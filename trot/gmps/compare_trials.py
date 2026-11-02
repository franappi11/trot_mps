"""Compare two saved DMRG trials of the same system (prepare_trial_6x6.py's dmrg_chi<chi>.npz format).

    python compare_trials.py trial_a.npz trial_b.npz

Checks h1, u and nelec match, then reports each trial's saved and recomputed variational energy (the Hubbard MPO, so
both are on the same footing whatever the file's `energy` field holds), their state overlap and fidelity (a direct
MPS-MPS contraction via trot.gmps.utils.contract_real: works even though the two trials generally have different
bond dimensions and an unrelated gauge, since it is a standard sequential overlap, not a tensor-by-tensor diff), and
the rdm1 difference. Two independently DMRG-converged trials at the same finite bond need not be identical (nor even
very close) if either run landed in a different near-degenerate branch; this is what tells you whether they did.

CPU only, a few seconds at chi ~1024 (one contract_real pass per norm/overlap): run_mps_sweep.sh, no GPU needed.
"""

from trot.config import configure_once

configure_once()  # float64: the overlap/norm values are compared near 1.0, at the 1e-6 level and below

import argparse

import numpy as np

from trot.gmps import dense_engine
from trot.gmps.prepare_htrial_saved import read_saved_trial
from trot.gmps.utils import contract_real
from trot.meas.mps import hubbard_mpo_from_h1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trial_a", type=str)
    parser.add_argument("trial_b", type=str)
    args = parser.parse_args(argv)

    h1_a, u_a, nelec_a, tensors_a, charges_a, energy_a, rdm1_a = read_saved_trial(args.trial_a)
    h1_b, u_b, nelec_b, tensors_b, charges_b, energy_b, rdm1_b = read_saved_trial(args.trial_b)

    if nelec_a != nelec_b or u_a != u_b or not np.allclose(h1_a, h1_b):
        raise SystemExit(f"not the same system: nelec {nelec_a} vs {nelec_b}, u {u_a} vs {u_b}, "
                         f"h1 matches: {np.allclose(h1_a, h1_b)}")

    # The saved `energy` fields need not mean the same thing (prepare_trial_6x6.py stores the variational energy of
    # the saved state, drmg_trial_6x6.py pyblock3's last Davidson sweep energy), so both are recomputed here the same
    # way: <T|H|T>/<T|T> with the Hubbard MPO (the MPO energy kernel's, validated against the stored chi 512/1024
    # energies to 1e-12).
    mpo = dense_engine.mpo_site_tensors(hubbard_mpo_from_h1(h1_a, u_a))
    variational_a = dense_engine.mpo_trial_energy(mpo, tensors_a)
    variational_b = dense_engine.mpo_trial_energy(mpo, tensors_b)
    bonds_a = max(len(q) for q in charges_a)
    bonds_b = max(len(q) for q in charges_b)
    print(f"{args.trial_a}: bond {bonds_a}, saved energy {energy_a:.10f}, <T|H|T>/<T|T> {variational_a:.10f}")
    print(f"{args.trial_b}: bond {bonds_b}, saved energy {energy_b:.10f}, <T|H|T>/<T|T> {variational_b:.10f}")
    print(f"variational energy difference (a - b): {variational_a - variational_b:+.6f}")

    norm_a = float(contract_real(tensors_a, tensors_a))
    norm_b = float(contract_real(tensors_b, tensors_b))
    overlap = float(contract_real(tensors_a, tensors_b))
    fidelity = overlap ** 2 / (norm_a * norm_b)
    print(f"norms: {norm_a:.10f}, {norm_b:.10f} (should be 1 for a normalized trial)")
    print(f"overlap <a|b> = {overlap:.10f}, fidelity |<a|b>|^2 / (<a|a><b|b>) = {fidelity:.10f} "
          f"(1 - fidelity = {1 - fidelity:.3e})")

    if rdm1_a is not None and rdm1_b is not None:
        diff = np.asarray(rdm1_a) - np.asarray(rdm1_b)
        print(f"rdm1 difference: max |diff| = {np.abs(diff).max():.3e}, "
              f"Frobenius norm = {np.linalg.norm(diff):.3e}")
        # the charge and spin pattern of each trial on the lattice: where the holes sit and the S^z order
        side = int(round(len(h1_a) ** 0.5))
        if side * side == len(h1_a):
            for name, rdm1 in ((args.trial_a, rdm1_a), (args.trial_b, rdm1_b)):
                n_up, n_dn = (np.diag(np.asarray(rdm1)[s]) for s in range(2))
                for label, grid in (("hole density 1 - n", 1 - n_up - n_dn), ("S^z", 0.5 * (n_up - n_dn))):
                    rows = "\n".join("  " + " ".join(f"{v:+.2f}" for v in row) for row in grid.reshape(side, side))
                    print(f"{name}: {label}, row y, column x:\n{rows}")
    else:
        print("rdm1 missing from at least one file: skipped that check")


if __name__ == "__main__":
    raise SystemExit(main())
