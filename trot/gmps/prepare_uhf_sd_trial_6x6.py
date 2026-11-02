"""The UHF determinant behind the doped 6x6 trials as an MPS trial file, for an MPS-CPMC run with the trial of
uhf_6x6_doped.py's first run (the same determinant, rotated 90 degrees about y, exact overlaps): the two runs then
differ only in the MPS machinery (walker conversion, truncation, frozen plan) and in this file's compression.

1. The entanglement of each spin channel, exactly. At cut k the Schmidt values of a determinant come from the
   eigenvalues n of the overlap matrix of its orbitals restricted to the sites left of the cut: each mode with
   0 < n < 1 contributes the factors n, 1 - n to the Schmidt probabilities. Printed per cut: the active modes (log2 of
   the exact rank) per channel, and the best discarded weight of the d=4 trial (the product of the channels) at each
   --caps bond.
2. The trial: sd_to_gmps(Ca, Cb, chi=--channel-chi) (channel bond --channel-chi, d=4 bond up to its square), then
   compress_mps_qn(..., max_bond=--chi) (charge-blocked, labels kept), its fidelity with the uncompressed MPS and its
   energy with the Hubbard MPO (pyblock3). The first run (7256271) used pyblock3's compression instead: its result
   could not be converted back (densify: "bond 32 differs between neighboring tensors"), and --channel-chi 32 alone
   lost 1.3% of the weight.
3. Written in the prepare_trial_6x6.py format (A{i}, q{i}, energy, rdm1, h1, u, nelec) to <out>, by default
   state_00/uhf_sd_chi<chi>.npz next to uhf.npz, for prepare_htrial_saved.py --cutoff 1e-9 and then
   ex_6x6_doped_diag.py --trial-file <out>.

CPU only, from this folder:

    sbatch -p genx -C "icelake&rocky9" --cpus-per-task=8 --mem=64G --time=01:00:00 -J uhf_sd_trial \\
        --export=ALL,TARGET=prepare_uhf_sd_trial_6x6.py run_mps_sweep.sh --chi 256

The d=4 MPS before compression has bonds up to --channel-chi squared (4096 at 64: about 0.5 GB per site, dense).
"""
from trot.config import configure_once

configure_once()
import argparse
import time
from pathlib import Path

import numpy as np

STATE = Path("/mnt/ceph/users/fnappi/trot_trials/uhf_dmrg_6x6_U8/state_00")


def schmidt_probabilities(n, keep, tolerance=1e-14):
    """The `keep` largest Schmidt probabilities of a determinant whose left-block occupations are n (products of
    n_j or 1 - n_j over the modes), sorted, and the number of active modes. Keeping only the largest `keep` partial
    products at every step is exact for the largest `keep` final ones."""
    active = n[(n > tolerance) & (n < 1.0 - tolerance)]
    p = np.array([1.0])
    for x in active:
        p = np.concatenate((p * x, p * (1.0 - x)))
        if len(p) > keep:
            p = np.partition(p, len(p) - keep)[len(p) - keep:]
    return np.sort(p)[::-1], len(active)


def channel_entanglement(C, keep):
    """Per cut k = 1 .. L-1: (Schmidt probabilities, active modes) of the determinant with orbitals C."""
    Q = np.linalg.qr(np.asarray(C, dtype=float))[0]
    return [schmidt_probabilities(np.linalg.eigvalsh(Q[:k].T @ Q[:k]), keep) for k in range(1, len(Q))]


def best_discarded(pa, pb, cap):
    """The smallest discarded weight of a bond-`cap` cut of the d=4 state, whose Schmidt probabilities are the
    products of the two channels'."""
    products = np.outer(pa[:cap], pb[:cap]).ravel()
    kept = np.partition(products, len(products) - cap)[len(products) - cap:] if len(products) > cap else products
    return max(0.0, 1.0 - float(kept.sum()))


def overlap(a, b):
    """<a|b> of two real MPS on the same sites, contracted with tensordot (the environment is the only intermediate)."""
    env = np.ones((1, 1))
    for A, B in zip(a, b):
        env = np.tensordot(np.tensordot(env, np.asarray(A), axes=(0, 0)), np.asarray(B), axes=([0, 1], [0, 1]))
    return float(env.reshape(()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uhf", type=Path, default=STATE / "uhf.npz")
    parser.add_argument("--chi", type=int, default=256, help="bond of the trial written (default 256)")
    parser.add_argument("--channel-chi", type=int, default=64, help="channel bond of the conversion (default 64)")
    parser.add_argument("--caps", type=int, nargs="*", default=[256, 512, 1024])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out = args.out or args.uhf.with_name(f"uhf_sd_chi{args.chi}.npz")

    from pyblock3.algebra.mpe import MPE

    from trot.gmps.dmrg import hubbard_pyblock3_mpo, make_pyblock3_hamiltonian
    from trot.gmps.dmrg_szsz_6x6 import save_trial
    from trot.gmps.ex_6x6_doped_diag import U, hubbard_h1
    from trot.gmps.uhf_dmrg_scan import gmps_to_pyblock3, quiet
    from trot.gmps.utils import sd_to_gmps
    from trot.trial.mps import compress_mps_qn, make_mps_trial

    with np.load(args.uhf) as data:
        Ca, Cb, e_uhf, seed = np.asarray(data["Ca"]), np.asarray(data["Cb"]), float(data["energy"]), int(data["seed"])
    h1 = hubbard_h1()
    nelec = (Ca.shape[1], Cb.shape[1])
    print(f"{args.uhf}: seed {seed}, E_UHF {e_uhf:.10f}, nelec {nelec}", flush=True)

    keep = max(max(args.caps), args.chi)
    spectra = [channel_entanglement(C, keep) for C in (Ca, Cb)]
    print(f"cut  active modes up/dn  (exact d=4 rank 2^sum)  best discarded weight at bond "
          + ", ".join(map(str, args.caps)), flush=True)
    worst = {cap: 0.0 for cap in args.caps}
    for k, ((pa, na), (pb, nb)) in enumerate(zip(*spectra), start=1):
        lost = {cap: best_discarded(pa, pb, cap) for cap in args.caps}
        worst = {cap: max(worst[cap], lost[cap]) for cap in args.caps}
        print(f"{k:3d}  {na:2d} / {nb:2d}  (2^{na + nb:2d})  " + ", ".join(f"{lost[c]:.2e}" for c in args.caps),
              flush=True)
    print("largest over the cuts: " + ", ".join(f"bond {c}: {worst[c]:.2e}" for c in args.caps), flush=True)

    start = time.perf_counter()
    gmps = sd_to_gmps(Ca, Cb, chi=args.channel_chi)
    full = [np.asarray(A) for A in gmps.tensors]
    print(f"channel bond {args.channel_chi}: discarded weight {gmps.discarded:.3e}, d=4 bonds max "
          f"{max(len(q) for q in gmps.charges)}; {time.perf_counter() - start:.0f} s", flush=True)
    start = time.perf_counter()
    small, small_charges = compress_mps_qn(full, gmps.charges, max_bond=args.chi)
    # <full|full> = 1 - discarded (the channel truncations only remove norm); the overlap of the two MPS
    fidelity = overlap(full, small) ** 2 / ((1.0 - gmps.discarded) * overlap(small, small))
    del full
    print(f"compressed to bond {args.chi} (bonds max {max(map(len, small_charges))}) in "
          f"{time.perf_counter() - start:.0f} s: fidelity with the uncompressed {fidelity:.8f}, about "
          f"{fidelity * (1.0 - gmps.discarded):.8f} with the determinant", flush=True)

    hamiltonian = make_pyblock3_hamiltonian(h1, nelec)
    with quiet():
        mpo = hubbard_pyblock3_mpo(hamiltonian, h1, U)
    psi = gmps_to_pyblock3(small, small_charges)
    e_trial = float(MPE(psi, mpo, psi)[0:2].expectation) / float(psi @ psi)
    print(f"E {e_trial:.10f} (E_UHF {e_uhf:.10f}, difference {e_trial - e_uhf:+.2e})", flush=True)

    trial = make_mps_trial(small, small_charges, nelec=nelec)
    save_trial(out, trial.tensors, trial.charge_arrays(), e_trial, h1, U, nelec, rdm1=np.asarray(trial.rdm1),
               uhf_seed=seed, uhf_energy=e_uhf, chi=args.chi, chi_channel=args.channel_chi,
               gmps_discarded=gmps.discarded, fidelity_uncompressed=fidelity)
    print(f"saved {out} (bonds max {max(len(q) for q in trial.charge_arrays())}); next: "
          f"prepare_htrial_saved.py {out} --cutoff 1e-9", flush=True)


if __name__ == "__main__":
    main()
