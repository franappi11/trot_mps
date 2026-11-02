"""Compress a saved DMRG trial to a smaller bond (the converged chi 2048 one to chi 1024 by default), keeping its
(N_up, N_dn) bond labels, and save it in prepare_trial_6x6.py's format for ex_6x6_doped_diag.py --trial-file.

    python compress_trial_6x6.py                                  # dmrg_chi2048.npz -> dmrg_chi1024_from2048.npz
    python compress_trial_6x6.py --input X.npz --chi 512 --out Y.npz

One SVD truncation sweep (trot.trial.mps.compress_mps_qn with max_bond): the largest Schmidt values across a bond's
sectors are kept, as a single-sweep truncation, not the variational optimum at the new bond (a few DMRG sweeps at chi
would lower the energy further). The result keeps the parent's state (its hole and spin pattern), which a DMRG run at
chi from the UHF seed need not find (the two ramp runs did not). Reported: the exact infidelity with the parent (an MPS
overlap) and the compressed trial's variational energy through the Hubbard MPO (the parent's is the saved one).

CPU only, on genx through run_mps_sweep.sh: the parent is 3.4 GB, the overlaps contract bonds 2048 x 1024.
"""

from trot.config import configure_once

configure_once()  # float64, before jax (contract_real, mpo_trial_energy)

import argparse
import time
from pathlib import Path

import numpy as np

from trot.gmps import dense_engine
from trot.gmps.dmrg_szsz_6x6 import STATE, save_trial
from trot.gmps.prepare_htrial_saved import read_saved_trial
from trot.gmps.utils import contract_real
from trot.meas.mps import hubbard_mpo_from_h1
from trot.trial.mps import compress_mps_qn, one_rdm


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=STATE / "dmrg_chi2048.npz", help="the saved trial to compress")
    parser.add_argument("--chi", type=int, default=1024, help="the new bond dimension")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <input folder>/dmrg_chi<chi>_from<input bond>.npz")
    args = parser.parse_args(argv)

    h1, u, nelec, tensors, charges, energy, _ = read_saved_trial(args.input)
    bond = max(len(q) for q in charges)
    out = args.out or args.input.parent / f"dmrg_chi{args.chi}_from{bond}.npz"
    if out.exists():  # never overwrite a trial other runs may read
        raise SystemExit(f"{out} exists: pass another --out")
    print(f"{args.input}: bond {bond}, saved energy {energy:.10f}; compressing to bond {args.chi}", flush=True)

    start = time.perf_counter()
    small, labels = compress_mps_qn(tensors, charges, max_bond=args.chi)
    norm2 = float(contract_real(small, small))
    small[0] = small[0] / np.sqrt(norm2)  # normalized, as the saved trials are
    print(f"compressed in {time.perf_counter() - start:.0f} s: bond {max(len(q) for q in labels)}, kept norm^2 "
          f"{norm2:.8f}", flush=True)

    start = time.perf_counter()
    overlap = float(contract_real(tensors, small))
    parent_norm2 = float(contract_real(tensors, tensors))
    infidelity = 1.0 - overlap ** 2 / parent_norm2
    mpo = dense_engine.mpo_site_tensors(hubbard_mpo_from_h1(h1, u))
    e_small = dense_engine.mpo_trial_energy(mpo, small)
    print(f"infidelity with the parent {infidelity:.3e}; <T|H|T>/<T|T> {e_small:.10f} (parent {energy:.10f}, "
          f"difference {e_small - energy:+.6f}); {time.perf_counter() - start:.0f} s", flush=True)

    start = time.perf_counter()
    rdm1 = np.stack(one_rdm(small))
    print(f"rdm1 in {time.perf_counter() - start:.0f} s", flush=True)
    save_trial(out, small, labels, e_small, h1, u, nelec, rdm1, chi=args.chi, parent_file=str(args.input),
               parent_energy=energy, compression_infidelity=infidelity)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
