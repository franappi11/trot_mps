"""Block-form H|trial> of a saved trial, cached next to it, for a later CPMC run.

The trial is a file of prepare_trial_6x6.py (dmrg_chi<chi>.npz: A{i}, q{i} with (N_up, N_dn) bond labels, energy, rdm1,
h1, u, nelec). H is applied block by block and the product compressed at --cutoff (trot.gmps.trials.load_or_make_htrial),
and written to <trial file>_htrial.npz, or <trial file>_htrial_tol<cutoff>.npz for a cutoff other than 1e-13. A run
loads it with trials.load_or_make_htrial(trial_file, h1, u, tensors, charges, energy, relative_tolerance=cutoff) and,
for the as-is spin-rotated trial, trials.rotate_htrial_blocks. CPU only; one job per trial, from this folder:

    sbatch -C "icelake&rocky9" --cpus-per-task=16 --mem=32G --time=00:25:00 --export=ALL,TARGET=prepare_htrial_saved.py \\
        run_mps_sweep.sh /mnt/ceph/users/fnappi/trot_trials/uhf_dmrg_6x6_U8/state_00/dmrg_chi512.npz --cutoff 1e-9
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_saved_trial(path):
    """(h1, u, nelec, tensors, charges, energy, rdm1) of a saved trial, with the checks a wrong file would fail."""
    with np.load(path, allow_pickle=False) as data:
        h1, u = np.asarray(data["h1"], dtype=float), float(data["u"])
        nelec = tuple(int(x) for x in data["nelec"])
        n = len(h1)
        tensors = [np.asarray(data[f"A{i}"]) for i in range(n)]
        charges = [np.asarray(data[f"q{i}"]) for i in range(n + 1)]
        energy = float(data["energy"])
        rdm1 = np.asarray(data["rdm1"]) if "rdm1" in data.files else None
    if any(np.iscomplexobj(A) or not np.isfinite(A).all() for A in tensors):
        raise ValueError(f"{path}: the tensors must be real and finite")
    if any(q.ndim != 2 or q.shape[1] != 2 for q in charges) or tuple(int(x) for x in charges[-1][0]) != nelec:
        raise ValueError(f"{path}: the trial needs (N_up, N_dn) bond labels that end at nelec = {nelec}")
    return h1, u, nelec, tensors, charges, energy, rdm1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trial", type=Path, help="a dmrg_chi<chi>.npz of prepare_trial_6x6.py")
    parser.add_argument("--cutoff", type=float, default=1.0e-13,
                        help="relative singular-value cut of the compression (default: roundoff)")
    args = parser.parse_args(argv)
    if not args.cutoff > 0.0:
        parser.error("--cutoff must be positive")

    from trot import config

    config.configure_once()
    from trot.gmps import trials

    h1, u, nelec, tensors, charges, energy, rdm1 = read_saved_trial(args.trial)
    print(f"{args.trial}: {len(h1)} sites, nelec {nelec}, U = {u:g}, trial bonds max {max(len(q) for q in charges)}, "
          f"E = {energy:.10f}; compression cutoff {args.cutoff:g}", flush=True)
    (_, labels), info = trials.load_or_make_htrial(args.trial, h1, u, tensors, charges, energy,
                                                   relative_tolerance=args.cutoff)
    if rdm1 is not None:  # the file's rdm1 (make_mps_trial's) against the one of the cache (one_rdm of the tensors)
        gap = float(np.max(np.abs(info["gamma"] - rdm1)))
        print(f"rdm1 of the cache against the saved one: max difference {gap:.1e}", flush=True)
        if gap > 1.0e-8:
            raise SystemExit(f"the cache's rdm1 differs from {args.trial}'s by {gap:.1e}")
    print(f"H|trial> in {trials.htrial_cache_file(args.trial, args.cutoff)}: bonds max {max(len(q) for q in labels)}, "
          f"<trial|H|trial> = {info['trial_energy']:.12f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
