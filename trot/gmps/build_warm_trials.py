"""Wall-free low-bond DMRG trials for the GPU runs, written in the trial-cache format of trot.gmps.trials.

Random-start DMRG at small bond on long chains converges to states with domain walls of the staggered
magnetisation (see rotated_dmrg_trial_study.warm_dmrg). This script runs DMRG at --warm (random start),
compresses that state to each --chi, re-optimises it there, and saves every trial as
<out>/L{L}_n{N_up}-{N_dn}_t{t}_U{U}_chi{chi}_sw{sweeps}_seed{seed}.npz, the chain cache name of
trot.gmps.trials.trial_cache_file, with the keys it reads (A{i}, q{i}, energy) plus a description. Point
run_mps_cpmc.py at the directory with --trial-cache <out>; --dmrg-sweeps and --dmrg-seed must match --sweeps
and --seed.

    ~/.trot/bin/python trot/gmps/build_warm_trials.py --L 100 --U 8 --chi 8 16 --warm 32 \\
        --out trot/gmps/trial_cache_warm
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--U", type=float, default=8.0)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--chi", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--warm", type=int, default=32, help="bond of the random-start DMRG state")
    parser.add_argument(
        "--warm-sweeps", type=int, default=30, help="sweeps of the warm source (must end wall-free)"
    )
    parser.add_argument(
        "--sweeps", type=int, default=30, help="file-name tag (--dmrg-sweeps of the GPU run)"
    )
    parser.add_argument("--seed", type=int, default=0, help="DMRG seed, also the --dmrg-seed tag")
    parser.add_argument(
        "--reference", type=float, default=None, help="reference energy for the printout"
    )
    parser.add_argument("--out", default="trial_cache_warm")
    args = parser.parse_args()

    from trot.config import configure_once

    configure_once()
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rotated_dmrg_trial_study import diagnostics, domain_walls, hubbard_chain, warm_dmrg

    if args.t != 1.0:
        raise SystemExit("only t = 1 (hubbard_chain) is supported")
    h1, ham, sys_ = hubbard_chain(args.L, args.U)
    nup, ndn = (int(n) for n in sys_.nelec)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for chi in args.chi:
        start = time.perf_counter()
        trial = warm_dmrg(ham, sys_, chi, args.warm, seed=args.seed, warm_sweeps=args.warm_sweeps)
        d = diagnostics(trial.tensors, h1, args.U, args.reference or 0.0)
        walls, max_m, _ = domain_walls(trial.rdm1)
        name = f"L{args.L}_n{nup}-{ndn}_t{args.t:g}_U{args.U:g}_chi{chi}_sw{args.sweeps}_seed{args.seed}.npz"
        info = dict(
            built_by="warm_dmrg",
            chi=chi,
            warm_chi=args.warm,
            dmrg_seed=args.seed,
            e_var=d["e_var"],
            walls=walls,
            max_staggered_m=max_m,
            p0=d["p0"],
            singlet_fraction_after_rotation=d["singlet_fraction_after_rotation"],
        )
        np.savez(
            out / name,
            energy=d["e_var"],
            description=json.dumps(info),
            **{f"A{i}": np.asarray(A) for i, A in enumerate(trial.tensors)},
            **{f"q{i}": np.asarray(q) for i, q in enumerate(trial.charge_arrays())},
        )
        gap = f" (E - ref {d['e_var'] - args.reference:+.4f})" if args.reference is not None else ""
        print(
            f"chi={chi} warm from {args.warm}: E_var {d['e_var']:.6f}{gap}, walls {walls}, max|m| {max_m:.3f}, "
            f"p0 {d['p0']:.3f}, singlet after rotation {d['singlet_fraction_after_rotation']:.3f} -> {out / name} "
            f"({time.perf_counter() - start:.0f} s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
