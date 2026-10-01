"""DMRG reference energy for the open Hubbard chain at sizes where build_qc_mpo is too big.

This uses mps_cpmc_gpu's term-built Hubbard MPO (bond dimension about 6) and its
warm-up schedule, the same DMRG the cluster runs use for their trials. dmrg_reference.py
goes through build_qc_mpo, whose bond dimension is about L^2/2.

Appends one JSON line with two energies:
- e_mps: <H> of the final MPS. It is variational, so use it as the reference.
- e_davidson: pyblock3's last Davidson energy, i.e. the two-site wavefunction before
  the final truncation.

    ~/.trot/bin/python chain_dmrg_reference.py --L 100 --U 8 --chi 200 --out cpmc_L100/L100_U8/dmrg_reference.jsonl
"""
import argparse
import json
import time

from pyblock3.algebra.mpe import MPE

import mps_cpmc_gpu as g

parser = argparse.ArgumentParser()
parser.add_argument("--L", type=int, required=True)
parser.add_argument("--U", type=float, required=True)
parser.add_argument("--t", type=float, default=1.0)
parser.add_argument("--n-up", type=int, default=None, help="default L // 2")
parser.add_argument("--n-down", type=int, default=None, help="default L // 2")
parser.add_argument("--chi", type=int, default=200)
parser.add_argument("--sweeps", type=int, default=30, help="Config.dmrg_sweeps, as in the cluster trials")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", required=True)
parser.add_argument("--tag", default="")
args = parser.parse_args()

n_up = args.L // 2 if args.n_up is None else args.n_up
n_down = args.L // 2 if args.n_down is None else args.n_down
cfg = g.Config(L=args.L, n_up=n_up, n_down=n_down, hopping=args.t, interaction=args.U,
               trial_chi=args.chi, dmrg_sweeps=args.sweeps, dmrg_seed=args.seed)
start = time.time()
hamiltonian = g.build_dmrg_hamiltonian(cfg)
mps, e_davidson = g.run_dmrg(hamiltonian, cfg)
mpo = g.hubbard_dmrg_mpo(hamiltonian, cfg)
e_mps = float(MPE(mps, mpo, mps)[0:2].expectation) / float(mps @ mps)
tensors, _ = g.densify_with_charges(mps, args.L)
record = dict(kind="dmrg", tag=args.tag, L=args.L, n_up=n_up, n_down=n_down, hopping=args.t,
              interaction=args.U, chi=args.chi, sweeps=args.sweeps, seed=args.seed,
              e_mps=e_mps, e_davidson=e_davidson,
              bond_dims=[int(A.shape[0]) for A in tensors] + [int(tensors[-1].shape[-1])],
              seconds=time.time() - start)
with open(args.out, "a") as stream:
    stream.write(json.dumps(record) + "\n")
print(json.dumps(record))
