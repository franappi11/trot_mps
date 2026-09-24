"""DMRG reference energy for the open Hubbard chain at half filling.

Appends one JSON line to --out with pyblock3's final Davidson energy and the
variational energy of the densified MPS in this code's convention (the same pair
as study_dmrg_ref.jsonl).

    ~/.trot/bin/python dmrg_reference.py --L 32 --U 8 --chi 200 --out sweep_L32_T8_U8/reference.jsonl
"""
import argparse
import json
import time

import jax.numpy as jnp

import mps_cpmc_new as m

parser = argparse.ArgumentParser()
parser.add_argument("--L", type=int, required=True)
parser.add_argument("--U", type=float, required=True)
parser.add_argument("--chi", type=int, default=200)
parser.add_argument("--sweeps", type=int, default=20)
parser.add_argument("--out", required=True)
args = parser.parse_args()

cfg = m.Config(L=args.L, n_up=args.L // 2, n_down=args.L // 2, interaction=args.U,
               trial_chi=args.chi, dmrg_sweeps=args.sweeps)
start = time.time()
mps, e_davidson = m.run_dmrg(m.build_dmrg_hamiltonian(cfg), cfg)
trial_np, _ = m.densify_with_charges(mps, args.L)
Htrial = m.compress_mps(m.apply_mpo(m.hubbard_mpo(args.L, 1.0, args.U), trial_np))
trial, Htrial = tuple(map(jnp.asarray, trial_np)), tuple(map(jnp.asarray, Htrial))
e_variational = float(m.contract_real(Htrial, trial) / m.contract_real(trial, trial))
record = dict(L=args.L, U=args.U, chi=args.chi, sweeps=args.sweeps, e_davidson=e_davidson,
              e_variational=e_variational, bond_dims=[int(A.shape[0]) for A in trial_np],
              seconds=time.time() - start)
with open(args.out, "a") as stream:
    stream.write(json.dumps(record) + "\n")
print(json.dumps(record))
