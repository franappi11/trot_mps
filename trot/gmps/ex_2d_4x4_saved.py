from trot import config
# set the precision; before importing jax
config.configure_once()
import os
import sys as _sys
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from trot.core.system import System
from trot.driver import run_qmc
from trot.ham.hubbard import HamHubbard
from trot.meas.mps import make_mps_meas_ops_hubbard
from trot.prop import mps_cpmc
from trot.prop.types import QmcParamsMps
from trot.trial.mps import (make_mps_trial, make_mps_trial_ops, make_walker_plan, rotate_spin,
                            spin_rotation_y)

# ex_2d_4x4_test.py with the trial loaded from a saved DMRG state instead of built from a UHF guess:
# TRIAL_FILE=<dmrg_chi*.npz of uhf_dmrg_scan.py> (or the first argument); everything after the trial is the same.
HERE = Path(__file__).resolve().parent
TRIAL_FILE = os.environ.get("TRIAL_FILE") or (_sys.argv[1] if len(_sys.argv) > 1 else None)
if not TRIAL_FILE:
    raise SystemExit("give the saved trial: TRIAL_FILE=<path to dmrg_chi*.npz> (or as the first argument)")

L = 4
U = 8.0
t = 1.0
N = L * L

h1 = np.zeros((N, N))
for y in range(L):
    for x in range(L):
        site = y * L + x
        if x + 1 < L:
            h1[site, site + 1] = h1[site + 1, site] = -t
        if y + 1 < L:
            h1[site, site + L] = h1[site + L, site] = -t

n_up = n_dn = 7
energy_fci = -10.1218936956
#unrestricted walkers
sys = System(N, (n_up, n_dn), "unrestricted")
ham = HamHubbard(jnp.asarray(h1), U)

# ---- the saved DMRG trial (trial-cache format: A{i}, q{i}, energy, rdm1, h1, u, nelec)
path = Path(TRIAL_FILE) if Path(TRIAL_FILE).is_absolute() else Path.cwd() / TRIAL_FILE
with np.load(path) as z:
    if "h1" in z.files and not (z["h1"].shape == h1.shape and np.allclose(z["h1"], h1)):
        raise SystemExit(f"{path} was made for another h1")
    if "u" in z.files and float(z["u"]) != U:
        raise SystemExit(f"{path} was made for U = {float(z['u'])}, not {U}")
    if "nelec" in z.files and tuple(int(x) for x in z["nelec"]) != (n_up, n_dn):
        raise SystemExit(f"{path} was made for nelec {tuple(int(x) for x in z['nelec'])}, not {(n_up, n_dn)}")
    tensors = [z[f"A{i}"] for i in range(N)]
    charges = [z[f"q{i}"] for i in range(N + 1)]
    rdm1_saved = np.asarray(z["rdm1"]) if "rdm1" in z.files else None
    e_dmrg = float(z["energy"])
trial = make_mps_trial(tensors=tensors, charges=charges, nelec=(n_up, n_dn), rdm1=rdm1_saved)
print(f"trial loaded from {path}: bonds max {max(trial.bond_dims)}, variational energy {e_dmrg:.10f}, "
      f"E - energy_fci {e_dmrg - energy_fci:+.6f}", flush=True)

params = QmcParamsMps(
    n_walkers=400,
    #When using a slice, I was getting oom without chunking
    n_chunks=int(os.environ.get("N_CHUNKS", 1)),
    auto_n_chunks=False,
    n_eql_blocks=50,
    n_blocks=200,
    dt=0.005,
    n_prop_steps=50,
    weight_floor=1e-8,
    seed=1234,
    trial_chi=max(trial.bond_dims),
    dmrg_sweeps=20,  # DMRG from the UHF MPS below
    dmrg_seed=0,  # DMRG noise
    #Adaptive uses the smaller block size that satisfies the tolerance
    occupation_tolerance=1e-10,
    orbital_plan="adaptive",
    #Bond dimension per spin
    walker_channel_chi=int(os.environ.get("WALKER_CHI", 16)),
    plan_reference="natural",  # determinant that freezes the conversion circuit: the trial's natural orbitals
    walker_start="natural",  # every walker starts from the trial's natural determinant
)
print(f"trial bond dimension chi_T = {params.trial_chi}, walker bond per spin chi_w = {params.walker_channel_chi}, "
      f"n_chunks = {params.n_chunks}", flush=True)

# spin polarization of the DMRG trial
g = np.asarray(trial.rdm1)
sz = 0.5 * (np.diag(g[0]) - np.diag(g[1]))  # <S^z_i>
print(f"trial spin polarization: ||rdm1_up - rdm1_dn|| {np.linalg.norm(g[0] - g[1]):.3e}, max |<S^z_i>| "
      f"{np.abs(sz).max():.3f}", flush=True)
print("trial <n_i> (row y, column x):\n" + "\n".join(" ".join(f"{v:.3f}" for v in row)
                                                     for row in (np.diag(g[0]) + np.diag(g[1])).reshape(L, L)),
      flush=True)

#Taking 1rdm before rotation
rdm1 = np.asarray(trial.rdm1)
#Spin-projecting the trial
R = spin_rotation_y(beta_deg=90.0)  # exp(-i beta S^y), 90 degrees
trial = make_mps_trial(tensors=rotate_spin(tensors=trial.tensors, R=R), nelec=sys.nelec, rdm1=rdm1)  # N labels
print(f"rotated trial: bonds max {max(trial.bond_dims)}", flush=True)
#Setting up computation
plan = make_walker_plan(ham_data=ham, trial=trial, sys_=sys, params=params)
trial_ops = make_mps_trial_ops(plan=plan)
meas_ops = make_mps_meas_ops_hubbard(plan=plan, energy_kernel=params.energy_kernel, htrial=None)
prop_ops = mps_cpmc.make_prop_ops(ham_data=ham, sys=sys, plan=plan)
meas_ctx = meas_ops.build_meas_ctx(ham, trial)
prop_ctx = prop_ops.build_prop_ctx(ham, trial_ops.get_rdm1(trial), params)

run = run_qmc(
    sys=sys,
    params=params,
    ham_data=ham,
    trial_data=trial,
    meas_ops=meas_ops,
    trial_ops=trial_ops,
    prop_ops=prop_ops,
    block_fn=mps_cpmc.block,
    state=None,
    meas_ctx=meas_ctx,
    prop_ctx=prop_ctx,
    target_error=None,
    mesh=None,
    observable_names=(),
    runtime=None,
)

print(f"E = {float(run.mean_energy):.6f} +/- {float(run.stderr_energy):.6f}   (E - energy_fci {float(run.mean_energy) - energy_fci:+.6f})")
