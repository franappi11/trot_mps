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
from trot.meas.ghf import make_ghf_meas_ops_hubbard
from trot.prop import blocks, cpmc
from trot.prop.types import QmcParams
from trot.trial.ghf import GhfTrial, make_ghf_trial_ops

# CPMC with the spin-rotated UHF determinant as trial (the GHF template of uhf_rotation_cpmc.ipynb, cell 4), for a UHF
# state saved by uhf_dmrg_scan.py: UHF_FILE=<state_XX/uhf.npz> (or the first argument). The CPMC settings are those of
# ex_2d_4x4_saved.py (400 walkers, dt 0.005, 50 steps, 50 + 200 blocks, seed 1234, weight floor 1e-8).
# Walkers start from trot's default (the natural orbitals of the trial's rdm1).
UHF_FILE = os.environ.get("UHF_FILE") or (_sys.argv[1] if len(_sys.argv) > 1 else None)
if not UHF_FILE:
    raise SystemExit("give the saved UHF state: UHF_FILE=<path to state_XX/uhf.npz> (or as the first argument)")

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
params = QmcParams(
    n_walkers=400,
    n_chunks=int(os.environ.get("N_CHUNKS", 1)),
    auto_n_chunks=False,
    n_eql_blocks=int(os.environ.get("EQL", 20)),
    n_blocks=int(os.environ.get("BLOCKS", 200)),
    dt=0.005,
    n_prop_steps=50,
    weight_floor=1e-8,
    seed=1234,
)

# ---- the saved UHF state: occupied orbitals Ca (N, n_up), Cb (N, n_dn) and the UHF 1-RDM (2, N, N)
path = Path(UHF_FILE) if Path(UHF_FILE).is_absolute() else Path.cwd() / UHF_FILE
with np.load(path) as z:
    Ca, Cb = np.asarray(z["Ca"]), np.asarray(z["Cb"])
    rdm1_uhf = np.asarray(z["rdm1"])
    e_uhf = float(z["energy"])
if Ca.shape != (N, n_up) or Cb.shape != (N, n_dn):
    raise SystemExit(f"{path}: orbitals {Ca.shape}, {Cb.shape} do not match {N} sites and {(n_up, n_dn)} electrons")
print(f"UHF state loaded from {path}: UHF energy {e_uhf:.10f}, n_chunks = {params.n_chunks}", flush=True)
rdm_up, rdm_dn = (np.diag(dm) for dm in rdm1_uhf)
print("n_up - n_dn:\n", np.round((rdm_up - rdm_dn).reshape(L, L), 2))
print("n_up + n_dn:\n", np.round((rdm_up + rdm_dn).reshape(L, L), 2), flush=True)

# ---- the rotated determinant, as uhf_rotation_cpmc.ipynb (cell 4): rows 0..N-1 spin up, N..2N-1 spin down;
# up orbitals -> (up + down), down orbitals -> (up - down), i.e. the unnormalized 90 degree rotation about y
trial_data = np.block([
    [Ca, Cb],
    [Ca, -Cb],
])
trial = GhfTrial(jnp.asarray(trial_data))
print(f"rotated GHF trial: mo_coeff {tuple(trial_data.shape)}", flush=True)

trial_ops = make_ghf_trial_ops(sys)
meas_ops = make_ghf_meas_ops_hubbard(sys)
prop_ops = cpmc.make_prop_ops(ham, "unrestricted", trial_ops)
prop_ctx = prop_ops.build_prop_ctx(ham, None, params)

run = run_qmc(
    sys=sys,
    params=params,
    ham_data=ham,
    trial_data=trial,
    trial_ops=trial_ops,
    meas_ops=meas_ops,
    prop_ops=prop_ops,
    prop_ctx=prop_ctx,
    block_fn=blocks.block,
)

print(f"E = {float(run.mean_energy):.6f} +/- {float(run.stderr_energy):.6f}   (E - energy_fci {float(run.mean_energy) - energy_fci:+.6f})")
