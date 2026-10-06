from trot import config

config.configure_once()  # set the precision; before importing jax
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from trot.core.system import System
from trot.driver import run_qmc
from trot.gmps.dmrg import make_dmrg_trial
from trot.ham.hubbard import HamHubbard
from trot.meas.mps import make_mps_meas_ops_hubbard
from trot.prop import mps_cpmc
from trot.prop.types import QmcParamsMps
from trot.trial.mps import make_mps_trial, make_mps_trial_ops, make_walker_plan, rotate_spin, spin_rotation_y

from trot.trial.mps_rotation import rotate_mps_trial, rotated_rdm1



L = 6 
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

#unrestricted walkers
sys = System(N, (N // 2, N // 2), "unrestricted")
ham = HamHubbard(jnp.asarray(h1), U)  
params = QmcParamsMps(
    n_walkers=400,
    n_chunks=4,  # 4 chunks of 100 walkers, so each fits in the 20 GB MIG slice
    auto_n_chunks=False,  # skip the automatic chunk search
    n_eql_blocks=50,  
    n_blocks=500,
    dt=0.005,
    n_prop_steps=50,
    weight_floor=1e-8,
    seed=1234,
    trial_chi=128,
    dmrg_sweeps=30,
    dmrg_seed=0,
    #Neel prodcut state is the default
    dmrg_init="auto",  
    #Adaptive uses the smaller block size that satisfies the tolerance
    occupation_tolerance=1e-10,
    orbital_plan="adaptive",
    #Bond dimensio per spin
    walker_channel_chi=32,
    plan_reference="natural",  # determinant that freezes the conversion circuit: the trial's natural orbitals
    walker_start="natural",  # every walker starts from the trial's natural determinant
)

# pyblock3 DMRG trial. The lattice is bipartite at half filling, so it starts from the Neel product state.
dmrg = make_dmrg_trial(ham, sys, chi=params.trial_chi, n_sweeps=params.dmrg_sweeps, seed=params.dmrg_seed,
                       init=params.dmrg_init)
trial = dmrg.trial
print(f"DMRG trial ({dmrg.init} start): bonds max {max(trial.bond_dims)}, "
      f"variational energy {dmrg.variational_energy:.10f}", flush=True)
# spin polarization of the DMRG trial: near 0 means the rotated rdm1 equals the unrotated one (site i = y * L + x)
g = np.asarray(trial.rdm1)
sz = 0.5 * (np.diag(g[0]) - np.diag(g[1]))  # <S^z_i>
print(f"trial spin polarization: ||rdm1_up - rdm1_dn|| {np.linalg.norm(g[0] - g[1]):.3e}, max |<S^z_i>| "
      f"{np.abs(sz).max():.3f}, staggered m {np.mean((-1.0) ** (np.arange(N) % L + np.arange(N) // L) * sz):+.3f}",
      flush=True)
# rdm1 whose natural orbitals set the walker plan and start: the unrotated DMRG trial's (L=100: better start overlap,
# fewer nodes, faster steps than the rotated trial's spin-averaged rotated_rdm1(rdm1, R), same energy)
rdm1 = np.asarray(trial.rdm1)  # the unrotated DMRG trial's rdm1: walker start and conversion plan
R = spin_rotation_y(beta_deg=90.0)  # exp(-i beta S^y), 90 degrees
trial = make_mps_trial(tensors=rotate_spin(tensors=trial.tensors, R=R), nelec=sys.nelec, rdm1=rdm1)  # N labels
print(f"rotated trial: bonds max {max(trial.bond_dims)}", flush=True)
plan = make_walker_plan(ham_data=ham, trial=trial, sys_=sys, params=params)  # frozen on the trial's natural orbitals
trial_ops = make_mps_trial_ops(plan=plan)
meas_ops = make_mps_meas_ops_hubbard(plan=plan, energy_kernel=params.energy_kernel, htrial=None)
prop_ops = mps_cpmc.make_prop_ops(ham_data=ham, sys=sys, plan=plan)
meas_ctx = meas_ops.build_meas_ctx(ham, trial)  # H|trial> 
prop_ctx = prop_ops.build_prop_ctx(ham, trial_ops.get_rdm1(trial), params)  # (ham_data, rdm1, params)

# Walkers at the start and after every SAVE_EVERY-th block (equilibration blocks count too), one npz each:
# up, dn of shape (n_walkers, N, n_up / n_dn). After a block's comb every walker has the same weight.
OUT = Path("/mnt/ceph/users/fnappi/trot_walkers/sq6x6_U8_ex2d")
OUT.mkdir(parents=True, exist_ok=True)
SAVE_EVERY = 50
state = prop_ops.init_prop_state(sys=sys, ham_data=ham, trial_ops=trial_ops, trial_data=trial, meas_ops=meas_ops,
                                 params=params, meas_ctx=meas_ctx)


def save_walkers(path, up, dn, **info):
    np.savez(path, up=np.asarray(up), dn=np.asarray(dn), **info)
    print(f"walkers saved to {path}", flush=True)


save_walkers(OUT / "walkers_block0000.npz", *state.walkers, block=0, tau=0.0)
blocks_done = [0]


def record(up, dn, pre_comb_weights, comb_index, energy, weight, e_estimate, node_encounters):
    """Called by the MPS block after every block with the walkers after its comb."""
    blocks_done[0] += 1
    b = blocks_done[0]
    if b % SAVE_EVERY == 0:
        save_walkers(OUT / f"walkers_block{b:04d}.npz", up, dn, block=b, tau=b * params.n_prop_steps * params.dt,
                     energy=float(energy), weight=float(weight), e_estimate=float(e_estimate),
                     node_encounters=int(node_encounters))
    return np.int32(0)


run = run_qmc(
    sys=sys,
    params=params,
    ham_data=ham,
    trial_data=trial,
    meas_ops=meas_ops,
    trial_ops=trial_ops,
    prop_ops=prop_ops,
    block_fn=mps_cpmc.make_block(record=record),  # the MPS block, calling record after every block
    state=state,  # the start saved above
    meas_ctx=meas_ctx,
    prop_ctx=prop_ctx,
    target_error=None,
    mesh=None,
    observable_names=(),
    runtime=None,
)

print(f"E = {float(run.mean_energy):.6f} +/- {float(run.stderr_energy):.6f}")
