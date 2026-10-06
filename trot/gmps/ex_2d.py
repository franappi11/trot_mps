from trot import config

config.configure_once()  # set the precision; before importing jax
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
# rdm1 whose natural orbitals set the walker plan and start: the rotated trial's (trot's convention);
# np.asarray(trial.rdm1) keeps the unrotated trial's instead
R = spin_rotation_y(beta_deg=90.0)  # exp(-i beta S^y), 90 degrees
rdm1 = rotated_rdm1(rdm1=np.asarray(trial.rdm1), R=R)
rdm1 = rotated_rdm1(rdm1=np.asarray(trial.rdm1), R=R)
trial = make_mps_trial(tensors=rotate_spin(tensors=trial.tensors, R=R), nelec=sys.nelec, rdm1=rdm1)  # N labels
print(f"rotated trial: bonds max {max(trial.bond_dims)}", flush=True)
plan = make_walker_plan(ham_data=ham, trial=trial, sys_=sys, params=params)  # frozen on the trial's natural orbitals
trial_ops = make_mps_trial_ops(plan=plan)
meas_ops = make_mps_meas_ops_hubbard(plan=plan, energy_kernel=params.energy_kernel, htrial=None)
prop_ops = mps_cpmc.make_prop_ops(ham_data=ham, sys=sys, plan=plan)
meas_ctx = meas_ops.build_meas_ctx(ham, trial)  # H|trial> 
prop_ctx = prop_ops.build_prop_ctx(ham, trial_ops.get_rdm1(trial), params)  # (ham_data, rdm1, params)

run = run_qmc(
    sys=sys,
    params=params,
    ham_data=ham,
    trial_data=trial,
    meas_ops=meas_ops,
    trial_ops=trial_ops,
    prop_ops=prop_ops,
    block_fn=mps_cpmc.block,  # the MPS block; trot.prop.blocks.block gives the same blocks, with 3 more conversions
    state=None,  # None: prop_ops.init_prop_state starts every walker from params.walker_start
    meas_ctx=meas_ctx,
    prop_ctx=prop_ctx,
    target_error=None,
    mesh=None,
    observable_names=(),
    runtime=None,
)

print(f"E = {float(run.mean_energy):.6f} +/- {float(run.stderr_energy):.6f}")
