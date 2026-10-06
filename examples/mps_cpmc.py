"""CPMC for the Hubbard chain with a DMRG (MPS) trial and Slater-determinant walkers.

Needs pyblock3 for the DMRG trial (pip install -e .[mps]). Exact energy: -4.235806999130.
"""

import jax.numpy as jnp

from trot import config

config.configure_once()

from trot.core.system import System
from trot.driver import run_qmc
from trot.gmps.dmrg import make_dmrg_trial
from trot.ham.hubbard import HamHubbard, hopping_matrix
from trot.meas.mps import make_mps_meas_ops_hubbard
from trot.prop import mps_cpmc
from trot.prop.types import QmcParamsMps
from trot.trial.mps import make_mps_trial_ops, make_walker_plan

L, U = 8, 4.0
sys = System(L, (L // 2, L // 2), "unrestricted")
ham = HamHubbard(jnp.asarray(hopping_matrix(L, 1.0)), U)
params = QmcParamsMps(
    n_walkers=50,
    n_eql_blocks=10,
    n_blocks=20,
    dt=0.01,
    n_prop_steps=20,
    weight_floor=1e-8,
    seed=2,
    trial_chi=16,
    walker_channel_chi=4,
)

# Call pyblock3 drmg to initilize the trial and the Hamiltonian MPO
trial = make_dmrg_trial(ham, sys, chi=params.trial_chi, n_sweeps=params.dmrg_sweeps).trial
plan = make_walker_plan(ham, trial, sys, params)  # the walker conversion, frozen on the trial's natural orbitals
trial_ops = make_mps_trial_ops(plan)
meas_ops = make_mps_meas_ops_hubbard(plan, energy_kernel=params.energy_kernel)
prop_ops = mps_cpmc.make_prop_ops(ham, sys, plan)
run = run_qmc(
    sys=sys,
    params=params,
    ham_data=ham,
    trial_data=trial,
    trial_ops=trial_ops,
    meas_ops=meas_ops,
    prop_ops=prop_ops,
    prop_ctx=prop_ops.build_prop_ctx(ham, trial_ops.get_rdm1(trial), params),
    block_fn=mps_cpmc.block,  # the MPS block; trot.prop.blocks.block gives the same blocks, with 3 more conversions
)
print(f"E = {float(run.mean_energy):.6f} +/- {float(run.stderr_energy):.6f}")
