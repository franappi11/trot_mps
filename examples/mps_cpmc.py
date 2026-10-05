"""CPMC for the Hubbard chain with a DMRG (MPS) trial and Slater-determinant walkers.

Needs pyblock3 for the DMRG trial (pip install -e .[mps]). Exact energy: -4.235806999130.
"""

import jax.numpy as jnp

from trot import config

config.configure_once()

from trot.core.system import System
from trot.driver import run_qmc
from trot.gmps.dmrg import make_dmrg_trial
from trot.gmps.driver import make_mps_cpmc_ops, run_qmc_mps
from trot.ham.hubbard import HamHubbard, hopping_matrix
from trot.prop import blocks
from trot.prop.types import QmcParamsMps

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

#Call pyblock3 drmg to initilize the trial and the Hamiltonian MPO
trial = make_dmrg_trial(ham, sys, chi=params.trial_chi, n_sweeps=params.dmrg_sweeps).trial
ops = make_mps_cpmc_ops(ham, trial, sys, params)
prop_ctx = ops.prop_ops.build_prop_ctx(ham, ops.trial_ops.get_rdm1(trial), params)
run = run_qmc(
    sys=sys,
    params=params,
    ham_data=ham,
    trial_data=trial,
    trial_ops=ops.trial_ops,
    meas_ops=ops.meas_ops,
    prop_ops=ops.prop_ops,
    prop_ctx=prop_ctx,
    block_fn=blocks.block,
)
print(f"E = {float(run.mean_energy):.6f} +/- {float(run.stderr_energy):.6f}")
