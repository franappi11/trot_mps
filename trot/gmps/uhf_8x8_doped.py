from trot.config import configure_once
configure_once()  # float64 for the GMPS conversion
import numpy as np
import jax.numpy as jnp
from trot.trial.ghf import make_ghf_trial_ops
from trot.trial.ghf import GhfTrial
from trot.ham.hubbard import HamHubbard
from trot.driver import run_qmc
from trot.core.system import System
from trot.prop.types import QmcParams
from trot.meas.ghf import make_ghf_meas_ops_hubbard
from trot.prop.cpmc_slow import make_prop_ops
from trot.prop import blocks
from trot.prop.blocks import make_periodic_state_saver
from trot.trial.multi_ghf import MultiGhfTrial
from trot.trial.multi_ghf import MultiGhfTrial, make_multi_ghf_trial_ops
from trot.meas.multi_ghf import make_multi_ghf_meas_ops_hubbard

L, U, t = 8, 8.0, 1.0
N = L * L
n_up = n_dn = 28
h1 = np.zeros((N, N))
for y in range(L):
    for x in range(L):
        site = y * L + x
        if x + 1 < L:
            h1[site, site + 1] = h1[site + 1, site] = -t
        if y + 1 < L:
            h1[site, site + L] = h1[site + L, site] = -t

eigenvalues, eigenvectors = np.linalg.eigh(h1)

sys = System(N, (n_up, n_dn), 'unrestricted')
params = QmcParams(n_walkers=1000, n_blocks=1000, n_eql_blocks=100, seed=2)
ham = HamHubbard(jnp.array(h1),U)
occ = eigenvectors[:, :n_up]
mo = np.zeros((2 * N, n_up + n_dn))
mo[:N, :n_up] = occ  
mo[N:, n_up:] = occ 
trial = GhfTrial(jnp.array(mo))
meas_ops = make_ghf_meas_ops_hubbard(sys)
trial_ops = make_ghf_trial_ops(sys)
prop_ops = make_prop_ops(ham, 'unrestricted')
prop_ctx = prop_ops.build_prop_ctx(ham, None, params)
block_fn, save_walkers = make_periodic_state_saver(
    "/mnt/ceph/users/fnappi/trot_walkers/uhf_8x8_doped_U8", every=20, base_block_fn=blocks.block)
meas_ctx = meas_ops.build_meas_ctx(ham, trial)

state = prop_ops.init_prop_state(
    sys=sys,
    ham_data=ham,
    trial_ops=trial_ops,
    trial_data=trial,
    meas_ops=meas_ops,
    params=params,
    meas_ctx=meas_ctx,
    initial_e_estimate=jnp.asarray(-34.41),  
)

run = run_qmc(
    sys=sys,
    params=params,
    ham_data=ham,
    trial_data=trial,
    trial_ops=trial_ops,
    meas_ops=meas_ops,
    prop_ops=prop_ops,
    prop_ctx=prop_ctx,
    meas_ctx=meas_ctx,
    block_fn=block_fn,
    state=state,
)

print("walkers saved to", save_walkers())
