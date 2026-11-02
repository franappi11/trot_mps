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

L, U, t = 6, 8.0, 1.0
N = L * L
n_up = n_dn = 16
h1 = np.zeros((N, N))
for y in range(L):
    for x in range(L):
        site = y * L + x
        if x + 1 < L:
            h1[site, site + 1] = h1[site + 1, site] = -t
        if y + 1 < L:
            h1[site, site + L] = h1[site + L, site] = -t

eigenvalues, eigenvectors = np.linalg.eigh(h1)
sys = System(N,(n_up, n_dn),'unrestricted')
params = QmcParams(n_walkers=400,
    dt=0.005,
    n_blocks=500,
    n_eql_blocks=50,
    seed=2)

ham = HamHubbard(jnp.array(h1),U)
# with np.load("/mnt/ceph/users/fnappi/trot_trials/uhf_dmrg_6x6_U8/state_00/uhf.npz") as data:
#     seed      = int(data["seed"])       
#     energy    = float(data["energy"])
#     Ca        = data["Ca"]              
#     Cb        = data["Cb"]             
#     rdm1      = data["rdm1"]           
#     converged = bool(data["converged"]) 
#     stable    = bool(data["stable"])    

# trial_data = np.block([
#     [ Ca, Cb],
#     [ Ca, -Cb,]
# ])   
# trial = GhfTrial(jnp.array(trial_data))
# meas_ops = make_ghf_meas_ops_hubbard(sys)
# trial_ops = make_ghf_trial_ops(sys)
# prop_ops = make_prop_ops(ham,'unrestricted')
# prop_ctx = prop_ops.build_prop_ctx(ham, None, params)

# block_fn_uhf, save_walkers_uhf = make_periodic_state_saver(
#     "/mnt/ceph/users/fnappi/trot_walkers/uhf_6x6_doped_U8", every=20, prefix="walkers_uhf", base_block_fn=blocks.block)
# run = run_qmc(
#     sys=sys,
#     params=params,
#     ham_data=ham,
#     trial_data=trial,
#     trial_ops=trial_ops,
#     meas_ops=meas_ops,
#     prop_ops=prop_ops,
#     prop_ctx=prop_ctx,
#     block_fn=block_fn_uhf,
# )
# print("walkers saved to", save_walkers_uhf())



core_idx = np.arange(15)        # levels below ε = 0, doubly occupied
active_idx = np.arange(15, 21)
core = eigenvectors[:, core_idx]
A = eigenvectors[:, active_idx]
nact = A.shape[1]
rho = np.sum(core**2, axis=1)  # Core density per spin.
Ecore = (2 * np.sum(eigenvalues[core_idx])+ U * np.dot(rho, rho))
h1_cas = A.T @ h1 @ A + U * (A.T @ (rho[:, None] * A))
h2_cas = U * np.einsum("pi,pk,pj,pl->ijkl", A, A, A, A)
I = np.eye(nact)
H = (Ecore * np.eye(nact**2)+ np.kron(h1_cas, I)+ np.kron(I, h1_cas)+ h2_cas.reshape(nact**2, nact**2))

basis = []
for i in range(nact):
    Cij = np.zeros((nact, nact))
    Cij[i, i] = 1.0
    basis.append(Cij.ravel())

for i in range(nact):
    for j in range(i + 1, nact):
        Cij = np.zeros((nact, nact))
        Cij[i, j] = Cij[j, i] = 1 / np.sqrt(2)
        basis.append(Cij.ravel())

B = np.column_stack(basis)
H_singlet = B.T @ H @ B
energies, vectors = np.linalg.eigh(H_singlet)
C = (B @ vectors[:, 0]).reshape(nact, nact)
pairs = [(i, j) for i in range(nact) for j in range(nact)]
ci = np.array([C[i, j] for i, j in pairs])
mo = np.zeros((len(pairs), 2 * N, n_up + n_dn))
for k, (i, j) in enumerate(pairs):
    mo[k, :N, :n_up] = np.column_stack((core, A[:, i]))  # up block
    mo[k, N:, n_up:] = np.column_stack((core, A[:, j]))  # down block
keep = np.abs(ci) > 1e-12

trial = MultiGhfTrial(
    ci_coeffs=jnp.array(ci[keep]),
    mo_coeffs=jnp.array(mo[keep]),
    green_real_dtype=jnp.float64,    
    green_complex_dtype=jnp.complex128,)
    
meas_ops = make_multi_ghf_meas_ops_hubbard(sys)
trial_ops = make_multi_ghf_trial_ops(sys)
prop_ops = make_prop_ops(ham,'unrestricted')
prop_ctx = prop_ops.build_prop_ctx(ham, None, params)

block_fn_cas, save_walkers_cas = make_periodic_state_saver(
    "/mnt/ceph/users/fnappi/trot_walkers/uhf_6x6_doped_U8", every=20, prefix="walkers_cas", base_block_fn=blocks.block)
run = run_qmc(
    sys=sys,
    params=params,
    ham_data=ham,
    trial_data=trial,
    trial_ops=trial_ops,
    meas_ops=meas_ops,
    prop_ops=prop_ops,
    prop_ctx=prop_ctx,
    block_fn=block_fn_cas,
)
print("walkers saved to", save_walkers_cas())