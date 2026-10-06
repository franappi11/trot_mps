from trot import config

config.configure_once()  # set the precision; before importing jax
import jax.numpy as jnp
import numpy as np
from pyblock3.algebra.core import SparseTensor, SubTensor
from pyblock3.algebra.mpe import MPE
from pyblock3.algebra.mps import MPS
from pyblock3.algebra.symmetry import SZ
from pyscf import ao2mo, gto, scf

from trot.core.system import System
from trot.driver import run_qmc
from trot.gmps.dmrg import hubbard_pyblock3_mpo, make_pyblock3_hamiltonian
from trot.gmps.utils import sd_to_gmps
from trot.ham.hubbard import HamHubbard
from trot.meas.mps import make_mps_meas_ops_hubbard
from trot.prop import mps_cpmc
from trot.prop.types import QmcParamsMps
from trot.trial.mps import (make_mps_trial, make_mps_trial_ops, make_walker_plan, mps_trial_from_pyblock3, rotate_spin,
                            spin_rotation_y)

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

FILLING = 0.875  # electrons per site: 14 electrons on 16 sites
n_up = n_dn = round(FILLING * N / 2)  # 7 and 7
E_FCI = -10.1218936956  # fci_4x4_hubbard.ipynb, (7, 7)
#unrestricted walkers
sys = System(N, (n_up, n_dn), "unrestricted")
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
    dmrg_sweeps=20,  # DMRG from the UHF MPS below
    dmrg_seed=0,  # DMRG noise
    #Adaptive uses the smaller block size that satisfies the tolerance
    occupation_tolerance=1e-10,
    orbital_plan="adaptive",
    #Bond dimensio per spin
    walker_channel_chi=32,
    plan_reference="natural",  # determinant that freezes the conversion circuit: the trial's natural orbitals
    walker_start="natural",  # every walker starts from the trial's natural determinant
)

# ---- trial: UHF (pyscf) from a doped Neel guess, as an MPS (sd_to_gmps), then DMRG at params.trial_chi from it
holes = [(L // 2 - 1, L // 2 - 1), (L // 2, L // 2)]  # (x, y) of the two holes, both on the up sublattice
flip = (L // 2 - 1, L // 2)  # a down site turned up, so that n_up = n_dn = 7

guess = np.zeros((2, N, N))  # diagonal density matrices of the product state
for y in range(L):
    for x in range(L):
        if (x, y) not in holes:
            spin = 0 if (x + y) % 2 == 0 or (x, y) == flip else 1
            guess[spin, y * L + x, y * L + x] = 1.0
print("guess (u = up, d = down, . = hole):")
print("\n".join(" ".join("u" if guess[0, y * L + x, y * L + x] else "d" if guess[1, y * L + x, y * L + x] else "."
                          for x in range(L)) for y in range(L)))

eri = np.zeros((N,) * 4)
for i in range(N):
    eri[i, i, i, i] = U
mol = gto.M(verbose=0)
mol.incore_anyway = True
mol.nelec = (n_up, n_dn)
mf = scf.UHF(mol)
mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: np.eye(N)
mf._eri = ao2mo.restore(8, eri, N)
mf.kernel(guess)
Ca, Cb = mf.mo_coeff[0][:, :n_up], mf.mo_coeff[1][:, :n_dn]

rdm_up, rdm_dn = (np.diag(dm) for dm in mf.make_rdm1())
print(f"\nUHF energy {mf.e_tot:.10f} (converged: {mf.converged})")
print("n_up - n_dn:\n", np.round((rdm_up - rdm_dn).reshape(L, L), 2))
print("n_up + n_dn:\n", np.round((rdm_up + rdm_dn).reshape(L, L), 2))
PHYSICAL = [(0, 0), (1, 0), (0, 1), (1, 1)]  # (n_up, n_dn) of the local states |0>, |up>, |dn>, |up dn>


def gmps_to_pyblock3(tensors, charges):
    """Dense charge-labelled site tensors (sd_to_gmps) as a flat pyblock3 SZ MPS."""
    q = lambda a, b: SZ(a + b, a - b, 0)
    out = []
    for A, q_left, q_right in zip(tensors, charges[:-1], charges[1:]):
        A, blocks = np.asarray(A), []
        for left in sorted(set(map(tuple, q_left))):
            rows = np.flatnonzero((q_left == left).all(axis=1))
            for p, (a, b) in enumerate(PHYSICAL):
                right = (left[0] + a, left[1] + b)
                cols = np.flatnonzero((q_right == right).all(axis=1))
                block = A[np.ix_(rows, [p], cols)]
                if np.any(block):
                    blocks.append(SubTensor(reduced=block, q_labels=(q(*left), q(a, b), q(*right))))
        out.append(SparseTensor(blocks=blocks))
    return MPS(tensors=out).to_flat()


chi_channel = 16
gmps = sd_to_gmps(Ca, Cb, chi=chi_channel)
mps = gmps_to_pyblock3(gmps.tensors, gmps.charges)

hamiltonian = make_pyblock3_hamiltonian(h1, (n_up, n_dn))
mpo = hubbard_pyblock3_mpo(hamiltonian, h1, U)
energy = lambda psi: float(MPE(psi, mpo, psi)[0:2].expectation) / float(psi @ psi)
print(f"MPS bonds {mps.show_bond_dims()}, discarded weight {gmps.discarded:.1e}")
print(f"<MPS|H|MPS> = {energy(mps):.10f}   UHF {mf.e_tot:.10f}")
np.random.seed(params.dmrg_seed)  # DMRG noise
MPE(mps, mpo, mps).dmrg(bdims=[params.trial_chi] * 6, noises=[1e-5] * 4 + [0.0] * 2, dav_thrds=[1e-9], iprint=0,
                        n_sweeps=params.dmrg_sweeps)
e_dmrg = energy(mps)
trial = mps_trial_from_pyblock3(mps, nelec=(n_up, n_dn))
print(f"DMRG trial (from UHF): bonds max {max(trial.bond_dims)}, variational energy {e_dmrg:.10f}, "
      f"E - E_FCI {e_dmrg - E_FCI:+.6f}", flush=True)
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

print(f"E = {float(run.mean_energy):.6f} +/- {float(run.stderr_energy):.6f}   (E - E_FCI {float(run.mean_energy) - E_FCI:+.6f})")
