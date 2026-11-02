"""drmg_trial_6x6.py with block2 instead of pyblock3, plus all-pairs <S^z_i S^z_j> of the result (the CPMC szsz).

Warm start from the chi 5000 block2 trial (job 7317904), swept up through bond_dims, saved in the trial-cache format,
then <S^z_i S^z_j> for every pair with block2's get_npdm, checked against trot's szsz_dense (the convention the CPMC
uses: S^z = (n_up - n_dn) / 2, SZ_LOCAL). Last, the trial rotated as the CPMC uses it (90 degrees about y) and its
<S^z_i S^z_j> with szsz_dense again: the unrotated state's <S^x_i S^x_j>, checked against block2's, equal to the first
matrix only for a singlet. Runs in ~/.trot312_block2 (block2 and pyblock3); environment variables:
START (warm-start trial), OUT, STACK_GB, B2_SCRATCH/TMPDIR, OMP_NUM_THREADS.
"""

from trot.config import configure_once

configure_once()  # float64, before jax/pyblock3

import os
import shutil
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyblock3.algebra.mpe import MPE
from pyblock3.block2.io import MPSTools

from trot.gmps.block2_tools import prune_sectors, right_canonicalize, site_blocks, to_block2_mps
from trot.gmps.dmrg import hubbard_pyblock3_mpo, make_pyblock3_hamiltonian
from trot.gmps.engine import szsz_dense
from trot.gmps.prepare_htrial_saved import read_saved_trial
from trot.trial.mps import mps_trial_from_pyblock3, rotate_spin, spin_rotation_y

L = 6
U = 8.0
t = 1.0
N = L * L
n_up = n_dn = 16
bonds = [(y * L + x, y * L + x + 1) for y in range(L) for x in range(L - 1)]
bonds += [(y * L + x, (y + 1) * L + x) for y in range(L - 1) for x in range(L)]
h1 = np.zeros((N, N))  # row-major, as drmg_trial_6x6.py
for i, j in bonds:
    h1[i, j] = h1[j, i] = -t

bond_dims = [6000] * 10 + [7000] * 10 + [8000] * 10 
n_sweeps = len(bond_dims)
n_noisy = n_sweeps - 5
noises = [1e-5] * n_noisy + [0.0] * (n_sweeps - n_noisy)
thrds = [1e-9] * n_sweeps

N_THREADS = int(os.environ.get("OMP_NUM_THREADS", 4))
STACK_GB = int(os.environ.get("STACK_GB", 32))
SCRATCH = Path(os.environ.get("B2_SCRATCH") or os.environ.get("TMPDIR") or "/tmp") / f"block2_{os.environ.get('SLURM_JOB_ID', os.getpid())}"
START = Path(os.environ.get("START", "/mnt/ceph/users/fnappi/trot_trials/uhf_dmrg_6x6_U8/state_00/"
                                     "dmrg_chi5000_from2048_down_block2.npz"))
OUT = Path(os.environ.get("OUT", START.parent / f"dmrg_chi{bond_dims[-1]}_from5000_block2.npz"))

# warm start: the chi 5000 trial (itself from the converged chi 2048 one), so the result stays in its state (the ramp
# runs from the UHF seed ended in another, higher one)
h1_start, u_start, nelec_start, tensors, charges, e_start, _ = read_saved_trial(START)
if not (np.allclose(h1_start, h1) and u_start == U and nelec_start == (n_up, n_dn)):
    raise SystemExit(f"{START} was made for another system")
print(f"warm start from {START}: bond {max(len(q) for q in charges)}, saved E = {e_start:.10f}", flush=True)
sites = [site_blocks(A, ql, qr) for A, ql, qr in zip(tensors, charges[:-1], charges[1:])]
del tensors
prune_sectors(sites)
right_canonicalize(sites)

mpo_pb3 = hubbard_pyblock3_mpo(make_pyblock3_hamiltonian(h1, (n_up, n_dn), U), h1, U)
pb3_energy = lambda psi: float(MPE(psi, mpo_pb3, psi)[0:2].expectation) / float(psi @ psi)

SCRATCH.mkdir(parents=True, exist_ok=True)
driver = DMRGDriver(stack_mem=STACK_GB << 30, scratch=str(SCRATCH), symm_type=SymmetryTypes.SZ, n_threads=N_THREADS)
driver.initialize_system(n_sites=N, n_elec=n_up + n_dn, spin=n_up - n_dn)
hop = np.array([[i, j, j, i] for i, j in bonds]).flatten()  # c+_i c_j and c+_j c_i of every bond
b = driver.expr_builder()
b.add_term("cd", hop, -t)
b.add_term("CD", hop, -t)
b.add_term("cdCD", np.repeat(np.arange(N), 4), U)
mpo = driver.get_mpo(b.finalize(), iprint=2)
identity = driver.get_identity_mpo()
b2_energy = lambda ket: driver.expectation(ket, mpo, ket) / driver.expectation(ket, identity, ket)

ket = to_block2_mps(sites, n_up, n_dn, driver, tag="KET")
del sites
e_b2 = b2_energy(ket)
print(f"converted to block2: E = {e_b2:.10f} (saved {e_start:.10f}, difference {e_b2 - e_start:.1e})", flush=True)
if abs(e_b2 - e_start) > 1.0e-8 * abs(e_start):  # a conversion that changed the state would waste the whole run
    raise SystemExit("the converted MPS does not have the saved energy")

start = time.perf_counter()
# tol=0: every sweep of the schedule runs (block2 otherwise stops once the energy change falls below tol)
e_dmrg = driver.dmrg(mpo, ket, n_sweeps=n_sweeps, bond_dims=bond_dims, noises=noises, thrds=thrds, cutoff=0, tol=0,
                     iprint=2)
dmrg_seconds = time.perf_counter() - start
sweep_bond_dims, sweep_discarded, sweep_energies = driver.get_dmrg_results()
sweep_energies = np.array([float(np.ravel(e)[0]) for e in sweep_energies])
e_var = b2_energy(ket)
print(f"block2 DMRG: {dmrg_seconds:.0f} s, last sweep E = {float(e_dmrg):.10f}, variational E = {e_var:.10f}", flush=True)
mps_out = MPSTools.from_block2(ket).to_flat()
e_out = pb3_energy(mps_out)
print(f"back in pyblock3: E = {e_out:.10f} (block2 {e_var:.10f}, difference {e_out - e_var:.1e})", flush=True)
if abs(e_out - e_var) > 1e-6:
    print("WARNING: the saved state does not have block2's energy: do not use this file before checking", flush=True)

# save in the trial-cache format (A{i}, q{i}, energy, rdm1, h1, u, nelec) every other script here reads
trial = mps_trial_from_pyblock3(mps_out, nelec=(n_up, n_dn))
OUT.parent.mkdir(parents=True, exist_ok=True)
tmp = OUT.with_name(f"{OUT.stem}.tmp.npz")
np.savez(tmp, energy=e_out, rdm1=np.asarray(trial.rdm1), h1=h1, u=U, nelec=np.array([n_up, n_dn]),
         **{f"A{i}": np.asarray(A) for i, A in enumerate(trial.tensors)},
         **{f"q{i}": q for i, q in enumerate(trial.charge_arrays())},
         energy_block2=e_var, energy_last_sweep=float(e_dmrg), sweep_energies=sweep_energies,
         sweep_bond_dims=sweep_bond_dims, sweep_discarded=sweep_discarded, dmrg_seconds=dmrg_seconds,
         chi=bond_dims[-1], sweeps=n_sweeps, start_file=str(START), start_energy=e_start, code="block2")
os.replace(tmp, OUT)  # atomic: a job killed mid-write never leaves a truncated trial behind
print(f"saved to {OUT}", flush=True)

# <S^z_i S^z_j> for all pairs: block2's <n_i,s n_j,s'> (mask [0, 0, 1, 1]: c+_i c_i c+_j c_j, an (N, N) array with the
# diagonal i = j included) for the four spin pairs, S^z = (n_up - n_dn) / 2. get_npdm works on a copy of the ket.
start = time.perf_counter()
nn = np.array(driver.get_npdm(ket, npdm_expr=["cdcd", "cdCD", "CDcd", "CDCD"], mask=[0, 0, 1, 1], iprint=1))
npdm_seconds = time.perf_counter() - start
C = 0.25 * (nn[0] - nn[1] - nn[2] + nn[3])
# fixed particle numbers: sum_j <n_i,s n_j,s'> = N_s' <n_i,s>, with <n_i,s> = <n_i,s n_i,s> (checks the i = j terms)
n_a, n_b = np.diag(nn[0]), np.diag(nn[3])
sum_rule_nn = max(np.abs(nn[0].sum(1) - n_up * n_a).max(), np.abs(nn[1].sum(1) - n_dn * n_a).max(),
                  np.abs(nn[2].sum(1) - n_up * n_b).max(), np.abs(nn[3].sum(1) - n_dn * n_b).max())
print(f"get_npdm in {npdm_seconds:.0f} s; sum rule sum_ij C_ij = {C.sum():.2e} (0 for S_z = 0), "
      f"sum_j <n_i n_j> = N <n_i> to {sum_rule_nn:.1e}, N_up {n_a.sum():.10f}, N_dn {n_b.sum():.10f}", flush=True)

# the same matrix from the saved trial with trot's szsz_dense (the CPMC convention), as a check of the above
start = time.perf_counter()
C_trot = np.asarray(szsz_dense([jnp.asarray(A) for A in trial.tensors], [jnp.asarray(A) for A in trial.tensors]))
print(f"szsz_dense in {time.perf_counter() - start:.0f} s; max |C_block2 - C_trot| = {np.abs(C - C_trot).max():.1e}",
      flush=True)

out = OUT.with_name(f"{OUT.stem}_szsz.npz")  # the format of dmrg_szsz_6x6.py, plus the block2 pieces
np.savez(out, C=C, C_trot=C_trot, nn=nn, energy=e_out, h1=h1, u=U, nelec=np.array([n_up, n_dn]), sum_rule=C.sum(),
         trial_file=str(OUT), npdm_seconds=npdm_seconds)
print(f"saved {out}", flush=True)

# <S^x_i S^x_j> = (<S^+_i S^-_j> + <S^-_i S^+_j>) / 4 from block2 (S^+S^+ and S^-S^- vanish at fixed S_z), S^+_i =
# c+_i,up c_i,dn: what the rotated trial's <S^z_i S^z_j> below must be, since R(90)^dag S^z R(90) = -+S^x. Its diagonal
# is <(S^x_i)^2> = <(S^z_i)^2>, the diagonal of C. A failure here costs only this check.
C_x = None
try:
    start = time.perf_counter()
    pm = np.array(driver.get_npdm(ket, npdm_expr=["cDCd", "CdcD"], mask=[0, 0, 1, 1], iprint=1))
    C_x = 0.25 * (pm[0] + pm[1])
    print(f"block2 <S^x_i S^x_j> in {time.perf_counter() - start:.0f} s; its diagonal against C's: "
          f"{np.abs(np.diag(C_x) - np.diag(C)).max():.1e}", flush=True)
except Exception as error:
    print(f"block2 <S^x_i S^x_j> failed ({type(error).__name__}: {error}); going on without that check", flush=True)
driver.finalize()
shutil.rmtree(SCRATCH, ignore_errors=True)

# the trial as the CPMC uses it, rotated about y (ex_6x6_doped_diag.load_trial: rotate_spin(tensors,
# spin_rotation_y(90))), and its <S^z_i S^z_j> with szsz_dense again; 2 sum_ij C_rot = <S^2> of the unrotated state
ROTATION = 90.0
start = time.perf_counter()
tensors = list(trial.tensors)
del trial, mps_out  # one copy of the trial on the host: each site is replaced by its rotation
R = spin_rotation_y(ROTATION)
for i in range(N):
    tensors[i] = jnp.asarray(rotate_spin([tensors[i]], R)[0])
C_rot = np.asarray(szsz_dense(tensors, tensors))
del tensors
s2 = 2.0 * float(C_rot.sum())
check = "" if C_x is None else f"; max |C_rot - C_x(block2)| = {np.abs(C_rot - C_x).max():.1e}"
print(f"rotated by {ROTATION:g} degrees and szsz_dense in {time.perf_counter() - start:.0f} s: <S^2> = 2 sum_ij C_rot "
      f"= {s2:.6f}; max |C_rot - C| = {np.abs(C_rot - C).max():.4e}{check}", flush=True)
out = OUT.with_name(f"{OUT.stem}_szsz_rot{ROTATION:g}.npz")
np.savez(out, C_rot=C_rot, C=C, C_x=np.full_like(C, np.nan) if C_x is None else C_x, s2=s2, rotation_degrees=ROTATION,
         energy=e_out, h1=h1, u=U, nelec=np.array([n_up, n_dn]), trial_file=str(OUT))
print(f"saved {out}", flush=True)
