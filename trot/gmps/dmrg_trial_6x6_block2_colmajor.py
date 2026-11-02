"""dmrg_trial_6x6_block2.py's block2 DMRG with the sites on the chain in column-major order, ramped from the UHF seed.

The other 6x6 scripts put the physical site (x, y) at chain position y * L + x (row-major, x fastest); here it sits at
x * L + y (column-major, the order of trot's square_hopping_matrix). On the open 6x6 square both orders give the same
h1 matrix (bonds (k, k + 1) inside each group of L, and (k, k + L)), so the order only changes where the physical state
lies along the chain: this run is the row-major run of the transposed seed. The start-up line "seed against its
transpose" says whether that is a different calculation at all: a zero there means the same run up to a symmetry
that the row-major chain keeps (chain reversal = 180 degree rotation, spin flip).

The UHF determinant behind the row-major trials (state_00/uhf.npz) is put into chain order, made an MPS (sd_to_gmps,
chi 16 per spin as prepare_trial_6x6.py) and ramped up in block2 with noise, as slurm/dmrg_trial_8x8_block2.py. The
trial is saved in the trial-cache format in chain order (its h1 is the chain-order one, the same matrix, so every
reader here takes it), with order[k] = the row-major site at chain position k. Then <S^z_i S^z_j> and <S^x_i S^x_j>
with get_npdm: in chain order (C, as the other _szsz.npz files and szsz_rotated_6x6.py expect) and in row-major
labels (C_rowmajor, to compare with the row-major trials), and <S^2>. ORDER=row runs the same ramp in row-major order
(a control with the same code and schedule). Runs in ~/.trot312_block2 (block2 and pyblock3); environment variables:
ORDER, UHF_STATE, OUT, STACK_GB, B2_SCRATCH/TMPDIR, OMP_NUM_THREADS.
"""

from trot.config import configure_once

configure_once()  # float64, before jax/pyblock3

import os
import shutil
import time
from pathlib import Path

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyblock3.algebra.mpe import MPE
from pyblock3.block2.io import MPSTools

from trot.gmps.block2_tools import prune_sectors, right_canonicalize, site_blocks, to_block2_mps
from trot.gmps.dmrg import hubbard_pyblock3_mpo, make_pyblock3_hamiltonian
from trot.gmps.utils import sd_to_gmps
from trot.trial.mps import mps_trial_from_pyblock3

L = 6
U = 8.0
t = 1.0
N = L * L
n_up = n_dn = 16
bonds = [(y * L + x, y * L + x + 1) for y in range(L) for x in range(L - 1)]  # row-major sites, as drmg_trial_6x6.py
bonds += [(y * L + x, (y + 1) * L + x) for y in range(L - 1) for x in range(L)]
h1_row = np.zeros((N, N))
for i, j in bonds:
    h1_row[i, j] = h1_row[j, i] = -t

transpose = np.array([y * L + x for x in range(L) for y in range(L)])  # transpose[x * L + y] = y * L + x
ORDER = os.environ.get("ORDER", "column")
if ORDER not in ("column", "row"):
    raise SystemExit(f"ORDER must be column or row, not {ORDER}")
order = transpose if ORDER == "column" else np.arange(N)  # order[k]: the row-major site at chain position k
inv = np.argsort(order)  # inv[r]: the chain position of row-major site r
h1 = h1_row[np.ix_(order, order)]
chain_bonds = [(int(inv[i]), int(inv[j])) for i, j in bonds]
print(f"{ORDER}-major chain: h1 the same matrix as the row-major one: {np.array_equal(h1, h1_row)}", flush=True)

# ramp from the determinant. Warm sweeps on 64 Ice Lake cores took 263 s at bond 2048, 651 at 3024, 1362 at 4098 and
# 2269 at 5000 (job 7317904), so ~11 h; noise on all but the last 5 sweeps, as the warm runs
bond_dims = [512] * 6 + [1024] * 6 + [2048] * 8 + [3000] * 8 + [4000] * 8 + [5000] * 10
n_sweeps = len(bond_dims)
noises = [1e-5] * (n_sweeps - 5) + [0.0] * 5
thrds = [1e-9] * n_sweeps

N_THREADS = int(os.environ.get("OMP_NUM_THREADS", 4))
STACK_GB = int(os.environ.get("STACK_GB", 32))
SCRATCH = Path(os.environ.get("B2_SCRATCH") or os.environ.get("TMPDIR") or "/tmp") / f"block2_{os.environ.get('SLURM_JOB_ID', os.getpid())}"
UHF_STATE = Path(os.environ.get("UHF_STATE", "/mnt/ceph/users/fnappi/trot_trials/uhf_dmrg_6x6_U8/state_00/uhf.npz"))
OUT = Path(os.environ.get("OUT", UHF_STATE.parent / f"dmrg_chi{bond_dims[-1]}_{ORDER}major_block2.npz"))

with np.load(UHF_STATE) as z:
    Ca, Cb = np.asarray(z["Ca"]), np.asarray(z["Cb"])  # rows: row-major sites
    seed, e_uhf = int(z["seed"]), float(z["energy"])
if Ca.shape != (N, n_up) or Cb.shape != (N, n_dn):
    raise SystemExit(f"{UHF_STATE} was made for another system")
print(f"UHF seed {seed} from {UHF_STATE}: E_UHF = {e_uhf:.8f}", flush=True)

# the transposed seed against the seed, also after a 180 degree rotation (anti-transpose) and with the spins flipped:
# a zero means the column-major run is the row-major one up to symmetry
P = np.stack([Ca @ Ca.T, Cb @ Cb.T])  # the determinant's spin densities, row-major
asym = {(name, flip): max(np.abs(P[s][np.ix_(g, g)] - P[s ^ flip]).max() for s in (0, 1))
        for name, g in (("transpose", transpose), ("anti-transpose", N - 1 - transpose)) for flip in (0, 1)}
print("seed against its " + "; ".join(f"{name}{' + spin flip' if flip else ''}: max |dP| = {d:.2e}"
                                      for (name, flip), d in asym.items()), flush=True)

chi_channel = 16  # as prepare_trial_6x6.py
gmps = sd_to_gmps(Ca[order], Cb[order], chi=chi_channel)
sites = [site_blocks(A, ql, qr) for A, ql, qr in zip(gmps.tensors, gmps.charges[:-1], gmps.charges[1:])]
prune_sectors(sites)
right_canonicalize(sites)
print(f"determinant MPS in chain order (chi_channel {chi_channel}, discarded {gmps.discarded:.2e})", flush=True)

mpo_pb3 = hubbard_pyblock3_mpo(make_pyblock3_hamiltonian(h1, (n_up, n_dn), U), h1, U)
pb3_energy = lambda psi: float(MPE(psi, mpo_pb3, psi)[0:2].expectation) / float(psi @ psi)

SCRATCH.mkdir(parents=True, exist_ok=True)
driver = DMRGDriver(stack_mem=STACK_GB << 30, scratch=str(SCRATCH), symm_type=SymmetryTypes.SZ, n_threads=N_THREADS)
driver.initialize_system(n_sites=N, n_elec=n_up + n_dn, spin=n_up - n_dn)
hop = np.array([[i, j, j, i] for i, j in chain_bonds]).flatten()  # c+_i c_j and c+_j c_i of every bond
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
print(f"determinant in block2: E = {e_b2:.10f} (E_UHF {e_uhf:.10f}, equal when nothing was discarded)", flush=True)

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

# save in the trial-cache format (A{i}, q{i}, energy, rdm1, h1, u, nelec) every other script here reads, in chain order
trial = mps_trial_from_pyblock3(mps_out, nelec=(n_up, n_dn))
del mps_out
OUT.parent.mkdir(parents=True, exist_ok=True)
tmp = OUT.with_name(f"{OUT.stem}.tmp.npz")
np.savez(tmp, energy=e_out, rdm1=np.asarray(trial.rdm1), h1=h1, u=U, nelec=np.array([n_up, n_dn]),
         **{f"A{i}": np.asarray(A) for i, A in enumerate(trial.tensors)},
         **{f"q{i}": q for i, q in enumerate(trial.charge_arrays())},
         energy_block2=e_var, energy_last_sweep=float(e_dmrg), sweep_energies=sweep_energies,
         sweep_bond_dims=sweep_bond_dims, sweep_discarded=sweep_discarded, dmrg_seconds=dmrg_seconds,
         chi=bond_dims[-1], sweeps=n_sweeps, chi_channel=chi_channel, gmps_discarded=gmps.discarded,
         start_energy_block2=e_b2, uhf_seed=seed, uhf_energy=e_uhf, uhf_state=str(UHF_STATE), order=order,
         site_order=f"{ORDER}-major: chain position k holds row-major site order[k] = y * L + x", code="block2")
os.replace(tmp, OUT)  # atomic: a job killed mid-write never leaves a truncated trial behind
print(f"saved to {OUT}", flush=True)
del trial

# <S^z_i S^z_j> for all pairs: block2's <n_i,s n_j,s'> (mask [0, 0, 1, 1]: c+_i c_i c+_j c_j, an (N, N) array with the
# diagonal i = j included) for the four spin pairs, S^z = (n_up - n_dn) / 2; <S^x_i S^x_j> = (<S^+_i S^-_j> +
# <S^-_i S^+_j>) / 4 with S^+_i = c+_i,up c_i,dn, as dmrg_trial_6x6_block2.py. get_npdm works on a copy of the ket.
start = time.perf_counter()
nn = np.array(driver.get_npdm(ket, npdm_expr=["cdcd", "cdCD", "CDcd", "CDCD"], mask=[0, 0, 1, 1], iprint=1))
C = 0.25 * (nn[0] - nn[1] - nn[2] + nn[3])
pm = np.array(driver.get_npdm(ket, npdm_expr=["cDCd", "CdcD"], mask=[0, 0, 1, 1], iprint=1))
C_x = 0.25 * (pm[0] + pm[1])
npdm_seconds = time.perf_counter() - start
# fixed particle numbers: sum_j <n_i,s n_j,s'> = N_s' <n_i,s>, with <n_i,s> = <n_i,s n_i,s> (checks the i = j terms)
n_a, n_b = np.diag(nn[0]), np.diag(nn[3])
sum_rule_nn = max(np.abs(nn[0].sum(1) - n_up * n_a).max(), np.abs(nn[1].sum(1) - n_dn * n_a).max(),
                  np.abs(nn[2].sum(1) - n_up * n_b).max(), np.abs(nn[3].sum(1) - n_dn * n_b).max())
s2 = float(C.sum() + 2.0 * C_x.sum())  # sum_ij <S_i . S_j>, with <S^y_i S^y_j> = <S^x_i S^x_j> at fixed S_z
print(f"get_npdm in {npdm_seconds:.0f} s; sum rule sum_ij C_ij = {C.sum():.2e} (0 for S_z = 0), "
      f"sum_j <n_i n_j> = N <n_i> to {sum_rule_nn:.1e}, N_up {n_a.sum():.10f}, N_dn {n_b.sum():.10f}; "
      f"diagonals of C_x and C differ by {np.abs(np.diag(C_x) - np.diag(C)).max():.1e}; <S^2> = {s2:.6f}", flush=True)
driver.finalize()
shutil.rmtree(SCRATCH, ignore_errors=True)

out = OUT.with_name(f"{OUT.stem}_szsz.npz")  # the format of dmrg_trial_6x6_block2.py's, plus the row-major copies
np.savez(out, C=C, C_x=C_x, nn=nn, C_rowmajor=C[np.ix_(inv, inv)], C_x_rowmajor=C_x[np.ix_(inv, inv)], s2=s2,
         order=order, energy=e_out, h1=h1, u=U, nelec=np.array([n_up, n_dn]), sum_rule=C.sum(), trial_file=str(OUT),
         npdm_seconds=npdm_seconds)
print(f"saved {out}", flush=True)
