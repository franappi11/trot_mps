from trot.config import configure_once
configure_once()  # float64, before jax/pyblock3
import os
import shutil
import time
from pathlib import Path
import block2 as bf
import block2.sz as bx
import numpy as np
from block2 import SZ as BSZ
from block2 import Global, VectorInt
from block2.sz import MPSInfo, StateInfo, VectorStateInfo
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyblock3.algebra.core import SparseTensor, SubTensor
from pyblock3.algebra.mpe import MPE
from pyblock3.algebra.mps import MPS
from pyblock3.algebra.symmetry import SZ
from pyblock3.block2.io import MPSTools
from trot.gmps.dmrg import hubbard_pyblock3_mpo, make_pyblock3_hamiltonian
from trot.gmps.prepare_htrial_saved import read_saved_trial
from trot.trial.mps import mps_trial_from_pyblock3

t = 1
U = 8.0
START = Path(os.environ.get("START", "/mnt/ceph/users/fnappi/trot_trials/uhf_dmrg_8x8_U8/state_00/dmrg_chi2048_block2.npz"))
N_THREADS = int(os.environ.get("OMP_NUM_THREADS", 4))
SCRATCH = Path(os.environ.get("B2_SCRATCH") or os.environ.get("TMPDIR") or "/tmp") / f"block2_{os.environ.get('SLURM_JOB_ID', os.getpid())}"
PHYSICAL = [(0, 0), (1, 0), (0, 1), (1, 1)]  # (n_up, n_dn) of the local states |0>, |up>, |dn>, |up dn>
chi_max = 4096
STACK_GB = int(os.environ.get("STACK_GB", 32))
OUT = Path(os.environ.get("OUT", START.parent / f"dmrg_chi{chi_max}_from2048_block2.npz"))  # never the START file

def shift(sector, p):
    """The (n_up, n_dn) sector after adding the local state p."""
    return (sector[0] + PHYSICAL[p][0], sector[1] + PHYSICAL[p][1])


def site_blocks(A, q_left, q_right):
    """One site's dense tensor (left, 4, right) as {(left sector, p): matrix (rows in that sector, columns in the
    right sector)}, the nonzero blocks only (the blocks gmps_to_pyblock3 makes, as plain arrays)."""
    A, blocks = np.asarray(A), {}
    for left in sorted(set(tuple(int(x) for x in row) for row in q_left)):
        rows = np.flatnonzero((q_left == left).all(axis=1))
        for p in range(4):
            cols = np.flatnonzero((q_right == shift(left, p)).all(axis=1))
            block = A[np.ix_(rows, [p], cols)][:, 0, :]
            if np.any(block):
                blocks[(left, p)] = block
    return blocks


def right_canonicalize(sites):
    """Sector-wise LQ sweep from the last site to site 1 (in place), what pyblock3's canonicalize(0) does: sites 1..N-1
    become right-canonical and site 0 keeps the norm, the form block2 expects ("K" at site 0, then "R"). The L factor
    of each left sector goes into the right index of the previous site; sectors no block reaches are dropped."""
    for i in range(len(sites) - 1, 0, -1):
        blocks, new, factors = sites[i], {}, {}
        for s in sorted({left for left, _ in blocks}):
            keys = [(s, p) for p in range(4) if (s, p) in blocks]
            Q_t, R_t = np.linalg.qr(np.hstack([blocks[k] for k in keys]).T)  # M^T = Q_t R_t, so M = R_t^T Q_t^T
            Q, col = Q_t.T, 0
            for k in keys:
                width = blocks[k].shape[1]
                new[k] = Q[:, col:col + width]
                col += width
            factors[s] = R_t.T  # (rows, k)
        sites[i] = new
        sites[i - 1] = {(s, p): B @ factors[shift(s, p)] for (s, p), B in sites[i - 1].items() if shift(s, p) in factors}


def prune_sectors(sites):
    """Drop the charge sectors that do not run from the vacuum to the target: a bond sector one neighbour never reaches
    or never leaves (zero-amplitude completion vectors). Forward and backward until nothing changes."""
    changed = True
    while changed:
        changed = False
        for i in range(len(sites) - 1):  # site i+1 keeps the left sectors that site i reaches
            reached = {shift(s, p) for (s, p) in sites[i]}
            kept = {k: B for k, B in sites[i + 1].items() if k[0] in reached}
            changed |= len(kept) != len(sites[i + 1])
            sites[i + 1] = kept
        for i in range(len(sites) - 1, 0, -1):  # site i-1 keeps the right sectors that site i starts from
            starts = {s for (s, p) in sites[i]}
            kept = {k: B for k, B in sites[i - 1].items() if shift(*k) in starts}
            changed |= len(kept) != len(sites[i - 1])
            sites[i - 1] = kept


def to_block2_mps(sites, n_up, n_dn, tag="KET"):
    """Right-canonical sector blocks as a block2 MPS (centre at site 0, one-site form): the block2 half of
    pyblock3.block2.io.MPSTools.to_block2, fed from the blocks instead of a pyblock3 MPS. It saves the MPS info
    into the driver's scratch (Global.frame), so the DMRGDriver must exist."""
    n = len(sites)
    left = [{s: B.shape[0] for (s, _), B in site.items()} for site in sites]
    right = [{shift(s, p): B.shape[1] for (s, p), B in site.items()} for site in sites]
    for i in range(n - 1):
        if left[i + 1] != right[i]:
            common = set(left[i + 1]) & set(right[i])
            raise AssertionError(
                f"bond {i + 1}: only in site {i}'s right leg {sorted(set(right[i]) - set(left[i + 1]))}; "
                f"only in site {i + 1}'s left leg {sorted(set(left[i + 1]) - set(right[i]))}; "
                f"dimensions differ (sector, right, left) "
                f"{[(s, right[i][s], left[i + 1][s]) for s in sorted(common) if right[i][s] != left[i + 1][s]]}")
    assert left[0] == {(0, 0): 1} and right[-1] == {(n_up, n_dn): 1}, "the MPS does not run from vacuum to target"

    q = lambda s: BSZ(s[0] + s[1], s[0] - s[1], 0)

    def state_info(dims):  # {(n_up, n_dn): number of states} -> block2 StateInfo
        info = StateInfo()
        info.allocate(len(dims))
        for ix, (s, count) in enumerate(sorted(dims.items())):
            info.quanta[ix] = q(s)
            info.n_states[ix] = count
        info.sort_states()
        return info

    vacuum, target = BSZ(0, 0, 0), BSZ(n_up + n_dn, n_up - n_dn, 0)
    basis = [state_info({s: 1 for s in PHYSICAL}) for _ in range(n)]  # the complete local basis on every site
    minfo = MPSInfo(n, vacuum, target, VectorStateInfo(basis))
    minfo.set_bond_dimension_full_fci(vacuum, vacuum)
    minfo.left_dims[0] = StateInfo(vacuum)
    for i in range(n):
        info = minfo.left_dims[i + 1]  # bond after site i
        info.allocate(len(right[i]))
        for ix, (s, count) in enumerate(sorted(right[i].items())):
            info.quanta[ix] = q(s)
            info.n_states[ix] = count
        info.sort_states()
    minfo.right_dims[n] = StateInfo(vacuum)
    for i in range(n):
        info = minfo.right_dims[i]  # bond before site i, labelled by what the right part carries
        info.allocate(len(left[i]))
        for ix, (s, count) in enumerate(sorted(left[i].items())):
            info.quanta[ix] = target - q(s)
            info.n_states[ix] = count
        info.sort_states()
    minfo.tag = tag
    minfo.bond_dim = minfo.get_max_bond_dimension()
    minfo.save_mutable()
    minfo.save_data(f"{Global.frame.save_dir}/{tag}-mps_info.bin")

    tensors = []
    for site, local in zip(sites, basis):
        tensor = bx.SparseTensor()
        tensor.data = bx.VectorVectorPSSTensor([bx.VectorPSSTensor() for _ in range(local.n)])
        for (s, p), B in site.items():
            im = local.find_state(q(PHYSICAL[p]))
            assert im != -1
            tensor.data[im].append(((q(s), q(shift(s, p))), bf.Tensor(VectorInt([B.shape[0], 1, B.shape[1]]))))
            np.array(tensor.data[im][-1][1], copy=False)[:] = B[:, None, :]
        tensors.append(tensor)
    umps = bx.UnfusedMPS()
    umps.info = minfo
    umps.n_sites = n
    umps.canonical_form = "K" + "R" * (n - 1)
    umps.center = 0
    umps.dot = 1
    umps.tensors = bx.VectorSpTensor(tensors)
    ket = umps.finalize()
    ket, _ = driver.adjust_mps(ket, dot=2)  # two-site form: the first sweep contracts sites 0 and 1
    return ket


def gmps_to_pyblock3(tensors, charges):
    """Dense charge-labelled site tensors (sd_to_gmps) as a flat pyblock3 SZ MPS. Only used to check the direct
    conversion (CHECK_START)."""
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

# warm start from the chi 2048 trial (job 7317956: 120 sweeps from the UHF seed, variational E -43.5434), so the
# schedule below only runs the bonds above 2048
h1, u_start, (n_up, n_dn), tensors, charges, e_start, _ = read_saved_trial(START)
with np.load(START) as z:  # the UHF seed behind the start trial, carried into the output
    seed = int(z["uhf_seed"]) if "uhf_seed" in z.files else -1
    e_uhf = float(z["uhf_energy"]) if "uhf_energy" in z.files else float("nan")
N = len(h1)
L = int(round(N ** 0.5))
bonds = [(y * L + x, y * L + x + 1) for y in range(L) for x in range(L - 1)]
bonds += [(y * L + x, (y + 1) * L + x) for y in range(L - 1) for x in range(L)]
h1_lattice = np.zeros((N, N))
for i, j in bonds:
    h1_lattice[i, j] = h1_lattice[j, i] = - t
if not (np.allclose(h1, h1_lattice) and u_start == U):
    raise SystemExit(f"{START} was made for another system")
print(f"warm start from {START}: {L}x{L}, {n_up} up + {n_dn} down, bond {max(len(q) for q in charges)}, "
      f"saved E = {e_start:.10f} (UHF seed {seed}, E_UHF {e_uhf:.8f})", flush=True)
sites = [site_blocks(A, ql, qr) for A, ql, qr in zip(tensors, charges[:-1], charges[1:])]
del tensors
prune_sectors(sites)
right_canonicalize(sites)

mpo_pb3 = hubbard_pyblock3_mpo(make_pyblock3_hamiltonian(h1, (n_up, n_dn)), h1, U)
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
ket = to_block2_mps(sites, n_up, n_dn, tag="KET")
del sites
e_b2 = b2_energy(ket)
print(f"converted to block2: E = {e_b2:.10f} (saved {e_start:.10f}, difference {e_b2 - e_start:.1e})", flush=True)
if abs(e_b2 - e_start) > 1.0e-8 * abs(e_start):  # a conversion that changed the state would waste the whole run
    raise SystemExit("the converted MPS does not have the saved energy")
# only the bonds above the start's 2048; noise on all but the last 10 sweeps
bond_dims = [2500] * 20 + [3000] * 20 + [3500] * 20 + [4096] * 20
n_sweeps = len(bond_dims)
n_noisy = n_sweeps - 10
noises = [1e-5] * n_noisy + [0.0] * (n_sweeps - n_noisy)
thrds = [1e-9] * n_sweeps
start = time.perf_counter()
e_dmrg = driver.dmrg(mpo, ket, n_sweeps=n_sweeps, bond_dims=bond_dims, noises=noises, thrds=thrds, cutoff=0, iprint=2)
dmrg_seconds = time.perf_counter() - start
sweep_energies = np.array([float(np.ravel(e)[0]) for e in driver._dmrg.energies])
e_var = b2_energy(ket)
print(f"block2 DMRG: {dmrg_seconds:.0f} s, last sweep E = {float(e_dmrg):.10f}, variational E = {e_var:.10f}", flush=True)
mps_out = MPSTools.from_block2(ket).to_flat()
e_out = pb3_energy(mps_out)
print(f"back in pyblock3: E = {e_out:.10f} (block2 {e_var:.10f}, difference {e_out - e_var:.1e})", flush=True)
if abs(e_out - e_var) > 1e-6:
    print("WARNING: the saved state does not have block2's energy: do not use this file before checking", flush=True)

trial = mps_trial_from_pyblock3(mps_out, nelec=(n_up, n_dn))
OUT.parent.mkdir(parents=True, exist_ok=True)
tmp = OUT.with_name(f"{OUT.stem}.tmp.npz")
np.savez(tmp, energy=e_out, rdm1=np.asarray(trial.rdm1), h1=h1, u=U, nelec=np.array([n_up, n_dn]),
         **{f"A{i}": np.asarray(A) for i, A in enumerate(trial.tensors)},
         **{f"q{i}": q for i, q in enumerate(trial.charge_arrays())},
         energy_block2=e_var, energy_last_sweep=float(e_dmrg), sweep_energies=sweep_energies,
         dmrg_seconds=dmrg_seconds, chi=bond_dims[-1], sweeps=n_sweeps, start_file=str(START), start_energy=e_start,
         start_energy_block2=e_b2, uhf_seed=seed, uhf_energy=e_uhf, code="block2")

os.replace(tmp, OUT)  # atomic: a job killed mid-write never leaves a truncated trial behind
print(f"saved to {OUT}", flush=True)
del trial, mps_out  # free the dense trial before the npdm

# <S^z_i S^z_j> for all pairs (row-major sites y * L + x): block2's <n_i,s n_j,s'> (mask [0, 0, 1, 1]: c+_i c_i c+_j c_j,
# an (N, N) array with the diagonal i = j included) for the four spin pairs, S^z = (n_up - n_dn) / 2; <S^x_i S^x_j> =
# (<S^+_i S^-_j> + <S^-_i S^+_j>) / 4 with S^+_i = c+_i,up c_i,dn, as dmrg_trial_6x6_block2_colmajor.py. get_npdm works
# on a copy of the ket.
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

out = OUT.with_name(f"{OUT.stem}_szsz.npz")  # the format of the 6x6 block2 _szsz.npz files
np.savez(out, C=C, C_x=C_x, nn=nn, s2=s2, energy=e_out, h1=h1, u=U, nelec=np.array([n_up, n_dn]), sum_rule=C.sum(),
         trial_file=str(OUT), npdm_seconds=npdm_seconds)
print(f"saved {out}", flush=True)
