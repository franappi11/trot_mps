"""CPMC with a DMRG trial -- the MPS route, end to end.

Self-contained. Builds the Hubbard model, runs DMRG with pyblock3 for the trial,
converts every walker to an MPS with a frozen Fishman-White plan, and contracts
<psi_T|phi> charge-blocked, driven by the fast sweep: ONE conversion per
propagation step instead of 2L+2 recontractions.

Extracted from `dmrg_trial_cpmc.ipynb`. The MSD cross-check route is not here --
see `sampled_msd_cpmc.py` for the sampling route.

Everything adjustable is in the CONFIG block below.
"""
import time
from collections import namedtuple

import numpy as np
import scipy.linalg
import jax
import jax.numpy as jnp
import jax.scipy.linalg

jax.config.update("jax_enable_x64", True)

from pyblock3.fcidump import FCIDUMP
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.algebra.mpe import MPE
from pyblock3.algebra.symmetry import SZ

from trot.core.system import System
from trot.core.ops import k_energy, MeasOps
from trot.ham.hubbard import HamHubbard
from trot.trial.uhf import UhfTrial, get_rdm1 as uhf_get_rdm1
from trot.trial.auto import make_auto_trial_ops
from trot.prop import blocks
from trot.prop.cpmc import init_prop_state
from trot.prop.hubbard_cpmc_ops import _build_prop_ctx, make_hubbard_cpmc_ops
from trot.prop.types import PropOps, PropState, QmcParams
from trot import walkers as wk
from trot.walkers import _qr as _qr_t
from trot.driver import run_qmc_energy

# ============================================================== CONFIG
L, n_up, n_down = 24, 12, 12
t, U            = 1.0, 4.0          # hopping, on-site repulsion

CHI             = 64
PLAN_B          = 'adaptive'
DMRG_SWEEPS     = 14
DMRG_SEED       = 0

CHI_WALKER      = 4
CUTOFF_WALKER   = 0.0               # additionally drop s < CUTOFF * s_max per split
BLOCK_ENERGY    = False             # charge-block the energy too (measured: slower)

N_WALKERS   = 200
N_BLOCKS    = 200
N_EQL       = 50
N_PROP      = 20
DT          = 0.01
SEED        = 1234
RESULT_JSON = '/Users/fnappi/trot_mps/trot/gmps/study_results.jsonl'
TAG         = 'mps_L24_cw4'
# ======================================================================

h1 = np.zeros((L, L))
for i in range(L - 1):
    h1[i, i + 1] = h1[i + 1, i] = -t

ham = HamHubbard(h1=jnp.asarray(h1), u=U)
sys_ = System(norb=L, nelec=(n_up, n_down), walker_kind="unrestricted")
eps, mo = np.linalg.eigh(h1)
Ca, Cb = mo[:, :n_up].copy(), mo[:, :n_down].copy()
e_hf = 2.0 * eps[:n_up].sum() + U * sum((Ca[i] @ Ca[i]) * (Cb[i] @ Cb[i]) for i in range(L))
trial_data = UhfTrial(mo_coeff_a=jnp.asarray(Ca), mo_coeff_b=jnp.asarray(Cb))
print(f"L={L} ({n_up},{n_down})  t={t} U={U}   E_HF = {e_hf:.12f}")

def plan_channel_maxB(C):
    """Fishman-White plan with B = n-k always: the remaining block is a genuine
    projector, so its eigenvalues are exactly 0/1 for any walker."""
    U, n = np.asarray(C, float).copy(), C.shape[0]
    occ, Bs, vrefs = np.zeros(n, int), [], []
    for k in range(n - 1):
        B = n - k
        w, W = np.linalg.eigh((U @ U.T)[k : k + B, k : k + B])
        v, occ[k] = (W[:, 0], 0) if w[0] <= 1.0 - w[-1] else (W[:, -1], 1)
        Bs.append(B); vrefs.append(v.copy())
        for j in range(B - 1, 0, -1):
            th = np.arctan2(v[j], v[j - 1])
            c, s = np.cos(th), np.sin(th)
            v[j - 1], v[j] = c * v[j - 1] + s * v[j], 0.0
            U[k + j - 1], U[k + j] = (c * U[k + j - 1] + s * U[k + j],
                                      -s * U[k + j - 1] + c * U[k + j])
    occ[n - 1] = int(round(float(U[n - 1] @ U[n - 1])))
    assert occ.sum() == C.shape[1], "particle number lost during compression"
    return occ, np.array(Bs), vrefs


def plan_channel(C, eps_occ=1e-10):
    """Fishman-White plan with B grown only until a mode isolates.

    The block is enlarged until some eigenvalue of Lambda_B comes within eps_occ
    of 0 or 1. For a low-entangled state that happens at small B, so both the
    gate count and the bond dimension stay far below the maximal-B plan -- which
    is what makes L > 16 reachable at all, since maximal B gives chi = 2^(L/2)
    (1024 at L = 20, 4096 at L = 24).

    The price: the retained block is only APPROXIMATELY idempotent, so the
    replay must use mode="eigh". See `channel_angles`.
    """
    U, n = np.asarray(C, float).copy(), C.shape[0]
    occ, Bs, vrefs = np.zeros(n, int), [], []
    for k in range(n - 1):
        Lam = U @ U.T
        for B in range(2, n - k + 1):                  # grow until a mode isolates
            w, W = np.linalg.eigh(Lam[k : k + B, k : k + B])
            if min(w[0], 1.0 - w[-1]) < eps_occ:
                break
        v, occ[k] = (W[:, 0], 0) if w[0] <= 1.0 - w[-1] else (W[:, -1], 1)
        Bs.append(B); vrefs.append(v.copy())
        for j in range(B - 1, 0, -1):
            th = np.arctan2(v[j], v[j - 1])
            c, s = np.cos(th), np.sin(th)
            v[j - 1], v[j] = c * v[j - 1] + s * v[j], 0.0
            U[k + j - 1], U[k + j] = (c * U[k + j - 1] + s * U[k + j],
                                      -s * U[k + j - 1] + c * U[k + j])
    occ[n - 1] = int(round(float(U[n - 1] @ U[n - 1])))
    assert occ.sum() == C.shape[1], "particle number lost during compression"
    return occ, np.array(Bs), vrefs



def channel_angles(C, plan, xp=jnp):
    """Replay the Givens angles from C with the plan frozen. C must be ORTHONORMAL.   
    """
    occ, Bs, vrefs = plan
    C = xp.asarray(C)
    rows = [C[i] for i in range(C.shape[0])]
    angles = []
    for k, (B, vref) in enumerate(zip(Bs, vrefs)):
        B = int(B)
        Uk = xp.stack(rows[k : k + B])
        M = Uk @ Uk.T
        vr = xp.asarray(vref)
        _, W = xp.linalg.eigh(M)
        v = W[:, -1] if occ[k] == 1 else W[:, 0]
        v = v * xp.sign(v @ vr)
        v = [v[i] for i in range(B)]
        for j in range(B - 1, 0, -1):
            th = xp.arctan2(v[j], v[j - 1])
            c, s = xp.cos(th), xp.sin(th)
            v[j - 1] = c * v[j - 1] + s * v[j]
            p = k + j - 1
            rows[p], rows[p + 1] = (c * rows[p] + s * rows[p + 1],
                                    -s * rows[p] + c * rows[p + 1])
            angles.append((p, th))
    return angles, rows


def V_hat(th):
    """The two-site gate, as a (2,2,2,2) tensor. The reference definition;
    `_gate_pair` fuses it into the contraction."""
    c, s = jnp.cos(th), jnp.sin(th)
    g = jnp.eye(4).at[1, 1].set(c).at[1, 2].set(s).at[2, 1].set(-s).at[2, 2].set(c)
    return g.reshape(2, 2, 2, 2)


def _gate_pair(A, B, th, xp=jnp):
    """(A B) with the gate folded in, as (Dl, 2, 2, Dr)."""
    c, s = xp.cos(th), xp.sin(th)
    t00 = A[:, 0, :] @ B[:, 0, :]
    t01 = A[:, 0, :] @ B[:, 1, :]
    t10 = A[:, 1, :] @ B[:, 0, :]
    t11 = A[:, 1, :] @ B[:, 1, :]
    return xp.stack([xp.stack([t00, c * t01 + s * t10], 1),
                     xp.stack([-s * t01 + c * t10, t11], 1)], 1)


def sector_plan(ql, qr):
    """Static per-charge row/column index sets for one two-site split, plus the
    middle labels and the gather maps back to full shape. NumPy, hence cached."""
    nl, nr = len(ql), len(qr)
    rc = (ql[:, None] + np.arange(2)[None, :]).ravel()
    cc = (qr[None, :] - np.arange(2)[:, None]).ravel()
    secs, qm, rcat, ccat = [], [], [], []
    for nm in sorted(set(rc.tolist()) & set(cc.tolist())):
        r, c = np.where(rc == nm)[0], np.where(cc == nm)[0]
        k = min(len(r), len(c))
        secs.append((r, c, k)); qm += [nm] * k
        rcat.append(r); ccat.append(c)
    rcat, ccat = np.concatenate(rcat), np.concatenate(ccat)
    rmap = np.full(2 * nl, len(rcat), int); rmap[rcat] = np.arange(len(rcat))
    cmap = np.full(2 * nr, len(ccat), int); cmap[ccat] = np.arange(len(ccat))
    return secs, np.array(qm, int), rmap, cmap


_SEC_CACHE = {}

def _sectors(ql, qr):
    key = (ql.tobytes(), len(ql), qr.tobytes(), len(qr))
    if key not in _SEC_CACHE:
        _SEC_CACHE[key] = sector_plan(ql, qr)
    return _SEC_CACHE[key]


def split_full(T, ql, qr):
    """Exact split, full rank in each particle-number sector, via QR."""
    Dl, _, _, Dr = T.shape
    M = T.reshape(Dl * 2, 2 * Dr)
    secs, qm, rmap, cmap = _sectors(ql, qr)
    As, Bs = [], []
    for r, c, k in secs:
        q, rr = jnp.linalg.qr(M[np.ix_(r, c)], mode="reduced")
        As.append(q[:, :k]); Bs.append(rr[:k])
    A = jax.scipy.linalg.block_diag(*As)
    B = jax.scipy.linalg.block_diag(*Bs)
    A = jnp.concatenate([A, jnp.zeros((1, A.shape[1]), A.dtype)], 0)[rmap]
    B = jnp.concatenate([B, jnp.zeros((B.shape[0], 1), B.dtype)], 1)[:, cmap]
    return A.reshape(Dl, 2, -1), B.reshape(-1, 2, Dr), qm




def channel_mps(C, plan):
    """Product state |occ> -> gates in reverse derivation order -> one d=2 MPS.

    Returns (tensors, bond labels, gauge sign). This notebook fixes the gauge by
    the amplitude ratio in Part 3 instead, so the third value is unused here.
    """
    one_hot = (jnp.array([[[1.0], [0.0]]]), jnp.array([[[0.0], [1.0]]]))

    occ = plan[0]
    ts = [one_hot[int(o)] for o in occ]
    qn = [np.zeros(1, int)]
    for o in occ:
        qn.append(qn[-1] + int(o))
    angles, rows = channel_angles(C, plan)
    for p, th in reversed(angles):
        ts[p], ts[p + 1], qn[p + 1] = split_full(_gate_pair(ts[p], ts[p + 1], th),
                                                 qn[p], qn[p + 2])
    gauge = jnp.linalg.det(jnp.stack([rows[i] for i in np.where(occ == 1)[0]]))
    return ts, qn, gauge


def combine(Aa, qna, Ab, qnb):
    """Interleave two d=2 channels into one d=4 MPS, local index n_a + 2 n_b."""
    ts, qn = [], [np.zeros((1, 2), int)]
    for i in range(len(Aa)):
        Dal, _, Dar = Aa[i].shape
        Dbl, _, Dbr = Ab[i].shape
        out = jnp.zeros((Dal, Dbl, 4, Dar, Dbr))
        for na in (0, 1):
            sgn = (-1.0) ** (na * qnb[i])                 # the Jordan-Wigner sign
            for nb in (0, 1):
                out = out.at[:, :, na + 2 * nb, :, :].set(
                    jnp.einsum("ar,b,bs->abrs", Aa[i][:, na, :], sgn, Ab[i][:, nb, :]))
        ts.append(out.reshape(Dal * Dbl, 4, Dar * Dbr))
        qn.append(np.stack([np.repeat(qna[i + 1], Dbr), np.tile(qnb[i + 1], Dar)], 1))
    return ts, qn


def sd_to_mps_qn(ca, cb, plan_a, plan_b):
    """Slater determinant -> (d=4 MPS, per-bond (n_a, n_b) labels). ORTHONORMAL ca/cb."""
    ta, qna, _ = channel_mps(ca, plan_a)
    tb, qnb, _ = channel_mps(cb, plan_b)
    return combine(ta, qna, tb, qnb)


def sd_to_mps(ca, cb, plan_a, plan_b):

    return sd_to_mps_qn(ca, cb, plan_a, plan_b)[0]


def mps_overlap(bra, ket):
    """<bra|ket>, dense. The charge-blocked version is in Part 3."""
    e = jnp.ones((bra[0].shape[0], ket[0].shape[0]))
    for a, b in zip(bra, ket):
        e = jnp.tensordot(a, jnp.tensordot(e, b, ([1], [0])), ([0, 1], [0, 1]))
    return e.reshape(())


def hubbard_mpo(L, t, U):
    """Dw=6 MPO. Bond basis: 0 = nothing started, 1..4 = a hop is pending, 5 = done."""
    I4 = np.eye(4)
    cr_a = np.zeros((4, 4)); cr_a[1, 0] = 1.0; cr_a[3, 2] = 1.0
    cr_b = np.zeros((4, 4)); cr_b[2, 0] = 1.0; cr_b[3, 1] = -1.0
    an_a, an_b = cr_a.T.copy(), cr_b.T.copy()
    n_a, n_b = np.diag([0., 1., 0., 1.]), np.diag([0., 0., 1., 1.])
    P_a, P_b = np.diag([1., -1., 1., -1.]), np.diag([1., 1., -1., -1.])
    W = np.zeros((L, 6, 4, 4, 6))
    for i in range(L):
        W[i, 0, :, :, 0] = I4
        W[i, 5, :, :, 5] = I4
        W[i, 0, :, :, 5] = U * (n_a @ n_b)
        if i < L - 1:
            W[i, 0, :, :, 1] = cr_a @ P_b
            W[i, 0, :, :, 2] = an_a @ P_b
            W[i, 0, :, :, 3] = P_a @ cr_b
            W[i, 0, :, :, 4] = P_a @ an_b
        if i > 0:
            W[i, 1, :, :, 5] = -t * an_a
            W[i, 2, :, :, 5] = -t * cr_a
            W[i, 3, :, :, 5] = -t * an_b
            W[i, 4, :, :, 5] = -t * cr_b
    return W


def apply_mpo(W, ts):
    """(W psi)[i] has bond dimension Dw*chi; the boundary MPO bonds are projected out."""
    out = []
    for i, (w, A) in enumerate(zip(W, ts)):
        A = np.asarray(A)
        if i == 0:
            w = w[0:1]
        if i == len(ts) - 1:
            w = w[:, :, :, 5:6]
        T = np.einsum("apqb,cqd->acpbd", w, A)
        dl, cl, p, dr, cr = T.shape
        out.append(T.reshape(dl * cl, p, dr * cr))
    return out


def compress(ts, tol=1e-13):
    """Left-to-right QR to canonicalise, then right-to-left SVD. Exact at this tol."""
    ts = [np.asarray(t) for t in ts]
    for i in range(len(ts) - 1):
        Dl, d, Dr = ts[i].shape
        q, r = np.linalg.qr(ts[i].reshape(Dl * d, Dr))
        ts[i] = q.reshape(Dl, d, -1)
        ts[i + 1] = np.tensordot(r, ts[i + 1], axes=([1], [0]))
    for i in range(len(ts) - 1, 0, -1):
        Dl, d, Dr = ts[i].shape
        u, sv, vt = np.linalg.svd(ts[i].reshape(Dl, d * Dr), full_matrices=False)
        k = max(int((sv > tol * max(sv[0], 1e-300)).sum()), 1)
        ts[i] = vt[:k].reshape(k, d, Dr)
        ts[i - 1] = np.tensordot(ts[i - 1], u[:, :k] * sv[:k], axes=([2], [0]))
    return ts

def build_hamil(L, U, t=1.0):
    h1e = np.zeros((L, L))
    for i in range(L - 1):
        h1e[i, i + 1] = h1e[i + 1, i] = -t
    g2e = np.zeros((L,) * 4)
    for i in range(L):
        g2e[i, i, i, i] = U
    fd = FCIDUMP(pg="c1", n_sites=L, n_elec=n_up + n_down, twos=n_up - n_down,
                 ipg=0, h1e=h1e, g2e=g2e)
    return Hamiltonian(fd, flat=True)


def run_dmrg(hamil, bdim, n_sweeps=14, seed=0):
    
    np.random.seed(seed)    
    mpo, _ = hamil.build_qc_mpo().compress(cutoff=1e-12)
    mps = hamil.build_mps(bdim)
    dmrg = MPE(mps, mpo, mps).dmrg(bdims=[bdim] * n_sweeps, noises=[1e-5] * 6 + [0],
                                   dav_thrds=[1e-10], iprint=-1, n_sweeps=n_sweeps)
    return mps, float(dmrg.energies[-1])


def spin_occ(q):
    """(n_alpha, n_beta) of an SZ label: n = na + nb, 2Sz = na - nb."""
    return (int(q.n) + int(q.twos)) // 2, (int(q.n) - int(q.twos)) // 2


def flat_blocks(mps, i):
    """(q_labels, shape, data) for every block of site i, with Python-int labels."""
    t = mps[i]
    for k in range(t.n_blocks):
        q = tuple(SZ.from_flat(int(x)) for x in t.q_labels[k])
        sh = tuple(int(x) for x in t.shapes[k])
        yield q, sh, np.asarray(t.data[t.idxs[k]:t.idxs[k + 1]]).reshape(sh)


def densify(mps, L):
    """flat pyblock3 MPS -> dense (Dl, 4, Dr) arrays, local index n_a + 2 n_b."""
    qkey = lambda q: (int(q.n), int(q.twos))
    left, right = [], []
    for i in range(L):
        lo, ro = {}, {}
        for (ql, qp, qr), sh, _ in flat_blocks(mps, i):
            lo[qkey(ql)], ro[qkey(qr)] = sh[0], sh[2]
        left.append(lo); right.append(ro)
    for i in range(L - 1):
        assert left[i + 1] == right[i], f"bond {i+1} mismatch between sites"

    offs = []
    for b in [left[i] for i in range(L)] + [right[L - 1]]:
        o, acc = {}, 0
        for k in sorted(b):
            o[k] = (acc, b[k]); acc += b[k]
        offs.append((o, acc))

    out = []
    for i in range(L):
        A = np.zeros((offs[i][1], 4, offs[i + 1][1]))
        for (ql, qp, qr), sh, dat in flat_blocks(mps, i):
            assert sh[1] == 1, f"physical block dim {sh[1]} != 1"
            na, nb = spin_occ(qp)
            ol, dl = offs[i][0][qkey(ql)]
            orr, dr = offs[i + 1][0][qkey(qr)]
            A[ol:ol + dl, na + 2 * nb, orr:orr + dr] = dat[:, 0, :]
        out.append(A)
    return out


def bond_qns(mps, L):
    """Per-bond (n_a, n_b) label for every index of the densified MPS.
    The same offset bookkeeping `densify` uses, so the labels and the dense
    arrays are guaranteed to agree index for index. This is what the
    charge-blocked contraction in Part 3 needs.
    """
    qkey = lambda q: (int(q.n), int(q.twos))
    left, right = [], []
    for i in range(L):
        lo, ro = {}, {}
        for (ql, qp, qr), sh, _ in flat_blocks(mps, i):
            lo[qkey(ql)], ro[qkey(qr)] = sh[0], sh[2]
        left.append(lo); right.append(ro)
    out = []
    for b in [left[i] for i in range(L)] + [right[L - 1]]:
        lab = []
        for k in sorted(b):
            lab += [((k[0] + k[1]) // 2, (k[0] - k[1]) // 2)] * b[k]   # (n,2Sz)->(na,nb)
        out.append(np.array(lab, int))
    return out

def mps_amp(ts, occ_a, occ_b):
    """<d|MPS>, a bond-dimension-1 contraction."""
    v = np.ones((1, 1))
    for A, a, b in zip(ts, occ_a, occ_b):
        v = v @ np.asarray(A)[:, int(a) + 2 * int(b), :]
    return float(v[0, 0])


def interleaving_sign(ra, rb):
    """(-1)^K relating 'all alpha then all beta' to the interleaved lattice order."""
    return (-1.0) ** sum(int((rb < i).sum()) for i in ra)

# CHI = 64                        # <- set in the CONFIG block at the top
hamil = build_hamil(L, U)
mps_dmrg, E_dmrg_dav = run_dmrg(hamil, CHI, n_sweeps=DMRG_SWEEPS, seed=DMRG_SEED)

LOCAL = {spin_occ(SZ.from_flat(int(c))): k for k, c in enumerate(hamil.basis[0])}
ket_T_np = densify(mps_dmrg, L)
ket_T = [jnp.asarray(t) for t in ket_T_np]

print(f"DMRG chi={CHI}:  bond dims {mps_dmrg.show_bond_dims()}")
print(f"local index map read from hamil.basis: {LOCAL}   -> l = n_a + 2 n_b")
print(f"dense bond dims : {[t.shape[0] for t in ket_T] + [ket_T[-1].shape[-1]]}")
print(f"<psi_T|psi_T>   = {float(mps_overlap(ket_T, ket_T)):.12f}")

#COnverting the trial in the det basis and applying H to compare later.
def interleaving_sign_all(occ):
    """(-1)^K for EVERY (a, b) pair at once.

    K = sum_{i in r_a} #{j in r_b : j < i} is bilinear in the occupation vectors,
    K[a,b] = occ_a . LOW . occ_b with LOW[i,j] = 1 iff j < i, so the whole NS x NS
    sign matrix is two matmuls instead of NS^2 python calls. Bit-identical to
    `interleaving_sign`; at L=16 half filling it is 1.4 s against ~20 minutes,
    and 158 MB as int8 against 1.2 GB as float64.
    """
    n = occ.shape[1]
    o = occ.astype(np.int64)
    LOW = np.tril(np.ones((n, n), np.int64), -1)
    return (1 - 2 * ((o @ LOW @ o.T) & 1)).astype(np.int8)


def mps_amp_all(ts, occ, chunk=1 << 20):
    """<d|MPS> for every (alpha, beta) pair, as an (NS, NS) array.

    The same contraction as `mps_amp`, but all four local states are contracted
    and THEN selected, so each site is one GEMM over a batch of determinants
    rather than NS^2 python-level calls -- 1.8 us per amplitude against 24 us,
    and the two agree to 3e-13. Chunked so the intermediate stays bounded; the
    output itself is NS^2 floats and that is irreducible (1.2 GB at L=16 half
    filling), which is the MSD route being exponential, as advertised.
    """
    ns = occ.shape[0]
    A = [jnp.asarray(t) for t in ts]
    flat = np.empty(ns * ns)
    for lo in range(0, ns * ns, chunk):
        hi = min(lo + chunk, ns * ns)
        ia, ib = np.divmod(np.arange(lo, hi), ns)
        cfg = jnp.asarray(occ[ia] + 2 * occ[ib])
        v = jnp.ones((hi - lo, 1))
        for x, Ax in enumerate(A):
            w = jnp.tensordot(v, Ax, axes=([1], [0]))          # (N, 4, chi')
            v = jnp.take_along_axis(w, cfg[:, x][:, None, None], axis=1)[:, 0, :]
        flat[lo:hi] = np.asarray(v[:, 0])
    return flat.reshape(ns, ns)



Hket_T = [jnp.asarray(t) for t in compress(apply_mpo(hubbard_mpo(L, t, U), ket_T_np))]


e_T_mps = float(mps_overlap(Hket_T, ket_T) / mps_overlap(ket_T, ket_T))


_mk_plan = plan_channel_maxB if PLAN_B == "maximal" else plan_channel
plan_a, plan_b = _mk_plan(Ca), _mk_plan(Cb)
print(f"plan: {PLAN_B} B, {int((plan_a[1]-1).sum())} gates, "
      f"max B {int(plan_a[1].max())}, eigh filter")

ROWS_A, ROWS_B = np.where(plan_a[0] == 1)[0], np.where(plan_b[0] == 1)[0]
ISIGN_REF = interleaving_sign(ROWS_A, ROWS_B)
LOC_REF = [int(a) + 2 * int(b) for a, b in zip(plan_a[0], plan_b[0])]


def amp_exact_ref(ca, cb):
    return ISIGN_REF * jnp.linalg.det(ca[ROWS_A, :]) * jnp.linalg.det(cb[ROWS_B, :])


def amp_mps_ref(ts):
    v = jnp.ones((1, 1))
    for A, l in zip(ts, LOC_REF):
        v = v @ A[:, l, :]
    return v[0, 0]


def _bra(walker):
    ca, cb = walker
    qa, _ = jnp.linalg.qr(ca)
    qb, _ = jnp.linalg.qr(cb)
    return sd_to_mps(qa, qb, plan_a, plan_b), ca, cb


def mps_overlap_T(walker, trial_data=None):
    bra, ca, cb = _bra(walker)
    return (amp_exact_ref(ca, cb) / amp_mps_ref(bra)) * mps_overlap(bra, ket_T)


def mps_energy_T(walker, ham_data=None, meas_ctx=None, trial_data=None):
    bra, _, _ = _bra(walker)
    return mps_overlap(bra, Hket_T) / mps_overlap(bra, ket_T)

"""The charge-blocked overlap: dense, batched, GPU-shaped.

Both MPSs conserve (n_alpha, n_beta), so the environment E[c, r] is nonzero only
where the bra bond index c and the ket bond index r carry the SAME charge. The
point is to use that WITHOUT sparse kernels, which do not port to a GPU: sort
every bond by charge, pad each charge sector to a common size, and carry the
environment as one dense array

    E[q, a, b]        q = charge, a = bra index within q, b = ket index within q

A site update is then two BATCHED GEMMs and one segment-sum, every shape static,
no data-dependent indexing:

    Ein[t] = E[src[t]]                           gather, static indices
    tmp[t] = bra_blk[t]^T Ein[t]                 batched GEMM
    out[t] = tmp[t] ket_blk[t]                   batched GEMM
    E'[q'] = sum_{t : dst[t] = q'} out[t]        segment-sum

`t` runs over the allowed (charge in, physical index, charge out) transitions,
fixed by the bond labels, so the whole layout is planned once in NumPy -- the
same plan/replay split used for the two-site splits.

Padding is zero and the padded entries of the blocks are zero, so this is EXACT:
it returns the same number as the dense contraction, not an approximation.

Two things make it pay here beyond the charge blocking itself:

  * only charges present on BOTH sides survive. A bra charge the ket does not
    have can never reach the last bond, so all work on it is dropped. With a
    chi=8 trial against a chi=256 walker that removes most of the walker.
  * the ket is FIXED, so its padded blocks are extracted once, in NumPy, and
    there is no runtime gather on that side.
"""
PHYS_NAB = np.array([[l % 2, l // 2] for l in range(4)])      # l -> (n_a, n_b)


def _charge_index(qn):
    """{(na,nb): array of bond indices} for one bond's label array."""
    out = {}
    for i, q in enumerate(map(tuple, np.asarray(qn).tolist())):
        out.setdefault(q, []).append(i)
    return {k: np.array(v, int) for k, v in out.items()}


def block_plan(qn_bra, qn_ket):
    """Static layout and transition table for every site. All NumPy."""
    n = len(qn_bra) - 1
    bidx = [_charge_index(q) for q in qn_bra]
    kidx = [_charge_index(q) for q in qn_ket]
    charges, A, B = [], [], []
    for x in range(n + 1):
        q = sorted(set(bidx[x]) & set(kidx[x]))
        charges.append(q)
        A.append(max([len(bidx[x][c]) for c in q], default=0))
        B.append(max([len(kidx[x][c]) for c in q], default=0))

    sites = []
    for x in range(n):
        pos_in = {c: i for i, c in enumerate(charges[x])}
        pos_out = {c: i for i, c in enumerate(charges[x + 1])}
        src, dst, ri, ci, mb, lid = [], [], [], [], [], []
        for c in charges[x]:
            rows = bidx[x][c]
            for l in range(4):
                c2 = (c[0] + PHYS_NAB[l][0], c[1] + PHYS_NAB[l][1])
                if c2 not in pos_out:
                    continue
                cols = bidx[x + 1][c2]
                R = np.zeros((A[x], A[x + 1]), int)      # padded index grids; the
                C = np.zeros((A[x], A[x + 1]), int)      # padding points at 0 and
                M = np.zeros((A[x], A[x + 1]))           # is masked to zero
                R[:len(rows), :len(cols)] = rows[:, None]
                C[:len(rows), :len(cols)] = cols[None, :]
                M[:len(rows), :len(cols)] = 1.0
                src.append(pos_in[c]); dst.append(pos_out[c2])
                ri.append(R); ci.append(C); mb.append(M); lid.append(l)
        sites.append(dict(src=np.array(src, int), dst=np.array(dst, int),
                          ri=np.stack(ri), ci=np.stack(ci), mb=np.stack(mb),
                          lid=np.array(lid, int), nq_out=len(charges[x + 1])))
    return dict(sites=sites, charges=charges, A=A, B=B, bidx=bidx, kidx=kidx, n=n)


def ket_blocks(ket, plan):
    """Pre-extract the fixed side's padded blocks once: no runtime gather there."""
    out = []
    for x, st in enumerate(plan["sites"]):
        blk = np.zeros((len(st["src"]), plan["B"][x], plan["B"][x + 1]))
        for t, (qi, qo, l) in enumerate(zip(st["src"], st["dst"], st["lid"])):
            c, c2 = plan["charges"][x][qi], plan["charges"][x + 1][qo]
            rows, cols = plan["kidx"][x][c], plan["kidx"][x + 1][c2]
            blk[t, :len(rows), :len(cols)] = \
                np.asarray(ket[x])[np.ix_(rows, [l], cols)][:, 0, :]
        out.append(jnp.asarray(blk))
    return out


def blocked_overlap(bra, kblk, plan):
    """<bra|ket> with the charge structure carried as dense padded blocks."""
    E = jnp.ones((1, plan["A"][0], plan["B"][0]))
    for x, (st, kb) in enumerate(zip(plan["sites"], kblk)):
        bb = bra[x][st["ri"], st["lid"][:, None, None], st["ci"]] * st["mb"]
        Ein = E[st["src"]]                                  # (T, A_x,   B_x)
        tmp = jnp.einsum("tij,tik->tjk", bb, Ein)           # (T, A_x+1, B_x)
        out = jnp.einsum("tjk,tkl->tjl", tmp, kb)           # (T, A_x+1, B_x+1)
        E = jax.ops.segment_sum(out, st["dst"], num_segments=st["nq_out"])
    return E.reshape(())


def plan_report(plan):
    """stored environment entries: padded blocks vs exact blocks vs dense."""
    pad = sum(len(plan["charges"][x]) * plan["A"][x] * plan["B"][x]
              for x in range(plan["n"] + 1))
    exact = sum(sum(len(plan["bidx"][x][c]) * len(plan["kidx"][x][c])
                    for c in plan["charges"][x]) for x in range(plan["n"] + 1))
    dense = sum(sum(len(v) for v in plan["bidx"][x].values())
                * sum(len(v) for v in plan["kidx"][x].values())
                for x in range(plan["n"] + 1))
    return dict(dense=dense, padded_blocks=pad, exact_blocks=exact,
                transitions=sum(len(st["src"]) for st in plan["sites"]))



qn_ket = bond_qns(mps_dmrg, L)

"""Two knobs, and a gauge fix that survives them.

WALKER COMPRESSION (CHI_WALKER) caps the CHANNEL bond dimension of the walker
conversion. Carried over from `mps_trial_cpmc.ipynb`: truncation is a discrete
decision, so it goes in the plan -- a NumPy dry run records how many states each
charge sector of each split keeps, which keeps shapes static under vmap while the
singular values are still recomputed from every walker. The splits never need an
SVD of the big matrix; following arXiv:2212.09782, reduce with a QR and read the
Schmidt values off the small hermitian R R^T,

    M = Q R,   R R^T = V S^2 V^T,   M_k = (Q V_k) (V_k^T R)

which is the exact rank-k truncation at about a third of an SVD's cost.

Unlike everything else here this one is an APPROXIMATION: it makes the walker MPS
inexact, so the MPS and MSD routes stop agreeing to machine precision.
CHI_WALKER = None (the default) keeps the conversion exact and the agreement at
1e-14. Note chi_channel is at most 16 at L=8 half filling, so CHI_WALKER >= 16 is
no truncation at all.

THE GAUGE HAD TO CHANGE FOR IT TO BE USABLE. Part 3 fixes the scale between the
walker MPS and the true determinant by matching ONE reference amplitude,

    amp_exact_ref(ca, cb) / amp_mps_ref(bra)

which is exact for an exact MPS -- any amplitude would do -- but divides by a
single truncated number, so it amplifies truncation error violently. The robust
alternative uses no amplitude at all: the conversion is orthogonal up to a sign,

    <MPS|SD(q)> = det(U_rot[occ, :]) = +-1      and     |SD(C)> = det(R) |SD(Q)>

so  g = det(R_a) det(R_b) g_a g_b  with g_sigma handed back by `channel_mps_c`.
Measured against the MSD overlap on the perturbed batch:

    CHI_WALKER   discarded   ratio gauge   det gauge
      None        0.0e+00      3.2e-14      3.1e-15
      8           5.5e-04      6.3e+02      2.6e-01
      4           5.7e-01          inf       9.0e-01

The det gauge is better even in the exact case, and ~2400x better at chi=8. The
residual error at chi=8 is much larger than `discarded` because the bond plan is
frozen on the HF reference while the walkers have drifted away from it.
"""

# CHI_WALKER = 4            # None = exact; int caps the CHANNEL bond dimension      # <- set in the CONFIG block at the top
# CUTOFF_WALKER = 0.0          # additionally drop s < CUTOFF * s_max per split      # <- set in the CONFIG block at the top

BondPlan = namedtuple("BondPlan", "ks qn chi discarded")


def sector_plan_ks(ql, qr, ks=None):
    """sector_plan, keeping only ks[s] states in charge sector s."""
    nl, nr = len(ql), len(qr)
    rc = (ql[:, None] + np.arange(2)[None, :]).ravel()
    cc = (qr[None, :] - np.arange(2)[:, None]).ravel()
    secs, qm, rcat, ccat = [], [], [], []
    for s, nm in enumerate(sorted(set(rc.tolist()) & set(cc.tolist()))):
        r, c = np.where(rc == nm)[0], np.where(cc == nm)[0]
        k = min(len(r), len(c)) if ks is None else int(ks[s])
        if k == 0:
            continue
        secs.append((r, c, k)); qm += [nm] * k
        rcat.append(r); ccat.append(c)
    rcat, ccat = np.concatenate(rcat), np.concatenate(ccat)
    rmap = np.full(2 * nl, len(rcat), int); rmap[rcat] = np.arange(len(rcat))
    cmap = np.full(2 * nr, len(ccat), int); cmap[ccat] = np.arange(len(ccat))
    return secs, np.array(qm, int), rmap, cmap


_SEC_KS_CACHE = {}

def _sectors_ks(ql, qr, ks):
    key = (ql.tobytes(), len(ql), qr.tobytes(), len(qr), ks)
    if key not in _SEC_KS_CACHE:
        _SEC_KS_CACHE[key] = sector_plan_ks(ql, qr, ks)
    return _SEC_KS_CACHE[key]


def split_trunc(T, ql, qr, ks):
    """Compressing split, exact rank-k per sector, via QR + eigh of the small R R^T."""
    Dl, _, _, Dr = T.shape
    M = T.reshape(Dl * 2, 2 * Dr)
    secs, qm, rmap, cmap = _sectors_ks(ql, qr, tuple(ks))
    As, Bs = [], []
    for r, c, k in secs:
        q, rr = jnp.linalg.qr(M[np.ix_(r, c)], mode="reduced")
        _, V = jnp.linalg.eigh(rr @ rr.T)
        Vk = V[:, ::-1][:, :k]
        As.append(q @ Vk); Bs.append(Vk.T @ rr)
    A = jax.scipy.linalg.block_diag(*As)
    B = jax.scipy.linalg.block_diag(*Bs)
    A = jnp.concatenate([A, jnp.zeros((1, A.shape[1]), A.dtype)], 0)[rmap]
    B = jnp.concatenate([B, jnp.zeros((B.shape[0], 1), B.dtype)], 1)[:, cmap]
    return A.reshape(Dl, 2, -1), B.reshape(-1, 2, Dr), qm


def plan_bonds(C, plan, chi_max=None, cutoff=0.0):
    """NumPy dry run: freeze how each split spends its bond budget."""
    occ = plan[0]
    ts = [np.eye(2)[int(o)].reshape(1, 2, 1) for o in occ]
    qn = [np.zeros(1, int)]
    for o in occ:
        qn.append(qn[-1] + int(o))
    angles, _ = channel_angles(np.asarray(C, float), plan, xp=np)

    ks_all, discarded = [], 0.0
    for p, th in reversed(angles):
        T = _gate_pair(ts[p], ts[p + 1], th, xp=np)
        Dl, _, _, Dr = T.shape
        M = T.reshape(Dl * 2, 2 * Dr)
        secs, _, _, _ = sector_plan_ks(qn[p], qn[p + 2])
        svs, blocks = [], []
        for r, c, kfull in secs:
            u, sv, vt = np.linalg.svd(M[np.ix_(r, c)], full_matrices=False)
            svs.append(sv[:kfull]); blocks.append((u, sv, vt))
        flat = np.concatenate(svs)
        order = np.argsort(-flat)
        sel = order[: len(flat) if chi_max is None else min(int(chi_max), len(flat))]
        if cutoff > 0.0 and len(flat):
            sel = sel[flat[sel] > cutoff * flat[order[0]]]
        keep = np.zeros(len(flat), bool); keep[sel] = True
        discarded += float((flat[~keep] ** 2).sum())
        ks, off = [], 0
        for sv in svs:
            ks.append(int(keep[off:off + len(sv)].sum())); off += len(sv)
        ks_all.append(tuple(ks))

        _, qm, rmap, cmap = sector_plan_ks(qn[p], qn[p + 2], tuple(ks))
        As, Bs = [], []
        for (u, sv, vt), k in zip(blocks, ks):
            if k:
                As.append(u[:, :k]); Bs.append(sv[:k, None] * vt[:k])
        A = scipy.linalg.block_diag(*As); B = scipy.linalg.block_diag(*Bs)
        A = np.concatenate([A, np.zeros((1, A.shape[1]))], 0)[rmap]
        B = np.concatenate([B, np.zeros((B.shape[0], 1))], 1)[:, cmap]
        ts[p] = A.reshape(Dl, 2, -1); ts[p + 1] = B.reshape(-1, 2, Dr)
        qn[p + 1] = qm
    return BondPlan(ks=ks_all, qn=qn, chi=max(len(q) for q in qn),
                    discarded=discarded)


def channel_mps_c(C, plan, bond_plan=None):
    """channel_mps with optional compression. Returns (tensors, labels, gauge sign)."""
    
    one_hot = (jnp.array([[[1.0], [0.0]]]), jnp.array([[[0.0], [1.0]]]))
    occ = plan[0]
    ts = [one_hot[int(o)] for o in occ]
    qn = [np.zeros(1, int)]
    for o in occ:
        qn.append(qn[-1] + int(o))
    angles, rows = channel_angles(C, plan)
    for gi, (p, th) in enumerate(reversed(angles)):
        T = _gate_pair(ts[p], ts[p + 1], th)
        if bond_plan is None:
            ts[p], ts[p + 1], qn[p + 1] = split_full(T, qn[p], qn[p + 2])
        else:
            ts[p], ts[p + 1], qn[p + 1] = split_trunc(T, qn[p], qn[p + 2],
                                                      bond_plan.ks[gi])
    return ts, qn, jnp.linalg.det(jnp.stack([rows[i] for i in np.where(occ == 1)[0]]))


# ------------------------------------------------- the walker conversion we use
bp_walk_a = bp_walk_b = None
if CHI_WALKER is not None or CUTOFF_WALKER > 0.0:
    bp_walk_a = plan_bonds(Ca, plan_a, chi_max=CHI_WALKER, cutoff=CUTOFF_WALKER)
    bp_walk_b = plan_bonds(Cb, plan_b, chi_max=CHI_WALKER, cutoff=CUTOFF_WALKER)


def convert_walker(ca, cb):
    """(d=4 tensors, bond labels, gauge) for one walker. No reference amplitude."""
    qa, ra = _qr_t(ca)
    qb, rb = _qr_t(cb)
    ta, qna, ga = channel_mps_c(qa, plan_a, bp_walk_a)
    tb, qnb, gb = channel_mps_c(qb, plan_b, bp_walk_b)
    ts, qn = combine(ta, qna, tb, qnb)
    return ts, qn, ra * rb * ga * gb


# the block plan has to be rebuilt: truncation changes the walker's bond labels
_bra_ref, qn_bra, _ = convert_walker(jnp.asarray(Ca), jnp.asarray(Cb))
blk_plan = block_plan(qn_bra, qn_ket)
ket_T_blk = ket_blocks(ket_T_np, blk_plan)

print("bond dims      walker", [len(q) for q in qn_bra])
print("               trial ", [len(q) for q in qn_ket])
print("charges/bond   walker", [len(set(map(tuple, q.tolist()))) for q in qn_bra])
print("               trial ", [len(set(map(tuple, q.tolist()))) for q in qn_ket])
print("               shared", [len(c) for c in blk_plan["charges"]],
      "  <- the chi=8 trial is what limits this")
print("padded block   bra   ", blk_plan["A"])
print("               ket   ", blk_plan["B"])
_rep = plan_report(blk_plan)
print(f"""
environment entries   dense {_rep['dense']}
                      charge blocks {_rep['exact_blocks']}
                      padded to rectangles {_rep['padded_blocks']}"""
      f"  ({_rep['dense'] / _rep['padded_blocks']:.1f}x fewer than dense)")
print(f"batched transitions per sweep: {_rep['transitions']}")



def blocked_overlap_T(walker, trial_data=None):
    """<psi_T|SD(C)>, charge-blocked, with the det gauge."""
    ca, cb = walker
    bra, _, gc = convert_walker(ca, cb)
    return gc * blocked_overlap(bra, ket_T_blk, blk_plan)

"""Blocking the ENERGY too: what it takes, and why it is off by default.

The overlap could be blocked because both sides carry (n_a, n_b) labels. The
energy needs <phi|H|psi_T>, and `Hket_T = compress(apply_mpo(W, ket_T))` has no
labels. They are lost in two distinct places:

  * `apply_mpo` -- RECOVERABLE. The output bond is the flattened pair (MPO bond
    dw, MPS bond c), so its charge is mpo_q[dw] + mps_q[c], and the Dw=6 Hubbard
    MPO's bond charges follow from how `hubbard_mpo` builds it: states 0
    ("nothing started") and 5 ("done") carry (0,0); state 1 is entered by cr_a so
    it carries (+1,0); state 2 by an_a -> (-1,0); 3 by cr_b -> (0,+1); 4 by an_b
    -> (0,-1). `apply_mpo_qn` below just carries that through, and the assertion
    checks every nonzero obeys q_left + q_phys = q_right.

  * `compress` -- ACTUALLY DESTROYS THEM. It QRs and SVDs the dense reshaped
    matrices with no charge sorting: the returned basis comes back in
    singular-value order, which interleaves charges arbitrarily; a degenerate
    singular value lets the factorisation MIX vectors of different charge; and
    the rank cut `(sv > tol*sv[0]).sum()` is one global count, so it can keep a
    partial sector. A charge-aware compress (sort by charge, factorise per
    sector, truncate per sector) would fix it -- that is the `sector_plan`
    machinery again, on a d=4 chain.

The way out taken here is simpler: SKIP `compress`. It is exact at tol=1e-13
anyway, so the uncompressed H|psi_T> is the same state, just at bond dimension
6*chi = 48 instead of 38. Verified below on random determinant amplitudes.

AND IT STILL DOES NOT PAY. Measured below: the blocked energy is exact to ~1e-14
but SLOWER than the dense one. The H side has 17 shared charges at the middle
bond with very uneven sector sizes, so padding every sector to a rectangle wastes
7378/1782 = 4.1x, which cancels most of the 11x block sparsity, and the extra
transitions cost more XLA ops. Fixing it needs size-bucketed padding (pad within
groups of similar-sized sectors instead of to one global max), which trades flops
for op count -- a bad trade at L=8, where these kernels are launch-bound.

It is also nearly irrelevant: the propagation needs the overlap at every one of
the 2L+2 = 18 field decisions per step against ONE energy evaluation per block.
At 20 prop steps that is 360 against 1, and the fast sweep serves all 360 from
one conversion plus a site loop. So the energy is ~1% of the work either
way. BLOCK_ENERGY is left False; the machinery is here and verified so the
measurement can be repeated rather than re-argued.
"""
# BLOCK_ENERGY = False      # <- set in the CONFIG block at the top

# entering MPO bond state k has created this much charge
MPO_QN = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]])


def apply_mpo_qn(W, ts, qn):
    """apply_mpo, carrying the bond labels.

    `apply_mpo` flattens the left bond as a*cl + c with the MPO index major, so a
    combined index has charge mpo_q[a] + mps_q[c].
    """
    out, qout = [], []
    for i, (w, A) in enumerate(zip(W, ts)):
        A = np.asarray(A)
        mq_l = MPO_QN[0:1] if i == 0 else MPO_QN
        if i == 0:
            w = w[0:1]
        if i == len(ts) - 1:
            w = w[:, :, :, 5:6]
        T = np.einsum("apqb,cqd->acpbd", w, A)
        dl, cl, p, dr, cr = T.shape
        out.append(T.reshape(dl * cl, p, dr * cr))
        qout.append(np.concatenate([mq_l[a][None, :] + qn[i] for a in range(dl)], 0))
    qout.append(MPO_QN[5][None, :] + qn[len(ts)])
    return out, qout


Hts_qn, Hqn = apply_mpo_qn(hubbard_mpo(L, t, U), ket_T_np, qn_ket)

_bad = 0
for _x, _A in enumerate(Hts_qn):
    for _l in range(4):
        _dq = np.array([_l % 2, _l // 2])
        for _r, _c in np.argwhere(np.abs(_A[:, _l, :]) > 1e-12):
            _bad += not np.array_equal(Hqn[_x][_r] + _dq, Hqn[_x + 1][_c])
print(f"H|psi_T>  uncompressed dims {[t.shape[0] for t in Hts_qn] + [1]}")
print(f"          compressed   dims {[t.shape[0] for t in Hket_T] + [1]}")
print(f"          charge-violating nonzeros: {_bad}   (must be 0)")

_rng2 = np.random.default_rng(1)
_e = 0.0
for _ in range(20):
    _oa = np.zeros(L, int); _oa[_rng2.choice(L, n_up, replace=False)] = 1
    _ob = np.zeros(L, int); _ob[_rng2.choice(L, n_down, replace=False)] = 1
    _e = max(_e, abs(mps_amp(Hts_qn, _oa, _ob)
                     - mps_amp([np.asarray(t) for t in Hket_T], _oa, _ob)))
print(f"          max |amp(uncompressed) - amp(compressed)| = {_e:.2e}  (same state)")

H_plan = block_plan(qn_bra, Hqn)
Hket_blk = ket_blocks(Hts_qn, H_plan)
_repH = plan_report(H_plan)
print(f"\nH-side env entries: dense {_repH['dense']}, blocks {_repH['exact_blocks']},"
      f" padded {_repH['padded_blocks']}"
      f"   -> {_repH['dense']/_repH['exact_blocks']:.1f}x sparsity but"
      f" {_repH['padded_blocks']/_repH['exact_blocks']:.1f}x padding waste")


def blocked_energy_T(walker, ham_data=None, meas_ctx=None, trial_data=None):
    """E_loc with BOTH contractions charge-blocked."""
    ca, cb = walker
    bra, _, _ = convert_walker(ca, cb)
    return (blocked_overlap(bra, Hket_blk, H_plan)
            / blocked_overlap(bra, ket_T_blk, blk_plan))


def dense_energy_T(walker, ham_data=None, meas_ctx=None, trial_data=None):
    """E_loc through the dense contraction against the compressed H|psi_T>."""
    ca, cb = walker
    bra, _, _ = convert_walker(ca, cb)
    return mps_overlap(bra, Hket_T) / mps_overlap(bra, ket_T)

energy_T = blocked_energy_T if BLOCK_ENERGY else dense_energy_T

"""THE FAST SWEEP, charge-blocked.

Reconverting the walker to an MPS and recontracting the overlap once per field
trial costs 2L+2 = 18 contractions per propagation step. This does ONE per step and then walks the
site loop with environments, which is where nearly all of the speed comes from.
It carries over from `mps_trial_cpmc.ipynb` unchanged in spirit -- it never used
the spin factorisation, only the fact that the discrete Hubbard-Stratonovich
operator is DIAGONAL in the local basis, so it applies to a spin-entangled DMRG
trial exactly as well.

Two reuses, both inside the blocked representation:

  * one pass of batched GEMMs per site serves both the field marginal and the
    pushed-forward left environment, because both need the same intermediate
        Q[t] = bra_blk[t]^T Lenv[src[t]] ket_blk[t]
    from which  M[l] = sum_{t: l(t)=l} <Q[t], R[x+1][dst[t]]>  and
                Lenv'[q'] = sum_{t: dst[t]=q'} D[l(t)] Q[t].
  * the right environments are built once per walker, backwards, and stay valid:
    sites already passed are folded into Lenv and sites still ahead are untouched.

The gauge prefactor is constant through the sweep, so it is computed once up
front: `convert_walker` returns det(R_a) det(R_b) g_a g_b, which fixes the scale
between the MPS and the true determinant without reference to any amplitude, and
the field operator is diagonal so it scales both descriptions identically.
"""

def right_envs_blocked(bra, kblk, plan):
    """R[x][q, a, b] = contraction of sites x..n-1, charge-blocked.

    R[0][0, 0, 0] is the full overlap, so the pre-loop value comes out free.
    """
    R = [jnp.ones((1, plan["A"][plan["n"]], plan["B"][plan["n"]]))]
    for x in range(plan["n"] - 1, -1, -1):
        st, kb = plan["sites"][x], kblk[x]
        bb = bra[x][st["ri"], st["lid"][:, None, None], st["ci"]] * st["mb"]
        Rin = R[-1][st["dst"]]                                # (T, A_x+1, B_x+1)
        tmp = jnp.einsum("tij,tjk->tik", bb, Rin)             # (T, A_x,   B_x+1)
        out = jnp.einsum("tik,tlk->til", tmp, kb)             # (T, A_x,   B_x)
        R.append(jax.ops.segment_sum(out, st["src"],
                                     num_segments=len(plan["charges"][x])))
    return R[::-1]


def fast_sweep(ca, cb, rns, hs, w_floor):
    """One walker's whole site loop, off a single conversion.

    Returns (ca, cb, overlap before the loop, overlap after, weight factor, nodes).
    """
    bra, _, pref = convert_walker(ca, cb)
    R = right_envs_blocked(bra, ket_T_blk, blk_plan)
    ov_in = pref * R[0][0, 0, 0]

    Lenv = jnp.ones((1, blk_plan["A"][0], blk_plan["B"][0]))
    ov, logw = ov_in, jnp.zeros(())
    nodes = jnp.zeros((), jnp.int32)
    # the two field choices as diagonal one-site operators, l = n_a + 2 n_b
    D0 = jnp.array([1.0, hs[0, 0], hs[0, 1], hs[0, 0] * hs[0, 1]])
    D1 = jnp.array([1.0, hs[1, 0], hs[1, 1], hs[1, 0] * hs[1, 1]])

    for x in range(L):
        st, kb = blk_plan["sites"][x], ket_T_blk[x]
        bb = bra[x][st["ri"], st["lid"][:, None, None], st["ci"]] * st["mb"]
        P = jnp.einsum("tij,tik->tjk", bb, Lenv[st["src"]])    # (T, A_x+1, B_x)
        Q = jnp.einsum("tjk,tkl->tjl", P, kb)                  # (T, A_x+1, B_x+1)

        # the field marginal, per local index, from the same Q
        w = jnp.einsum("tjl,tjl->t", Q, R[x + 1][st["dst"]])   # (T,)
        M = jax.ops.segment_sum(w, st["lid"], num_segments=4)

        ov0, ov1 = pref * (D0 @ M), pref * (D1 @ M)
        r0 = jnp.where(0.5 * (ov0 / ov) < w_floor, 0.0, 0.5 * (ov0 / ov))
        r1 = jnp.where(0.5 * (ov1 / ov) < w_floor, 0.0, 0.5 * (ov1 / ov))
        nodes = nodes + (r0 <= 0.0) + (r1 <= 0.0)        # the constrained-path test
        norm = r0 + r1 + 1.0e-13
        take0 = rns[x] < r0 / norm
        D = jnp.where(take0, D0, D1)
        ov = jnp.where(take0, ov0, ov1)
        logw = logw + jnp.log(norm)
        ca = ca.at[x, :].mul(jnp.where(take0, hs[0, 0], hs[1, 0]))
        cb = cb.at[x, :].mul(jnp.where(take0, hs[0, 1], hs[1, 1]))
        Lenv = jax.ops.segment_sum(D[st["lid"]][:, None, None] * Q, st["dst"],
                                   num_segments=len(blk_plan["charges"][x + 1]))
    return ca, cb, ov_in, ov, jnp.exp(logw), nodes


def init_prop_state_pinned(**kwargs):
    """trot's initializer with a strongly typed node counter.

    `trot.prop.cpmc.init_prop_state` sets node_encounters = jnp.asarray(0), which
    is WEAKLY typed, while every propagation step hands it back strongly typed.
    The aval of the state therefore differs between the first and second call of
    the jitted run_blocks, so the whole block scan is traced and compiled TWICE.
    Pinning the dtype costs one compile instead of two. Nothing under trot/ is
    modified.
    """
    return init_prop_state(**kwargs)._replace(
        node_encounters=jnp.zeros((), dtype=int))


def make_fast_prop_ops(ham_data, walker_kind, overlap_fn):
    """PropOps using the sweep instead of 2L+2 reconversions.

    `overlap_fn` is used for the two one-body half steps, so a run is consistent
    end to end with whichever estimator it is given. Same RNG stream, same weight
    clamps and same population control as trot's reference CPMC step, so two runs
    that differ only in their estimators stay comparable step for step rather
    than merely statistically.
    """
    cpmc_ops = make_hubbard_cpmc_ops(ham_data, walker_kind)

    def step(state, *, params, ham_data, trial_data, trial_ops,
             meas_ops, meas_ctx, prop_ctx):
        key, subkey = jax.random.split(state.rng_key)
        nw = wk.n_walkers(state.walkers)
        rns = jax.random.uniform(subkey, (nw, cpmc_ops.n_sites()))
        w_floor = float(getattr(params, "weight_floor", 1.0e-8))
        w_cap = float(getattr(params, "weight_cap", 100.0))
        damping = float(getattr(params, "pop_control_damping", 0.1))

        # --- first one-body half step, then the whole site loop on one conversion ---
        walkers = cpmc_ops.apply_one_body_half(state.walkers, prop_ctx)
        ca, cb, ov_half, overlaps, wfac, nod = jax.vmap(
            fast_sweep, in_axes=(0, 0, 0, None, None))(
            walkers[0], walkers[1], rns, prop_ctx.hs_constant, w_floor)

        ratio = jnp.real(jnp.real(ov_half) / state.overlaps)
        ratio = jnp.where(ratio < w_floor, 0.0, ratio)
        nodes = jnp.sum(ratio <= 0.0) + jnp.sum(nod)
        weights = jnp.where(state.weights * ratio > w_cap, 0.0, state.weights * ratio)
        weights = weights * wfac
        walkers = (ca, cb)

        # --- second one-body half step: a general rotation, so a full overlap ---
        walkers = cpmc_ops.apply_one_body_half(walkers, prop_ctx)
        overlaps_new = jnp.real(
            jax.vmap(overlap_fn, in_axes=(0, None))(walkers, trial_data))
        ratio = jnp.real(overlaps_new / overlaps)
        ratio = jnp.where(ratio < w_floor, 0.0, ratio)
        nodes = nodes + jnp.sum(ratio <= 0.0)
        weights = jnp.where(weights * ratio > w_cap, 0.0, weights * ratio)

        weights = weights * jnp.exp(prop_ctx.dt * state.pop_control_ene_shift)
        weights = jnp.where(weights > w_cap, 0.0, weights)
        avg_w = jnp.clip(jnp.mean(weights), min=1.0e-300)
        return PropState(
            walkers=walkers, weights=weights, overlaps=overlaps_new, rng_key=key,
            pop_control_ene_shift=state.e_estimate
            - damping * (jnp.log(avg_w) / prop_ctx.dt),
            e_estimate=state.e_estimate,
            node_encounters=state.node_encounters + nodes,
        )

    return PropOps(init_prop_state=init_prop_state_pinned,
                   build_prop_ctx=lambda h, t, p: _build_prop_ctx(h, p.dt),
                   step=step)


                          


# ----------------------------------------------------------------------- run it
params = QmcParams(dt=DT, n_walkers=N_WALKERS, n_prop_steps=N_PROP,
                   n_blocks=N_BLOCKS, n_eql_blocks=N_EQL,
                   weight_floor=1e-8, seed=SEED)


def run(overlap_fn, energy_fn, prop_ops):
    trial_ops = make_auto_trial_ops(sys_, overlap_u=overlap_fn, get_rdm1=uhf_get_rdm1)
    meas_ops = MeasOps(overlap=overlap_fn, kernels={k_energy: energy_fn})
    return run_qmc_energy(sys=sys_, params=params, ham_data=ham, trial_data=trial_data,
                          meas_ops=meas_ops, trial_ops=trial_ops, prop_ops=prop_ops,
                          block_fn=blocks.block)


if __name__ == "__main__":
    prop_ops_fast = make_fast_prop_ops(ham, sys_.walker_kind, blocked_overlap_T)
    _t0 = time.time()
    mean, err, be, bw = run(blocked_overlap_T, energy_T, prop_ops_fast)
    t_run = time.time() - _t0
    _f = lambda x: None if x is None else float(x)
    print(f"\n<psi_T|H|psi_T>  (DMRG trial) {float(e_T_mps):.12f}")
    print(f"CPMC, DMRG trial via MPS-MPS  {_f(mean)} +- {_f(err)}")
    print(f"wall time                     {t_run:.0f} s")
    if RESULT_JSON:
        import json as _json
        rec = dict(kind="mps", tag=TAG, L=L, n_up=n_up, U=U, chi_trial=CHI,
                   plan_b=PLAN_B, chi_walker=CHI_WALKER, n_walkers=N_WALKERS,
                   n_blocks=N_BLOCKS, n_eql=N_EQL, n_prop=N_PROP, dt=DT, seed=SEED,
                   e_hf=float(e_hf), e_trial=float(e_T_mps),
                   e_cpmc=_f(mean), err_cpmc=_f(err), t_run=t_run,
                   chi_walker_d4=int(max(len(q) for q in qn_bra)),
                   gates=int((plan_a[1] - 1).sum()))
        with open(RESULT_JSON, "a") as fh:
            fh.write(_json.dumps(rec) + "\n"); fh.flush()
        print("saved ->", RESULT_JSON)
