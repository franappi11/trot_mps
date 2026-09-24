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

CHI             = 64                # DMRG bond dimension of the trial
PLAN_B          = "adaptive"        # "adaptive" (plan_channel) or "maximal"
DMRG_SWEEPS     = 14
DMRG_SEED       = 0

CHI_WALKER      = 20                # None = exact conversion; int caps the CHANNEL bond
CUTOFF_WALKER   = 0.0               # additionally drop s < CUTOFF * s_max per split
BLOCK_ENERGY    = False             # charge-block the energy too (measured: slower)

N_WALKERS   = 32
N_BLOCKS    = 40
N_EQL       = 15
N_PROP      = 20                    # propagation steps per block
DT          = 0.01
SEED        = 1234
RESULT_JSON = ""              # if set, append a JSON record here
TAG         = ""
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
import json as _json, time as _time
_t0 = _time.time()
_h200 = build_hamil(L, U)
_m200, _e200 = run_dmrg(_h200, 200, n_sweeps=DMRG_SWEEPS, seed=DMRG_SEED)
_t200 = _time.time() - _t0
_k200 = [jnp.asarray(x) for x in densify(_m200, L)]
_H200 = [jnp.asarray(x) for x in compress(apply_mpo(hubbard_mpo(L, t, U),
                                                    densify(_m200, L)))]
_e200_var = float(mps_overlap(_H200, _k200) / mps_overlap(_k200, _k200))
_rec = dict(L=L, n_up=n_up, U=U, chi_trial=CHI,
            e_trial_chi64=float(mps_overlap(Hket_T, ket_T) / mps_overlap(ket_T, ket_T)),
            bond_dims_chi64=[int(x.shape[0]) for x in ket_T],
            e_dmrg_chi200=_e200_var, e_dmrg_chi200_dav=float(_e200),
            bond_dims_chi200=[int(x.shape[0]) for x in _k200],
            t_dmrg200=_t200, e_hf=float(e_hf))
with open("/Users/fnappi/trot_mps/trot/gmps/study_dmrg_ref.jsonl", "a") as fh:
    fh.write(_json.dumps(_rec) + "\n"); fh.flush()
print("ref saved", _rec["e_trial_chi64"], _rec["e_dmrg_chi200"])
