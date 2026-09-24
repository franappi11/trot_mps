"""CPMC on an MSD trial SELECTED BY SAMPLING the DMRG state.

Self-contained, and deliberately never builds the full CI expansion. Perfect
sampling draws determinants from |<d|Psi_T>|^2; each draw carries its own exact
amplitude, so the trial is assembled from ONLY the determinants that came up:

    |Psi_tilde> = sum_{d in sampled} c_d |d>,     c_d = <d|Psi_T>

There is no NS x NS matrix anywhere -- not M, not MH, not DOCC. Everything scales
with the number of DISTINCT determinants drawn, not with C(L, n_up)^2. At L=16
half filling that is the difference between 1.66e8 coefficients (1.2 GB) and
however many the sampler actually visits.

The local energy is the Slater-Condon form rather than H applied in the CI basis,
for the same reason: for a product-state bra,

    <d|H|phi> / <d|phi> = k1_a + k1_b + U sum_i n_ia n_ib
    k1_sigma = tr(h1[occ, :] C (C[occ, :])^-1)

so it needs one small solve per DISTINCT alpha and beta string, and no knowledge
of determinants outside the sampled set.

Everything adjustable is in the CONFIG block below.
"""
import dataclasses
import time
from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg

jax.config.update("jax_enable_x64", True)

from pyblock3.fcidump import FCIDUMP
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.algebra.mpe import MPE
from pyblock3.algebra.symmetry import SZ

from trot.core.system import System
from trot.ham.hubbard import HamHubbard
from trot.trial.uhf import UhfTrial, get_rdm1 as uhf_get_rdm1
from trot.prop import blocks, cpmc_slow
from trot.prop.cpmc import init_prop_state
from trot.prop.hubbard_cpmc_ops import make_hubbard_cpmc_ops, _build_prop_ctx
from trot.prop.types import PropOps, PropState, QmcParams
from trot import walkers as wk
from trot.core.ops import k_energy, MeasOps
from trot.trial.auto import make_auto_trial_ops
from trot.driver import run_qmc_energy
from jax import lax

# ============================================================== CONFIG
L, n_up, n_down = 16, 8, 8
t, U            = 1.0, 4.0

CHI             = 16
DMRG_SWEEPS     = 14
DMRG_SEED       = 0

FAST_SWEEP      = True              # False falls back to trot's cpmc_slow
TABLE_MSD       = True              # table/excitation minors; False = brute force
N_REF           = 0                 # table references per channel; 0 = nA/200
N_CHUNKS        = 0                 # walker micro-batches; 0 = size from MEM_BUDGET_GB
MEM_BUDGET_GB   = 6.0               # cap on the per-kernel walker-batched intermediate

N_SAMPLES       = 10000
SAMPLE_SEED     = 1
SAMPLE_CHUNK    = 100_000           # draws per batch, to bound memory

N_WALKERS   = 200
N_BLOCKS    = 200
N_EQL       = 50
N_PROP      = 20
DT          = 0.01
SEED        = 1234
RESULT_JSON = '/Users/fnappi/trot_mps/trot/gmps/study_results.jsonl'
TAG         = 'smp16_L16_N10000'
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


def mps_overlap(bra, ket):
    """<bra|ket> for two MPS given as lists of (Dl, d, Dr) arrays."""
    e = jnp.ones((bra[0].shape[0], ket[0].shape[0]))
    for a, b in zip(bra, ket):
        e = jnp.tensordot(a, jnp.tensordot(e, b, ([1], [0])), ([0, 1], [0, 1]))
    return e.reshape(())


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
    """Return the MPS and its TRUE <Psi_T|H|Psi_T>.

    `dmrg.energies[-1]` is the two-site Davidson eigenvalue from the last sweep,
    which is the energy of the state BEFORE that sweep's bond truncation -- not
    the energy of the MPS actually returned, densified and sampled. The gap is
    monotone in the truncation (at L=8: -5.1e-2, -5.0e-3, -1.1e-3, -1.1e-5 for
    chi = 4, 8, 16, 32), exactly as a pre-truncation eigenvalue must be, and at
    the chi used here it is ~2e-3 -- small, but it is the wrong quantity and it
    was being reported as the trial energy. Take the expectation value against
    the returned MPS instead.
    """
    np.random.seed(seed)
    mpo, _ = hamil.build_qc_mpo().compress(cutoff=1e-12)
    mps = hamil.build_mps(bdim)
    dmrg = MPE(mps, mpo, mps).dmrg(bdims=[bdim] * n_sweeps, noises=[1e-5] * 6 + [0],
                                   dav_thrds=[1e-10], iprint=-1, n_sweeps=n_sweeps)
    e_expect = float(np.dot(mps, mpo @ mps) / np.dot(mps, mps))
    return mps, e_expect, float(dmrg.energies[-1])


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
mps_dmrg, E_dmrg_T, E_dmrg_dav = run_dmrg(hamil, CHI, n_sweeps=DMRG_SWEEPS, seed=DMRG_SEED)

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



Sample = namedtuple("Sample", "config logp amp")


# ---------------------------------------------------------------- canonical form
def right_canonicalize(ts, normalize=True):
    """Right-to-left sweep to right-canonical form, orthogonality centre on site 0.

    sum_l B_x[:, l, :] B_x[:, l, :]^T = I for every x >= 1 -- the property that
    makes the sampling conditionals normalised by construction.

    Done as an LQ, i.e. a QR of the transpose: with M = A_x.reshape(Dl, d*Dr)
    and M^T = Q R, the right factor is Q^T (orthonormal ROWS, so right
    isometric) and R^T is pushed one site left. Bond dimensions can only shrink,
    to min(Dl, d*Dr), and they are python ints, so shapes stay static.

    Returns (tensors, norm). Run it ONCE, then draw as many samples as you like.
    """
    ts = [jnp.asarray(t) for t in ts]
    for x in range(len(ts) - 1, 0, -1):
        Dl, d, Dr = ts[x].shape
        q, r = jnp.linalg.qr(ts[x].reshape(Dl, d * Dr).conj().T, mode="reduced")
        ts[x] = q.conj().T.reshape(-1, d, Dr)
        ts[x - 1] = jnp.tensordot(ts[x - 1], r.conj().T, axes=([2], [0]))
    norm = jnp.linalg.norm(ts[0])
    if normalize:
        ts[0] = ts[0] / jnp.where(norm == 0.0, 1.0, norm)
    return ts, norm


def canonical_error(ts):
    """max_x || sum_l B_x B_x^T - I ||_inf over x >= 1. Zero iff right-canonical."""
    return max(float(jnp.abs(jnp.einsum("axc,bxc->ab", t, t.conj())
                             - jnp.eye(t.shape[0])).max()) for t in ts[1:])


# --------------------------------------------------------------------- the draw
def perfect_sample(ts, key=None, us=None, eps=1e-300):
    """One perfect sample from a NORMALISED RIGHT-CANONICAL MPS.

    Returns (config, logp, amp): the local indices l_x, log p(config), and
    <config|psi> of the normalised state, sign included. |amp|^2 == exp(logp).

    Randomness enters only as L uniforms, either from `key` or supplied as `us`
    -- the same pattern as the fast sweep, so a sampled step can be driven off
    the same RNG stream as a propagation step.

    The site index is an unrolled inverse CDF (d-1 comparisons summed) rather
    than `jnp.searchsorted`, which XLA cannot flatten at d = 4, and rather than
    `jax.random.categorical`, which would take the log of a probability that is
    legitimately zero. Worth 1.5x at this size. `logp` is not accumulated
    separately: the state is normalised, so log p = 2 log|amp| identically.
    """
    n = len(ts)
    us = jax.random.uniform(key, (n,)) if us is None else us
    v = jnp.ones((1,), ts[0].dtype)
    cfg, logamp = [], jnp.zeros(())
    for x in range(n):
        w = jnp.tensordot(v, ts[x], axes=([0], [0]))       # (d, Dr)
        p = jnp.einsum("sr,sr->s", w, w.conj()).real       # sums to ||v||^2 = 1
        c = jnp.cumsum(p)
        u = us[x] * c[-1]
        s = sum((u >= c[k]).astype(jnp.int32) for k in range(p.shape[0] - 1))
        nrm = jnp.sqrt(jnp.maximum(p[s], eps))
        v = w[s] / nrm
        cfg.append(s)
        logamp = logamp + jnp.log(nrm)
    return Sample(jnp.stack(cfg).astype(jnp.int32), 2.0 * logamp,
                  jnp.exp(logamp) * v[0])


def perfect_sample_batch(ts, key, n_samples):
    """`n_samples` independent perfect samples, vmapped over split keys."""
    return jax.vmap(perfect_sample, in_axes=(None, 0))(ts, jax.random.split(key, n_samples))


def sample_mps(ts, key, n_samples=None):
    """Canonicalise, normalise, then draw. Returns (samples, norm)."""
    ts, norm = right_canonicalize(ts)
    s = perfect_sample(ts, key) if n_samples is None else perfect_sample_batch(ts, key, n_samples)
    return s, norm


def mps_amplitude(ts, config):
    """<config|psi>. Contracts all d and then picks, so the shared tensor stays a
    GEMM: the obvious `v @ A[:, config[x], :]` gathers a whole chi x chi matrix
    per sample and goes memory bound under vmap (7-38x slower, measured in the
    companion notebook)."""
    v = jnp.ones((1,), ts[0].dtype)
    for x, A in enumerate(ts):
        v = jnp.tensordot(v, A, axes=([0], [0]))[config[x]]
    return v[0]


def mps_amplitudes(ts, configs):
    return jax.vmap(mps_amplitude, in_axes=(None, 0))(ts, configs)




# ==================================================== sample the trial's determinants
def _pack_config(cfg):
    """Exact distinct-configuration key for any L, two bits per site.

    The obvious key, `cfg @ 4**arange(L)`, silently overflows int64 at L >= 33:
    `4**32` wraps to 0, so every site from 32 on contributes nothing and two
    different determinants collide into one trial entry with no error raised.
    That is in range for this study -- L=48 is on the grid -- and it would show
    up only as a quietly wrong trial. Packing four base-4 digits into each byte
    is exact for every L and lets np.unique compare whole rows at C speed.
    """
    m, n = cfg.shape
    a = np.zeros((m, n + (-n) % 4), np.uint8)
    a[:, :n] = cfg
    g = a.reshape(m, -1, 4).astype(np.uint16)
    return (g * np.array([1, 4, 16, 64], np.uint16)).sum(2).astype(np.uint8)


def sample_determinants(ts, n_samples, key, chunk):
    """Draw determinants from |<d|Psi_T>|^2 and keep the DISTINCT ones.

    Returns (occ_a, occ_b, coeff, hits): occupied SITE INDICES per determinant and
    the exact coefficient c_d = <d|Psi_T>, taken from the sampler's own amplitude
    (the interleaved one) times the interleaving sign, which puts it in the
    all-alpha-then-all-beta convention trot's multi-determinant trial expects.

    Nothing here touches, or needs, the coefficient of a determinant that was not
    drawn -- there is no C(L, n_up)^2 object anywhere in this script.
    """
    can, _ = right_canonicalize(ts)
    n = len(ts)
    draw = jax.jit(perfect_sample_batch, static_argnums=2)
    seen = {}
    left = int(n_samples)
    while left > 0:
        key, sub = jax.random.split(key)
        m = min(chunk, left); left -= m
        s = draw(can, sub, m)
        cfg, amp = np.asarray(s.config), np.asarray(s.amp)
        packed = _pack_config(cfg)
        uniq, first, cnt = np.unique(packed, axis=0, return_index=True,
                                     return_counts=True)
        for u, i, k in zip(uniq, first, cnt):
            c = u.tobytes()
            if c in seen:
                seen[c][2] += int(k)
            else:
                seen[c] = [cfg[i], float(amp[i]), int(k)]
    keys = sorted(seen)
    occ_a = np.array([np.where(seen[c][0] % 2 == 1)[0] for c in keys])
    occ_b = np.array([np.where(seen[c][0] // 2 == 1)[0] for c in keys])
    sgn = np.array([interleaving_sign(a, b) for a, b in zip(occ_a, occ_b)])
    coeff = sgn * np.array([seen[c][1] for c in keys])
    hits = np.array([seen[c][2] for c in keys])
    return occ_a, occ_b, coeff, hits



# ===================================================== table-based MSD minors
"""All nA minor determinants det(C[s, :]) from a handful of factorisations.

The brute force evaluates every string's nocc x nocc minor independently, which
is the entire cost of this calculation. It is avoidable: pick a REFERENCE string
r, factorise once, form the table T = C inv(C[r, :]), and then for a string s
differing from r by k orbitals

    det(C[s,:]) = (-1)^(sum(I) + sum(J)) det(T[cre, J]) det(C[r,:])

      cre = sorted(s - r)   created SITE indices (k of them)
      ann = sorted(r - s)   annihilated SITE indices
      I   = positions of cre within sorted s
      J   = positions of ann within sorted r

so each string costs O(k^3) instead of O(nocc^3). Verified exhaustively at
L=9 nocc=4 (max rel err 1.8e-14, every k); dropping the sign breaks 55 of the
126 strings, so the check is sensitive to the failure mode that matters.

SINGLE REFERENCE IS NOT ENOUGH HERE, and it is worth saying why. Measured on the
actual sampled trials, the mean excitation level off the highest-weight string is
exactly nocc/2 at every system size (4.01 at L=16, 6.01 at L=24, 7.99 at L=32),
which is a flat 6x in flops and no better at large L. These strings are in fact
BROADER than uniformly random ones, because at U/t=4 half filling the Neel string
and its particle-hole conjugate at k = nocc both carry large weight -- 11% each
at L=32. There is no low-excitation core to exploit and truncating on excitation
level would not help either, since the weight is spread flat across k.

What works is several references, chosen by a greedy k-medoids minimisation of
sum_s min_j k(s, r_j)^3, with each string assigned to its nearest. That pulls the
mean k down (4.01 -> 2.16 at L=16 with 8 references) and measures 18x at L=16 and
26x at L=24. The optimum is around nref ~ nA/200: past it the nref table builds
cost more than the k x k determinants they save.

The index tables depend only on the TRIAL, never on the walker, so all of the
bookkeeping below is precomputed once and each excitation group becomes exactly
one gather plus one batched determinant.

THE ONE HAZARD is that the table divides by det(C[r_j, :]), so a reference whose
minor is near a node poisons every string assigned to it, where the brute force
would just contribute zero. Measured over 256 walkers this stays at 1e-15 with
trot's reorthogonalisation, and only reaches 1e-13 with none at all for 120
steps. It is NOT guarded by a runtime fallback: a per-walker branch under vmap
executes both sides and would erase the speedup. Instead the run reports the
smallest reference minor it saw, relative to the median, so a walker drifting
onto a node shows up in the log rather than silently.
"""


def pick_refs(A, L, nref, weights=None, mode="greedy", ncand=512, seed=0):
    """Choose `nref` reference strings from A (nA, nocc) and assign each string.

    `mode="weight"` takes the single highest-weight string.  `mode="greedy"` is
    a k-medoids-style greedy minimisation of sum_s min_j k(s, r_j)^3, with the
    candidate references restricted to `ncand` strings (the highest-weight ones
    plus a random sample) so the cost matrix is ncand x nA instead of nA x nA --
    at nA ~ 2e4 the full matrix would be 3 GB.
    """
    nA, nocc = A.shape
    b = np.zeros((nA, L), np.int16)
    b[np.arange(nA)[:, None], A] = 1
    if mode == "weight":
        chosen = [int(np.argmax(weights))]
    else:
        rng = np.random.default_rng(seed)
        top = np.argsort(-weights)[:ncand // 2] if weights is not None else np.array([], int)
        rest = rng.choice(nA, size=min(nA, ncand), replace=False)
        cand = np.unique(np.concatenate([top, rest]))
        Kc = (nocc - b[cand] @ b.T).astype(np.int64)      # (ncand, nA)
        C3 = Kc ** 3
        chosen, best = [], None
        for _ in range(nref):
            tot = C3.sum(axis=1) if best is None else np.minimum(best[None], C3).sum(axis=1)
            j = int(np.argmin(tot))
            chosen.append(int(cand[j]))
            best = C3[j] if best is None else np.minimum(best, C3[j])
    refs = np.array(chosen, int)
    ov = b @ b[refs].T                                    # (nA, nref)
    assign = np.argmax(ov, axis=1)
    k = nocc - ov[np.arange(nA), assign]
    return refs, assign, k


def build_tables(A, L, refs, assign, k):
    """Group the strings by excitation level; return per-k static index arrays.

    flat[kk] indexes into T.reshape(nref*L*nocc) so a whole (n_k, k, k) batch of
    submatrices is ONE gather.  All of this depends only on the TRIAL, never on
    the walker, so it is precomputed once.
    """
    nA, nocc = A.shape
    R = A[refs]
    bA = np.zeros((nA, L), bool); bA[np.arange(nA)[:, None], A] = True
    bR = np.zeros((len(refs), L), bool); bR[np.arange(len(refs))[:, None], R] = True
    groups = {}
    for kk in range(0, nocc + 1):
        sel = np.where(k == kk)[0]
        if len(sel) == 0:
            continue
        rj = assign[sel]
        if kk == 0:
            groups[0] = dict(dst=sel, ref=rj)
            continue
        n = len(sel)
        # created / annihilated, vectorised over the group
        isc = bA[sel] & ~bR[rj]                      # (n, L) created sites
        isa = bR[rj] & ~bA[sel]                      # (n, L) annihilated sites
        cre = np.where(isc)[1].reshape(n, kk)        # sorted ascending by np.where
        ann = np.where(isa)[1].reshape(n, kk)
        # position of each created site within the sorted string s, and of each
        # annihilated site within the sorted reference r: a cumulative count
        I = (bA[sel].cumsum(1)[np.arange(n)[:, None], cre] - 1)
        J = (bR[rj].cumsum(1)[np.arange(n)[:, None], ann] - 1)
        sgn = (-1.0) ** ((I.sum(1) + J.sum(1)) & 1)
        flat = ((rj[:, None, None] * L + cre[:, :, None].astype(np.int64)) * nocc
                + J[:, None, :]).astype(np.int32).reshape(-1)
        groups[kk] = dict(dst=sel, ref=rj, sgn=sgn, flat=flat, n=n)
    return groups


def pack_channel(A, L, nref=1, weights=None, mode="greedy"):
    refs, assign, k = pick_refs(A, L, nref, weights, mode)
    groups = build_tables(A, L, refs, assign, k)
    out = dict(nA=len(A), nocc=A.shape[1], L=L, nref=len(refs),
               refs=jnp.asarray(A[refs]), k=k)
    g = []
    for kk in sorted(groups):
        d = groups[kk]
        if kk == 0:
            g.append((0, jnp.asarray(d["dst"]), jnp.asarray(d["ref"]), None, None))
        else:
            g.append((kk, jnp.asarray(d["dst"]), jnp.asarray(d["ref"]),
                      jnp.asarray(d["sgn"]), jnp.asarray(d["flat"])))
    out["groups"] = g
    return out


# ------------------------------------------------------------------- the kernel
def _small_det(M, k):
    """det of (..., k, k) with closed forms for k <= 3 (no LU, pure arithmetic)."""
    if k == 1:
        return M[..., 0, 0]
    if k == 2:
        return M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
    if k == 3:
        a, b, c = M[..., 0, 0], M[..., 0, 1], M[..., 0, 2]
        d, e, f = M[..., 1, 0], M[..., 1, 1], M[..., 1, 2]
        g, h, i = M[..., 2, 0], M[..., 2, 1], M[..., 2, 2]
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return jnp.linalg.det(M)


def dets_from_table(C, refs, groups, nA):
    """All nA minor determinants det(C[s,:]) from `nref` reference factorisations.

    C: (L, nocc).  refs: (nref, nocc) reference row sets.
    """
    L, nocc = C.shape
    nref = refs.shape[0]
    Cr = C[refs]                                        # (nref, nocc, nocc)
    dref = jnp.linalg.det(Cr)                           # (nref,)
    # T_j = C inv(Cr_j):  solve Cr_j^T X = C^T, then T = X^T
    X = jnp.linalg.solve(jnp.swapaxes(Cr, -1, -2),
                         jnp.broadcast_to(C.T, (nref, nocc, L)))
    T = jnp.swapaxes(X, -1, -2)                         # (nref, L, nocc)
    Tf = T.reshape(-1)
    out = jnp.zeros((nA,))
    for (k, dst, rj, sgn, flat) in groups:
        if k == 0:
            out = out.at[dst].set(dref[rj])
            continue
        sub = Tf[flat].reshape(-1, k, k)
        out = out.at[dst].set(sgn * _small_det(sub, k) * dref[rj])
    return out

def sampled_trial_energy(occ_a, occ_b, coeff, h1_mat, u_int, nsite):
    """Variational energy of the SAMPLED expansion, <Psi~|H|Psi~> / <Psi~|Psi~>.

    This is the quantity that says what the truncation actually costs, and it is
    not the DMRG energy: |Psi~> keeps only the determinants that came up, with
    their exact coefficients, so it is a strict variational state above the
    DMRG trial it was drawn from.

    Cheap to evaluate despite the pair sum, for two reasons specific to this
    model. The interaction is DIAGONAL in the determinant basis, contributing
    U * (double occupancy) per determinant. The hopping connects only pairs
    differing by moving one electron between ADJACENT sites within one spin
    channel -- and adjacency is what makes the fermionic sign trivial: the usual
    (-1)^(occupied orbitals strictly between i and j) has nothing between i and
    i+1, so every matrix element is exactly h1[i, i+1] with no sign bookkeeping.
    (Verified: over 61,904 non-vanishing adjacent hops up to L=24 the sign is +1
    every time, while 47,827 NON-adjacent hops split 22,156/25,671 between +1 and
    -1 -- so the test would have caught a sign had there been one.) The
    coefficients are already in the all-alpha-then-all-beta convention, so the
    two channels do not interleave either.

    `h1_mat` is read rather than assumed: the adjacency-implies-no-sign argument
    is what is special to nearest-neighbour hopping, and taking the matrix keeps
    this honest for any chain h1 rather than silently hardcoding -t.

    Determinants reached by a hop that were never sampled are simply absent from
    the expansion and contribute nothing -- which is the point: their absence is
    the truncation error being measured.
    """
    n = len(coeff)
    cfg = np.zeros((n, nsite), np.uint8)
    rows = np.arange(n)[:, None]
    cfg[rows, occ_a] += 1
    cfg[rows, occ_b] += 2
    keys = _pack_config(cfg)
    kv = np.ascontiguousarray(keys).view([("", np.uint8)] * keys.shape[1]).ravel()
    order = np.argsort(kv, kind="stable")
    sorted_kv = kv[order]

    def find(sub_cfg):
        """Index of each row of sub_cfg in the expansion, or -1 if absent."""
        q = _pack_config(sub_cfg)
        qv = np.ascontiguousarray(q).view([("", np.uint8)] * q.shape[1]).ravel()
        pos = np.searchsorted(sorted_kv, qv)
        pos = np.clip(pos, 0, len(sorted_kv) - 1)
        hit = sorted_kv[pos] == qv
        return np.where(hit, order[pos], -1)

    c = np.asarray(coeff, float)
    norm = float(c @ c)
    e = u_int * float(((cfg == 3).sum(1) * c * c).sum())

    for bit, other in ((1, 2), (2, 1)):          # alpha channel, then beta
        occ = (cfg & bit) > 0
        for i in range(nsite - 1):
            for src, dst in ((i, i + 1), (i + 1, i)):
                sel = np.where(occ[:, src] & ~occ[:, dst])[0]
                if len(sel) == 0:
                    continue
                sub = cfg[sel].copy()
                sub[:, src] -= bit
                sub[:, dst] += bit
                j = find(sub)
                ok = j >= 0
                if ok.any():
                    e += h1_mat[dst, src] * float((c[sel[ok]] * c[j[ok]]).sum())
    return e / norm


def make_uhf_msd_kernels(occ_a, occ_b, coeff, h1, u_int):
    """UHF (spin-resolved) multi-determinant kernels for |Psi_tilde>.

    trot's only multi-determinant trial is MultiGhfTrial, which stores each
    determinant as a (2L, ne) spin-orbital matrix and carries a (2L, 2L) Green's
    function per (walker, determinant) pair. Our determinants conserve spin, so
    both are exactly block diagonal and half of each is structurally zero -- and
    the Green's storage, nw x nd x (2L)^2, is what puts a wall in front of the
    large-L runs.

    So the trial is kept spin resolved. For a product-state bra,

        <d|phi>          = det(Ca[occ_a, :]) det(Cb[occ_b, :])
        <d|H|phi>/<d|phi> = k1_a + k1_b + U sum_i n_ia n_ib
        k1_sigma          = tr(h1[occ, :] C (C[occ, :])^-1)

    so the cost is one small solve per DISTINCT alpha and beta string plus O(nd)
    arithmetic, and NO Green's function is stored at all. The estimators plug
    into trot through `make_auto_trial_ops` / `MeasOps` exactly as every other
    trial in this project does, and trot's cpmc_slow propagates them (it
    "requires only overlap for a single walker").
    """
    A, ia = np.unique(occ_a, axis=0, return_inverse=True)
    B, ib = np.unique(occ_b, axis=0, return_inverse=True)
    RA, RB = jnp.asarray(A), jnp.asarray(B)

    # reference sets for the table form; nref ~ nA/200 is the measured optimum
    nsite_ = h1.shape[0]
    _wA = np.zeros(len(A)); np.add.at(_wA, ia, coeff ** 2)
    _wB = np.zeros(len(B)); np.add.at(_wB, ib, coeff ** 2)
    if TABLE_MSD:
        nra = N_REF if N_REF > 0 else int(np.clip(len(A) // 200, 1, 64))
        nrb = N_REF if N_REF > 0 else int(np.clip(len(B) // 200, 1, 64))
        pa = pack_channel(A, nsite_, nra, _wA, "greedy" if nra > 1 else "weight")
        pb = pack_channel(B, nsite_, nrb, _wB, "greedy" if nrb > 1 else "weight")
        dets_a = lambda ca: dets_from_table(ca, pa["refs"], pa["groups"], pa["nA"])
        dets_b = lambda cb: dets_from_table(cb, pb["refs"], pb["groups"], pb["nA"])
        print(f"   table MSD: {nra}/{nrb} references, mean excitation "
              f"{pa['k'].mean():.2f}/{pb['k'].mean():.2f} of nocc={A.shape[1]}")
    else:
        pa = pb = None
        dets_a = lambda ca: jnp.linalg.det(ca[RA])
        dets_b = lambda cb: jnp.linalg.det(cb[RB])
    IA, IB = jnp.asarray(ia), jnp.asarray(ib)
    CK = jnp.asarray(coeff)
    h1j = jnp.asarray(h1)
    hA, hB = h1j[RA], h1j[RB]
    docc = jnp.asarray(np.array([len(set(a.tolist()) & set(b.tolist()))
                                 for a, b in zip(occ_a, occ_b)], float))
    # occupation indicators, (L, nA) and (L, nB): MA[x, k] = 1 iff site x is in
    # alpha string k. This is all the fast sweep needs to know about the trial.
    nsite = h1.shape[0]
    MA = np.zeros((nsite, len(A))); MA[A.T, np.arange(len(A))[None, :]] = 1.0
    MB = np.zeros((nsite, len(B))); MB[B.T, np.arange(len(B))[None, :]] = 1.0
    print(f"   UHF MSD trial: {len(coeff)} determinants, {len(A)} distinct alpha "
          f"strings, {len(B)} distinct beta, no Green's functions stored")

    def _dets_k1(C, rows, hrow):
        M = C[rows]                                         # (nA, nocc, nocc)
        d = jnp.linalg.det(M)
        X = jnp.swapaxes(jnp.linalg.solve(
            jnp.swapaxes(M, -1, -2),
            jnp.broadcast_to(C.T, (rows.shape[0],) + C.T.shape)), -1, -2)
        return d, jnp.einsum("kln,knl->k", hrow, X)

    def overlap(walker, trial_data=None):
        ca, cb = walker
        return jnp.sum(CK * dets_a(ca)[IA] * dets_b(cb)[IB])

    def energy(walker, ham_data=None, meas_ctx=None, trial_data=None):
        ca, cb = walker
        da, k1a = _dets_k1(ca, RA, hA)
        db, k1b = _dets_k1(cb, RB, hB)
        w = CK * da[IA] * db[IB]
        return jnp.sum(w * (k1a[IA] + k1b[IB] + u_int * docc)) / jnp.sum(w)

    pack = dict(RA=RA, RB=RB, IA=IA, IB=IB, CK=CK, dets_a=dets_a, dets_b=dets_b,
                refsA=None if pa is None else pa["refs"],
                refsB=None if pb is None else pb["refs"],
                MA=jnp.asarray(MA), MB=jnp.asarray(MB))
    return overlap, energy, pack


# --------------------------------------------------------------- fast sweep
def make_msd_fast_prop_ops(ham_data, walker_kind, overlap_fn, pack, n_chunks=1):
    """PropOps that walks the site loop off ONE determinant evaluation per step.

    The same idea as `make_fast_prop_ops` in `dmrg_trial_cpmc.ipynb`, carried to
    a multi-determinant bra instead of an MPS one. trot's `cpmc_slow` recomputes
    the full overlap for each of the 2L+2 field trials in a step, and for this
    trial one overlap costs a batched determinant over every DISTINCT alpha and
    beta string -- at L=16 that is 2676 8x8 determinants per walker, 34 times a
    step, which is where all the time went.

    It is avoidable because the discrete Hubbard-Stratonovich operator is
    DIAGONAL in site occupation: a field at site x multiplies row x of the walker
    by a scalar h, and for a determinant whose occupied rows are `r`,

        det(C'[r, :]) = h^{[x in r]} det(C[r, :]),

    so the minors do not have to be recomputed at all -- they are rescaled, and
    only for the strings that contain x. One batched determinant at the top of
    the step gives `da`, `db`; every field trial after that is two O(n_det)
    gathers and a dot product. Cost per step falls from (2L+2) batched
    determinants to 2, and the site loop becomes O(L * n_det) arithmetic.

    Same RNG stream, same weight floor and cap, same node test and same
    population control as `cpmc_slow`, so the two are comparable step for step
    and not merely statistically -- which is what `VALIDATE_FAST` checks.
    """
    cpmc_ops = make_hubbard_cpmc_ops(ham_data, walker_kind)
    IA, IB, CK = pack["IA"], pack["IB"], pack["CK"]
    dets_a, dets_b = pack["dets_a"], pack["dets_b"]
    refsA, refsB = pack["refsA"], pack["refsB"]
    MA, MB = pack["MA"], pack["MB"]
    nsite = MA.shape[0]

    def _contract(da, db):
        return jnp.sum(CK * da[IA] * db[IB])

    def msd_sweep(ca, cb, rns, hs, w_floor):
        """One walker's whole site loop off a single pair of minor batches."""
        da, db = dets_a(ca), dets_b(cb)
        ov_in = _contract(da, db)

        def body(carry, x):
            ca, cb, da, db, ov, logw, nodes = carry
            ma, mb = MA[x], MB[x]                      # (nA,), (nB,) in {0, 1}
            da0, db0 = da * (1.0 + ma * (hs[0, 0] - 1.0)), db * (1.0 + mb * (hs[0, 1] - 1.0))
            da1, db1 = da * (1.0 + ma * (hs[1, 0] - 1.0)), db * (1.0 + mb * (hs[1, 1] - 1.0))
            ov0, ov1 = _contract(da0, db0), _contract(da1, db1)
            r0 = jnp.where(0.5 * (ov0 / ov) < w_floor, 0.0, 0.5 * (ov0 / ov))
            r1 = jnp.where(0.5 * (ov1 / ov) < w_floor, 0.0, 0.5 * (ov1 / ov))
            nodes = nodes + (r0 <= 0.0) + (r1 <= 0.0)
            norm = r0 + r1 + 1.0e-13
            take0 = rns[x] < r0 / norm
            da, db = jnp.where(take0, da0, da1), jnp.where(take0, db0, db1)
            ov = jnp.where(take0, ov0, ov1)
            ca = ca.at[x, :].mul(jnp.where(take0, hs[0, 0], hs[1, 0]))
            cb = cb.at[x, :].mul(jnp.where(take0, hs[0, 1], hs[1, 1]))
            return (ca, cb, da, db, ov, logw + jnp.log(norm), nodes), None

        (ca, cb, da, db, ov, logw, nodes), _ = lax.scan(
            body, (ca, cb, da, db, ov_in, jnp.zeros(()), jnp.zeros((), jnp.int32)),
            jnp.arange(nsite))
        return ca, cb, ov_in, ov, jnp.exp(logw), nodes

    def init_prop_state_pinned(**kwargs):
        # trot's initializer sets node_encounters weakly typed, which makes the
        # jitted block scan compile twice. Pinning the dtype costs one compile.
        return init_prop_state(**kwargs)._replace(
            node_encounters=jnp.zeros((), dtype=int))

    def step(state, *, params, ham_data, trial_data, trial_ops,
             meas_ops, meas_ctx, prop_ctx):
        key, subkey = jax.random.split(state.rng_key)
        nw = wk.n_walkers(state.walkers)
        rns = jax.random.uniform(subkey, (nw, cpmc_ops.n_sites()))
        w_floor = float(getattr(params, "weight_floor", 1.0e-8))
        w_cap = float(getattr(params, "weight_cap", 100.0))
        damping = float(getattr(params, "pop_control_damping", 0.1))

        walkers = cpmc_ops.apply_one_body_half(state.walkers, prop_ctx)
        ca, cb, ov_half, overlaps, wfac, nod = wk.vmap_chunked(
            msd_sweep, n_chunks, in_axes=(0, 0, 0, None, None))(
            walkers[0], walkers[1], rns, prop_ctx.hs_constant, w_floor)

        ratio = jnp.real(jnp.real(ov_half) / state.overlaps)
        ratio = jnp.where(ratio < w_floor, 0.0, ratio)
        nodes = jnp.sum(ratio <= 0.0) + jnp.sum(nod)
        weights = jnp.where(state.weights * ratio > w_cap, 0.0, state.weights * ratio)
        weights = weights * wfac
        walkers = (ca, cb)

        # the second half step is a general rotation, not diagonal, so the minors
        # genuinely change and this one overlap has to be paid in full
        walkers = cpmc_ops.apply_one_body_half(walkers, prop_ctx)
        overlaps_new = jnp.real(wk.vmap_chunked(
            overlap_fn, n_chunks, in_axes=(0, None))(walkers, trial_data))
        ratio = jnp.where(jnp.real(overlaps_new / overlaps) < w_floor, 0.0,
                          jnp.real(overlaps_new / overlaps))
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

if __name__ == "__main__":
    _t0 = time.perf_counter()
    occ_a, occ_b, coeff, hits = sample_determinants(
        ket_T, N_SAMPLES, jax.random.PRNGKey(SAMPLE_SEED), SAMPLE_CHUNK)
    t_sample = time.perf_counter() - _t0
    print(f"\n{N_SAMPLES} draws in {t_sample:.1f} s")
    print(f"   distinct determinants kept : {len(coeff)}")
    print(f"   weight captured sum|c_k|^2 : {float((coeff**2).sum()):.9f}")
    print(f"   rarest kept, p_hat         : {hits.min()/N_SAMPLES:.2e}")

    _t0 = time.perf_counter()
    e_trial_samp = sampled_trial_energy(occ_a, occ_b, coeff, h1, U, L)
    print(f"   DMRG trial <Psi_T|H|Psi_T> (chi={CHI}) {float(E_dmrg_T):.9f}"
          f"   (Davidson eigenvalue {float(E_dmrg_dav):.9f})")
    print(f"   sampled trial energy <H> variational {e_trial_samp:.9f}"
          f"   ({time.perf_counter()-_t0:.1f} s)")

    overlap_s, energy_s, pack = make_uhf_msd_kernels(occ_a, occ_b, coeff, h1, U)

    # The kernels are batched over walkers, so the peak live array is
    # n_walkers x n_strings x nocc x max(L, nocc) doubles -- the broadcast C^T
    # inside the energy's solve, the largest of the three. At L=16 that is
    # 0.4 GB and one chunk is fine; at L=48 with 20k distinct strings it is tens
    # of GB, which is what CHUNKS is for. Chunking costs nothing but serialism.
    n_chunks = N_CHUNKS
    if n_chunks <= 0:
        _ns = max(pack["MA"].shape[1], pack["MB"].shape[1])
        _bytes = 3 * N_WALKERS * _ns * max(n_up, n_down) * max(L, n_up) * 8
        n_chunks = max(1, int(np.ceil(_bytes / (MEM_BUDGET_GB * 1e9))))
    params = params._replace(n_chunks=n_chunks) if hasattr(params, "_replace") \
        else dataclasses.replace(params, n_chunks=n_chunks)
    print(f"   peak walker-batched intermediate -> n_chunks = {n_chunks}")
    # trial_data is the UHF determinant, used only to initialise the walkers;
    # the sampled trial itself lives in the two closures above.
    trial_ops = make_auto_trial_ops(sys_, overlap_u=overlap_s, get_rdm1=uhf_get_rdm1)
    meas_ops = MeasOps(overlap=overlap_s, kernels={k_energy: energy_s})
    if FAST_SWEEP:
        prop_ops = make_msd_fast_prop_ops(ham, sys_.walker_kind, overlap_s,
                                          pack, n_chunks)
    else:
        prop_ops = cpmc_slow.make_prop_ops(ham, sys_.walker_kind)

    if TABLE_MSD:
        # The table divides by det(C[ref, :]), so a reference minor near a node
        # costs digits where the brute force would simply contribute zero. Check
        # it on walkers that have actually moved, not on the initial ones, and
        # check the overlap itself against the brute force while we are here.
        _rng = np.random.default_rng(7)
        _pert = lambda C, n: jnp.asarray(np.stack(
            [np.linalg.qr(C + 0.3 * _rng.standard_normal(C.shape))[0] for _ in range(n)]))
        _W = (_pert(np.asarray(Ca), 64), _pert(np.asarray(Cb), 64))
        _dr = np.abs(np.asarray(jax.vmap(lambda c: jnp.linalg.det(c[pack["refsA"]]))(_W[0])))
        _dm = np.abs(np.asarray(jax.vmap(lambda c: jnp.linalg.det(c[pack["RA"]]))(_W[0])))
        _ovf = np.asarray(jax.jit(jax.vmap(overlap_s))(_W))
        _ovb = np.asarray(jax.jit(jax.vmap(lambda w: jnp.sum(
            pack["CK"] * jnp.linalg.det(w[0][pack["RA"]])[pack["IA"]]
            * jnp.linalg.det(w[1][pack["RB"]])[pack["IB"]])))(_W))
        print(f"   table vs brute force, 64 perturbed walkers: max rel err "
              f"{np.abs(_ovf / _ovb - 1).max():.2e}")
        print(f"   worst reference minor / median minor        "
              f"{(_dr.min() / np.median(_dm)):.2e}   (small = losing digits)")

    _t0 = time.perf_counter()
    mean, err, be, bw = run_qmc_energy(
        sys=sys_, params=params, ham_data=ham, trial_data=trial_data,
        meas_ops=meas_ops, trial_ops=trial_ops, prop_ops=prop_ops,
        block_fn=blocks.block)
    t_run = time.perf_counter() - _t0
    _f = lambda x: None if x is None else float(x)
    print(f"\nCPMC, sampled UHF-MSD trial  {_f(mean)} +- {_f(err)}")
    print(f"wall time                    {t_run:.0f} s")
    if RESULT_JSON:
        import json as _json
        rec = dict(kind="sampling", tag=TAG, L=L, n_up=n_up, U=U, chi_trial=CHI,
                   n_samples=N_SAMPLES, n_dets=int(len(coeff)),
                   n_alpha_strings=int(len(np.unique(occ_a, axis=0))),
                   n_beta_strings=int(len(np.unique(occ_b, axis=0))),
                   weight=float((coeff**2).sum()), trial="uhf_msd",
                   n_walkers=N_WALKERS, n_blocks=N_BLOCKS, n_eql=N_EQL,
                   n_prop=N_PROP, dt=DT, seed=SEED, e_hf=float(e_hf),
                   e_dmrg_trial=float(E_dmrg_T), e_dmrg_davidson=float(E_dmrg_dav), e_trial_sampled=float(e_trial_samp),
                   fast_sweep=bool(FAST_SWEEP), n_chunks=int(n_chunks),
                   e_cpmc=_f(mean), err_cpmc=_f(err), t_run=t_run,
                   t_sample=t_sample)
        with open(RESULT_JSON, "a") as fh:
            fh.write(_json.dumps(rec) + "\n"); fh.flush()
        print("saved ->", RESULT_JSON)
