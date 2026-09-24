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
from trot.core.ops import k_energy, MeasOps
from trot.trial.auto import make_auto_trial_ops
from trot.prop.types import QmcParams
from trot.driver import run_qmc_energy

# ============================================================== CONFIG
L, n_up, n_down = 16, 8, 8
t, U            = 1.0, 4.0

CHI             = 64
DMRG_SWEEPS     = 14
DMRG_SEED       = 0

N_SAMPLES       = 4000
SAMPLE_SEED     = 1
SAMPLE_CHUNK    = 100_000           # draws per batch, to bound memory

N_WALKERS   = 200
N_BLOCKS    = 200
N_EQL       = 50
N_PROP      = 20
DT          = 0.01
SEED        = 1234
RESULT_JSON = '/Users/fnappi/trot_mps/trot/gmps/study_results.jsonl'
TAG         = 'smp_L16_N4000'
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
        code = cfg @ (4 ** np.arange(n))
        uniq, first, cnt = np.unique(code, return_index=True, return_counts=True)
        for c, i, k in zip(uniq, first, cnt):
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
    IA, IB = jnp.asarray(ia), jnp.asarray(ib)
    CK = jnp.asarray(coeff)
    h1j = jnp.asarray(h1)
    hA, hB = h1j[RA], h1j[RB]
    docc = jnp.asarray(np.array([len(set(a.tolist()) & set(b.tolist()))
                                 for a, b in zip(occ_a, occ_b)], float))
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
        return jnp.sum(CK * jnp.linalg.det(ca[RA])[IA] * jnp.linalg.det(cb[RB])[IB])

    def energy(walker, ham_data=None, meas_ctx=None, trial_data=None):
        ca, cb = walker
        da, k1a = _dets_k1(ca, RA, hA)
        db, k1b = _dets_k1(cb, RB, hB)
        w = CK * da[IA] * db[IB]
        return jnp.sum(w * (k1a[IA] + k1b[IB] + u_int * docc)) / jnp.sum(w)

    return overlap, energy


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

    overlap_s, energy_s = make_uhf_msd_kernels(occ_a, occ_b, coeff, h1, U)
    # trial_data is the UHF determinant, used only to initialise the walkers;
    # the sampled trial itself lives in the two closures above.
    trial_ops = make_auto_trial_ops(sys_, overlap_u=overlap_s, get_rdm1=uhf_get_rdm1)
    meas_ops = MeasOps(overlap=overlap_s, kernels={k_energy: energy_s})
    prop_ops = cpmc_slow.make_prop_ops(ham, sys_.walker_kind)

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
                   e_cpmc=_f(mean), err_cpmc=_f(err), t_run=t_run,
                   t_sample=t_sample)
        with open(RESULT_JSON, "a") as fh:
            fh.write(_json.dumps(rec) + "\n"); fh.flush()
        print("saved ->", RESULT_JSON)
