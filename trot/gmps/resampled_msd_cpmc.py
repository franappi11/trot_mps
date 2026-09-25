"""CPMC with an MSD trial that is RE-DRAWN from the DMRG state as the walk proceeds.

`sampled_msd_cpmc.py` draws determinants once and keeps the distinct ones with
their exact coefficients, so the CPMC trial is one fixed truncation of the DMRG
state -- 0.6-2% of its norm at L=24 -- and the constraint the walkers feel is
that truncation's, not the DMRG state's. Here the determinant set is replaced
every RESAMPLE_EVERY steps with fresh perfect samples, so over a run the walkers
see the whole DMRG state instead of one fixed slice of it.

Three things have to be right for a moving trial to be a legitimate importance
function.

1. COEFFICIENTS. Each redraw must be an UNBIASED estimate of Psi_T, otherwise the
   average over redraws is some other state. With N draws n_1..n_N from
   p(n) = |Psi_T(n)|^2, duplicates included, the trial is notes.pdf eqs. (9)-(10):

       |Psi~> = (1/N) sum_k |n_k> / Psi_T(n_k),
       E[<Psi~|phi>] = sum_n p(n) phi(n) / Psi_T(n) = <Psi_T|phi>   for every phi.

   sampled_msd_cpmc.py's coefficients -- the exact c_d on the distinct draws -- are
   NOT unbiased: their mean is sum_d P(d drawn) c_d |d>, which weights determinants
   by ~N c_d^3 when draws are rare.

2. WEIGHTS ACROSS A REDRAW. An importance-sampled weight is w = W <Psi~|phi>, so
   swapping the trial must multiply w by <Psi~_new|phi> / <Psi~_old|phi>. trot's
   step already does this: its first ratio is <Psi~_new|B phi> / state.overlaps,
   and the stored overlaps are the old trial's, so the ratio is the switch times
   the usual half-step ratio. A NEGATIVE switch means the two trials disagree on
   the walker's sign, and trot's constraint kills it.

3. NOISE. The switch is a ratio of two independent estimates, each with relative
   spread ~ sqrt((1/cos^2 - 1) / N) (notes eqs. 11-12; cos is the normalised
   overlap of the walker with Psi_T). If that is not small, weights diffuse and
   the sign test kills walkers at random -- a new bias in place of the truncation
   one. The run prints the spread of <Psi~|phi_RHF> over a few redraws up front;
   equilibrated walkers overlap Psi_T less than phi_RHF does, so treat it as a
   lower bound.

HOW IT SITS ON TROT. All of the QMC is trot's: prop/cpmc.py's fast-update step,
blocks.block for measurement and SR, driver.run_qmc_energy for equilibration,
sampling and blocking. This file supplies only
  * the trial, as trot TrialOps / MeasOps. For a bra determinant in the SITE
    basis the Green's function diagonal is the occupation itself, G^k_xx = n_x(k),
    and a row rescale leaves it unchanged, so trot's determinant-lemma ratio for a
    field at x is exactly (1 + u_up n_x,up(k)) (1 + u_dn n_x,dn(k)) per determinant
    and update_green only rescales the determinant weights. No 2L x 2L Green's
    function is stored; trot's MultiGhfTrial would carry nw x ndet x (2L)^2 of
    them, 8 GB at L=16, N=5000, in its default complex64.
  * the local energy, in the same Slater-Condon form sampled_msd_cpmc.py uses:
    E_L = sum_k w_k [k1_a + k1_b + U docc_k] / sum_k w_k,
    k1_sigma = tr(h1[occ, :] C (C[occ, :])^-1).
  * a block_fn around trot's: the steps run through trot's own step with a fresh
    trial every RESAMPLE_EVERY steps, then trot's block runs with n_prop_steps = 0,
    so it measures and reconfigures with the trial the last step used -- the one
    the stored overlaps refer to. With RESAMPLE_EVERY = 0 it is trot's block.

COST. The minors det(C[occ, :]) are the block: measured at L=16 and 24, N=5000,
LAPACK minors, the overlaps and fast-update weights take 84-87% of it, the local
energy 7-8%, the redraw 1%, the site loop 2-5%. trot's step needs three minor
batches per step (the overlap after each half step and the site-loop weights; the
first and last are the same minors, and sharing them would need a change to
trot's step). Two things cut them down:
  * minors are taken once per DISTINCT alpha / beta string of the current draws.
    jit needs static shapes and the number of distinct strings is random, so
    jnp.unique pads to a cap: STRING_CAP, or 0 to fit it from six pilot draws
    (largest count + 10 sqrt(count) + 32, at most N; a trial that ever exceeds it
    prints a warning). It pays only where strings repeat: at L=16, N=5000 (~1500
    distinct per spin) 10.2 s/block against 21.2 s with STRING_CAP = N,
    bit-identical energies; at L=24 the pilot already returns N and it is a no-op.
  * TABLE_MSD: reference tables, as in sampled_msd_cpmc.py, with the references
    fit once to pilot draws instead of to one fixed set (section below).
Measured, N=5000, 200 walkers, 20 steps per block, after compile; block energies
identical to 10 digits with and without the tables:
                                   LAPACK       TABLE_MSD
    L=16  (7 refs, mean k 2.3)    10.2 s/block   2.27 s/block
    L=24 (21 refs, mean k 2.8)    56   s/block   12.6 s/block
With the tables the local energy (brute-force solve, once per block) is ~30% of a
block at L=24.

VALIDATION (measured; the comparisons below used two other coefficient choices,
since removed: sampled_msd_cpmc.py's exact c_d, and Horvitz-Thompson c_d / pi_d)
  * With a fixed trial carrying sampled_msd_cpmc.py's coefficients and the same
    seeds, block energies equal that script's to 1e-15 over 8 blocks at L=8 -- its
    own fast sweep against trot's fast step on these TrialOps. On 32 random
    non-orthogonal walkers the overlap agrees to 1.8e-13 relative and E_L to
    1.5e-10; calc_overlap_ratio and update_green agree with a recomputed overlap
    to 1e-15.
  * Unbiasedness at L=8, against the full 4900-determinant expansion, over 4000
    redraws of N=50: <Psi~|phi> and <Psi~|H|phi> within 1.6 sigma of exact (the
    exact-c_d trial is off by hundreds of sigma).
  * L=10 (exact -5.380619; chi=64 DMRG is exact there), N_SAMPLES = 100 (captured
    weight ~0.23), 200 walkers, 50 + 200 blocks, independent seeds:
        fixed trial, exact c_d (sampled_msd_cpmc.py)   bias +0.054 +- 0.015   5 runs
        redrawn every step (this script)               bias +0.003 +- 0.008   9 runs
        redrawn every step, Horvitz-Thompson           bias +0.019 +- 0.009   8 runs
    with the same ~0.02-0.03 error bar per run for all three. Redrawing removes the
    truncation bias.
  * trot's block replaces E_L by E_est when |E_L - E_est| > sqrt(2/dt). A redrawn
    trial makes E_L noisier: at L=10, N=100 about 0.2% of local energies were
    clipped, and removing the clip moved the energy by -0.0065 +- 0.0006 over 4
    paired runs -- small next to the error bar, but systematic.

Usage: edit CONFIG, or override any CONFIG name on the command line:
    ~/.trot/bin/python resampled_msd_cpmc.py N_SAMPLES=20000 L=24
"""
import ast
import dataclasses
import json
import sys
import time
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
jax.config.update("jax_enable_x64", True)
from pyblock3.fcidump import FCIDUMP
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.algebra.mpe import MPE
from pyblock3.algebra.symmetry import SZ
from trot.core.ops import MeasOps, TrialOps, k_energy
from trot.core.system import System
from trot.driver import run_qmc_energy
from trot.ham.hubbard import HamHubbard
from trot.prop import blocks, cpmc
from trot.prop.types import QmcParams

_before_config = set(globals())
# ============================================================== CONFIG
L               = 16
N_UP            = None              # electrons per spin; None = half filling
N_DN            = None
t, U            = 1.0, 4.0

CHI             = 16                # DMRG bond dimension of Psi_T
DMRG_SWEEPS     = 14
DMRG_SEED       = 0

N_SAMPLES       = 5000              # draws per trial, duplicates included
RESAMPLE_EVERY  = 1                 # steps per trial; 0 = never redraw (fixed trial)
TRIAL_SEED      = 1                 # the first trial; redraws are keyed off the walk
STRING_CAP      = 0                 # distinct strings per spin per trial; 0 = from a pilot
TABLE_MSD       = True              # reference-table minors; False = one LAPACK det per string
N_REF           = 0                 # table references per spin; 0 = distinct strings / 200

N_WALKERS       = 200
N_BLOCKS        = 200
N_EQL           = 50
N_PROP          = 20
DT              = 0.01
SEED            = 1234
WEIGHT_FLOOR    = 1e-8
N_CHUNKS        = 0                 # walker micro-batches; 0 = size from MEM_BUDGET_GB
MEM_BUDGET_GB   = 4.0
RESULT_JSON     = ""                # if set, append a JSON record here
TAG             = ""
# ======================================================================
CONFIG_NAMES = sorted(set(globals()) - _before_config - {"_before_config"})


def apply_overrides(argv, g):
    """KEY=VALUE arguments override the CONFIG entry of the same name."""
    for arg in argv:
        key, sep, val = arg.partition("=")
        if not sep or key not in CONFIG_NAMES:
            raise SystemExit(f"expected KEY=VALUE with KEY in CONFIG, got {arg!r}")
        try:
            g[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            g[key] = val                                  # bare strings: RESULT_JSON=out.jsonl


# ============================================================ DMRG trial
def build_hamil(nsite, n_up, n_dn, u_int, t_hop=1.0):
    h1e = np.zeros((nsite, nsite))
    for i in range(nsite - 1):
        h1e[i, i + 1] = h1e[i + 1, i] = -t_hop
    g2e = np.zeros((nsite,) * 4)
    for i in range(nsite):
        g2e[i, i, i, i] = u_int
    fd = FCIDUMP(pg="c1", n_sites=nsite, n_elec=n_up + n_dn, twos=n_up - n_dn,
                 ipg=0, h1e=h1e, g2e=g2e)
    return Hamiltonian(fd, flat=True)


def run_dmrg(hamil, bdim, n_sweeps=14, seed=0):
    """The MPS and its TRUE <Psi_T|H|Psi_T> (not the pre-truncation Davidson value)."""
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
    ten = mps[i]
    for k in range(ten.n_blocks):
        q = tuple(SZ.from_flat(int(x)) for x in ten.q_labels[k])
        sh = tuple(int(x) for x in ten.shapes[k])
        yield q, sh, np.asarray(ten.data[ten.idxs[k]:ten.idxs[k + 1]]).reshape(sh)


def densify(mps, nsite):
    """flat pyblock3 MPS -> dense (Dl, 4, Dr) arrays, local index n_a + 2 n_b."""
    qkey = lambda q: (int(q.n), int(q.twos))
    left, right = [], []
    for i in range(nsite):
        lo, ro = {}, {}
        for (ql, qp, qr), sh, _ in flat_blocks(mps, i):
            lo[qkey(ql)], ro[qkey(qr)] = sh[0], sh[2]
        left.append(lo); right.append(ro)
    for i in range(nsite - 1):
        assert left[i + 1] == right[i], f"bond {i+1} mismatch between sites"
    offs = []
    for b in [left[i] for i in range(nsite)] + [right[nsite - 1]]:
        o, acc = {}, 0
        for k in sorted(b):
            o[k] = (acc, b[k]); acc += b[k]
        offs.append((o, acc))
    out = []
    for i in range(nsite):
        A = np.zeros((offs[i][1], 4, offs[i + 1][1]))
        for (ql, qp, qr), sh, dat in flat_blocks(mps, i):
            assert sh[1] == 1, f"physical block dim {sh[1]} != 1"
            na, nb = spin_occ(qp)
            ol, dl = offs[i][0][qkey(ql)]
            orr, dr = offs[i + 1][0][qkey(qr)]
            A[ol:ol + dl, na + 2 * nb, orr:orr + dr] = dat[:, 0, :]
        out.append(A)
    return out


def right_canonicalize(ts):
    """Right-canonical form with the centre on site 0, normalised. Returns (ts, norm).

    sum_l B_x[:, l, :] B_x[:, l, :]^T = I for every x >= 1, which is what makes
    the sampling conditionals normalised by construction.
    """
    ts = [jnp.asarray(a) for a in ts]
    for x in range(len(ts) - 1, 0, -1):
        Dl, d, Dr = ts[x].shape
        q, r = jnp.linalg.qr(ts[x].reshape(Dl, d * Dr).conj().T, mode="reduced")
        ts[x] = q.conj().T.reshape(-1, d, Dr)
        ts[x - 1] = jnp.tensordot(ts[x - 1], r.conj().T, axes=([2], [0]))
    norm = jnp.linalg.norm(ts[0])
    ts[0] = ts[0] / jnp.where(norm == 0.0, 1.0, norm)
    return ts, norm


class Sample(NamedTuple):
    config: jax.Array
    amp: jax.Array


def perfect_sample(ts, key, eps=1e-300):
    """One perfect sample from a normalised right-canonical MPS: (config, <config|psi>).

    Same draw as sampled_msd_cpmc.py (L uniforms, unrolled inverse CDF), so the
    two scripts see identical determinants from identical keys.
    """
    n = len(ts)
    us = jax.random.uniform(key, (n,))
    v = jnp.ones((1,), ts[0].dtype)
    cfg, logamp = [], jnp.zeros(())
    for x in range(n):
        w = jnp.tensordot(v, ts[x], axes=([0], [0]))       # (d, Dr)
        p = jnp.einsum("sr,sr->s", w, w.conj()).real
        c = jnp.cumsum(p)
        u = us[x] * c[-1]
        s = sum((u >= c[k]).astype(jnp.int32) for k in range(p.shape[0] - 1))
        nrm = jnp.sqrt(jnp.maximum(p[s], eps))
        v = w[s] / nrm
        cfg.append(s)
        logamp = logamp + jnp.log(nrm)
    return Sample(jnp.stack(cfg).astype(jnp.int32), jnp.exp(logamp) * v[0])


def perfect_sample_batch(ts, key, n_samples):
    return jax.vmap(perfect_sample, in_axes=(None, 0))(ts, jax.random.split(key, n_samples))


# ============================================================ the trial
class MsdTrial(NamedTuple):
    """One draw of the trial. A pytree of fixed shapes, so redraws never recompile."""
    occ_a: jax.Array  # (SA, n_up) occupied sites of each distinct alpha string (padded)
    occ_b: jax.Array  # (SB, n_dn)
    ia: jax.Array     # (N,) draw -> row of occ_a
    ib: jax.Array     # (N,)
    coef: jax.Array   # (N,) 1 / (N Psi_T(n_k)) for each draw k
    docc: jax.Array   # (N,) double occupancy of each draw
    code: jax.Array   # (N, L) uint8 local code n_a + 2 n_b of each draw
    rdm1: jax.Array   # (2, L, L) free-electron density; only seeds trot's walkers
    n_str: jax.Array  # (2,) true number of distinct alpha / beta strings
    tab_a: tuple | None = None  # reference-table bookkeeping for occ_a (TABLE_MSD), or None
    tab_b: tuple | None = None


def _distinct_strings(bits, cap, nocc):
    """Distinct occupation strings among the draws, padded to a static `cap`.
    bits (N, L) in {0, 1}; the key is the string's bitmask (exact for L <= 62).
    Padding repeats the first string, so padded minors are finite and unused.
    """
    nsite = bits.shape[1]
    key = jnp.sum(bits.astype(jnp.int64) << jnp.arange(nsite, dtype=jnp.int64), axis=1)
    uniq, inv = jnp.unique(key, size=cap, fill_value=key[0], return_inverse=True)
    srt = jnp.sort(key)
    count = 1 + jnp.sum(srt[1:] != srt[:-1])
    ubits = (uniq[:, None] >> jnp.arange(nsite, dtype=jnp.int64)) & 1
    occ = jax.vmap(lambda r: jnp.nonzero(r, size=nocc)[0])(ubits)
    return occ, inv.reshape(-1), count


def _warn_caps(n_str, caps):
    if np.any(np.asarray(n_str) > np.asarray(caps)):
        print(f"\n*** {np.asarray(n_str)} distinct strings exceeded STRING_CAP {caps}: "
              f"this trial is truncated, rerun with a larger STRING_CAP ***\n", flush=True)


def build_trial(sample, caps, nocc, rdm1, tables=None):
    """Perfect samples -> MsdTrial. `caps` = static (SA, SB), `nocc` = (n_up, n_dn).
    `tables` = ((refs_a, kcaps_a), (refs_b, kcaps_b)) from pilot_tables, or None."""
    cfg = sample.config
    a, b = cfg & 1, cfg >> 1
    occ_a, ia, na = _distinct_strings(a, caps[0], nocc[0])
    occ_b, ib, nb = _distinct_strings(b, caps[1], nocc[1])
    jax.debug.callback(_warn_caps, jnp.stack([na, nb]), caps)
    tab_a = tab_b = None
    if tables is not None:
        (refs_a, kcaps_a), (refs_b, kcaps_b) = tables
        tab_a = _table_groups(occ_a, na, refs_a, kcaps_a, cfg.shape[1])
        tab_b = _table_groups(occ_b, nb, refs_b, kcaps_b, cfg.shape[1])
    # (-1)^K, K = sum_{i in r_a} #{j in r_b : j < i}: interleaved -> all-alpha-then-beta
    K = jnp.sum(a * (jnp.cumsum(b, axis=1) - b), axis=1)
    c = (1 - 2 * (K & 1)) * sample.amp                # Psi_T(n_k), all-alpha-then-beta
    coef = 1.0 / (c.shape[0] * c)                     # notes eq. (9): phi(n) / Psi_T(n)
    return MsdTrial(occ_a=occ_a, occ_b=occ_b, ia=ia, ib=ib, coef=coef,
                    docc=jnp.sum(cfg == 3, axis=1).astype(c.dtype),
                    code=cfg.astype(jnp.uint8), rdm1=rdm1, n_str=jnp.stack([na, nb]),
                    tab_a=tab_a, tab_b=tab_b)


# ------------------------------------------------ reference-table minors
"""det(C[s, :]) for every string s of a trial from a handful of factorisations.

The table form of sampled_msd_cpmc.py: pick REFERENCE strings r_j, form
T_j = C (C[r_j, :])^-1 once per walker, and for a string s at excitation level k
from its nearest reference

    det(C[s, :]) = (-1)^(sum(I) + sum(J)) det(T_j[cre, J]) det(C[r_j, :])

    cre = sorted(s - r_j)   created sites,        I = their positions in sorted s
    ann = sorted(r_j - s)   annihilated sites,    J = their positions in sorted r_j

so each string costs a k x k determinant (closed form for k <= 3) instead of an
nocc x nocc one. The fixed script fits the references to its one set of strings;
here every trial is a fresh draw from the same |Psi_T|^2, so the references are
fit ONCE to pilot draws (pick_refs' greedy sum_s min_j k(s, r_j)^3) and every trial
only assigns its strings to them and groups them by k. jit needs static group
sizes and the count at each k is random, so each group is padded to a cap fit from
the same pilots (overflow is checked and warned, like STRING_CAP).

THE HAZARD is the one sampled_msd_cpmc.py documents: the table divides by
det(C[r_j, :]), so a reference minor near a node costs digits. main() prints the
worst reference minor relative to the median minor, and the overlap's error
against LAPACK, on perturbed walkers.
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


def _bits(occ, nsite):
    return jnp.zeros((occ.shape[0], nsite), jnp.int32).at[jnp.arange(occ.shape[0])[:, None], occ].set(1)


def _table_groups(occ, n_valid, refs, kcaps, nsite):
    """A trial's strings -> (refs, per-k groups, per-k counts), all static shapes.

    Group k holds (idx, r, cre, J, sgn): the rows of `occ` at excitation level k
    from their nearest reference r, padded to kcaps[k] with the out-of-range row
    index S so the scatter in _table_minors drops them. Rows past n_valid are
    jnp.unique's padding and join no group.
    """
    S, nocc = occ.shape
    bs, br = _bits(occ, nsite), _bits(refs, nsite)
    shared = bs @ br.T                                          # (S, nref)
    rj = jnp.argmax(shared, axis=1)
    k = jnp.where(jnp.arange(S) < n_valid, nocc - jnp.max(shared, axis=1), -1)
    groups, counts = [], []
    for kk, cap in enumerate(kcaps):
        counts.append(jnp.sum(k == kk))
        idx = jnp.nonzero(k == kk, size=cap, fill_value=S)[0]
        rows = jnp.minimum(idx, S - 1)                          # padded slots: any valid row
        r = rj[rows]
        if kk == 0:
            groups.append((idx, r))
            continue
        s_b, r_b = bs[rows], br[r]
        cre = jax.vmap(lambda m: jnp.nonzero(m, size=kk)[0])(s_b * (1 - r_b))
        ann = jax.vmap(lambda m: jnp.nonzero(m, size=kk)[0])(r_b * (1 - s_b))
        I = jnp.take_along_axis(jnp.cumsum(s_b, axis=1), cre, axis=1) - 1
        J = jnp.take_along_axis(jnp.cumsum(r_b, axis=1), ann, axis=1) - 1
        sgn = (1 - 2 * ((I.sum(1) + J.sum(1)) & 1)).astype(jnp.float64)
        groups.append((idx, r, cre, J, sgn))
    counts = jnp.stack(counts)
    jax.debug.callback(_warn_kcaps, counts, jnp.asarray(kcaps))
    return refs, tuple(groups)


def _warn_kcaps(counts, kcaps):
    over = np.asarray(counts) > np.asarray(kcaps)
    if np.any(over):
        print(f"\n*** table group(s) k={np.nonzero(over)[0].tolist()} overflowed "
              f"({np.asarray(counts)[over].tolist()} > {np.asarray(kcaps)[over].tolist()}): "
              f"this trial is truncated, rerun with TABLE_MSD=False or more N_REF ***\n",
              flush=True)


def _table_minors(C, tab, n_rows):
    """det(C[s, :]) for every row s of the trial's occ, from the reference tables."""
    refs, groups = tab
    Cr = C[refs]                                                # (nref, n, n)
    dref = jnp.linalg.det(Cr)
    T = jnp.swapaxes(jnp.linalg.solve(                          # C Cr^-1, (nref, L, n)
        jnp.swapaxes(Cr, -1, -2), jnp.broadcast_to(C.T, (refs.shape[0],) + C.T.shape)), -1, -2)
    out = jnp.zeros(n_rows, C.dtype)
    for kk, g in enumerate(groups):
        if kk == 0:
            idx, r = g
            out = out.at[idx].set(dref[r], mode="drop")
        else:
            idx, r, cre, J, sgn = g
            sub = T[r[:, None, None], cre[:, :, None], J[:, None, :]]      # (cap, k, k)
            out = out.at[idx].set(sgn * _small_det(sub, kk) * dref[r], mode="drop")
    return out


# ------------------------------------------------ trot TrialOps / MeasOps
def _minors(C, occ, tab=None):
    """det(C[s, :]) for every row s of occ: reference tables if the trial has them."""
    return jnp.linalg.det(C[occ]) if tab is None else _table_minors(C, tab, occ.shape[0])


def _det_weights(walker, tr):
    """w_k = coef_k det(Ca[occ_a(k), :]) det(Cb[occ_b(k), :]) for every draw k."""
    ca, cb = walker
    return (tr.coef * _minors(ca, tr.occ_a, tr.tab_a)[tr.ia]
            * _minors(cb, tr.occ_b, tr.tab_b)[tr.ib])


def overlap(walker, tr):
    return jnp.sum(_det_weights(walker, tr))


def calc_green(walker, tr):
    """trot's 'greens' for a site-basis MSD: the determinant weights, plus the
    occupations G^k_xx = n_x(k) -- all a diagonal field update ever reads."""
    return {"w": _det_weights(walker, tr), "code": tr.code}


def _field_factor(greens, update_indices, update_constants):
    """(1 + u_up n_x,up(k)) (1 + u_dn n_x,dn(k)) for every determinant k.
    trot's update_indices are [[0, x], [1, x]]: both spins at the same site x."""
    c = greens["code"][:, update_indices[0, 1]]
    return ((1.0 + update_constants[0] * (c & 1)) * (1.0 + update_constants[1] * (c >> 1)))


def calc_overlap_ratio(greens, update_indices, update_constants):
    w = greens["w"]
    return jnp.sum(w * _field_factor(greens, update_indices, update_constants)) / jnp.sum(w)


def update_green(greens, update_indices, update_constants):
    return {"w": greens["w"] * _field_factor(greens, update_indices, update_constants),
            "code": greens["code"]}


def energy_kernel(walker, ham_data, meas_ctx, tr):
    """<Psi~|H|phi> / <Psi~|phi>, Slater-Condon for a product-state bra:
    <d|H|phi>/<d|phi> = k1_a + k1_b + U docc, k1 = tr(h1[occ, :] C (C[occ, :])^-1)."""

    ca, cb = walker
    h1 = ham_data.h1

    def dets_k1(C, occ):
        M = C[occ]                                                    # (S, n, n)
        X = jnp.swapaxes(jnp.linalg.solve(                            # C M^-1, (S, L, n)
            jnp.swapaxes(M, -1, -2), jnp.broadcast_to(C.T, (occ.shape[0],) + C.T.shape)), -1, -2)
        return jnp.linalg.det(M), jnp.einsum("kln,knl->k", h1[occ], X)

    da, ka = dets_k1(ca, tr.occ_a)
    db, kb = dets_k1(cb, tr.occ_b)
    w = tr.coef * da[tr.ia] * db[tr.ib]
    return jnp.sum(w * (ka[tr.ia] + kb[tr.ib] + ham_data.u * tr.docc)) / jnp.sum(w)


TRIAL_OPS = TrialOps(overlap=overlap, get_rdm1=lambda tr: tr.rdm1, calc_green=calc_green,
                     calc_overlap_ratio=calc_overlap_ratio, update_green=update_green)
MEAS_OPS = MeasOps(overlap=overlap, kernels={k_energy: energy_kernel})


def make_redraw_block(draw_trial, every):
    """trot's blocks.block, with a fresh trial every `every` steps.

    The steps run here through trot's own step (prop_ops.step) with the current
    trial; trot's block is then called with n_prop_steps = 0, so it only measures
    and reconfigures, with the trial the last step used. The trial keys are folded
    off the walk's key, which leaves trot's RNG stream untouched.
    """
    def block_fn(state, *, params, trial_data, prop_ops, **kw):
        step_kw = {k: kw[k] for k in ("ham_data", "trial_ops", "meas_ops", "meas_ctx", "prop_ctx")}

        def segment(carry, _):
            st, _, key = carry
            key, sub = jax.random.split(key)
            tr = draw_trial(sub)
            st, _ = lax.scan(lambda s, _: (prop_ops.step(s, params=params, trial_data=tr,
                                                         **step_kw), None),
                             st, None, length=every)
            return (st, tr, key), None

        key = jax.random.fold_in(state.rng_key, 1_000_003)
        (state, tr, _), _ = lax.scan(segment, (state, trial_data, key), None,
                                     length=params.n_prop_steps // every)
        return blocks.block(state, params=dataclasses.replace(params, n_prop_steps=0),
                            trial_data=tr, prop_ops=prop_ops, **kw)

    return block_fn

"""
What the string caps are. For each walker, the script computes the minor det(C[occ,:]) 
once per distinct α and β string in the current draw, not once per draw. 
Under jit every array needs a fixed shape, but the number of distinct strings in a 
draw is random. So jnp.unique pads its output to a fixed size, and that size is the cap. 
STRING_CAP=0 fits the cap automatically.What pilot_caps does. It makes six throwaway draws of 
N samples and counts the distinct strings in each. It then sets the cap to the 
largest count plus 10√count + 32, never above N. 
A trial that somehow exceeded the cap would silently lose strings, 
so the script prints a warning if that ever happens.
"""
def pilot_caps(can, n_samples, seed):
    """Distinct-string caps from a few pilot draws, with ~10 sigma of slack."""
    key = jax.random.PRNGKey(seed + 7919)
    draw = jax.jit(perfect_sample_batch, static_argnums=2)
    counts = []
    for _ in range(6):
        key, sub = jax.random.split(key)
        cfg = np.asarray(draw(can, sub, n_samples).config)
        counts.append([len(np.unique(cfg & 1, axis=0)), len(np.unique(cfg >> 1, axis=0))])
    counts = np.array(counts)
    caps = tuple(int(min(n_samples, m + 10 * np.sqrt(m) + 32)) for m in counts.max(0))
    return caps, counts.mean(0)


def pilot_tables(can, n_samples, seed, caps, nocc, n_ref, nsite):
    """References and per-k group caps for the table minors, from pilot draws.

    References: pick_refs' greedy fit to the distinct strings of two pilot draws,
    weighted by how often each was drawn; N_REF of them, or distinct strings / 200
    (sampled_msd_cpmc.py's measured optimum). Group caps: the most strings seen at
    each excitation level over six pilot draws plus ~10 sigma of slack (16 for a
    level never seen), at most the string cap.
    Returns (((refs_a, kcaps_a), (refs_b, kcaps_b)), [(nref, mean k, kcaps)] * 2).
    """
    key = jax.random.PRNGKey(seed + 104723)
    draw = jax.jit(perfect_sample_batch, static_argnums=2)
    cfgs = []
    for _ in range(6):
        key, sub = jax.random.split(key)
        cfgs.append(np.asarray(draw(can, sub, n_samples).config))
    as_occ = lambda bits, n: np.nonzero(bits)[1].reshape(len(bits), n)   # rows hold n ones
    tables, info = [], []
    for spin, cap, n in ((0, caps[0], nocc[0]), (1, caps[1], nocc[1])):
        per_draw = [np.unique((c >> spin) & 1, axis=0) for c in cfgs]
        pool, hits = np.unique(np.concatenate([(c >> spin) & 1 for c in cfgs[:2]]),
                               axis=0, return_counts=True)
        A = as_occ(pool, n)
        nref = n_ref or int(np.clip(np.mean([len(u) for u in per_draw]) // 200, 1, 64))
        ridx, _, _ = pick_refs(A, nsite, nref, hits.astype(float),
                               "greedy" if nref > 1 else "weight")
        br = pool[ridx].astype(int)
        hist = np.array([np.bincount(n - (u.astype(int) @ br.T).max(1), minlength=n + 1)
                         for u in per_draw])
        kcaps = tuple(int(min(cap, m + 10 * np.sqrt(m) + 16)) for m in hist.max(0))
        tables.append((jnp.asarray(A[ridx]), kcaps))
        info.append((nref, float(np.arange(n + 1) @ hist.mean(0) / hist.mean(0).sum()), kcaps))
    return tuple(tables), info


# ============================================================== the run
def main():
    apply_overrides(sys.argv[1:], globals())
    n_up = N_UP if N_UP is not None else L // 2
    n_dn = N_DN if N_DN is not None else L // 2
    nocc = (n_up, n_dn)
    assert RESAMPLE_EVERY == 0 or N_PROP % RESAMPLE_EVERY == 0, "RESAMPLE_EVERY must divide N_PROP"
    assert L <= 62, "string keys are int64 bitmasks"

    h1 = np.zeros((L, L))
    for i in range(L - 1):
        h1[i, i + 1] = h1[i + 1, i] = -t
    eps, mo = np.linalg.eigh(h1)
    Ca, Cb = mo[:, :n_up], mo[:, :n_dn]
    e_hf = eps[:n_up].sum() + eps[:n_dn].sum() + U * float(
        np.sum(np.einsum("ik,ik->i", Ca, Ca) * np.einsum("ik,ik->i", Cb, Cb)))
    print(f"L={L} ({n_up},{n_dn})  t={t} U={U}   E_HF = {e_hf:.12f}")

    mps, e_T, e_dav = run_dmrg(build_hamil(L, n_up, n_dn, U, t), CHI, DMRG_SWEEPS, DMRG_SEED)
    can, _ = right_canonicalize(densify(mps, L))
    print(f"DMRG chi={CHI}: <Psi_T|H|Psi_T> = {e_T:.9f}   (Davidson {e_dav:.9f})")

    if STRING_CAP > 0:
        caps = (STRING_CAP, STRING_CAP)
    else:
        caps, mean_str = pilot_caps(can, N_SAMPLES, TRIAL_SEED)
        print(f"distinct strings per trial (pilot mean) {mean_str[0]:.0f} / {mean_str[1]:.0f}"
              f"  -> caps {caps[0]} / {caps[1]} of N = {N_SAMPLES}")
    tables = None
    if TABLE_MSD:
        tables, tinfo = pilot_tables(can, N_SAMPLES, TRIAL_SEED, caps, nocc, N_REF, L)
        for spin, (nref, mean_k, kcaps) in zip("ab", tinfo):
            print(f"table minors ({spin}): {nref} references, mean excitation {mean_k:.2f} "
                  f"of nocc={nocc[0]}, group caps {kcaps}")
    rdm1 = jnp.asarray(np.stack([Ca @ Ca.T, Cb @ Cb.T]))
    draw_trial = lambda key: build_trial(perfect_sample_batch(can, key, N_SAMPLES),
                                         caps, nocc, rdm1, tables)
    # first trial: the same key split as sampled_msd_cpmc.sample_determinants
    trial0 = jax.jit(draw_trial)(jax.random.split(jax.random.PRNGKey(TRIAL_SEED))[1])

    pairs = np.unique(np.stack([np.asarray(trial0.ia), np.asarray(trial0.ib)], 1),
                      axis=0, return_index=True)[1]
    amp0 = np.asarray(perfect_sample_batch(
        can, jax.random.split(jax.random.PRNGKey(TRIAL_SEED))[1], N_SAMPLES).amp)
    w0 = float(np.sum(amp0[pairs] ** 2))
    print(f"first trial: {len(pairs)} distinct dets of {N_SAMPLES} draws, weight {w0:.5f}")
    phi = (jnp.asarray(Ca), jnp.asarray(Cb))
    f_ov = jax.jit(lambda k: overlap(phi, draw_trial(k)))
    ovs = np.array([float(f_ov(k)) for k in jax.random.split(jax.random.PRNGKey(TRIAL_SEED + 1), 16)])
    print(f"<Psi~|phi_RHF> over 16 redraws: mean {ovs.mean():+.4e}  rel. spread "
          f"{ovs.std() / abs(ovs.mean()):.3f}   (the per-step switch noise is at least this)")
    if tables is not None:
        # the table divides by det(C[ref, :]): check it on walkers that have moved
        rng = np.random.default_rng(7)
        pert = lambda C: jnp.asarray(np.stack(
            [np.linalg.qr(C + 0.3 * rng.standard_normal(C.shape))[0] for _ in range(64)]))
        W = (pert(Ca), pert(Cb))
        ov_b = jax.jit(jax.vmap(overlap, in_axes=(0, None)))
        rel = np.abs(np.asarray(ov_b(W, trial0))
                     / np.asarray(ov_b(W, trial0._replace(tab_a=None, tab_b=None))) - 1)
        d_ref = np.abs(np.asarray(jax.vmap(lambda c: jnp.linalg.det(c[tables[0][0]]))(W[0])))
        d_all = np.abs(np.asarray(jax.vmap(lambda c: jnp.linalg.det(c[trial0.occ_a]))(W[0])))
        print(f"table vs LAPACK, 64 perturbed walkers: overlap max rel err {rel.max():.2e};  "
              f"worst reference minor / median minor {d_ref.min() / np.median(d_all):.2e}")


    n_chunks = N_CHUNKS
    if n_chunks <= 0:                                 # the energy's broadcast C^T dominates
        n_chunks = max(1, int(np.ceil(3 * N_WALKERS * max(caps) * max(nocc) * L * 8
                                      / (MEM_BUDGET_GB * 1e9))))
    params = QmcParams(dt=DT, n_walkers=N_WALKERS, n_prop_steps=N_PROP, n_blocks=N_BLOCKS,
                       n_eql_blocks=N_EQL, weight_floor=WEIGHT_FLOOR, seed=SEED,
                       n_chunks=n_chunks)
    ham = HamHubbard(h1=jnp.asarray(h1), u=U)
    prop_ops = cpmc.make_prop_ops(ham, "unrestricted", TRIAL_OPS)
    # trot's initializer leaves node_encounters weakly typed, which makes the jitted
    # block scan compile twice; pinning the dtype costs one compile
    prop_ops = dataclasses.replace(prop_ops, init_prop_state=lambda **kw: cpmc.init_prop_state(
        **kw)._replace(node_encounters=jnp.zeros((), int)))
    block_fn = blocks.block if RESAMPLE_EVERY == 0 else make_redraw_block(draw_trial, RESAMPLE_EVERY)
    print(f"trial: N_SAMPLES={N_SAMPLES} RESAMPLE_EVERY={RESAMPLE_EVERY}  "
          f"n_chunks={n_chunks}")

    t0 = time.perf_counter()
    mean, err, block_e, block_w = run_qmc_energy(
        sys=System(norb=L, nelec=nocc, walker_kind="unrestricted"), params=params,
        ham_data=ham, trial_data=trial0, meas_ops=MEAS_OPS, trial_ops=TRIAL_OPS,
        prop_ops=prop_ops, block_fn=block_fn)
    t_run = time.perf_counter() - t0

    _f = lambda x: None if x is None else float(x)
    print(f"\nCPMC, re-drawn MSD trial (N={N_SAMPLES})  {_f(mean)} +- {_f(err)}")
    print(f"DMRG trial <Psi_T|H|Psi_T>                  {e_T:.9f}")
    print(f"wall time {t_run:.0f} s")
    if RESULT_JSON:
        rec = dict(kind="resampled", tag=TAG, e_cpmc=_f(mean), err_cpmc=_f(err), t_run=t_run,
                   e_dmrg_trial=e_T, e_dmrg_davidson=e_dav, e_hf=float(e_hf),
                   first_trial_weight=w0, ovl_rel_spread=float(ovs.std() / abs(ovs.mean())),
                   caps=list(caps), n_chunks=n_chunks,
                   config={k: globals()[k] for k in CONFIG_NAMES if k not in ("RESULT_JSON", "TAG")},
                   # trot's arrays: [initial estimate, N_EQL equilibration blocks, sampling
                   # blocks left after its outlier rejection]; sampling = [1 + N_EQL:]
                   block_energies=np.asarray(block_e).real.tolist(),
                   block_weights=np.asarray(block_w).real.tolist(), n_eql=N_EQL)
        with open(RESULT_JSON, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print("saved ->", RESULT_JSON)


if __name__ == "__main__":
    main()
