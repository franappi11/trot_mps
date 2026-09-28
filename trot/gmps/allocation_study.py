#!/usr/bin/env python
"""Frozen vs per-walker (padded) kept-count allocation for the gMPS walker truncation.

Production (mps_cpmc_new.py) truncates every walker's Fishman-White gMPS with the number of kept states per
charge sector frozen once, on a reference determinant (plan_bonds). A per-walker allocation is much closer to
the optimum (entanglement_vs_gmps.ipynb), but jitted code needs static shapes: every sector must be padded to
the largest count any walker uses, and the padded bond D is what the contraction costs. Does paying for that
padding beat spending the same bond on a frozen allocation?

For each (L, U, trial) two independent CPMC populations are run (train and test seeds). For every nominal bond
chi, the padding is learned on the training walkers, and every test walker (spin up) gets the infidelity of:

  scheme      kept counts per charge sector                                          cost (max bond)
  optimum     none: the smallest error any MPS with that bond can have (worst cut,
              from the correlation spectrum; exact)                                  chi, and D
  frozen      production: frozen on the plan reference (plan_bonds)                  chi
  own         the walker's own best counts (the ideal; not jittable)                 chi
  padded      own counts capped by the training padding: the jittable dynamic
              scheme (a walker needing more than the padding is clipped)             D
  union       the padding used as a frozen allocation (every padded slot filled)     D
  frozen_D    production at the padded bond                                          D
  frozen_W    production at the largest bond whose work (sum over gates of bond^3)
              does not exceed the padded scheme's                                    equal work

D is the largest padded bond over the circuit's gates. The overflow rate is the fraction of test walkers whose
own allocation the training padding cannot hold at some gate.

Walkers come from production-like CPMC with the plan reference production uses (plan_reference="natural"):
  dmrg, L <= DENSE_TRIAL_MAX_L   trot CPMC with the DMRG trial exact in the determinant basis (no truncation)
  dmrg, larger L                 production's MPS-CPMC (make_walker_ops + make_fast_prop_ops), walker bond RUN_CHI_W
  uhf                            trot CPMC with a Neel UHF trial (plan reference = the UHF determinant)
Infidelities are exact against the dense walker state for L <= DENSE_MAX_L, otherwise against the walker's
untruncated own-plan gMPS, whose own error (the "floor", <~1e-9) is recorded.

Usage (from trot/gmps; everything is cached in --out and re-used):
  ~/.trot/bin/python allocation_study.py --L 8 12 --U 4 8
  ~/.trot/bin/python allocation_study.py --selftest
"""
from __future__ import annotations

import argparse
import heapq
import json
import sys
import time
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import NamedTuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import mps_cpmc_new as mcn  # noqa: E402
from mps_cpmc_new import (_assemble, _move_centre, channel_angles, channel_mps, gate_pair,  # noqa: E402
                          hopping_matrix, make_orbital_plan, plan_bonds, sector_plan)
from trot.core.ops import MeasOps, TrialOps, k_energy  # noqa: E402
from trot.core.system import System  # noqa: E402
from trot.driver import make_run_blocks  # noqa: E402
from trot.ham.chol import HamChol  # noqa: E402
from trot.ham.hubbard import HamHubbard  # noqa: E402
from trot.meas.uhf import make_uhf_meas_ops  # noqa: E402
from trot.prop import blocks, cpmc_slow  # noqa: E402
from trot.prop.types import QmcParams  # noqa: E402
from trot.trial.auto import make_auto_trial_ops  # noqa: E402
from trot.trial.uhf import UhfTrial, get_rdm1 as uhf_get_rdm1, make_uhf_trial_ops  # noqa: E402

EPS = 1.0e-10              # purity tolerance of the adaptive orbital plans (mps_cpmc_new default)
DENSE_TRIAL_MAX_L = 12     # DMRG trial exact in the determinant basis up to here; production MPS-CPMC above
DENSE_MAX_L = 16           # exact dense walker states up to here; reference gMPS above
WEIGHT_FLOOR = 1.0e-8      # mps_cpmc_new's Config.weight_floor


# ---------------------------------------------------------------------------------------------- dense states
@lru_cache(maxsize=None)
def _basis(L, N):
    combos = np.array(list(combinations(range(L), N)))
    return combos, (1 << (L - 1 - combos)).sum(axis=1)


def orthonormal(W):
    return np.linalg.qr(W)[0]


def dense_state(Q):
    """<n|phi> = det Q[occupied rows of n] for every occupation string n (site 0 most significant)."""
    L, N = Q.shape
    combos, index = _basis(L, N)
    psi = np.zeros(2 ** L)
    psi[index] = np.linalg.det(Q[combos])
    return psi


def mps_to_dense(tensors):
    v = np.asarray(tensors[0])[0]
    for t in tensors[1:]:
        v = np.tensordot(v, np.asarray(t), axes=([-1], [0])).reshape(-1, np.asarray(t).shape[-1])
    return v.reshape(-1)


def mps_dot(A, B):
    E = np.ones((1, 1))
    for a, b in zip(A, B):
        E = np.einsum("ab,apc,bpd->cd", E, np.asarray(a), np.asarray(b))
    return E[0, 0]


# ------------------------------------------------------------------------ optimum from the correlation spectrum
def cut_spectra(Q):
    L = Q.shape[0]
    side = lambda b: Q[:b] if b <= L - b else Q[b:]
    return [np.clip(np.linalg.eigvalsh(side(b) @ side(b).T), 0.0, 1.0) for b in range(1, L)]


def top_schmidt(nu, chi):
    """The chi largest Schmidt values prod_k (nu_k or 1 - nu_k) of a Gaussian state (best-first search)."""
    p = np.maximum(nu, 1 - nu)
    w = np.sort(np.minimum(nu, 1 - nu) / p)[::-1]
    w = w[w > 0]
    P0 = float(np.prod(p))
    out, heap = [P0], []
    if len(w):
        heapq.heappush(heap, (-P0 * w[0], 0, P0 * w[0]))
    while heap and len(out) < chi:
        _, j, val = heapq.heappop(heap)
        out.append(val)
        if j + 1 < len(w):
            heapq.heappush(heap, (-val * w[j + 1], j + 1, val * w[j + 1]))
            swap = val / w[j] * w[j + 1]
            heapq.heappush(heap, (-swap, j + 1, swap))
    return np.array(out)


def optimum_discarded(Q, chis):
    """Worst-cut optimal discarded weight max_b (1 - sum of the chi largest Schmidt values) for each chi."""
    kmax = max(chis)
    worst = np.zeros(len(chis))
    for nu in cut_spectra(Q):
        cum = np.cumsum(top_schmidt(nu, kmax))
        worst = np.maximum(worst, [max(1.0 - cum[min(c, len(cum)) - 1], 0.0) for c in chis])
    return worst


# ------------------------------------------------------------------------------------------ gMPS with any allocation
def gmps(Q, plan, chi=None, counts=None, caps=None):
    """Fishman-White gMPS of one orthonormal spin channel, in NumPy, with a chosen truncation.

    The same gates, orthogonality-centre moves and charge-blocked splits as mps_cpmc_new.channel_mps and its
    dry run plan_bonds. The number of states kept in each charge sector at gate g is
      counts[g] = {charge: k}   given: a frozen allocation (capped at the sector's rank), or
      chosen on Q itself        the chi largest squared singular values of the split (plan_bonds' rule),
                                optionally at most caps[g] = {charge: k_max} per sector, or
      everything                chi = counts = None (untruncated).
    Returns tensors, gauge (phi = gauge * MPS), chosen[g] = {charge: k}, the discarded weight, the kept bond per
    gate, and whether a cap was ever binding.
    """
    occ = plan.occupation
    tensors = [np.eye(2)[int(o)].reshape(1, 2, 1) for o in occ]
    charges = [np.zeros(1, int)]
    for o in occ:
        charges.append(charges[-1] + int(o))
    angles, rows = channel_angles(Q, plan, xp=np)
    gauge = float(np.linalg.det(np.stack([rows[i] for i in np.flatnonzero(occ)])))
    centre, chosen, discarded, bonds, capped = None, [], 0.0, [], False
    for g, (site, theta) in enumerate(reversed(angles)):
        centre = _move_centre(tensors, charges, centre, site, xp=np)
        pair = gate_pair(tensors[site], tensors[site + 1], theta, xp=np)
        Dl, _, _, Dr = pair.shape
        M = pair.reshape(2 * Dl, 2 * Dr)
        ql, qr = charges[site], charges[site + 2]
        row_charge = (np.asarray(ql)[:, None] + np.arange(2)).ravel()
        full = sector_plan(ql, qr)
        qvals = [int(row_charge[r[0]]) for r, _, _ in full.sectors]
        decs = [np.linalg.svd(M[np.ix_(r, c)], full_matrices=False) for r, c, _ in full.sectors]
        ranks = [len(d[1]) for d in decs]
        if counts is not None:
            kept = [min(int(counts[g].get(q, 0)), n) for q, n in zip(qvals, ranks)]
        elif chi is None:
            kept = list(ranks)
        else:
            values = np.concatenate([d[1] ** 2 for d in decs])
            owner = np.concatenate([np.full(n, i) for i, n in enumerate(ranks)])
            cap = None if caps is None else [int(caps[g].get(q, 0)) for q in qvals]
            kept, total = [0] * len(decs), 0
            for idx in np.argsort(-values, kind="stable"):
                if total >= chi:
                    break
                i = owner[idx]
                if cap is not None and kept[i] >= cap[i]:
                    capped = True
                    continue
                kept[i] += 1
                total += 1
        if sum(kept) == 0:                       # a cap left nothing: keep the largest value regardless
            kept[int(np.argmax([d[1][0] for d in decs]))] = 1
            capped = True
        chosen.append({q: k for q, k in zip(qvals, kept) if k})
        truncated = sector_plan(ql, qr, tuple(kept))
        left, right = [], []
        for (u, s, vh), k in zip(decs, kept):
            discarded += float(np.sum(s[k:] ** 2))
            if k:
                left.append(u[:, :k])
                right.append(s[:k, None] * vh[:k])
        A, B = _assemble(left, right, truncated, xp=np)
        tensors[site], tensors[site + 1] = A.reshape(Dl, 2, -1), B.reshape(-1, 2, Dr)
        charges[site + 1] = truncated.middle_charges
        centre = site + 1
        bonds.append(int(sum(kept)))
    return tensors, gauge, chosen, discarded, bonds, capped


def padding(chosen_list):
    """Per gate and charge sector, the largest count any walker chose: the static shapes jit needs."""
    pad = [dict() for _ in chosen_list[0]]
    for chosen in chosen_list:
        for g, gate in enumerate(chosen):
            for q, k in gate.items():
                pad[g][q] = max(pad[g].get(q, 0), k)
    return pad


# --------------------------------------------------------------------------------------------------- trials
def uhf_scf(h1, u, na, nb, guess=0.3, mix=0.5, tol=1e-12, iters=5000):
    n = h1.shape[0]
    stag = (-1.0) ** np.arange(n)
    da, db = na / n + guess * stag, nb / n - guess * stag
    for _ in range(iters):
        _, va = np.linalg.eigh(h1 + u * np.diag(db))
        _, vb = np.linalg.eigh(h1 + u * np.diag(da))
        Ca, Cb = va[:, :na], vb[:, :nb]
        new_a, new_b = np.einsum("ik,ik->i", Ca, Ca), np.einsum("ik,ik->i", Cb, Cb)
        change = max(abs(new_a - da).max(), abs(new_b - db).max())
        da, db = (1 - mix) * da + mix * new_a, (1 - mix) * db + mix * new_b
        if change < tol:
            break
    return Ca, Cb


def dense4(tensors):
    """Amplitudes of a d=4 MPS over all 4^L local states, contracted from both ends."""
    L, h = len(tensors), len(tensors) // 2
    left = np.asarray(tensors[0])[0]
    for t in tensors[1:h]:
        left = np.tensordot(left, np.asarray(t), axes=([-1], [0])).reshape(-1, np.asarray(t).shape[-1])
    right = np.asarray(tensors[-1])[:, :, 0]
    for t in reversed(tensors[h:-1]):
        right = np.tensordot(np.asarray(t), right, axes=([-1], [0])).reshape(np.asarray(t).shape[0], -1)
    return (left @ right).reshape(-1)


def sector_matrix(vec4, L, N):
    """<I_up J_dn|state>, all up modes before all down modes (the d=4 MPS orders them site by site)."""
    combos, _ = _basis(L, N)
    occ = np.zeros((len(combos), L), int)
    occ[np.arange(len(combos))[:, None], combos] = 1
    local = occ[:, None, :] + 2 * occ[None, :, :]
    dn_left = np.concatenate([np.zeros((len(combos), 1), int), np.cumsum(occ, axis=1)[:, :-1]], axis=1)
    sign = (-1.0) ** (occ[:, None, :] * dn_left[None, :, :]).sum(-1)
    return vec4[local @ (4 ** np.arange(L - 1, -1, -1))] * sign


def dmrg_trial_mps(L, U, chi_t, sweeps):
    N = L // 2
    cfg = mcn.Config(L=L, n_up=N, n_down=N, interaction=U, trial_chi=chi_t, dmrg_sweeps=sweeps)
    mps, e_dmrg = mcn.run_dmrg(mcn.build_dmrg_hamiltonian(cfg), cfg)
    trial_np, trial_charges = mcn.densify_with_charges(mps, L)
    H_np = mcn.compress_mps(mcn.apply_mpo(mcn.hubbard_mpo(L, 1.0, U), trial_np))
    gamma = np.stack(mcn.one_rdm(trial_np))
    nos = tuple(mcn.natural_orbitals(g, N)[0] for g in gamma)
    return trial_np, trial_charges, H_np, gamma, nos, e_dmrg


class DmrgTrial(NamedTuple):
    M_T: jax.Array
    M_H: jax.Array
    rdm1: jax.Array


def make_dmrg_ops(L, N):
    combos = jnp.asarray(_basis(L, N)[0])
    amps = lambda w: jnp.linalg.det(w[combos])

    def overlap(walker, trial):
        return amps(walker[0]) @ trial.M_T @ amps(walker[1])

    def energy(walker, ham_data, meas_ctx, trial):
        a, b = amps(walker[0]), amps(walker[1])
        return (a @ trial.M_H @ b) / (a @ trial.M_T @ b)

    return (TrialOps(overlap=overlap, get_rdm1=lambda trial: trial.rdm1),
            MeasOps(overlap=overlap, kernels={k_energy: energy}))


# ------------------------------------------------------------------------------------------------ CPMC runs
def _run(prop_ops, trial_ops, meas_ops, tdata, ham_prop, ham_meas, system, params, n_blocks_total):
    run_blocks = make_run_blocks(block_fn=blocks.block, sys=system, params=params,
                                 trial_ops=trial_ops, meas_ops=meas_ops, prop_ops=prop_ops)
    ctx = dict(ham_data=ham_meas, trial_data=tdata, meas_ctx=meas_ops.build_meas_ctx(ham_meas, tdata),
               prop_ctx=prop_ops.build_prop_ctx(ham_prop, trial_ops.get_rdm1(tdata), params))
    state = prop_ops.init_prop_state(sys=system, ham_data=ham_meas, trial_ops=trial_ops,
                                     trial_data=tdata, meas_ops=meas_ops, params=params)
    ups, dns, energies = [np.asarray(state.walkers[0])], [np.asarray(state.walkers[1])], []
    for _ in range(n_blocks_total):
        state, scalars, _ = run_blocks(state, **ctx, n_blocks=1)
        ups.append(np.asarray(state.walkers[0]))
        dns.append(np.asarray(state.walkers[1]))
        energies.append(float(scalars["energy"][0]))
    return np.stack(ups), np.stack(dns), np.array(energies)


def run_population(L, U, trial, seed, args):
    """One CPMC population: every block's walkers, and the plan reference (spin up) production would use."""
    N = L // 2
    h1 = hopping_matrix(L, 1.0)
    ham = HamHubbard(h1=jnp.asarray(h1), u=U)
    system = System(norb=L, nelec=(N, N), walker_kind="unrestricted")
    params = QmcParams(dt=args.dt, n_walkers=args.walkers, n_prop_steps=args.steps, n_blocks=args.blocks,
                       n_eql_blocks=args.eql, weight_floor=WEIGHT_FLOOR, seed=seed)
    total = args.eql + args.blocks
    if trial == "uhf":
        Ca, Cb = uhf_scf(h1, U, N, N)
        onsite = np.zeros((L, L, L))
        onsite[np.arange(L), np.arange(L), np.arange(L)] = np.sqrt(U)
        ham_meas = HamChol(h0=jnp.zeros(()), h1=jnp.asarray(h1), chol=jnp.asarray(onsite))
        tdata = UhfTrial(mo_coeff_a=jnp.asarray(Ca), mo_coeff_b=jnp.asarray(Cb))
        up, dn, e = _run(cpmc_slow.make_prop_ops(ham, "unrestricted"), make_uhf_trial_ops(system),
                         make_uhf_meas_ops(system), tdata, ham, ham_meas, system, params, total)
        return dict(up=up, dn=dn, energies=e, reference=Ca, how="trot CPMC, UHF trial")
    trial_np, trial_charges, H_np, gamma, nos, _ = dmrg_trial_mps(L, U, args.trial_chi, args.dmrg_sweeps)
    if L <= DENSE_TRIAL_MAX_L:
        T, H = sector_matrix(dense4(trial_np), L, N), sector_matrix(dense4(H_np), L, N)
        norm = np.sqrt(np.sum(T ** 2))
        tdata = DmrgTrial(jnp.asarray(T / norm), jnp.asarray(H / norm), jnp.asarray(gamma))
        trial_ops, meas_ops = make_dmrg_ops(L, N)
        up, dn, e = _run(cpmc_slow.make_prop_ops(ham, "unrestricted"), trial_ops, meas_ops, tdata,
                         ham, ham, system, params, total)
        how = "trot CPMC, DMRG trial exact in the determinant basis"
    else:
        plans = [make_orbital_plan(C, "adaptive", EPS) for C in nos]
        bonds_run = [plan_bonds(C, p, args.run_chi_w, 0.0) for C, p in zip(nos, plans)]
        ops = mcn.make_walker_ops(*nos, *plans, *bonds_run, trial_np, trial_charges,
                                  tuple(jnp.asarray(A) for A in H_np))
        prop_ops = mcn.make_fast_prop_ops(ham, "unrestricted", ops.overlap, ops.sweep)
        trial_ops = make_auto_trial_ops(system, overlap_u=ops.overlap, get_rdm1=uhf_get_rdm1)
        meas_ops = MeasOps(overlap=ops.overlap, kernels={k_energy: ops.energy})
        tdata = UhfTrial(mo_coeff_a=jnp.asarray(nos[0]), mo_coeff_b=jnp.asarray(nos[1]))   # walker start
        up, dn, e = _run(prop_ops, trial_ops, meas_ops, tdata, ham, ham, system, params, total)
        how = f"production MPS-CPMC, walker bond {args.run_chi_w}"
    return dict(up=up, dn=dn, energies=e, reference=nos[0], how=how)


def population(L, U, trial, seed, args):
    path = Path(args.out) / "walkers" / f"L{L}_U{U:g}_{trial}_chiT{args.trial_chi}_nw{args.walkers}_b{args.eql}+{args.blocks}_s{seed}.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        pop = run_population(L, U, trial, seed, args)
        np.savez_compressed(path, **{k: v for k, v in pop.items() if k != "how"}, how=pop["how"])
        print(f"  CPMC L={L} U={U:g} {trial} seed {seed}: {pop['how']}, {time.perf_counter() - start:.0f} s", flush=True)
    return dict(np.load(path))


def sampling_walkers(pop, n_snaps, eql):
    """Spin-up walkers of n_snaps evenly spaced sampling-phase snapshots (snapshot b follows block b)."""
    idx = np.arange(eql + 1, pop["up"].shape[0])
    idx = idx[np.linspace(0, len(idx) - 1, min(n_snaps, len(idx))).round().astype(int)]
    return [orthonormal(W) for s in idx for W in pop["up"][s]]


# ---------------------------------------------------------------------------------------------------- study
def fidelity_loss(exact, tensors):
    kind, x = exact
    if kind == "dense":
        v = mps_to_dense(tensors)
        return float(1 - (x @ v) ** 2 / ((x @ x) * (v @ v)))
    return float(1 - mps_dot(x, tensors) ** 2 / (mps_dot(x, x) * mps_dot(tensors, tensors)))


def exact_state(Q):
    L = Q.shape[0]
    if L <= DENSE_MAX_L:
        return ("dense", dense_state(Q)), 0.0
    own = make_orbital_plan(Q, "adaptive", EPS)
    tensors, gauge, *_ = gmps(Q, own)                               # untruncated own-plan gMPS
    return ("mps", tensors), 1.0 - gauge ** 2                       # floor: the norm it misses


def study(L, U, trial, args):
    tag = f"L{L}_U{U:g}_{trial}"
    out_npz = Path(args.out) / f"{tag}.npz"
    train_pop = population(L, U, trial, args.seed, args)
    test_pop = population(L, U, trial, args.seed + args.test_seed_offset, args)
    ref = train_pop["reference"]
    plan = make_orbital_plan(ref, "adaptive", EPS)
    train = sampling_walkers(train_pop, args.train_snaps, args.eql)
    test = sampling_walkers(test_pop, args.test_snaps, args.eql)
    start = time.perf_counter()
    exacts, floors = zip(*[exact_state(Q) for Q in test])
    chis = sorted(args.chis)
    res = {k: np.full((len(test), len(chis)), np.nan) for k in
           ("optimum", "frozen", "own", "padded", "union", "frozen_D", "frozen_W", "optimum_D")}
    res["overflow"] = np.zeros((len(test), len(chis)), bool)
    meta = []
    for j, chi in enumerate(chis):
        frozen_counts = gmps(ref, plan, chi=chi)[2]
        pad = padding([gmps(Q, plan, chi=chi)[2] for Q in train])
        union_bonds = gmps(ref, plan, counts=pad)[4]
        D = max(union_bonds)
        frozen_D_counts = gmps(ref, plan, chi=D)[2]
        frozen_bonds = gmps(ref, plan, counts=frozen_counts)[4]
        frozen_D_bonds = gmps(ref, plan, counts=frozen_D_counts)[4]
        work = lambda b: int(np.sum(np.asarray(b) ** 3))
        chi_W = chi                                   # frozen at equal work: largest bond within the padded work
        for cand in range(chi, D + 1):
            if work(gmps(ref, plan, counts=gmps(ref, plan, chi=cand)[2])[4]) <= work(union_bonds):
                chi_W = cand
        frozen_W_counts = gmps(ref, plan, chi=chi_W)[2]
        frozen_W_bonds = gmps(ref, plan, counts=frozen_W_counts)[4]
        for w, (Q, exact) in enumerate(zip(test, exacts)):
            res["frozen"][w, j] = fidelity_loss(exact, gmps(Q, plan, counts=frozen_counts)[0])
            res["own"][w, j] = fidelity_loss(exact, gmps(Q, plan, chi=chi)[0])
            tensors, _, _, _, _, capped = gmps(Q, plan, chi=chi, caps=pad)
            res["padded"][w, j], res["overflow"][w, j] = fidelity_loss(exact, tensors), capped
            res["union"][w, j] = fidelity_loss(exact, gmps(Q, plan, counts=pad)[0])
            res["frozen_D"][w, j] = fidelity_loss(exact, gmps(Q, plan, counts=frozen_D_counts)[0])
            res["frozen_W"][w, j] = fidelity_loss(exact, gmps(Q, plan, counts=frozen_W_counts)[0])
            res["optimum"][w, j], res["optimum_D"][w, j] = optimum_discarded(Q, [chi, D])
        cost = lambda b: dict(max=int(max(b)), sum_cubed=int(np.sum(np.asarray(b) ** 3)))
        meta.append(dict(chi=chi, D=int(D), chi_W=int(chi_W), frozen=cost(frozen_bonds), union=cost(union_bonds),
                         frozen_D=cost(frozen_D_bonds), frozen_W=cost(frozen_W_bonds)))
        print(f"    chi={chi:3d}: padded bond D={D:3d}   ({time.perf_counter() - start:.0f} s)", flush=True)
    np.savez_compressed(out_npz, chis=np.array(chis), floor=np.array(floors), **res, meta=json.dumps(meta))
    return res, meta, np.array(floors), dict(train=len(train), test=len(test), how=str(train_pop["how"]),
                                             energy_train=float(np.mean(train_pop["energies"][args.eql:])),
                                             gates=int((plan.block_sizes - 1).sum()), plan_max_B=int(plan.block_sizes.max()))


def summarize(L, U, trial, res, meta, floors, info, args):
    stats = lambda x: dict(median=float(np.median(x)), p90=float(np.percentile(x, 90)),
                           p99=float(np.percentile(x, 99)), mean=float(np.mean(x)), max=float(np.max(x)))
    records = []
    print(f"\n  L={L} U={U:g} {trial}: {info['how']}; {info['train']} training and {info['test']} test walkers; "
          f"circuit {info['gates']} gates (max B {info['plan_max_B']}); reference floor max {floors.max():.1e}")
    print(f"  {'chi':>4s} {'D':>4s} {'optimum':>9s} {'frozen':>9s} {'own':>9s} | {'padded':>9s} {'union':>9s} "
          f"{'frozen_D':>9s} {'opt(D)':>9s} | {'chi_W':>5s} {'frozen_W':>9s} | {'overflow':>8s}  best at equal work")
    for j, m in enumerate(meta):
        row = {k: stats(res[k][:, j]) for k in ("optimum", "frozen", "own", "padded", "union", "frozen_D", "frozen_W", "optimum_D")}
        at_W = {k: row[k]["median"] for k in ("padded", "union", "frozen_W")}
        best = min(at_W, key=at_W.get) if max(at_W.values()) > 1e-13 else "tie (roundoff)"
        print(f"  {m['chi']:4d} {m['D']:4d} " + " ".join(f"{row[k]['median']:9.1e}" for k in ("optimum", "frozen", "own"))
              + " | " + " ".join(f"{row[k]['median']:9.1e}" for k in ("padded", "union", "frozen_D", "optimum_D"))
              + f" | {m['chi_W']:5d} {row['frozen_W']['median']:9.1e} | {np.mean(res['overflow'][:, j]):8.0%}  {best}")
        viol = int(np.sum(np.column_stack([res[k][:, j] for k in ("frozen", "own", "frozen_W")])
                          < res["optimum"][:, j, None] - 1e-12 - 10 * floors[:, None])
                   + np.sum(np.column_stack([res[k][:, j] for k in ("padded", "union", "frozen_D")])
                            < res["optimum_D"][:, j, None] - 1e-12 - 10 * floors[:, None]))
        records.append(dict(L=L, U=U, trial=trial, chi=m["chi"], D=m["D"], chi_W=m["chi_W"],
                            costs={k: m[k] for k in ("frozen", "union", "frozen_D", "frozen_W")},
                            overflow=float(np.mean(res["overflow"][:, j])), violations=viol, floor_max=float(floors.max()),
                            n_train=info["train"], n_test=info["test"], how=info["how"], energy_train=info["energy_train"],
                            trial_chi=args.trial_chi, **{k: v for k, v in row.items()}))
    return records


def plot(records, out):
    import matplotlib.pyplot as plt
    ink2, grid, surface = "#52514e", "#e6e5e1", "#fcfcfb"
    plt.rcParams.update({"figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface,
                         "axes.edgecolor": grid, "axes.labelcolor": ink2, "axes.titlesize": 14, "axes.labelsize": 10,
                         "xtick.color": ink2, "ytick.color": ink2, "axes.grid": True, "grid.color": grid,
                         "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
                         "legend.frameon": False, "font.size": 10})
    style = {"frozen": ("#eb6834", "-", "frozen (production), cost chi"),
             "own": ("#d4a20f", "--", "own counts, cost chi (not jittable)"),
             "padded": ("#2a78d6", "-", "padded own counts, cost D"),
             "union": ("#1baf7a", "-", "union of paddings (frozen), cost D"),
             "frozen_D": ("#8e5bd0", "-", ""),
             "optimum": ("#52514e", ":", "optimum, any MPS")}
    for key in sorted({(r["trial"], r["U"], r["L"]) for r in records}):
        rows = sorted([r for r in records if (r["trial"], r["U"], r["L"]) == key], key=lambda r: r["chi"])
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        for name, (col, ls, lab) in style.items():
            if name == "frozen_D":
                continue
            if name == "frozen":                   # production at every bond it was run at: chi, chi_W and D
                pts = sorted({(r["chi"], r["frozen"]["median"]) for r in rows}
                             | {(r["chi_W"], r["frozen_W"]["median"]) for r in rows}
                             | {(r["D"], r["frozen_D"]["median"]) for r in rows})
                x, y = [p[0] for p in pts], [max(p[1], 1e-17) for p in pts]
                lab = "frozen (production), cost = its bond"
            else:
                x = [r["D"] if name in ("padded", "union") else r["chi"] for r in rows]
                y = [max(r[name]["median"], 1e-17) for r in rows]
            ax.plot(x, y, ls, marker="o", ms=4, color=col, label=lab)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("cost: largest bond per spin channel"); ax.set_ylabel(r"$1-F$ per spin channel (median)")
        trial, U, L = key
        ax.set_title(f"Allocation vs cost: L={L}, U={U:g}, {trial.upper()} trial")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        path = Path(out) / f"allocation_L{L}_U{U:g}_{trial}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path}")


# ------------------------------------------------------------------------------------------------ self-test
def selftest(args):
    """Small, exact checks of every piece against production code and dense states (L=8)."""
    print("self-test (L=8, U=4):")
    L, N, U = 8, 4, 4.0
    rng = np.random.default_rng(0)
    trial_np, trial_charges, _, gamma, nos, _ = dmrg_trial_mps(L, U, 8, 10)
    plan = make_orbital_plan(nos[0], "adaptive", EPS)
    walkers = [orthonormal(nos[0] + 0.3 * rng.standard_normal(nos[0].shape)) for _ in range(6)]
    worst = dict(frozen_vs_production=0.0, own_discarded_vs_plan_bonds=0.0, reference_vs_dense=0.0)
    violations = 0
    for chi in (2, 4, 8):
        bp = plan_bonds(nos[0], plan, chi, 0.0)
        frozen_counts = gmps(nos[0], plan, chi=chi)[2]
        f = jax.jit(jax.vmap(lambda q: channel_mps(q, plan, bp)[::2]))
        tj, gj = jax.tree_util.tree_map(np.asarray, f(jnp.asarray(np.stack(walkers))))
        for w, Q in enumerate(walkers):
            psi = dense_state(Q)
            mine, g, *_ = gmps(Q, plan, counts=frozen_counts)
            prod_v, mine_v = gj[w] * mps_to_dense([t[w] for t in tj]), g * mps_to_dense(mine)
            worst["frozen_vs_production"] = max(worst["frozen_vs_production"], np.abs(prod_v - mine_v).max())
            own = gmps(Q, plan, chi=chi)
            worst["own_discarded_vs_plan_bonds"] = max(worst["own_discarded_vs_plan_bonds"],
                                                       abs(own[3] - plan_bonds(Q, plan, chi, 0.0).reference_discarded_weight))
            ref = ("mps", gmps(Q, make_orbital_plan(Q, "adaptive", EPS))[0])
            for tensors in (mine, own[0]):
                worst["reference_vs_dense"] = max(worst["reference_vs_dense"],
                                                  abs(fidelity_loss(ref, tensors) - fidelity_loss(("dense", psi), tensors)))
                violations += int(fidelity_loss(("dense", psi), tensors) < optimum_discarded(Q, [chi])[0] - 1e-12)
    for k, v in worst.items():
        print(f"  max |{k}| = {v:.1e}")
    print(f"  errors below the optimum: {violations}")
    ok = worst["frozen_vs_production"] < 1e-10 and worst["own_discarded_vs_plan_bonds"] < 1e-10 \
        and worst["reference_vs_dense"] < 1e-8 and violations == 0
    print("  self-test", "passed" if ok else "FAILED")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--L", type=int, nargs="+", default=[8, 12, 16, 24, 32])
    parser.add_argument("--U", type=float, nargs="+", default=[4.0, 8.0])
    parser.add_argument("--trial", nargs="+", default=["dmrg"], choices=["dmrg", "uhf"])
    parser.add_argument("--chis", type=int, nargs="+", default=[2, 3, 4, 6, 8, 12, 16], help="nominal bonds per spin channel")
    parser.add_argument("--trial-chi", type=int, default=8, help="DMRG trial bond (the L=32 sweeps used 8)")
    parser.add_argument("--dmrg-sweeps", type=int, default=14)
    parser.add_argument("--run-chi-w", type=int, default=8, help="walker bond of the production MPS-CPMC runs (L > 12)")
    parser.add_argument("--walkers", type=int, default=50)
    parser.add_argument("--eql", type=int, default=10)
    parser.add_argument("--blocks", type=int, default=30)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234, help="training population; test uses seed + offset")
    parser.add_argument("--test-seed-offset", type=int, default=1000)
    parser.add_argument("--train-snaps", type=int, default=4)
    parser.add_argument("--test-snaps", type=int, default=4)
    parser.add_argument("--out", default=str(HERE / "allocation_study_data"))
    parser.add_argument("--selftest", action="store_true", help="run only the self-test")
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if not selftest(args):
        sys.exit(1)
    if args.selftest:
        return
    results = Path(args.out) / "results.jsonl"
    records = []
    for trial in args.trial:
        for U in args.U:
            for L in args.L:
                print(f"\n== L={L} U={U:g} trial={trial}", flush=True)
                res, meta, floors, info = study(L, U, trial, args)
                recs = summarize(L, U, trial, res, meta, floors, info, args)
                with results.open("a") as stream:
                    for r in recs:
                        stream.write(json.dumps(r) + "\n")
                records += recs
    plot(records, args.out)


if __name__ == "__main__":
    main()
