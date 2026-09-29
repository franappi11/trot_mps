#!/usr/bin/env python
"""Given a target accuracy, what is the cheapest walker truncation that reaches it: padding or not?

Production (mps_cpmc_new.py, mps_cpmc_gpu.py) truncates every walker's Fishman-White gMPS with the number of
kept states per charge sector frozen once, on a reference determinant (plan_bonds). A per-walker allocation is
much closer to the optimum (entanglement_vs_gmps.ipynb), but jitted code needs static shapes: every sector must
be padded to the largest count any walker uses, and the padded bond D is what the contraction costs. So for each
target accuracy: is it cheaper to reach it with a frozen allocation at a larger bond, or with a per-walker
allocation inside a padding?

For each (L, U, trial) two independent CPMC populations are run (train and test seeds). Paddings are learned on
the training walkers; every test walker (spin up) gets the infidelity 1-F of:

  scheme      kept counts per charge sector                                          evaluated at
  frozen      production: frozen on the plan reference (plan_bonds' rule)            every bond of a grid (the
                                                                                     nominal chis and every D)
  padded      the walker's own counts at nominal bond chi, capped by the training
              padding: the jittable per-walker scheme (GPU: dynamic_chi)             each chi (static shapes D)
  union       the padding used as a frozen allocation (every padded slot filled)     each chi (static shapes D)
  own         the walker's own best counts at chi (the ideal; not jittable)          each chi, reference only
  optimum     none: the smallest error any MPS with bond chi (or D) can have
              (worst cut, from the correlation spectrum; exact)                      each chi, reference only

Every jittable scheme gets a cost in three models:
  bond   the largest bond over the circuit's gates (memory, contraction size)
  work   the sum over gates of bond^3 (a CPU-like proxy for the conversion)
  gpu    the measured time per walker-step of mps_cpmc_gpu on this device (--gpu): conversion, contraction
         with the production DMRG trial and the HS sweep, both spins, at --gpu-walkers walkers. The padded scheme
         runs its real per-walker selection (eigh at every gate, top-chi within the caps), union and frozen their
         frozen allocations; each GPU conversion is checked against this file's NumPy gmps.
For each target (median, or --stats p90, of 1-F per spin channel) the study reports each scheme's cheapest
measured configuration that reaches it, and which scheme wins; schemes within --tie (5%) of each other are a
tie. No interpolation between configurations: small systems and latency-bound GPU steps have flat, noisy costs.
--report redoes the tables and plots from <out>/results.jsonl without computing anything.
D is the largest padded bond over the circuit's gates; the overflow rate is the fraction of test walkers whose
own allocation the training padding cannot hold at some gate.

Walkers come from production-like CPMC with the plan reference production uses (plan_reference="natural"):
  dmrg, L <= DENSE_TRIAL_MAX_L   trot CPMC with the DMRG trial exact in the determinant basis (no truncation)
  dmrg, larger L                 production's MPS-CPMC (make_walker_ops + make_fast_prop_ops), walker bond RUN_CHI_W
  uhf                            trot CPMC with a Neel UHF trial (plan reference = the UHF determinant)
Infidelities are exact against the dense walker state for L <= DENSE_MAX_L, otherwise against the walker's
untruncated own-plan gMPS, whose own error (the "floor", <~1e-9) is recorded.

Usage (from trot/gmps; populations, trials and GPU timings are cached in --out and re-used):
  ~/.trot/bin/python allocation_study.py --L 8 12 --U 4 8
  ~/.trot/bin/python allocation_study.py --selftest
  sbatch --export=ALL,TARGET=allocation_study.py run_mps_gpu.sh --L 8 12 16 24 32 --U 4 8 --gpu
  ~/.trot/bin/python allocation_study.py --report            # tables and plots from saved results only
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

import mps_cpmc_gpu as gpu  # noqa: E402
import mps_cpmc_new as mcn  # noqa: E402
from mps_cpmc_new import (_assemble, _move_centre, channel_angles, channel_mps, gate_pair,  # noqa: E402
                          hopping_matrix, make_orbital_plan, plan_bonds, sector_plan)
from jax import lax  # noqa: E402
from trot.core.ops import MeasOps, TrialOps, k_energy  # noqa: E402
from trot.core.system import System  # noqa: E402
from trot.driver import make_run_blocks  # noqa: E402
from trot.ham.chol import HamChol  # noqa: E402
from trot.ham.hubbard import HamHubbard  # noqa: E402
from trot.meas.uhf import make_uhf_meas_ops  # noqa: E402
from trot.prop import blocks, cpmc_slow  # noqa: E402
from trot.prop.hubbard_cpmc_ops import _build_prop_ctx  # noqa: E402
from trot.prop.types import PropState, QmcParams  # noqa: E402
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
def gmps(Q, plan, chi=None, counts=None, caps=None, trace=None):
    """Fishman-White gMPS of one orthonormal spin channel, in NumPy, with a chosen truncation.

    The same gates, orthogonality-centre moves and charge-blocked splits as mps_cpmc_new.channel_mps and its
    dry run plan_bonds. The number of states kept in each charge sector at gate g is
      counts[g] = {charge: k}   given: a frozen allocation (capped at the sector's rank), or
      chosen on Q itself        the chi largest squared singular values of the split (plan_bonds' rule),
                                optionally at most caps[g] = {charge: k_max} per sector, or
      everything                chi = counts = None (untruncated).
    Returns tensors, gauge (phi = gauge * MPS), chosen[g] = {charge: k}, the discarded weight, the kept bond per
    gate, and whether a cap was ever binding. With trace a list, one dict per gate is appended: how close its
    truncation came to a tie, relative to the gate's largest squared singular value (within: the smallest gap
    between a sector's last kept and first dropped value; across: smallest kept minus largest dropped overall).
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
        if trace is not None:
            top = max(float(d[1][0]) ** 2 for d in decs) or 1.0
            within = [(d[1][k - 1] ** 2 - d[1][k] ** 2) / top for d, k in zip(decs, kept) if 0 < k < len(d[1])]
            kept_v = [float(d[1][k - 1]) ** 2 for d, k in zip(decs, kept) if k]
            dropped = [float(d[1][k]) ** 2 for d, k in zip(decs, kept) if k < len(d[1])]
            trace.append(dict(gate=g, site=int(site), within=min(within, default=np.inf),
                              across=(min(kept_v) - max(dropped)) / top if dropped else np.inf,
                              discarded=float(sum(np.sum(d[1][k:] ** 2) for d, k in zip(decs, kept)))))
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


def population_path(L, U, trial, seed, args):
    return Path(args.out) / "walkers" / f"L{L}_U{U:g}_{trial}_chiT{args.trial_chi}_nw{args.walkers}_b{args.eql}+{args.blocks}_s{seed}.npz"


def population(L, U, trial, seed, args):
    path = population_path(L, U, trial, seed, args)
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


def timing_walkers(pop):
    """Both spins of the population's last snapshot, orthonormalised: the batch the GPU cost is timed on."""
    return (np.stack([orthonormal(W) for W in pop["up"][-1]]), np.stack([orthonormal(W) for W in pop["dn"][-1]]))


# ------------------------------------------------------------------------------------------------- GPU cost
def gpu_setup(L, U, args):
    """The production DMRG trial (cached), its charge-labelled H|trial> and the propagator of mps_cpmc_gpu."""
    cfg = gpu.Config(L=L, n_up=L // 2, n_down=L // 2, interaction=U, trial_chi=args.trial_chi,
                     dmrg_sweeps=args.dmrg_sweeps, trial_cache=str(Path(args.out) / "trial_cache"))
    trial_np, trial_charges, _ = gpu.load_or_run_trial(cfg)
    htrial = gpu.compress_mps_qn(*gpu.apply_mpo_qn(gpu.hubbard_mpo(L, 1.0, U), trial_np, trial_charges))
    prop_ctx = _build_prop_ctx(HamHubbard(h1=jnp.asarray(hopping_matrix(L, 1.0)), u=U), args.dt)
    return trial_np, trial_charges, htrial, prop_ctx


def time_on_device(plan, bond, dynamic_chi, reference, setup, sample, args):
    """Best-of time of args.gpu_steps CPMC steps of mps_cpmc_gpu for one allocation, and the largest
    infidelity between its conversion and the NumPy reference(Q) on a few sample walkers."""
    trial_np, trial_charges, htrial, prop_ctx = setup
    ops = gpu.make_gpu_ops(plan, plan, bond, bond, trial_np, trial_charges, htrial, prop_ctx, linalg="batched",
                           walker_qr="cholesky", spin_batch=True, energy="blocked", dynamic_chi=dynamic_chi)
    up, dn = sample
    nw = args.gpu_walkers
    reps = -(-nw // len(up))
    wu = jnp.asarray(np.concatenate([up] * reps)[:nw])
    wd = jnp.asarray(np.concatenate([dn] * reps)[:nw])
    params = QmcParams(dt=args.dt, n_walkers=nw, n_prop_steps=1, n_blocks=1, n_eql_blocks=0,
                       weight_floor=WEIGHT_FLOOR, seed=0)
    memory, limit = gpu.memory_model(ops), gpu.device_bytes_limit()
    budget = None if limit is None else args.gpu_mem_fraction * limit - memory["data_bytes"]
    n_chunks = gpu.choose_chunks(nw, memory["step_bytes_per_walker"], budget)
    half = gpu.make_half_step(ops, params, n_chunks)
    n_half = 2 * args.gpu_steps
    propagate = jax.jit(lambda s, d: lax.scan(lambda st, i: (half(st, i, d), None), s, jnp.arange(n_half))[0])
    shift = -0.5 * up.shape[1]  # any finite energy: only the timing matters
    state = PropState((wu, wd), jnp.ones(nw, jnp.float64), jax.jit(ops.overlaps)(wu, wd, ops.data),
                      jax.random.PRNGKey(0), jnp.asarray(shift, jnp.float64), jnp.asarray(shift, jnp.float64),
                      jnp.zeros((), jnp.int64))
    t0 = time.perf_counter()
    compiled = propagate.lower(state, ops.data).compile()
    compile_seconds = time.perf_counter() - t0
    jax.block_until_ready(compiled(state, ops.data))
    best = float("inf")
    for _ in range(args.gpu_repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(compiled(state, ops.data))
        best = min(best, time.perf_counter() - t0)
    # Checks per sample walker: how far the GPU state is from the NumPy one, and which is closer to the exact walker.
    # Under heavy truncation a nearly degenerate mode (channel_angles) or a nearly tied cut can be resolved
    # differently by the two eigensolvers: equally valid conversions whose errors differ either way. A bug would
    # instead make the GPU worse on (nearly) every differing walker.
    convert, check, error_gap = jax.jit(ops.converter.convert), 0.0, 0.0
    differing = gpu_worse = 0
    for q_up, q_dn in zip(up[:args.gpu_check], dn[:args.gpu_check]):
        alpha, beta, _ = convert(jnp.asarray(q_up), jnp.asarray(q_dn))
        for Q, tensors in ((q_up, alpha), (q_dn, beta)):
            dev, want = [np.asarray(t) for t in tensors], reference(Q)
            state = fidelity_loss(("mps", want), dev)
            check = max(check, state)
            exact, _ = exact_state(Q)
            e_numpy, e_gpu = fidelity_loss(exact, want), fidelity_loss(exact, dev)
            error_gap = max(error_gap, abs(e_gpu - e_numpy) / max(e_numpy, 1e-12))
            if state > 1e-9:
                differing += 1
                gpu_worse += int(e_gpu > e_numpy * (1 + 1e-3))
    device = jax.devices()[0]
    return dict(seconds=best, ms_per_step=1e3 * best / args.gpu_steps, us_per_walker_step=1e6 * best / (nw * args.gpu_steps),
                compile_seconds=compile_seconds, n_chunks=n_chunks, walkers=nw, device=device.device_kind,
                backend=jax.default_backend(), circuit=gpu.circuit_stats(ops.converter.circuits[0]),
                gpu_vs_numpy=check, error_gap=error_gap, differing=differing, gpu_worse=gpu_worse)


def gpu_costs(L, U, plan, grid, frozen_counts, chis, pads, sample, pop_tag, args):
    """Measured device cost of every jittable point: frozen at each grid bond, union and padded at each chi.
    Cached in <out>/gpu_costs.jsonl by population, scheme, parameter, batch, device and JAX version."""
    cache_path = Path(args.out) / "gpu_costs.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            record = json.loads(line)
            cache[record["key"]] = record
    setup = None
    device = jax.devices()[0].device_kind
    configs = [("frozen", b, frozen_counts[b], None, lambda Q, c=frozen_counts[b]: gmps(Q, plan, counts=c)[0])
               for b in grid]
    configs += [("union", chi, pad, None, lambda Q, c=pad: gmps(Q, plan, counts=c)[0]) for chi, pad in zip(chis, pads)]
    configs += [("padded", chi, pad, chi, lambda Q, c=pad, x=chi: gmps(Q, plan, chi=x, caps=c)[0])
                for chi, pad in zip(chis, pads)]
    out = {}
    for name, param, counts, dynamic_chi, reference in configs:
        key = (f"{pop_tag}|train{args.train_snaps}|{name}|{param}|nw{args.gpu_walkers}|steps{args.gpu_steps}"
               f"|chiT{args.trial_chi}|{device}|jax{jax.__version__}")
        if key in cache and not args.gpu_retime:
            out[name, param] = cache[key]
            continue
        setup = setup or gpu_setup(L, U, args)
        record = time_on_device(plan, gpu.counts_bond_plan(plan, counts), dynamic_chi, reference, setup, sample, args)
        record.update(key=key, L=L, U=U, scheme=name, param=int(param))
        with cache_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        out[name, param] = record
        n, worse = record.get("differing", 0), record.get("gpu_worse", 0)
        if record["gpu_vs_numpy"] <= 1e-9:
            flag = ""
        elif n >= 3 and worse == n:
            flag = f"   <-- GPU worse on all {n} differing walkers: run allocation_gpu_check.py"
        else:
            flag = f"   ({n} walkers differ, GPU worse on {worse}: near-degenerate choices, either way)"
        print(f"    gpu {name:>6s} {param:3d}: {record['us_per_walker_step']:9.3f} us/walker-step, "
              f"compile {record['compile_seconds']:5.1f} s, vs NumPy {record['gpu_vs_numpy']:.1e}, "
              f"error gap {record['error_gap']:.1e}{flag}", flush=True)
    return out


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

    # paddings learned on the training walkers; their padded bond D
    pads = [padding([gmps(Q, plan, chi=chi)[2] for Q in train]) for chi in chis]
    union_bonds = [gmps(ref, plan, counts=pad)[4] for pad in pads]
    Ds = [int(max(b)) for b in union_bonds]
    # production on a bond grid that covers every padded scheme's cost
    grid = sorted(set(chis) | set(Ds) | set(args.extra_bonds))
    frozen_counts = {b: gmps(ref, plan, chi=b)[2] for b in grid}
    frozen_bonds = {b: gmps(ref, plan, counts=frozen_counts[b])[4] for b in grid}

    res = {k: np.full((len(test), len(chis)), np.nan) for k in ("optimum", "optimum_D", "own", "padded", "union")}
    res["overflow"] = np.zeros((len(test), len(chis)), bool)
    res["frozen"] = np.full((len(test), len(grid)), np.nan)
    for w, (Q, exact) in enumerate(zip(test, exacts)):
        for i, b in enumerate(grid):
            res["frozen"][w, i] = fidelity_loss(exact, gmps(Q, plan, counts=frozen_counts[b])[0])
        for j, (chi, pad, D) in enumerate(zip(chis, pads, Ds)):
            res["own"][w, j] = fidelity_loss(exact, gmps(Q, plan, chi=chi)[0])
            tensors, *_, capped = gmps(Q, plan, chi=chi, caps=pad)
            res["padded"][w, j], res["overflow"][w, j] = fidelity_loss(exact, tensors), capped
            res["union"][w, j] = fidelity_loss(exact, gmps(Q, plan, counts=pad)[0])
            res["optimum"][w, j], res["optimum_D"][w, j] = optimum_discarded(Q, [chi, D])
    print(f"    errors of {len(test)} test walkers at {len(grid)} frozen bonds and {len(chis)} nominal chis "
          f"({time.perf_counter() - start:.0f} s)", flush=True)

    work = lambda b: int(np.sum(np.asarray(b) ** 3))
    points = dict(
        frozen=[dict(param=int(b), column=i, cost=dict(bond=int(max(frozen_bonds[b])), work=work(frozen_bonds[b])))
                for i, b in enumerate(grid)],
        padded=[dict(param=int(chi), column=j, D=D, cost=dict(bond=D, work=work(ub)))
                for j, (chi, D, ub) in enumerate(zip(chis, Ds, union_bonds))],
        union=[dict(param=int(chi), column=j, D=D, cost=dict(bond=D, work=work(ub)))
               for j, (chi, D, ub) in enumerate(zip(chis, Ds, union_bonds))])
    if args.gpu:
        pop_tag = Path(population_path(L, U, trial, args.seed, args)).stem
        costs = gpu_costs(L, U, plan, grid, frozen_counts, chis, pads, timing_walkers(test_pop), pop_tag, args)
        for name, rows in points.items():
            for p in rows:
                record = costs[name, p["param"]]
                p["cost"]["gpu"] = record["us_per_walker_step"]
                p["gpu_vs_numpy"] = record["gpu_vs_numpy"]
                p["error_gap"] = record.get("error_gap")
    np.savez_compressed(out_npz, chis=np.array(chis), grid=np.array(grid), D=np.array(Ds), floor=np.array(floors),
                        **res, points=json.dumps(points))
    info = dict(train=len(train), test=len(test), how=str(train_pop["how"]),
                energy_train=float(np.mean(train_pop["energies"][args.eql:])),
                gates=int((plan.block_sizes - 1).sum()), plan_max_B=int(plan.block_sizes.max()),
                device=jax.devices()[0].device_kind if args.gpu else None)
    return res, points, np.array(floors), info


# ------------------------------------------------------------------------------------------ cost to a target
COST_MODELS = {"bond": "largest bond", "work": "sum over gates of bond^3", "gpu": "GPU us per walker-step"}


def cost_to_reach(points, target, model, stat):
    """The cheapest measured configuration of the scheme whose error meets target: (cost, param, how), how "<="
    when even the scheme's cheapest configuration meets it (the true cheapest cost may be lower), None if no
    configuration on the grid does. No interpolation: on a flat or noisy cost curve (small L, or a latency-bound
    GPU step) interpolating between configurations invents costs no configuration has."""
    meeting = [(p["cost"][model], p["param"]) for p in points if model in p["cost"] and p["err"][stat] <= target]
    if not meeting:
        return None, None, None
    cost, param = min(meeting)
    cheapest = min(p["cost"][model] for p in points if model in p["cost"])
    return cost, param, "<=" if cost <= cheapest else "measured"


def summarize(L, U, trial, res, points, floors, info, args):
    stats = lambda x: dict(median=float(np.median(x)), p90=float(np.percentile(x, 90)),
                           p99=float(np.percentile(x, 99)), mean=float(np.mean(x)), max=float(np.max(x)))
    for name, rows in points.items():
        for p in rows:
            p["err"] = stats(res[name][:, p["column"]])
    chis = [p["param"] for p in points["padded"]]
    print(f"\n  L={L} U={U:g} {trial}: {info['how']}; {info['train']} training and {info['test']} test walkers; "
          f"circuit {info['gates']} gates (max B {info['plan_max_B']}); reference floor max {floors.max():.1e}")
    print(f"  median 1-F per spin channel at each nominal chi (padded and union cost D):")
    print(f"  {'chi':>4s} {'D':>4s} {'optimum':>9s} {'own':>9s} | {'padded':>9s} {'union':>9s} {'opt(D)':>9s} | "
          f"{'overflow':>8s} {'violations':>10s}")
    violations = 0
    for j, chi in enumerate(chis):
        D = points["padded"][j]["D"]
        viol = int(np.sum(res["own"][:, j] < res["optimum"][:, j] - 1e-12 - 10 * floors)
                   + np.sum(np.column_stack([res[k][:, j] for k in ("padded", "union")])
                            < res["optimum_D"][:, j, None] - 1e-12 - 10 * floors[:, None]))
        violations += viol
        med = lambda k: float(np.median(res[k][:, j]))
        print(f"  {chi:4d} {D:4d} {med('optimum'):9.1e} {med('own'):9.1e} | {med('padded'):9.1e} {med('union'):9.1e} "
              f"{med('optimum_D'):9.1e} | {np.mean(res['overflow'][:, j]):8.0%} {viol:10d}")
    frozen = points["frozen"]
    print("  frozen (production): " + "  ".join(f"b={p['param']}: {p['err']['median']:.1e}" for p in frozen))
    worst_check = max((p.get("gpu_vs_numpy", 0.0) for rows in points.values() for p in rows), default=0.0)
    worst_gap = max((p.get("error_gap") or 0.0 for rows in points.values() for p in rows), default=0.0)
    record = dict(L=L, U=U, trial=trial, trial_chi=args.trial_chi, n_train=info["train"], n_test=info["test"],
                  how=info["how"], energy_train=info["energy_train"], floor_max=float(floors.max()),
                  violations=violations, device=info["device"], points=points,
                  overflow={int(chi): float(np.mean(res["overflow"][:, j])) for j, chi in enumerate(chis)},
                  own={int(chi): float(np.median(res["own"][:, j])) for j, chi in enumerate(chis)},
                  optimum={int(chi): float(np.median(res["optimum"][:, j])) for j, chi in enumerate(chis)},
                  gpu_vs_numpy_max=worst_check, error_gap_max=worst_gap)
    record["cost_to_target"] = report_costs(record, args)
    return [record]


def report_costs(record, args):
    """Print, for every statistic and cost model, the cheapest measured configuration of each scheme that meets
    each target; a scheme within args.tie of the cheapest is reported as a tie (timing noise, flat costs)."""
    points, schemes = record["points"], ("frozen", "padded", "union")
    label = dict(frozen="b", padded="chi", union="chi")
    models = [m for m in COST_MODELS if all(m in p["cost"] for rows in points.values() for p in rows)]
    table = []
    for stat in args.stats:
        for model in models:
            print(f"\n  cheapest configuration reaching {stat} 1-F <= target; cost = {COST_MODELS[model]}"
                  + (f" ({record['device']})" if model == "gpu" else ""))
            print(f"  {'target':>8s} " + " ".join(f"{s:>16s}" for s in schemes) + "   best")
            for target in args.targets:
                row = {s: cost_to_reach(points[s], target, model, stat) for s in schemes}
                reached = {s: r[0] for s, r in row.items() if r[0] is not None}
                if reached:
                    best = min(reached, key=reached.get)
                    ties = [s for s in reached if s != best and reached[s] <= (1 + args.tie) * reached[best]]
                    others = sorted(v for s, v in reached.items() if s != best)
                    if ties:
                        verdict = f"tie: {best}, {', '.join(ties)} (within {args.tie:.0%})"
                    elif others:
                        verdict = f"{best} ({others[0] / reached[best]:.2f}x cheaper than the next)"
                    else:
                        verdict = f"{best} (the only scheme that reaches it)"
                    if row[best][2] == "<=":
                        verdict += "; met at its cheapest configuration (true cost may be lower)"
                else:
                    best, verdict = None, "no configuration on the grid reaches it"

                def cell(s):
                    cost, param, how = row[s]
                    if cost is None:
                        return "-"
                    return f"{'<=' if how == '<=' else ''}{cost:.4g} ({label[s]}{param})"
                print(f"  {target:8.0e} " + " ".join(f"{cell(s):>16s}" for s in schemes) + f"   {verdict}")
                table.append(dict(stat=stat, model=model, target=target, best=best,
                                  ties=ties if reached else [],
                                  **{s: dict(cost=row[s][0], param=row[s][1], how=row[s][2]) for s in row}))
    if any("gpu" in p["cost"] for rows in points.values() for p in rows):
        worst, gap = record.get("gpu_vs_numpy_max", 0.0), record.get("error_gap_max")
        print(f"\n  GPU conversions vs NumPy gmps: largest state infidelity {worst:.1e}"
              + ("" if gap is None else f", largest relative difference of their errors {gap:.1e}")
              + ("" if worst <= 1e-9 else
                 "\n  (the errors above are NumPy's and the GPU costs depend only on the shapes, so neither depends on"
                 " how near-degenerate choices are resolved; allocation_gpu_check.py shows where the states differ)"))
    return table


def plot(records, args):
    """One figure per system, one panel per cost model. Tick labels are plain text (no mathtext: FreeType failed
    rendering a log-axis label with 'raster overflow' in job 7126084), and a failing figure is reported, not fatal:
    the results are already in results.jsonl and --report can redraw them."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter
    ink2, grid, surface = "#52514e", "#e6e5e1", "#fcfcfb"
    plt.rcParams.update({"figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface,
                         "axes.edgecolor": grid, "axes.labelcolor": ink2, "axes.titlesize": 12, "axes.labelsize": 10,
                         "xtick.color": ink2, "ytick.color": ink2, "axes.grid": True, "grid.color": grid,
                         "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
                         "legend.frameon": False, "font.size": 10})
    style = {"frozen": ("#eb6834", "frozen (production)"), "padded": ("#2a78d6", "padded (per walker)"),
             "union": ("#1baf7a", "union of paddings (frozen)")}
    stat = args.stats[0]
    for r in records:
        models = [m for m in COST_MODELS if all(m in p["cost"] for rows in r["points"].values() for p in rows)]
        fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.6), squeeze=False)
        for ax, model in zip(axes[0], models):
            for name, (col, lab) in style.items():
                pts = sorted((p["cost"][model], max(p["err"][stat], 1e-17)) for p in r["points"][name])
                ax.plot([c for c, _ in pts], [e for _, e in pts], "-", marker="o", ms=4, color=col, label=lab)
            if model == "bond":  # references with no jittable cost: the ideal own counts and the optimum, at chi
                own, opt = sorted(r["own"].items()), sorted(r["optimum"].items())
                ax.plot([c for c, _ in own], [max(e, 1e-17) for _, e in own], "--", color="#d4a20f", label="own (not jittable)")
                ax.plot([c for c, _ in opt], [max(e, 1e-17) for _, e in opt], ":", color=ink2, label="optimum, any MPS")
            for target in args.targets:
                ax.axhline(target, color=grid, lw=0.8, zorder=0)
            ax.set_xscale("log", base=2 if model == "bond" else 10)
            ax.set_yscale("log")
            ax.set_ylim(3e-17, 1.0)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            # the GPU costs span less than a decade: label its minor ticks, or the axis has no labels
            ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}") if model == "gpu" else NullFormatter())
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0e}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(COST_MODELS[model] + (f"\n({r['device']})" if model == "gpu" else ""))
            ax.set_ylabel(f"{stat} 1-F per spin channel")
        axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.suptitle(f"Accuracy vs cost: L={r['L']}, U={r['U']:g}, {r['trial'].upper()} trial (trial chi {r['trial_chi']})")
        path = Path(args.out) / f"allocation_L{r['L']}_U{r['U']:g}_{r['trial']}.png"
        write_curves(r, path.with_suffix(".csv"))
        error = None
        # Retry without FreeType hinting: jobs 7126084/7126438 hit "raster overflow" in FT_Render_Glyph with the
        # module's matplotlib even with plain-text labels.
        for settings in ({}, {"text.hinting": "no_hinting", "text.hinting_factor": 1}):
            try:
                with plt.rc_context(settings):
                    fig.savefig(path, dpi=200, bbox_inches="tight")
                print(f"  saved {path}" + (" (without text hinting)" if settings else ""))
                error = None
                break
            except Exception as caught:  # the numbers are saved; a figure is not worth losing a run for
                error = caught
        if error is not None:
            print(f"  could not draw {path}: {type(error).__name__}: {error}; its data is in {path.with_suffix('.csv')}")
        plt.close(fig)


def write_curves(record, path):
    """The plotted curves as CSV (one row per configuration and reference point), to plot anywhere."""
    lines = ["scheme,param,D,cost_bond,cost_work,cost_gpu_us_per_walker_step,err_median,err_p90,err_p99,err_mean,err_max"]
    for name, rows in record["points"].items():
        for p in rows:
            c, e = p["cost"], p["err"]
            lines.append(",".join(str(x) for x in (name, p["param"], p.get("D", p["param"]), c["bond"], c["work"],
                                                   c.get("gpu", ""), e["median"], e["p90"], e["p99"], e["mean"], e["max"])))
    for name in ("own", "optimum"):  # references at nominal chi: medians only
        for chi, err in sorted(record[name].items()):
            lines.append(f"{name},{chi},,,,,{err},,,,")
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path}")


# ------------------------------------------------------------------------------------------------ self-test
def selftest(args):
    """Small, exact checks of every piece against production code and dense states (L=8), including the GPU
    conversions of every scheme (frozen, union, padded) against this file's NumPy gmps."""
    print("self-test (L=8, U=4):")
    L, N, U = 8, 4, 4.0
    rng = np.random.default_rng(0)
    trial_np, trial_charges, _, gamma, nos, _ = dmrg_trial_mps(L, U, 8, 10)
    plan = make_orbital_plan(nos[0], "adaptive", EPS)
    walkers = [orthonormal(nos[0] + 0.3 * rng.standard_normal(nos[0].shape)) for _ in range(6)]
    others = [orthonormal(nos[0] + 0.6 * rng.standard_normal(nos[0].shape)) for _ in range(4)]
    worst = dict(frozen_vs_production=0.0, own_discarded_vs_plan_bonds=0.0, reference_vs_dense=0.0,
                 gpu_frozen_vs_numpy=0.0, gpu_union_vs_numpy=0.0, gpu_padded_vs_numpy=0.0)
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
    for chi in (2, 3, 4):  # GPU conversions of the three jittable schemes, capped walkers included
        pad = padding([gmps(Q, plan, chi=chi)[2] for Q in walkers[:4]])
        frozen_counts = gmps(nos[0], plan, chi=chi)[2]
        cases = [("gpu_frozen_vs_numpy", frozen_counts, None, lambda Q: gmps(Q, plan, counts=frozen_counts)),
                 ("gpu_union_vs_numpy", pad, None, lambda Q: gmps(Q, plan, counts=pad)),
                 ("gpu_padded_vs_numpy", pad, chi, lambda Q: gmps(Q, plan, chi=chi, caps=pad))]
        for name, counts, dynamic_chi, reference in cases:
            bond = gpu.counts_bond_plan(plan, counts)
            convert = jax.jit(gpu.make_converter(plan, plan, bond, bond, linalg="batched", spin_batch=True,
                                                 dynamic_chi=dynamic_chi).convert)
            for Qa, Qb in zip(walkers + others, (others + walkers)[::-1]):
                alpha, beta, gauges = convert(jnp.asarray(Qa), jnp.asarray(Qb))
                for Q, tensors, gauge in ((Qa, alpha, gauges[0]), (Qb, beta, gauges[1])):
                    want, g_want, *_ = reference(Q)
                    worst[name] = max(worst[name], np.abs(float(gauge) * mps_to_dense([np.asarray(t) for t in tensors])
                                                          - g_want * mps_to_dense(want)).max())
    for k, v in worst.items():
        print(f"  max |{k}| = {v:.1e}")
    print(f"  errors below the optimum: {violations}")
    ok = worst["frozen_vs_production"] < 1e-10 and worst["own_discarded_vs_plan_bonds"] < 1e-10 \
        and worst["reference_vs_dense"] < 1e-8 and violations == 0 \
        and all(worst[k] < 1e-10 for k in ("gpu_frozen_vs_numpy", "gpu_union_vs_numpy", "gpu_padded_vs_numpy"))
    print("  self-test", "passed" if ok else "FAILED")
    return ok


# options that change a system's results: a saved record is reused only if all of them match
SETTINGS = ("chis", "extra_bonds", "trial_chi", "dmrg_sweeps", "run_chi_w", "walkers", "eql", "blocks", "steps",
            "dt", "seed", "test_seed_offset", "train_snaps", "test_snaps", "gpu", "gpu_walkers", "gpu_steps",
            "gpu_repeats")


def settings_of(args):
    return {k: getattr(args, k) for k in SETTINGS}


def reusable(previous, args):
    """A saved record can stand in for recomputing its system. Records written before settings were saved
    (e.g. by job 7126487) match on what they contain: the chi grid, the test-walker count and GPU costs."""
    if previous is None:
        return False
    if "settings" in previous:
        return previous["settings"] == settings_of(args)
    chis = sorted(p["param"] for p in previous["points"]["padded"])
    has_gpu = all("gpu" in p["cost"] for rows in previous["points"].values() for p in rows)
    return chis == sorted(args.chis) and previous["n_test"] == args.walkers * args.test_snaps and has_gpu == args.gpu


def load_records(results):
    """Every record of results.jsonl (later ones win), keyed by (L, U, trial, trial_chi)."""
    latest = {}
    if results.exists():
        for line in results.read_text().splitlines():
            r = json.loads(line)
            for key in ("own", "optimum", "overflow"):  # JSON turned the integer keys into strings
                r[key] = {int(k): v for k, v in r[key].items()}
            latest[r["L"], r["U"], r["trial"], r["trial_chi"]] = r
    return latest


def report(results, args):
    """Redo the cost-to-target tables and plots from saved results: the latest record of every system in
    results.jsonl, restricted to --L/--U/--trial when they are given on the command line."""
    chosen = [r for _, r in sorted(load_records(results).items())
              if (not args.only_given or (r["L"] in args.L and r["U"] in args.U and r["trial"] in args.trial))]
    for r in chosen:
        print(f"\n== L={r['L']} U={r['U']:g} trial={r['trial']} (trial chi {r['trial_chi']}): {r['how']}; "
              f"{r['n_test']} test walkers")
        r["cost_to_target"] = report_costs(r, args)
    plot(chosen, args)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--L", type=int, nargs="+", default=[8, 12, 16, 24, 32])
    parser.add_argument("--U", type=float, nargs="+", default=[4.0, 8.0])
    parser.add_argument("--trial", nargs="+", default=["dmrg"], choices=["dmrg", "uhf"])
    parser.add_argument("--chis", type=int, nargs="+", default=[2, 3, 4, 6, 8, 12, 16], help="nominal bonds per spin channel")
    parser.add_argument("--extra-bonds", type=int, nargs="*", default=[],
                        help="more bonds for the frozen curve (it always includes every chi and every padded D)")
    parser.add_argument("--targets", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
                        help="target accuracies (1-F per spin channel) for the cost-to-target tables")
    parser.add_argument("--stats", nargs="+", default=["median"], choices=["median", "p90", "p99", "mean", "max"],
                        help="which statistic of 1-F over test walkers must meet the target (the first one is plotted)")
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
    parser.add_argument("--gpu", action="store_true", help="measure each scheme's mps_cpmc_gpu step time on this device")
    parser.add_argument("--gpu-walkers", type=int, default=1024, help="walker batch the GPU cost is timed at")
    parser.add_argument("--gpu-steps", type=int, default=10, help="CPMC steps per timed call")
    parser.add_argument("--gpu-repeats", type=int, default=5, help="timed calls per configuration (best is kept)")
    parser.add_argument("--tie", type=float, default=0.05,
                        help="schemes within this relative cost of the cheapest are reported as a tie")
    parser.add_argument("--gpu-check", type=int, default=4, help="walkers whose GPU conversion is checked against NumPy")
    parser.add_argument("--gpu-mem-fraction", type=float, default=0.75)
    parser.add_argument("--gpu-retime", action="store_true", help="ignore cached GPU timings")
    parser.add_argument("--compile-cache", default=str(Path.home() / ".cache" / "trot_jax_compile"))
    parser.add_argument("--out", default=str(HERE / "allocation_study_data"))
    parser.add_argument("--selftest", action="store_true", help="run only the self-test")
    parser.add_argument("--report", action="store_true",
                        help="only redo the cost-to-target tables and plots from <out>/results.jsonl (no computation)")
    parser.add_argument("--recompute", action="store_true",
                        help="recompute systems even when results.jsonl has them with the same settings")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.only_given = any(getattr(args, k) != parser.get_default(k) for k in ("L", "U", "trial"))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    results = Path(args.out) / "results.jsonl"
    if args.report:
        report(results, args)
        return
    if args.gpu and args.compile_cache:
        jax.config.update("jax_compilation_cache_dir", args.compile_cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    if args.gpu and jax.default_backend() != "gpu":
        print(f"note: --gpu on the {jax.default_backend()} backend: the 'gpu' cost is that device's time")
    if not selftest(args):
        sys.exit(1)
    if args.selftest:
        return
    records, saved = [], load_records(results)
    for trial in args.trial:
        for U in args.U:
            for L in args.L:
                print(f"\n== L={L} U={U:g} trial={trial}", flush=True)
                previous = saved.get((L, U, trial, args.trial_chi))
                if reusable(previous, args) and not args.recompute:
                    print(f"  reusing the saved result (same settings; --recompute to redo it)")
                    previous["cost_to_target"] = report_costs(previous, args)
                    records.append(previous)
                    continue
                res, points, floors, info = study(L, U, trial, args)
                recs = summarize(L, U, trial, res, points, floors, info, args)
                with results.open("a") as stream:
                    for r in recs:
                        r["settings"] = settings_of(args)
                        stream.write(json.dumps(r) + "\n")
                records += recs
    plot(records, args)


if __name__ == "__main__":
    main()
