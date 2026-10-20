"""Kept counts per charge sector trained on walkers, instead of frozen on the walker plan's reference determinant.

A walker plan freezes, at every gate of the conversion, how many states each charge sector keeps
(BondPlan.kept_per_sector): every walker's channel MPS has those bonds, the static shapes the batched engine compiles.
plan_bonds freezes them on the reference, its own top chi. Here they come from many walkers: each walker's own top-chi
choice on the plan's gates is recorded with the Schmidt weights it saw (walker_allocation), a rule turns them into one
allocation per gate (train_counts), and plan_bonds(..., counts=) makes it a bond plan for the same gates
(trained_bond_plans, trained_walker_plan). At the doped 6x6 (chi_T 256, chi_w 16) counts frozen on the trial's
natural determinant discard 21-25% of a walker's norm, its own counts 3-4%, consensus counts trained on 256 walkers 11%.

Rules, at bond chi (the cost of the frozen counts at chi):
  consensus   per gate the chi slots (charge q, j-th state) of largest Schmidt weight s^2 summed over the walkers: the
              most kept weight on average;
  amplitude   the same with the singular values s: more say to the small components the local energy is sensitive to;
  vote        the chi slots most often among the walkers' own top chi: as many of each walker's own states as possible.
Each walker's weights fall with j, so do the scores: the slots a rule keeps in a sector are its first ones. With a
larger bond: "p<k>" keeps per gate and sector the k-th percentile over the walkers of their own counts, "union" the
maximum (what per-walker counts would cost on static shapes).
"""
from __future__ import annotations

import numpy as np

from trot.gmps.utils import plan_bonds

RULES = ("consensus", "amplitude", "vote")


def walker_allocation(job):
    """One walker channel's own top-chi counts on given gates: job = (C, orbital_plan, chi) -> (per gate {charge:
    kept}, per gate {charge: squared singular values, descending}, the weight it discards). Picklable for a pool."""
    C, orbital_plan, chi = job
    Q = np.linalg.qr(np.asarray(C, dtype=float))[0]
    spectra = []
    bond = plan_bonds(Q, orbital_plan, chi, spectra=spectra)
    counts = [dict(zip(gate.keys(), kept)) for gate, kept in zip(spectra, bond.kept_per_sector)]
    return counts, spectra, bond.reference_discarded_weight


def allocations_of(walkers, orbital_plan, chi, pool=None):
    """walker_allocation of every walker channel C in walkers (a sequence of L x N arrays), in a pool if given."""
    jobs = [(C, orbital_plan, chi) for C in walkers]
    return list(pool.map(walker_allocation, jobs, chunksize=1) if pool is not None else map(walker_allocation, jobs))


def train_counts(allocations, chi=None, rule="consensus", weights=None):
    """One allocation per gate, [{charge: kept}], from the walkers' allocations (walker_allocation) by a rule of the
    module docstring: consensus, amplitude or vote at bond chi, or "p<k>" / "union" (chi unused). weights: optional
    per-walker weights of the sums (equal by default)."""
    weights = np.ones(len(allocations)) if weights is None else np.asarray(weights, dtype=float)
    n_gates = len(allocations[0][0])
    if rule == "union" or rule.startswith("p"):
        percentile = 100.0 if rule == "union" else float(rule[1:])
        out = []
        for g in range(n_gates):
            charges = set().union(*(counts[g] for counts, _, _ in allocations))
            out.append({q: int(np.ceil(np.percentile([counts[g].get(q, 0) for counts, _, _ in allocations],
                                                     percentile))) for q in charges})
        return out
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, 'p<percentile>' or 'union', got {rule!r}")
    if chi is None:
        raise ValueError(f"rule {rule!r} needs a bond chi")
    out = []
    for g in range(n_gates):
        scores = {}
        for weight, (counts, spectra, _) in zip(weights, allocations):
            for q, s2 in spectra[g].items():
                if rule == "vote":
                    value = (np.arange(len(s2)) < counts[g].get(q, 0)).astype(float)
                else:
                    value = s2 if rule == "consensus" else np.sqrt(np.maximum(s2, 0.0))
                acc = scores.get(q, np.zeros(0))
                if len(value) > len(acc):
                    acc = np.concatenate((acc, np.zeros(len(value) - len(acc))))
                acc[:len(value)] += weight * value
                scores[q] = acc
        slots = sorted(((value, q, j) for q, acc in scores.items() for j, value in enumerate(acc)),
                       key=lambda x: (-x[0], x[1], x[2]))[:int(chi)]
        kept = {}
        for _, q, _ in slots:
            kept[q] = kept.get(q, 0) + 1
        out.append(kept)
    return out


def trained_bond_plans(references, orbital_plans, walkers, chi=None, rule="consensus", own_chi=None, pool=None,
                       weights=None):
    """Bond plans (per spin) of counts trained on walkers = (up, dn) stacks of channel orbitals, on the gates
    orbital_plans frozen on references. own_chi: the bond of the walkers' own choices (default chi; set it for the
    percentile rules, which take no chi)."""
    own_chi = chi if own_chi is None else own_chi
    if own_chi is None:
        raise ValueError("give chi or own_chi")
    plans = []
    for R, orbital_plan, channel in zip(references, orbital_plans, walkers):
        allocations = allocations_of(list(channel), orbital_plan, own_chi, pool)
        plans.append(plan_bonds(R, orbital_plan, counts=train_counts(allocations, chi, rule, weights)))
    return tuple(plans)


def channel_loss(job):
    """(plan loss, discarded weight) of one walker channel: job = (C, orbital_plan, bond_plan). The plan loss
    1 - |gauge|^2 is what the gates cannot represent at all; the discarded weight what the kept counts cut, 1 - <MPS|MPS>
    of the host conversion (engine.channel_mps_host). Picklable for a pool."""
    from trot.gmps import engine
    from trot.gmps.utils import channel_angles

    C, orbital_plan, bond = job
    Q = np.linalg.qr(np.asarray(C, dtype=float))[0]
    _, rows = channel_angles(Q, orbital_plan, xp=np)
    gauge = float(np.linalg.det(np.stack([rows[i] for i in np.flatnonzero(orbital_plan.occupation)])))
    tensors, _, _ = engine.channel_mps_host(Q, orbital_plan, bond)
    return 1.0 - gauge**2, 1.0 - engine.mps_overlap_host(tensors, tensors)


def discarded_under(walkers, orbital_plans, bond_plans, pool=None):
    """(n, 2) discarded weights of walkers = (up, dn) stacks under per-spin gates and bond plans (channel_loss)."""
    out = []
    for channel, orbital_plan, bond in zip(walkers, orbital_plans, bond_plans):
        jobs = [(C, orbital_plan, bond) for C in channel]
        losses = pool.map(channel_loss, jobs, chunksize=1) if pool is not None else map(channel_loss, jobs)
        out.append([w for _, w in losses])
    return np.asarray(out).T


def slots_changed(bonds_a, bonds_b):
    """Fraction of kept slots (gate, charge, state) of bonds_a that bonds_b does not keep, over both spins; the
    allocations must share their gates. Counts are compared per gate and sector position."""
    changed = total = 0
    for a, b in zip(bonds_a, bonds_b):
        for gate_a, gate_b in zip(a.kept_per_sector, b.kept_per_sector):
            if len(gate_a) == len(gate_b):
                changed += sum(max(x - y, 0) for x, y in zip(gate_a, gate_b))
            else:  # another sector structure at this gate: every slot counts as changed
                changed += sum(gate_a)
            total += sum(gate_a)
    return changed / max(total, 1)


def drift_check(plan, walkers, chi, rule="consensus", gain=0.10, subsample=128, pool=None):
    """Whether counts trained on the current population (walkers = (up, dn) stacks) beat the plan's own: trains a
    candidate plan on all walkers (trained_walker_plan, the plan's gates), compares the mean discarded weight of an
    evenly spaced subsample under both bond plans and returns (candidate or None, info). The candidate is returned
    when it lowers the mean discarded weight by more than `gain` (relative)."""
    walkers = tuple(np.asarray(w) for w in walkers)
    candidate = trained_walker_plan(plan, walkers, chi, rule, pool=pool)
    n = len(walkers[0])
    index = np.linspace(0, n - 1, min(subsample, n)).round().astype(int)
    sample = tuple(w[index] for w in walkers)
    current = float(np.mean(discarded_under(sample, plan.orbital_plans, plan.bond_plans, pool)))
    trained = float(np.mean(discarded_under(sample, candidate.orbital_plans, candidate.bond_plans, pool)))
    accept = trained < (1.0 - gain) * current
    info = dict(discarded_current=current, discarded_candidate=trained, accepted=bool(accept),
                slots_changed=slots_changed(plan.bond_plans, candidate.bond_plans))
    return (candidate if accept else None), info


def trained_walker_plan(plan, walkers, chi, rule="consensus", pool=None, weights=None):
    """A copy of the walker plan with the same reference and gates and kept counts trained on walkers (up, dn)."""
    from trot.trial.mps import make_walker_plan_from_plans

    bonds = trained_bond_plans(plan.reference, plan.orbital_plans, walkers, chi, rule, pool=pool, weights=weights)
    return make_walker_plan_from_plans(*plan.reference, plan.orbital_plans, bonds, sector_buckets=plan.sector_buckets,
                                       walker_qr=plan.walker_qr)
