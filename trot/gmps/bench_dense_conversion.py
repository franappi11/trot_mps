"""Benchmark of the dense-bond conversion (trot.gmps.dense_conversion) against the engine's frozen-count conversion.

For the walkers of one saved population, on the device and with the gates of the walker plan (an orbital plan per
spin channel, both conversions use the same angles), the script reports

  frozen   engine.converter_for(plan): the kept counts frozen on the plan's reference (what the production runs
           use). Compile and run time of the conversion of every walker (both spins), and the mean discarded weight
           of the walkers under those counts (host oracle, on the checked walkers).
  dense    for every --chi and every --moves method: the same conversion with dense bonds of size chi, each walker
           keeping its own chi largest Schmidt states. Compile and run time (alpha, beta, total, and the ratio to
           frozen), the mean discarded weight over all walkers, and against the host oracle for the checked walkers (the
           same truncation done by trot.gmps.utils.plan_bonds(Q, plan, chi) + engine.channel_mps_host, the "own
           counts"): the largest difference of the discarded weights and the largest relative error of
           g <host|dense> against g <host|host>. The charge check of the alpha channel: the largest impurity of the
           weighted kept states over all factorisations and walkers, and the share of the final tensors' weight that
           breaks charge conservation under the carried labels (both ~1e-12 or less when the labels are usable).

Label diagnostics (summarize_labels): where the sector blocks' exact zeros are first lost (the "seed": an off-sector
entry in a new isometry), how many factorisations and walkers carry one, how heavy the mislabeled states get, and how
much of a factored matrix lies outside its carried blocks. --variants chi:move:eigh:snap|nosnap compares eigensolvers
(XLA's default or Jacobi forced) and snapping (dense_conversion.make_dense_converter); --sign-check B contracts the
checked walkers with a random d=4 MPS of bond B and compares with the own-counts oracle, whose labels are exact: wrong
signs on weighted states show there, truncation near-ties only at the level of the state errors.

The walkers are an .npz file: up / dn (the run folders of ex_6x6_doped_diag.py, walkers_block*.npz) or walkers_0 /
walkers_1 (PropState dumps, e.g. the free-electron runs uhf_6x6_doped.py, uhf_8x8_doped.py). The gates come from the
run's walker_plan.pkl (--plan; its bond plans give the frozen baseline) or, without it, from the free-fermion
determinant of the open --side x --side square lattice, with counts frozen on it at --chi[0] (the natural reference of
a trial is not at hand for the 8x8).

    # 6x6: the walkers of a doped DMRG run, with that run's frozen plan (chi_w 16)
    ./submit_ex_2d.sh h100 bench_dense_conversion.py /mnt/ceph/users/fnappi/trot_walkers/sq6x6_doped_diag/dmrg_chi256_w16_m64/walkers_block0100.npz \\
        --plan /mnt/ceph/users/fnappi/trot_walkers/sq6x6_doped_diag/dmrg_chi256_w16_m64/walker_plan.pkl --chi 16 32
    # 8x8: walkers of the free-electron CPMC run, plan frozen on the free-fermion determinant
    ./submit_ex_2d.sh h100 bench_dense_conversion.py /mnt/ceph/users/fnappi/trot_walkers/uhf_8x8_doped_U8/walkers_00340.npz \\
        --side 8 --chi 16 32
"""
from trot.config import configure_once

configure_once()  # float64

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from trot.gmps import allocation, dense_engine, engine, utils
from trot.gmps import dense_conversion as dc
from trot.trial.mps import make_walker_plan_from_plans

# a rerun with the same shapes skips the compiles (the conversion circuits are slow to compile)
jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/trot_jax_compile"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)


def square_h1(side, t=1.0):
    """Open side x side square lattice, site y * side + x (as ex_6x6_doped_diag.hubbard_h1)."""
    n = side * side
    h1 = np.zeros((n, n))
    for y in range(side):
        for x in range(side):
            site = y * side + x
            if x + 1 < side:
                h1[site, site + 1] = h1[site + 1, site] = -t
            if y + 1 < side:
                h1[site, site + side] = h1[site + side, site] = -t
    return h1


def load_walkers(path, n):
    """(up, dn, distinct) of the first n walkers of a saved population; distinct are the indices of walkers that are
    not copies of each other (the comb of a run duplicates walkers)."""
    with np.load(path) as data:
        if "up" in data.files:
            up, dn = np.asarray(data["up"]), np.asarray(data["dn"])
            if "comb_index" in data.files:
                _, first = np.unique(np.asarray(data["comb_index"]), return_index=True)
                distinct = np.sort(first)
            else:
                distinct = np.arange(len(up))
        elif "walkers_0" in data.files:
            up, dn = np.asarray(data["walkers_0"]), np.asarray(data["walkers_1"])
            distinct = np.arange(len(up))
        else:
            raise SystemExit(f"{path}: expected up/dn or walkers_0/walkers_1, found {data.files}")
    n = min(n, len(up))
    return up[:n], dn[:n], distinct[distinct < n]


def build_plans(args, up):
    """(reference, orbital plans, bond plans, source) of the frozen walker plan."""
    if args.plan:
        with open(args.plan, "rb") as handle:
            saved = pickle.load(handle)
        return tuple(saved["reference"]), tuple(saved["orbital_plans"]), tuple(saved["bond_plans"]), "saved plan"
    if not args.side:
        raise SystemExit("give --plan (a run's walker_plan.pkl) or --side (the free-fermion reference)")
    n_sites, n_elec = up.shape[1], up.shape[2]
    if n_sites != args.side ** 2:
        raise SystemExit(f"--side {args.side} has {args.side ** 2} sites, the walkers {n_sites}")
    _, orbitals = np.linalg.eigh(square_h1(args.side))
    reference = (orbitals[:, :n_elec].copy(), orbitals[:, :n_elec].copy())
    plans = tuple(utils.make_orbital_plan(R, args.orbital_plan, args.tolerance) for R in reference)
    bonds = tuple(utils.plan_bonds(R, p, args.chi[0]) for R, p in zip(reference, plans))
    return reference, plans, bonds, f"free-fermion reference, {args.orbital_plan} plan"


def compile_and_time(fn, args, repeats):
    """(compiled, compile seconds, median run seconds, best run seconds) of jax.jit(fn)."""
    start = time.perf_counter()
    compiled = jax.jit(fn).lower(*args).compile()
    compile_seconds = time.perf_counter() - start
    jax.block_until_ready(compiled(*args))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        times.append(time.perf_counter() - start)
    return compiled, compile_seconds, float(np.median(times)), float(np.min(times))


def host_reference(Q, plan, chi):
    """The own-counts oracle: (tensors, gauge, discarded weight, exact integer bond labels) of one channel truncated
    by plan_bonds at chi."""
    bond = utils.plan_bonds(Q, plan, chi)
    tensors, charges, gauge = engine.channel_mps_host(Q, plan, bond)
    return tensors, float(gauge), float(bond.reference_discarded_weight), charges


def overlap_error(host, dense_tensors, gauge):
    """|g <host|dense> - g_host <host|host>| / |g_host <host|host>|, as engine.conversion_self_check."""
    tensors, g_host = host[0], host[1]
    reference = g_host * engine.mps_overlap_host(tensors, tensors)
    return abs(gauge * engine.mps_overlap_host(tensors, dense_tensors) - reference) / abs(reference)


def random_trial(n_sites, bond, seed=7):
    """A random real d=4 MPS of bond `bond` with no particle-number structure: against a walker it weighs every
    component with an arbitrary amplitude, so a wrong fermionic sign on any weighted state shows in the overlap."""
    rng = np.random.default_rng(seed)
    dims = [1] + [bond] * (n_sites - 1) + [1]
    return [rng.normal(size=(dims[i], 4, dims[i + 1])) for i in range(n_sites)]


def contract_with(blocks, alpha, beta, beta_charges):
    """<trial|walker channels> by dense_engine's contraction, the fermionic sign taken from the beta bond labels."""
    dressed = [dense_engine._dressed(jnp.asarray(B), dense_engine._parity_signs(jnp.asarray(q, dtype=float)))
               for B, q in zip(beta, beta_charges[:-1])]
    return float(dense_engine.left_contract([jnp.asarray(A) for A in alpha], dressed, blocks))


def sign_errors(blocks, trial_norm, hosts, dense):
    """End-to-end check of the labels on the checked walkers: |g <T|dense walker> - g <T|own-counts walker>| over
    sqrt(<T|T>) (a bound on |<T|walker>|) and over |<T|own-counts walker>|. hosts: per walker ((alpha host), (beta host))
    host_reference tuples; dense: per walker (alpha tensors, beta tensors, beta labels, gauge product). A wrong sign on a
    state of weight w changes the overlap by about sqrt(w) of the scale; the truncation differences (near-ties) are of
    the size of the state errors."""
    scaled, relative = [], []
    for (ha, hb), (alpha, beta, qb, gauge) in zip(hosts, dense):
        reference = ha[1] * hb[1] * contract_with(blocks, ha[0], hb[0], hb[3])
        value = gauge * contract_with(blocks, alpha, beta, qb)
        scaled.append(abs(value - reference) / trial_norm)
        relative.append(abs(value - reference) / max(abs(reference), 1.0e-300))
    return max(scaled), max(relative)


def parse_variant(text):
    """'chi:move:eigh:snap' (snap: snap / nosnap), e.g. '32:cans:jacobi:snap'."""
    chi, move, eigh, snap = text.split(":")
    if move not in dc.MOVES or eigh not in dc.EIGH or snap not in ("snap", "nosnap"):
        raise SystemExit(f"bad variant {text!r}: chi:{'|'.join(dc.MOVES)}:{'|'.join(dc.EIGH)}:snap|nosnap")
    return int(chi), move, eigh, snap == "snap"


def summarize_labels(stats):
    """The label diagnostics of a batch of conversions (dense_conversion.make_dense_converter's stats, arrays of shape
    (walkers, factorisations) plus the per-walker leakage) as plain numbers:
      seeded_pairs / seeded_walkers  share of (walker, factorisation) pairs / of walkers with an off-sector entry (offlabel
                                     > 0): exactly 0 while the sector blocks' zeros survive every step;
      first_seed                     the first factorisation index with one, over the walkers (-1: none);
      offlabel_max                   the largest off-sector entry of a new isometry;
      mislabeled_pairs / _weight     pairs with a kept state of impurity > MISLABEL, and the largest weight of such a state;
      input_leak_max                 the largest share of a factored matrix outside its carried sector blocks;
      impurity_max, leakage_max      as before."""
    off, weight, leak = stats["offlabel"], stats["mislabeled_weight"], stats["input_leak"]
    seeded = off > 0.0
    first = np.where(seeded.any(axis=1), seeded.argmax(axis=1), -1)
    return dict(seeded_pairs=float(seeded.mean()), seeded_walkers=float(seeded.any(axis=1).mean()),
                first_seed=int(first[first >= 0].min()) if (first >= 0).any() else -1,
                offlabel_max=float(off.max()), mislabeled_pairs=float((weight > 0.0).mean()),
                mislabeled_weight_max=float(weight.max()), input_leak_max=float(leak.max()),
                impurity_max=float(stats["impurity"].max()), leakage_max=float(np.max(stats["leakage"])))


def format_labels(s):
    return (f"seeded {100 * s['seeded_pairs']:.3f}% of pairs ({100 * s['seeded_walkers']:.1f}% of walkers, first at op "
            f"{s['first_seed']}), offlabel max {s['offlabel_max']:.1e}; mislabeled (impurity > {dc.MISLABEL:g}) "
            f"{100 * s['mislabeled_pairs']:.3f}% of pairs, heaviest {s['mislabeled_weight_max']:.2e}; input leak max "
            f"{s['input_leak_max']:.1e}; impurity max {s['impurity_max']:.1e}; leakage max {s['leakage_max']:.1e}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("walkers", type=Path, help="walkers .npz (up/dn or walkers_0/walkers_1)")
    parser.add_argument("--plan", type=Path, default=None, help="walker_plan.pkl of a run (gates + frozen counts)")
    parser.add_argument("--side", type=int, default=0, help="without --plan: the free-fermion reference of the side x side lattice")
    parser.add_argument("--orbital-plan", default="adaptive", choices=["adaptive", "rank_exact", "maximal"])
    parser.add_argument("--tolerance", type=float, default=1.0e-10, help="occupation tolerance of the adaptive plan")
    parser.add_argument("--chi", type=int, nargs="+", default=[16], help="dense bond sizes (and, without --plan, the frozen counts' bond)")
    parser.add_argument("--moves", nargs="+", default=list(dc.MOVES), choices=list(dc.MOVES))
    parser.add_argument("--n-walkers", type=int, default=400)
    parser.add_argument("--check", type=int, default=6, help="distinct walkers per spin checked against the host oracle")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sector-buckets", type=int, nargs="*", default=[8, 16], help="size buckets of the frozen conversion")
    parser.add_argument("--cans-lo", type=float, default=1.0e-9, help="smallest relative singular value the cans polar factor resolves")
    parser.add_argument("--gram-floor", type=float, default=1.0e-10, help="relative singular value floor of the Gram-based factors")
    parser.add_argument("--charge-gap", type=float, default=1.0e-7,
                        help="per-charge shift of the gate's eigh in units of trace(G); above 1 the sectors' eigenvalue bands "
                             "cannot overlap (needed where the eigh is not the Jacobi one, matrices above 32 x 32)")
    parser.add_argument("--eigh", default="default", choices=list(dc.EIGH),
                        help="the dense conversion's eigh: XLA's choice, or Jacobi forced at every size (GPU only)")
    parser.add_argument("--snap", action="store_true",
                        help="snap every new isometry to its sectors (dense_conversion.make_dense_converter)")
    parser.add_argument("--variants", nargs="*", default=[],
                        help="dense variants chi:move:eigh:snap|nosnap (e.g. 32:cans:jacobi:snap) instead of --chi x --moves "
                             "with --eigh / --snap")
    parser.add_argument("--sign-check", type=int, default=0,
                        help="trial bond of the end-to-end label check on the checked walkers (0: off): the dense walkers' "
                             "overlap with a random d=4 MPS against the own-counts oracle's, with its exact labels")
    parser.add_argument("--unroll", type=int, default=1, help="unroll of the cans iteration (0: all steps)")
    parser.add_argument("--no-frozen", action="store_true", help="skip the frozen-count baseline")
    parser.add_argument("--out", type=Path, default=None, help="write the rows as JSON")
    args = parser.parse_args(argv)

    up, dn, distinct = load_walkers(args.walkers, args.n_walkers)
    n, n_sites, n_elec = up.shape
    Qa, Qb = np.linalg.qr(up)[0], np.linalg.qr(dn)[0]
    reference, orbital_plans, bond_plans, source = build_plans(args, up)
    chi_plan = bond_plans[0].max_bond
    check = distinct[np.linspace(0, len(distinct) - 1, min(args.check, len(distinct))).round().astype(int)]
    device = jax.devices()[0]
    print(f"{args.walkers}: {n} walkers ({len(distinct)} distinct), {n_sites} sites, {n_elec} electrons per spin, "
          f"device {device.device_kind}")
    print(f"gates ({source}): {[len(dc.dense_ops(p)) for p in orbital_plans]} factorisations "
          f"({[len(engine.gate_sites(p)) for p in orbital_plans]} gates); frozen counts at bond {chi_plan}; "
          f"checked walkers: {list(map(int, check))}")
    qa, qb = jnp.asarray(Qa), jnp.asarray(Qb)
    rows = []

    variants = ([parse_variant(v) for v in args.variants] if args.variants else
                [(chi, move, args.eigh, args.snap) for chi in args.chi for move in args.moves])
    chis = sorted({v[0] for v in variants})

    # ---- host oracles: own counts at every chi, the frozen counts' discarded weight ----------------------------
    start = time.perf_counter()
    own = {chi: [[host_reference(Q[i], p, chi) for i in check] for Q, p in zip((Qa, Qb), orbital_plans)]
           for chi in chis}
    print(f"host oracle (own counts, {len(check)} walkers x 2 spins x {len(chis)} bonds): "
          f"{time.perf_counter() - start:.0f} s")
    if args.sign_check:
        trial = random_trial(n_sites, args.sign_check)
        blocks = engine.dense_trial_blocks(trial, xp=np, device=True)
        trial_norm = float(np.sqrt(engine.mps_overlap_host(trial, trial)))
        print(f"sign check: random d=4 trial of bond {args.sign_check} against the own-counts oracle's exact labels")
    for chi in chis:
        discarded = np.mean([h[2] for spin in own[chi] for h in spin])
        print(f"  own counts, bond {chi}: mean discarded weight {discarded:.4e}")

    # ---- frozen counts ------------------------------------------------------------------------------------------
    frozen_seconds = None
    if not args.no_frozen:
        plan = make_walker_plan_from_plans(*reference, orbital_plans, bond_plans, sector_buckets=args.sector_buckets,
                                           walker_qr="auto")
        convert = engine.converter_for(plan).convert
        _, compile_seconds, median, best = compile_and_time(lambda a, b: jax.vmap(convert)(a, b), (qa, qb),
                                                            args.repeats)
        frozen_seconds = median
        lost = allocation.discarded_under((Qa[check], Qb[check]), orbital_plans, bond_plans).mean()
        print(f"frozen   bond {chi_plan} (counts on the reference, buckets {args.sector_buckets}): "
              f"compile {compile_seconds:6.1f} s  run {1e3 * median:8.1f} ms (best {1e3 * best:.1f})  "
              f"discarded {lost:.4e}")
        rows.append(dict(kind="frozen", chi=chi_plan, compile_s=compile_seconds, run_ms=1e3 * median,
                         best_ms=1e3 * best, discarded=float(lost)))

    # ---- dense bonds -------------------------------------------------------------------------------------------
    norm2 = jax.jit(jax.vmap(dc.mps_norm2))
    for chi, move, eigh, snap in variants:
        unroll = True if args.unroll == 0 else args.unroll
        converters = [dc.make_dense_converter(p, chi, move, cans_lo=args.cans_lo, gram_floor=args.gram_floor,
                                              charge_gap=args.charge_gap, unroll=unroll, eigh=eigh, snap=snap)
                      for p in orbital_plans]
        runs, compile_total, discarded, errors, diffs, outputs = [], 0.0, [], [], [], []
        for spin, (Q, conv) in enumerate(zip((qa, qb), converters)):
            compiled, compile_seconds, median, _ = compile_and_time(jax.vmap(conv.convert), (Q,), args.repeats)
            tensors, charges, gauge = compiled(Q)
            outputs.append((tensors, charges, gauge))
            lost = 1.0 - np.asarray(norm2(tensors))
            discarded.append(lost)
            compile_total += compile_seconds
            runs.append(median)
            for j, i in enumerate(check):
                host = own[chi][spin][j]
                dense = [np.asarray(A[i]) for A in tensors]
                errors.append(overlap_error(host, dense, float(gauge[i])))
                diffs.append(abs(lost[i] - host[2]))
        total = sum(runs)
        # the charge diagnostics on the alpha channel (a property of the algorithm, so one spin is enough)
        stats = jax.jit(jax.vmap(lambda C, f=converters[0].convert: f(C, stats=True)))
        s = {k: np.asarray(v) for k, v in stats(qa)[3].items()}
        label = summarize_labels(s)
        sizes = sorted({min(shape) for (kind, *_), shape in zip(converters[0].ops, converters[0].shapes)
                        if kind == "gate"})
        ratio = "" if frozen_seconds is None else f"  x{total / frozen_seconds:5.2f} of frozen"
        steps = f", {converters[0].polar_steps} polar steps" if move == "cans" else ""
        print(f"dense {move:<5} bond {chi:3d} eigh {eigh} snap {snap}: compile {compile_total:6.1f} s  run "
              f"{1e3 * total:8.1f} ms (alpha {1e3 * runs[0]:.1f}, beta {1e3 * runs[1]:.1f}){ratio}{steps}; gate eigh "
              f"sizes up to {max(sizes)}")
        print(f"    discarded {np.mean(np.concatenate(discarded)):.4e} (all walkers)  vs host own counts "
              f"max |diff| {max(diffs):.2e}  overlap error max {max(errors):.2e}")
        if label["first_seed"] >= 0:
            kind, site, step, _ = converters[0].ops[label["first_seed"]]
            label["first_seed_op"] = f"{kind} site {site} step {step} matrix {converters[0].shapes[label['first_seed']]}"
        print(f"    charges (alpha; gap {args.charge_gap:g}): " + format_labels(label)
              + (f"; first seeded op: {label['first_seed_op']}" if "first_seed_op" in label else ""))
        if args.sign_check:
            hosts = [(own[chi][0][j], own[chi][1][j]) for j in range(len(check))]
            (ta, _, ga), (tb, qb_dense, gb) = outputs
            dense = [([np.asarray(A[i]) for A in ta], [np.asarray(B[i]) for B in tb],
                      [np.asarray(q[i]) for q in qb_dense], float(ga[i] * gb[i])) for i in check]
            scaled, relative = sign_errors(blocks, trial_norm, hosts, dense)
            label.update(sign_error_scaled=scaled, sign_error_relative=relative)
            print(f"    sign check: |dense - oracle| max {scaled:.2e} of sqrt(<T|T>), {relative:.2e} relative")
        rows.append(dict(kind="dense", move=move, chi=chi, charge_gap=args.charge_gap, gram_floor=args.gram_floor,
                         eigh=eigh, snap=snap, compile_s=compile_total, run_ms=1e3 * total,
                         alpha_ms=1e3 * runs[0], beta_ms=1e3 * runs[1], discarded=float(np.mean(np.concatenate(discarded))),
                         discarded_diff_max=float(max(diffs)), overlap_error_max=float(max(errors)),
                         polar_steps=converters[0].polar_steps, **label))
    if args.out:
        args.out.write_text(json.dumps(dict(walkers=str(args.walkers), n=n, sites=n_sites, electrons=n_elec,
                                            device=device.device_kind, rows=rows), indent=1))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
