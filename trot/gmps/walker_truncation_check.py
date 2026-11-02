"""How much the frozen walker plan and the walker bond lose on saved MPS-CPMC walkers (ex_6x6_doped_diag.py), and what
that does to their overlaps and local energies.

The walker plan is frozen on one reference determinant (the trial's natural orbitals): the gate sites and the
occupied/empty pattern of the adaptive orbital plan, and the number of states kept in every charge sector at every
gate (BondPlan.kept_per_sector). Per walker only the gate angles and the vectors inside each sector change. The run's
log reports the loss of the reference only; this script measures it on the propagated walkers.

--part norms (CPU, NumPy, one process per core): for every distinct walker of each saved block and each spin channel,
  with Q its orthonormal orbitals,
    plan_loss    1 - |gauge|^2 of the frozen orbital plan (driver._plan_fidelity): the weight of Q the frozen adaptive
                 gate plan cannot represent at all (0 for the reference, and for rank_exact / maximal plans);
    w_frozen     the weight the production conversion truncates: frozen gates and frozen kept counts per sector
                 (engine.channel_mps_host with the plan's bond plan; 1 - <MPS|MPS>);
    w_alloc[c]   the frozen gates with the walker's own best kept counts at bond c (utils.plan_bonds on Q);
    w_own[c]     the walker's own adaptive gate plan and kept counts at bond c (a plan made for this walker);
    fidelity     |gauge|^2 (1 - w_frozen): |<Q|MPS>|^2 / <MPS|MPS> of the production conversion.
  Reading: the frozen gate plan costs plan_loss; the frozen sectors w_frozen - w_alloc[chi_w]; the frozen gate sites
  w_alloc[c] - w_own[c]; the bond itself w_own[c] against c. Block 0 is the reference (every walker starts there):
  there plan_loss ~ 0 and w_frozen equals the log's "reference discarded weight", which the script checks.

--part allocations (CPU): fixed kept counts per charge sector trained on the walkers' own choices (--train-blocks)
  instead of frozen on the reference, scored by the weight they discard on the training walkers and on --blocks; see
  the section "allocations" below. Writes <walkers>/allocations.pkl for --part overlaps.

--part overlaps (GPU): <trial|walker> and the local energy of the same walkers (all of them, so the mean local energy
  is the population's mixed estimate: equal weights after the comb) under several conversions,
    adaptive:<c>     the production plan (frozen on the run's reference) at walker bond c;
    rank_exact:<c>   the rank_exact plan (exact for every walker before truncation, so no plan loss) at bond c;
    population:<c>   the adaptive plan frozen on the natural orbitals of the block's own walkers at bond c;
    trained:<gates>/<name>  an allocation of --part allocations, e.g. natural/reference, natural/consensus16,
                     rank_exact/vote16, rhf/union (the gate set's reference and orbital plans, the trained counts);
  and, against --reference-config, the shift of the mixed energy (paired over the same walkers), the fraction of
  overlaps with the other sign (spurious node crossings of the constraint) and the relative overlap errors.

The trial and H|trial> are rebuilt from the run's manifest.json exactly as the run built them. Output:
<walkers>/truncation_<part>.npz and .json. From this folder, one job per part:

    sbatch -p genx -C "icelake&rocky9" --cpus-per-task=16 --mem=16G --time=01:00:00 -J trunc_norms \\
        --export=ALL,TARGET=walker_truncation_check.py run_mps_sweep.sh <walkers dir> --part norms
    sbatch -p genx -C "icelake&rocky9" --cpus-per-task=16 --mem=16G --time=01:00:00 -J trunc_alloc \\
        --export=ALL,TARGET=walker_truncation_check.py run_mps_sweep.sh <walkers dir> --part allocations \\
        --train-blocks 100 140 180 220 --blocks 260 300 --gates natural rank_exact rhf population
    sbatch --gpus-per-task=h100:1 --constraint=h100-sxm5 --cpus-per-task=4 --mem=64G --time=03:00:00 \\
        -J trunc_overlaps -o "slurm/%x-%j.out" --export=ALL,TARGET=walker_truncation_check.py run_mps_gpu.sh \\
        <walkers dir> --part overlaps
"""
from trot.config import configure_once

configure_once()
import argparse
import json
import multiprocessing
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CHIS = (16, 32, 64, 128)
CONFIGS = ("adaptive:16", "adaptive:32", "adaptive:64", "rank_exact:16", "rank_exact:32", "rank_exact:64",
           "population:16")


def walker_files(folder, blocks=None):
    """{block: path} of the saved walkers, all of them or the given blocks."""
    files = {int(p.stem.removeprefix("walkers_block")): p for p in sorted(Path(folder).glob("walkers_block*.npz"))}
    if not files:
        raise SystemExit(f"no walkers_block*.npz in {folder}")
    if blocks:
        missing = sorted(set(blocks) - set(files))
        if missing:
            raise SystemExit(f"blocks {missing} are not saved in {folder} (saved: {sorted(files)})")
        files = {b: files[b] for b in blocks}
    return files


def distinct_walkers(data):
    """Indices of the distinct walkers of a saved block: after the comb walker i is the pre-comb walker comb_index[i],
    so the copies share it; the start (block 0) is one determinant repeated."""
    if "comb_index" not in data.files:
        return np.array([0])
    _, first = np.unique(np.asarray(data["comb_index"]), return_index=True)
    return np.sort(first)


# ------------------------------------------------------------------------------------------------------------------
# norms
# ------------------------------------------------------------------------------------------------------------------

def _channel_norms(C, plan, bond, chis, tolerance):
    from trot.gmps import engine, utils
    from trot.gmps.driver import _plan_fidelity

    Q = np.linalg.qr(np.asarray(C, dtype=float))[0]
    fidelity = _plan_fidelity(Q, plan)
    tensors, _, _ = engine.channel_mps_host(Q, plan, bond)
    w_frozen = 1.0 - engine.mps_overlap_host(tensors, tensors)
    own = utils.make_orbital_plan(Q, "adaptive", tolerance)
    w_alloc = [utils.plan_bonds(Q, plan, c).reference_discarded_weight for c in chis]
    w_own = [utils.plan_bonds(Q, own, c).reference_discarded_weight for c in chis]
    return 1.0 - fidelity, w_frozen, fidelity * (1.0 - w_frozen), w_alloc, w_own


def _walker_norms(job):
    up, dn, plans, bonds, chis, tolerance = job
    return [_channel_norms(C, plan, bond, chis, tolerance) for C, plan, bond in zip((up, dn), plans, bonds)]


def frozen_plans(folder, params):
    """(reference, orbital plans, bond plans, source) of the run's frozen walker plan: walker_plan.pkl when the run
    saved it (ex_6x6_doped_diag.py since 2026-10-18), else rebuilt from the saved reference, which can pick other
    near-degenerate gates than the run did (dmrg_chi256_w16: up-channel reference loss 0.166 rebuilt vs 0.156)."""
    path = Path(folder) / "walker_plan.pkl"
    if path.exists():
        with open(path, "rb") as handle:
            saved = pickle.load(handle)
        return tuple(saved["reference"]), tuple(saved["orbital_plans"]), tuple(saved["bond_plans"]), "saved"
    from trot.gmps import utils

    with np.load(Path(folder) / "walkers_block0000.npz") as data:
        references = (np.asarray(data["reference_up"]), np.asarray(data["reference_dn"]))
    plans = tuple(utils.make_orbital_plan(R, params["orbital_plan"], params["occupation_tolerance"])
                  for R in references)
    bonds = tuple(utils.plan_bonds(R, plan, params["walker_channel_chi"]) for R, plan in zip(references, plans))
    return references, plans, bonds, "rebuilt"


def run_norms(args, manifest, files, out):
    params = manifest["params"]
    chi_w, mode, tolerance = params["walker_channel_chi"], params["orbital_plan"], params["occupation_tolerance"]
    _, plans, bonds, source = frozen_plans(args.walkers, params)
    print(f"frozen plan ({source}): {mode}, chi_w {chi_w}; reference discarded weight "
          f"{bonds[0].reference_discarded_weight:.4e}, {bonds[1].reference_discarded_weight:.4e}", flush=True)
    chis = tuple(sorted(set(args.chis) | {chi_w}))
    workers = args.workers or len(os.sched_getaffinity(0))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"  # one core per process (the spawned workers read it at start)
    results, summary = {}, {}
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for block, path in files.items():
            start = time.perf_counter()
            with np.load(path) as data:
                index = distinct_walkers(data)
                if args.walkers_per_block and len(index) > args.walkers_per_block:
                    index = index[np.linspace(0, len(index) - 1, args.walkers_per_block).round().astype(int)]
                up, dn = np.asarray(data["up"])[index], np.asarray(data["dn"])[index]
            jobs = [(up[k], dn[k], plans, bonds, chis, tolerance) for k in range(len(index))]
            rows = list(pool.map(_walker_norms, jobs, chunksize=1))
            arrays = dict(
                walker=index,
                plan_loss=np.array([[r[s][0] for s in range(2)] for r in rows]),
                w_frozen=np.array([[r[s][1] for s in range(2)] for r in rows]),
                fidelity=np.array([[r[s][2] for s in range(2)] for r in rows]),
                w_alloc=np.array([[r[s][3] for s in range(2)] for r in rows]),
                w_own=np.array([[r[s][4] for s in range(2)] for r in rows]))
            results[block] = arrays
            summary[block] = summarize_norms(arrays, chis, chi_w)
            print_norms(block, summary[block], chis, len(index), time.perf_counter() - start)
    if 0 in results:  # block 0 is the reference: its losses are the plan's own
        loss, frozen = results[0]["plan_loss"][0], results[0]["w_frozen"][0]
        want = np.array([b.reference_discarded_weight for b in bonds])
        print(f"check, block 0 (the reference): plan loss {loss}, w_frozen {frozen} vs the plan's {want}", flush=True)
        # the truncation (eigh of the Gram matrix) and plan_bonds (SVD) agree to ~1e-8 at bond 16 (7263021: 2e-8)
        if np.max(np.abs(frozen - want)) > 1e-6 or np.max(loss) > 1e-6:
            print("WARNING: block 0 does not reproduce the plan's reference losses", flush=True)
    np.savez(out.with_suffix(".npz"), chis=np.asarray(chis), blocks=np.asarray(list(results)),
             **{f"b{b}_{k}": v for b, a in results.items() for k, v in a.items()})
    out.with_suffix(".json").write_text(json.dumps(dict(chis=chis, chi_w=chi_w, summary=summary), indent=2) + "\n")
    print(f"saved {out.with_suffix('.npz')} and .json", flush=True)


def summarize_norms(a, chis, chi_w):
    stat = lambda x: dict(mean=float(np.mean(x)), median=float(np.median(x)), p90=float(np.percentile(x, 90)),
                          max=float(np.max(x)))
    k = chis.index(chi_w)
    out = {}
    for s, spin in enumerate(("up", "dn")):
        out[spin] = dict(plan_loss=stat(a["plan_loss"][:, s]), w_frozen=stat(a["w_frozen"][:, s]),
                         fidelity=stat(a["fidelity"][:, s]),
                         frozen_sectors=stat(a["w_frozen"][:, s] - a["w_alloc"][:, s, k]),
                         w_alloc={str(c): stat(a["w_alloc"][:, s, j]) for j, c in enumerate(chis)},
                         w_own={str(c): stat(a["w_own"][:, s, j]) for j, c in enumerate(chis)})
    return out


def print_norms(block, summary, chis, n, seconds):
    print(f"block {block}: {n} distinct walkers ({seconds:.0f} s); mean (90th percentile) per channel", flush=True)
    for spin, s in summary.items():
        line = lambda d: f"{d['mean']:.3e} ({d['p90']:.3e})"
        print(f"  {spin}: plan loss {line(s['plan_loss'])}, w_frozen {line(s['w_frozen'])}, "
              f"fidelity {line(s['fidelity'])}, frozen sectors cost {line(s['frozen_sectors'])}", flush=True)
        print("      w_alloc " + ", ".join(f"chi {c}: {line(s['w_alloc'][str(c)])}" for c in chis), flush=True)
        print("      w_own   " + ", ".join(f"chi {c}: {line(s['w_own'][str(c)])}" for c in chis), flush=True)


# ------------------------------------------------------------------------------------------------------------------
# allocations: kept counts per charge sector trained on the walkers, on several gate sets and with several rules
# ------------------------------------------------------------------------------------------------------------------
#
# A batched engine has static shapes, so a per-walker choice of kept counts needs every sector padded to the largest
# count any walker uses there (the union), and the contractions cost the padded sizes; keeping the union counts for
# every walker then keeps everything a walker's own choice keeps, and more. So the alternatives to the counts frozen
# on the reference are fixed allocations trained on the walkers' own top-chi_w choices (trot.gmps.allocation):
#   <gates>/reference      the counts frozen on the gate set's own reference determinant (production: natural/reference);
#   <gates>/<rule><c>      a training rule (consensus, amplitude, vote) at bond c: the cost of the frozen counts at c;
#   <gates>/p<k>, /union   the k-th percentile / maximum of the walkers' own counts (bond > chi_w: what per-walker
#                          counts would cost on static shapes);
#   <gates>/own            each walker's own top-chi_w counts (the ideal per-walker choice; no fixed bond plan).
# Gate sets (--gates): natural (the run's frozen gates on the trial's natural determinant), rank_exact (gates exact
# for every walker, on the same determinant), rhf (adaptive gates on the free-electron determinant of h1), population
# (adaptive gates on the natural orbitals of the training walkers' mean density). Gates frozen on one determinant may
# lose part of another one outright (plan loss, 1 - |gauge|^2), so a walker's score is its total infidelity
# 1 - (1 - plan loss)(1 - discarded weight). Trained on --train-blocks, scored on --blocks, written to
# <walkers>/allocations.pkl for --part overlaps (configs trained:<gates>/<name>).

GATES = ("natural", "rank_exact", "rhf", "population")


def _plan_loss(job):
    """1 - |gauge|^2 of one walker channel under given gates: the part of it they cannot represent at all."""
    from trot.gmps.driver import _plan_fidelity

    C, orbital_plan = job
    return 1.0 - _plan_fidelity(np.linalg.qr(np.asarray(C, dtype=float))[0], orbital_plan)


def _sample_walkers(files, per_block):
    walkers = []
    for path in files.values():
        with np.load(path) as data:
            index = distinct_walkers(data)
            if per_block and len(index) > per_block:
                index = index[np.linspace(0, len(index) - 1, per_block).round().astype(int)]
            up, dn = np.asarray(data["up"]), np.asarray(data["dn"])
            walkers += [(up[k], dn[k]) for k in index]
    return walkers


def gate_sets(names, args, params, train):
    """{name: (reference (Ra, Rb), orbital plans)} of the requested gate sets."""
    from trot.gmps import utils
    from trot.trial.mps import natural_orbitals

    references, plans, _, source = frozen_plans(args.walkers, params)
    mode, tolerance = params["orbital_plan"], params["occupation_tolerance"]
    nelec = (train[0][0].shape[1], train[0][1].shape[1])
    out = {}
    for name in names:
        if name == "natural":
            out[name] = (references, plans)
            continue
        if name == "rank_exact":
            R, gate_mode = references, "rank_exact"
        elif name == "rhf":
            from trot.gmps.ex_6x6_doped_diag import hubbard_h1

            orbitals = np.linalg.eigh(hubbard_h1())[1]
            R, gate_mode = (orbitals[:, :nelec[0]].copy(), orbitals[:, :nelec[1]].copy()), mode
        else:  # population
            gamma = [np.mean([q @ q.T for q in (np.linalg.qr(w[s])[0] for w in train)], axis=0) for s in range(2)]
            R, gate_mode = tuple(natural_orbitals(gamma[s], nelec[s])[0] for s in range(2)), mode
        out[name] = (R, tuple(utils.make_orbital_plan(r, gate_mode, tolerance) for r in R))
    print(f"gate sets {list(out)} (natural: {source} run plan)", flush=True)
    return out


def run_allocations(args, manifest, files, out):
    from trot.gmps import allocation, utils

    params = manifest["params"]
    chi_w = params["walker_channel_chi"]
    train_files = walker_files(args.walkers, args.train_blocks)
    workers = args.workers or len(os.sched_getaffinity(0))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    train = _sample_walkers(train_files, args.walkers_per_block)
    test = _sample_walkers(files, args.walkers_per_block)
    gates = gate_sets(args.gates, args, params, train)
    print(f"chi_w {chi_w}; training on blocks {sorted(train_files)} ({len(train)} walkers), scoring on blocks "
          f"{sorted(files)} ({len(test)} walkers); rules {args.rules} at bonds {args.alloc_chis}, percentiles "
          f"{args.percentiles}", flush=True)
    allocations, scores, start = {}, {}, time.perf_counter()
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for gate_name, (references, plans) in gates.items():
            own = [allocation.allocations_of([w[s] for w in train], plans[s], chi_w, pool) for s in range(2)]
            found = {"reference": tuple(utils.plan_bonds(references[s], plans[s], chi_w) for s in range(2))}
            for rule in args.rules:
                for c in args.alloc_chis:
                    found[f"{rule}{c}"] = tuple(utils.plan_bonds(
                        references[s], plans[s], counts=allocation.train_counts(own[s], c, rule)) for s in range(2))
            for p in args.percentiles:
                found["union" if p == 100 else f"p{p:g}"] = tuple(utils.plan_bonds(
                    references[s], plans[s], counts=allocation.train_counts(own[s], rule=f"p{p:g}")) for s in range(2))
            print(f"{gate_name}: trained {len(found) - 1} allocations ({time.perf_counter() - start:.0f} s)", flush=True)
            for name, bonds in found.items():
                allocations[f"{gate_name}/{name}"] = bonds
                result = [list(pool.map(allocation.channel_loss, [(w[s], plans[s], bonds[s]) for w in test],
                                        chunksize=1))
                          for s in range(2)]
                scores[f"{gate_name}/{name}"] = np.array(result).transpose(1, 0, 2)  # (walker, spin, [plan, w])
            own_test = [allocation.allocations_of([w[s] for w in test], plans[s], chi_w, pool) for s in range(2)]
            plan_loss = [list(pool.map(_plan_loss, [(w[s], plans[s]) for w in test], chunksize=1)) for s in range(2)]
            scores[f"{gate_name}/own"] = np.array([[[plan_loss[s][k], own_test[s][k][2]] for s in range(2)]
                                                   for k in range(len(test))])
            print(f"{gate_name}: scored ({time.perf_counter() - start:.0f} s)", flush=True)

    summary = {}
    print("infidelity 1 - (1 - plan loss)(1 - discarded) per channel on the scored walkers, mean (90th percentile); "
          "plan loss mean; d=4-free channel bonds max / mean", flush=True)
    for name, s in scores.items():
        infidelity = 1.0 - (1.0 - s[..., 0]) * (1.0 - s[..., 1])
        bonds = allocations.get(name)
        sizes = None if bonds is None else [[len(q) for q in b.charges] for b in bonds]
        summary[name] = dict(
            infidelity={spin: dict(mean=float(np.mean(infidelity[:, k])), p90=float(np.percentile(infidelity[:, k], 90)))
                        for k, spin in enumerate(("up", "dn"))},
            plan_loss=[float(np.mean(s[:, k, 0])) for k in range(2)],
            max_bond=None if sizes is None else [max(x) for x in sizes],
            mean_bond=None if sizes is None else [float(np.mean(x)) for x in sizes])
        row = " ".join(f"{np.mean(infidelity[:, k]):.3e} ({np.percentile(infidelity[:, k], 90):.3e})" for k in range(2))
        plan = " ".join(f"{np.mean(s[:, k, 0]):.1e}" for k in range(2))
        bonds_text = "per walker" if sizes is None else ", ".join(f"{max(x)} / {np.mean(x):.1f}" for x in sizes)
        print(f"  {name:>24}: {row}; plan loss {plan}; bonds {bonds_text}", flush=True)
    with open(args.walkers / "allocations.pkl", "wb") as handle:
        pickle.dump(dict(gates=gates, allocations=allocations, chi_w=chi_w, train_blocks=sorted(train_files)), handle)
    out.with_suffix(".json").write_text(json.dumps(dict(chi_w=chi_w, train_blocks=sorted(train_files),
                                                        test_blocks=sorted(files), summary=summary), indent=2) + "\n")
    np.savez(out.with_suffix(".npz"), **{name.replace("/", "__"): s for name, s in scores.items()})
    print(f"saved {args.walkers / 'allocations.pkl'}, {out.with_suffix('.json')} and .npz", flush=True)


# ------------------------------------------------------------------------------------------------------------------
# overlaps and local energies
# ------------------------------------------------------------------------------------------------------------------

def run_overlaps(args, manifest, files, out):
    import jax
    import jax.numpy as jnp

    from trot.gmps import engine
    from trot.gmps.ex_6x6_doped_diag import U, hubbard_h1, load_trial
    from trot.ham.hubbard import HamHubbard
    from trot.meas.mps import build_mps_meas_ctx
    from trot.prop import mps_cpmc
    from trot.trial.mps import make_walker_plan_from_plans, make_walker_plan_from_reference, natural_orbitals

    params = manifest["params"]
    trained = None
    if any(c.startswith("trained:") for c in args.configs):
        with open(args.walkers / "allocations.pkl", "rb") as handle:  # --part allocations
            trained = pickle.load(handle)
        missing = [c for c in args.configs if c.startswith("trained:") and c[8:] not in trained["allocations"]]
        if missing:
            raise SystemExit(f"{missing} not in allocations.pkl ({sorted(trained['allocations'])})")
    h1 = hubbard_h1()
    trial, htrial, nelec = load_trial(manifest["trial_file"], float(manifest["rotation_degrees"]),
                                      float(manifest["htrial_cutoff"]))
    ham = HamHubbard(jnp.asarray(h1), U)
    with np.load(Path(args.walkers) / "walkers_block0000.npz") as data:
        references = (np.asarray(data["reference_up"]), np.asarray(data["reference_dn"]))
    blocks = {}
    for block, path in files.items():
        with np.load(path) as data:
            blocks[block] = dict(up=np.asarray(data["up"]), dn=np.asarray(data["dn"]),
                                 energy=float(data["energy"]) if "energy" in data.files else float("nan"))
    print(f"trial {manifest['trial_file']} rotated {manifest['rotation_degrees']} deg; blocks {sorted(blocks)}; "
          f"configs {args.configs}, reference config {args.reference_config}", flush=True)

    values = {}  # (config, block) -> (overlaps, energies)
    for config in args.configs:
        mode, chi = config.split(":")
        ctx = None
        for block, walkers in blocks.items():
            start = time.perf_counter()
            if mode == "trained" and ctx is None:  # the orbital plans and a bond plan of --part allocations
                # the gate set's own reference, under another name: `references` (the run's) builds the other configs
                # (7279383/4 overwrote it, so their rank_exact:64 was frozen on the rhf / population determinant)
                gate_references, orbital_plans = trained["gates"][chi.split("/")[0]]  # chi is "<gates>/<name>" here
                plan = make_walker_plan_from_plans(*gate_references, orbital_plans, trained["allocations"][chi],
                                                   sector_buckets=tuple(params["sector_buckets"]), walker_qr="auto")
                ctx = build_mps_meas_ctx(ham, trial, plan=plan, kernel="blocked", htrial=htrial)
            elif mode == "population" or ctx is None:  # one plan per config; "population" one per block
                chi = int(chi)
                if mode == "population":  # frozen on the natural orbitals of this block's walkers
                    Q = [np.linalg.qr(walkers[s])[0] for s in ("up", "dn")]
                    gamma = [np.mean(q @ np.swapaxes(q, 1, 2), axis=0) for q in Q]
                    Ra, Rb = (natural_orbitals(gamma[s], nelec[s])[0] for s in range(2))
                    plan_mode = "adaptive"
                else:  # frozen on the run's reference
                    (Ra, Rb), plan_mode = references, mode
                plan = make_walker_plan_from_reference(
                    Ra, Rb, orbital_plan=plan_mode, occupation_tolerance=params["occupation_tolerance"],
                    walker_channel_chi=chi, sector_buckets=tuple(params["sector_buckets"]), walker_qr="auto")
                ctx = build_mps_meas_ctx(ham, trial, plan=plan, kernel="blocked", htrial=htrial)
            n = len(walkers["up"])
            overlaps, energies = mps_cpmc._start_values(ctx)(jnp.asarray(walkers["up"]), jnp.asarray(walkers["dn"]),
                                                             ctx, n_chunks=engine.measure_chunks(n, 1))
            values[config, block] = (np.asarray(overlaps), np.asarray(energies))
            print(f"{config:>14} block {block:4d}: mean E_L {np.mean(values[config, block][1]):.6f} "
                  f"(the run's block energy {walkers['energy']:.6f}); {time.perf_counter() - start:.0f} s", flush=True)
        del ctx, plan  # this config's device blocks and kernels
        jax.clear_caches()

    summary = {}
    for block in blocks:
        ref_ov, ref_e = values[args.reference_config, block]
        rows = {}
        for config in args.configs:
            ov, e = values[config, block]
            de = e - ref_e
            rel = np.abs(ov - ref_ov) / np.abs(ref_ov)
            rows[config] = dict(mixed_energy=float(np.mean(e)), mixed_energy_err=float(np.std(e) / np.sqrt(len(e))),
                                shift=float(np.mean(de)), shift_err=float(np.std(de) / np.sqrt(len(de))),
                                sign_flips=float(np.mean(np.sign(ov) != np.sign(ref_ov))),
                                relative_overlap_error=dict(median=float(np.median(rel)),
                                                            p90=float(np.percentile(rel, 90)),
                                                            max=float(np.max(rel))),
                                outside_window=float(np.mean(np.abs(e - np.mean(ref_e)) > np.sqrt(2.0 / params["dt"]))))
        summary[block] = rows
        print(f"block {block} against {args.reference_config}:", flush=True)
        print(f"  {'config':>14} {'mixed E':>12} {'shift':>18} {'sign flips':>10} {'|dov|/|ov| median':>18} "
              f"{'p90':>10}", flush=True)
        for config, r in rows.items():
            print(f"  {config:>14} {r['mixed_energy']:12.6f} {r['shift']:+10.6f}({r['shift_err']:.1e}) "
                  f"{r['sign_flips']:10.4f} {r['relative_overlap_error']['median']:18.3e} "
                  f"{r['relative_overlap_error']['p90']:10.3e}", flush=True)
    np.savez(out.with_suffix(".npz"), configs=np.asarray(args.configs), blocks=np.asarray(list(blocks)),
             **{f"{c.replace(':', '_').replace('/', '__')}_b{b}_overlaps": v[0] for (c, b), v in values.items()},
             **{f"{c.replace(':', '_').replace('/', '__')}_b{b}_energies": v[1] for (c, b), v in values.items()})
    out.with_suffix(".json").write_text(json.dumps(dict(reference_config=args.reference_config,
                                                        summary={str(b): r for b, r in summary.items()}),
                                                   indent=2) + "\n")
    print(f"saved {out.with_suffix('.npz')} and .json", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("walkers", type=Path, help="the output folder of ex_6x6_doped_diag.py")
    parser.add_argument("--part", required=True, choices=["norms", "allocations", "overlaps"])
    parser.add_argument("--blocks", type=int, nargs="*", default=None,
                        help="saved blocks to use (norms: all saved; allocations: the blocks scored; overlaps: the "
                             "last three)")
    parser.add_argument("--chis", type=int, nargs="*", default=list(CHIS), help="norms: walker bonds to scan")
    parser.add_argument("--train-blocks", type=int, nargs="*", default=None,
                        help="allocations: the blocks the counts are trained on")
    parser.add_argument("--alloc-chis", type=int, nargs="*", default=[16, 24],
                        help="allocations: bonds of the trained allocations (default 16 24)")
    parser.add_argument("--gates", nargs="*", default=["natural"], choices=list(GATES),
                        help="allocations: gate sets to train on (default natural, the run's)")
    parser.add_argument("--rules", nargs="*", default=["consensus", "amplitude", "vote"],
                        help="allocations: training rules at the --alloc-chis bonds (trot.gmps.allocation.RULES)")
    parser.add_argument("--percentiles", type=float, nargs="*", default=[90, 100],
                        help="allocations: percentiles of the walkers' own counts (100: the union)")
    parser.add_argument("--walkers-per-block", type=int, default=64,
                        help="norms, allocations: distinct walkers per block, evenly spaced (0: all)")
    parser.add_argument("--workers", type=int, default=0,
                        help="norms, allocations: processes (default: the job's cores)")
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS),
                        help="overlaps: <plan>:<walker bond>, or trained:<gates>/<name> of --part allocations")
    parser.add_argument("--reference-config", default="rank_exact:64", help="overlaps: the most accurate config")
    args = parser.parse_args(argv)

    manifest = json.loads((args.walkers / "manifest.json").read_text())
    blocks = args.blocks
    if args.part == "overlaps":
        if args.reference_config not in args.configs:
            parser.error("--reference-config must be one of --configs")
        if not blocks:
            saved = sorted(walker_files(args.walkers))
            blocks = [b for b in saved if b][-3:]
    if args.part == "allocations" and not (args.train_blocks and blocks):
        parser.error("--part allocations needs --train-blocks and --blocks (the scored blocks)")
    files = walker_files(args.walkers, blocks)
    out = args.walkers / f"truncation_{args.part}"
    print(f"{args.walkers}: {manifest['tag']}, blocks {sorted(files)}", flush=True)
    dict(norms=run_norms, allocations=run_allocations, overlaps=run_overlaps)[args.part](args, manifest, files, out)


if __name__ == "__main__":
    main()
