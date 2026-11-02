"""Compress a saved charge-labelled trial, ready for production's rotate-as-is loader.

Example (from the repository root)::

    python -m trot.gmps.compress_saved_trial \
        --source /Users/fnappi/trot_trials/uhf_dmrg_6x6_U8/dmrg_chi2048.npz \
        --baseline /Users/fnappi/trot_trials/uhf_dmrg_6x6_U8/dmrg_chi1024.npz \
        --chis 1024 --out diagnostics/trial_transfer_6x6/compressed

Archives contain UNROTATED tensors and their freshly computed RDM and energy.
Use ex_6x6_doped_diag.py --trial-file <archive> --rotation 90 --mpo --no-szsz.
That loader rotates the physical legs and uses the state as-is, without projection.
The stored unrotated RDM sets its natural reference, matching production convention.
Compression preserves N_up,N_down, but is not SU(2)-adapted: its spin content can
change. The subsequent unitary spin rotation preserves that compressed state's
total-spin content. No CPMC accuracy claim follows from compression fidelity.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np


def block_observables(blocks, *, say=lambda message: None):
    """Exact real, SZ-labelled MPS one-RDM, double occupations, and <S^2>.

    Charge-block environments avoid dense bond-cubed contractions. The interleaved
    local basis is |0>, |up>, |dn>, |up dn>; all returned observables are normalized.
    """
    sites = []
    for site in blocks:
        by_left = {}
        for (ql, p, qr), a in site.items():
            by_left.setdefault(ql, {})[p] = (qr, a)
        sites.append(by_left)

    def advance(env, i, operator):
        entries = [(int(p), int(q), operator[p, q]) for p, q in zip(*np.nonzero(operator))]
        result = {}
        for (lb, lk), e in env.items():
            bra, ket = sites[i].get(lb, {}), sites[i].get(lk, {})
            for p, q, coefficient in entries:
                if p not in bra or q not in ket:
                    continue
                rb, a = bra[p]
                rk, b = ket[q]
                value = coefficient * (a.T @ e @ b)
                key = (rb, rk)
                result[key] = result[key] + value if key in result else value
        return result

    identity = np.eye(4)
    zero = next(iter(sites[0]))
    left = [{(zero, zero): np.ones((1, 1))}]
    for i in range(len(sites)):
        left.append(advance(left[-1], i, identity))
    norm = float(sum(x.item() for x in left[-1].values()))
    if norm <= 0:
        raise ValueError("Non-positive trial norm")
    last = next(iter(blocks[-1]))[2]
    right = [None] * (len(sites) + 1)
    right[-1] = {last: np.ones((1, 1))}
    for i in reversed(range(len(sites))):
        result = {}
        for (ql, _, qr), a in blocks[i].items():
            if qr in right[i+1]:
                term = a @ right[i+1][qr] @ a.T
                result[ql] = result[ql] + term if ql in result else term
        right[i] = result

    def close(env, boundary):
        return sum(float(np.sum(e * right[boundary][qb]))
                   for (qb, qk), e in env.items() if qb == qk and qb in right[boundary]) / norm

    def local(operator):
        return np.array([close(advance(left[i], i, operator), i+1) for i in range(len(sites))])

    ca, cb = np.zeros((4, 4)), np.zeros((4, 4))
    ca[1, 0] = ca[3, 2] = 1
    cb[2, 0], cb[3, 1] = 1, -1
    pa, pb = np.diag([1., -1., 1., -1.]), np.diag([1., 1., -1., -1.])
    rdm = np.zeros((2, len(sites), len(sites)))
    for spin, (opening, closing, number) in enumerate((
        (ca @ pb, ca.T, np.diag([0., 1., 0., 1.])),
        (pa @ cb, cb.T, np.diag([0., 0., 1., 1.])),
    )):
        np.fill_diagonal(rdm[spin], local(number))
        for i in range(len(sites)):
            env = advance(left[i], i, opening)
            for j in range(i+1, len(sites)):
                rdm[spin, i, j] = rdm[spin, j, i] = close(advance(env, j, closing), j+1)
                env = advance(env, j, pa @ pb)
            if i % 6 == 0 or i == len(sites)-1:
                say(f"RDM spin {spin}, row {i+1}/{len(sites)}")
    double = local(np.diag([0., 0., 0., 1.]))
    plus = np.zeros((4, 4))
    plus[1, 2] = 1.
    # S^2 = Sz^2 - Sz + S+ S-; the input has definite Sz.
    sz = float((np.trace(rdm[0])-np.trace(rdm[1])) / 2)
    s2 = sz*sz - sz + float(np.trace(rdm[0])-double.sum())
    for i in range(len(sites)):
        env = advance(left[i], i, plus)
        for j in range(i+1, len(sites)):
            s2 += 2 * close(advance(env, j, plus.T), j+1)
            env = advance(env, j, identity)
    return dict(norm=norm, rdm1=rdm, double_occupancy=double, s2=float(s2))


def load_trial(path):
    with np.load(path, allow_pickle=False) as z:
        n = len(z["h1"])
        arrays = {key: z[key].copy() for key in ("h1", "u", "nelec", "energy", "rdm1", "chi")}
        tensors = [z[f"A{i}"] for i in range(n)]
        charges = tuple(z[f"q{i}"] for i in range(n+1))
    if any(q.shape[1] != 2 for q in charges) or not np.array_equal(charges[-1], [arrays["nelec"]]):
        raise ValueError("Expected an unrotated trial with (N_up,N_down) bond labels")
    return tensors, charges, arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--chis", nargs="+", type=int, default=[1024])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if any(chi <= 0 for chi in args.chis):
        parser.error("All bonds must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    destinations = [args.out / f"{args.source.stem}_to_chi{chi}.npz" for chi in args.chis]
    if any(path.exists() for path in destinations):
        raise FileExistsError("A destination trial already exists; choose a new output directory")

    from trot.gmps import trials
    from trot.trial.mps import compress_mps_qn, spin_rotation_unitary, spin_rotation_y

    start = time.monotonic()
    def say(message):
        print(f"[{time.monotonic()-start:7.1f}s] {message}", flush=True)

    say(f"Loading {args.source}")
    source, sq, meta = load_trial(args.source)
    if any(chi >= max(len(q) for q in sq) for chi in args.chis):
        raise ValueError("Choose target bonds below the source maximum")
    source_blocks = trials.to_blocks(source, sq)
    source_norm = trials.overlap_blocks(source_blocks, source_blocks)
    baseline_blocks = None
    baseline_info = None
    if args.baseline:
        ba, bq, bm = load_trial(args.baseline)
        for key in ("h1", "u", "nelec"):
            np.testing.assert_allclose(meta[key], bm[key])
        baseline_blocks = trials.to_blocks(ba, bq)
        bn = trials.overlap_blocks(baseline_blocks, baseline_blocks)
        cross = trials.overlap_blocks(source_blocks, baseline_blocks)
        baseline_info = dict(path=str(args.baseline.resolve()), energy=float(bm["energy"]),
                             norm=bn, source_fidelity=cross**2 / (bn*source_norm))
        say(f"Baseline vs source: {baseline_info}")
        del ba
        gc.collect()

    # Local physical rotation is unitary and does not change any bond dimension.
    rotation = spin_rotation_unitary(spin_rotation_y(90.))
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(4), atol=1e-14)
    number = np.diag([0., 1., 1., 2.])
    np.testing.assert_allclose(rotation @ number, number @ rotation, atol=1e-14)
    for chi, destination in zip(args.chis, destinations):
        say(f"Compressing source to bond {chi}, preserving (N_up,N_down)")
        a, q = compress_mps_qn(source, sq, relative_tolerance=1e-13, max_bond=chi)
        blocks = trials.to_blocks(a, q)
        norm = trials.overlap_blocks(blocks, blocks)
        cross = trials.overlap_blocks(source_blocks, blocks)
        fidelity = cross**2 / (source_norm*norm)
        # Align overall sign to the source and normalize without modifying it.
        a[0] *= (1 if cross >= 0 else -1) / np.sqrt(norm)
        blocks = trials.to_blocks(a, q)
        say(f"Compressed norm before normalization {norm:.12g}; source fidelity {fidelity:.12g}")
        obs = block_observables(blocks, say=say)
        rdm = obs["rdm1"]
        np.testing.assert_allclose(np.trace(rdm, axis1=1, axis2=2), meta["nelec"], atol=1e-8)
        np.testing.assert_allclose(obs["norm"], 1., atol=1e-10)
        occupations = np.linalg.eigvalsh(rdm)
        if occupations.min() < -1e-8 or occupations.max() > 1+1e-8:
            raise AssertionError("Nonphysical one-body occupations")
        kinetic = float(np.einsum("ij,sji->", meta["h1"], rdm))
        energy = kinetic + float(meta["u"])*obs["double_occupancy"].sum()
        info = dict(source_file=str(args.source.resolve()), source_energy=float(meta["energy"]),
                    source_chi=int(meta["chi"]), source_norm=source_norm, target_chi=chi,
                    source_fidelity=float(fidelity), source_infidelity=float(1-fidelity),
                    energy=float(energy), kinetic_energy=kinetic, s2=obs["s2"],
                    norm_before_normalization=float(norm), max_bond=max(map(len, q)),
                    stored_rotation_degrees=0., production_rotation_degrees=90.,
                    rotated_trial="as_is", baseline=baseline_info,
                    note="No explicit spin projection; no CPMC run performed. Compression can change S^2.")
        if baseline_blocks is not None:
            overlap = trials.overlap_blocks(baseline_blocks, blocks)
            info["baseline_fidelity"] = overlap**2 / baseline_info["norm"]
        values = dict(h1=meta["h1"], u=meta["u"], nelec=meta["nelec"], energy=energy,
                      rdm1=rdm, chi=chi, s2=obs["s2"], double_occupancy=obs["double_occupancy"],
                      source_file=str(args.source.resolve()), source_chi=int(meta["chi"]),
                      compression_fidelity=fidelity, stored_rotation_degrees=0.,
                      recommended_rotation_degrees=90., rotated_trial="as_is",
                      compression_method="charge-preserving QR/SVD sweep; global singular-value rank cap",
                      metadata_json=json.dumps(info))
        values.update({f"A{i}": x for i, x in enumerate(a)})
        values.update({f"q{i}": x for i, x in enumerate(q)})
        temporary = destination.with_suffix(".partial.npz")
        trials._save_npz(temporary, values.items())
        temporary.replace(destination)
        destination.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
        say(f"Saved {destination}: E={energy:.12f}, S^2={obs['s2']:.8f}")
        del values, a, q, blocks, obs
        gc.collect()


if __name__ == "__main__":
    main()
