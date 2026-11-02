"""Deterministic diagnostics for reconstructed Hubbard DMRG trials.

From the repository root (NumPy/SciPy/JAX, PySCF and pyblock3 required for the CLI)::

    python -m pytest -q tests/test_trial_bias_diagnostics.py
    python -m trot.gmps.trial_bias_diagnostics --chis 128 --seeds 0 --out diagnostics/4x4
    python -m trot.gmps.trial_bias_diagnostics --chis 128 256 --seeds 0 1 --out diagnostics/scan

Small smoke run::

    python -m trot.gmps.trial_bias_diagnostics --shape 2 2 --chis 4 --sweeps 6 --walkers 2 --out /tmp/trial-smoke

No FCI or constrained-path simulation is run. The default converter is the engine's
NumPy host implementation; --device-conversion uses its compiled JAX implementation.
Both use the production orbital/bond plans. Near-node local energies are omitted,
not clipped. Generated trials are reconstructions, not historical cluster artifacts.
Imports do not configure JAX, run computations, parse arguments, or create files.

Read summary.json for a comparison across trials, and each trial's diagnostics.json
for spin weights, profiles, and per-walker errors relative to rank_exact_uncapped.
trial.npz contains A0..A15 and q0..q16, the unrotated RDM and frozen reference.
walkers.npz is shared by every trial. --walker-files accepts existing up/dn snapshots;
--walkers chooses how many to read, in file order. Sign comparisons are undefined
when either overlap is below --near-zero. Density symmetry residuals do not assign
wavefunction irreducible representations. No reference ground-state energy is assumed.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import platform
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander


SZ = np.diag([0., .5, -.5, 0.])
PLUS = np.zeros((4, 4))
PLUS[1, 2] = 1.
SX = (PLUS + PLUS.T) / 2
NUMBER = np.diag([0., 1., 1., 2.])
STARTS = ("random", "neel", "uhf_current", "uhf_opposite")


def overlap(bra, ket):
    """Real MPS inner product, with no dense many-body expansion."""
    env = np.ones((1, 1))
    for a, b in zip(bra, ket, strict=True):
        env = np.einsum("ab,apc,bpd->cd", env, a, b, optimize=True)
    return float(env.item())


def site_sum(tensors, operator):
    """Apply sum_i operator_i, an even on-site operator, as a bond-two MPO."""
    result = []
    for i, a in enumerate(tensors):
        dl, d, dr = a.shape
        b = np.zeros((2, dl, d, 2, dr))
        b[0, :, :, 0] = b[1, :, :, 1] = a
        b[0, :, :, 1] = np.einsum("pq,aqb->apb", operator, a)
        if i == 0:
            b = b[:1]
        if i == len(tensors) - 1:
            b = b[:, :, :, 1:]
        result.append(b.reshape(b.shape[0] * dl, d, b.shape[3] * dr))
    return result


def spin_squared(tensors):
    """<S²> = ||S+ psi||² + ||Sz psi||² + <Sz>, valid beyond M=0."""
    raised, z = site_sum(tensors, PLUS), site_sum(tensors, SZ)
    return (overlap(raised, raised) + overlap(z, z) + overlap(tensors, z)) / overlap(tensors, tensors)


def singlet_quadrature(nsites, nelec):
    """(angles in degrees, weights) for P_M=0 sum_k w_k Ry(beta_k) P_M=0.

    This is a singlet projector only between M=0 states. Four nodes suffice for
    fourteen electrons; omitting the M=0 projection does not give a pure singlet.
    """
    if nelec[0] != nelec[1]:
        raise ValueError("singlet quadrature requires M=0")
    smax = min(sum(nelec), 2 * nsites - sum(nelec)) // 2
    x, w = leggauss((smax + 2) // 2)
    return np.rad2deg(np.arccos(x)), w / 2


def local_elements(bra, ket, operator):
    """Unnormalized <bra|operator_i|ket> for every site (real, even operator)."""
    right = [np.ones((1, 1))]
    for a, b in zip(bra[::-1], ket[::-1], strict=True):
        right.append(np.einsum("apc,bpd,cd->ab", a, b, right[-1], optimize=True))
    right.reverse()
    left, values = np.ones((1, 1)), []
    for i, (a, b) in enumerate(zip(bra, ket, strict=True)):
        values.append(np.einsum("ab,apc,pq,bqd,cd->", left, a, operator, b, right[i + 1], optimize=True))
        left = np.einsum("ab,apc,bpd->cd", left, a, b, optimize=True)
    return np.asarray(values)


def number_blocks(blocks, charges, rotation=None):
    """Relabel SZ blocks by N and optionally rotate physical legs, without densifying H|T>."""
    from trot.trial.mps import spin_rotation_unitary

    labels = tuple(np.asarray(q).sum(axis=1, keepdims=True) for q in charges)
    maps = []
    for old, new in zip(charges, labels, strict=True):
        mapping = {}
        for q in set(map(tuple, old)):
            inds = np.flatnonzero(np.all(old == q, axis=1))
            group = np.flatnonzero(new[:, 0] == sum(q))
            mapping[q] = (np.searchsorted(group, inds), len(group))
        maps.append(mapping)
    matrix = np.eye(4) if rotation is None else spin_rotation_unitary(rotation)
    out = []
    for i, site in enumerate(blocks):
        target = {}
        for (ql, p, qr), block in site.items():
            il, nl = maps[i][ql]
            ir, nr = maps[i + 1][qr]
            for pp in np.flatnonzero(matrix[:, p]):
                key = ((sum(ql),), int(pp), (sum(qr),))
                if key not in target:
                    target[key] = np.zeros((nl, nr))
                target[key][np.ix_(il, ir)] += matrix[pp, p] * block
        out.append(target)
    return out, labels


def prepare_htrial(tensors, charges, h1, u):
    """Block-form H|T>, energy and variance; compression is only at 1e-13 roundoff tolerance."""
    from trot.gmps import trials
    from trot.meas.mps import hubbard_mpo_from_h1

    blocks = trials.to_blocks(tensors, charges)
    hp, hq = trials.compress_blocks_qn(
        *trials.trial_times_h_blocks(hubbard_mpo_from_h1(h1, u), blocks, charges), consume=True)
    norm = trials.overlap_blocks(blocks, blocks)
    energy = trials.overlap_blocks(blocks, hp) / norm
    variance = trials.overlap_blocks(hp, hp) / norm - energy**2
    if variance < -1e-8:
        raise AssertionError(f"Negative energy variance: {variance}")
    return hp, hq, float(energy), float(variance)


def spin_diagnostics(tensors, charges, nelec, htrial=None):
    """Spin weights and profiles for T, R90 T, and P_M=0 R90 T.

    Requires an SZ-labelled M=0 input. Eight nodes resolve every spin up to seven;
    for larger spins the node count is increased. Scalar charge/energy kernels
    resolve the same spin sectors. Effective-state weights are w_S P_S(0)^2.
    """
    from trot.gmps import trials
    from trot.trial.mps import rotate_spin, spin_rotation_y

    if nelec[0] != nelec[1] or any(np.asarray(q).shape[1] != 2 for q in charges):
        raise ValueError("spin_diagnostics requires an SZ-labelled M=0 input")
    norm = overlap(tensors, tensors)
    if norm <= 0:
        raise ValueError("zero-norm trial")
    smax = min(sum(nelec), 2 * len(tensors) - sum(nelec)) // 2
    spins = np.arange(smax + 1)
    x, weights = leggauss(max(8, smax + 1))
    coefficients = (2 * spins + 1)[None, :] * weights[:, None] * legvander(x, smax) / 2
    trial_blocks = trials.to_blocks(tensors, charges)
    tn = number_blocks(trial_blocks, charges)[0]
    hn = None if htrial is None else number_blocks(*htrial)[0]
    f, charge_kernels, energy_kernels = [], [], []
    for xx in x:
        rotation = spin_rotation_y(np.rad2deg(np.arccos(xx)))
        r = rotate_spin(tensors, rotation)
        f.append(overlap(tensors, r) / norm)
        charge_kernels.append(local_elements(tensors, r, NUMBER) / norm)
        if hn is not None:
            rn = number_blocks(trial_blocks, charges, rotation=rotation)[0]
            energy_kernels.append(trials.overlap_blocks(hn, rn) / norm)
    ws = coefficients.T @ f
    charge_s = coefficients.T @ charge_kernels
    direct_s2 = spin_squared(tensors)
    np.testing.assert_allclose(ws.sum(), 1., atol=1e-9, rtol=0)
    if ws.min() < -1e-9:
        raise AssertionError(f"Negative spin weight: {ws.min()}")
    np.testing.assert_allclose(ws @ (spins * (spins + 1)), direct_s2, atol=1e-8, rtol=0)
    factors = legvander([0.], smax)[0] ** 2
    sector_weight = float(ws @ factors)
    if sector_weight <= 1e-12:
        raise ValueError("R90 trial has negligible M=0 weight")
    effective_ws = ws * factors / sector_weight
    r90 = spin_rotation_y(90.)
    rotated = rotate_spin(tensors, r90)
    original = dict(spin_weights=ws, s2=direct_s2, singlet_weight=float(ws[0]),
                    density=local_elements(tensors, tensors, NUMBER) / norm,
                    sz=local_elements(tensors, tensors, SZ) / norm,
                    sx=local_elements(tensors, tensors, SX) / norm)
    full = dict(spin_weights=ws.copy(), s2=spin_squared(rotated), singlet_weight=float(ws[0]),
                density=local_elements(rotated, rotated, NUMBER) / norm,
                sz=local_elements(rotated, rotated, SZ) / norm,
                sx=local_elements(rotated, rotated, SX) / norm)
    effective = dict(spin_weights=effective_ws, s2=float(effective_ws @ (spins * (spins + 1))),
                     singlet_weight=float(effective_ws[0]), sector_weight=sector_weight,
                     density=factors @ charge_s / sector_weight,
                     # M=0 eliminates transverse moments; the remaining even-S sectors eliminate Sz moments.
                     sz=np.zeros(len(tensors)), sx=np.zeros(len(tensors)))
    np.testing.assert_allclose(full["s2"], direct_s2, atol=1e-8, rtol=0)
    np.testing.assert_allclose(full["sx"], original["sz"], atol=1e-9, rtol=0)
    np.testing.assert_allclose(full["density"], original["density"], atol=1e-9, rtol=0)
    np.testing.assert_allclose(effective["density"].sum(), sum(nelec), atol=1e-8, rtol=0)
    if hn is not None:
        es = coefficients.T @ energy_kernels
        original["energy"] = float(trials.overlap_blocks(hn, tn) / norm)
        hr, _ = number_blocks(*htrial, rotation=r90)
        rn = number_blocks(trial_blocks, charges, rotation=r90)[0]
        full["energy"] = float(trials.overlap_blocks(hr, rn) / norm)
        effective["energy"] = float(es @ factors / sector_weight)
        np.testing.assert_allclose(full["energy"], original["energy"], atol=1e-8, rtol=0)
        np.testing.assert_allclose(es.sum(), original["energy"], atol=1e-8, rtol=0)
        singlet_energy = float(es[0] / ws[0]) if ws[0] > 1e-12 else None
    else:
        singlet_energy = None
    return dict(original=original, rotated=full, effective=effective,
                singlet_projected_energy=singlet_energy, quadrature_nodes=len(x))


def channel_amplitudes(tensors, gauge=1.):
    """At most 2^16 entries for a 4x4 spin channel, including its determinant gauge."""
    v = np.ones((1, 1))
    for a in tensors:
        v = np.einsum("xa,apb->xpb", v, a).reshape(-1, a.shape[2])
    return v[:, 0] * float(gauge)


def determinant_amplitudes(orbitals):
    """Independent determinant-minor oracle, site zero the most significant bit."""
    nsites, particles = orbitals.shape
    rows = np.array(list(itertools.combinations(range(nsites), particles)), dtype=int)
    indices = ((1 << (nsites - 1 - rows))).sum(axis=1)
    v = np.zeros(2**nsites)
    v[indices] = np.linalg.det(orbitals[rows])
    return v


def channel_error(tensors, gauge, exact):
    got = channel_amplitudes(tensors, gauge)
    ref_norm, got_norm = float(exact @ exact), float(got @ got)
    return dict(relative_l2=float(np.linalg.norm(got - exact) / np.sqrt(ref_norm)),
                infidelity=float(1 - (got @ exact)**2 / (got_norm * ref_norm)) if got_norm > 0 else 1.,
                norm2=got_norm, gauge=float(gauge))


class ChannelContraction:
    """NumPy contraction using the production factorized layout, including fermionic signs."""

    def __init__(self, qa, qb, tensors, charges):
        from trot.gmps import engine

        self.plan = engine.make_factorized_plan(qa, qb, tensors, charges)

    def __call__(self, alpha, beta):
        env = np.ones((1, 1, 1, 1))
        for a, b, fixed, sp in zip(alpha, beta, self.plan.blocks, self.plan.sites, strict=True):
            aa = np.append(np.asarray(a).ravel(), 0.)[sp.gather_a]
            bb = np.append(np.asarray(b).ravel(), 0.)[sp.gather_b]
            value, indices = env[sp.src], "abk"
            for leg in sp.left_order:
                lo, hi = ("aA", "bB", "kK")[leg]
                new = indices.replace(lo, hi)
                value = np.einsum(f"t{indices},t{lo}{hi}->t{new}", value, (aa, bb, fixed)[leg], optimize=True)
                indices = new
            value *= sp.sign
            padded = np.concatenate((value, np.zeros((1,) + value.shape[1:])))
            env = padded[sp.into].sum(axis=1)
        return float(env.item())


def compare_conversions(tensors, charges, htrial, reference, walkers, *, cap=32,
                        near_zero=1e-10, device=False, say=print):
    """Four conversion controls on identical walkers, using the fully rotated trial.

    near_zero is an absolute overlap threshold for normalized trials/walkers.
    Raw overlaps/numerators are always retained, even when local energies are undefined.
    """
    from trot.gmps import engine
    from trot.gmps.utils import make_orbital_plan, plan_bonds
    from trot.trial.mps import rotate_spin, spin_rotation_y

    r = spin_rotation_y(90.)
    rotated = rotate_spin(tensors, r)
    qn = tuple(np.asarray(q).sum(axis=1, keepdims=True) for q in charges)
    hr, hq = number_blocks(*htrial, rotation=r)
    oracles = [[determinant_amplitudes(c) for c in pair] for pair in walkers]
    results = {}
    for mode, chi in (("rank_exact", None), ("rank_exact", cap), ("adaptive", None), ("adaptive", cap)):
        tag = f"{mode}_{'uncapped' if chi is None else f'chi{chi}'}"
        say(f"  conversion {tag}", flush=True)
        plans = [make_orbital_plan(c, mode, 1e-10) for c in reference]
        bonds = [None if chi is None else plan_bonds(c, p, chi, 0.)
                 for c, p in zip(reference, plans, strict=True)]
        if device:
            import jax
            import jax.numpy as jnp

            converter = engine.make_converter(*plans, *bonds)
            convert = jax.jit(converter.convert)
        records, contract, hcontract = [], None, None
        for iw, pair in enumerate(walkers):
            if device:
                alpha, beta, gauges = convert(*map(jnp.asarray, pair))
                alpha, beta = [np.asarray(a) for a in alpha], [np.asarray(b) for b in beta]
                qa, qb = converter.charges
                ga, gb = map(float, gauges)
            else:
                alpha, qa, ga = engine.channel_mps_host(pair[0], plans[0], bonds[0])
                beta, qb, gb = engine.channel_mps_host(pair[1], plans[1], bonds[1])
            if contract is None:
                contract = ChannelContraction(qa, qb, rotated, qn)
                hcontract = ChannelContraction(qa, qb, hr, hq)
            errors = [channel_error(a, g, exact) for a, g, exact in
                      zip((alpha, beta), (ga, gb), oracles[iw], strict=True)]
            if mode == "rank_exact" and chi is None:
                for error in errors:
                    if error["relative_l2"] > 1e-9:
                        raise AssertionError(f"Exact conversion disagrees with determinants: {error}")
            ov = ga * gb * contract(alpha, beta)
            numerator = ga * gb * hcontract(alpha, beta)
            small = abs(ov) <= near_zero
            records.append(dict(walker=iw, overlap=ov, numerator=numerator, near_zero=small,
                                local_energy=None if small else numerator / ov, channels=errors))
        results[tag] = dict(walkers=records, block_sizes=[p.block_sizes for p in plans],
                            reference_discarded=[0. if b is None else float(b.reference_discarded_weight)
                                                 for b in bonds])
    exact = results["rank_exact_uncapped"]["walkers"]
    for result in results.values():
        for row, ref in zip(result["walkers"], exact, strict=True):
            row["overlap_error"] = row["overlap"] - ref["overlap"]
            reliable = not (row["near_zero"] or ref["near_zero"])
            row["relative_overlap_error"] = row["overlap_error"] / abs(ref["overlap"]) if not ref["near_zero"] else None
            row["sign_disagreement"] = bool(np.sign(row["overlap"]) != np.sign(ref["overlap"])) if reliable else None
            row["local_energy_error"] = row["local_energy"] - ref["local_energy"] if reliable else None
    return results


def product_guess(shape, current=False):
    """Two-hole Néel guess; current=True reproduces ex_2d_4x4's two holes and spin flip."""
    ly, lx = shape
    y, x = np.indices(shape)
    states = np.where((x + y) % 2 == 0, 1, 2)
    x0, y0 = lx // 2 - 1, ly // 2 - 1
    holes = [(y0, x0), (y0 + 1, x0 + 1)] if current else [(y0, x0), (y0, x0 + 1)]
    for yy, xx in holes:
        states[yy, xx] = 0
    if current:
        # The diagonal holes have the same spin; flip one site of the opposite spin.
        states[y0 + 1, x0] = 1 if (x0 + y0) % 2 == 0 else 2
    return states.ravel()


def _gmps_to_pyblock3(tensors, charges):
    from pyblock3.algebra.core import SparseTensor, SubTensor
    from pyblock3.algebra.mps import MPS
    from pyblock3.algebra.symmetry import SZ as Charge

    q = lambda a, b: Charge(a + b, a - b, 0)
    result = []
    for a, left, right in zip(tensors, charges[:-1], charges[1:], strict=True):
        blocks = []
        for pair in sorted(set(map(tuple, left))):
            rows = np.flatnonzero(np.all(left == pair, axis=1))
            for p, delta in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
                target = np.asarray(pair) + delta
                cols = np.flatnonzero(np.all(right == target, axis=1))
                block = np.asarray(a)[np.ix_(rows, [p], cols)]
                if np.any(block):
                    blocks.append(SubTensor(reduced=block, q_labels=(q(*pair), q(*delta), q(*target))))
        result.append(SparseTensor(blocks=blocks))
    return MPS(tensors=result).to_flat()


def build_trial(h1, u, nelec, shape, start, chi, seed, sweeps):
    """Controlled schedule shared by all starts; no early stopping or UHF stability reruns."""
    from pyblock3.algebra.mpe import MPE
    from trot.gmps.dmrg import hubbard_pyblock3_mpo, make_pyblock3_hamiltonian, product_mps
    from trot.trial.mps import mps_trial_from_pyblock3

    ham = make_pyblock3_hamiltonian(h1, nelec)
    mpo = hubbard_pyblock3_mpo(ham, h1, u)
    metadata = dict(start=start, seed=seed, chi=chi, sweeps_requested=sweeps)
    arrays = {}
    energy = lambda mps: float(MPE(mps, mpo, mps)[0:2].expectation) / float(mps @ mps)
    rng_state = np.random.get_state()
    try:
        np.random.seed(seed)
        if start == "random":
            mps = ham.build_mps(chi)
        else:
            states = product_guess(shape, current=start == "uhf_current")
            arrays["guess_states"] = states
            if start == "neel":
                mps = product_mps(ham, states)
            else:
                from pyscf import ao2mo, gto, scf
                from trot.gmps.utils import sd_to_gmps

                n = len(h1)
                eri = np.zeros((n,) * 4)
                idx = np.arange(n)
                eri[idx, idx, idx, idx] = u
                mol = gto.M(verbose=0)
                mol.incore_anyway, mol.nelec = True, nelec
                mf = scf.UHF(mol)
                mf.get_hcore = lambda *args: h1
                mf.get_ovlp = lambda *args: np.eye(n)
                mf._eri = ao2mo.restore(8, eri, n)
                guess = np.stack([np.diag(states == 1), np.diag(states == 2)]).astype(float)
                mf.kernel(guess)
                metadata.update(uhf_converged=bool(mf.converged), uhf_energy=float(mf.e_tot))
                if not mf.converged:
                    raise RuntimeError(f"UHF did not converge for {start}; no DMRG comparison was made")
                ca, cb = (mf.mo_coeff[s][:, mf.mo_occ[s] > 0] for s in (0, 1))
                gmps = sd_to_gmps(ca, cb, chi=16)
                mps = _gmps_to_pyblock3(gmps.tensors, gmps.charges)
                e_start = energy(mps)
                metadata.update(uhf_mps_channel_chi=16, uhf_mps_discarded=float(gmps.discarded),
                                uhf_mps_energy=e_start, uhf_mps_energy_error=e_start - mf.e_tot)
                arrays.update(uhf_up=ca, uhf_dn=cb, uhf_guess=guess)
        metadata["initial_mps_energy"] = energy(mps)
        np.random.seed(seed)  # identical noise seed, independently of constructing the initial state
        result = MPE(mps, mpo, mps).dmrg(
            bdims=[chi] * sweeps, noises=[1e-5] * 4 + [0.] * (sweeps - 4),
            dav_thrds=[1e-9] * sweeps, n_sweeps=sweeps, tol=0., iprint=-1)
        metadata.update(sweep_energies=[float(e) for e in result.energies],
                        sweeps_run=len(result.energies), pyblock3_energy=energy(mps))
        if len(result.energies) != sweeps:
            raise AssertionError("DMRG did not execute the full controlled schedule")
        trial = mps_trial_from_pyblock3(mps, nelec=nelec)
    finally:
        np.random.set_state(rng_state)
    return trial, metadata, arrays


def make_probe_walkers(h1, u, nelec, count=8, seed=1729, steps=200, dt=.005):
    """Shared unconstrained, unguided HS probes. These are not an equilibrated CPMC sample."""
    from scipy.linalg import expm

    rng = np.random.default_rng(seed)
    orbitals = np.linalg.eigh(h1)[1]
    half = expm(-dt * h1 / 2)
    gamma = np.arccosh(np.exp(dt * u / 2))
    out = []
    for k in range(count):
        pair = [orbitals[:, :n].copy() if k < (count + 1) // 2 else
                np.linalg.qr(rng.normal(size=(len(h1), n)))[0] for n in nelec]
        for _ in range(steps):
            field = rng.choice([-1., 1.], len(h1))
            pair = [np.linalg.qr(half @ (np.exp(sign * gamma * field)[:, None] * (half @ c)))[0]
                    for sign, c in zip((1, -1), pair, strict=True)]
        out.append(tuple(pair))
    return out


def load_probe_walkers(paths, nsites, nelec, count):
    """Load up/dn snapshots without QR rephasing; require normalized real determinants."""
    out = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            up, dn = data["up"], data["dn"]
            if up.ndim != 3 or dn.ndim != 3 or len(up) != len(dn):
                raise ValueError(f"{path}: expected paired batches up/dn")
            if "nelec" in data and not np.array_equal(data["nelec"], nelec):
                raise ValueError(f"{path}: mismatched nelec")
            for pair in zip(up, dn, strict=True):
                for c, n in zip(pair, nelec, strict=True):
                    if c.shape != (nsites, n) or np.iscomplexobj(c) or not np.isfinite(c).all():
                        raise ValueError(f"{path}: expected finite real ({nsites}, {n}) orbitals")
                    if not np.allclose(c.T @ c, np.eye(n), rtol=0, atol=1e-8):
                        raise ValueError(f"{path}: walkers must already be orthonormal")
                out.append(tuple(np.array(c, dtype=float) for c in pair))
    if len(out) < count:
        raise ValueError(f"Requested {count} walkers, but snapshots contain only {len(out)}")
    return out[:count]


def profile_symmetry(density, shape):
    """Density-profile residuals, not wavefunction irrep assignments."""
    d = np.asarray(density).reshape(shape)
    images = dict(reflection_x=d[:, ::-1], reflection_y=d[::-1, :], rotation_180=np.rot90(d, 2))
    if shape[0] == shape[1]:
        images.update(rotation_90=np.rot90(d), diagonal_reflection=d.T)
    return {name: float(np.linalg.norm(d - other)) for name, other in images.items()}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", type=int, nargs=2, default=(4, 4), metavar=("ROWS", "COLS"))
    parser.add_argument("--chis", type=int, nargs="+", default=[128])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="random-MPS and DMRG-noise seeds; UHF guesses are fixed")
    parser.add_argument("--starts", nargs="+", choices=STARTS, default=list(STARTS))
    parser.add_argument("--sweeps", type=int, default=20)
    parser.add_argument("--walkers", type=int, default=8)
    parser.add_argument("--walker-files", type=Path, nargs="+")
    parser.add_argument("--walker-cap", type=int, default=32)
    parser.add_argument("--probe-seed", type=int, default=1729)
    parser.add_argument("--near-zero", type=float, default=1e-10)
    parser.add_argument("--device-conversion", action="store_true")
    parser.add_argument("--out", type=Path, required=True, help="new output directory; existing directories are refused")
    args = parser.parse_args(argv)
    shape = tuple(args.shape)
    if any(n < 2 or n % 2 for n in shape) or np.prod(shape) > 16:
        parser.error("shape must have even sides >= 2 and at most 16 sites")
    if min(args.chis) < 1 or min(args.seeds) < 0 or args.sweeps < 6 or args.walkers < 1 or args.walker_cap < 1:
        parser.error("positive chis/walkers/cap, nonnegative seeds and at least six sweeps are required")
    if not np.isfinite(args.near_zero) or args.near_zero <= 0:
        parser.error("near-zero must be finite and positive")
    if args.out.exists():
        parser.error(f"output directory already exists: {args.out}; choose a new directory")

    from trot import config

    config.configure_once(single_precision=False, matmul_precision="highest")
    import jax
    from trot.ham.hubbard import square_hopping_matrix
    from trot.trial.mps import natural_orbitals

    if not jax.config.jax_enable_x64:
        raise RuntimeError("Diagnostics require JAX float64")
    nsites, u = int(np.prod(shape)), 8.
    nelec = (nsites // 2 - 1,) * 2
    h1 = np.asarray(square_hopping_matrix(*shape, 1.), dtype=float)
    walkers = (load_probe_walkers(args.walker_files, nsites, nelec, args.walkers) if args.walker_files else
               make_probe_walkers(h1, u, nelec, args.walkers, args.probe_seed))
    versions = {}
    for package in ("numpy", "scipy", "jax", "jaxlib", "pyscf", "pyblock3"):
        versions[package] = importlib.metadata.version(package)
    args.out.mkdir(parents=True)
    manifest = dict(parameters=vars(args), shape=shape, nelec=nelec, u=u, t=1., boundaries="open",
                    site_order="site = y * columns + x", versions=versions, python=platform.python_version(),
                    device=str(jax.devices()[0]), conversion_backend="jax" if args.device_conversion else "numpy_host",
                    trial_provenance="reconstructions, not historical cluster artifacts", reference_energy=None,
                    walker_source="snapshots" if args.walker_files else "unguided HS probes, not CPMC samples",
                    probe_dt=.005, probe_steps=200, near_zero_convention="absolute overlap, normalized states")
    _write_json(args.out / "manifest.json", manifest)
    np.savez(args.out / "walkers.npz", up=np.stack([p[0] for p in walkers]), dn=np.stack([p[1] for p in walkers]),
             h1=h1, u=u, nelec=nelec)
    summary = []
    for chi, seed, start in itertools.product(dict.fromkeys(args.chis), dict.fromkeys(args.seeds),
                                             dict.fromkeys(args.starts)):
        tag = f"{start}_chi{chi}_seed{seed}"
        folder = args.out / tag
        folder.mkdir()
        print(f"{tag}: building reconstructed trial", flush=True)
        trial, meta, arrays = build_trial(h1, u, nelec, shape, start, chi, seed, args.sweeps)
        tensors = [np.asarray(a) for a in trial.tensors]
        charges = trial.charge_arrays()
        reference = tuple(natural_orbitals(np.asarray(trial.rdm1)[s], nelec[s])[0] for s in (0, 1))
        np.savez(folder / "trial.npz", h1=h1, u=u, nelec=nelec, rdm1=np.asarray(trial.rdm1),
                 reference_up=reference[0], reference_dn=reference[1], **arrays,
                 **{f"A{i}": a for i, a in enumerate(tensors)}, **{f"q{i}": q for i, q in enumerate(charges)})
        _write_json(folder / "dmrg.json", meta)
        print(f"{tag}: energy moments and spin decomposition", flush=True)
        hp, hq, energy, variance = prepare_htrial(tensors, charges, h1, u)
        np.testing.assert_allclose(energy, meta["pyblock3_energy"], rtol=0, atol=1e-8)
        spin = spin_diagnostics(tensors, charges, nelec, (hp, hq))
        for state in ("original", "rotated", "effective"):
            spin[state]["density_symmetry_residuals"] = profile_symmetry(spin[state]["density"], shape)
        spin["original"]["variance"] = spin["rotated"]["variance"] = variance
        result = dict(dmrg=meta, energy=energy, variance=variance,
                      retained_minus_last_sweep=energy - meta["sweep_energies"][-1], spin=spin)
        _write_json(folder / "diagnostics.json", result)
        result["conversions"] = compare_conversions(tensors, charges, (hp, hq), reference, walkers,
                                                    cap=args.walker_cap, near_zero=args.near_zero,
                                                    device=args.device_conversion)
        _write_json(folder / "diagnostics.json", result)
        production = result["conversions"][f"adaptive_chi{args.walker_cap}"]["walkers"]
        row = dict(trial=tag, energy=energy, variance=variance, s2=spin["original"]["s2"],
                   effective_s2=spin["effective"]["s2"], effective_singlet_weight=spin["effective"]["singlet_weight"],
                   max_channel_relative_l2=max(c["relative_l2"] for r in production for c in r["channels"]),
                   max_abs_local_energy_error=max((abs(r["local_energy_error"]) for r in production
                                                  if r["local_energy_error"] is not None), default=None),
                   near_zero_pairs=sum(r["sign_disagreement"] is None for r in production),
                   sign_disagreements=sum(r["sign_disagreement"] is True for r in production))
        summary.append(row)
        _write_json(args.out / "summary.json", summary)
        print(f"{tag}: E={energy:.10f}, variance={variance:.3e}, S²={row['s2']:.6f}, "
              f"effective S²={row['effective_s2']:.6f}, max channel error={row['max_channel_relative_l2']:.3e}",
              flush=True)
    print(f"Results saved to {args.out}. Trial rankings are measurements, not assertions of CPMC bias.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
