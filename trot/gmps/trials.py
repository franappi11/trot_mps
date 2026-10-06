"""Trials for MPS-CPMC on any Hubbard h1 (trot.gmps.driver.run_qmc_mps), chain or lattice alike.

The geometry is read off h1 (describe_h1); it only names cache files and runs and picks the DMRG defaults, so every
other step is the same code for a chain, a square lattice or any other h1:

* describe_h1(h1): open/periodic chain, Lx x Ly square lattice (open, periodic or antiperiodic sides, site
  x * Ly + y) or a general h1 (named by a hash).
* load_or_make_dmrg_trial(...): pyblock3 DMRG trial (trot.gmps.dmrg.dmrg_h1) with a file cache. The file names of
  mps_cpmc_gpu (chain) and mps_cpmc_2d_gpu (square lattice) are kept, so their caches load as they are.
* make_trial(...): the MpsTrial a run uses: optionally spin-rotated by exp(-i beta S^y), projected
  onto the walkers' sector or used as it is, with the rdm1 whose natural orbitals set the walker plan and start.
* export_trial(...): the trial in the notebooks' npz format.
* Block-form H|trial> (to_blocks ... load_or_make_htrial): the compressed, charge-labelled H|trial> without dense
  tensors, for 6x6 and larger lattices whose dense H|trial> does not fit in memory, cached next to the trial.
"""

from __future__ import annotations

import hashlib
import itertools
import os
import time
import zipfile
from pathlib import Path
from typing import NamedTuple

import numpy as np

from trot.ham.hubbard import hopping_matrix, square_hopping_matrix
from trot.meas.mps import CHANNEL_CHARGE, apply_mpo, compress_mps, hubbard_mpo_from_h1
from trot.trial.mps import (
    PHYSICAL_CHARGE,
    _charge_index,
    _label_sectors,
    make_mps_trial,
    one_rdm,
    rotate_spin,
    spin_rotation_y,
)
from trot.trial.mps_rotation import rotate_mps_trial, rotated_rdm1

BOUNDARY_CODE = {"open": "o", "periodic": "p", "antiperiodic": "a"}


# ---------------------------------------------------------------------------------------------
# Geometry from h1
# ---------------------------------------------------------------------------------------------


class Lattice(NamedTuple):
    """What describe_h1 finds in an h1: kind "chain", "square" or "general", the name used for cache files and run
    tags ("L100", "L16p", "sq4x4oo", "h1-<hash>"), the site count, the hopping amplitude and, when it applies, the
    square lattice's sides and boundaries (a chain is Lx = n_sites, Ly = 1)."""

    kind: str
    name: str
    n_sites: int
    hopping: float
    Lx: int | None = None
    Ly: int | None = None
    boundary_x: str | None = None
    boundary_y: str | None = None

    def record(self) -> dict:
        """The result-record keys the plot scripts read (L for a chain, Lx/Ly/boundaries for a lattice)."""
        keys = dict(lattice=self.name, n_sites=self.n_sites, hopping=self.hopping)
        if self.kind == "chain":
            keys.update(L=self.n_sites, boundary_x=self.boundary_x)
        elif self.kind == "square":
            keys.update(Lx=self.Lx, Ly=self.Ly, boundary_x=self.boundary_x, boundary_y=self.boundary_y)
        return keys


def _same(a, b) -> bool:
    return a.shape == b.shape and np.allclose(a, b, rtol=0.0, atol=1.0e-12)


def describe_h1(h1) -> Lattice:
    """The lattice behind a real symmetric one-body matrix: an open, periodic or antiperiodic chain (sites in order),
    an Lx x Ly square lattice of trot.ham.hubbard.square_hopping_matrix (site x * Ly + y), or "general"."""
    h1 = np.asarray(h1, dtype=float)
    n = len(h1)
    off = h1[~np.eye(n, dtype=bool)]
    hops = np.abs(off[off != 0])
    if hops.size and not np.any(np.diag(h1)) and np.allclose(hops, hops[0], rtol=0.0, atol=1.0e-12):
        t = float(hops[0])
        if _same(h1, hopping_matrix(n, t)):
            return Lattice("chain", f"L{n}", n, t, n, 1, "open", "open")
        for boundary in ("periodic", "antiperiodic"):
            if n > 2 and _same(h1, square_hopping_matrix(n, 1, t, boundary, "open")):
                return Lattice("chain", f"L{n}{BOUNDARY_CODE[boundary]}", n, t, n, 1, boundary, "open")
        for Lx in range(2, n // 2 + 1):
            if n % Lx:
                continue
            Ly = n // Lx
            for bx, by in itertools.product(BOUNDARY_CODE, repeat=2):
                if _same(h1, square_hopping_matrix(Lx, Ly, t, bx, by)):
                    name = f"sq{Lx}x{Ly}{BOUNDARY_CODE[bx]}{BOUNDARY_CODE[by]}"
                    return Lattice("square", name, n, t, Lx, Ly, bx, by)
    digest = hashlib.sha1(np.round(h1, 12).tobytes()).hexdigest()[:10]
    return Lattice("general", f"h1-{digest}", n, float(hops.max()) if hops.size else 0.0)


def dmrg_defaults(lattice: Lattice) -> dict:
    """The DMRG options the production scripts used: on a chain the term-built MPO with a warm-up schedule
    (mps_cpmc_gpu), elsewhere build_qc_mpo with the plain schedule and tol 1e-6 (mps_cpmc_2d_gpu)."""
    if lattice.kind == "chain":
        return dict(mpo="terms", schedule="warmup", tol=None)
    return dict(mpo="qc", schedule="plain", tol=1.0e-6)


# ---------------------------------------------------------------------------------------------
# DMRG trial and its cache
# ---------------------------------------------------------------------------------------------


class DmrgTrialData(NamedTuple):
    tensors: list
    charges: tuple
    davidson_energy: float  # two-site energy before the last truncation, not variational
    variational_energy: float | None  # <mps|H|mps>/<mps|mps> (None in old chain cache files)
    path: Path | None  # the cache file, if any


def trial_cache_file(cache_dir, lattice: Lattice, nelec, u, *, chi, sweeps, seed=0, mpo, schedule, bdims=(),
                     tol=None) -> Path:
    """Cache file name of a DMRG trial: mps_cpmc_gpu's for an open chain with the chain defaults, mps_cpmc_2d_gpu's
    otherwise (lattice name, schedule, tolerance and MPO in the name)."""
    nup, ndn = (int(x) for x in nelec)
    t, n = lattice.hopping, lattice.n_sites
    open_chain = lattice.name == f"L{n}"
    if open_chain and (mpo, schedule, tol) == ("terms", "warmup", None) and not bdims:
        name = f"L{n}_n{nup}-{ndn}_t{t:g}_U{u:g}_chi{chi}_sw{sweeps}_seed{seed}"
    else:
        ramp = "-".join(map(str, bdims)) if bdims else str(chi)
        name = f"{lattice.name}_n{nup}-{ndn}_t{t:g}_U{u:g}_chi{ramp}_sw{sweeps}"
        name += "" if tol is None else f"_tol{tol:g}"
        # the qc MPO is the unnamed default except on an open chain, where the unnamed name is the chain default's
        name += f"_seed{seed}" + ("" if mpo == "qc" and not open_chain else f"_mpo{mpo}")
        name += "" if schedule == "plain" else f"_{schedule}"
    return Path(cache_dir) / f"{name}.npz"


def load_or_make_dmrg_trial(h1, u, nelec, *, chi, sweeps, seed=0, cache_dir="", mpo=None, schedule=None, bdims=(),
                            tol="default", say=print) -> DmrgTrialData:
    """The DMRG trial of HamHubbard(h1, u) at nelec: read from cache_dir when present, else made by
    trot.gmps.dmrg.dmrg_h1 (pyblock3, imported only then) and written there. mpo, schedule and tol default to
    dmrg_defaults(describe_h1(h1)). A cached file that stores h1 must match this h1."""
    h1 = np.asarray(h1, dtype=float)
    lattice = describe_h1(h1)
    defaults = dmrg_defaults(lattice)
    mpo = mpo or defaults["mpo"]
    schedule = schedule or ("plain" if bdims else defaults["schedule"])
    tol = defaults["tol"] if tol == "default" else tol
    n = len(h1)
    path = None
    if cache_dir:
        path = trial_cache_file(cache_dir, lattice, nelec, u, chi=chi, sweeps=sweeps, seed=seed, mpo=mpo,
                                schedule=schedule, bdims=tuple(bdims), tol=tol)
    if path is not None and path.exists():
        with np.load(path) as data:
            if "h1" in data.files and not _same(np.asarray(data["h1"]), h1):
                raise ValueError(f"{path} was made for a different h1")
            tensors = [np.asarray(data[f"A{i}"]) for i in range(n)]
            charges = tuple(np.asarray(data[f"q{i}"]) for i in range(n + 1))
            davidson = float(data["energy"])
            variational = float(data["mps_energy"]) if "mps_energy" in data.files else None
        say(f"trial loaded from {path}", flush=True)
        return DmrgTrialData(tensors, charges, davidson, variational, path)

    from trot.gmps.dmrg import dmrg_h1

    tensors, charges, davidson, sweep_energies, variational = dmrg_h1(
        h1, u, nelec, chi=chi, n_sweeps=sweeps, seed=seed, mpo=mpo, schedule=schedule, bdims=tuple(bdims), tol=tol)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, energy=davidson, mps_energy=variational, sweep_energies=np.asarray(sweep_energies), h1=h1,
                 dmrg_mpo=mpo, **{f"A{i}": A for i, A in enumerate(tensors)},
                 **{f"q{i}": q for i, q in enumerate(charges)})
        os.replace(tmp, path)
        say(f"trial saved to {path}", flush=True)
    return DmrgTrialData(tensors, charges, davidson, variational, path)


# ---------------------------------------------------------------------------------------------
# The trial a run uses: rotation and the rdm1 behind the walker plan and start
# ---------------------------------------------------------------------------------------------


def make_trial(tensors, charges, nelec, *, rotation=0.0, rotated_trial="projected", natural_rdm1="after",
               rdm1=None):
    """The MpsTrial for a DMRG trial with (N_up, N_dn) bond labels.

    rotation: spin rotation exp(-i beta S^y) in degrees (trot.trial.mps.spin_rotation_y); 90 removes every odd total
      spin from the walkers' sector (one-node spin projection).
    rotated_trial: "projected" (trot.trial.mps_rotation.rotate_mps_trial: the S_z-conserving rotation MPO, straight
      into the walkers' (N_up, N_dn) sector, exact labels) or "as_is" (make_mps_trial of the rotated tensors: bonds
      unchanged, particle-number labels). Both give the same local energies and CPMC trajectory.
    natural_rdm1: the rdm1 whose natural orbitals give the plan reference, the bond plans and the walker start, the
      rotated trial's ("after", trot's default; spin averaged at 90 deg) or the unrotated trial's ("before").
    rdm1: the unrotated trial's rdm1 when already known (e.g. from the block-form H|trial> cache), to skip one_rdm.

    Returns (trial, info) with info = dict(trial_rotation, rotated_trial, natural_rdm1, sector_weight).
    """
    nelec = (int(nelec[0]), int(nelec[1]))
    tensors = [np.asarray(A) for A in tensors]
    info = dict(trial_rotation=float(rotation), rotated_trial=None, natural_rdm1=None, sector_weight=1.0)
    trial = make_mps_trial(tensors, charges, nelec=nelec, rdm1=rdm1)
    if not rotation:
        return trial, info
    if natural_rdm1 not in ("after", "before"):
        raise ValueError("natural_rdm1 must be 'after' or 'before'")
    R = spin_rotation_y(rotation)
    # the rotated trial's rdm1 before any projection follows from the unrotated one (no second one_rdm)
    start = np.asarray(trial.rdm1) if natural_rdm1 == "before" else rotated_rdm1(np.asarray(trial.rdm1), R)
    if rotated_trial == "projected":
        trial = rotate_mps_trial(trial, R, rdm1=start)
    elif rotated_trial == "as_is":
        trial = make_mps_trial(rotate_spin(tensors, R), nelec=nelec, rdm1=start)
    else:
        raise ValueError("rotated_trial must be 'projected' or 'as_is'")
    info.update(rotated_trial=rotated_trial, natural_rdm1=natural_rdm1, sector_weight=trial.sector_weight)
    return trial, info


def export_trial(path, tensors, charges, h1, u, dmrg_energy, references, start, *, gamma=None, htrial_file=None):
    """The trial in the format the analysis notebooks load: T{i} (d=4 tensors, local index n_up + 2 n_dn), H{i}
    (H|trial>, dense-compressed), q{i} (bond labels), e_dmrg, gamma (spin-resolved rdm1), h1, the plan reference
    (reference_up/dn) and the walkers' start (start_up/dn). With htrial_file (block-form H|trial>) no H{i} is written
    and the file's path is stored instead. Written atomically: runs sharing a trial may export it together."""
    h1 = np.asarray(h1, dtype=float)
    tensors = [np.asarray(A) for A in tensors]
    extra = {}
    if htrial_file is not None:
        H = []
        extra["htrial_file"] = str(htrial_file)
    else:
        H = compress_mps(apply_mpo(hubbard_mpo_from_h1(h1, float(u)), tensors))
    gamma = np.stack(one_rdm(tensors)) if gamma is None else np.asarray(gamma)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, e_dmrg=dmrg_energy, gamma=gamma, h1=h1, **extra,
                        reference_up=np.asarray(references[0]), reference_dn=np.asarray(references[1]),
                        start_up=np.asarray(start[0]), start_dn=np.asarray(start[1]),
                        **{f"T{i}": A for i, A in enumerate(tensors)},
                        **{f"H{i}": np.asarray(A) for i, A in enumerate(H)},
                        **{f"q{i}": np.asarray(q) for i, q in enumerate(charges)})
    os.replace(tmp, path)
    print(f"saved the trial to {path}", flush=True)


# ---------------------------------------------------------------------------------------------
# H|trial> in block form (6x6 and larger lattices)
# ---------------------------------------------------------------------------------------------
#
# trial_times_h and compress_mps_qn keep every tensor dense. The uncompressed bond of H|trial> is (2 + 4 Ly) chi_T
# and compression keeps about 4 Ly chi_T of it (4x4 at chi_T 256: 4608 -> 4092), so at 8x8 and chi_T 512 the dense
# tensors need ~600 GB. The block form keeps only charge-allowed blocks: site k is a dict {(q_left, p, q_right):
# block}, next to the same full label arrays as before. A block's rows and columns are the bond indices with those
# labels in increasing order (_charge_index's order), and a missing block is zero. The functions below repeat
# trial_times_h and compress_mps_qn on this form: the same bonds, labels, sector matrices and rank cut.

PHYSICAL_INDEX = {tuple(int(x) for x in q): p for p, q in enumerate(PHYSICAL_CHARGE)}


def _shift(q, dq):
    return (q[0] + int(dq[0]), q[1] + int(dq[1]))


def to_blocks(tensors, charges):
    """Dense charge-labelled MPS -> block form (blocks that are not all zero)."""
    out = []
    for k, A in enumerate(tensors):
        A = np.asarray(A)
        left, right = _charge_index(charges[k]), _charge_index(charges[k + 1])
        site = {}
        for ql, rows in left.items():
            for p, dq in enumerate(PHYSICAL_CHARGE):
                qr = _shift(ql, dq)
                if qr in right:
                    block = A[np.ix_(rows, [p], right[qr])][:, 0]
                    if block.any():
                        site[(ql, p, qr)] = block
        out.append(site)
    return out


def from_blocks(blocks, charges):
    """Block form -> dense tensors (for small lattices and the tests)."""
    tensors = []
    for k, site in enumerate(blocks):
        left, right = _charge_index(charges[k]), _charge_index(charges[k + 1])
        A = np.zeros((len(charges[k]), 4, len(charges[k + 1])))
        for (ql, p, qr), block in site.items():
            A[np.ix_(left[ql], [p], right[qr])] = block[:, None, :]
        tensors.append(A)
    return tensors


def overlap_blocks(bra, ket):
    """<bra|ket> of two real block-form MPS, one charge sector at a time."""
    env = {}
    for k, (A, B) in enumerate(zip(bra, ket)):
        new = {}
        for key, X in A.items():
            Y = B.get(key)
            E = np.ones((1, 1)) if k == 0 else env.get(key[0])
            if Y is None or E is None:
                continue
            term = X.T @ E @ Y
            new[key[2]] = new[key[2]] + term if key[2] in new else term
        env = new
    return float(sum(v.sum() for v in env.values()))  # the last bond is 1 x 1


def _htrial_bonds(W, trial_charges):
    """trial_times_h's bond layout: each MPO channel's charge, the channels each bond keeps, and the
    (N_alpha, N_beta) labels of the flattened (channel, trial) bonds."""
    n, D = len(trial_charges) - 1, W.shape[1]
    delta = np.zeros((D, 2), int)
    delta[1:-1] = np.tile(CHANNEL_CHARGE, ((D - 2) // 4, 1))
    active = ([np.zeros(1, int)]
              + [np.flatnonzero(np.any(W[b - 1] != 0, axis=(0, 1, 2))) for b in range(1, n)]
              + [np.full(1, D - 1)])
    charges = tuple((delta[active[b]][:, None, :] + np.asarray(trial_charges[b])[None]).reshape(-1, 2)
                    for b in range(n + 1))
    return delta, active, charges


def _htrial_layout(W, trial_charges):
    """_htrial_bonds plus, per bond, where channel j's trial indices of label q sit inside H|trial>'s label group
    delta_j + q, and each group's size. The flattened bond is channel-major, so a group lists channel 0's trial
    indices first, then channel 1's."""
    delta, active, charges = _htrial_bonds(W, trial_charges)
    layout = []
    for b, labels in enumerate(trial_charges):
        offset, size = {}, {}
        groups = _charge_index(labels)
        for j, a in enumerate(active[b]):
            for q, indices in groups.items():
                L = _shift(q, delta[a])
                offset[(j, q)] = size.get(L, 0)
                size[L] = size.get(L, 0) + len(indices)
        layout.append((offset, size))
    return delta, active, charges, layout


def _htrial_terms(W, trial_blocks, delta, active, k):
    """Every (H|trial> block key, channel positions, trial block labels, MPO coefficient, trial block) that
    trial_times_h's site-k product sums."""
    for jl, a in enumerate(active[k]):
        for jr, b in enumerate(active[k + 1]):
            O = W[k][a, :, :, b]
            if not O.any():
                continue
            for (qc, q, qd), B in trial_blocks[k].items():
                L, R = _shift(qc, delta[a]), _shift(qd, delta[b])
                for p in np.flatnonzero(O[:, q]):
                    yield (L, int(p), R), jl, jr, qc, qd, O[p, q], B


def htrial_block_bytes(W, trial_blocks, trial_charges):
    """Bytes the uncompressed block-form H|trial> takes (for the log, before it is made)."""
    delta, active, _, layout = _htrial_layout(W, trial_charges)
    total = 0
    for k in range(len(trial_blocks)):
        keys = {key for key, *_ in _htrial_terms(W, trial_blocks, delta, active, k)}
        total += sum(layout[k][1][L] * layout[k + 1][1][R] for L, _, R in keys)
    return 8 * total


def trial_times_h_blocks(W, trial_blocks, trial_charges):
    """trial_times_h in block form: the same bonds, labels and entries (each entry is one MPO coefficient times one
    trial entry, as in the dense product), no dense tensors."""
    delta, active, charges, layout = _htrial_layout(W, trial_charges)
    tensors = []
    for k in range(len(trial_blocks)):
        (left_offset, left_size), (right_offset, right_size) = layout[k], layout[k + 1]
        site = {}
        for key, jl, jr, qc, qd, coefficient, B in _htrial_terms(W, trial_blocks, delta, active, k):
            if key not in site:
                L, p, R = key
                if _shift(L, PHYSICAL_CHARGE[p]) != R:
                    raise AssertionError(f"an MPO channel does not conserve the charge at site {k}")
                site[key] = np.zeros((left_size[L], right_size[R]))
            r0, c0 = left_offset[(jl, qc)], right_offset[(jr, qd)]
            site[key][r0:r0 + B.shape[0], c0:c0 + B.shape[1]] += coefficient * B
        tensors.append(site)
    return tensors, charges


def compress_blocks_qn(tensors, charges, relative_tolerance=1.0e-13, consume=False):
    """compress_mps_qn in block form. Every sector matrix is assembled with its rows and columns in compress_mps_qn's
    order (zero rows and columns included), so the QR, the SVD and the rank cut are the same. consume=True replaces
    the input list's entries as the sweeps go, so the uncompressed tensors are freed site by site."""
    A = tensors if consume else [dict(site) for site in tensors]
    Q = [np.asarray(q, int).reshape(-1, 2) for q in charges]
    d = len(PHYSICAL_CHARGE)
    for i in range(len(A) - 1):
        left = _charge_index(Q[i])
        rows = (Q[i][:, None, :] + PHYSICAL_CHARGE[None]).reshape(-1, 2)
        by_right, following = {}, {}
        for (ql, p, qr), X in A[i].items():
            by_right.setdefault(qr, []).append((ql, p, X))
        for (ql, p, qr), X in A[i + 1].items():
            following.setdefault(ql, []).append((p, qr, X))
        site, after, labels = {}, {}, []
        for c, r, k in _label_sectors(rows, Q[i + 1]):
            M = np.zeros((len(r), len(k)))  # rows l * d + p in increasing order: one p per l
            for ql, p, X in by_right.get(c, ()):
                M[np.searchsorted(r, left[ql] * d + p)] = X
            q, rr = np.linalg.qr(M)
            for ql, indices in left.items():
                p = PHYSICAL_INDEX.get((c[0] - ql[0], c[1] - ql[1]))
                if p is not None:
                    part = q[np.searchsorted(r, indices * d + p)]
                    if part.any():
                        site[(ql, p, c)] = part
            for p, qr, X in following.get(c, ()):
                after[(c, p, qr)] = rr @ X
            labels += [c] * q.shape[1]
        A[i], A[i + 1] = site, after
        Q[i + 1] = np.asarray(labels, int).reshape(-1, 2)
    for i in range(len(A) - 1, 0, -1):
        right = _charge_index(Q[i + 1])
        columns = (Q[i + 1][None, :, :] - PHYSICAL_CHARGE[:, None, :]).reshape(-1, 2)
        preceding = {}
        for (ql, p, qr), X in A[i - 1].items():
            preceding.setdefault(qr, []).append((ql, p, X))
        factors = []
        for c, r, k in _label_sectors(Q[i], columns):
            parts, width = [], 0  # columns p * Dr + r in increasing order: p-major
            for p, dq in enumerate(PHYSICAL_CHARGE):
                qr = _shift(c, dq)
                if qr in right:
                    parts.append((p, qr, width, len(right[qr])))
                    width += len(right[qr])
            if width != len(k):
                raise AssertionError(f"sector {c} at bond {i}: {width} columns, compress_mps_qn has {len(k)}")
            M = np.zeros((len(r), width))
            for p, qr, start, w in parts:
                X = A[i].get((c, p, qr))
                if X is not None:
                    M[:, start:start + w] = X
            u, s, vh = np.linalg.svd(M, full_matrices=False)
            factors.append((c, parts, u, s, vh))
        cut = relative_tolerance * max(max(f[3][0] for f in factors), 1e-300)
        site, before, labels = {}, {}, []
        for c, parts, u, s, vh in factors:
            n = int(np.sum(s > cut))
            if n == 0:
                continue
            for p, qr, start, w in parts:
                part = vh[:n, start:start + w]
                if part.any():
                    site[(c, p, qr)] = np.ascontiguousarray(part)
            us = u[:, :n] * s[:n]
            for ql, p, X in preceding.get(c, ()):
                before[(ql, p, c)] = X @ us
            labels += [c] * n
        del factors
        A[i], A[i - 1] = site, before
        Q[i] = np.asarray(labels, int).reshape(-1, 2)
    return A, tuple(Q)


def energy_plan_bytes(charges):
    """Upper bound of the device memory make_factorized_plan gives H|trial>'s blocks: every label shared with the
    walkers, per site transitions x largest left group x largest right group (the plan pads each bond to its largest
    group)."""
    groups = [_charge_index(q) for q in charges]
    total = 0
    for k in range(len(charges) - 1):
        transitions = sum(_shift(c, dq) in groups[k + 1] for c in groups[k] for dq in PHYSICAL_CHARGE)
        total += transitions * max(map(len, groups[k].values())) * max(map(len, groups[k + 1].values()))
    return 8 * total


def htrial_cache_file(trial_path) -> Path:
    path = Path(trial_path)
    return path.with_name(f"{path.stem}_htrial.npz")


def _pack_blocks(blocks, charges):
    """(name, array) pairs of the npz: the bond labels q{b} and, per site, K{k} (each block's q_left, p, q_right and
    shape) and V{k} (the blocks raveled one after another). A generator, so only one site's V{k} is copied at a time."""
    for b, q in enumerate(charges):
        yield f"q{b}", np.asarray(q)
    for k, site in enumerate(blocks):
        keys = [(*ql, p, *qr, *X.shape) for (ql, p, qr), X in site.items()]
        yield f"K{k}", np.asarray(keys, np.int64).reshape(-1, 7)
        yield f"V{k}", np.concatenate([X.ravel() for X in site.values()]) if site else np.zeros(0)


def _save_npz(path, arrays):
    """np.savez (uncompressed) for an iterable of (name, array), written one at a time."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, array in arrays:
            with archive.open(f"{name}.npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asanyarray(array), allow_pickle=False)


def _unpack_blocks(data, n):
    charges = tuple(np.asarray(data[f"q{b}"]) for b in range(n + 1))
    blocks = []
    for k in range(n):
        keys, values = np.asarray(data[f"K{k}"]), np.asarray(data[f"V{k}"])
        site, start = {}, 0
        for qla, qlb, p, qra, qrb, rows, cols in keys.tolist():
            site[((qla, qlb), p, (qra, qrb))] = values[start:start + rows * cols].reshape(rows, cols)
            start += rows * cols
        blocks.append(site)
    return blocks, charges


def load_or_make_htrial(trial_path, h1, u, trial_np, trial_charges, mps_energy, say=print):
    """The compressed, charge-labelled H|trial> in block form (tensors, labels) and a dict with <trial|H|trial>, the
    uncompressed bonds and the trial's rdm1 (gamma).

    With trial_path (the trial's cache file) they are read from, or written to, <trial file>_htrial.npz: made once on
    the CPU (run_mps_cpmc.py --prepare-only) for every GPU run of that trial. A file made for another h1 or another
    trial (its pyblock3 energy differs) is refused. Without a cache file the product is made and not saved."""
    h1 = np.asarray(h1, dtype=float)
    n = len(h1)
    path = htrial_cache_file(trial_path) if trial_path is not None else None
    if path is not None and path.exists():
        with np.load(path) as data:
            same_trial = mps_energy is None or float(data["trial_mps_energy"]) == mps_energy
            if not _same(np.asarray(data["h1"]), h1) or not same_trial:
                raise ValueError(f"{path} was made for another trial")
            blocks, charges = _unpack_blocks(data, n)
            info = dict(trial_energy=float(data["trial_energy"]),
                        uncompressed_bonds=[int(x) for x in data["uncompressed_bonds"]],
                        gamma=np.stack([np.asarray(data["gamma_a"]), np.asarray(data["gamma_b"])]))
        say(f"H|trial> loaded from {path}", flush=True)
        return (blocks, charges), info

    clock, seconds = time.perf_counter, {}
    W = hubbard_mpo_from_h1(h1, float(u))
    trial_blocks = to_blocks(trial_np, trial_charges)
    exact_bonds = [len(q) for q in _htrial_bonds(W, trial_charges)[2]]
    say(f"H|trial> (MPO bond {W.shape[1]}) uncompressed bonds {exact_bonds}: "
        f"{htrial_block_bytes(W, trial_blocks, trial_charges) / 1e9:.2f} GB in blocks", flush=True)
    start = clock()
    exact = trial_times_h_blocks(W, trial_blocks, trial_charges)
    seconds["product"] = clock() - start
    start = clock()
    blocks, charges = compress_blocks_qn(*exact, consume=True)
    del exact
    seconds["compress"] = clock() - start
    stored = sum(X.nbytes for site in blocks for X in site.values())
    energy = overlap_blocks(trial_blocks, blocks) / overlap_blocks(trial_blocks, trial_blocks)
    say(f"H|trial> compressed bonds {[len(q) for q in charges]}: {stored / 1e9:.2f} GB in blocks; "
        f"device energy-plan blocks at most {energy_plan_bytes(charges) / 1e9:.1f} GB", flush=True)
    if mps_energy is not None:
        say(f"<trial|H|trial> {energy:.12f}, pyblock3 {mps_energy:.12f}, difference {energy - mps_energy:.1e}",
            flush=True)
        if abs(energy - mps_energy) > 1e-8 * max(1.0, abs(mps_energy)):
            # two MPOs of the same Hamiltonian (hubbard_mpo_from_h1 here, the DMRG's in pyblock3)
            raise AssertionError(f"<trial|H|trial> = {energy} but pyblock3 gives {mps_energy}; H|trial> not saved")
    start = clock()
    gamma = np.stack(one_rdm(trial_np))
    seconds["rdm1"] = clock() - start
    say("H|trial> times: " + ", ".join(f"{k} {v:.1f} s" for k, v in seconds.items()), flush=True)
    if path is not None:
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        header = dict(h1=h1, trial_mps_energy=np.nan if mps_energy is None else mps_energy, trial_energy=energy,
                      relative_tolerance=1.0e-13, uncompressed_bonds=np.asarray(exact_bonds),
                      gamma_a=gamma[0], gamma_b=gamma[1])
        _save_npz(tmp, itertools.chain(header.items(), _pack_blocks(blocks, charges)))
        os.replace(tmp, path)
        say(f"H|trial> saved to {path}", flush=True)
    return (blocks, charges), dict(trial_energy=energy, uncompressed_bonds=exact_bonds, gamma=gamma)
