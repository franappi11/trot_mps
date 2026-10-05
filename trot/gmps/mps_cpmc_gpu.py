"""CPMC with a DMRG trial and SD walkers converted to MPS, built for one GPU.

The device engine (circuit compilation, batched factorisations, factorized
contractions, half steps) lives in trot/gmps/gpu.py and is re-exported here; the
same engine runs trot's native MPS-CPMC ops (QmcParamsMps.engine="batched").
This file keeps the run script: Config, the trial cache and rotation
(Config.trial_rotation), the measurement block, logging, snapshots and main().

Same algorithm as mps_cpmc_new.py:
the walkers stay Slater determinants, every overlap converts each spin channel
to a charge-labelled d=2 MPS with Fishman-White gates, truncating gate by gate
with the orthogonality centre on the gate, and one conversion plus cached right
environments serves the whole diagonal HS sweep. What changes is how the work is
laid out for a GPU, where throughput comes from large batches and few kernels:

* The gate circuit is compiled once on the host (compile_circuit): every centre
  move and split becomes a static gather into padded batches of charge-sector
  blocks. Each sector is factored by the same method as the original
  (_factor_block: closed form for one row or column, QR when exact, the Gram
  eigh when truncating), but on the device (linalg="batched") all sectors of one
  kind are a single call over walkers x spins x sectors. Both spin channels run
  in one batch whenever they share a gate sequence, each with its own sectors and
  kept ranks (spin_batch).
* Walker-trial contractions never form the d=4 walker. The environment is kept as
  (alpha, beta, trial) blocks per shared (N_alpha, N_beta) label and the three
  legs are contracted one at a time (make_factorized_plan), which costs
  Pa*Pb*Pt*(Pa+Pb+Pt) per transition instead of Pa^2*Pb^2*Pt. Sums over incoming
  transitions are gathers, not atomic scatter-adds, so runs are deterministic.
* The local energy uses a charge-labelled, sector-compressed H|trial>
  (compress_mps_qn) with the same factorized contraction. The dense path
  (energy="dense") is kept for validation only.
* Each CPMC step is a scan over half steps with one conversion call site; the
  HS sweep runs under lax.cond on even half steps. The measurement block rescales
  overlaps by det(R) after orthonormalisation and gathers them after the comb, so
  a block needs one conversion per half step plus one for the energy (trot's block
  needs three more).
* Trial blocks are jit arguments, not HLO constants; walkers are chunked with
  lax.map only as far as the memory model requires (n_chunks=0).

The implementation is real-valued and assumes the spatial local basis
|0>, |alpha>, |beta>, |alpha beta>, indexed by n_alpha + 2*n_beta.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Callable, NamedTuple

import jax
import jax.experimental
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
import scipy.linalg
from jax import lax

jax.config.update("jax_enable_x64", True)

try:  # DMRG only; a cached trial (Config.trial_cache) runs without pyblock3
    from pyblock3.algebra.mpe import MPE
    from pyblock3.algebra.symmetry import SZ
    from pyblock3.fcidump import FCIDUMP
    from pyblock3.hamiltonian import Hamiltonian
except ImportError:  # pragma: no cover
    MPE = SZ = FCIDUMP = Hamiltonian = None

from trot import walkers as wk
from trot.core.system import System
from trot.ham.hubbard import HamHubbard
from trot.prop.hubbard_cpmc_ops import _build_prop_ctx
from trot.prop.types import PropState, QmcParams
from trot.stat_utils import blocking_analysis_ratio, reject_outliers
from trot.trial.mps import _label_sectors, compress_mps_qn, label_array  # noqa: F401  (one copy, N labels too)
from trot.trial.uhf import UhfTrial, get_rdm1 as uhf_get_rdm1
from trot.walkers import _qr as qr_with_det
from trot.gmps.gpu import (  # noqa: F401  (the device engine; re-exported for scripts and tests)
    Circuit,
    ContractionPlan,
    Converter,
    DeviceData,
    FactorOp,
    GpuOps,
    N_PHYSICAL,
    ONE_HOT,
    PHYSICAL,
    SECTOR_KINDS,
    SectorClass,
    SelectSpec,
    SitePlan,
    _LEGS,
    _can_group,
    _charge_index,
    _cholesky_qr,
    _fixed_block,
    _fixed_key,
    _gate_op,
    _gather_flat,
    _group_op,
    _index_1d,
    _leg_order,
    _legs,
    _move_op,
    _reduce,
    _sector_kind,
    _select_per_walker,
    _table,
    _take_per_channel,
    channel_mps_host,
    cholesky_qr2,
    choose_chunks,
    chunked,
    circuit_stats,
    compile_circuit,
    conversion_self_check,
    counts_bond_plan,
    device_bytes_limit,
    device_channel_angles,
    factor_sectors,
    field_sweep,
    gate_pair_batched,
    gate_sites,
    left_contract,
    make_batch_qr,
    make_converter,
    make_factorized_plan,
    make_gpu_ops,
    make_half_step,
    memory_model,
    mps_overlap_host,
    number_labels,
    right_environments,
    run_circuit,
    trial_identity,
    trial_labels,
    walker_blocks,
)
from trot.gmps.utils import (  # noqa: F401  (byte-identical host primitives)
    BondPlan,
    OrbitalPlan,
    SectorPlan,
    _assemble,
    _block_plan,
    _factor_block,
    _key,
    _move_centre,
    _rotate_mode_to_front,
    _shift_centre,
    _vector_qr,
    channel_angles,
    channel_mps,
    combine_channels,
    combined_charges,
    gate_pair,
    make_orbital_plan,
    plan_bonds,
    sector_plan,
    split_pair,
)


@dataclass(frozen=True)
class Config:
    L: int = 16
    n_up: int = 8
    n_down: int = 8
    hopping: float = 1.0
    interaction: float = 4.0
    trial_chi: int = 64
    dmrg_sweeps: int = 14
    dmrg_seed: int = 0
    # DMRG initial state: "neel" (the Neel product state, sweeps directly at trial_chi; no domain walls), "random"
    # (random MPS with a warm-up at a larger bond; low trial_chi on long chains gets stuck with domain walls, e.g. the
    # L=100 chi=16 trial with 4), "auto" (neel at half filling, else random) or "warm" (load a build_warm_trials.py
    # trial from trial_cache, never run DMRG). Cache files are tagged _neel / _warm; random keeps the old names.
    dmrg_init: str = "auto"
    # rank_exact is exact for every full-rank walker before bond truncation.
    # adaptive is cheaper but is only guaranteed for the reference determinant.
    # maximal is exact but usually creates more gates than rank_exact.
    orbital_plan: str = "adaptive"  # rank_exact, adaptive, maximal
    occupation_tolerance: float = 1.0e-10
    walker_channel_chi: int | None = 4
    walker_cutoff: float = 0.0
    # Determinant that freezes the static structure: the gate circuit (make_orbital_plan)
    # and the per-sector kept counts (plan_bonds). "natural" is the determinant of the
    # trial's most occupied natural orbitals, "rhf" the free-fermion determinant.
    plan_reference: str = "natural"  # natural, rhf
    # Determinant every walker starts from. "natural" follows trot's convention
    # (natural orbitals of the trial's one-body density matrix).
    walker_start: str = "natural"  # natural, rhf
    n_walkers: int = 1024
    n_blocks: int = 40
    n_equilibration: int = 15
    n_steps: int = 20
    dt: float = 0.01
    weight_floor: float = 1.0e-8
    seed: int = 1234
    # --- GPU layout (none of these change the algorithm) ---
    n_chunks: int = 0  # walker chunks per half step; 0 = the fewest the memory model allows
    mem_fraction: float = 0.75  # share of the device memory the chunker may plan for
    linalg: str = "auto"  # batched (all sectors per op in one call per kind, GPUs), native (the original loop), auto
    walker_qr: str = "auto"  # cholesky (CholeskyQR2), native (Householder, trot's _qr), auto
    energy: str = "blocked"  # blocked (charge-labelled H|trial>), dense (original; small sizes)
    spin_batch: bool = True  # convert both spin channels as one batch when their circuits match
    self_check: bool = True  # check one device conversion against the NumPy original at start-up
    trial_cache: str = ""  # directory for cached DMRG trials (skips DMRG when present)
    # Rotate the DMRG trial by exp(-i beta S^y) on every site (degrees; trot.trial.mps.rotate_spin). 90 = one-node
    # spin projection: every odd total spin removed. rotated_trial="projected" projects it exactly onto the walkers'
    # (n_up, n_down) sector (make_mps_trial: (N_up, N_dn) labels); "as_is" keeps the rotated tensors and bonds
    # (trot.trial.mps_rotation.make_rotated_mps_trial) and blocks every contraction on particle number
    # (gpu.number_labels): overlaps are sqrt(sector weight) times the projected trial's, local energies and the run
    # are the same. Either way the plan reference and the walker start come from the rotated trial's rdm1 before
    # any projection (trot's MpsTrial / RotatedMpsTrial default), so both choices propagate the same walkers.
    trial_rotation: float = 0.0
    rotated_trial: str = "projected"  # projected, as_is
    # With trial_rotation: the rdm1 whose natural orbitals give the plan reference, the bond plans and the walker
    # start: the trial's "after" the rotation (spin averaged at 90 deg) or the DMRG trial's "before" it.
    natural_rdm1: str = "after"  # after, before
    compile_cache: str = ""  # JAX persistent compilation cache directory
    result_json: str = ""
    block_log: str = ""  # if set, append every block's scalars here as it finishes
    walker_snapshots: str = ""  # if set, every block's walkers go here (.npz, notebook format; see WalkerSnapshots)
    trial_export: str = ""  # if set, the DMRG trial goes here in the notebooks' format (see export_trial)
    tag: str = ""


CFG = Config()
PHYSICAL_CHARGE = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])








def hopping_matrix(n: int, hopping: float) -> np.ndarray:
    h1 = np.zeros((n, n))
    i = np.arange(n - 1)
    h1[i, i + 1] = h1[i + 1, i] = -hopping
    return h1




































def one_rdm(tensors):
    """Spin-resolved one-body density matrices <c^dag_i,sigma c_j,sigma> of a real
    d=4 MPS in the interleaved (alpha before beta on each site) ordering."""
    create_a = np.zeros((4, 4)); create_a[1, 0] = create_a[3, 2] = 1.0
    create_b = np.zeros((4, 4)); create_b[2, 0] = 1.0; create_b[3, 1] = -1.0
    parity_a, parity_b = np.diag([1.0, -1.0, 1.0, -1.0]), np.diag([1.0, 1.0, -1.0, -1.0])
    operators = {"a": (create_a @ parity_b, create_a.T, np.diag([0.0, 1.0, 0.0, 1.0])),
                 "b": (parity_a @ create_b, create_b.T, np.diag([0.0, 0.0, 1.0, 1.0]))}
    A = [np.asarray(t) for t in tensors]
    L = len(A)
    site = lambda E, x, O: np.einsum("ab,apc,pq,bqd->cd", E, A[x], O, A[x], optimize=True)
    left = [np.ones((1, 1))]
    for x in range(L):
        left.append(site(left[-1], x, np.eye(4)))
    right = [np.ones((1, 1))]
    for x in range(L - 1, -1, -1):
        right.insert(0, np.einsum("apc,bpd,cd->ab", A[x], A[x], right[0], optimize=True))
    gammas = []
    for create_i, annihilate_j, number in operators.values():
        gamma = np.zeros((L, L))
        for i in range(L):
            gamma[i, i] = np.sum(site(left[i], i, number) * right[i + 1])
            E = site(left[i], i, create_i)
            for j in range(i + 1, L):
                gamma[i, j] = gamma[j, i] = np.sum(site(E, j, annihilate_j) * right[j + 1])
                E = site(E, j, parity_a @ parity_b)  # Jordan-Wigner string between i and j
        gammas.append(gamma / left[-1][0, 0])
    return tuple(gammas)


def natural_orbitals(gamma, n):
    """The n most occupied natural orbitals and all occupations, descending."""
    occupations, vectors = np.linalg.eigh(gamma)
    return vectors[:, ::-1][:, :n].copy(), occupations[::-1]


def contract_real(left_mps, right_mps):
    """Real MPS contraction.  Complex support would require conjugating the left MPS."""
    env = jnp.ones((1, 1))
    for left, right in zip(left_mps, right_mps):
        env = jnp.einsum("ab,apr,bps->rs", env, left, right, optimize=True)
    return env.reshape(())


def hubbard_mpo(L, hopping, interaction):
    """Open-chain Hubbard MPO in the spatial local basis, virtual dimension six."""
    eye = np.eye(4)
    create_a = np.zeros((4, 4)); create_a[1, 0] = create_a[3, 2] = 1.0
    create_b = np.zeros((4, 4)); create_b[2, 0] = 1.0; create_b[3, 1] = -1.0
    annihilate_a, annihilate_b = create_a.T, create_b.T
    parity_a = np.diag([1.0, -1.0, 1.0, -1.0])
    parity_b = np.diag([1.0, 1.0, -1.0, -1.0])
    double = np.diag([0.0, 0.0, 0.0, 1.0])

    W = np.zeros((L, 6, 4, 4, 6))
    for i in range(L):
        W[i, 0, :, :, 0] = eye
        W[i, 5, :, :, 5] = eye
        W[i, 0, :, :, 5] = interaction * double
        if i < L - 1:
            W[i, 0, :, :, 1] = create_a @ parity_b
            W[i, 0, :, :, 2] = annihilate_a @ parity_b
            W[i, 0, :, :, 3] = parity_a @ create_b
            W[i, 0, :, :, 4] = parity_a @ annihilate_b
        if i:
            W[i, 1, :, :, 5] = -hopping * annihilate_a
            W[i, 2, :, :, 5] = -hopping * create_a
            W[i, 3, :, :, 5] = -hopping * annihilate_b
            W[i, 4, :, :, 5] = -hopping * create_b
    return W


def apply_mpo(W, tensors):
    out = []
    for i, (operator, A) in enumerate(zip(W, tensors)):
        if i == 0:
            operator = operator[:1]
        if i == len(tensors) - 1:
            operator = operator[:, :, :, 5:6]
        T = np.einsum("apqb,cqd->acpbd", operator, np.asarray(A))
        dl, cl, d, dr, cr = T.shape
        out.append(T.reshape(dl*cl, d, dr*cr))
    return out


def compress_mps(tensors, relative_tolerance=1.0e-13):
    """Compress a fixed MPS by a host QR/SVD sweep."""
    tensors = [np.array(A, copy=True) for A in tensors]
    for i in range(len(tensors) - 1):
        Dl, d, Dr = tensors[i].shape
        q, r = np.linalg.qr(tensors[i].reshape(Dl*d, Dr))
        tensors[i] = q.reshape(Dl, d, -1)
        tensors[i + 1] = np.tensordot(r, tensors[i + 1], axes=1)
    for i in range(len(tensors) - 1, 0, -1):
        Dl, d, Dr = tensors[i].shape
        u, s, vh = np.linalg.svd(tensors[i].reshape(Dl, d*Dr), full_matrices=False)
        rank = max(1, int(np.sum(s > relative_tolerance * max(s[0], 1e-300))))
        tensors[i] = vh[:rank].reshape(rank, d, Dr)
        tensors[i - 1] = np.tensordot(tensors[i - 1], u[:, :rank] * s[:rank], axes=1)
    return tensors


def build_dmrg_hamiltonian(cfg: Config):
    h1 = hopping_matrix(cfg.L, cfg.hopping)
    g2 = np.zeros((cfg.L,) * 4)
    i = np.arange(cfg.L)
    g2[i, i, i, i] = cfg.interaction
    fcidump = FCIDUMP(pg="c1", n_sites=cfg.L, n_elec=cfg.n_up + cfg.n_down,
                      twos=cfg.n_up - cfg.n_down, ipg=0, h1e=h1, g2e=g2)
    return Hamiltonian(fcidump, flat=True)


def hubbard_dmrg_mpo(hamiltonian, cfg: Config):
    """Hubbard-chain MPO from its ~6L operator terms (bond dimension ~6).
    build_qc_mpo treats g2 as a general two-electron tensor, which gives a bond
    dimension ~L^2/2 and tens of GB of DMRG environments at L=48, chi=200."""
    # Term encoding of pyblock3's flat builder (as in Hamiltonian.build_complex_qc_mpo):
    # operator index = OP * (0 for c+, 1 for c) + SITE * site + SPIN * spin, -1 pads.
    SPIN, SITE, OP = 1, 2, 16384
    C, D = 0 * OP, 1 * OP
    h1 = hopping_matrix(cfg.L, cfg.hopping)
    values, terms = [], []
    for i, j in zip(*np.nonzero(h1)):
        for s in (0, 1):
            values.append(h1[i, j])
            terms.append([C + i * SITE + s * SPIN, D + j * SITE + s * SPIN, -1, -1])
    for i in range(cfg.L):
        # n_up n_down = c+_up c+_down c_down c_up
        values.append(cfg.interaction)
        terms.append([C + i * SITE, C + i * SITE + SPIN, D + i * SITE + SPIN, D + i * SITE])
    gen = (np.array(values, dtype=np.float64), np.array(terms, dtype=np.int32))
    return hamiltonian.build_mpo(gen, cutoff=1.0e-12)


def run_dmrg(hamiltonian, cfg: Config):
    np.random.seed(cfg.dmrg_seed)
    mpo = hubbard_dmrg_mpo(hamiltonian, cfg)
    if dmrg_init(cfg) == "neel":
        # Neel product state, then sweeps directly at trial_chi: a warm-up at a larger bond with strong noise would
        # wash the seed out. trot.gmps.dmrg (imported here, it needs pyblock3) holds the one implementation.
        from trot.gmps.dmrg import neel_states, product_mps

        mps = product_mps(hamiltonian, neel_states(hopping_matrix(cfg.L, cfg.hopping), (cfg.n_up, cfg.n_down)))
        n = cfg.dmrg_sweeps
        result = MPE(mps, mpo, mps).dmrg(bdims=[cfg.trial_chi] * n, noises=[1.0e-6] * (n - 2) + [0.0] * 2,
                                         dav_thrds=[1.0e-10], iprint=-1, n_sweeps=n)
        print("DMRG initial state: Neel product state")
        return mps, float(result.energies[-1])
    print("DMRG initial state: random MPS (warm-up schedule)")
    mps = hamiltonian.build_mps(cfg.trial_chi)
    # Warm up at a larger bond dimension with stronger noise, then truncate to
    # trial_chi: sweeping at chi=8 from a random start got stuck at L=48,
    # U=8 and 12 (energy per site 10x further from chi=200 than elsewhere).
    chi = cfg.trial_chi
    warm = max(chi, min(4 * chi, 64))
    bdims = [warm] * 4 + [max(chi, warm // 2)] * 2 + [chi] * cfg.dmrg_sweeps
    noises = [1.0e-4] * 4 + [1.0e-5] * 2 + [1.0e-6] * (cfg.dmrg_sweeps - 2) + [0.0] * 2
    result = MPE(mps, mpo, mps).dmrg(
        bdims=bdims, noises=noises, dav_thrds=[1.0e-10], iprint=-1, n_sweeps=len(bdims))
    return mps, float(result.energies[-1])


def spin_occupations(charge):
    return ((int(charge.n) + int(charge.twos)) // 2,
            (int(charge.n) - int(charge.twos)) // 2)


def flat_blocks(mps, site):
    tensor = mps[site]
    for k in range(tensor.n_blocks):
        labels = tuple(SZ.from_flat(int(x)) for x in tensor.q_labels[k])
        shape = tuple(map(int, tensor.shapes[k]))
        data = np.asarray(tensor.data[tensor.idxs[k]:tensor.idxs[k + 1]]).reshape(shape)
        yield labels, shape, data


def densify_with_charges(mps, L):
    """Convert a pyblock3 MPS to dense tensors and matching (N_alpha,N_beta) labels."""
    key = lambda q: (int(q.n), int(q.twos))
    left, right = [], []
    for site in range(L):
        lo, ro = {}, {}
        for (ql, _, qr), shape, _ in flat_blocks(mps, site):
            lo[key(ql)], ro[key(qr)] = shape[0], shape[2]
        left.append(lo); right.append(ro)
    for site in range(L - 1):
        if left[site + 1] != right[site]:
            raise AssertionError(f"bond {site + 1} differs between neighboring tensors")

    bond_sectors = [left[i] for i in range(L)] + [right[-1]]
    offsets, bond_charges = [], []
    for sectors in bond_sectors:
        offset, labels, start = {}, [], 0
        for q, multiplicity in sorted(sectors.items()):
            offset[q] = (start, multiplicity)
            spin_charge = ((q[0] + q[1]) // 2, (q[0] - q[1]) // 2)
            labels.extend([spin_charge] * multiplicity)
            start += multiplicity
        offsets.append((offset, start))
        bond_charges.append(np.asarray(labels, int))

    dense = []
    for site in range(L):
        A = np.zeros((offsets[site][1], 4, offsets[site + 1][1]))
        for (ql, qp, qr), shape, data in flat_blocks(mps, site):
            if shape[1] != 1:
                raise AssertionError("expected one-dimensional physical charge blocks")
            na, nb = spin_occupations(qp)
            ol, dl = offsets[site][0][key(ql)]
            or_, dr = offsets[site + 1][0][key(qr)]
            A[ol:ol + dl, na + 2*nb, or_:or_ + dr] = data[:, 0, :]
        dense.append(A)
    return dense, tuple(bond_charges)




def constrain_ratio(ratio, weight_floor):
    """trot's cpmc_step rule: zero every overlap ratio at or below the floor.

    This is also the constrained-path condition (a sign change gives ratio <= 0),
    so it cannot be dropped; weight_floor=0 keeps only the constraint.
    """
    return jnp.where(ratio <= weight_floor, 0.0, ratio)


# ============================================================================
# Host: trial cache, reference conversion, charge-labelled H|trial>
# ============================================================================

def dmrg_init(cfg):
    """The DMRG initial state cfg asks for: "neel", "random" or "warm" ("auto": neel when the open chain's Neel
    state, up on even sites and down on odd ones or the reverse, has (n_up, n_down))."""
    if cfg.dmrg_init not in ("auto", "neel", "random", "warm"):
        raise ValueError(f"dmrg_init must be auto, neel, random or warm, got {cfg.dmrg_init!r}")
    neel = sorted((cfg.n_up, cfg.n_down)) == [cfg.L // 2, (cfg.L + 1) // 2]
    if cfg.dmrg_init == "neel" and not neel:
        raise ValueError(f"no Neel product state with (n_up, n_down) = ({cfg.n_up}, {cfg.n_down}) on L={cfg.L}")
    if cfg.dmrg_init == "auto":
        return "neel" if neel else "random"
    return cfg.dmrg_init


def _trial_cache_file(cfg):
    tag = {"random": "", "neel": "_neel", "warm": "_warm"}[dmrg_init(cfg)]
    name = (f"L{cfg.L}_n{cfg.n_up}-{cfg.n_down}_t{cfg.hopping:g}_U{cfg.interaction:g}"
            f"_chi{cfg.trial_chi}_sw{cfg.dmrg_sweeps}_seed{cfg.dmrg_seed}{tag}.npz")
    return Path(cfg.trial_cache) / name


def load_or_run_trial(cfg):
    """DMRG trial as dense tensors, (N_alpha, N_beta) bond labels and its energy.

    With cfg.trial_cache set the trial is read from, or written to, one npz file per
    (L, filling, t, U, chi, sweeps, seed), so GPU jobs and benchmarks skip DMRG.
    """
    path = _trial_cache_file(cfg) if cfg.trial_cache else None
    if path is not None and path.exists():
        with np.load(path) as data:
            tensors = [np.asarray(data[f"A{i}"]) for i in range(cfg.L)]
            charges = tuple(np.asarray(data[f"q{i}"]) for i in range(cfg.L + 1))
            energy = float(data["energy"])
        print(f"trial loaded from {path}")
        return tensors, charges, energy
    if dmrg_init(cfg) == "warm":
        raise FileNotFoundError(f"dmrg_init='warm' loads build_warm_trials.py output, but {path} does not exist")
    if MPE is None:
        raise ImportError("pyblock3 is needed to run DMRG (or point trial_cache at a cached trial)")
    mps, energy = run_dmrg(build_dmrg_hamiltonian(cfg), cfg)
    tensors, charges = densify_with_charges(mps, cfg.L)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, energy=energy, **{f"A{i}": A for i, A in enumerate(tensors)},
                 **{f"q{i}": q for i, q in enumerate(charges)})
        os.replace(tmp, path)
        print(f"trial saved to {path}")
    return tensors, charges, energy






# Entering MPO bond state k of hubbard_mpo has created this much (N_alpha, N_beta):
# 0 "nothing started" and 5 "done" carry nothing, 1..4 follow cr_a, an_a, cr_b, an_b.
MPO_QN = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]])


def apply_mpo_qn(W, tensors, charges):
    """apply_mpo, carrying the bond labels. The output bond flattens (MPO bond, MPS
    bond) with the MPO index major, so its label is MPO_QN[a] + q[c]; for particle-number
    labels (width 1, a rotated trial used as it is) the MPO state adds its N."""
    charges = [label_array(q) for q in charges]
    mpo_qn = MPO_QN if charges[0].shape[1] == 2 else MPO_QN.sum(axis=1, keepdims=True)
    out, labels = [], [mpo_qn[:1] + charges[0]]
    last = len(tensors) - 1
    for i, (w, A) in enumerate(zip(W, tensors)):
        if i == 0:
            w = w[:1]
        if i == last:
            w = w[:, :, :, 5:6]
        T = np.einsum("apqb,cqd->acpbd", w, np.asarray(A))
        dl, cl, d, dr, cr = T.shape
        out.append(T.reshape(dl * cl, d, dr * cr))
        mpo_right = mpo_qn[5:6] if i == last else mpo_qn
        labels.append((mpo_right[:, None, :] + charges[i + 1][None, :, :]).reshape(-1, mpo_qn.shape[1]))
    return out, tuple(labels)


def make_block(ops: GpuOps, params: QmcParams, n_chunks: int, energy_chunks: int, record_comb=False):
    """trot.prop.blocks.block for these ops: n_prop_steps steps, orthonormalise,
    measure the energy, comb. Overlaps after orthonormalisation are rescaled by
    det(R) and after the comb gathered, instead of reconverting every walker.
    record_comb: also return the comb's input, per walker: its weight at the
    measurement (pre_comb_weights) and the copy map (comb_index: walker i after the
    comb is a copy of walker comb_index[i] before it)."""
    half_step = make_half_step(ops, params, n_chunks)
    n_half = 2 * params.n_prop_steps

    def block(state: PropState, data: DeviceData):
        state, _ = lax.scan(lambda s, i: (half_step(s, i, data), None), state, jnp.arange(n_half))
        qu, du = ops.batch_qr(state.walkers[0])
        qd, dd = ops.batch_qr(state.walkers[1])
        overlaps = state.overlaps / (du * dd)

        e_samples = chunked(lambda a, b: ops.energies(a, b, data), energy_chunks, qu, qd)
        thresh = jnp.sqrt(2.0 / jnp.asarray(params.dt))
        e_ref = state.e_estimate
        is_nan = ~jnp.isfinite(e_samples)
        e_samples = jnp.where(is_nan | (jnp.abs(e_samples - e_ref) > thresh), e_ref, e_samples)
        weights = jnp.where(is_nan, 0.0, state.weights)
        w_sum = jnp.sum(weights)
        w_sum_safe = jnp.where(w_sum == 0, 1.0, w_sum)
        e_block = jnp.sum(weights * e_samples) / w_sum_safe
        e_block = jnp.where(w_sum == 0, e_ref, e_block)
        alpha = jnp.asarray(params.shift_ema, dtype=jnp.result_type(e_block))
        e_estimate = (1.0 - alpha) * state.e_estimate + alpha * e_block

        key, subkey = jax.random.split(state.rng_key)
        zeta = jax.random.uniform(subkey)
        n = weights.shape[0]
        idx = wk._sr_indices(weights, zeta, n)
        average = jnp.cumsum(jnp.abs(weights))[-1] / n
        state = PropState((qu[idx], qd[idx]), jnp.full((n,), average, weights.dtype), overlaps[idx], key,
                          state.pop_control_ene_shift, e_estimate, state.node_encounters)
        scalars = dict(energy=e_block, weight=w_sum)
        if record_comb:
            scalars.update(pre_comb_weights=weights, comb_index=idx.astype(jnp.int32))
        return state, scalars

    return block


def init_state(ops: GpuOps, system: System, trial_data, params: QmcParams, probe_fn=None):
    """trot's init_prop_state: every walker starts at the natural orbitals of the
    placeholder trial's rdm1; all are identical, so one walker is converted."""
    wu, wd = wk.init_walkers(sys=system, rdm1=uhf_get_rdm1(trial_data), n_walkers=params.n_walkers)
    wu, wd = jnp.real(wu), jnp.real(wd)
    probe = (probe_fn or jax.jit(ops.probe))(wu[0], wd[0], ops.data)
    n = params.n_walkers
    # Two separate buffers (the state is donated), and strong dtypes throughout:
    # a weakly typed scalar would not match the compiled run_blocks after one call.
    e = float(probe["energy"])
    state = PropState((wu.astype(jnp.float64), wd.astype(jnp.float64)), jnp.ones((n,), jnp.float64),
                      jnp.full((n,), probe["overlap"], jnp.float64), jax.random.PRNGKey(int(params.seed)),
                      jnp.asarray(e, jnp.float64), jnp.asarray(e, jnp.float64), jnp.zeros((), jnp.int64))
    return state, probe


def make_block_logger(path, n_equilibration, tag=""):
    """Append each block's scalars to a JSONL file the moment the block finishes:
    raw values, before outlier rejection, so partial runs survive."""
    counter = iter(range(1 << 62))
    start = time.perf_counter()
    result_spec = jax.ShapeDtypeStruct((), jnp.int32)

    def write(energy, weight, e_estimate, nodes):
        block = next(counter)
        record = dict(tag=tag, block=block,
                      phase="equilibration" if block < n_equilibration else "sampling",
                      energy=float(energy), weight=float(weight), e_estimate=float(e_estimate),
                      node_encounters=int(nodes), seconds=time.perf_counter() - start)
        with Path(path).open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        return np.int32(0)

    def log(state, scalars):
        jax.experimental.io_callback(write, result_spec, scalars["energy"], scalars["weight"],
                                     state.e_estimate, state.node_encounters, ordered=True)
    return log


def make_run_blocks(block_fn, logger=None):
    donate = () if jax.default_backend() == "cpu" else (0,)

    @partial(jax.jit, static_argnames=("n_blocks",), donate_argnums=donate)
    def run_blocks(state, data, n_blocks):
        def one(state, _):
            state, scalars = block_fn(state, data)
            if logger is not None:
                logger(state, scalars)
            return state, scalars
        return lax.scan(one, state, None, length=n_blocks)
    return run_blocks


def compiled_memory(compiled):
    try:
        m = compiled.memory_analysis()
        return dict(temp_bytes=int(m.temp_size_in_bytes), argument_bytes=int(m.argument_size_in_bytes),
                    output_bytes=int(m.output_size_in_bytes))
    except Exception:  # pragma: no cover - not every backend reports it
        return {}


def peak_device_bytes():
    try:
        return (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use")
    except Exception:  # pragma: no cover
        return None


class WalkerSnapshots:
    """Every block's walkers, in the format fixed_block_walkers.ipynb and entanglement_vs_gmps.ipynb load.

    Snapshot 0 is the starting population and snapshot b the population after block b, after its comb (where
    every walker has the same weight): up, dn of shape (n_snap, n_walkers, L, N_sigma), float64, at imaginary
    time tau_snapshots = b * n_steps * dt. Per block b (the one ending at tau_blocks[b] and producing snapshot
    b + 1): energies, weights (the block's energy and total weight), e_estimate and node_encounters (cumulative),
    and the comb's input: pre_comb_weights[b, j] is walker j's weight at the energy measurement, and
    comb_index[b, i] = j says walker i of snapshot b + 1 is a copy of that walker j. So walker i of snapshot
    b + 1 carried the weight pre_comb_weights[b, comb_index[b, i]] before the comb.

    Arrays are written block by block into <name>.parts/ (progress.json says how many are valid), so a crash keeps
    every finished block; finish() packs them with the config into <name>.npz and removes the parts.
    """

    def __init__(self, path, n_snap, n_walkers, L, n_up, n_down, config=None):
        from numpy.lib.format import open_memmap
        self.config = dict(config or {})  # stored in progress.json too, so unfinished runs can be plotted
        self.path = Path(path)
        self.parts = self.path.with_name(self.path.stem + ".parts")
        self.parts.mkdir(parents=True, exist_ok=True)
        memmap = lambda name, shape, dtype=np.float64: open_memmap(self.parts / f"{name}.npy", mode="w+",
                                                                   dtype=dtype, shape=shape)
        self.up = memmap("up", (n_snap, n_walkers, L, n_up))
        self.dn = memmap("dn", (n_snap, n_walkers, L, n_down))
        self.pre_comb_weights = memmap("pre_comb_weights", (n_snap - 1, n_walkers))
        self.comb_index = memmap("comb_index", (n_snap - 1, n_walkers), np.int32)
        self.snapshots, self.blocks = 0, []
        print(f"walker snapshots: {n_snap} x {n_walkers} walkers -> {self.path} "
              f"({(self.up.nbytes + self.dn.nbytes) / 1e9:.1f} GB)", flush=True)

    def add_snapshot(self, state):
        self.up[self.snapshots] = np.asarray(state.walkers[0])
        self.dn[self.snapshots] = np.asarray(state.walkers[1])
        self.snapshots += 1

    def add_block(self, state, scalars):
        """After one block (a run_blocks call with n_blocks=1): its scalars, the comb data and the new walkers."""
        b = len(self.blocks)
        self.pre_comb_weights[b] = np.asarray(scalars["pre_comb_weights"])[-1]
        self.comb_index[b] = np.asarray(scalars["comb_index"])[-1]
        self.blocks.append(dict(energy=float(np.asarray(scalars["energy"])[-1]),
                                weight=float(np.asarray(scalars["weight"])[-1]),
                                e_estimate=float(state.e_estimate), node_encounters=int(state.node_encounters)))
        self.add_snapshot(state)
        for array in (self.up, self.dn, self.pre_comb_weights, self.comb_index):
            array.flush()
        (self.parts / "progress.json").write_text(json.dumps(dict(snapshots=self.snapshots, blocks=self.blocks,
                                                                  config=self.config)))

    def finish(self, config, arrays=None):
        config = {**self.config, **config}
        n, nb = self.snapshots, len(self.blocks)
        step = config["N_PROP"] * config["DT"]
        column = lambda key, dtype=float: np.array([blk[key] for blk in self.blocks], dtype=dtype)
        np.savez(self.path, up=self.up[:n], dn=self.dn[:n], energies=column("energy"), weights=column("weight"),
                 e_estimate=column("e_estimate"), node_encounters=column("node_encounters", np.int64),
                 pre_comb_weights=self.pre_comb_weights[:nb], comb_index=self.comb_index[:nb],
                 tau_snapshots=np.arange(n) * step, tau_blocks=np.arange(1, nb + 1) * step,
                 config=json.dumps(config), **(arrays or {}))
        del self.up, self.dn, self.pre_comb_weights, self.comb_index
        import shutil
        shutil.rmtree(self.parts)
        print(f"saved {self.path}: {n} snapshots, {nb} blocks", flush=True)


def export_trial(path, cfg, trial_np, trial_charges, dmrg_energy, references, start):
    """The DMRG trial in the format fixed_block_walkers.ipynb loads (its DMRG_FILE): T{i} (d=4 tensors, local
    index n_up + 2 n_dn), H{i} (H|trial>, dense-compressed exactly as the notebook builds it), q{i} (the
    (N_up, N_dn) bond labels; particle numbers N for rotated_trial="as_is") and e_dmrg; plus its spin-resolved
    one-body density matrices (gamma), the plan
    reference determinant (reference_up/dn: the natural orbitals for plan_reference="natural") and the walkers'
    starting determinant (start_up/dn)."""
    H = compress_mps(apply_mpo(hubbard_mpo(cfg.L, cfg.hopping, cfg.interaction), trial_np))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, e_dmrg=dmrg_energy, gamma=np.stack(one_rdm(trial_np)),
                        reference_up=np.asarray(references[0]), reference_dn=np.asarray(references[1]),
                        start_up=np.asarray(start[0]), start_dn=np.asarray(start[1]),
                        **{f"T{i}": np.asarray(A) for i, A in enumerate(trial_np)},
                        **{f"H{i}": np.asarray(A) for i, A in enumerate(H)},
                        **{f"q{i}": np.asarray(q) for i, q in enumerate(trial_charges)})
    print(f"saved the DMRG trial to {path}", flush=True)


def run_qmc(state, data, run_blocks, *, n_eql, n_blocks, n_walkers, n_steps, snapshots=None):
    """run_qmc_fixed_chunks with an explicit AOT compile (timed) and throughput.
    With snapshots (a WalkerSnapshots), one block per call and every block's walkers saved.
    Returns (mean, stderr, energies, weights, collapsed_after_block, timing)."""
    chunk = 1 if snapshots is not None else math.gcd(n_eql, n_blocks)
    total = n_eql + n_blocks
    t0 = time.perf_counter()
    compiled = run_blocks.lower(state, data, n_blocks=chunk).compile()
    compile_seconds = time.perf_counter() - t0
    print(f"compiled run_blocks ({chunk} blocks) in {compile_seconds:.1f} s; memory {compiled_memory(compiled)}",
          flush=True)

    energies, weights, collapsed, chunk_times = [], [], None, []
    if snapshots is not None:
        snapshots.add_snapshot(state)  # snapshot 0, read before the first call donates the state
    start = time.perf_counter()
    for done in range(chunk, total + 1, chunk):
        state, scalars = compiled(state, data)
        if snapshots is not None:
            snapshots.add_block(state, scalars)
        e, w = np.asarray(scalars["energy"]), np.asarray(scalars["weight"])
        chunk_times.append((done, time.perf_counter() - start))
        energies.extend(e.tolist())
        weights.extend(w.tolist())
        print(f"[{'eql' if done <= n_eql else 'blk'} {done:4d}/{total}]  E_chunk {np.sum(e * w) / np.sum(w):14.10f}"
              f"  W {w.mean():12.6e}  nodes {int(state.node_encounters):10d}"
              f"  t {time.perf_counter() - start:8.1f} s", flush=True)
        if not w[-1] > 0.0:
            collapsed = done
            print(f"\nPopulation collapsed: total weight is zero after block {done}. Stopping.", flush=True)
            break

    seconds = chunk_times[-1][1] if chunk_times else 0.0
    rate = n_walkers * n_steps * (chunk_times[-1][0] if chunk_times else 0) / max(seconds, 1e-300)
    timing = dict(compile_seconds=compile_seconds, run_seconds=seconds, walker_steps_per_s=rate,
                  peak_bytes=peak_device_bytes(), **compiled_memory(compiled))
    print(f"throughput {rate:.1f} walker-steps/s over {seconds:.1f} s (compile excluded)", flush=True)
    if collapsed is not None:
        return float("nan"), float("nan"), np.asarray(energies), np.asarray(weights), collapsed, timing
    sampled = np.column_stack((energies[n_eql:], weights[n_eql:]))
    clean, _ = reject_outliers(sampled, obs=0)
    print(f"\nRejected {len(sampled) - len(clean)} outlier blocks.\n\nFinal blocking analysis:")
    stats = blocking_analysis_ratio(np.asarray(clean[:, 0]), np.asarray(clean[:, 1]), print_q=True)
    return stats["mu"], stats["se_star"], np.asarray(energies), np.asarray(weights), None, timing


def save_result(path, record):
    if not path:
        return
    with Path(path).open("a") as stream:
        stream.write(json.dumps(record) + "\n")


# ============================================================================
# Setup and main
# ============================================================================

class Setup(NamedTuple):
    cfg: Config
    system: System
    ham: HamHubbard
    ops: GpuOps
    params: QmcParams
    trial_data: UhfTrial
    plans: tuple
    bonds: tuple
    n_chunks: int
    energy_chunks: int
    memory: dict
    info: dict
    references: tuple  # plan reference determinants (Ra, Rb)
    trial: tuple  # (dense trial tensors, their labels)
    htrial: tuple = ()  # compressed H|trial> tensors the energy uses, if kept for export (mps_cpmc_2d_gpu)
    rdm1: np.ndarray | None = None  # (2, L, L) trial rdm1 the "natural" plan reference and walker start use


def resolve_backend_options(cfg: Config):
    cpu = jax.default_backend() == "cpu"
    linalg = cfg.linalg if cfg.linalg != "auto" else ("native" if cpu else "batched")
    linalg = "batched" if linalg == "eigh" else linalg  # the earlier name
    walker_qr = cfg.walker_qr if cfg.walker_qr != "auto" else ("native" if cpu else "cholesky")
    return linalg, walker_qr


def build(cfg: Config = CFG, verbose=True) -> Setup:
    """Everything up to the initial state: trial, plans, compiled circuit, device
    data and the chunking. main() runs it; the benchmark reuses it."""
    say = print if verbose else (lambda *a, **k: None)
    if cfg.compile_cache:
        jax.config.update("jax_compilation_cache_dir", cfg.compile_cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    device = jax.devices()[0]
    linalg, walker_qr = resolve_backend_options(cfg)
    say(f"jax {jax.__version__}, backend {jax.default_backend()}, device {device.device_kind}; "
        f"linalg={linalg}, walker_qr={walker_qr}, energy={cfg.energy}", flush=True)

    h1 = hopping_matrix(cfg.L, cfg.hopping)
    ham = HamHubbard(h1=jnp.asarray(h1), u=cfg.interaction)
    system = System(norb=cfg.L, nelec=(cfg.n_up, cfg.n_down), walker_kind="unrestricted")
    for option in ("plan_reference", "walker_start"):
        if getattr(cfg, option) not in ("rhf", "natural"):
            raise ValueError(f"{option} must be 'rhf' or 'natural'")
    _, orbitals = np.linalg.eigh(h1)
    Ca, Cb = orbitals[:, :cfg.n_up].copy(), orbitals[:, :cfg.n_down].copy()
    say(f"L={cfg.L} ({cfg.n_up},{cfg.n_down}), U={cfg.interaction}")

    trial_np, trial_charges, dmrg_energy = load_or_run_trial(cfg)
    sector_weight, trial_rdm1 = 1.0, None
    if cfg.trial_rotation:
        if cfg.natural_rdm1 == "before":
            trial_rdm1 = np.stack(one_rdm(trial_np))  # the DMRG trial's, before the rotation
        elif cfg.natural_rdm1 != "after":
            raise ValueError("natural_rdm1 must be 'after' or 'before'")
        from trot.trial.mps import make_mps_trial, rotate_spin, spin_rotation_y
        from trot.trial.mps_rotation import make_rotated_mps_trial

        R, nelec = spin_rotation_y(cfg.trial_rotation), (cfg.n_up, cfg.n_down)
        if cfg.rotated_trial == "projected":
            rotated = make_mps_trial(rotate_spin(trial_np, R), nelec=nelec)
            trial_np, trial_charges = [np.asarray(A) for A in rotated.tensors], rotated.charge_arrays()
            sector_weight = rotated.sector_weight
            say(f"trial rotated by {cfg.trial_rotation:g} deg about y and projected onto ({cfg.n_up},{cfg.n_down}): "
                f"sector weight {sector_weight:.6f}, bonds max {max(rotated.bond_dims)}")
        elif cfg.rotated_trial == "as_is":
            rotated = make_rotated_mps_trial(trial_np, R, nelec=nelec)
            trial_np, trial_charges = trial_labels(rotated)
            sector_weight = None  # not computed: nothing is projected
            say(f"trial rotated by {cfg.trial_rotation:g} deg about y, used as it is (particle-number labels): "
                f"bonds max {max(rotated.bond_dims)}")
        else:
            raise ValueError("rotated_trial must be 'projected' or 'as_is'")
        if trial_rdm1 is None:
            trial_rdm1 = np.asarray(rotated.rdm1)
        say(f"natural orbitals (plan reference, bond plans, walker start) from the trial's rdm1 "
            f"{cfg.natural_rdm1} the rotation")
    trial = tuple(jnp.asarray(A) for A in trial_np)
    np.testing.assert_allclose(float(contract_real(trial, trial)), 1.0, atol=1e-10)
    mpo = hubbard_mpo(cfg.L, cfg.hopping, cfg.interaction)
    htrial_qn = compress_mps_qn(*apply_mpo_qn(mpo, trial_np, trial_charges))
    Htrial = tuple(jnp.asarray(A) for A in htrial_qn[0])
    trial_energy = float(contract_real(Htrial, trial) / contract_real(trial, trial))
    say(f"DMRG Davidson energy={dmrg_energy:.12f}; dense-MPS expectation={trial_energy:.12f}")
    say("trial bonds:", [A.shape[0] for A in trial_np] + [trial_np[-1].shape[-1]])
    say("H|trial> compressed bonds (charge-labelled):", [A.shape[0] for A in htrial_qn[0]] + [1])

    determinants = {"rhf": (Ca, Cb)}
    if "natural" in (cfg.plan_reference, cfg.walker_start):
        if trial_rdm1 is None:
            trial_rdm1 = np.stack(one_rdm(trial_np))
            np.testing.assert_allclose(np.trace(trial_rdm1, axis1=1, axis2=2), [cfg.n_up, cfg.n_down], atol=1e-8)
        else:  # a rotated trial's <N_up> - <N_dn> is cos(beta) (N_up - N_dn) before the projection
            np.testing.assert_allclose(np.trace(trial_rdm1, axis1=1, axis2=2).sum(), cfg.n_up + cfg.n_down, atol=1e-8)
        gamma_a, gamma_b = trial_rdm1
        Na, occupations = natural_orbitals(gamma_a, cfg.n_up)
        Nb, _ = natural_orbitals(gamma_b, cfg.n_down)
        determinants["natural"] = (Na, Nb)
        say(f"trial natural orbitals: alpha gap n_N - n_N+1 = {occupations[cfg.n_up - 1] - occupations[cfg.n_up]:.3f}")

    Ra, Rb = determinants[cfg.plan_reference]
    plan_a = make_orbital_plan(Ra, cfg.orbital_plan, cfg.occupation_tolerance)
    plan_b = make_orbital_plan(Rb, cfg.orbital_plan, cfg.occupation_tolerance)
    gates = lambda plan: int(plan.block_sizes.sum() - len(plan.block_sizes))
    say(f"plan reference: {cfg.plan_reference} determinant; gates {gates(plan_a)}, {gates(plan_b)}")

    Sa, Sb = determinants[cfg.walker_start]
    trial_data = UhfTrial(mo_coeff_a=jnp.asarray(Sa), mo_coeff_b=jnp.asarray(Sb))
    Pa, Pb = Sa @ Sa.T, Sb @ Sb.T
    initial_energy = float(np.sum(h1 * (Pa + Pb)) + cfg.interaction * np.diag(Pa) @ np.diag(Pb))

    def plan_fidelity(S, plan):
        _, rows = channel_angles(S, plan, xp=np)
        return float(np.linalg.det(np.stack([rows[i] for i in np.flatnonzero(plan.occupation)]))) ** 2
    say(f"walkers start from the {cfg.walker_start} determinant, E={initial_energy:.12f}; orbital-plan infidelity "
        f"{1 - plan_fidelity(Sa, plan_a):.1e}, {1 - plan_fidelity(Sb, plan_b):.1e}")

    bond_a = bond_b = None
    if cfg.walker_channel_chi is not None or cfg.walker_cutoff:
        bond_a = plan_bonds(Ra, plan_a, cfg.walker_channel_chi, cfg.walker_cutoff)
        bond_b = plan_bonds(Rb, plan_b, cfg.walker_channel_chi, cfg.walker_cutoff)
        say(f"walker truncation reference discarded weights: {bond_a.reference_discarded_weight:.3e}, "
            f"{bond_b.reference_discarded_weight:.3e}")

    prop_ctx = _build_prop_ctx(ham, cfg.dt)
    htrial = htrial_qn if cfg.energy == "blocked" else tuple(
        jnp.asarray(A) for A in compress_mps(apply_mpo(mpo, trial_np)))
    ops = make_gpu_ops(plan_a, plan_b, bond_a, bond_b, trial_np, trial_charges, htrial, prop_ctx,
                       linalg=linalg, walker_qr=walker_qr, spin_batch=cfg.spin_batch, energy=cfg.energy)

    circuit = circuit_stats(ops.converter.circuits[0])
    qa, qb = ops.converter.charges
    say("walker channel bonds:", [len(q) for q in qa])
    say(f"conversion circuit: {circuit}; spin-batched={ops.converter.spin_batched}")
    if linalg == "batched" and not ops.converter.spin_batched:
        say("note: spins converted separately (their orbital plans differ in particle number or gate sequence)")
    if circuit["eigh_over_32"] and linalg == "batched":
        say(f"note: {circuit['eigh_over_32']} factorisations exceed 32x32; check gpu_linalg_bench.py for "
            "whether batched eigh stays batched beyond cuSOLVER's Jacobi limit")
    say("overlap plan:", ops.overlap_plan.stats)
    if ops.energy_plan is not None:
        say("energy plan: ", ops.energy_plan.stats)

    memory = memory_model(ops)
    limit = device_bytes_limit()
    budget = None if limit is None else cfg.mem_fraction * limit - memory["data_bytes"]
    n_chunks = cfg.n_chunks or choose_chunks(cfg.n_walkers, memory["step_bytes_per_walker"], budget)
    energy_chunks = cfg.n_chunks or choose_chunks(cfg.n_walkers, memory["energy_bytes_per_walker"], budget)
    if cfg.n_walkers % n_chunks or cfg.n_walkers % energy_chunks:
        raise ValueError("n_chunks must divide n_walkers")
    say(f"memory model: {memory}; device limit {limit}; n_chunks {n_chunks} (energy {energy_chunks})")

    params = QmcParams(dt=cfg.dt, n_walkers=cfg.n_walkers, n_prop_steps=cfg.n_steps,
                       n_blocks=cfg.n_blocks, n_eql_blocks=cfg.n_equilibration,
                       weight_floor=cfg.weight_floor, seed=cfg.seed, n_chunks=n_chunks)
    info = dict(dmrg_energy=dmrg_energy, trial_energy=trial_energy, initial_energy=initial_energy,
                trial_sector_weight=sector_weight, trial_bonds_max=max(A.shape[0] for A in trial_np),
                gates=gates(plan_a), circuit=circuit, linalg=linalg, walker_qr=walker_qr,
                spin_batched=ops.converter.spin_batched, device=device.device_kind,
                backend=jax.default_backend(), walker_d4_chi=max(len(x) * len(y) for x, y in zip(qa, qb)),
                overlap_plan=ops.overlap_plan.stats)
    return Setup(cfg, system, ham, ops, params, trial_data, (plan_a, plan_b), (bond_a, bond_b),
                 n_chunks, energy_chunks, memory, info, (Ra, Rb), (trial_np, trial_charges), rdm1=trial_rdm1)


def main(cfg=CFG):
    setup = build(cfg)
    ops, params = setup.ops, setup.params

    start = time.perf_counter()
    state, probe = init_state(ops, setup.system, setup.trial_data, params)
    print(f"initial overlap {float(probe['overlap']):.6e}, local energy {float(probe['energy']):.10f} "
          f"({time.perf_counter() - start:.1f} s incl. compile)", flush=True)
    if cfg.self_check:
        errors = conversion_self_check(ops, setup.plans, setup.bonds, probe)
        print(f"self-check vs NumPy channel_mps: relative error alpha {errors[0]:.1e}, beta {errors[1]:.1e}")
        if max(errors) > 1e-6:
            raise AssertionError(f"device conversion disagrees with the NumPy reference: {errors}")

    start_det = (np.asarray(setup.trial_data.mo_coeff_a), np.asarray(setup.trial_data.mo_coeff_b))
    if cfg.trial_export:
        export_trial(cfg.trial_export, cfg, *setup.trial, setup.info["dmrg_energy"], setup.references, start_det)

    logger = make_block_logger(cfg.block_log, cfg.n_equilibration, cfg.tag) if cfg.block_log else None
    snapshots = None
    if cfg.walker_snapshots:
        # the notebooks' config keys (N_PROP, DT, N_EQL, ...), and this run's settings
        config = dict(L=cfg.L, N_UP=cfg.n_up, N_DN=cfg.n_down, T=cfg.hopping, U=cfg.interaction,
                      N_WALKERS=cfg.n_walkers, N_EQL=cfg.n_equilibration, N_BLOCKS=cfg.n_blocks, N_PROP=cfg.n_steps,
                      DT=cfg.dt, SEED=cfg.seed, DMRG_CHI_T=cfg.trial_chi, DMRG_SWEEPS=cfg.dmrg_sweeps,
                      CHI_PROP=cfg.walker_channel_chi, E_DMRG=setup.info["dmrg_energy"],
                      E_TRIAL=setup.info["trial_energy"], plan_reference=cfg.plan_reference,
                      walker_start=cfg.walker_start, orbital_plan=cfg.orbital_plan, EPS=cfg.occupation_tolerance,
                      trial_rotation=cfg.trial_rotation, rotated_trial=cfg.rotated_trial,
                      natural_rdm1=cfg.natural_rdm1,
                      trial_file=cfg.trial_export, tag=cfg.tag, module="mps_cpmc_gpu", device=setup.info["device"])
        snapshots = WalkerSnapshots(cfg.walker_snapshots, cfg.n_equilibration + cfg.n_blocks + 1, cfg.n_walkers,
                                    cfg.L, cfg.n_up, cfg.n_down, config)
    block_fn = make_block(ops, params, setup.n_chunks, setup.energy_chunks, record_comb=snapshots is not None)
    run_blocks = make_run_blocks(block_fn, logger)

    start = time.perf_counter()
    mean, error, block_energies, block_weights, collapsed, timing = run_qmc(
        state, ops.data, run_blocks, n_eql=cfg.n_equilibration, n_blocks=cfg.n_blocks,
        n_walkers=cfg.n_walkers, n_steps=cfg.n_steps, snapshots=snapshots)
    elapsed = time.perf_counter() - start

    scalar = lambda x: None if x is None else float(x)
    print(f"CPMC energy = {scalar(mean)} +/- {scalar(error)}; elapsed={elapsed:.1f} s")
    if snapshots is not None:
        snapshots.finish(dict(E_CPMC=scalar(mean), E_CPMC_ERR=scalar(error), collapsed_after_block=collapsed),
                         dict(reference_up=np.asarray(setup.references[0]),
                              reference_dn=np.asarray(setup.references[1]),
                              start_up=start_det[0], start_dn=start_det[1]))
    record = asdict(cfg)
    info = dict(setup.info)
    record.update(
        kind="mps_gpu", initial_energy=info.pop("initial_energy"), dmrg_energy=info.pop("dmrg_energy"),
        trial_energy=info.pop("trial_energy"), cpmc_energy=scalar(mean), cpmc_error=scalar(error),
        seconds=elapsed, collapsed_after_block=collapsed, n_chunks_used=setup.n_chunks,
        energy_chunks=setup.energy_chunks, memory_model=setup.memory, **timing, **info)
    save_result(cfg.result_json, record)


if __name__ == "__main__":
    main()
