"""CPMC with a DMRG trial for the square-lattice Hubbard model, built for one GPU.

mps_cpmc_2d.py runs mps_cpmc_new.py on an Lx x Ly lattice; this file does the same
for mps_cpmc_gpu.py. None of mps_cpmc_gpu's device code assumes a chain (the compiled
conversion circuit, the factorized charge-blocked contractions, the half-step scan,
the measurement block, the walker chunking), so all of it is reused unchanged. Only
the host setup differs from mps_cpmc_gpu.build:

* h1 is square_hopping_matrix (site x*Ly + y, each side open, periodic or
  antiperiodic) and the trial is mps_cpmc_2d's DMRG, with its bond-dimension ramp.
* H|trial> is the general-h1 MPO applied with exact (N_alpha, N_beta) bond labels
  (trial_times_h), then compressed per charge sector by compress_mps_qn, as the chain
  H|trial> is in mps_cpmc_gpu. mps_cpmc_2d contracts it uncompressed, and so does
  compress_htrial=False.
* The trial cache is keyed by the lattice, its boundaries and the DMRG schedule.
* From 6x6 on the dense H|trial> does not fit in memory: cache_htrial=True makes it in
  block form (charge-allowed blocks only, the same arithmetic as compress_mps_qn) and
  caches it with the trial's rdm1. prepare_sq_trial.py makes trial and H|trial> once on
  the CPU; GPU runs with the same trial_cache load both. dmrg_mpo="terms" builds the
  DMRG MPO from the Hubbard terms instead of pyblock3's quantum-chemistry MPO.

The lattice code is copied from mps_cpmc_2d instead of imported: importing it imports
mps_cpmc_new and pyblock3, and GPU jobs depend on neither (a cached trial runs without
pyblock3).

    python mps_cpmc_2d_gpu.py cpmc Lx=4 Ly=4 n_up=8 n_down=8 trial_chi=256 trial_cache=trial_cache
    sbatch --export=ALL,TARGET=mps_cpmc_2d_gpu.py run_mps_gpu.sh cpmc Lx=4 Ly=4 n_up=8 n_down=8 trial_chi=256

Arguments are Config fields as key=value, as in mps_cpmc_2d. The leading cpmc is
optional. mps_cpmc_2d's blocked_energy=False is energy=dense here. DMRG references
(mps_cpmc_2d's dmrg mode) run on the host and stay in mps_cpmc_2d.py.
"""
from __future__ import annotations

import ast
import itertools
import os
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from trot.core.system import System
from trot.gmps import mps_cpmc_gpu as g
from trot.ham.hubbard import HamHubbard
from trot.prop.hubbard_cpmc_ops import _build_prop_ctx
from trot.prop.types import QmcParams
from trot.trial.uhf import UhfTrial


@dataclass(frozen=True)
class Config:
    Lx: int = 4
    Ly: int = 2
    boundary_x: str = "open"  # open, periodic, antiperiodic
    boundary_y: str = "open"
    n_up: int = 4
    n_down: int = 4
    hopping: float = 1.0
    interaction: float = 4.0
    trial_chi: int = 64
    # Bond-dimension ramp, one entry per sweep, with the last entry repeating. If set it
    # replaces trial_chi, and noise stays on until the ramp is done.
    dmrg_bdims: tuple[int, ...] = ()
    dmrg_sweeps: int = 14
    dmrg_tol: float = 1.0e-6
    dmrg_seed: int = 0
    # DMRG MPO: "qc" is pyblock3's build_qc_mpo (mps_cpmc_2d's; U as a general two-electron
    # tensor), "terms" is built from the hopping and U terms, as mps_cpmc_gpu does for the
    # chain (bond 2 + 4 Ly). Both are the same operator; "terms" trials get their own cache names.
    dmrg_mpo: str = "qc"
    # rank_exact is exact for every walker. The adaptive plan is only guaranteed for the
    # reference, which a degenerate Fermi shell makes arbitrary.
    orbital_plan: str = "rank_exact"
    occupation_tolerance: float = 1.0e-10
    walker_channel_chi: int | None = None
    walker_cutoff: float = 0.0
    plan_reference: str = "natural"  # natural, rhf
    walker_start: str = "natural"  # natural, rhf
    n_walkers: int = 1024
    n_blocks: int = 40
    n_equilibration: int = 15
    n_steps: int = 20
    dt: float = 0.01
    weight_floor: float = 1.0e-8
    seed: int = 1234
    # --- GPU layout (none of these change the algorithm), as in mps_cpmc_gpu ---
    n_chunks: int = 0  # walker chunks per half step; 0 = the fewest the memory model allows
    mem_fraction: float = 0.75  # share of the device memory the chunker may plan for
    linalg: str = "auto"  # batched (all sectors per op in one call per kind, GPUs), native (the original loop), auto
    walker_qr: str = "auto"  # cholesky (CholeskyQR2), native (Householder, trot's _qr), auto
    energy: str = "blocked"  # blocked (charge-labelled H|trial>), dense (original; small sizes)
    # Compress H|trial> per charge sector (relative cut 1e-13). False contracts the exact
    # product from trial_times_h, as mps_cpmc_2d does.
    compress_htrial: bool = True
    spin_batch: bool = True  # convert both spin channels as one batch when their circuits match
    self_check: bool = True  # check one device conversion against the NumPy original at start-up
    trial_cache: str = ""  # directory for cached DMRG trials (skips DMRG when present)
    # Read (or make and write) the compressed H|trial>, in block form, and the trial's rdm1
    # next to the cached trial (load_or_make_htrial). Needs trial_cache, compress_htrial and
    # energy="blocked"; the dense H|trial> of 6x6 and larger lattices does not fit in memory.
    cache_htrial: bool = False
    compile_cache: str = ""  # JAX persistent compilation cache directory
    result_json: str = ""
    block_log: str = ""  # if set, append every block's scalars here as it finishes
    walker_snapshots: str = ""  # if set, every block's walkers go here (.npz, see mps_cpmc_gpu.WalkerSnapshots)
    trial_export: str = ""  # if set, the DMRG trial goes here (see export_trial)
    tag: str = ""

    @property
    def n_sites(self) -> int:
        return self.Lx * self.Ly


CFG = Config()


# ============================================================================
# Lattice, MPO and DMRG trial (copied from mps_cpmc_2d)
# ============================================================================

def square_hopping_matrix(Lx, Ly, hopping, boundary_x="open", boundary_y="open"):
    """Nearest-neighbour hopping on an Lx x Ly lattice, site x*Ly + y.

    Bonds accumulate, so a periodic side of length 2 carries a doubled bond.
    """
    signs = {"open": 0.0, "periodic": 1.0, "antiperiodic": -1.0}
    if boundary_x not in signs or boundary_y not in signs:
        raise ValueError("boundaries must be 'open', 'periodic', or 'antiperiodic'")
    bonds = []
    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y
            if x + 1 < Lx:
                bonds.append((i, i + Ly, 1.0))
            elif Lx > 1 and signs[boundary_x]:
                bonds.append((i, y, signs[boundary_x]))
            if y + 1 < Ly:
                bonds.append((i, i + 1, 1.0))
            elif Ly > 1 and signs[boundary_y]:
                bonds.append((i, x * Ly, signs[boundary_y]))
    h1 = np.zeros((Lx * Ly, Lx * Ly))
    for i, j, sign in bonds:
        h1[i, j] -= sign * hopping
        h1[j, i] -= sign * hopping
    return h1


def lattice_hopping(cfg: Config) -> np.ndarray:
    return square_hopping_matrix(cfg.Lx, cfg.Ly, cfg.hopping, cfg.boundary_x, cfg.boundary_y)


def hubbard_mpo_from_h1(h1, interaction):
    """Hubbard MPO for any real symmetric one-body matrix, in the spatial local basis.

    A finite-state automaton. Channel 0 has placed nothing and the last channel holds a
    finished term. A hopping opened on site i travels in four channels, one per opening
    operator, and carries the Jordan-Wigner string Pa Pb to its partner site. Bonds are
    padded to a common width. A nearest-neighbour chain gives hubbard_mpo exactly.
    """
    h1 = np.asarray(h1)
    if np.iscomplexobj(h1) or not np.array_equal(h1, h1.T):
        raise ValueError("h1 must be real symmetric")
    n = len(h1)
    eye = np.eye(4)
    create_a = np.zeros((4, 4)); create_a[1, 0] = create_a[3, 2] = 1.0
    create_b = np.zeros((4, 4)); create_b[2, 0] = 1.0; create_b[3, 1] = -1.0
    annihilate_a, annihilate_b = create_a.T, create_b.T
    parity_a = np.diag([1.0, -1.0, 1.0, -1.0])
    parity_b = np.diag([1.0, 1.0, -1.0, -1.0])
    double = np.diag([0.0, 0.0, 0.0, 1.0])
    number = np.diag([0.0, 1.0, 1.0, 2.0])
    opening = (create_a @ parity_b, annihilate_a @ parity_b,
               parity_a @ create_b, parity_a @ annihilate_b)
    closing = (annihilate_a, create_a, annihilate_b, create_b)

    upper = np.triu(h1, 1) != 0
    reach = [np.flatnonzero(row).max(initial=i) for i, row in enumerate(upper)]
    slots = [{i: 1 + 4 * rank for rank, i in enumerate(i for i in range(b) if reach[i] >= b)}
             for b in range(n + 1)]
    D = 2 + 4 * max(map(len, slots))

    W = np.zeros((n, D, 4, 4, D))
    for k in range(n):
        left, right = slots[k], slots[k + 1]
        W[k, 0, :, :, 0] = W[k, -1, :, :, -1] = eye
        W[k, 0, :, :, -1] = interaction * double + h1[k, k] * number
        for op in range(4):
            if k in right:
                W[k, 0, :, :, right[k] + op] = opening[op]
            for i, slot in left.items():
                if h1[i, k]:
                    W[k, slot + op, :, :, -1] = h1[i, k] * closing[op]
                if i in right:
                    W[k, slot + op, :, :, right[i] + op] = parity_a @ parity_b
    return W


def build_dmrg_hamiltonian(cfg: Config, h1):
    n = cfg.n_sites
    g2 = np.zeros((n,) * 4)
    i = np.arange(n)
    g2[i, i, i, i] = cfg.interaction
    fcidump = g.FCIDUMP(pg="c1", n_sites=n, n_elec=cfg.n_up + cfg.n_down,
                        twos=cfg.n_up - cfg.n_down, ipg=0, h1e=h1, g2e=g2)
    return g.Hamiltonian(fcidump, flat=True)


def dmrg_schedule(cfg: Config):
    """Per-sweep bond dimensions and noises. Without a ramp this is mps_cpmc_new.run_dmrg's schedule."""
    if not cfg.dmrg_bdims:
        return [cfg.trial_chi] * cfg.dmrg_sweeps, [1.0e-5] * min(6, cfg.dmrg_sweeps) + [0.0]
    ramp = list(cfg.dmrg_bdims)
    if len(ramp) >= cfg.dmrg_sweeps:
        raise ValueError("dmrg_sweeps must exceed len(dmrg_bdims) so the final bond gets noiseless sweeps")
    bdims = ramp + [ramp[-1]] * (cfg.dmrg_sweeps - len(ramp))
    return bdims, [1.0e-5] * max(6, len(ramp)) + [0.0]


def hubbard_terms_mpo(hamiltonian, h1, interaction):
    """pyblock3 MPO of the Hubbard model from its operator terms: one hopping term per
    nonzero h1 entry and spin, one U n_up n_down per site. mps_cpmc_gpu.hubbard_dmrg_mpo
    for any h1. build_qc_mpo instead treats U as a general two-electron tensor, whose MPO
    has bond ~n^2/2 before compression; this one compresses to the nearest-neighbour
    bond, 2 + 4 Ly on the square lattice."""
    # pyblock3's flat term encoding (as in Hamiltonian.build_complex_qc_mpo):
    # operator index = OP * (0 for c+, 1 for c) + SITE * site + SPIN * spin, -1 pads.
    SPIN, SITE, OP = 1, 2, 16384
    C, D = 0 * OP, 1 * OP
    values, terms = [], []
    for i, j in zip(*np.nonzero(h1)):
        for s in (0, 1):
            values.append(h1[i, j])
            terms.append([C + i * SITE + s * SPIN, D + j * SITE + s * SPIN, -1, -1])
    for i in range(len(h1)):
        # n_up n_down = c+_up c+_down c_down c_up
        values.append(interaction)
        terms.append([C + i * SITE, C + i * SITE + SPIN, D + i * SITE + SPIN, D + i * SITE])
    gen = (np.array(values, dtype=np.float64), np.array(terms, dtype=np.int32))
    return hamiltonian.build_mpo(gen, cutoff=1.0e-12)


def build_dmrg_mpo(hamiltonian, cfg: Config):
    if cfg.dmrg_mpo == "qc":
        return hamiltonian.build_qc_mpo().compress(cutoff=1.0e-12)[0]
    if cfg.dmrg_mpo == "terms":
        return hubbard_terms_mpo(hamiltonian, lattice_hopping(cfg), cfg.interaction)
    raise ValueError("dmrg_mpo must be 'qc' or 'terms'")


def run_dmrg(hamiltonian, cfg: Config, iprint=-1):
    """Returns the MPS, the last Davidson energy, the per-sweep energies and <H> of the MPS.

    Davidson energies belong to the two-site wavefunction before truncation. Only the
    last value, from pyblock3's block-sparse MPO, is variational for the returned MPS.
    """
    np.random.seed(cfg.dmrg_seed)
    bdims, noises = dmrg_schedule(cfg)
    mpo = build_dmrg_mpo(hamiltonian, cfg)
    if iprint >= 0 and hasattr(mpo, "show_bond_dims"):
        print(f"DMRG MPO ({cfg.dmrg_mpo}) bonds: {mpo.show_bond_dims()}", flush=True)
    mps = hamiltonian.build_mps(bdims[0])
    result = g.MPE(mps, mpo, mps).dmrg(bdims=bdims, noises=noises, dav_thrds=[1.0e-10],
                                       iprint=iprint, n_sweeps=cfg.dmrg_sweeps, tol=cfg.dmrg_tol)
    energies = [float(e) for e in result.energies]
    mps_energy = float(g.MPE(mps, mpo, mps)[0:2].expectation) / float(mps @ mps)
    return mps, energies[-1], energies, mps_energy


def describe(cfg: Config, h1):
    """Print the lattice and the free-fermion gaps at the Fermi level."""
    eps = np.linalg.eigvalsh(h1)
    gap = lambda N: eps[N] - eps[N - 1] if 0 < N < len(eps) else float("inf")
    print(f"{cfg.Lx}x{cfg.Ly} lattice, boundaries x={cfg.boundary_x} y={cfg.boundary_y}, "
          f"({cfg.n_up},{cfg.n_down}), U={cfg.interaction}")
    print(f"free-fermion gap eps_N - eps_N-1: {gap(cfg.n_up):.3e}, {gap(cfg.n_down):.3e}"
          " (about 0 means the rhf determinant is an arbitrary pick in a degenerate shell)")


def densified_trial(cfg: Config, h1, iprint=-1):
    hamiltonian = build_dmrg_hamiltonian(cfg, h1)
    dmrg_mps, dmrg_energy, sweep_energies, mps_energy = run_dmrg(hamiltonian, cfg, iprint)
    trial_np, trial_charges = g.densify_with_charges(dmrg_mps, cfg.n_sites)
    return trial_np, trial_charges, dmrg_energy, sweep_energies, mps_energy


CHANNEL_CHARGE = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])  # c†a, ca, c†b, cb left open


def trial_times_h(W, trial_np, trial_charges):
    """H|trial>, uncompressed, with exact (N_alpha, N_beta) bond labels.

    Bond index (channel, trial) is labelled with the trial's label plus the charge that
    the channel's open operator has put left of the cut (the opening order of
    hubbard_mpo_from_h1). That lets <H trial|walker> use the blocked overlap machinery.
    Padding channels that a bond never uses are dropped.
    """
    n = len(trial_np)
    _, active, charges = _htrial_bonds(W, trial_charges)
    tensors = []
    for k in range(n):
        T = np.einsum("apqb,cqd->acpbd", W[k][active[k]][..., active[k + 1]], np.asarray(trial_np[k]))
        dl, cl, d, dr, cr = T.shape
        tensors.append(T.reshape(dl*cl, d, dr*cr))
    return tensors, charges


def _htrial_bonds(W, trial_charges):
    """trial_times_h's bond layout: each MPO channel's charge, the channels each bond
    keeps, and the (N_alpha, N_beta) labels of the flattened (channel, trial) bonds."""
    n, D = len(trial_charges) - 1, W.shape[1]
    delta = np.zeros((D, 2), int)
    delta[1:-1] = np.tile(CHANNEL_CHARGE, ((D - 2) // 4, 1))
    active = ([np.zeros(1, int)]
              + [np.flatnonzero(np.any(W[b - 1] != 0, axis=(0, 1, 2))) for b in range(1, n)]
              + [np.full(1, D - 1)])
    charges = tuple((delta[active[b]][:, None, :] + np.asarray(trial_charges[b])[None]).reshape(-1, 2)
                    for b in range(n + 1))
    return delta, active, charges


# ============================================================================
# Host: trial cache and export
# ============================================================================

BOUNDARY_CODE = {"open": "o", "periodic": "p", "antiperiodic": "a"}


def _trial_cache_file(cfg: Config):
    ramp = "-".join(map(str, cfg.dmrg_bdims)) if cfg.dmrg_bdims else str(cfg.trial_chi)
    name = (f"sq{cfg.Lx}x{cfg.Ly}{BOUNDARY_CODE[cfg.boundary_x]}{BOUNDARY_CODE[cfg.boundary_y]}"
            f"_n{cfg.n_up}-{cfg.n_down}_t{cfg.hopping:g}_U{cfg.interaction:g}_chi{ramp}"
            f"_sw{cfg.dmrg_sweeps}_tol{cfg.dmrg_tol:g}_seed{cfg.dmrg_seed}"
            f"{'' if cfg.dmrg_mpo == 'qc' else '_mpo' + cfg.dmrg_mpo}.npz")
    return Path(cfg.trial_cache) / name


def load_or_run_trial(cfg: Config, h1, iprint=-1):
    """mps_cpmc_2d's DMRG trial as dense tensors, (N_alpha, N_beta) bond labels, the
    last Davidson energy and pyblock3's <H> of the MPS.

    With cfg.trial_cache set the trial is read from, or written to, one npz file per
    (lattice, boundaries, filling, t, U, DMRG schedule, seed), so GPU jobs skip DMRG.
    The file keeps h1, and a trial whose h1 differs from this run's is refused.
    """
    n = cfg.n_sites
    path = _trial_cache_file(cfg) if cfg.trial_cache else None
    if path is not None and path.exists():
        with np.load(path) as data:
            if not np.array_equal(data["h1"], h1):
                raise ValueError(f"{path} was made for a different h1")
            tensors = [np.asarray(data[f"A{i}"]) for i in range(n)]
            charges = tuple(np.asarray(data[f"q{i}"]) for i in range(n + 1))
            energy, mps_energy = float(data["energy"]), float(data["mps_energy"])
        print(f"trial loaded from {path}")
        return tensors, charges, energy, mps_energy
    if g.MPE is None:
        raise ImportError("pyblock3 is needed to run DMRG (or point trial_cache at a cached trial)")
    tensors, charges, energy, sweep_energies, mps_energy = densified_trial(cfg, h1, iprint)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, energy=energy, mps_energy=mps_energy, sweep_energies=np.asarray(sweep_energies), h1=h1,
                 dmrg_mpo=cfg.dmrg_mpo,
                 **{f"A{i}": A for i, A in enumerate(tensors)}, **{f"q{i}": q for i, q in enumerate(charges)})
        os.replace(tmp, path)
        print(f"trial saved to {path}")
    return tensors, charges, energy, mps_energy


def export_trial(path, cfg, trial_np, trial_charges, dmrg_energy, references, start, htrial=()):
    """mps_cpmc_gpu.export_trial on the square lattice: T{i}, H{i}, q{i}, e_dmrg, gamma,
    reference_up/dn and start_up/dn in the same format, plus the lattice's h1.

    H{i} is the compressed H|trial> the run's energy used (htrial: build's charge-labelled
    compression for energy="blocked", the dense one for energy="dense"), so it is not
    built and compressed a second time. Without it (compress_htrial=False) the general-h1
    MPO is applied to the trial and dense-compressed here, as before. htrial=None
    (cache_htrial) writes no H{i}: the block-form H|trial> and the rdm1 are read from the
    H|trial> cache file, whose path is stored as htrial_file."""
    h1 = lattice_hopping(cfg)
    extra = {}
    if htrial is None:
        H, cache = [], _htrial_cache_file(cfg)
        with np.load(cache) as data:
            gamma = np.stack([data["gamma_a"], data["gamma_b"]])
        extra["htrial_file"] = str(cache)
    else:
        H = list(htrial) if len(htrial) else g.compress_mps(
            trial_times_h(hubbard_mpo_from_h1(h1, cfg.interaction), trial_np, trial_charges)[0])
        gamma = np.stack(g.one_rdm(trial_np))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # runs that share the trial (another walker bond) may export it at the same time
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, e_dmrg=dmrg_energy, gamma=gamma, h1=h1, **extra,
                        reference_up=np.asarray(references[0]), reference_dn=np.asarray(references[1]),
                        start_up=np.asarray(start[0]), start_dn=np.asarray(start[1]),
                        **{f"T{i}": np.asarray(A) for i, A in enumerate(trial_np)},
                        **{f"H{i}": np.asarray(A) for i, A in enumerate(H)},
                        **{f"q{i}": np.asarray(q) for i, q in enumerate(trial_charges)})
    os.replace(tmp, path)
    print(f"saved the DMRG trial to {path}", flush=True)


# ============================================================================
# Host: H|trial> in block form (6x6 and larger)
# ============================================================================
#
# trial_times_h and compress_mps_qn keep every tensor dense. The uncompressed bond of
# H|trial> is (2 + 4 Ly) chi_T and compression keeps about 4 Ly chi_T of it (4x4 at
# chi_T 256: 4608 -> 4092), so at 8x8 and chi_T 512 the dense tensors need ~600 GB.
# The block form keeps only charge-allowed blocks: site k is a dict {(q_left, p,
# q_right): block}, next to the same full label arrays as before. A block's rows and
# columns are the bond indices with those labels in increasing order (_charge_index's
# order), and a missing block is zero. The functions below repeat trial_times_h and
# compress_mps_qn on this form: the same bonds, labels, sector matrices and rank cut.

PHYSICAL_INDEX = {tuple(int(x) for x in q): p for p, q in enumerate(g.PHYSICAL_CHARGE)}


def _shift(q, dq):
    return (q[0] + int(dq[0]), q[1] + int(dq[1]))


def to_blocks(tensors, charges):
    """Dense charge-labelled MPS -> block form (blocks that are not all zero)."""
    out = []
    for k, A in enumerate(tensors):
        A = np.asarray(A)
        left, right = g._charge_index(charges[k]), g._charge_index(charges[k + 1])
        site = {}
        for ql, rows in left.items():
            for p, dq in enumerate(g.PHYSICAL_CHARGE):
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
        left, right = g._charge_index(charges[k]), g._charge_index(charges[k + 1])
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


def _htrial_layout(W, trial_charges):
    """_htrial_bonds plus, per bond, where channel j's trial indices of label q sit inside
    H|trial>'s label group delta_j + q, and each group's size. The flattened bond is
    channel-major, so a group lists channel 0's trial indices first, then channel 1's."""
    delta, active, charges = _htrial_bonds(W, trial_charges)
    layout = []
    for b, labels in enumerate(trial_charges):
        offset, size = {}, {}
        groups = g._charge_index(labels)
        for j, a in enumerate(active[b]):
            for q, indices in groups.items():
                L = _shift(q, delta[a])
                offset[(j, q)] = size.get(L, 0)
                size[L] = size.get(L, 0) + len(indices)
        layout.append((offset, size))
    return delta, active, charges, layout


def _htrial_terms(W, trial_blocks, delta, active, k):
    """Every (H|trial> block key, channel positions, trial block labels, MPO coefficient,
    trial block) that trial_times_h's site-k product sums."""
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
    """trial_times_h in block form: the same bonds, labels and entries (each entry is one
    MPO coefficient times one trial entry, as in the dense product), no dense tensors."""
    delta, active, charges, layout = _htrial_layout(W, trial_charges)
    tensors = []
    for k in range(len(trial_blocks)):
        (left_offset, left_size), (right_offset, right_size) = layout[k], layout[k + 1]
        site = {}
        for key, jl, jr, qc, qd, coefficient, B in _htrial_terms(W, trial_blocks, delta, active, k):
            if key not in site:
                L, p, R = key
                if _shift(L, g.PHYSICAL_CHARGE[p]) != R:
                    raise AssertionError(f"an MPO channel does not conserve the charge at site {k}")
                site[key] = np.zeros((left_size[L], right_size[R]))
            r0, c0 = left_offset[(jl, qc)], right_offset[(jr, qd)]
            site[key][r0:r0 + B.shape[0], c0:c0 + B.shape[1]] += coefficient * B
        tensors.append(site)
    return tensors, charges


def compress_blocks_qn(tensors, charges, relative_tolerance=1.0e-13, consume=False):
    """compress_mps_qn in block form. Every sector matrix is assembled with its rows and
    columns in compress_mps_qn's order (zero rows and columns included), so the QR, the
    SVD and the rank cut are the same. consume=True replaces the input list's entries as
    the sweeps go, so the uncompressed tensors are freed site by site."""
    A = tensors if consume else [dict(site) for site in tensors]
    Q = [np.asarray(q, int).reshape(-1, 2) for q in charges]
    d = len(g.PHYSICAL_CHARGE)
    for i in range(len(A) - 1):
        left = g._charge_index(Q[i])
        rows = (Q[i][:, None, :] + g.PHYSICAL_CHARGE[None]).reshape(-1, 2)
        by_right, following = {}, {}
        for (ql, p, qr), X in A[i].items():
            by_right.setdefault(qr, []).append((ql, p, X))
        for (ql, p, qr), X in A[i + 1].items():
            following.setdefault(ql, []).append((p, qr, X))
        site, after, labels = {}, {}, []
        for c, r, k in g._label_sectors(rows, Q[i + 1]):
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
        right = g._charge_index(Q[i + 1])
        columns = (Q[i + 1][None, :, :] - g.PHYSICAL_CHARGE[:, None, :]).reshape(-1, 2)
        preceding = {}
        for (ql, p, qr), X in A[i - 1].items():
            preceding.setdefault(qr, []).append((ql, p, X))
        factors = []
        for c, r, k in g._label_sectors(Q[i], columns):
            parts, width = [], 0  # columns p * Dr + r in increasing order: p-major
            for p, dq in enumerate(g.PHYSICAL_CHARGE):
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
    """Upper bound of the device memory make_factorized_plan gives H|trial>'s blocks:
    every label shared with the walkers, per site transitions x largest left group x
    largest right group (the plan pads each bond to its largest group)."""
    groups = [g._charge_index(q) for q in charges]
    total = 0
    for k in range(len(charges) - 1):
        transitions = sum(_shift(c, dq) in groups[k + 1] for c in groups[k] for dq in g.PHYSICAL_CHARGE)
        total += transitions * max(map(len, groups[k].values())) * max(map(len, groups[k + 1].values()))
    return 8 * total


def _htrial_cache_file(cfg: Config):
    path = _trial_cache_file(cfg)
    return path.with_name(f"{path.stem}_htrial.npz")


def _pack_blocks(blocks, charges):
    """(name, array) pairs of the npz: the bond labels q{b} and, per site, K{k} (each
    block's q_left, p, q_right and shape) and V{k} (the blocks raveled one after another).
    A generator, so only one site's V{k} is copied at a time."""
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


def load_or_make_htrial(cfg: Config, h1, trial_np, trial_charges, mps_energy, say=print):
    """The compressed, charge-labelled H|trial> in block form (tensors, labels) and a dict
    with <trial|H|trial>, the uncompressed bonds and the trial's rdm1 (gamma_a, gamma_b).

    With cfg.trial_cache set they are read from, or written to, <trial file>_htrial.npz:
    made once on the CPU (prepare_sq_trial.py) for every GPU run of that trial. A file
    made for another h1 or another trial (its pyblock3 energy differs) is refused."""
    n = cfg.n_sites
    path = _htrial_cache_file(cfg) if cfg.trial_cache else None
    if path is not None and path.exists():
        with np.load(path) as data:
            if not np.array_equal(data["h1"], h1) or float(data["trial_mps_energy"]) != mps_energy:
                raise ValueError(f"{path} was made for another trial")
            blocks, charges = _unpack_blocks(data, n)
            info = dict(trial_energy=float(data["trial_energy"]),
                        uncompressed_bonds=[int(x) for x in data["uncompressed_bonds"]],
                        gamma=(np.asarray(data["gamma_a"]), np.asarray(data["gamma_b"])))
        say(f"H|trial> loaded from {path}", flush=True)
        return (blocks, charges), info

    clock, seconds = time.perf_counter, {}
    W = hubbard_mpo_from_h1(h1, cfg.interaction)
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
    say(f"<trial|H|trial> {energy:.12f}, pyblock3 {mps_energy:.12f}, difference {energy - mps_energy:.1e}", flush=True)
    if abs(energy - mps_energy) > 1e-8 * max(1.0, abs(mps_energy)):
        # two MPOs of the same Hamiltonian (hubbard_mpo_from_h1 here, the DMRG's in pyblock3)
        raise AssertionError(f"<trial|H|trial> = {energy} but pyblock3 gives {mps_energy}; H|trial> not saved")
    start = clock()
    gamma = g.one_rdm(trial_np)
    seconds["rdm1"] = clock() - start
    say("H|trial> times: " + ", ".join(f"{k} {v:.1f} s" for k, v in seconds.items()), flush=True)
    if path is not None:
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        header = dict(h1=h1, trial_mps_energy=mps_energy, trial_energy=energy, relative_tolerance=1.0e-13,
                      uncompressed_bonds=np.asarray(exact_bonds), gamma_a=gamma[0], gamma_b=gamma[1])
        _save_npz(tmp, itertools.chain(header.items(), _pack_blocks(blocks, charges)))
        os.replace(tmp, path)
        say(f"H|trial> saved to {path}", flush=True)
    return (blocks, charges), dict(trial_energy=energy, uncompressed_bonds=exact_bonds, gamma=gamma)


# ============================================================================
# Setup and main
# ============================================================================

def build(cfg: Config = CFG, verbose=True) -> g.Setup:
    """mps_cpmc_gpu.build on the square lattice: trial, plans, compiled circuit, device
    data and the chunking, everything up to the initial state."""
    say = print if verbose else (lambda *a, **k: None)
    for option in ("plan_reference", "walker_start"):
        if getattr(cfg, option) not in ("rhf", "natural"):
            raise ValueError(f"{option} must be 'rhf' or 'natural'")
    if cfg.compile_cache:
        jax.config.update("jax_compilation_cache_dir", cfg.compile_cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    device = jax.devices()[0]
    linalg, walker_qr = g.resolve_backend_options(cfg)
    host_cpus = len(os.sched_getaffinity(0))
    say(f"jax {jax.__version__}, backend {jax.default_backend()}, device {device.device_kind}; "
        f"linalg={linalg}, walker_qr={walker_qr}, energy={cfg.energy}; host CPUs {host_cpus} "
        f"(OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')})", flush=True)
    seconds, clock = {}, time.perf_counter  # host setup time per stage, for the log and the result record

    h1 = lattice_hopping(cfg)
    n = cfg.n_sites
    ham = HamHubbard(h1=jnp.asarray(h1), u=cfg.interaction)
    system = System(norb=n, nelec=(cfg.n_up, cfg.n_down), walker_kind="unrestricted")
    _, orbitals = np.linalg.eigh(h1)
    Ca, Cb = orbitals[:, :cfg.n_up].copy(), orbitals[:, :cfg.n_down].copy()
    if verbose:
        describe(cfg, h1)

    start = clock()
    trial_np, trial_charges, dmrg_energy, mps_energy = load_or_run_trial(cfg, h1)
    seconds["trial"] = clock() - start
    trial = tuple(jnp.asarray(A) for A in trial_np)
    np.testing.assert_allclose(float(g.contract_real(trial, trial)), 1.0, atol=1e-10)

    start = clock()
    W = hubbard_mpo_from_h1(h1, cfg.interaction)
    mpo_bond = W.shape[1]
    bonds_of = lambda tensors: [A.shape[0] for A in tensors] + [tensors[-1].shape[-1]]
    gamma = None
    if cfg.cache_htrial:
        if not (cfg.trial_cache and cfg.compress_htrial and cfg.energy == "blocked"):
            raise ValueError("cache_htrial needs trial_cache, compress_htrial=True and energy='blocked'")
        # block form, from the cache when prepare_sq_trial.py has made it
        htrial_qn, cached = load_or_make_htrial(cfg, h1, trial_np, trial_charges, mps_energy, say)
        trial_energy, exact_bonds, gamma = cached["trial_energy"], cached["uncompressed_bonds"], cached["gamma"]
    else:
        htrial_exact = trial_times_h(W, trial_np, trial_charges)
        htrial_qn = g.compress_mps_qn(*htrial_exact) if cfg.compress_htrial else htrial_exact
        Htrial = tuple(jnp.asarray(A) for A in htrial_qn[0])
        trial_energy = float(g.contract_real(Htrial, trial) / g.contract_real(trial, trial))
        exact_bonds = bonds_of(htrial_exact[0])
    seconds["htrial"] = clock() - start
    say(f"DMRG Davidson energy={dmrg_energy:.12f} (two-site, before truncation); trial energy: "
        f"pyblock3 {mps_energy:.12f}, dense MPO {trial_energy:.12f}")
    say("trial bonds:", bonds_of(trial_np))
    say(f"H|trial> bonds (MPO bond {mpo_bond}): uncompressed {exact_bonds}"
        + (f"; charge-labelled compressed {[len(q) for q in htrial_qn[1]]}" if cfg.compress_htrial else ""))

    start = clock()
    determinants = {"rhf": (Ca, Cb)}
    if "natural" in (cfg.plan_reference, cfg.walker_start):
        gamma_a, gamma_b = gamma if gamma is not None else g.one_rdm(trial_np)
        np.testing.assert_allclose([np.trace(gamma_a), np.trace(gamma_b)], [cfg.n_up, cfg.n_down], atol=1e-8)
        Na, occupations = g.natural_orbitals(gamma_a, cfg.n_up)
        Nb, _ = g.natural_orbitals(gamma_b, cfg.n_down)
        determinants["natural"] = (Na, Nb)
        if cfg.n_up < n:
            say(f"trial natural orbitals: alpha gap n_N - n_N+1 = {occupations[cfg.n_up - 1] - occupations[cfg.n_up]:.3f}")
    seconds["natural_orbitals"] = clock() - start

    Ra, Rb = determinants[cfg.plan_reference]
    plan_a = g.make_orbital_plan(Ra, cfg.orbital_plan, cfg.occupation_tolerance)
    plan_b = g.make_orbital_plan(Rb, cfg.orbital_plan, cfg.occupation_tolerance)
    gates = lambda plan: int(plan.block_sizes.sum() - len(plan.block_sizes))
    say(f"plan reference: {cfg.plan_reference} determinant; gates {gates(plan_a)}, {gates(plan_b)}")

    Sa, Sb = determinants[cfg.walker_start]
    trial_data = UhfTrial(mo_coeff_a=jnp.asarray(Sa), mo_coeff_b=jnp.asarray(Sb))
    Pa, Pb = Sa @ Sa.T, Sb @ Sb.T
    initial_energy = float(np.sum(h1 * (Pa + Pb)) + cfg.interaction * np.diag(Pa) @ np.diag(Pb))

    def plan_fidelity(S, plan):
        _, rows = g.channel_angles(S, plan, xp=np)
        return float(np.linalg.det(np.stack([rows[i] for i in np.flatnonzero(plan.occupation)]))) ** 2
    say(f"walkers start from the {cfg.walker_start} determinant, E={initial_energy:.12f}; orbital-plan infidelity "
        f"{1 - plan_fidelity(Sa, plan_a):.1e}, {1 - plan_fidelity(Sb, plan_b):.1e}")

    start = clock()
    bond_a = bond_b = None
    if cfg.walker_channel_chi is not None or cfg.walker_cutoff:
        bond_a = g.plan_bonds(Ra, plan_a, cfg.walker_channel_chi, cfg.walker_cutoff)
        bond_b = g.plan_bonds(Rb, plan_b, cfg.walker_channel_chi, cfg.walker_cutoff)
        say(f"walker truncation reference discarded weights: {bond_a.reference_discarded_weight:.3e}, "
            f"{bond_b.reference_discarded_weight:.3e}")
    seconds["bond_plans"] = clock() - start

    start = clock()
    prop_ctx = _build_prop_ctx(ham, cfg.dt)
    # htrial_mps: the compressed H|trial> the energy contracts, kept so export_trial need not
    # build and compress H|trial> a second time (empty: only the uncompressed product exists;
    # None: block form, which export_trial points to in its cache file)
    if cfg.energy == "blocked":
        htrial = htrial_qn
        htrial_mps = None if cfg.cache_htrial else tuple(htrial_qn[0]) if cfg.compress_htrial else ()
    else:
        htrial_mps = tuple(g.compress_mps(htrial_exact[0]))
        htrial = tuple(jnp.asarray(A) for A in htrial_mps)
    ops = g.make_gpu_ops(plan_a, plan_b, bond_a, bond_b, trial_np, trial_charges, htrial, prop_ctx,
                         linalg=linalg, walker_qr=walker_qr, spin_batch=cfg.spin_batch, energy=cfg.energy)
    seconds["ops"] = clock() - start

    circuit = g.circuit_stats(ops.converter.circuits[0])
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

    memory = g.memory_model(ops)
    limit = g.device_bytes_limit()
    budget = None if limit is None else cfg.mem_fraction * limit - memory["data_bytes"]
    n_chunks = cfg.n_chunks or g.choose_chunks(cfg.n_walkers, memory["step_bytes_per_walker"], budget)
    energy_chunks = cfg.n_chunks or g.choose_chunks(cfg.n_walkers, memory["energy_bytes_per_walker"], budget)
    if cfg.n_walkers % n_chunks or cfg.n_walkers % energy_chunks:
        raise ValueError("n_chunks must divide n_walkers")
    say(f"memory model: {memory}; device limit {limit}; n_chunks {n_chunks} (energy {energy_chunks})")

    params = QmcParams(dt=cfg.dt, n_walkers=cfg.n_walkers, n_prop_steps=cfg.n_steps,
                       n_blocks=cfg.n_blocks, n_eql_blocks=cfg.n_equilibration,
                       weight_floor=cfg.weight_floor, seed=cfg.seed, n_chunks=n_chunks)
    info = dict(dmrg_energy=dmrg_energy, mps_energy=mps_energy, trial_energy=trial_energy,
                initial_energy=initial_energy, gates=gates(plan_a), mpo_bond=mpo_bond,
                htrial_bond=max(len(q) for q in htrial_qn[1]), circuit=circuit, linalg=linalg, walker_qr=walker_qr,
                spin_batched=ops.converter.spin_batched, device=device.device_kind,
                backend=jax.default_backend(), walker_d4_chi=max(len(x) * len(y) for x, y in zip(qa, qb)),
                overlap_plan=ops.overlap_plan.stats, host_cpus=host_cpus,
                setup_seconds={k: round(v, 1) for k, v in seconds.items()})
    say("setup times: " + ", ".join(f"{k} {v:.1f} s" for k, v in seconds.items()), flush=True)
    return g.Setup(cfg, system, ham, ops, params, trial_data, (plan_a, plan_b), (bond_a, bond_b),
                   n_chunks, energy_chunks, memory, info, (Ra, Rb), (trial_np, trial_charges), htrial=htrial_mps)


def main(cfg: Config = CFG):
    """mps_cpmc_gpu.main on the square lattice. The device code is unchanged."""
    setup = build(cfg)
    ops, params = setup.ops, setup.params

    start = time.perf_counter()
    state, probe = g.init_state(ops, setup.system, setup.trial_data, params)
    print(f"initial overlap {float(probe['overlap']):.6e}, local energy {float(probe['energy']):.10f} "
          f"({time.perf_counter() - start:.1f} s incl. compile)", flush=True)
    if cfg.self_check:
        errors = g.conversion_self_check(ops, setup.plans, setup.bonds, probe)
        print(f"self-check vs NumPy channel_mps: relative error alpha {errors[0]:.1e}, beta {errors[1]:.1e}")
        if max(errors) > 1e-6:
            raise AssertionError(f"device conversion disagrees with the NumPy reference: {errors}")
    seconds = setup.info["setup_seconds"]
    seconds["init"] = round(time.perf_counter() - start, 1)

    start_det = (np.asarray(setup.trial_data.mo_coeff_a), np.asarray(setup.trial_data.mo_coeff_b))
    if cfg.trial_export:
        start = time.perf_counter()
        export_trial(cfg.trial_export, cfg, *setup.trial, setup.info["dmrg_energy"], setup.references, start_det,
                     setup.htrial)
        seconds["export"] = round(time.perf_counter() - start, 1)
        print(f"trial export took {seconds['export']:.1f} s", flush=True)

    logger = g.make_block_logger(cfg.block_log, cfg.n_equilibration, cfg.tag) if cfg.block_log else None
    snapshots = None
    if cfg.walker_snapshots:
        # mps_cpmc_gpu's snapshot keys (L is the number of sites), plus the lattice
        config = dict(L=cfg.n_sites, LX=cfg.Lx, LY=cfg.Ly, BOUNDARY_X=cfg.boundary_x, BOUNDARY_Y=cfg.boundary_y,
                      N_UP=cfg.n_up, N_DN=cfg.n_down, T=cfg.hopping, U=cfg.interaction,
                      N_WALKERS=cfg.n_walkers, N_EQL=cfg.n_equilibration, N_BLOCKS=cfg.n_blocks, N_PROP=cfg.n_steps,
                      DT=cfg.dt, SEED=cfg.seed, DMRG_CHI_T=dmrg_schedule(cfg)[0][-1],
                      DMRG_BDIMS=list(cfg.dmrg_bdims), DMRG_SWEEPS=cfg.dmrg_sweeps,
                      CHI_PROP=cfg.walker_channel_chi, E_DMRG=setup.info["dmrg_energy"],
                      E_TRIAL=setup.info["trial_energy"], plan_reference=cfg.plan_reference,
                      walker_start=cfg.walker_start, orbital_plan=cfg.orbital_plan, EPS=cfg.occupation_tolerance,
                      trial_file=cfg.trial_export, tag=cfg.tag, module="mps_cpmc_2d_gpu", device=setup.info["device"])
        snapshots = g.WalkerSnapshots(cfg.walker_snapshots, cfg.n_equilibration + cfg.n_blocks + 1, cfg.n_walkers,
                                      cfg.n_sites, cfg.n_up, cfg.n_down, config)
    block_fn = g.make_block(ops, params, setup.n_chunks, setup.energy_chunks, record_comb=snapshots is not None)
    run_blocks = g.make_run_blocks(block_fn, logger)

    start = time.perf_counter()
    mean, error, _, _, collapsed, timing = g.run_qmc(
        state, ops.data, run_blocks, n_eql=cfg.n_equilibration, n_blocks=cfg.n_blocks,
        n_walkers=cfg.n_walkers, n_steps=cfg.n_steps, snapshots=snapshots)
    elapsed = time.perf_counter() - start

    scalar = lambda x: None if x is None else float(x)
    print(f"CPMC energy = {scalar(mean)} +/- {scalar(error)}; elapsed={elapsed:.1f} s")
    if snapshots is not None:
        snapshots.finish(dict(E_CPMC=scalar(mean), E_CPMC_ERR=scalar(error), collapsed_after_block=collapsed),
                         dict(reference_up=np.asarray(setup.references[0]),
                              reference_dn=np.asarray(setup.references[1]),
                              start_up=start_det[0], start_dn=start_det[1], h1=lattice_hopping(cfg)))
    record = asdict(cfg)
    info = dict(setup.info)
    record.update(
        kind="mps_2d_gpu", n_sites=cfg.n_sites, initial_energy=info.pop("initial_energy"),
        dmrg_energy=info.pop("dmrg_energy"), trial_energy=info.pop("trial_energy"),
        cpmc_energy=scalar(mean), cpmc_error=scalar(error), seconds=elapsed, collapsed_after_block=collapsed,
        n_chunks_used=setup.n_chunks, energy_chunks=setup.energy_chunks, memory_model=setup.memory,
        **timing, **info)
    g.save_result(cfg.result_json, record)
    return record


def config_from_args(pairs):
    values = {}
    for pair in pairs:
        key, value = pair.split("=", 1)
        try:
            values[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            values[key] = value  # bare strings such as boundary_x=periodic
    if isinstance(values.get("dmrg_bdims"), int):
        values["dmrg_bdims"] = (values["dmrg_bdims"],)
    defaults = Config()
    for key, value in values.items():
        default = getattr(defaults, key)
        if isinstance(default, (int, float)) and not isinstance(default, bool) and isinstance(value, str):
            raise ValueError(f"{key}={value!r} is not a number (one key=value per argument)")
    return Config(**values)


if __name__ == "__main__":
    pairs = sys.argv[1:]
    if pairs[:1] == ["dmrg"]:
        raise SystemExit("DMRG references run on the host: mps_cpmc_2d.py dmrg [field=value ...]")
    if pairs[:1] == ["cpmc"]:
        pairs = pairs[1:]
    if any("=" not in pair for pair in pairs):
        raise SystemExit("usage: mps_cpmc_2d_gpu.py [cpmc] [field=value ...]")
    main(config_from_args(pairs))
