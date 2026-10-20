from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, NamedTuple, Protocol

import jax
import numpy as np
from jax.sharding import Mesh

from numpy.typing import NDArray

from ..core.ops import MeasOps, TrialOps
from ..core.system import System


def _random_seed() -> int:
    return int(np.random.randint(0, int(1e6)))


class PropState(NamedTuple):
    walkers: Any
    weights: jax.Array
    overlaps: jax.Array
    rng_key: jax.Array
    pop_control_ene_shift: jax.Array
    e_estimate: jax.Array
    node_encounters: jax.Array


@dataclass(frozen=True)
class QmcParamsBase:
    dt: float = 0.005
    # Walker micro-batches per data shard (global population on one device).
    n_chunks: int = 1
    n_exp_terms: int = 6
    n_prop_steps: int = 50
    n_blocks: int = 200
    n_walkers: int = 200
    seed: int = field(default_factory=_random_seed)


@dataclass(frozen=True)
class QmcParams(QmcParamsBase):
    # By default, the driver compiles the full block starting from
    # ``n_chunks`` and increases it only if the compiler-reported peak memory
    # does not fit conservatively in the device allocator limit.
    # Set False to keep an explicitly chosen walker chunk count fixed.
    auto_n_chunks: bool = True
    pop_control_damping: float = 0.1
    weight_floor: float = 1.0e-3
    weight_cap: float = 100.0
    shift_ema: float = 0.1
    n_eql_blocks: int = 20
    error_method: Literal["gamma", "blocking"] = "gamma"
    # Frozen CISD/UCISD and PT component pair samplers use local Cholesky
    # strata on model or combined data/model meshes. False retains global sampling.
    local_cholesky_sampling: bool = field(default=True, kw_only=True)


@dataclass(frozen=True)
class QmcParamsLno(QmcParams):
    prjlo: NDArray | None = None


_MPS_CHOICES = {
    "orbital_plan": ("rank_exact", "adaptive", "maximal"),
    "plan_reference": ("natural", "rhf"),
    "walker_start": ("natural", "rhf"),
    "energy_kernel": ("blocked", "dense", "mpo"),
    "walker_qr": ("auto", "cholesky", "native"),
    "dmrg_init": ("auto", "neel", "random"),
    "walker_conversion": ("sector", "dense"),
    "dense_move": ("gram", "cans"),
}


@dataclass(frozen=True)
class QmcParamsMps(QmcParams):
    """QmcParams for CPMC with an MPS trial (trot.trial.mps, trot.meas.mps, trot.prop.mps_cpmc).

    trot's base defaults are kept (dt=0.005, n_prop_steps=50, weight_floor=1e-3); the production command line
    (trot/gmps/run_mps_cpmc.py) uses 0.01, 20 and 1e-8.

    trial_chi, dmrg_sweeps, dmrg_seed: pyblock3 DMRG, used when run_qmc_mps builds the trial.
    dmrg_init: DMRG initial state (trot.gmps.dmrg.make_dmrg_trial): "auto" (Neel product state where
      defined, else random), "neel" or "random" (random MPS with a warm-up at a larger bond).
    orbital_plan: gate plan of the walker conversion. "rank_exact" and "maximal" are exact for
      every walker before truncation; "adaptive" is cheaper but exact only for the reference.
    occupation_tolerance: purity threshold of the "adaptive" plan.
    walker_channel_chi, walker_cutoff: per-spin-channel walker bond cap and relative singular
      value cutoff (None and 0.0 keep the conversion exact).
    plan_reference: determinant that freezes the gate circuit and kept counts, "natural" (most
      occupied natural orbitals of the trial) or "rhf" (free-fermion determinant of h1).
    walker_start: determinant every walker starts from, "natural" (trot's convention) or "rhf".
    energy_kernel: "blocked" (charge-labelled H|trial>, blocked contraction), "dense"
      (d=4 walker MPS against a densely compressed H|trial>; for validation) or "mpo" (no stored H|trial>: the MPO is
      applied on the fly with dense walker legs at the bond measure_channel_chi, default walker_channel_chi; the memory
      is the trial plus a few MB instead of H|trial>'s ~19 GB at the doped 6x6's chi_T 512; no szsz).
    walker_qr: walker orthonormalisation, "cholesky" (CholeskyQR2 with a Householder fallback, for
      accelerators), "native" (trot's Householder _qr) or "auto" (cholesky on accelerators, native on CPU).
    sector_buckets: increasing size bounds that split each factorisation's charge sectors into classes padded to
      their own largest member (trot.gmps.engine.compile_circuit): the same factorisations on smaller padded
      matrices. (8, 16) measured 1.19-1.38x faster per step at L=100; () keeps one class per kind.
    measure_channel_chi: per-channel walker bond of the measurements (local energy, observables), None for the
      propagation's walker_channel_chi. The walkers propagate at walker_channel_chi; the energy and observables
      convert them with the same gates at this bond instead (trot.trial.mps.measurement_plan). At the doped 6x6 the
      local energy of chi-16 walkers is biased by +0.25 (+0.06 at 32, ~0 at 64) while their sampling is not.
    walker_conversion: "sector" (the walker bonds keep frozen counts per electron-number sector: static shapes) or
      "dense" (trot.gmps.dense_engine: bonds of the fixed size walker_channel_chi, each walker keeping its own largest
      Schmidt states; no counts to freeze or retrain, the trial is contracted densely). Dense walkers propagate only:
      it needs measure_channel_chi, the sector plan the energy and observables use.
    dense_move: how the dense conversion moves its orthogonality centre, "gram" (polar factor from one eigh) or "cans"
      (polar factor by a polynomial iteration, ~1.4x slower and more accurate at large bonds).
    dense_charge_gap: the dense conversion's per-charge eigh shift in units of trace(G); None is 1e-7.
    dense_snap: zero the off-sector entries of every new isometry of the dense conversion (default). Above bond 16 the
      gates' eigh no longer keeps the sector blocks' zeros and the charge labels fail without it; with it they are
      exact at no measurable cost (trot.gmps.dense_conversion.DenseSpec).
    """

    trial_chi: int = 64
    dmrg_sweeps: int = 14
    dmrg_seed: int = 0
    dmrg_init: Literal["auto", "neel", "random"] = "auto"
    orbital_plan: Literal["rank_exact", "adaptive", "maximal"] = "adaptive"
    occupation_tolerance: float = 1.0e-10
    walker_channel_chi: int | None = 4
    walker_cutoff: float = 0.0
    plan_reference: Literal["natural", "rhf"] = "natural"
    walker_start: Literal["natural", "rhf"] = "natural"
    energy_kernel: Literal["blocked", "dense", "mpo"] = "blocked"
    walker_qr: Literal["auto", "cholesky", "native"] = "auto"
    sector_buckets: tuple[int, ...] = (8, 16)
    measure_channel_chi: int | None = None
    walker_conversion: Literal["sector", "dense"] = "sector"
    dense_move: Literal["gram", "cans"] = "gram"
    dense_charge_gap: float | None = None
    dense_snap: bool = True

    def __post_init__(self) -> None:
        for name, allowed in _MPS_CHOICES.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        buckets = tuple(int(b) for b in self.sector_buckets)
        if any(b < 1 for b in buckets) or list(buckets) != sorted(set(buckets)):
            raise ValueError(f"sector_buckets must be increasing positive sizes, got {self.sector_buckets!r}")
        object.__setattr__(self, "sector_buckets", buckets)
        if self.trial_chi < 1:
            raise ValueError("trial_chi must be positive")
        if self.dmrg_sweeps < 2:
            raise ValueError("dmrg_sweeps must be at least 2 (the last two sweeps are noiseless)")
        if self.walker_channel_chi is not None and self.walker_channel_chi < 1:
            raise ValueError("walker_channel_chi must be None or positive")
        if self.walker_cutoff < 0:
            raise ValueError("walker_cutoff must be non-negative")
        if self.measure_channel_chi is not None and self.measure_channel_chi < 1:
            raise ValueError("measure_channel_chi must be None or positive")
        if self.energy_kernel == "mpo" and self.walker_channel_chi is None and self.measure_channel_chi is None:
            raise ValueError("energy_kernel='mpo' contracts dense walker legs: it needs measure_channel_chi or "
                             "walker_channel_chi (the bond they are converted at)")
        if self.walker_conversion == "dense":
            if self.walker_channel_chi is None:
                raise ValueError("walker_conversion='dense' needs walker_channel_chi (the dense bond)")
            if self.measure_channel_chi is None and self.energy_kernel != "mpo":
                raise ValueError("walker_conversion='dense' needs measure_channel_chi: the energy and observables are "
                                 "measured with a sector plan (or energy_kernel='mpo')")
            if self.dense_charge_gap is not None and self.dense_charge_gap <= 0:
                raise ValueError("dense_charge_gap must be None or positive")


@dataclass(frozen=True)
class QmcParamsFp(QmcParamsBase):
    dt: float = 0.05
    n_prop_steps: int = 20
    n_qr_blocks: int = 10
    n_blocks: int = 5
    ene0: float | None = None
    n_traj: int = 10


class StepKernel(Protocol):

    def __call__(
        self,
        state: PropState,
        *,
        params: Any,
        ham_data: Any,
        trial_data: Any,
        trial_ops: TrialOps,
        meas_ops: MeasOps,
        meas_ctx: Any,
        prop_ctx: Any,
    ) -> PropState: ...


class InitPropState(Protocol):

    def __call__(
        self,
        *,
        sys: System,
        ham_data: Any,
        trial_ops: TrialOps,
        trial_data: Any,
        meas_ops: MeasOps,
        params: Any,
        meas_ctx: Any | None = None,
        initial_walkers: Any | None = None,
        initial_e_estimate: jax.Array | None = None,
        rdm1: jax.Array | None = None,
        mesh: Mesh | None = None,
    ) -> PropState: ...


@dataclass(frozen=True)
class PropOps:
    init_prop_state: InitPropState
    build_prop_ctx: Callable[[Any, jax.Array, Any], Any]  # (ham_data, rdm1, params) -> prop_ctx
    step: StepKernel
