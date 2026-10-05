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
    "energy_kernel": ("blocked", "dense"),
    "propagator": ("fast", "slow"),
    "engine": ("auto", "batched", "reference"),
    "linalg": ("auto", "batched", "native"),
    "walker_qr": ("auto", "cholesky", "native"),
    "dmrg_init": ("auto", "neel", "random"),
}


@dataclass(frozen=True)
class QmcParamsMps(QmcParams):
    """QmcParams for CPMC with an MPS trial (trot.trial.mps, trot.gmps.driver.run_qmc_mps).

    Field names follow trot/gmps/mps_cpmc_new.py's Config. trot's base defaults are kept
    (dt=0.005, n_prop_steps=50, weight_floor=1e-3); mps_cpmc_new used 0.01, 20 and 1e-8.

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
    energy_kernel: "blocked" (charge-labelled H|trial>, blocked contraction) or "dense"
      (d=4 walker MPS against a densely compressed H|trial>).
    propagator: "fast" (one walker conversion per HS sweep with cached environments) or
      "slow" (trot.prop.cpmc_slow: a full conversion and overlap for every field proposal).
    engine: "reference" (trot.trial.mps / trot.meas.mps / trot.prop.mps_cpmc), "batched" (the GPU engine of
      trot/gmps/gpu.py: batched sector factorisations, factorized contractions, device data as jit arguments;
      "fast" propagator only) or "auto" (batched on a GPU backend, reference on CPU). Same results up to rounding.
    linalg, walker_qr: batched engine only. linalg "batched" factors all sectors of one kind in one call,
      "native" runs the per-sector loop; walker_qr "cholesky" (CholeskyQR2, Householder fallback) or "native".
      "auto": batched and cholesky on accelerators, native on CPU.
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
    energy_kernel: Literal["blocked", "dense"] = "blocked"
    propagator: Literal["fast", "slow"] = "fast"
    engine: Literal["auto", "batched", "reference"] = "auto"
    linalg: Literal["auto", "batched", "native"] = "auto"
    walker_qr: Literal["auto", "cholesky", "native"] = "auto"

    def __post_init__(self) -> None:
        for name, allowed in _MPS_CHOICES.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        if self.trial_chi < 1:
            raise ValueError("trial_chi must be positive")
        if self.dmrg_sweeps < 2:
            raise ValueError("dmrg_sweeps must be at least 2 (the last two sweeps are noiseless)")
        if self.walker_channel_chi is not None and self.walker_channel_chi < 1:
            raise ValueError("walker_channel_chi must be None or positive")
        if self.walker_cutoff < 0:
            raise ValueError("walker_cutoff must be non-negative")


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
