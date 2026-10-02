"""Local energy of SD walkers against a RotatedMpsTrial (an MPS without definite (N_up, N_dn)).

E_loc = <T|H|W> / <T|W> with dense contractions: the walker's spin channels are combined into
one d=4 MPS and contracted with H|T> and with T. H|T> is applied once on the host with the
automaton MPO of trot.meas.mps and compressed densely (compress_mps); it carries no bond labels.
H conserves S_z, so E_loc equals that of the trial projected onto the walkers' sector.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

from trot.core.ops import MeasOps, k_energy
from trot.gmps.utils import contract_real
from trot.meas.mps import _dense_overlap, apply_mpo, compress_mps, hubbard_h1, hubbard_mpo_from_h1
from trot.trial.mps import MpsWalkerPlan
from trot.trial.mps_rotation import (
    RotatedMpsTrial,
    check_rotated_trial,
    dense_walker_mps,
    rotated_overlap_fn,
)


@tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class RotatedMeasCtx:
    """Per-run measurement data for a RotatedMpsTrial, built once by build_rotated_meas_ctx.

    h_tensors: the densely compressed H|trial> tensors.
    key: static aux (plan, trial bond dimensions, trial energy, H|trial> bond dimensions).
    """

    h_tensors: tuple
    key: tuple

    @property
    def plan(self) -> MpsWalkerPlan:
        return self.key[0]

    @property
    def trial_bond_dims(self) -> tuple:
        return self.key[1]

    @property
    def trial_energy(self) -> float:
        """<trial|H|trial> / <trial|trial> over all sectors of the (unprojected) trial."""
        return self.key[2]

    @property
    def h_bond_dims(self) -> tuple:
        return self.key[3]

    def tree_flatten(self):
        return (self.h_tensors,), self.key

    @classmethod
    def tree_unflatten(cls, key, children):
        return cls(h_tensors=children[0], key=key)


def build_rotated_meas_ctx(ham_data, trial_data, *, plan: MpsWalkerPlan) -> RotatedMeasCtx:
    """Host-side precomputation (called eagerly by trot's drivers): the dense H|trial>."""
    check_rotated_trial(trial_data, plan)
    trial_np = [np.asarray(A) for A in trial_data.tensors]
    W = hubbard_mpo_from_h1(hubbard_h1(ham_data), float(ham_data.u))
    h_np = compress_mps(apply_mpo(W, trial_np))
    energy = _dense_overlap(h_np, trial_np) / _dense_overlap(trial_np, trial_np)
    bonds = tuple(int(A.shape[0]) for A in h_np) + (int(h_np[-1].shape[-1]),)
    key = (plan, trial_data.bond_dims, energy, bonds)
    return RotatedMeasCtx(h_tensors=tuple(jnp.asarray(A) for A in h_np), key=key)


def check_rotated_meas_ctx(meas_ctx, plan: MpsWalkerPlan, trial_data) -> None:
    if not isinstance(meas_ctx, RotatedMeasCtx):
        raise ValueError(
            "a RotatedMpsTrial needs meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data) "
            f"(got {type(meas_ctx).__name__})"
        )
    if meas_ctx.plan is not plan:
        raise ValueError("meas_ctx was built for a different walker plan")
    if meas_ctx.trial_bond_dims != trial_data.bond_dims:
        raise ValueError("meas_ctx was built for a different trial (bond dimensions differ)")


def rotated_energy(
    walker, ham_data, meas_ctx: RotatedMeasCtx, trial_data: RotatedMpsTrial, plan: MpsWalkerPlan
):
    """<H trial|walker> / <trial|walker> with the dense d=4 walker MPS."""
    check_rotated_meas_ctx(meas_ctx, plan, trial_data)
    tensors, _ = dense_walker_mps(walker, plan)
    return contract_real(tensors, meas_ctx.h_tensors) / contract_real(
        tensors, tuple(trial_data.tensors)
    )


def rotated_energy_fn(plan: MpsWalkerPlan):
    """The jitted rotated-trial local energy of a plan (cached on the plan)."""
    fn = plan.caches.get("rotated_energy")
    if fn is None:
        fn = jax.jit(lambda walker, ham, ctx, trial: rotated_energy(walker, ham, ctx, trial, plan))
        plan.caches["rotated_energy"] = fn
    return fn


def make_rotated_meas_ops_hubbard(plan: MpsWalkerPlan) -> MeasOps:
    """MeasOps for a RotatedMpsTrial and a HamHubbard: overlap, build_meas_ctx and the energy."""
    return MeasOps(
        overlap=rotated_overlap_fn(plan),
        build_meas_ctx=partial(build_rotated_meas_ctx, plan=plan),
        kernels={k_energy: rotated_energy_fn(plan)},
    )
