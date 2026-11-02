"""Maximal orbital plans convert every walker exactly, including at blocks with B <= N.

channel_angles takes an empty mode from the last complete-QR vector of the block only when the block has more rows
than the walker has orbitals (B > N); there that vector lies in null(block^T) for any walker. For B <= N (the late
gates of a maximal plan) it would be a null vector only when the block's leading columns are independent, which
fails for special walkers: the natural orbitals of the L=8 U=8 DMRG chi=4 singlet (degenerate occupations)
converted with 100% error. Those gates use the eigh branch.
"""

from trot import config

config.configure_once()

import jax.numpy as jnp
import numpy as np
import pytest

from tests.helpers import hubbard_fock as hf
from trot.gmps.utils import channel_angles, channel_mps, combine_channels, make_orbital_plan
from trot.trial.mps import convert_walker, make_walker_plan_from_reference, natural_orbitals


def _channel_vector(tensors, gauge):
    v = np.ones((1, 1))
    for A in tensors:
        A = np.asarray(A)
        v = np.einsum("xa,apb->xpb", v, A).reshape(-1, A.shape[2])
    return v[:, 0] * float(gauge)


def _walkers(L, n, seed):
    rng = np.random.default_rng(seed)
    h1 = -(np.diag(np.ones(L - 1), 1) + np.diag(np.ones(L - 1), -1))
    out = {
        "random": np.linalg.qr(rng.standard_normal((L, n)))[0],
        "free fermions": np.linalg.eigh(h1)[1][:, :n],
        "localized first column": np.linalg.qr(
            np.column_stack(
                [np.eye(L)[:, 0], rng.standard_normal((L, n - 1)) * (np.arange(L) > 0)[:, None]]
            )
        )[0],
    }
    return out


@pytest.mark.parametrize("L, n", [(8, 4), (8, 6), (7, 5), (6, 3)])
def test_maximal_plan_empty_modes_vanish_and_amplitudes_match(L, n):
    """After the circuit every empty site's row is zero, and the channel MPS equals the determinants."""
    rows_sector = hf.sector_basis(L, n)[0]
    index = [int(sum(1 << (L - 1 - i) for i in r)) for r in rows_sector]
    for name, C in _walkers(L, n, seed=L + n).items():
        plan = make_orbital_plan(C, "maximal")
        assert np.any(plan.block_sizes <= n), "the case must exercise blocks with B <= N"
        _, rotated = channel_angles(jnp.asarray(C), plan)
        empty = np.flatnonzero(plan.occupation == 0)
        assert max(float(np.abs(np.asarray(rotated[i])).max()) for i in empty) < 1e-12, name
        tensors, _, gauge = channel_mps(jnp.asarray(C), plan)
        vector = _channel_vector(tensors, gauge)
        dets = np.array([np.linalg.det(C[r]) for r in rows_sector])
        np.testing.assert_allclose(vector[index], dets, rtol=0, atol=1e-12 * np.abs(dets).max())
        assert np.abs(np.delete(vector, index)).max() < 1e-12


def test_degenerate_natural_orbitals_of_a_dmrg_singlet_convert_exactly():
    """The case that failed: a walker equal to the plan's reference, natural orbitals of the chi=4 trial."""
    pytest.importorskip("pyblock3")
    import contextlib
    import io

    from trot.gmps.dmrg import make_dmrg_trial
    from trot.ham.hubbard import HamHubbard, hopping_matrix
    from trot.core.system import System

    L = 8
    ham = HamHubbard(h1=jnp.asarray(hopping_matrix(L, 1.0)), u=8.0)
    with contextlib.redirect_stdout(io.StringIO()):
        trial = make_dmrg_trial(
            ham, System(L, (4, 4), "unrestricted"), chi=4, n_sweeps=14, seed=0
        ).trial
    rdm1 = np.asarray(trial.rdm1)
    occupations = np.linalg.eigvalsh(rdm1[0])[::-1]
    assert (
        abs(occupations[0] - occupations[2]) < 1e-8
    ), "the case needs degenerate natural occupations"
    Ra, Rb = natural_orbitals(rdm1[0], 4)[0], natural_orbitals(rdm1[1], 4)[0]
    plan = make_walker_plan_from_reference(Ra, Rb, orbital_plan="maximal")
    alpha, qa, beta, qb, prefactor = convert_walker((jnp.asarray(Ra), jnp.asarray(Rb)), plan)
    tensors, _ = combine_channels(alpha, qa, beta, qb)
    got = float(prefactor) * hf.mps_sector_amplitudes(tensors, 4, 4)
    want = hf.sd_amplitudes(Ra, Rb)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12 * np.abs(want).max())
