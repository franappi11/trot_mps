"""Independent small-Fock-space checks of compressed-trial diagnostics and rotation."""
import numpy as np
import pytest

from tests.helpers import hubbard_fock as hf
from trot.gmps.compress_saved_trial import block_observables
from trot.gmps.trials import to_blocks
from trot.trial.mps import compress_mps_qn, rotate_spin, spin_rotation_y


def site_sum(vector, operator, length):
    shaped = vector.reshape((4,) * length)
    out = np.zeros_like(shaped)
    for i in range(length):
        out += np.moveaxis(np.tensordot(operator, shaped, axes=(1, i)), 0, i)
    return out.ravel()


@pytest.mark.parametrize("nelec", [(2, 2), (2, 1)])
def test_observables_and_as_is_rotation(nelec):
    length = 4
    shape = tuple(len(hf.sector_basis(length, n)[0]) for n in nelec)
    psi = np.random.default_rng(716).normal(size=shape)
    psi /= np.linalg.norm(psi)
    a, q = hf.exact_mps_from_sector_state(psi, length, *nelec)
    # Exercise the new diagnostic on a truncated, non-normalized state.
    a, q = compress_mps_qn(a, q, max_bond=4)
    values = block_observables(to_blocks(a, q))
    vector = hf.mps_full_vector(a)
    norm = vector @ vector
    normalized = vector / np.sqrt(norm)
    sector = hf.sector_from_full(normalized, length, *nelec)
    np.testing.assert_allclose(values["norm"], norm, atol=1e-12)
    np.testing.assert_allclose(values["rdm1"], hf.dense_rdm1(sector, length, *nelec), atol=1e-12)
    doublon = np.diag([0., 0., 0., 1.])
    np.testing.assert_allclose(values["double_occupancy"].sum(),
                               normalized @ site_sum(normalized, doublon, length), atol=1e-12)
    plus = np.zeros((4, 4))
    plus[1, 2] = 1.
    z = np.diag([0., .5, -.5, 0.])
    s2v = site_sum(site_sum(normalized, plus, length), plus.T, length)
    s2v += site_sum(site_sum(normalized, z, length), z, length) + site_sum(normalized, z, length)
    np.testing.assert_allclose(values["s2"], normalized @ s2v, atol=1e-12)
    h1 = np.zeros((length, length))
    for i in range(length-1):
        h1[i, i+1] = h1[i+1, i] = -1.
    energy = np.einsum("ij,sji->", h1, values["rdm1"]) + 8*values["double_occupancy"].sum()
    np.testing.assert_allclose(energy, normalized @ hf.apply_hubbard_full(h1, 8., normalized), atol=1e-12)
    rotated = rotate_spin(a, spin_rotation_y(90.))
    assert [x.shape for x in rotated] == [x.shape for x in a]
    rv = hf.mps_full_vector(rotated) / np.sqrt(norm)
    np.testing.assert_allclose(rv @ rv, 1., atol=1e-12)
    np.testing.assert_allclose(rv @ hf.apply_hubbard_full(h1, 8., rv), energy, atol=1e-12)
    rotated_s2v = site_sum(site_sum(rv, plus, length), plus.T, length)
    rotated_s2v += site_sum(site_sum(rv, z, length), z, length) + site_sum(rv, z, length)
    np.testing.assert_allclose(rv @ rotated_s2v, values["s2"], atol=1e-12)
    sx = (plus + plus.T) / 2
    for i in range(length):
        before = np.moveaxis(np.tensordot(z, normalized.reshape((4,)*length), axes=(1,i)), 0,i)
        after = np.moveaxis(np.tensordot(sx, rv.reshape((4,)*length), axes=(1,i)), 0,i)
        np.testing.assert_allclose(normalized @ before.ravel(), rv @ after.ravel(), atol=1e-12)
    # Rotation is as-is: the component outside the original Sz sector is retained.
    assert np.linalg.norm(hf.sector_from_full(rv, length, *nelec)) < .999


def test_archive_contains_unrotated_recomputed_trial(tmp_path):
    from trot.gmps.compress_saved_trial import main, load_trial
    psi = np.random.default_rng(713).normal(size=(6, 6))
    psi /= np.linalg.norm(psi)
    a, q = hf.exact_mps_from_sector_state(psi, 4, 2, 2)
    source = tmp_path / "source.npz"
    h1 = np.diag([-1., -1., -1.], 1) + np.diag([-1., -1., -1.], -1)
    full = hf.mps_full_vector(a)
    np.savez(source, h1=h1, u=8., nelec=(2, 2), chi=max(map(len, q)),
             energy=full @ hf.apply_hubbard_full(h1, 8., full),
             rdm1=hf.dense_rdm1(psi, 4, 2, 2),
             **{f"A{i}": x for i, x in enumerate(a)}, **{f"q{i}": x for i, x in enumerate(q)})
    main(["--source", str(source), "--baseline", str(source), "--chis", "4", "--out", str(tmp_path / "out")])
    destination = tmp_path / "out/source_to_chi4.npz"
    ca, cq, meta = load_trial(destination)
    state = hf.mps_full_vector(ca)
    np.testing.assert_allclose(state @ state, 1., atol=1e-12)
    np.testing.assert_allclose(meta["energy"], state @ hf.apply_hubbard_full(h1, 8., state), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(hf.sector_from_full(state, 4, 2, 2)), 1., atol=1e-12)
    with np.load(destination) as z:
        assert float(z["stored_rotation_degrees"]) == 0.
        assert float(z["recommended_rotation_degrees"]) == 90.
        assert str(z["rotated_trial"]) == "as_is"
        np.testing.assert_allclose(z["compression_fidelity"], (state @ full)**2, atol=1e-12)
