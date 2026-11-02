"""A saved trial control must preserve the tensors, Hamiltonian, and unrotated RDM."""

from trot import config

config.configure_once()

import hashlib

import numpy as np
import pytest

from trot.gmps.run_saved_trial_cpmc import load_saved_trial
from trot.ham.hubbard import square_hopping_matrix


def _artifact(path, **overrides):
    states = [1, 0, 2, 0]
    tensors = [np.eye(4)[p].reshape(1, 4, 1) for p in states]
    deltas = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])[states]
    charges = np.vstack((np.zeros((1, 2), int), np.cumsum(deltas, axis=0)))[:, None, :]
    rdm = np.stack([np.diag(np.array(states) == p) for p in (1, 2)]).astype(float)
    data = dict(h1=np.asarray(square_hopping_matrix(2, 2, 1.)), u=8., nelec=(1, 1), rdm1=rdm,
                **{f"A{i}": a for i, a in enumerate(tensors)}, **{f"q{i}": q for i, q in enumerate(charges)})
    data.update(overrides)
    np.savez(path, **data)
    return data


def test_loading_preserves_the_saved_trial_and_unrotated_rdm(tmp_path):
    path = tmp_path / "trial.npz"
    data = _artifact(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    h1, u, nelec, trial = load_saved_trial(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    np.testing.assert_array_equal(h1, data["h1"])
    assert u == 8. and nelec == (1, 1)
    np.testing.assert_array_equal(trial.rdm1, data["rdm1"])
    assert not np.array_equal(trial.rdm1[0], trial.rdm1[1])
    for i, a in enumerate(trial.tensors):
        np.testing.assert_array_equal(a, data[f"A{i}"])
    for i, q in enumerate(trial.charge_arrays()):
        np.testing.assert_array_equal(q, data[f"q{i}"])


def test_loader_rejects_inconsistent_particle_labels(tmp_path):
    path = tmp_path / "trial.npz"
    _artifact(path, q4=np.array([[2, 0]]))
    with pytest.raises(ValueError, match="charge labels"):
        load_saved_trial(path)
