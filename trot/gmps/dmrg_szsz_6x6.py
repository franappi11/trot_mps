from trot.config import configure_once

configure_once()  # float64, before jax

import os
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from trot.gmps.engine import szsz_dense
from trot.gmps.prepare_htrial_saved import read_saved_trial

STATE = Path("/mnt/ceph/users/fnappi/trot_trials/uhf_dmrg_6x6_U8/state_00")
CHIS = [512, 1024, 2048]


def save_trial(path, tensors, charges, energy, h1, u, nelec, rdm1=None, **meta):
    """The trial-cache format of prepare_trial_6x6.py (A{i}, q{i}, energy, rdm1, h1, u, nelec), written atomically.
    Not used below: compress_trial_6x6.py and prepare_uhf_sd_trial_6x6.py import it (and STATE) from here."""
    arrays = dict(energy=energy, h1=h1, u=u, nelec=np.asarray(nelec), **meta)
    if rdm1 is not None:
        arrays["rdm1"] = rdm1
    arrays.update({f"A{i}": np.asarray(A) for i, A in enumerate(tensors)})
    arrays.update({f"q{i}": np.asarray(q) for i, q in enumerate(charges)})
    path = Path(path)
    tmp = path.with_name(f"{path.stem}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


if __name__ == "__main__":
    for chi in CHIS:
        trial = STATE / f"dmrg_chi{chi}.npz"
        h1, u, nelec, tensors, charges, energy, _ = read_saved_trial(trial)
        print(f"chi {chi}: {trial}, E = {energy:.10f}", flush=True)

        # <S^z_i S^z_j> of the saved trial with itself: a (36, 36) matrix
        tensors = [jnp.asarray(A) for A in tensors]
        start = time.perf_counter()
        C = np.asarray(szsz_dense(tensors, tensors))
        print(f"szsz_dense in {time.perf_counter() - start:.0f} s; sum rule sum_ij C_ij = {C.sum():.2e} "
              f"(0 for S_z = 0)", flush=True)

        out = trial.with_name(f"{trial.stem}_szsz.npz")
        np.savez(out, C=C, energy=energy, h1=h1, u=u, nelec=np.asarray(nelec), sum_rule=C.sum(),
                 trial_file=str(trial))
        print(f"saved {out}", flush=True)
        del tensors
