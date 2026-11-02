from trot.config import configure_once

configure_once()  # float64, before jax/pyblock3

import os
from pathlib import Path

import numpy as np
from pyblock3.algebra.core import SparseTensor, SubTensor
from pyblock3.algebra.mpe import MPE
from pyblock3.algebra.mps import MPS
from pyblock3.algebra.symmetry import SZ

from trot.gmps.dmrg import hubbard_pyblock3_mpo, make_pyblock3_hamiltonian
from trot.gmps.prepare_htrial_saved import read_saved_trial
from trot.trial.mps import mps_trial_from_pyblock3

PHYSICAL = [(0, 0), (1, 0), (0, 1), (1, 1)]  # (n_up, n_dn) of the local states |0>, |up>, |dn>, |up dn>


def gmps_to_pyblock3(tensors, charges):
    """Dense charge-labelled site tensors (sd_to_gmps) as a flat pyblock3 SZ MPS."""
    q = lambda a, b: SZ(a + b, a - b, 0)
    out = []
    for A, q_left, q_right in zip(tensors, charges[:-1], charges[1:]):
        A, blocks = np.asarray(A), []
        for left in sorted(set(map(tuple, q_left))):
            rows = np.flatnonzero((q_left == left).all(axis=1))
            for p, (a, b) in enumerate(PHYSICAL):
                right = (left[0] + a, left[1] + b)
                cols = np.flatnonzero((q_right == right).all(axis=1))
                block = A[np.ix_(rows, [p], cols)]
                if np.any(block):
                    blocks.append(SubTensor(reduced=block, q_labels=(q(*left), q(a, b), q(*right))))
        out.append(SparseTensor(blocks=blocks))
    return MPS(tensors=out).to_flat()


L = 6
U = 8.0
t = 1.0
N = L * L
n_up = n_dn = 16
h1 = np.zeros((N, N))
for y in range(L):
    for x in range(L):
        site = y * L + x
        if x + 1 < L:
            h1[site, site + 1] = h1[site + 1, site] = -t
        if y + 1 < L:
            h1[site, site + L] = h1[site + L, site] = -t

# warm start: the converged chi 2048 trial (dmrg_szsz_6x6.py, 8 sweeps), swept down to a lower bond, so the result
# stays in its state (the ramp runs from the UHF seed ended in another, higher one: fidelity 0.20 with dmrg_chi1024.npz)
START = Path("/mnt/ceph/users/fnappi/trot_trials/uhf_dmrg_6x6_U8/state_00/dmrg_chi2048.npz")
h1_start, u_start, nelec_start, tensors, charges, e_start, _ = read_saved_trial(START)
if not (np.allclose(h1_start, h1) and u_start == U and nelec_start == (n_up, n_dn)):
    raise SystemExit(f"{START} was made for another system")
print(f"warm start from {START}: bond {max(len(q) for q in charges)}, saved E = {e_start:.10f}", flush=True)

hamiltonian = make_pyblock3_hamiltonian(h1, (n_up, n_dn), U)
mpo = hubbard_pyblock3_mpo(hamiltonian, h1, U)
energy = lambda psi: float(MPE(psi, mpo, psi)[0:2].expectation) / float(psi @ psi)  # variational, as the files store

mps = gmps_to_pyblock3(tensors, charges)
del tensors
e = energy(mps)
print(f"converted to pyblock3: E = {e:.10f} (saved {e_start:.10f}, difference {e - e_start:.1e})", flush=True)
if abs(e - e_start) > 1.0e-8 * abs(e_start):  # a conversion that changed the state would waste the whole run
    raise SystemExit("the converted MPS does not have the saved energy")

bond_dims = [2048] * 2 + [1500] * 4 + [1024] * 6  # down only: a little refinement at 2048, then the lower bonds
n_sweeps = len(bond_dims)
noises = [0.0] * n_sweeps
thrds = [1e-6] * n_sweeps

dmrg = MPE(mps, mpo, mps).dmrg(bdims=bond_dims, noises=noises, dav_thrds=thrds, iprint=2, n_sweeps=n_sweeps)
ener = energy(mps)
print("Energy = %20.12f (variational; last sweep %20.12f)" % (ener, dmrg.energies[-1]), flush=True)

# save in the trial-cache format (A{i}, q{i}, energy, rdm1, h1, u, nelec) every other script here reads
trial = mps_trial_from_pyblock3(mps, nelec=(n_up, n_dn))
out = START.parent / f"dmrg_chi{bond_dims[-1]}_from2048_down.npz"  # distinct from dmrg_chi1024.npz and _from2048.npz
tmp = out.with_name(f"{out.stem}.tmp.npz")
np.savez(tmp, energy=float(ener), rdm1=np.asarray(trial.rdm1), h1=h1, u=U, nelec=np.array([n_up, n_dn]),
         **{f"A{i}": np.asarray(A) for i, A in enumerate(trial.tensors)},
         **{f"q{i}": q for i, q in enumerate(trial.charge_arrays())})
os.replace(tmp, out)  # atomic: a job killed mid-write never leaves a truncated trial behind
print(f"saved to {out}", flush=True)
