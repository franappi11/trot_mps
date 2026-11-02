import numpy as np  
import jax.numpy as jax
from pyblock3.algebra.mpe import MPE
from pyblock3.fcidump import FCIDUMP
from pyblock3.hamiltonian import Hamiltonian
from trot.gmps.utils import sd_to_gmps
from trot.config import configure_once

configure_once()  # float64 for the GMPS conversion

import os
import sys
from pathlib import Path

import numpy as np
from pyblock3.algebra.core import SparseTensor, SubTensor
from pyblock3.algebra.mpe import MPE
from pyblock3.algebra.mps import MPS
from pyblock3.algebra.symmetry import SZ
from pyscf import ao2mo, gto, scf
from trot.gmps.dmrg import  hubbard_pyblock3_mpo, make_pyblock3_hamiltonian
from trot.gmps.utils import sd_to_gmps
from trot.trial.mps import mps_trial_from_pyblock3

def make_pyblock3_hamiltonian(h1, nelec, u=None):
    """pyblock3 Hamiltonian (flat) for len(h1) spatial sites and nelec = (N_up, N_dn).

    The term-built MPO needs only the orbital and symmetry data. With u given, the dense
    diagonal g2 is stored as well (legacy behaviour, L^4 doubles), which build_qc_mpo needs.
    """

    h1 = np.asarray(h1, dtype=float)
    n = len(h1)
    nup, ndn = (int(x) for x in nelec)
    g2 = None
    if u is not None:
        g2 = np.zeros((n,) * 4)
        i = np.arange(n)
        g2[i, i, i, i] = u
    fcidump = FCIDUMP(pg="c1", n_sites=n, n_elec=nup + ndn, twos=nup - ndn, ipg=0, h1e=h1, g2e=g2)
    return Hamiltonian(fcidump, flat=True)

def hubbard_pyblock3_mpo(hamiltonian, h1, u):
    """Hubbard MPO from its operator terms: every nonzero h1[i, j] and U on every site.

    build_qc_mpo treats g2 as a general two-electron tensor, which gives a bond dimension
    ~L^2/2 and tens of GB of DMRG environments at L=48, chi=200.
    """
    # Term encoding of pyblock3's flat builder (as in Hamiltonian.build_complex_qc_mpo):
    # operator index = OP * (0 for c+, 1 for c) + SITE * site + SPIN * spin, -1 pads.
    SPIN, SITE, OP = 1, 2, 16384
    C, D = 0 * OP, 1 * OP
    h1 = np.asarray(h1, dtype=float)
    values, terms = [], []
    for i, j in zip(*np.nonzero(h1)):
        for s in (0, 1):
            values.append(h1[i, j])
            terms.append([C + i * SITE + s * SPIN, D + j * SITE + s * SPIN, -1, -1])
    for i in range(len(h1)):
        # n_up n_down = c+_up c+_down c_down c_up
        values.append(u)
        terms.append([C + i * SITE, C + i * SITE + SPIN, D + i * SITE + SPIN, D + i * SITE])
    gen = (np.array(values, dtype=np.float64), np.array(terms, dtype=np.int32))
    return hamiltonian.build_mpo(gen, cutoff=1.0e-12)

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
n_up = 16
n_dn = 16
h1 = np.zeros((N, N))
for y in range(L):
    for x in range(L):
        site = y * L + x
        if x + 1 < L:
            h1[site, site + 1] = h1[site + 1, site] = -t
        if y + 1 < L:
            h1[site, site + L] = h1[site + L, site] = -t

hamiltonian = make_pyblock3_hamiltonian(h1,(n_up,n_dn),U)
mpo = hubbard_pyblock3_mpo(cutoff=1.0e-12)
n_sweeps = 20
chi_channel = 256
bond_dims = [500] * 10 + [800] * 5 + [1024] * 5 
noises = [1E-5] * 15 + [0] * 5
thrds = [1E-6] * n_sweeps
#TODO:LOAd CA,CB
gmps = sd_to_gmps(Ca, Cb, chi=chi_channel)
mps = gmps_to_pyblock3(gmps.tensors, gmps.charges)
dmrg = MPE(mps, mpo, mps).dmrg(bdims=[bond_dims], noises=noises,
    dav_thrds=thrds, iprint=2, n_sweeps=n_sweeps)
ener = dmrg.energies[-1]
print("Energy = %20.12f" % ener)
#TODO: Save results