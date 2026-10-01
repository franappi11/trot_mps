"""pyblock3 DMRG trials for MPS-CPMC (Hubbard model, any real symmetric h1).

make_dmrg_trial(ham_data, sys, chi=..., n_sweeps=..., seed=...) runs DMRG with the term-built
Hubbard MPO (bond dimension ~6 on a chain, 2 + 4 x open hoppings in general) and returns the
densified trial (trot.trial.mps.MpsTrial) together with the Davidson and variational energies.

The cfg-based build_dmrg_hamiltonian / hubbard_dmrg_mpo / run_dmrg are the legacy API of
trot/gmps/mps_cpmc_new.py (open chain from cfg.L and cfg.hopping); they keep their behaviour
exactly, including run_dmrg seeding numpy's global RNG with cfg.dmrg_seed.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from pyblock3.algebra.mpe import MPE
from pyblock3.fcidump import FCIDUMP
from pyblock3.hamiltonian import Hamiltonian

from trot.ham.hubbard import hopping_matrix
from trot.meas.mps import hubbard_h1
from trot.trial.mps import MpsTrial, mps_trial_from_pyblock3


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


def dmrg_schedule(chi, n_sweeps):
    """Warm-up schedule: a larger bond with stronger noise first, then chi, last two sweeps clean.

    Sweeping at chi=8 from a random start got stuck at L=48, U=8 and 12 (energy per site 10x
    further from chi=200 than elsewhere).
    """
    warm = max(chi, min(4 * chi, 64))
    bdims = [warm] * 4 + [max(chi, warm // 2)] * 2 + [chi] * n_sweeps
    noises = [1.0e-4] * 4 + [1.0e-5] * 2 + [1.0e-6] * (n_sweeps - 2) + [0.0] * 2
    return bdims, noises


def dmrg_core(hamiltonian, mpo, *, chi, n_sweeps):
    """Random initial MPS (numpy's global RNG, seed it first) optimised by DMRG.

    Returns (mps, last two-site Davidson energy).
    """
    mps = hamiltonian.build_mps(chi)
    bdims, noises = dmrg_schedule(chi, n_sweeps)
    result = MPE(mps, mpo, mps).dmrg(
        bdims=bdims, noises=noises, dav_thrds=[1.0e-10], iprint=-1, n_sweeps=len(bdims)
    )
    return mps, float(result.energies[-1])


class DmrgTrial(NamedTuple):
    trial: MpsTrial
    davidson_energy: float  # two-site energy before the last truncation, not variational
    variational_energy: float  # <mps|H|mps> / <mps|mps>
    mps: Any  # the pyblock3 MPS


def make_dmrg_trial(ham_data, sys, *, chi, n_sweeps=14, seed=0) -> DmrgTrial:
    """DMRG ground state of a HamHubbard (any real symmetric h1) as an MPS trial.

    numpy's global RNG is seeded with seed for the random initial MPS and restored afterwards,
    so later QmcParams() default seeds are unaffected.
    """
    if n_sweeps < 2:
        raise ValueError("n_sweeps must be at least 2 (the last two sweeps are noiseless)")
    h1 = hubbard_h1(ham_data)
    u = float(ham_data.u)
    nelec = (int(sys.nelec[0]), int(sys.nelec[1]))
    hamiltonian = make_pyblock3_hamiltonian(h1, nelec)
    mpo = hubbard_pyblock3_mpo(hamiltonian, h1, u)
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        mps, davidson = dmrg_core(hamiltonian, mpo, chi=int(chi), n_sweeps=int(n_sweeps))
    finally:
        np.random.set_state(state)
    variational = float(MPE(mps, mpo, mps)[0:2].expectation) / float(mps @ mps)
    trial = mps_trial_from_pyblock3(mps, nelec=nelec)
    return DmrgTrial(trial, davidson, variational, mps)


# Legacy cfg-based API of trot/gmps/mps_cpmc_new.py (cfg needs L, hopping, interaction, n_up,
# n_down, trial_chi, dmrg_sweeps, dmrg_seed). Duck-typed: importing Config would be circular.


def build_dmrg_hamiltonian(cfg):
    return make_pyblock3_hamiltonian(
        hopping_matrix(cfg.L, cfg.hopping), (cfg.n_up, cfg.n_down), u=cfg.interaction
    )


def hubbard_dmrg_mpo(hamiltonian, cfg):
    """Hubbard-chain MPO from its ~6L operator terms (bond dimension ~6)."""
    return hubbard_pyblock3_mpo(hamiltonian, hopping_matrix(cfg.L, cfg.hopping), cfg.interaction)


def run_dmrg(hamiltonian, cfg):
    np.random.seed(cfg.dmrg_seed)
    mpo = hubbard_dmrg_mpo(hamiltonian, cfg)
    return dmrg_core(hamiltonian, mpo, chi=cfg.trial_chi, n_sweeps=cfg.dmrg_sweeps)
