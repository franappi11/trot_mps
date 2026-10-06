"""pyblock3 DMRG trials for MPS-CPMC (Hubbard model, any real symmetric h1).

make_dmrg_trial(ham_data, sys, chi=..., n_sweeps=..., seed=..., init=...) runs DMRG with the
term-built Hubbard MPO (bond dimension ~6 on a chain, 2 + 4 x open hoppings in general) and returns
the densified trial (trot.trial.mps.MpsTrial) together with the Davidson and variational energies.

dmrg_h1 is the general entry point (any h1, the "terms" or "qc" MPO, the chain's warm-up schedule or the lattice's
plain schedule and bond ramp) that trot.gmps.trials caches.

Initial state (init, both entry points): "neel" starts from the Neel product state (up on the
sublattice of site 0, down on the other; bipartite h1 with matching N_up, N_dn); "random" starts
from a random MPS. "auto" (default) is "neel" where the Neel state is defined, else "random". Random
starts at low chi on long chains get stuck in states with domain walls of the staggered
magnetisation (L=32 chi=6: 6 walls for two of three seeds; the cluster's L=100 chi=16 trial: 4
walls); the Neel start seeds the uniform pattern (0 walls). With the chain's warm-up schedule a Neel
start sweeps directly at chi instead (neel_schedule): a warm-up at a larger bond with strong noise
washes the seed out. The lattice's plain schedule (noise 1e-5) is the same for both starts.

pyblock3 is imported inside the functions that run DMRG, so neel_states and resolve_init (which
name cache files) work without it.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from trot.meas.mps import hubbard_h1
from trot.trial.mps import MpsTrial, mps_trial_from_pyblock3


def make_pyblock3_hamiltonian(h1, nelec, u=None):
    """pyblock3 Hamiltonian (flat) for len(h1) spatial sites and nelec = (N_up, N_dn).

    The term-built MPO needs only the orbital and symmetry data. With u given, the dense
    diagonal g2 is stored as well (legacy behaviour, L^4 doubles), which build_qc_mpo needs.
    """
    from pyblock3.fcidump import FCIDUMP
    from pyblock3.hamiltonian import Hamiltonian

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


def neel_schedule(chi, n_sweeps):
    """The warm-up schedule's replacement for a product (Neel) start: n_sweeps directly at chi with
    noise 1e-6, the last two without noise."""
    return [chi] * n_sweeps, [1.0e-6] * (n_sweeps - 2) + [0.0] * 2


def lattice_dmrg_schedule(chi, n_sweeps, bdims=()):
    """The schedule mps_cpmc_2d_gpu used: chi (or a ramp, last entry repeating) with noise 1e-5 on the first six
    sweeps (or until the ramp is done), the rest noiseless."""
    if not bdims:
        return [chi] * n_sweeps, [1.0e-5] * min(6, n_sweeps) + [0.0]
    ramp = list(bdims)
    if len(ramp) >= n_sweeps:
        raise ValueError("n_sweeps must exceed the ramp's length so the final bond gets noiseless sweeps")
    return ramp + [ramp[-1]] * (n_sweeps - len(ramp)), [1.0e-5] * max(6, len(ramp)) + [0.0]


def neel_states(h1, nelec):
    """Site states of the Neel product state (1 = up, 2 = down) on the lattice of h1, or None.

    The sites are 2-coloured along the nonzero off-diagonal h1 entries (the lowest site of every
    connected component gets the first colour). Up electrons sit on the colour of site 0 and down
    electrons on the other, or the other way round when that matches nelec; None when h1 is not
    bipartite or neither assignment gives (N_up, N_dn).
    """
    h1 = np.asarray(h1)
    n = len(h1)
    colour = np.full(n, -1)
    for root in range(n):
        if colour[root] >= 0:
            continue
        colour[root], stack = 0, [root]
        while stack:
            i = stack.pop()
            for j in np.flatnonzero(h1[i]):
                if j == i:
                    continue
                if colour[j] < 0:
                    colour[j] = 1 - colour[i]
                    stack.append(j)
                elif colour[j] == colour[i]:
                    return None
    nup, ndn = (int(x) for x in nelec)
    a, b = int(np.sum(colour == 0)), int(np.sum(colour == 1))
    if (nup, ndn) == (a, b):
        return np.where(colour == 0, 1, 2)
    if (nup, ndn) == (b, a):
        return np.where(colour == 0, 2, 1)
    return None


def product_mps(hamiltonian, states):
    """pyblock3 MPS of the product state with site states 0 (empty), 1 (up), 2 (down), 3 (up down).

    Every bond keeps only the quantum number of that path, SZ(N so far, 2S_z so far, 0), with
    dimension 1, so the MPS is the product state up to normalisation.
    """
    from pyblock3.algebra.mps import MPS, MPSInfo
    from pyblock3.algebra.symmetry import SZ

    states = [int(x) for x in states]
    n_left = np.cumsum([0] + [(0, 1, 1, 2)[x] for x in states])
    twos_left = np.cumsum([0] + [(0, 1, -1, 0)[x] for x in states])
    L = hamiltonian.n_sites
    info = MPSInfo(L, hamiltonian.vacuum, hamiltonian.target, hamiltonian.basis)

    def keep_path(info):
        for d in range(1, L):
            q = SZ(int(n_left[d]), int(twos_left[d]), 0)
            if not isinstance(next(iter(info.left_dims[d].keys())), SZ):
                q = q.to_flat()  # flat Hamiltonian: bond infos are keyed by flat ints
            for dims in (info.left_dims, info.right_dims):
                bond = dims[d].__class__()
                bond[q] = 1
                dims[d] = bond

    info.set_bond_dimension_fci(call_back=keep_path)
    mps = MPS.ones(info)
    return mps / np.linalg.norm(mps)


DMRG_INITS = ("auto", "neel", "random")


def resolve_init(init, h1, nelec):
    """("neel", site states) or ("random", None) for init in DMRG_INITS."""
    if init not in DMRG_INITS:
        raise ValueError(f"init must be one of {DMRG_INITS}, got {init!r}")
    if init == "random":
        return "random", None
    states = neel_states(h1, nelec)
    if states is None:
        if init == "neel":
            raise ValueError(f"no Neel product state for this h1 and nelec={tuple(nelec)}")
        return "random", None
    return "neel", states


def dmrg_core(hamiltonian, mpo, *, chi, n_sweeps, initial=None):
    """DMRG at bond chi. Returns (mps, last two-site Davidson energy).

    initial=None: a random MPS (numpy's global RNG, seed it first) with the warm-up schedule of
    dmrg_schedule. initial=<pyblock3 MPS> (e.g. product_mps): neel_schedule.
    """
    from pyblock3.algebra.mpe import MPE

    if initial is None:
        mps = hamiltonian.build_mps(chi)
        bdims, noises = dmrg_schedule(chi, n_sweeps)
    else:
        mps = initial.copy()
        bdims, noises = neel_schedule(chi, n_sweeps)
    result = MPE(mps, mpo, mps).dmrg(
        bdims=bdims, noises=noises, dav_thrds=[1.0e-10], iprint=-1, n_sweeps=len(bdims)
    )
    return mps, float(result.energies[-1])


def dmrg_h1(h1, u, nelec, *, chi, n_sweeps, seed=0, init="auto", mpo="terms", schedule="warmup", bdims=(),
            tol=None, iprint=-1):
    """DMRG ground state of the Hubbard model with one-body matrix h1 (any real symmetric h1).

    init: "auto" (the Neel product state where defined, else random), "neel" or "random"; see the module docstring.
    mpo: "terms" (built from h1's nonzero entries and U, bond 2 + 4 x open hoppings) or "qc" (pyblock3's
      build_qc_mpo, U as a general two-electron tensor; bond ~n^2/2 before compression).
    schedule: "warmup" (dmrg_schedule: a larger bond and stronger noise first, n_sweeps at chi after it; the chain
      production schedule; neel_schedule from a Neel start) or "plain" (lattice_dmrg_schedule: chi or the ramp bdims
      for n_sweeps sweeps; the square-lattice production schedule). tol is passed to pyblock3 when given.
    numpy's global RNG is seeded with seed and restored afterwards.

    Returns (tensors, charges, davidson_energy, sweep_energies, variational_energy): dense tensors with (N_up, N_dn)
    bond labels (trot.gmps.utils.densify_with_charges), the last two-site Davidson energy (not variational), all
    sweeps' energies and <mps|H|mps>/<mps|mps>.
    """
    from pyblock3.algebra.mpe import MPE

    from trot.gmps.utils import densify_with_charges

    h1 = np.asarray(h1, dtype=float)
    nelec = (int(nelec[0]), int(nelec[1]))
    _, states = resolve_init(init, h1, nelec)
    hamiltonian = make_pyblock3_hamiltonian(h1, nelec, u=u if mpo == "qc" else None)
    if mpo == "terms":
        operator = hubbard_pyblock3_mpo(hamiltonian, h1, u)
    elif mpo == "qc":
        operator = hamiltonian.build_qc_mpo().compress(cutoff=1.0e-12)[0]
    else:
        raise ValueError("mpo must be 'terms' or 'qc'")
    if schedule == "warmup":
        if bdims:
            raise ValueError("a bond ramp (bdims) needs schedule='plain'")
        schedule_fn = dmrg_schedule if states is None else neel_schedule
        bond_dims, noises = schedule_fn(int(chi), int(n_sweeps))
    elif schedule == "plain":
        bond_dims, noises = lattice_dmrg_schedule(int(chi), int(n_sweeps), tuple(bdims))
    else:
        raise ValueError("schedule must be 'warmup' or 'plain'")
    options = dict(bdims=bond_dims, noises=noises, dav_thrds=[1.0e-10], iprint=iprint,
                   n_sweeps=len(bond_dims) if schedule == "warmup" else int(n_sweeps))
    if tol is not None:
        options["tol"] = tol
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        if states is not None:
            mps = product_mps(hamiltonian, states)
        else:
            mps = hamiltonian.build_mps(bond_dims[0] if schedule == "plain" else int(chi))
        result = MPE(mps, operator, mps).dmrg(**options)
    finally:
        np.random.set_state(state)
    energies = [float(e) for e in result.energies]
    variational = float(MPE(mps, operator, mps)[0:2].expectation) / float(mps @ mps)
    tensors, charges = densify_with_charges(mps, len(h1))
    return tensors, charges, energies[-1], energies, variational


class DmrgTrial(NamedTuple):
    trial: MpsTrial
    davidson_energy: float  # two-site energy before the last truncation, not variational
    variational_energy: float  # <mps|H|mps> / <mps|mps>
    mps: Any  # the pyblock3 MPS
    init: str = "random"  # the initial state actually used: "neel" or "random"


def make_dmrg_trial(ham_data, sys, *, chi, n_sweeps=14, seed=0, init="auto") -> DmrgTrial:
    """DMRG ground state of a HamHubbard (any real symmetric h1) as an MPS trial.

    init: "auto" (Neel product state where defined, else random), "neel" or "random"; see the
    module docstring. numpy's global RNG is seeded with seed (random MPS, DMRG noise) and restored
    afterwards, so later QmcParams() default seeds are unaffected.
    """
    from pyblock3.algebra.mpe import MPE

    if n_sweeps < 2:
        raise ValueError("n_sweeps must be at least 2 (the last two sweeps are noiseless)")
    h1 = hubbard_h1(ham_data)
    u = float(ham_data.u)
    nelec = (int(sys.nelec[0]), int(sys.nelec[1]))
    used, states = resolve_init(init, h1, nelec)
    hamiltonian = make_pyblock3_hamiltonian(h1, nelec)
    mpo = hubbard_pyblock3_mpo(hamiltonian, h1, u)
    initial = None if states is None else product_mps(hamiltonian, states)
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        mps, davidson = dmrg_core(
            hamiltonian, mpo, chi=int(chi), n_sweeps=int(n_sweeps), initial=initial
        )
    finally:
        np.random.set_state(state)
    variational = float(MPE(mps, mpo, mps)[0:2].expectation) / float(mps @ mps)
    trial = mps_trial_from_pyblock3(mps, nelec=nelec)
    return DmrgTrial(trial, davidson, variational, mps, used)
