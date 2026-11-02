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

from trot.gmps.dmrg import dmrg_with_diagnostics, hubbard_pyblock3_mpo, make_pyblock3_hamiltonian
from trot.gmps.utils import sd_to_gmps
from trot.trial.mps import mps_trial_from_pyblock3

L, U, t = 8, 8.0, 1.0
N = L * L
n_up = n_dn = 28

h1 = np.zeros((N, N))
for y in range(L):
    for x in range(L):
        site = y * L + x
        if x + 1 < L:
            h1[site, site + 1] = h1[site + 1, site] = -t
        if y + 1 < L:
            h1[site, site + L] = h1[site + L, site] = -t

# One job per stage (submit_prepare_trial_6x6.sh):
#   python prepare_trial_6x6.py            only the UHF stage: out/scan.npz and out/state_XX/{guess.npy, uhf.npz}
#   python prepare_trial_6x6.py 128 256    DMRG at these chi from the saved UHF states, out/state_XX/dmrg_chi<chi>.npz
# A DMRG job runs the UHF stage itself when scan.npz is missing.
chi_trial = [int(c) for c in sys.argv[1:]]

# Same layout as the 4x4 trials: out/scan.npz, out/state_XX/{guess.npy, uhf.npz, dmrg_chi*.npz}
# TRIAL_OUT redirects everything (e.g. a timing re-run that must not overwrite the saved trials); it needs a copy of
# scan.npz and state_XX/uhf.npz when only DMRG stages are run.
out = Path(os.environ.get("TRIAL_OUT", "/mnt/ceph/users/fnappi/trot_trials/uhf_dmrg_6x6_U8"))
out.mkdir(parents=True, exist_ok=True)

if chi_trial and (out / "scan.npz").exists():  # the UHF stage of an earlier job, written last
    with np.load(out / "scan.npz") as z:
        seeds, energies, chosen = z["seeds"], z["energies"], [int(k) for k in z["chosen"]]
    Cas, Cbs = {}, {}
    for rank, k in enumerate(chosen):
        with np.load(out / f"state_{rank:02d}" / "uhf.npz") as z:
            Cas[k], Cbs[k] = z["Ca"], z["Cb"]
    print(f"UHF stage loaded from {out}: {len(chosen)} states, seeds {[int(seeds[k]) for k in chosen]}", flush=True)
else:
    n_tries = 10000
    seeds = np.arange(n_tries)
    one_rdms = np.zeros((n_tries,2,N,N))
    Cas = np.zeros((n_tries,N,n_up))
    Cbs = np.zeros((n_tries,N,n_up))
    energies = np.zeros(n_tries)
    converged = np.zeros(n_tries, dtype=bool)
    stable = np.zeros(n_tries, dtype=bool)
    guesses = np.zeros((n_tries,2,N,N))
    for k,seed in enumerate(seeds):
        np.random.seed(seed)
        guess = np.random.rand(2, N, N)  # diagonal density matrices of the product state
        guesses[k] = guess
        eri = np.zeros((N,) * 4)
        for i in range(N):
            eri[i, i, i, i] = U
        mol = gto.M(verbose=0)
        mol.incore_anyway = True
        mol.nelec = (n_up, n_dn)
        mf = scf.UHF(mol)
        mf.get_hcore = lambda *args: h1
        mf.get_ovlp = lambda *args: np.eye(N)
        mf._eri = ao2mo.restore(8, eri, N)
        mf.kernel(guess)
        mo1 = mf.stability()[0]
        dm1 = mf.make_rdm1(mo1, mf.mo_occ)
        mf = mf.run(dm1)
        stable[k] = mf.stability(return_status=True)[2]
        converged[k] = mf.converged
        Ca, Cb = mf.mo_coeff[0][:, :n_up], mf.mo_coeff[1][:, :n_dn]
        Cas[k] = Ca
        Cbs[k] = Cb
        rdm_up, rdm_dn = (np.diag(dm) for dm in mf.make_rdm1())
        one_rdm =  mf.make_rdm1()
        one_rdms[k] = one_rdm.copy()
        energies[k] = mf.e_tot

    n_states = 1

    chosen = []
    for k in np.argsort(energies):
        if converged[k] and stable[k] and all(abs(energies[k] - energies[j]) > 1e-6 for j in chosen):
            chosen.append(int(k))
        if len(chosen) == n_states:
            break
    print(f"{(converged & stable).sum()} of {n_tries} UHF runs converged and stable; chose {len(chosen)} of {n_states} "
          f"states", flush=True)

    for rank, k in enumerate(chosen):
        folder = out / f"state_{rank:02d}"
        folder.mkdir(exist_ok=True)
        np.save(folder / "guess.npy", guesses[k])
        np.savez(folder / "uhf.npz", seed=int(seeds[k]), energy=energies[k], Ca=Cas[k], Cb=Cbs[k], rdm1=one_rdms[k],
                 converged=converged[k], stable=stable[k])
    # last, so scan.npz means the whole UHF stage is on disk
    np.savez(out / "scan.npz", seeds=seeds, energies=energies, converged=converged, stable=stable,
             chosen=np.array(chosen))
    print(f"UHF stage saved to {out}", flush=True)

import contextlib
import ctypes
import os
import sys
import time
import warnings

from pyblock3.symbolic.expr import OpElement, OpNames
from pyblock3.symbolic.symbolic_mpo import SiteMPO

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


@contextlib.contextmanager
def quiet():
    """Silence block3's 'MPO site ...' lines: compiled code writes them straight to file descriptor 1."""
    libc = ctypes.CDLL(None)
    sys.stdout.flush()
    libc.fflush(None)
    saved = os.dup(1)
    with open(os.devnull, "w") as null:
        os.dup2(null.fileno(), 1)
        try:
            yield
        finally:
            libc.fflush(None)
            os.dup2(saved, 1)
            os.close(saved)


def annihilator(i, s):

    op = OpElement(OpNames.D, (i, s), q_label=SZ(-1, -1 if s == 0 else 1, hamiltonian.orb_sym[i]))
    site = SiteMPO(hamiltonian, op)
    return MPS(tensors=site.tensors, const=0.0, dq=site.dq).to_sparse().to_flat()


def one_rdm(psi):
    """Spin-resolved 1-RDM <psi|c+_{i,s} c_{j,s}|psi> / <psi|psi> = <c_{i,s} psi|c_{j,s} psi> / <psi|psi>, as in the
    pyblock3 docs' 1-PDM example. s = 0 up, 1 down."""
    rdm, norm = np.zeros((2, N, N)), float(psi @ psi)
    with warnings.catch_warnings():  # pyblock3's quantum-number arithmetic overflows harmlessly for c's labels
        warnings.filterwarnings("ignore", "overflow encountered in scalar", RuntimeWarning)
        for s in (0, 1):
            kets = [annihilator(i, s) @ psi for i in range(N)]
            for i in range(N):
                for j in range(i, N):
                    rdm[s, i, j] = rdm[s, j, i] = float(np.dot(kets[i].conj(), kets[j])) / norm
    return rdm



hamiltonian = make_pyblock3_hamiltonian(h1, (n_up, n_dn))
with quiet():
    mpo = hubbard_pyblock3_mpo(hamiltonian, h1, U)
energy = lambda psi: float(MPE(psi, mpo, psi)[0:2].expectation) / float(psi @ psi)

chi_channel = 16
n_sweeps = 20
for chi in chi_trial:
    for rank, k in enumerate(chosen):
        start = time.perf_counter()
        folder = out / f"state_{rank:02d}"
        seed = int(seeds[k])
        gmps = sd_to_gmps(Cas[k], Cbs[k], chi=chi_channel)
        mps = gmps_to_pyblock3(gmps.tensors, gmps.charges)
        e_start = energy(mps)
        np.random.seed(0)  # DMRG noise
        result, diag = dmrg_with_diagnostics(MPE(mps, mpo, mps), bdims=[chi] * 6, noises=[1e-5] * 4 + [0.0] * 2,
                                             dav_thrds=[1e-9], n_sweeps=n_sweeps)
        e_dmrg = energy(mps)
        sweep_energies = np.array([float(e) for e in result.energies])
        print(f"  chi {chi}: last sweep DE = {diag.de[-1]:.2e}, max discarded weight = {diag.mdw[-1]:.2e} "
              f"(not converged if either is still large; dmrg_with_diagnostics' sweep log is above)", flush=True)
        # The trial as make_dmrg_trial gives it to ex_2d_6x6.py: dense (N_up, N_dn)-labelled tensors, normalized, and
        # the rdm1 make_mps_trial computes from them (the unrotated rdm1 behind the walker plan and start).
        trial = mps_trial_from_pyblock3(mps, nelec=(n_up, n_dn))
        # Trial-cache format, read back by ex_2d_4x4_saved.py / run_saved_trial_cpmc.load_saved_trial:
        # A{i} (N sites), q{i} (N + 1 bonds), energy, rdm1, h1, u, nelec. Written to a temporary name first, so a
        # job killed at its time limit never leaves a truncated trial behind.
        path = folder / f"dmrg_chi{chi}.npz"
        tmp = path.with_name(f"{path.stem}.tmp.npz")
        np.savez(tmp, energy=e_dmrg, rdm1=np.asarray(trial.rdm1), h1=h1, u=U, nelec=np.array([n_up, n_dn]),
                 **{f"A{i}": np.asarray(A) for i, A in enumerate(trial.tensors)},
                 **{f"q{i}": q for i, q in enumerate(trial.charge_arrays())},
                 sweep_energies=sweep_energies, sweep_de=diag.de, sweep_mdw=diag.mdw,
                 sweep_time_s=diag.sweep_time_s, chi=chi, sweeps=n_sweeps, chi_channel=chi_channel,
                 gmps_discarded=gmps.discarded, start_energy=e_start, uhf_seed=seed, uhf_energy=energies[k])
        os.replace(tmp, path)
        print(f"state {rank}: seed {seed}, chi {chi}, E_UHF {energies[k]:.8f}, E_DMRG = {e_dmrg:.8f} "
              f"({time.perf_counter() - start:.0f} s) -> {path}", flush=True)
