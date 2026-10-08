from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import sys
import time
from pathlib import Path

from trot.config import configure_once

configure_once()  # float64 for the GMPS conversion

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyblock3.algebra.core import SparseTensor, SubTensor
from pyblock3.algebra.mpe import MPE
from pyblock3.algebra.mps import MPS
from pyblock3.algebra.symmetry import SZ
from pyscf import ao2mo, gto, scf

from trot.gmps.dmrg import hubbard_pyblock3_mpo, make_pyblock3_hamiltonian
from trot.gmps.utils import sd_to_gmps

E_FCI = {(4, 8.0, 7, 7): -10.1218936956}  # fci_4x4_hubbard.ipynb
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0", "#d4a20f", "#d6457a"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
PHYSICAL = [(0, 0), (1, 0), (0, 1), (1, 1)]  # (n_up, n_dn) of the local states |0>, |up>, |dn>, |up dn>

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 14,
    "axes.labelsize": 11, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})


def square_h1(L, t=1.0):
    h1 = np.zeros((L * L, L * L))
    for y in range(L):
        for x in range(L):
            site = y * L + x
            if x + 1 < L:
                h1[site, site + 1] = h1[site + 1, site] = -t
            if y + 1 < L:
                h1[site, site + L] = h1[site + L, site] = -t
    return h1


def run_uhf(h1, U, nelec, guess, max_cycle=200, max_reruns=5):
    """pyscf UHF from the density-matrix guess, rerun from the unstable direction until internally stable."""
    N = len(h1)
    eri = np.zeros((N,) * 4)
    for i in range(N):
        eri[i, i, i, i] = U
    mol = gto.M(verbose=0)
    mol.incore_anyway = True
    mol.nelec = nelec
    mf = scf.UHF(mol)
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(N)
    mf._eri = ao2mo.restore(8, eri, N)
    mf.max_cycle = max_cycle
    mf.kernel(guess)
    stable = False
    for _ in range(max_reruns):
        mo, _, stable, _ = mf.stability(return_status=True)
        if stable:
            break
        mf.kernel(mf.make_rdm1(mo, mf.mo_occ))
    return mf, bool(stable)


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


def one_rdm(psi, hamiltonian):
    """Spin-resolved 1-RDM <psi|c+_{i,s} c_{j,s}|psi> / <psi|psi> with pyblock3: one MPO per element, in the term
    encoding of hubbard_pyblock3_mpo (c+ = 0, c = 16384, + 2 * site + spin)."""
    N = hamiltonian.n_sites
    rdm, norm = np.zeros((2, N, N)), float(psi @ psi)
    with quiet():
        for s in (0, 1):
            for i in range(N):
                for j in range(i, N):
                    term = np.array([[2 * i + s, 16384 + 2 * j + s, -1, -1]], dtype=np.int32)
                    op = hamiltonian.build_mpo((np.array([1.0]), term))
                    rdm[s, i, j] = rdm[s, j, i] = float(MPE(psi, op, psi)[0:2].expectation) / norm
    return rdm


def occupation_plot(rdm1, L, title, path):
    """n_up, n_dn, density, magnetisation and staggered magnetisation on the lattice, fixed colour scales so all
    folders and both methods compare directly."""
    n_up, n_dn = (np.diag(rdm1[s]).reshape(L, L) for s in (0, 1))
    sign = (-1.0) ** np.add.outer(np.arange(L), np.arange(L))  # (-1)^(x+y), indexed [y, x]
    density = n_up + n_dn
    panels = [(n_up, "n_up", "Blues", 0.0, 1.0),
              (n_dn, "n_dn", "Blues", 0.0, 1.0),
              (density, "n_up + n_dn", "Blues", 0.0, max(1.0, density.max())),
              (n_up - n_dn, "n_up - n_dn", "RdBu_r", -1.0, 1.0),
              (0.5 * sign * (n_up - n_dn), "(-1)^(x+y) <S^z_i>", "RdBu_r", -0.5, 0.5)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4), layout="constrained")
    for ax, (values, name, cmap, vmin, vmax) in zip(axes, panels):
        image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xticks(range(L))
        ax.set_yticks(range(L))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.spines[:].set_visible(False)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle(title, fontsize=16, color=INK)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def energies_plot(energies, converged, chosen, path):
    order = np.argsort(np.where(converged, energies, np.inf))[: int(converged.sum())]
    rank = {k: r for r, k in enumerate(order)}
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.plot(np.arange(len(order)), energies[order], ".", ms=3, color=PALETTE[0], label="converged UHF runs, sorted")
    ax.plot([rank[k] for k in chosen], energies[chosen], "o", ms=7, mfc="none", color=PALETTE[1],
            label=f"selected states ({len(chosen)})")
    ax.grid(color=GRID)
    ax.set_xlabel("rank")
    ax.set_ylabel("UHF energy")
    ax.set_title(f"UHF energies of {len(energies)} random guesses")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--U", type=float, default=8.0)
    parser.add_argument("--guesses", type=int, default=1000, help="random UHF guesses (seeds 0 .. guesses-1)")
    parser.add_argument("--states", type=int, default=10, help="lowest distinct UHF solutions to keep")
    parser.add_argument("--energy-tol", type=float, default=1e-6, help="UHF energies closer than this are copies")
    parser.add_argument("--chi-channel", type=int, default=16, help="GMPS bond per spin channel (chi_channel**2 total)")
    parser.add_argument("--chi", type=int, nargs="+", default=[128],
                        help="DMRG bond dimension(s); each starts again from the same GMPS")
    parser.add_argument("--sweeps", type=int, default=20, help="DMRG sweeps; noise 1e-5 on the first four")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    L, U = args.L, args.U
    N = L * L
    nelec = (N // 2 - 1, N // 2 - 1)  # two holes
    e_ref = E_FCI.get((L, U, *nelec))
    out = (args.out or Path.home() / "Desktop" / f"uhf_dmrg_{L}x{L}_U{U:g}").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    h1 = square_h1(L)

    # 1. UHF from random guesses
    start = time.perf_counter()
    seeds = np.arange(args.guesses)
    energies, converged = np.zeros(len(seeds)), np.zeros(len(seeds), bool)
    for k, seed in enumerate(seeds):
        np.random.seed(seed)
        mf, stable = run_uhf(h1, U, nelec, np.random.rand(2, N, N))
        energies[k], converged[k] = mf.e_tot, mf.converged and stable
    print(f"{len(seeds)} UHF runs in {time.perf_counter() - start:.0f} s, {converged.sum()} converged and stable",
          flush=True)

    # 2. the lowest distinct solutions
    chosen = []
    for k in np.argsort(energies):
        if converged[k] and all(abs(energies[k] - energies[j]) > args.energy_tol for j in chosen):
            chosen.append(int(k))
        if len(chosen) == args.states:
            break
    np.savez(out / "scan.npz", seeds=seeds, energies=energies, converged=converged, chosen=np.array(chosen))
    energies_plot(energies, converged, chosen, out / "uhf_energies.png")

    # 3. DMRG from each
    hamiltonian = make_pyblock3_hamiltonian(h1, nelec)
    with quiet():
        mpo = hubbard_pyblock3_mpo(hamiltonian, h1, U)
    energy = lambda psi: float(MPE(psi, mpo, psi)[0:2].expectation) / float(psi @ psi)
    rows = []
    for rank, k in enumerate(chosen):
        start = time.perf_counter()
        folder = out / f"state_{rank:02d}"
        folder.mkdir(exist_ok=True)
        seed = int(seeds[k])
        np.random.seed(seed)
        guess = np.random.rand(2, N, N)
        mf, _ = run_uhf(h1, U, nelec, guess)
        Ca, Cb = (mf.mo_coeff[s][:, mf.mo_occ[s] > 0] for s in (0, 1))
        rdm_uhf = np.asarray(mf.make_rdm1())
        np.save(folder / "guess.npy", guess)
        np.savez(folder / "uhf.npz", seed=seed, energy=mf.e_tot, Ca=Ca, Cb=Cb, rdm1=rdm_uhf)
        occupation_plot(rdm_uhf, L, f"UHF, seed {seed}: E = {mf.e_tot:.6f}", folder / "uhf_occupations.png")

        gmps = sd_to_gmps(Ca, Cb, chi=args.chi_channel)
        for chi in args.chi:
            mps = gmps_to_pyblock3(gmps.tensors, gmps.charges)
            e_start = energy(mps)
            np.random.seed(0)  # DMRG noise
            result = MPE(mps, mpo, mps).dmrg(bdims=[chi] * args.sweeps,
                                             noises=[1e-5] * 4 + [0.0] * max(args.sweeps - 4, 1),
                                             dav_thrds=[1e-9], iprint=-1, n_sweeps=args.sweeps)
            e_dmrg = energy(mps)
            rdm_dmrg = one_rdm(mps, hamiltonian)
            sweep_energies = np.array([float(e) for e in result.energies])
            np.savez(folder / f"dmrg_chi{chi}.npz", energy=e_dmrg, sweep_energies=sweep_energies, rdm1=rdm_dmrg,
                     chi=chi, sweeps=args.sweeps, chi_channel=args.chi_channel,
                     gmps_discarded=gmps.discarded, start_energy=e_start)
            title = f"DMRG from UHF seed {seed}, chi = {chi}: E = {e_dmrg:.6f}"
            if e_ref is not None:
                title += f"  (E - E_FCI = {e_dmrg - e_ref:.2e})"
            occupation_plot(rdm_dmrg, L, title, folder / f"dmrg_chi{chi}_occupations.png")

            row = dict(rank=rank, seed=seed, e_uhf=float(mf.e_tot), e_gmps_start=e_start, e_dmrg=e_dmrg,
                       sweeps_run=len(sweep_energies), gmps_discarded=float(gmps.discarded), chi=chi,
                       chi_channel=args.chi_channel, e_fci=e_ref)
            rows.append(row)
            print(f"state {rank:2d}  seed {seed:4d}  chi {chi:5d}  E_UHF {mf.e_tot:.8f}  E_DMRG {e_dmrg:.8f}"
                  + ("" if e_ref is None else f"  E_DMRG - E_FCI {e_dmrg - e_ref:.2e}")
                  + f"  ({time.perf_counter() - start:.0f} s)", flush=True)
        (folder / "summary.json").write_text(json.dumps([r for r in rows if r["rank"] == rank], indent=2))

    header = f"{L}x{L} open, U = {U:g}, nelec = {nelec}, {len(seeds)} guesses ({converged.sum()} converged and " \
             f"stable), DMRG chi = {args.chi}, {args.sweeps} sweeps, GMPS chi_channel = {args.chi_channel}\n"
    if e_ref is not None:
        header += f"E_FCI = {e_ref}\n"
    lines = [f"{'state':>5} {'seed':>5} {'chi':>5} {'E_UHF':>14} {'E_DMRG':>14}" + ("" if e_ref is None else f" {'E_DMRG-E_FCI':>13}")]
    for r in rows:
        lines.append(f"{r['rank']:5d} {r['seed']:5d} {r['chi']:5d} {r['e_uhf']:14.8f} {r['e_dmrg']:14.8f}"
                     + ("" if e_ref is None else f" {r['e_dmrg'] - e_ref:13.2e}"))
    (out / "summary.txt").write_text(header + "\n".join(lines) + "\n")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
