"""Exact ground state (pyscf FCI) of the Hubbard model on an open Lx x Ly lattice, and its structure.

The lowest --nroots states of the (N_up, N_dn) sector: energy, <S^2>, the point-group characters and irrep weights
(trot.gmps.fci_tools.square_point_group: C4v for a square). For the ground state:
  * entanglement: Schmidt coefficients across every cut of the site order (site x * Ly + y), with the entropy, the
    weight per left sector (N_left, 2 S^z_left) and the smallest discarded weight possible at each bond dimension,
    i.e. the best fidelity any MPS of that bond can reach in this site order;
  * correlations: n_i, <n_i n_j>, <S^z_i S^z_j>, double occupancy, empty-site (hole) correlations <h_i h_j>, and
    singlet-pair correlations on the bonds with their d-wave combination;
  * distance averages: staggered spin, hole-hole and d-wave pair correlations against distance.
It is the reference that trial_spin_check.py --fci compares trials with.

    # CPU only: one genx Ice Lake job, from trot/gmps. pyscf's Davidson keeps (12 + 4 (nroots - 1)) * 2 + 3 nroots
    # vectors of 1.05 GB in memory at 4x4 7+7 (63 GB for 4 roots); each H|c> is ~5e12 flops
    sbatch -p genx -C "icelake&rocky9" -J fci_4x4 --time=04:00:00 --cpus-per-task=32 --mem=100G \\
        -o slurm/fci_4x4-%j.out --export=ALL,TARGET=fci_reference.py run_mps_sweep.sh --Lx 4 --Ly 4 \\
        --n-up 7 --n-down 7 --U 8 --nroots 4

Outputs: <vectors>/<tag>_ci0.npy (the ground state's CI matrix, pyscf string order), and in --out <tag>_fci.json
(energies, spins, irreps, summaries) and <tag>_fci.npz (h1, U, nelec, the structure arrays, the Schmidt spectra and
ci_path, the location of the CI matrix), which trial_spin_check.py --fci reads.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BOUNDARIES = ("open", "periodic", "antiperiodic")
CHIS = (32, 64, 128, 192, 256, 384, 512, 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--Lx", type=int, default=4)
    parser.add_argument("--Ly", type=int, default=4)
    parser.add_argument("--bx", default="open", choices=BOUNDARIES)
    parser.add_argument("--by", default="open", choices=BOUNDARIES)
    parser.add_argument("--n-up", type=int, default=None, help="default: half filling")
    parser.add_argument("--n-down", type=int, default=None, help="default: half filling")
    parser.add_argument("--U", type=float, default=8.0)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--nroots", type=int, default=4)
    parser.add_argument("--max-memory", type=float, default=80000, help="MB for pyscf's Davidson")
    parser.add_argument("--conv-tol", type=float, default=1e-10)
    parser.add_argument("--max-cycle", type=int, default=200)
    parser.add_argument("--reference", type=float, default=None, help="known ground-state energy to check against")
    parser.add_argument("--vectors", default="/mnt/ceph/users/fnappi/trot_trials/fci", help="folder of the CI matrix")
    parser.add_argument("--out", default=str(HERE / "fci_reference"))
    args = parser.parse_args()

    from pyscf import fci

    from trot.gmps import fci_tools, trials
    from trot.gmps.dmrg import bipartite_colouring
    from trot.ham.hubbard import square_hopping_matrix

    start = time.time()
    h1 = square_hopping_matrix(args.Lx, args.Ly, args.t, args.bx, args.by)
    n = len(h1)
    nelec = (args.n_up if args.n_up is not None else n // 2, args.n_down if args.n_down is not None else n // 2)
    tag = f"{trials.describe_h1(h1).name}_n{nelec[0]}-{nelec[1]}_U{args.U:g}"
    eri = np.zeros((n,) * 4)
    eri[np.arange(n), np.arange(n), np.arange(n), np.arange(n)] = args.U

    # ---- FCI (standard pyscf)
    solver = fci.direct_spin1.FCI()
    # verbose 5: pyscf prints one "davidson <iteration> ... |r|= ... e= ..." line per Davidson iteration (debug level)
    solver.max_memory, solver.conv_tol, solver.max_cycle, solver.verbose = args.max_memory, args.conv_tol, \
        args.max_cycle, 5
    energies, vectors = solver.kernel(h1, eri, n, nelec, nroots=args.nroots)
    energies = np.atleast_1d(energies)
    vectors = list(vectors) if args.nroots > 1 else [vectors]
    print(f"{tag}: FCI done ({time.time() - start:.0f} s), {vectors[0].size} determinants", flush=True)

    # ---- spin and point group of every root
    group = fci_tools.square_point_group(args.Lx, args.Ly)
    roots = []
    for k, (energy, ci) in enumerate(zip(energies, vectors)):
        s2, multiplicity = fci.spin_op.spin_square(ci, n, nelec)
        characters, weights = fci_tools.point_group_weights(ci, n, nelec, group)
        irrep = max(weights, key=weights.get)
        roots.append(dict(root=k, energy=float(energy), S2=float(s2), S=float((multiplicity - 1) / 2),
                          irrep=irrep, irrep_weights=weights, characters=characters))
        print(f"root {k}: E = {energy:.10f}" + (f" (- reference {energy - args.reference:+.2e})"
                                              if args.reference is not None and k == 0 else "")
              + f", <S^2> = {s2:.6f} (S = {roots[-1]['S']:.2f}), irrep {irrep} (weight {weights[irrep]:.6f}); "
              + "characters " + ", ".join(f"{g} {c:+.4f}" for g, c in characters.items()), flush=True)

    # ---- structure of the ground state
    psi = vectors[0] / np.linalg.norm(vectors[0])
    diag = fci_tools.diagonal_correlations(psi, n, nelec)
    holes, hole_hole = fci_tools.hole_correlations(psi, n, nelec)
    bonds = fci_tools.lattice_bonds(args.Lx, args.Ly)
    pair = fci_tools.pair_correlations(psi, n, nelec, bonds)
    d_wave = fci_tools.d_wave_correlations(pair, bonds, n)
    colour = bipartite_colouring(h1)
    stagger = np.where(colour == 0, 1.0, -1.0) if colour is not None else np.ones(n)
    curves = dict(
        staggered_szsz=fci_tools.distance_average(np.outer(stagger, stagger) * diag["szsz"], args.Lx, args.Ly, 1.0),
        hole_hole=fci_tools.distance_average(hole_hole, args.Lx, args.Ly, 1.0),
        d_wave=fci_tools.distance_average(d_wave, args.Lx, args.Ly, 2.0))
    print(f"structure done ({time.time() - start:.0f} s)", flush=True)
    schmidt, cut_summaries = {}, {}
    for cut in range(1, n):
        schmidt[cut] = fci_tools.schmidt_values(psi, n, nelec, cut)
        cut_summaries[cut] = fci_tools.schmidt_summary(schmidt[cut], CHIS)
        print(f"cut {cut:2d}: S = {cut_summaries[cut]['entropy']:.4f}, rank {cut_summaries[cut]['rank']}, discarded "
              + ", ".join(f"chi {chi} {w:.2e}" for chi, w in cut_summaries[cut]["discarded"].items()), flush=True)

    # ---- report
    occupations = diag["n_up"] + diag["n_dn"]
    rows = (args.Lx, args.Ly)
    print("<n_i> (row x, column y):\n" + "\n".join("    " + " ".join(f"{v:.4f}" for v in row)
                                                 for row in occupations.reshape(rows)))
    print("<h_i> (empty sites):\n" + "\n".join("    " + " ".join(f"{v:.4f}" for v in row)
                                             for row in holes.reshape(rows)))
    print(f"double occupancy: mean {diag['double'].mean():.4f}; S^zz(pi,pi) = "
          f"{float(stagger @ diag['szsz'] @ stagger) / n:.4f}")
    for name, curve in curves.items():
        print(f"{name} vs distance: " + ", ".join(f"r={r:.3f}: {v:+.5f}" for r, v in curve.items()))
    middle = cut_summaries[n // 2]
    top = sorted(middle["sectors"].items(), key=lambda kv: -kv[1])[:8]
    print("middle cut, weight per (N_left, 2Sz_left): " + ", ".join(f"({k}) {w:.4f}" for k, w in top))

    # ---- save
    vectors_dir, out = Path(args.vectors), Path(args.out)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    ci_path = vectors_dir / f"{tag}_ci0.npy"
    np.save(ci_path, psi)
    spectra = {f"schmidt_cut{cut}_{a}_{b}": s for cut, values in schmidt.items() for (a, b), s in values.items()}
    np.savez(out / f"{tag}_fci.npz", h1=h1, u=args.U, nelec=np.asarray(nelec), Lx=args.Lx, Ly=args.Ly,
             ci_path=str(ci_path), energies=energies, occupations=occupations, n_up=diag["n_up"], n_dn=diag["n_dn"],
             nn=diag["nn"], szsz=diag["szsz"], double=diag["double"], holes=holes, hole_hole=hole_hole,
             pair=pair, bonds=np.asarray(bonds), d_wave=d_wave, **spectra)
    summary = dict(tag=tag, Lx=args.Lx, Ly=args.Ly, boundaries=[args.bx, args.by], U=args.U, t=args.t,
                   nelec=list(nelec), ci_path=str(ci_path), roots=roots, curves=curves,
                   schmidt={str(cut): s for cut, s in cut_summaries.items()},
                   S_zz_pi=float(stagger @ diag["szsz"] @ stagger) / n, seconds=time.time() - start)
    json_path = out / f"{tag}_fci.json"
    tmp = json_path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(tmp, json_path)
    print(f"saved {ci_path}, {out / (tag + '_fci.npz')} and {json_path} ({time.time() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
