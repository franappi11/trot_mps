"""block2 reference for the MPS-CPMC <S^z_i S^z_j> mixed estimator (numpy + pyblock2 only: no trot, no JAX).

    ~/.trot/bin/python trot/gmps/szsz_block2_check.py file.npz [--tol 1e-12]

file.npz comes from trot/gmps/szsz_block2_export.py. The trial T is loaded from its determinant expansion and every
walker is expanded here, with numpy alone, as the determinant expansion of its two spin channels
(alpha-block amplitudes det(Ca[A]) det(Cb[B]) times the interleaving sign (-1)^K, K = #{i in A, j in B, j < i}).
block2 then gives the transition 2-PDM <T|n_i n_j|walker> through get_npdm(walker, bra=T, ...):

- case "sz": SymmetryTypes.SZ, npdm_expr = cdcd, cdCD, CDcd, CDCD with mask (0, 0, 1, 1); S^z S^z = (aa - ab - ba + bb)/4.
- case "nosz": SymmetryTypes.SGF on 2L spin orbitals (2i up, 2i+1 down), npdm_expr = CDCD; the trial is not an S_z
  eigenstate, so SZ symmetry cannot hold it. S^z S^z = sum_{s, s'} s s' <n_{i s} n_{j s'}> with s = +-1/2.

Two block2 routes give the numerator <T|S^z_i S^z_j|walker>: the transition 2-PDM above, and a direct expectation
<T|MPO_ij|walker> of the two-site operator (one expr_builder MPO per pair). The MPO route is exact to rounding for any
walker. get_npdm rotates to the eigenbasis of the ket's reduced density matrices and drops states below `cutoff`:
a walker close to a product state (a CPMC walker near the trial) has Schmidt weights down to ~1e-20 that it loses,
so it agrees with the exact value to ~1e-11 (best at cutoff 1e-20; not better with a smaller cutoff, which only adds
noise). A fresh random-orbital determinant, without any propagation, is reproduced by get_npdm to ~1e-15.

The mixed estimator is that value over <T|walker>, and the block value the weighted mean over the walkers. Also checked:
the normalised overlap <T|walker> against trot's and <T|H|T>/<T|T> against trot's trial energy, with a Hubbard
MPO built here from h1 and U (so that block2's determinant conventions are verified, not assumed).
"""

import argparse
import itertools
import sys
import tempfile

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes


STACK_MEM = 8 << 30  # block2's memory pool: the default 1 GiB is too small for 2L = 16 spin-orbital sites


def sector_rows(L, n):
    rows = np.array(list(itertools.combinations(range(L), n)), dtype=int).reshape(-1, n)
    occ = np.zeros((len(rows), L), dtype=int)
    occ[np.arange(len(rows))[:, None], rows] = 1
    return rows, occ


def sd_expansion(Ca, Cb):
    """Determinant codes (n_a n_b, L) with local state n_up + 2 n_dn and interleaved amplitudes of SD(Ca, Cb)."""
    L = Ca.shape[0]
    rows_a, occ_a = sector_rows(L, Ca.shape[1])
    rows_b, occ_b = sector_rows(L, Cb.shape[1])
    da = np.array([np.linalg.det(Ca[r]) for r in rows_a])
    db = np.array([np.linalg.det(Cb[r]) for r in rows_b])
    lower = np.tril(np.ones((L, L), dtype=int), -1)
    sign = 1 - 2 * ((occ_a @ lower @ occ_b.T) & 1)
    amplitudes = (da[:, None] * db[None, :] * sign).ravel()
    codes = (occ_a[:, None, :] + 2 * occ_b[None, :, :]).reshape(-1, L).astype(np.uint8)
    return codes, amplitudes


def spin_orbital_modes(codes):
    """(n, L) local codes -> (n, 2L) occupations, mode 2i up and 2i + 1 down."""
    n, L = codes.shape
    modes = np.zeros((n, 2 * L), dtype=np.uint8)
    modes[:, 0::2] = codes % 2
    modes[:, 1::2] = codes // 2
    return modes


class Block2:
    """block2 states and operators in SZ (site = orbital) or SGF (site = spin orbital) mode."""

    def __init__(self, case, L, nelec, h1, U, scratch):
        self.case, self.L, self.h1, self.U = case, L, h1, U
        N = int(sum(nelec))
        if case == "sz":
            self.driver = DMRGDriver(stack_mem=STACK_MEM, scratch=scratch, symm_type=SymmetryTypes.SZ, n_threads=1)
            self.driver.initialize_system(n_sites=L, n_elec=N, spin=int(nelec[0] - nelec[1]))
        else:
            self.driver = DMRGDriver(stack_mem=STACK_MEM, scratch=scratch, symm_type=SymmetryTypes.SGF, n_threads=1)
            self.driver.initialize_system(n_sites=2 * L, n_elec=N)
        self.identity = self.driver.get_identity_mpo()
        self.count = 0

    def state(self, codes, amplitudes):
        self.count += 1
        dets = codes if self.case == "sz" else spin_orbital_modes(codes)
        return self.driver.get_mps_from_csf_coefficients(dets, amplitudes, f"S{self.count}", dot=2, iprint=0)

    def overlap(self, bra, ket):
        return self.driver.expectation(bra, self.identity, ket)

    def hubbard_mpo(self):
        b, L = self.driver.expr_builder(), self.L
        for i, j in zip(*np.nonzero(self.h1)):
            if self.case == "sz":
                b.add_term("cd", [i, j], self.h1[i, j])
                b.add_term("CD", [i, j], self.h1[i, j])
            else:
                for s in (0, 1):
                    b.add_term("CD", [2 * i + s, 2 * j + s], self.h1[i, j])
        if self.case == "sz":
            b.add_term("cdCD", np.array([[i] * 4 for i in range(L)]).ravel(), self.U)
        else:
            b.add_term("CDCD", np.array([[2 * i, 2 * i, 2 * i + 1, 2 * i + 1] for i in range(L)]).ravel(), self.U)
        return self.driver.get_mpo(b.finalize(), iprint=0)

    def szsz_numerator(self, bra, ket, cutoff=1e-20):
        """<bra|S^z_i S^z_j|ket> (L, L) from the transition 2-PDM, without dividing by the overlap."""
        L, mask = self.L, (0, 0, 1, 1)
        if self.case == "sz":
            aa, ab, ba, bb = self.driver.get_npdm(
                ket, pdm_type=2, bra=bra, npdm_expr=["cdcd", "cdCD", "CDcd", "CDCD"], mask=mask, cutoff=cutoff, iprint=0
            )
            return 0.25 * (aa - ab - ba + bb)
        n = np.asarray(self.driver.get_npdm(ket, pdm_type=2, bra=bra, npdm_expr="CDCD", mask=mask, cutoff=cutoff, iprint=0))
        s = np.array([0.5, -0.5])
        return np.einsum("a,b,iajb->ij", s, s, n.reshape(L, 2, L, 2))


    def szsz_numerator_mpo(self, bra, ket):
        """<bra|S^z_i S^z_j|ket> (L, L) as direct expectations of one two-site MPO per pair i <= j."""
        L = self.L
        out = np.zeros((L, L))
        for i in range(L):
            for j in range(i, L):
                b = self.driver.expr_builder()
                for si, sj in itertools.product((0, 1), repeat=2):
                    c = 0.25 * (1.0 if si == sj else -1.0)
                    if self.case == "sz":
                        b.add_term(("cd", "CD")[si] + ("cd", "CD")[sj], [i, i, j, j], c)
                    else:
                        b.add_term("CDCD", [2 * i + si, 2 * i + si, 2 * j + sj, 2 * j + sj], c)
                mpo = self.driver.get_mpo(b.finalize(), iprint=0)
                out[i, j] = out[j, i] = self.driver.expectation(bra, mpo, ket)
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file")
    parser.add_argument("--tol", type=float, default=1e-12, help="tolerance of the exact (MPO) route and the energy")
    parser.add_argument("--npdm-tol", type=float, default=1e-9, help="tolerance of the get_npdm route (see the header)")
    parser.add_argument("--cutoff", type=float, default=1e-20, help="get_npdm density-matrix cutoff")
    args = parser.parse_args()

    data = np.load(args.file)
    case, L, U = str(data["case"]), int(data["L"]), float(data["U"])
    nelec, h1 = tuple(int(x) for x in data["nelec"]), data["h1"]
    walkers_up, walkers_dn, weights = data["walkers_up"], data["walkers_dn"], data["weights"]
    with tempfile.TemporaryDirectory() as scratch:
        b2 = Block2(case, L, nelec, h1, U, scratch)
        trial = b2.state(data["trial_codes"], data["trial_values"])
        norm_trial = b2.overlap(trial, trial)
        energy = b2.driver.expectation(trial, b2.hubbard_mpo(), trial) / norm_trial
        print(f"case {case}: L={L} U={U} nelec={nelec}, {len(data['trial_values'])} trial determinants, "
              f"{len(weights)} walkers")
        print(f"  <T|H|T>/<T|T>: block2 {energy:.14f}, trot {float(data['trial_energy']):.14f}, "
              f"diff {abs(energy - float(data['trial_energy'])):.2e}")

        values, exact, overlap_errors = [], [], []
        for w in range(len(weights)):
            walker = b2.state(*sd_expansion(walkers_up[w], walkers_dn[w]))
            overlap = b2.overlap(trial, walker)
            # trot's trial is normalised and the walker is an orthonormal determinant (unit norm)
            overlap_errors.append(abs(overlap / np.sqrt(norm_trial) - data["trot_overlaps"][w]))
            values.append(b2.szsz_numerator(trial, walker, args.cutoff) / overlap)
            exact.append(b2.szsz_numerator_mpo(trial, walker) / overlap)
        values, exact = np.array(values), np.array(exact)

    trot_walkers, trot_block = data["trot_szsz_walkers"], data["trot_szsz_block"]
    e_energy = abs(energy - float(data["trial_energy"]))
    report = {}
    for name, v in (("get_npdm (transition 2-PDM)", values), ("MPO expectation", exact)):
        per_walker = np.abs(v - trot_walkers).max()
        block = np.abs(np.tensordot(weights, v, axes=1) / weights.sum() - trot_block).max()
        report[name] = (per_walker, block)
    print(f"  normalised overlap <T|walker>: max |block2 - trot| = {max(overlap_errors):.2e}")
    print(f"  <S^z_i S^z_j> values up to {np.abs(exact).max():.3f}; the mixed estimator, max |block2 - trot|:")
    for name, (per_walker, block) in report.items():
        print(f"    {name:28s} per walker {per_walker:.2e}   block average {block:.2e}")
    print(f"    get_npdm vs MPO expectation (both block2): {np.abs(values - exact).max():.2e}")
    tol_npdm, tol_mpo = args.npdm_tol, args.tol
    ok = (max(report["MPO expectation"]) < tol_mpo and max(report["get_npdm (transition 2-PDM)"]) < tol_npdm
          and e_energy < tol_mpo and max(overlap_errors) < tol_mpo)
    print(f"  {'AGREE' if ok else 'NOT WITHIN TOLERANCE'} (MPO route and energy < {tol_mpo:g}, get_npdm < {tol_npdm:g})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
