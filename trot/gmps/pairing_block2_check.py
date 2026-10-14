"""block2 reference for the MPS-CPMC pairing correlations (numpy + pyblock2 only: no trot, no JAX).

    ~/.trot/bin/python trot/gmps/pairing_block2_check.py file.npz [--mpo-walkers 2] [--tol 1e-12]

file.npz comes from trot/gmps/pairing_block2_export.py. The trial T is loaded from its determinant expansion and every
walker is expanded here, with numpy alone, as the determinant expansion of its two spin channels (alpha-block
amplitudes det(Ca[A]) det(Cb[B]) times the interleaving sign (-1)^K, K = #{i in A, j in B, j < i}).

The mixed pairing correlation is P_a[j, l] = <T|Delta_a^dag(j) Delta_a(l)|walker> / <T|walker> with
Delta_a(l) = sum_delta f_a(delta) [c_{l up} c_{l+delta dn} - c_{l dn} c_{l+delta up}] on the open lattice (site = x * Ly +
y; f_s = 1, f_d = +1 for x bonds and -1 for y bonds). Expanding the product, every term is a value of
D[x, y, z, w] = <T|c+_{x up} c+_{y dn} c_{z dn} c_{w up}|walker> / <T|walker>:

    <(c_{l up} c_{m dn} - c_{l dn} c_{m up})^dag (c_{l' up} c_{m' dn} - c_{l' dn} c_{m' up})>
        = D[l,m,m',l'] + D[l,m,l',m'] + D[m,l,m',l'] + D[m,l,l',m'].

Three references, all independent of trot:
- "numpy": exact Fock-space vectors (the operators applied literally), no MPS and no block2;
- "get_npdm": block2's transition 2-PDM, npdm_expr "cCDd" in SZ mode (case "sz": a trial with (N_up, N_dn) labels) or
  "CCDD" in SGF mode on 2L spin orbitals, 2i up and 2i + 1 down (case "nosz": the trial is not an S_z eigenstate);
- "MPO": block2 expectations of four-operator MPOs, exact for any walker (a few walkers only: one MPO per term).
get_npdm drops tiny Schmidt weights of near-product walkers in its density-matrix rotation, so it agrees to ~1e-10 relative
(~1e-9 absolute for values of order 10), not to rounding; the MPO and numpy routes do. Also checked: the normalised overlap <T|walker> and <T|H|T>/<T|T> (a Hubbard
MPO built here from h1 and U), so that block2's determinant conventions are verified, not assumed.
"""

import argparse
import itertools
import sys
import tempfile

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

STACK_MEM = 40 << 30  # block2's memory pool: the default 1 GiB, and 8 GiB, is too small for the full 2L = 16 spin-orbital CCDD 2-PDM


def sector_rows(L, n):
    rows = np.array(list(itertools.combinations(range(L), n)), dtype=int).reshape(-1, n)
    occ = np.zeros((len(rows), L), dtype=int)
    occ[np.arange(len(rows))[:, None], rows] = 1
    return rows, occ


def sd_expansion(Ca, Cb):
    """Determinant codes (n_a n_b, L), local state n_up + 2 n_dn, and the interleaved amplitudes of SD(Ca, Cb)."""
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


def full_vector(codes, amplitudes, L):
    v = np.zeros(4**L)
    v[(codes.astype(np.int64) * 4 ** np.arange(L - 1, -1, -1)).sum(axis=1)] = amplitudes
    return v


_ANNIHILATE = {}


def annihilate_full(vector, L, site, spin):
    """c_{site, spin}|vector> on the full interleaved vector: mode 2 site + spin, sign (-1)^(occupied modes before)."""
    key = (L, site, spin)
    if key not in _ANNIHILATE:
        digits = np.indices((4,) * L).reshape(L, -1)
        flat = np.zeros((L, 2, digits.shape[1]), dtype=int)
        flat[:, 0], flat[:, 1] = digits % 2, digits // 2
        flat = flat.reshape(2 * L, -1)
        mode = 2 * site + spin
        has = flat[mode] == 1
        _ANNIHILATE[key] = (has, ((-1) ** flat[:mode].sum(axis=0))[has],
                            (np.arange(digits.shape[1]) - (1 if spin == 0 else 2) * 4 ** (L - 1 - site))[has])
    has, sign, target = _ANNIHILATE[key]
    out = np.zeros_like(vector)
    out[target] = sign * vector[has]
    return out


def lattice_terms(Lx, Ly):
    """(l, m, f_s, f_d) of every neighbour m = l + delta of every site l (open boundaries), delta = (+-1, 0), (0, +-1)."""
    out = []
    for x, y in itertools.product(range(Lx), range(Ly)):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if 0 <= x + dx < Lx and 0 <= y + dy < Ly:
                out.append((x * Ly + y, (x + dx) * Ly + y + dy, 1.0, 1.0 if dx else -1.0))
    return out


def pairing_from_values(value, Lx, Ly):
    """(2, L, L) from value(l, m, l2, m2) = <(c_{l up} c_{m dn} - c_{l dn} c_{m up})^dag (c_{l2 up} c_{m2 dn} - c_{l2 dn} c_{m2 up})>
    (divided by the overlap), summed over the neighbours with the form factors f_s = 1 and f_d = +1 (x), -1 (y)."""
    L = Lx * Ly
    P = np.zeros((2, L, L))
    terms = lattice_terms(Lx, Ly)
    for l, m, fs, fd in terms:
        for l2, m2, fs2, fd2 in terms:
            v = value(l, m, l2, m2)
            P[0, l, l2] += fs * fs2 * v
            P[1, l, l2] += fd * fd2 * v
    return P


def pair_value_from_D(D):
    """The four D entries of one pair of oriented bonds: D[l,m,m2,l2] + D[l,m,l2,m2] + D[m,l,m2,l2] + D[m,l,l2,m2]."""
    return lambda l, m, l2, m2: D[l, m, m2, l2] + D[l, m, l2, m2] + D[m, l, m2, l2] + D[m, l, l2, m2]


class Block2:
    """block2 states and operators in SZ (site = orbital) or SGF (site = spin orbital) mode."""

    def __init__(self, case, L, nelec, h1, U, scratch):
        self.case, self.L, self.h1, self.U = case, L, h1, U
        N = int(sum(nelec))
        sym = SymmetryTypes.SZ if case == "sz" else SymmetryTypes.SGF
        self.driver = DMRGDriver(stack_mem=STACK_MEM, scratch=scratch, symm_type=sym, n_threads=1)
        if case == "sz":
            self.driver.initialize_system(n_sites=L, n_elec=N, spin=int(nelec[0] - nelec[1]))
        else:
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

    def D_npdm(self, bra, ket, cutoff):
        """D[x, y, z, w] = <bra|c+_{x up} c+_{y dn} c_{z dn} c_{w up}|ket> from the transition 2-PDM."""
        if self.case == "sz":  # a list of one array in SZ mode
            out = self.driver.get_npdm(ket, pdm_type=2, bra=bra, npdm_expr="cCDd", cutoff=cutoff, iprint=0)
            return np.asarray(out[0] if isinstance(out, list) else out)
        M = np.asarray(self.driver.get_npdm(ket, pdm_type=2, bra=bra, npdm_expr="CCDD", cutoff=cutoff, iprint=0))
        return M[0::2, 1::2, 1::2, 0::2]  # (2x, 2y + 1, 2z + 1, 2w): up, down, down, up

    def pair_value_mpo(self, bra, ket, l, m, l2, m2):
        """The four D entries of (l, m; l2, m2) as one MPO expectation (the operators c+_x up c+_y dn c_z dn c_w up)."""
        b = self.driver.expr_builder()
        for x, y, z, w in ((l, m, m2, l2), (l, m, l2, m2), (m, l, m2, l2), (m, l, l2, m2)):
            if self.case == "sz":
                b.add_term("cCDd", [x, y, z, w], 1.0)
            else:
                b.add_term("CCDD", [2 * x, 2 * y + 1, 2 * z + 1, 2 * w], 1.0)
        return self.driver.expectation(bra, self.driver.get_mpo(b.finalize(), iprint=0), ket)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file")
    parser.add_argument("--tol", type=float, default=1e-12, help="tolerance of the exact routes and the energy")
    parser.add_argument("--npdm-tol", type=float, default=1e-8, help="tolerance of the get_npdm route (see the header)")
    parser.add_argument("--cutoff", type=float, default=1e-20, help="get_npdm density-matrix cutoff")
    parser.add_argument("--mpo-walkers", type=int, default=2, help="walkers for the (slow) MPO route")
    args = parser.parse_args()

    data = np.load(args.file, allow_pickle=False)
    case, L, U = str(data["case"]), int(data["L"]), float(data["U"])
    Lx, Ly = int(data["Lx"]), int(data["Ly"])
    nelec, h1 = tuple(int(x) for x in data["nelec"]), data["h1"]
    walkers_up, walkers_dn, weights = data["walkers_up"], data["walkers_dn"], data["weights"]
    n_walkers = len(weights)
    trot_walkers, trot_block = data["trot_pairing_walkers"], data["trot_pairing_block"]
    trial_vec = full_vector(data["trial_codes"], data["trial_values"], L)

    with tempfile.TemporaryDirectory() as scratch:
        b2 = Block2(case, L, nelec, h1, U, scratch)
        trial = b2.state(data["trial_codes"], data["trial_values"])
        norm_trial = b2.overlap(trial, trial)
        energy = b2.driver.expectation(trial, b2.hubbard_mpo(), trial) / norm_trial
        print(f"case {case}: {Lx}x{Ly} U={U} nelec={nelec}, {len(data['trial_values'])} trial determinants, "
              f"{n_walkers} walkers")
        print(f"  <T|H|T>/<T|T>: block2 {energy:.14f}, trot {float(data['trial_energy']):.14f}, "
              f"diff {abs(energy - float(data['trial_energy'])):.2e}")

        exact, npdm, mpo, overlap_errors = [], [], {}, []
        for w in range(n_walkers):
            codes, amplitudes = sd_expansion(walkers_up[w], walkers_dn[w])
            walker = b2.state(codes, amplitudes)
            overlap = b2.overlap(trial, walker)
            overlap_errors.append(abs(overlap / np.sqrt(norm_trial) - data["trot_overlaps"][w]))

            # numpy: the operators applied literally to the Fock vectors
            wv = full_vector(codes, amplitudes, L)
            bra = {(l, s): annihilate_full(trial_vec, L, l, s) for l in range(L) for s in (0, 1)}
            ket = {(l, s): annihilate_full(wv, L, l, s) for l in range(L) for s in (0, 1)}
            D = np.zeros((L,) * 4)
            for x, y, z, v in itertools.product(range(L), repeat=4):
                # <T|c+_{x up} c+_{y dn} c_{z dn} c_{v up}|W> = <c_{y dn} c_{x up} T | c_{z dn} c_{v up} W>
                D[x, y, z, v] = annihilate_full(bra[(x, 0)], L, y, 1) @ annihilate_full(ket[(v, 0)], L, z, 1)
            D /= trial_vec @ wv
            exact.append(pairing_from_values(pair_value_from_D(D), Lx, Ly))

            npdm.append(pairing_from_values(pair_value_from_D(b2.D_npdm(trial, walker, args.cutoff) / overlap), Lx, Ly))
            if w < args.mpo_walkers:
                cache = {}

                def value(l, m, l2, m2, walker=walker, overlap=overlap, cache=cache):
                    key = (tuple(sorted((l, m))), tuple(sorted((l2, m2))))  # S_uv is symmetric in u, v
                    if key not in cache:
                        cache[key] = b2.pair_value_mpo(trial, walker, *key[0], *key[1]) / overlap
                    return cache[key]
                mpo[w] = pairing_from_values(value, Lx, Ly)
                print(f"  walker {w}: MPO route done ({len(cache)} MPOs)", flush=True)
        exact, npdm = np.array(exact), np.array(npdm)

    def block(P):
        return np.tensordot(weights, P, axes=1) / weights.sum()

    e_energy = abs(energy - float(data["trial_energy"]))
    print(f"  normalised overlap <T|walker>: max |block2 - trot| = {max(overlap_errors):.2e}")
    print(f"  pairing correlations up to {np.abs(exact).max():.3f}; max |reference - trot|:")
    rows = {"numpy (exact Fock vectors)": (exact, None), "block2 get_npdm": (npdm, None)}
    report = {}
    for name, (P, _) in rows.items():
        report[name] = (np.abs(P - trot_walkers).max(), np.abs(block(P) - trot_block).max())
        print(f"    {name:28s} per walker {report[name][0]:.2e}   block average {report[name][1]:.2e}")
    if mpo:
        ws = sorted(mpo)
        err = max(np.abs(mpo[w] - trot_walkers[w]).max() for w in ws)
        print(f"    {'block2 MPO expectations':28s} per walker {err:.2e}   ({len(ws)} walkers)")
    print(f"    get_npdm vs numpy exact: {np.abs(npdm - exact).max():.2e}")
    ok = (report["numpy (exact Fock vectors)"][0] < args.tol and report["block2 get_npdm"][0] < args.npdm_tol
          and (not mpo or err < args.tol) and e_energy < args.tol and max(overlap_errors) < args.tol)
    print(f"  {'AGREE' if ok else 'NOT WITHIN TOLERANCE'} (exact routes and energy < {args.tol:g}, get_npdm < {args.npdm_tol:g})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
