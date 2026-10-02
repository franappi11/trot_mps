"""Why the 4x4 CPMC energy at tau = 0 lies below the trial energy for chi_T = 128 and above it for chi_T = 256.

Reads the production runs in --run-dir (the trial exports, the w64 logs) and tests two hypotheses.

H1  The chi_T = 128 DMRG trial breaks spin symmetry (antiferromagnetic spin density, gamma_a != gamma_b), so the
    natural-orbital determinant the walkers start from is Neel-like and low in energy (logged E_det = -1.86). The
    chi_T = 256 trial is nearly spin symmetric, its frontier natural orbitals are about half filled, and its natural
    determinant is essentially a restricted (RHF) one (logged E_det = +10.13).
H2  The tau = 0 energy is the mixed estimator of the start, E_L(phi) = <T|H|phi>/<T|phi>, not E_T = <T|H|T>:
    E_L - E_T = <R|phi>/<T|phi> with R = (H - E_T)|T>, which is orthogonal to T. Its sign belongs to the pair
    (T, phi), not to the trial, and |E_L - E_T| <= sigma_T sqrt((1 - F)/F) with F = |<T|phi>|^2 and
    sigma_T^2 = <T|H^2|T> - E_T^2. cos(R, phi_perp) = (E_L - E_T) <T|phi> / (sigma_T sqrt(1 - F)) is the signed
    fraction of that bound reached.

E_L and <T|phi> come from mps_cpmc_gpu's own probe (the runs' tau = 0 measurement), for each trial against four
starts: both natural determinants, the free-fermion RHF determinant and the Neel determinant. Each start is converted
on an adaptive orbital plan and chi_w = CHI_W kept counts built on itself, as the w64 runs converted theirs, so each
trial's own natural determinant must reproduce that run's logged tau = 0 numbers.

    sbatch --cpus-per-task=16 --mem=32G --time=00:30:00 -J sq4x4_start \
        --export=ALL,TARGET=sq4x4_start_energy.py run_mps_sweep.sh
"""
import argparse
import re
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from trot.gmps import mps_cpmc_2d_gpu as sg
from trot.gmps import mps_cpmc_gpu as g
from trot.ham.hubbard import HamHubbard
from trot.prop.hubbard_cpmc_ops import _build_prop_ctx

LX, LY, N_UP, N_DN, U = 4, 4, 8, 8, 8.0
CHIS = (128, 256)
CHI_W = 64  # kept counts per spin channel for each start, built on the start itself (the w64 runs' conversion)
EPS = 1.0e-10  # the runs' occupation_tolerance

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--run-dir", default="/mnt/ceph/users/fnappi/trot_walkers/sq4x4_U8")
args = parser.parse_args()
RUN_DIR = Path(args.run_dir)
N_SITES = LX * LY
STAGGER = (-1.0) ** np.sum(np.divmod(np.arange(N_SITES), LY), axis=0)  # (-1)^(x + y), site x*Ly + y


def load_trial(chi):
    z = np.load(RUN_DIR / f"dmrg_trial_sq4x4oo_U{U:g}_chi{chi}.npz")
    return dict(T=[z[f"T{i}"] for i in range(N_SITES)], q=tuple(z[f"q{i}"] for i in range(N_SITES + 1)),
                gamma=z["gamma"], natural=(z["reference_up"], z["reference_dn"]), h1=z["h1"],
                e_dmrg=float(z["e_dmrg"]))


def logged_start(chi):
    """The w64 run's own tau = 0 numbers: E_det of its start, <T|phi0> and E_L(phi0)."""
    text = (RUN_DIR / f"sq4x4oo_U{U:g}_T{chi}_w{CHI_W}_NOplan_NOstart_s1234.log").read_text()
    number = r"([-+.\deE]+)"
    e_det = float(re.search(rf"walkers start from the natural determinant, E={number};", text).group(1))
    overlap, e_local = map(float, re.search(rf"initial overlap {number}, local energy {number}", text).groups())
    return e_det, overlap, e_local


def grid(values):
    return "\n".join("      " + "  ".join(f"{values[x * LY + y]:+.3f}" for y in range(LY)) for x in range(LX))


def energy_parts(C, h1):
    """<phi|H|phi> of a determinant as (kinetic, U sum_i n_i,up n_i,dn), the formula build() logs as E."""
    Pa, Pb = (c @ c.T for c in C)
    return float(np.sum(h1 * (Pa + Pb))), float(U * np.diag(Pa) @ np.diag(Pb))


def trial_numbers(trial):
    """The trial's charge-labelled H|T> (as build() makes it), E_T and sigma_T."""
    W = sg.hubbard_mpo_from_h1(trial["h1"], U)
    ht = g.compress_mps_qn(*sg.trial_times_h(W, trial["T"], trial["q"]))
    T, H = tuple(jnp.asarray(a) for a in trial["T"]), tuple(jnp.asarray(a) for a in ht[0])
    norm = float(g.contract_real(T, T))
    e_t = float(g.contract_real(H, T)) / norm
    variance = float(g.contract_real(H, H)) / norm - e_t ** 2
    return ht, e_t, np.sqrt(max(variance, 0.0))


def start_numbers(trial, ht, C):
    """probe on one start: <T|phi> (T and phi normalised), E_L(phi) and the chi_w discarded weight on phi."""
    plans = [g.make_orbital_plan(c, "adaptive", EPS) for c in C]
    bonds = [g.plan_bonds(c, p, CHI_W, 0.0) for c, p in zip(C, plans)]
    prop = _build_prop_ctx(HamHubbard(h1=jnp.asarray(trial["h1"]), u=U), 0.01)  # unused by probe
    ops = g.make_gpu_ops(*plans, *bonds, trial["T"], trial["q"], ht, prop, linalg="native", walker_qr="native",
                         spin_batch=True, energy="blocked")
    out = jax.jit(ops.probe)(jnp.asarray(C[0]), jnp.asarray(C[1]), ops.data)
    return float(out["overlap"]), float(out["energy"]), max(b.reference_discarded_weight for b in bonds)


print(f"jax {jax.__version__} on {jax.default_backend()}; runs in {RUN_DIR}", flush=True)
trials = {chi: load_trial(chi) for chi in CHIS}
h1 = trials[CHIS[0]]["h1"]
assert all(np.array_equal(t["h1"], h1) for t in trials.values())
assert np.array_equal(h1, sg.square_hopping_matrix(LX, LY, 1.0)), "not the open 4x4 lattice"

# ---------------------------------------------------------------- H1: the trials' spin structure
print("\n== H1: spin structure of the trials (gamma from the trial exports)")
for chi, t in trials.items():
    ga, gb = t["gamma"]
    spin = np.diag(ga) - np.diag(gb)
    occ_a, occ_b = (np.linalg.eigvalsh(x)[::-1] for x in (ga, gb))
    Ca, Cb = t["natural"]
    print(f"chi_T={chi}: max|gamma_a - gamma_b| {np.abs(ga - gb).max():.3e};  staggered magnetisation "
          f"(1/N) sum (-1)^(x+y) (n_up - n_dn) {STAGGER @ spin / N_SITES:+.4f};  max|n_i - 1| "
          f"{np.abs(np.diag(ga + gb) - 1).max():.1e};  natural determinant ||P_a - P_b|| "
          f"{np.linalg.norm(Ca @ Ca.T - Cb @ Cb.T):.3f}")
    print("  spin density n_up - n_dn (rows x, columns y):")
    print(grid(spin))
    print(f"  natural occupations 5-12, alpha {np.round(occ_a[4:12], 3)}")
    print(f"                        beta  {np.round(occ_b[4:12], 3)}", flush=True)

# ---------------------------------------------------------------- H1: the starting determinants
eps, V = np.linalg.eigh(h1)
fermi = eps[N_UP - 1]
below, shell = np.flatnonzero(eps < fermi - 1e-8), np.flatnonzero(np.abs(eps - fermi) < 1e-8)
rhf = V[:, :N_UP]
starts = {f"NO(chi_T={chi})": t["natural"] for chi, t in trials.items()}
starts["RHF"] = (rhf, rhf.copy())
starts["Neel"] = (np.eye(N_SITES)[:, STAGGER > 0], np.eye(N_SITES)[:, STAGGER < 0])


def level_weights(c):
    """Occupied weight of one spin's determinant in the free-fermion levels below the Fermi shell, in it, above it."""
    w = np.einsum("ik,ij,jk->k", V, c @ c.T, V)
    return w[below].sum(), w[shell].sum(), w[shell.max() + 1:].sum()


print(f"\n== H1: the starting determinants (free fermions: {len(below)} levels below the Fermi level, a shell of "
      f"{len(shell)} at eps = {fermi:+.3f}; an RHF determinant puts weight {len(below)}, {N_UP - len(below)}, 0 "
      "below, in and above it)")
for name, C in starts.items():
    kinetic, hubbard = energy_parts(C, h1)
    weights = "; ".join(f"{s} " + ", ".join(f"{w:.3f}" for w in level_weights(c)) for s, c in zip("ab", C))
    print(f"{name:14s} E_det {kinetic + hubbard:+9.4f} = kinetic {kinetic:+9.4f} + U sum n_up n_dn {hubbard:+8.4f};"
          f"  weight below/in/above the shell: {weights};  ||P_a - P_b|| {np.linalg.norm(C[0] @ C[0].T - C[1] @ C[1].T):.3f}")
for chi in CHIS:
    kinetic, hubbard = energy_parts(trials[chi]["natural"], h1)
    print(f"  NO(chi_T={chi}) E_det logged by its w64 run: {logged_start(chi)[0]:+.9f} (here {kinetic + hubbard:+.9f})")

# ---------------------------------------------------------------- H2: tau = 0 local energies
print(f"\n== H2: tau = 0 local energy E_L(phi) = <T|H|phi>/<T|phi> of every start against every trial "
      f"(chi_w = {CHI_W}, plan and kept counts built on each start)", flush=True)
for chi, t in trials.items():
    start = time.perf_counter()
    ht, e_t, sigma = trial_numbers(t)
    print(f"\nchi_T={chi}: E_T = <T|H|T> {e_t:.10f} (DMRG Davidson {t['e_dmrg']:.10f}), sigma_T {sigma:.5f}, "
          f"H|T> bond {max(a.shape[0] for a in ht[0])} ({time.perf_counter() - start:.0f} s)", flush=True)
    print(f"  {'start':14s} {'<T|phi>':>10s} {'F':>7s} {'E_L':>10s} {'E_L - E_T':>10s} {'bound':>8s} "
          f"{'cos(R,phi_perp)':>15s}  chi_w discarded")
    for name, C in starts.items():
        start = time.perf_counter()
        overlap, e_local, discarded = start_numbers(t, ht, C)
        F = overlap ** 2
        bound = sigma * np.sqrt((1 - F) / F)
        cosine = (e_local - e_t) * overlap / (sigma * np.sqrt(1 - F))
        print(f"  {name:14s} {overlap:+10.6f} {F:7.4f} {e_local:+10.6f} {e_local - e_t:+10.5f} {bound:8.5f} "
              f"{cosine:+15.3f}  {discarded:.1e}  ({time.perf_counter() - start:.0f} s)", flush=True)
        if name == f"NO(chi_T={chi})":
            _, logged_overlap, logged_energy = logged_start(chi)
            print(f"  {'':14s} logged by the w64 run: <T|phi0> {logged_overlap:+.6e}, E_L {logged_energy:.10f}; "
                  f"difference {overlap - logged_overlap:+.1e}, {e_local - logged_energy:+.1e}")
