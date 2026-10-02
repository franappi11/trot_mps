"""Spin contamination and staggered magnetization of a cached DMRG trial (mps_cpmc_gpu's trial_cache).

The trial fixes N_alpha and N_beta (pyblock3's SZ symmetry), so S_z = 0, but S^2 is not fixed. Its spin
content comes from global rotations about y, R(beta) = exp(-i beta S^y). S^y_i is an even on-site operator,
so R(beta) is one real 4x4 matrix per site (rotation_matrix) and R(beta)|psi> keeps the trial's bonds.
For an S_z = 0 state, with x = cos(beta) and P_S the Legendre polynomials (<S,0|R|S,0> = P_S(cos beta)),

    f(x) = <psi|R|psi>      = sum_S w_S P_S(x),   w_S = weight of total spin S
    g(x) = <H psi|R|psi>    = sum_S e_S P_S(x),   e_S = <psi_S|H|psi_S>            ([H, R] = 0)
    h(x) = <psi|M.M R|psi>  = sum_S m_S P_S(x),   M = sum_i (-1)^i S_i, a scalar like H

f, g and h are polynomials of degree <= S_max = N/2 in x, so with K Gauss-Legendre nodes (x_k, c_k) every
coefficient a_S = (2S+1)/2 sum_k c_k a(x_k) P_S(x_k) with S <= 2K - 1 - S_max is exact; the default
K = 64 covers every S at L = 100. From them:

  * <S^2> = sum_S S(S+1) w_S, checked against |S^+ psi|^2 (equal to <S^2> when S_z = 0);
  * the singlet weight w_0 and the energy of the spin-projected trial, E_0 = e_0 / w_0;
  * for K' = 1, 2, 3, 4, 6, 8, 12, 16 the trial a CPMC with K' rotated copies would use,
    sum_k' c_k'/2 R(beta_k')|psi> = sum_S eps_S psi_S: its variational energy and singlet fraction;
  * the staggered magnetization: the profile <S^z_i>, m_s = <M^z>/L, the structure factors
    S^aa(pi) = <(M^a)^2>/L (a = x, y, z; all equal for a singlet) and, for the projected trial,
    S(pi) = m_0 / (3 L w_0). The projected trial has <S^z_i> = 0 on every site.

Two checks guard the rotation's sign convention, at one angle with the MPO applied directly:
<R psi|H|R psi> = <psi|H|psi> and <psi|H R psi> = <H psi|R psi>. Nothing is saved if either fails.
The numbers go to <out>/<tag>_spin_check.json and .npz before the plot <tag>_spin_check.pdf.

    # CPU only: one genx job per trial (never the login node), from trot/gmps
    for chi in 8 16 32; do
        sbatch -p genx -J spin_T$chi --time=00:05:00 --cpus-per-task=2 --mem=4G \\
            --export=ALL,TARGET=trial_spin_check.py run_mps_sweep.sh --trial-chi $chi
    done
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mps_cpmc_gpu as m  # noqa: E402  (the trial cache, hubbard_mpo, apply_mpo)

# as uhf_trial_cpmc_gpu.py: (L, U) at half filling, t = 1 -> DMRG reference
REFERENCES = {(100, 8.0): -32.545774923969844}
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a"]
INK2, GRID, SURFACE = "#52514e", "#e6e5e1", "#fcfcfb"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--L", type=int, default=100)
parser.add_argument("--U", type=float, default=8.0)
parser.add_argument("--t", type=float, default=1.0)
parser.add_argument("--trial-chi", type=int, required=True)
parser.add_argument("--dmrg-sweeps", type=int, default=30, help="as the L = 100 CPMC runs")
parser.add_argument("--dmrg-seed", type=int, default=0)
parser.add_argument("--nodes", type=int, default=64, help="Gauss-Legendre nodes in cos(beta)")
parser.add_argument("--trial-cache", default=str(HERE / "trial_cache"))
parser.add_argument("--out", default=str(HERE / "trial_spin_check"))
args = parser.parse_args()

L, U, CHI, K = args.L, args.U, args.trial_chi, args.nodes
N_UP = N_DN = L // 2
S_MAX = min(N_UP + N_DN, 2 * L - N_UP - N_DN) // 2
S_EXACT = min(S_MAX, 2 * K - 1 - S_MAX)   # w_S, e_S, m_S are exact up to this S
TAG = f"L{L}_U{U:g}_T{CHI}"
OUT = Path(args.out).resolve()
E_REF = REFERENCES.get((L, U)) if args.t == 1.0 else None
if S_EXACT < S_MAX:
    print(f"warning: with {K} nodes w_S, e_S, m_S are exact only up to S = {S_EXACT} (S_max = {S_MAX}); "
          f"use --nodes {S_MAX + 1} for all", flush=True)
start = time.time()

# ---- local operators in mps_cpmc_gpu's basis |0>, |up>, |dn>, |up dn> (index n_up + 2 n_dn), where
# c^dag_up c_dn = |up><dn| on a site. All are even, so a sum over sites needs no Jordan-Wigner string.
RAISE = np.zeros((4, 4)); RAISE[1, 2] = 1.0     # S^+_i
SZ_OP = np.diag([0.0, 0.5, -0.5, 0.0])
SX_OP = 0.5 * (RAISE + RAISE.T)
SY_REAL = 0.5 * (RAISE - RAISE.T)                # S^y_i = -i SY_REAL (real, antisymmetric)
NUMBER = np.diag([0.0, 1.0, 1.0, 2.0])
STAGGER = (-1.0) ** np.arange(L)


def rotation_matrix(beta):
    """exp(-i beta S^y) on one site = exp(-beta SY_REAL): |up> -> c|up> + s|dn>, |dn> -> -s|up> + c|dn>;
    |0> and |up dn> are on-site singlets and stay."""
    c, s = np.cos(beta / 2), np.sin(beta / 2)
    R = np.eye(4)
    R[1, 1], R[1, 2], R[2, 1], R[2, 2] = c, -s, s, c
    return R


def rotate(tensors, beta):
    R = rotation_matrix(beta)
    return [np.einsum("pq,lqr->lpr", R, A) for A in tensors]


def overlap(bra, ket):
    """<bra|ket> of two real open-boundary MPS."""
    env = np.ones((1, 1))
    for x, y in zip(bra, ket):
        env = np.tensordot(np.tensordot(env, x, axes=(0, 0)), y, axes=([0, 1], [0, 1]))
    return float(env.reshape(()))


def apply_site_sum(tensors, operators):
    """(sum_i operators[i] on site i)|psi> as a bond-2 MPO: channel 0 = not applied yet, 1 = applied."""
    out = []
    for i, (A, O) in enumerate(zip(tensors, operators)):
        Dl, d, Dr = A.shape
        B = np.zeros((2, Dl, d, 2, Dr))
        B[0, :, :, 0] = B[1, :, :, 1] = A
        B[0, :, :, 1] = np.einsum("pq,lqr->lpr", O, A)
        if i == 0:
            B = B[:1]
        if i == len(tensors) - 1:
            B = B[:, :, :, 1:]
        out.append(B.reshape(B.shape[0] * Dl, d, B.shape[3] * Dr))
    return out


def site_expectations(tensors, O):
    """<O_i> on every site of a normalized real MPS (O even and on-site)."""
    right = [np.ones((1, 1))]
    for A in reversed(tensors):
        right.append(np.einsum("apc,bpd,cd->ab", A, A, right[-1], optimize=True))
    right = right[::-1]                          # right[i]: sites i..L-1
    left, values = np.ones((1, 1)), []
    for i, A in enumerate(tensors):
        values.append(np.einsum("ab,apc,pq,bqd,cd->", left, A, O, A, right[i + 1], optimize=True))
        left = np.einsum("ab,apc,bpd->cd", left, A, A, optimize=True)
    return np.array(values)


# ---- the trial, as the CPMC runs load it
cfg = m.Config(L=L, n_up=N_UP, n_down=N_DN, hopping=args.t, interaction=U, trial_chi=CHI,
               dmrg_sweeps=args.dmrg_sweeps, dmrg_seed=args.dmrg_seed, trial_cache=args.trial_cache)
trial_file = m._trial_cache_file(cfg)
if not trial_file.exists():
    raise SystemExit(f"no cached trial {trial_file}; this script does not run DMRG")
psi, _, e_dmrg = m.load_or_run_trial(cfg)
norm = overlap(psi, psi)
psi[0] = psi[0] / np.sqrt(norm)
W = m.hubbard_mpo(L, args.t, U)
h_psi = m.apply_mpo(W, psi)
e_trial = overlap(h_psi, psi)
print(f"{TAG}: <psi|psi> = {norm:.12f}, DMRG Davidson {e_dmrg:.10f}, <psi|H|psi> = {e_trial:.10f}", flush=True)

# ---- the rotation's sign convention: R must commute with this H
beta_check = 1.0
r_psi = rotate(psi, beta_check)
checks = dict(beta=beta_check,
              energy_rotated=overlap(r_psi, m.apply_mpo(W, r_psi)) - e_trial,
              commutator=overlap(psi, m.apply_mpo(W, r_psi)) - overlap(h_psi, r_psi))
print(f"[H, R] checks at beta = {beta_check}: <R psi|H|R psi> - E = {checks['energy_rotated']:.2e}, "
      f"<psi|H R psi> - <H psi|R psi> = {checks['commutator']:.2e}", flush=True)
if max(abs(checks["energy_rotated"]), abs(checks["commutator"])) > 1e-8:
    raise SystemExit("the rotation does not commute with H: wrong sign convention, nothing saved")

# ---- f, g, h at the Gauss-Legendre nodes
spin_ops = {"x": SX_OP, "y": SY_REAL, "z": SZ_OP}   # y: <M^y a|M^y b> = <Y a|Y b> with Y = sum_i (-1)^i SY_REAL
stag_ops = {a: [s * O for s in STAGGER] for a, O in spin_ops.items()}
m_psi = {a: apply_site_sum(psi, ops) for a, ops in stag_ops.items()}
x, c = leggauss(K)
betas = np.arccos(x)
f, g, h = np.empty(K), np.empty(K), np.empty(K)
for k, beta in enumerate(betas):
    r_psi = rotate(psi, beta)
    f[k] = overlap(psi, r_psi)
    g[k] = overlap(h_psi, r_psi)
    h[k] = sum(overlap(m_psi[a], apply_site_sum(r_psi, ops)) for a, ops in stag_ops.items())
print(f"{K} nodes done ({time.time() - start:.0f} s)", flush=True)

spins = np.arange(S_MAX + 1)
coefficients = (2 * spins + 1) / 2 * (c[:, None] * legvander(x, S_MAX))   # (K, S_MAX + 1)
w_S, e_S, m_S = coefficients.T @ f, coefficients.T @ g, coefficients.T @ h

s2 = float(spins * (spins + 1) @ w_S)
raised = apply_site_sum(psi, [RAISE] * L)             # S^+|psi>
s2_direct = overlap(raised, raised)
m2 = {a: overlap(m_psi[a], m_psi[a]) for a in spin_ops}
checks.update(sum_w_minus_1=float(w_S.sum() - 1.0), sum_e_minus_energy=float(e_S.sum() - e_trial),
              sum_m_minus_MM=float(m_S.sum() - sum(m2.values())), s2_minus_direct=s2 - s2_direct,
              min_w=float(w_S.min()))
w0 = float(w_S[0])
e_projected = float(e_S[0] / w0)

few_nodes = []
for kp in (1, 2, 3, 4, 6, 8, 12, 16):
    xp, cp = leggauss(kp)
    eps = (cp[:, None] / 2 * legvander(xp, S_MAX)).sum(axis=0)   # the K'-node filter on each spin S
    weight = float(np.sum(eps ** 2 * w_S))
    few_nodes.append(dict(K=kp, e_variational=float(np.sum(eps ** 2 * e_S)) / weight, singlet_fraction=w0 / weight))

sz = site_expectations(psi, SZ_OP)
occupations = site_expectations(psi, NUMBER)
staggered = STAGGER * sz
bulk = slice(L // 4, 3 * L // 4)
magnetization = dict(m_s=float(staggered.mean()), m_s_bulk=float(staggered[bulk].mean()),
                     max_abs_sz=float(np.abs(sz).max()), sum_sz=float(sz.sum()), sum_n=float(occupations.sum()),
                     S_pi={a: m2[a] / L for a in spin_ops}, S_pi_projected=float(m_S[0] / w0) / (3 * L))

# ---- report and save
significant = [int(S) for S in spins if w_S[S] > 1e-10]
print(f"<S^2> = {s2:.6f} (|S^+ psi|^2 = {s2_direct:.6f}); singlet weight w_0 = {w0:.6f}")
print("  S   w_S           E_S")
for S in significant:
    print(f"  {S:<3d} {w_S[S]:.6e}  {e_S[S] / w_S[S]:.8f}")
print(f"energy: trial {e_trial:.8f}, spin-projected {e_projected:.8f} (change {e_projected - e_trial:+.2e})"
      + (f"; DMRG reference {E_REF:.8f}" if E_REF is not None else ""))
print("  K'  variational E of the K'-copy trial   singlet fraction")
for row in few_nodes:
    print(f"  {row['K']:<3d} {row['e_variational']:.8f}                        {row['singlet_fraction']:.6f}")
print(f"staggered magnetization m_s = {magnetization['m_s']:.3e} (bulk {magnetization['m_s_bulk']:.3e}), "
      f"max |<S^z_i>| = {magnetization['max_abs_sz']:.3e}; sum <S^z_i> = {magnetization['sum_sz']:.1e}, "
      f"sum <n_i> = {magnetization['sum_n']:.8f}")
print("S(pi): " + ", ".join(f"{a}{a} {v:.6f}" for a, v in magnetization["S_pi"].items())
      + f"; projected trial {magnetization['S_pi_projected']:.6f}")
print("checks: " + ", ".join(f"{k} {v:.1e}" for k, v in checks.items() if k != "beta"))

OUT.mkdir(parents=True, exist_ok=True)
result = dict(tag=TAG, trial_file=str(trial_file), L=L, U=U, t=args.t, n_up=N_UP, n_dn=N_DN, trial_chi=CHI,
              nodes=K, s_max=S_MAX, exact_up_to_S=S_EXACT, norm=norm, e_dmrg=e_dmrg, e_trial=e_trial,
              e_ref=E_REF, S2=s2, S2_direct=s2_direct, singlet_weight=w0, e_projected=e_projected,
              spin_weights={S: float(w_S[S]) for S in significant},
              spin_energies={S: float(e_S[S] / w_S[S]) for S in significant},
              few_node_trials=few_nodes, magnetization=magnetization, checks=checks)
json_path = OUT / f"{TAG}_spin_check.json"
tmp = json_path.with_suffix(".tmp.json")
tmp.write_text(json.dumps(result, indent=2) + "\n")
os.replace(tmp, json_path)
np.savez(OUT / f"{TAG}_spin_check.npz", x=x, c=c, beta=betas, f=f, g=g, h=h, w_S=w_S, e_S=e_S, m_S=m_S,
         sz=sz, occupations=occupations)
print(f"saved {json_path} and {TAG}_spin_check.npz ({time.time() - start:.0f} s)")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "font.size": 10})
    fig, (ax_m, ax_w) = plt.subplots(1, 2, figsize=(10.0, 3.6))
    ax_m.axhline(0.0, color=GRID, lw=1.0, zorder=0)
    ax_m.plot(np.arange(L), staggered, color=PALETTE[0], lw=1.5, marker="o", ms=3)
    ax_m.set_xlabel("site i")
    ax_m.set_ylabel(r"$(-1)^i\,\langle S^z_i\rangle$")
    ax_m.set_title(f"staggered moment, $m_s$ = {magnetization['m_s']:.2e}")
    shown = spins[w_S > 1e-15]
    ax_w.vlines(shown, 1e-16, w_S[shown], color=GRID, lw=1.0, zorder=0)
    ax_w.semilogy(shown, w_S[shown], ls="none", color=PALETTE[0], marker="o", ms=5)
    ax_w.set_ylim(bottom=max(1e-16, 0.3 * w_S[shown].min()))
    ax_w.set_xlabel("total spin S")
    ax_w.set_ylabel(r"weight $w_S$")
    ax_w.set_title(rf"spin weights, $\langle S^2\rangle$ = {s2:.3g}, $w_0$ = {w0:.4f}")
    fig.suptitle(rf"$\chi_T$ = {CHI} DMRG trial, L = {L}, U = {U:g}: E = {e_trial:.4f}, "
                 rf"spin-projected E = {e_projected:.4f}", color=INK2)
    fig.tight_layout()
    pdf = OUT / f"{TAG}_spin_check.pdf"   # a PDF: the module's matplotlib cannot rasterize text
    fig.savefig(pdf, bbox_inches="tight")
    print(f"saved {pdf}")
except Exception:
    traceback.print_exc()
    raise SystemExit(f"the plot failed; the numbers are in {json_path}")
