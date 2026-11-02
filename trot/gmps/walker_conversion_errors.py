"""Frozen vs dense walker conversion: state infidelity and exact local-energy errors against the walker bond.

The script version of frozen_vs_dense_walkers_6x6.ipynb, for the doped U = 8 Hubbard runs of uhf_4x4_doped.py,
uhf_6x6_doped.py and uhf_8x8_doped.py. One lattice per run, one GPU job each:

    ./submit_ex_2d.sh h100 walker_conversion_errors.py 4      # 4 (4x4), 6 (6x6) or 8 (8x8)

It takes --walkers random walkers of the run's last saved snapshot and converts them with the production engines
(frozen sector counts planned on the trial's natural orbitals, and dense per-walker bonds) at every bond of --chis,
for the adaptive and rank_exact gate plans. For every (plan, method, chi, walker) it measures:
- the total norm loss and the exact-state infidelity 1 - F against the original Slater walker;
- the local energy against the run's trial, and its error against the walker's exact local energy;
- the fidelity bound |dE| <= sqrt(1 - F) * residual / |normalized trial overlap| and the direction factor.
Nothing is plotted: everything goes into one npz, keys at the end of this docstring.

Trials. The 4x4 and 6x6 runs sampled with the CAS trial of the degenerate e = 0 shell: NCORE levels below it doubly
occupied, NACT active orbitals holding the last electron of each spin. The 8x8 run used the RHF determinant of the 28
lowest levels; it is written here as a CAS with one active orbital (the 28th level, which closes the degenerate pair
of levels 27 and 28), so that the same code measures all three.

Transitions. Every local energy needs <D|W> and <D|n_s|W> between a trial determinant D and a converted walker MPS W,
and the infidelity needs <walker|W>.
- 4x4 (7 electrons per spin) and 6x6 (16): exact, by notebook_walker_tools.determinant_mps_transition (cost 2**N, on
  the host), as in the notebook.
- 8x8 (28 per spin): 2**28 subsets per bond state, out of reach. The walkers and the trial determinants are therefore
  converted at the bond --reference-chi (default 1024; dense own Schmidt states, rank_exact gates, on the GPU), and the
  transitions are MPS-MPS contractions against those references. They are built one channel at a time; the walkers'
  are kept on the host and moved to the GPU one at a time, the trial determinants' stay on the GPU. At bond 4096 an
  8x8 reference is ~11 GB, so --mem must hold 2 x --walkers of them on top of the compiles (81.6 GB peak at 2048).
- reference_* in the npz check that reference against exact quantities of the original walkers: the trial overlap
  (exact Slater algebra) and the local energy.
- --reference-chi on 4x4 or 6x6 runs the reference path there too, to compare it with the exact one.

Output (walker_errors/walker_errors_<L>x<L>[_ref<chi>].npz next to this script; --out to change):
    plans, methods, chis, sample_indices, weights (the sample's population weights, normalized), snapshot, block
    norm2, loss, bond_loss, infidelity, state_error, energies, overlaps, energy_error, normalized_overlap,
      fidelity_energy_bound, energy_direction_factor: (plan, method, chi, walker)
    compile_s, run_ms: (plan, method, chi), conversion of the whole sample (compile times are cache loads on a rerun)
    E_exact, overlaps_exact: (walker,), the original walkers' local energy and trial overlap
    uncapped_plans, plan_loss, uncapped_energy_error: (3 plans, walker), the gate circuits alone (no bond cap)
    entropy (walker, cut) of the sample, entropy_population_mean (cut,), cas_energy, trial_mean, trial_variance
    reference path only: reference_walker_norm_loss (spin, walker), reference_trial_norm_loss (spin, determinant),
      reference_overlap_error (walker,) relative, reference_energy_error (walker,)
The file is rewritten after every (plan, method, chi), so a run that times out keeps what it measured (NaN elsewhere).
"""

from trot.config import configure_once

configure_once(single_precision=False, quiet=True)

import argparse
import os
import re
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from trot.gmps import dense_conversion as dc, engine, utils
from trot.gmps.notebook_walker_tools import cas_trial_moments, determinant_mps_transition
from trot.ham.hubbard import HamHubbard
from trot.meas.multi_ghf import energy_kernel_hubbard_u
from trot.prop.blocks import load_prop_state_npz
from trot.trial.mps import natural_orbitals
from trot.trial.multi_ghf import MultiGhfTrial, overlap_u

# persistent compile cache (as ex_6x6_doped_diag.py): a rerun with the same shapes loads the compiled conversions
jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/trot_jax_compile"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

ROOT = Path(__file__).resolve().parents[2]
WALKERS = Path("/mnt/ceph/users/fnappi/trot_walkers")
SYSTEMS = {
    4: dict(nelec=7, ncore=6, nact=4, folder="uhf_4x4_doped_U8", prefix="walkers", reference_chi=0),
    6: dict(nelec=16, ncore=15, nact=6, folder="uhf_6x6_doped_U8", prefix="walkers_cas", reference_chi=0),
    8: dict(nelec=28, ncore=27, nact=1, folder="uhf_8x8_doped_U8", prefix="walkers", reference_chi=1024),
}
PLANS, METHODS = ("adaptive", "rank_exact"), ("frozen", "dense")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("side", type=int, choices=sorted(SYSTEMS), help="the lattice: 4 (4x4), 6 (6x6) or 8 (8x8)")
parser.add_argument("--walkers", type=int, default=10, help="random walkers of the snapshot (default 10)")
parser.add_argument("--seed", type=int, default=0, help="seed of the random choice of walkers")
parser.add_argument("--snapshot", type=Path, default=None, help="walker snapshot (default: the run's last one)")
parser.add_argument("--chis", type=int, nargs="+", default=[16, 32, 64, 128, 256], help="per-spin walker bonds")
parser.add_argument("--reference-chi", type=int, default=None,
                    help="bond of the reference MPS the transitions are taken against; 0: exact transitions "
                         "(default: 0 on 4x4 and 6x6, 1024 on 8x8)")
parser.add_argument("--out", type=Path, default=None)
args = parser.parse_args()

SIDE, U = args.side, 8.0
system = SYSTEMS[SIDE]
L, N, NCORE, NACT = SIDE * SIDE, system["nelec"], system["ncore"], system["nact"]
CHIS = tuple(args.chis)
REFERENCE_CHI = system["reference_chi"] if args.reference_chi is None else args.reference_chi
OUT = args.out or ROOT / "trot/gmps/walker_errors" / (
    f"walker_errors_{SIDE}x{SIDE}" + (f"_ref{REFERENCE_CHI}" if REFERENCE_CHI else "") + ".npz")
OUT.parent.mkdir(parents=True, exist_ok=True)
try:
    revision = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
except (OSError, subprocess.CalledProcessError):
    revision = "unknown"
print(f"{SIDE}x{SIDE}, U = {U:g}, {N} + {N} electrons; bonds {CHIS}; transitions "
      f"{'against chi ' + str(REFERENCE_CHI) + ' references' if REFERENCE_CHI else 'exact'}; device "
      f"{jax.devices()[0].device_kind}; revision {revision}", flush=True)


# ============================================================================
# 1. Saved walkers: a random sample of the last snapshot
# ============================================================================

def orthonormal(C):
    Q, R = np.linalg.qr(C)
    return Q * np.where(np.diagonal(R, axis1=-2, axis2=-1) < 0, -1.0, 1.0)[..., None, :]


DATA_DIR = WALKERS / system["folder"]
numbered = re.compile(rf"{system['prefix']}_(\d+)")
files = sorted((p for p in DATA_DIR.glob(f"{system['prefix']}_*.npz") if numbered.fullmatch(p.stem)),
               key=lambda p: int(numbered.fullmatch(p.stem).group(1)))
if args.snapshot is None and not files:
    raise FileNotFoundError(f"no {system['prefix']}_<block>.npz walker snapshots in {DATA_DIR}")
WALKER_PATH = args.snapshot or files[-1]
match = numbered.fullmatch(WALKER_PATH.stem)
block = int(match.group(1)) if match else -1

state = load_prop_state_npz(WALKER_PATH)
raw_up, raw_dn = (np.asarray(x) for x in state.walkers)
assert raw_up.shape == raw_dn.shape and raw_up.shape[1:] == (L, N), raw_up.shape
population_weights = np.abs(np.asarray(state.weights))
all_up, all_dn = orthonormal(raw_up), orthonormal(raw_dn)
rng = np.random.default_rng(args.seed)
sample_indices = np.sort(rng.choice(len(raw_up), size=args.walkers, replace=False))
up, dn = all_up[sample_indices], all_dn[sample_indices]
up_device, dn_device = jnp.asarray(up), jnp.asarray(dn)
weights = population_weights[sample_indices] / population_weights[sample_indices].sum()
BATCH = len(sample_indices)

print(f"{len(files)} snapshots in {DATA_DIR}; selected {WALKER_PATH.name}, block {block}", flush=True)
print(f"population {raw_up.shape}; sampled indices {sample_indices}; saved energy estimate "
      f"{float(state.e_estimate):.6f}")
print(f"sample weights {weights}", flush=True)

# two-spin entanglement entropy of every walker at every cut
entropy = np.zeros((len(raw_up), L - 1))
for channel in (all_up, all_dn):
    for cut in range(1, L):
        nu = np.clip(np.linalg.svd(channel[:, :cut, :], compute_uv=False) ** 2, 0, 1)
        mixed = (nu > 1e-14) & (nu < 1 - 1e-14)
        x = np.where(mixed, nu, 0.5)
        entropy[:, cut - 1] += np.sum(np.where(mixed, -x * np.log(x) - (1 - x) * np.log1p(-x), 0), axis=1)


# ============================================================================
# 2. The run's trial (CAS of the e = 0 shell, or RHF on 8x8) and the gate plans
# ============================================================================

h1 = np.zeros((L, L))
for y in range(SIDE):
    for x in range(SIDE):
        i = y * SIDE + x
        for j in ([i + 1] if x + 1 < SIDE else []) + ([i + SIDE] if y + 1 < SIDE else []):
            h1[i, j] = h1[j, i] = -1
e, V = np.linalg.eigh(h1)
core, active = V[:, :NCORE], V[:, NCORE:NCORE + NACT]
rho = np.sum(core ** 2, axis=1)
Ecore = 2 * e[:NCORE].sum() + U * np.dot(rho, rho)
hcas = active.T @ h1 @ active + U * (active.T @ (rho[:, None] * active))
v = U * np.einsum("pi,pk,pj,pl->ijkl", active, active, active, active)
H = (Ecore * np.eye(NACT ** 2) + np.kron(hcas, np.eye(NACT)) + np.kron(np.eye(NACT), hcas)
     + v.reshape(NACT ** 2, NACT ** 2))
basis = []
for i in range(NACT):
    b = np.zeros((NACT, NACT))
    b[i, i] = 1
    basis.append(b.ravel())
for i in range(NACT):
    for j in range(i + 1, NACT):
        b = np.zeros((NACT, NACT))
        b[i, j] = b[j, i] = 1 / np.sqrt(2)
        basis.append(b.ravel())
B = np.column_stack(basis)
cas_energies, cas_vectors = np.linalg.eigh(B.T @ H @ B)
C = (B @ cas_vectors[:, 0]).reshape(NACT, NACT)
np.testing.assert_allclose(np.sum(C * C), 1, atol=3e-14)
rdms = (core @ core.T + active @ (C @ C.T) @ active.T, core @ core.T + active @ (C.T @ C) @ active.T)
reference, natural_occ = zip(*(natural_orbitals(g, N) for g in rdms))


def full_blocks_same_occ(R, occupation):
    """The adaptive plan's occupations with full remaining blocks: exact for every walker (a control)."""
    rows, refs = R.copy(), []
    for k, o in enumerate(occupation[:-1]):
        _, vectors = np.linalg.eigh(rows[k:] @ rows[k:].T)
        v = vectors[:, -1 if o else 0]
        refs.append(v.copy())
        utils._rotate_mode_to_front(rows, k, v)
    return utils.OrbitalPlan(np.asarray(occupation).copy(), np.arange(L, 1, -1), tuple(refs), True)


BASE_ADAPTIVE_EPS = 1e-10
adaptive = tuple(utils.make_orbital_plan(R, "adaptive", eps=BASE_ADAPTIVE_EPS) for R in reference)
plan_sets = {"adaptive": adaptive,
             "rank_exact": tuple(utils.make_orbital_plan(R, "rank_exact") for R in reference),
             "same occ, full blocks": tuple(full_blocks_same_occ(R, p.occupation) for R, p in zip(reference, adaptive))}
print(f"trial energy (CAS of {NACT} active orbitals, {NCORE} core): {cas_energies[0]:.10f}")
for name, ps in plan_sets.items():
    print(f"  {name}: gates {[int(p.block_sizes.sum() - len(p.block_sizes)) for p in ps]} (up, down)", flush=True)


# ============================================================================
# 3. Exact local energies of the sampled walkers, and the uncapped gate-plan errors
# ============================================================================

cas_determinants = [np.column_stack((core, active[:, i])) for i in range(NACT)]
for i, D in enumerate(cas_determinants):  # each determinant is an eigenstate of the one-body kinetic operator
    np.testing.assert_allclose(h1 @ D, D * np.r_[e[:NCORE], e[NCORE + i]], atol=1e-12)
pairs = [(i, j) for i in range(NACT) for j in range(NACT) if abs(C[i, j]) > 1e-12]
mo = np.zeros((len(pairs), 2 * L, 2 * N))
for k, (i, j) in enumerate(pairs):
    mo[k, :L, :N] = cas_determinants[i]
    mo[k, L:, N:] = cas_determinants[j]
cas_trial = MultiGhfTrial(ci_coeffs=jnp.asarray([C[i, j] for i, j in pairs]), mo_coeffs=jnp.asarray(mo),
                          green_real_dtype=jnp.float64, green_complex_dtype=jnp.complex128)
ham = HamHubbard(jnp.asarray(h1), U)
energy_batch = jax.jit(jax.vmap(lambda a, b: energy_kernel_hubbard_u((a, b), ham, None, cas_trial)))
overlap_batch = jax.jit(jax.vmap(lambda a, b: jnp.real(overlap_u((a, b), cas_trial))))
E_exact = np.asarray(energy_batch(up_device, dn_device))
overlaps_exact = np.asarray(overlap_batch(up_device, dn_device))
print(f"exact local energies {E_exact}")
print(f"exact trial overlaps {overlaps_exact}", flush=True)


def projected_orbitals(Q, plan):
    """The determinant the uncapped gate circuit of `plan` represents for walker Q, and its fidelity |gauge|^2. The
    rows are rotated as the conversion rotates them (R Q), the planned-empty rows dropped and the rotations undone:
    R^T Pi R Q, which is Q itself for exact plans."""
    angles, _ = utils.channel_angles(Q, plan, xp=np)
    R = np.eye(len(Q))
    for p, theta in angles:
        c, s = np.cos(theta), np.sin(theta)
        R[[p, p + 1]] = c * R[p] + s * R[p + 1], -s * R[p] + c * R[p + 1]
    rotated, kept = R @ Q, plan.occupation.astype(bool)
    return R[kept].T @ rotated[kept], np.linalg.det(rotated[kept]) ** 2


plan_losses, uncapped_energy_errors = {}, {}
for name, ps in plan_sets.items():
    projected = [[projected_orbitals(Q, p) for Q in channel] for channel, p in zip((up, dn), ps)]
    P = [np.stack([orbitals for orbitals, _ in spin]) for spin in projected]
    plan_losses[name] = 1 - np.prod([[fidelity for _, fidelity in spin] for spin in projected], axis=0)
    uncapped_energy_errors[name] = np.asarray(energy_batch(*map(jnp.asarray, P))) - E_exact
    print(f"uncapped {name}: plan loss max {plan_losses[name].max():.2e}, energy error max "
          f"{np.abs(uncapped_energy_errors[name]).max():.2e} (0 for the exact plans)", flush=True)

# The CAS shell is kinetically degenerate, so the exact spin-Schmidt rotation of its coefficients keeps every
# determinant a kinetic eigenstate and cuts the determinants the energy needs (2 per spin on 6x6, 1 on 8x8).
assert np.ptp(e[NCORE:NCORE + NACT]) < 1e-12, "the active shell must be degenerate"
uact, sact, vact = np.linalg.svd(C)
keep = sact > 1e-12
measurement_C = np.diag(sact[keep])
measurement_dets = ([np.column_stack((core, active @ uact[:, k])) for k in np.flatnonzero(keep)],
                    [np.column_stack((core, active @ vact.T[:, k])) for k in np.flatnonzero(keep)])
np.testing.assert_allclose(uact[:, keep] @ measurement_C @ vact[keep], C, atol=3e-14)
kappa = e[:NCORE].sum() + np.mean(e[NCORE:NCORE + NACT])
measurement_kinetic = (np.full(keep.sum(), kappa), np.full(keep.sum(), kappa))
NMEASURE = measurement_C.shape[0]
print(f"spin-Schmidt singular values {sact}; {NMEASURE} determinant(s) per spin in the energy", flush=True)


def cas_energy_from_transitions(ovs, dens):
    """(E_L, <T|W>) of a walker W from o_i = <D_i|W_spin> (2, NMEASURE) and d_i,s = <D_i|n_s|W_spin> (2, NMEASURE, L).
    Every D_i is a kinetic eigenstate, so <T|H|W> = sum_ij C_ij [(kappa_i + kappa_j) o_ia o_jb + U sum_s d_ia,s d_jb,s]
    holds for any W, the truncated (non-Gaussian) MPS walkers included."""
    s = np.einsum("ij,i,j", measurement_C, ovs[0], ovs[1])
    kinetic = np.einsum("ij,ij,i,j", measurement_C,
                        measurement_kinetic[0][:, None] + measurement_kinetic[1][None, :], ovs[0], ovs[1])
    interaction = U * np.einsum("ij,is,js", measurement_C, dens[0], dens[1])
    return (kinetic + interaction) / s, s


# check of the formula on an original saved walker, with determinant Green functions
check_ov, check_n = np.zeros((2, NMEASURE)), np.zeros((2, NMEASURE, L))
for spin, Q in enumerate((up[0], dn[0])):
    for i, D in enumerate(measurement_dets[spin]):
        O = D.T @ Q
        check_ov[spin, i] = np.linalg.det(O)
        check_n[spin, i] = check_ov[spin, i] * np.diag(Q @ np.linalg.solve(O, D.T))
check_energy, check_overlap = cas_energy_from_transitions(check_ov, check_n)
np.testing.assert_allclose(check_energy, E_exact[0], rtol=1e-8)
np.testing.assert_allclose(check_overlap, overlaps_exact[0], rtol=1e-8)
print("full-Hubbard CAS formula agrees with the production energy on an original saved walker", flush=True)

# full-Hubbard moments of the trial, for the fidelity bound
active_a = np.column_stack([D[:, -1] for D in measurement_dets[0]])
active_b = np.column_stack([D[:, -1] for D in measurement_dets[1]])
trial_mean, trial_variance = cas_trial_moments(h1, U, core, active_a, active_b, np.diag(measurement_C))
np.testing.assert_allclose(trial_mean, cas_energies[0], atol=3e-11)
residual_at_original = np.sqrt(trial_variance + np.abs(E_exact - trial_mean) ** 2)
print(f"full-Hubbard trial energy / variance: {trial_mean:.10f} / {trial_variance:.6f}", flush=True)


# ============================================================================
# 4. Transitions: exact (2**N subsets, 4x4 and 6x6) or against chi_ref references (8x8)
# ============================================================================

def channel_data(r, spin, w):
    A = [x[w] for x in r["out"][spin]]
    dense = len(r["out"]) == 4
    q = [x[w] if dense else x for x in r["labels"][spin]]
    return A, q, float(r["out"][2][spin][w])


def mps_transitions(bra, ket):
    """(<bra|ket>, [<bra|n_s|ket> for every site s]) of two real d=2 MPS, lists of (Dl, 2, Dr) tensors, without
    their gauges: the reference path's replacement for determinant_mps_transition."""
    right = [jnp.ones((1, 1))]
    for A, Bt in zip(bra[::-1], ket[::-1]):
        right.append(jnp.einsum("apc,bpd,cd->ab", A, Bt, right[-1]))
    right = right[::-1]  # right[s]: sites s, ..., L - 1
    left, density = jnp.ones((1, 1)), []
    for s, (A, Bt) in enumerate(zip(bra, ket)):
        density.append(jnp.einsum("ab,ac,bd,cd->", left, A[:, 1], Bt[:, 1], right[s + 1]))
        left = jnp.einsum("ab,apc,bpd->cd", left, A, Bt)
    return left[0, 0], jnp.stack(density)


def mps_overlap(bra, ket):
    """<bra|ket> of two real d=2 MPS, lists of (Dl, 2, Dr) tensors, without their gauges (one environment alive)."""
    left = jnp.ones((1, 1))
    for A, Bt in zip(bra, ket):
        left = jnp.einsum("ab,apc,bpd->cd", left, A, Bt)
    return left[0, 0]


fixed_bra_transitions = jax.jit(jax.vmap(mps_transitions, in_axes=(None, 0)))  # one bra, the batch of kets
single_transitions = jax.jit(mps_transitions)  # one pair: the reference check, a walker's reference as the ket
pair_overlap = jax.jit(mps_overlap)  # one pair: a walker's reference against its converted channel


def on_device(tensors):
    return [jnp.asarray(A) for A in tensors]


def reference_mps(Cs, plan):
    """The dense conversions of orthonormal channels Cs (batch, L, N) at REFERENCE_CHI with the rank_exact gates of
    `plan`: (tensors, scale, norm loss), tensors[w] the host list of channel w's (Dl, 2, Dr) site tensors and scale the
    gauge divided by the norm, so that scale * tensors approximates the normalized determinant. One channel at a time,
    each copied to the host: an 8x8 reference at bond 4096 is ~11 GB (40 sites of 4096 x 2 x 4096), so a batch of
    them does not fit on the GPU."""
    converter = dc.make_dense_converter(plan, REFERENCE_CHI, move="gram", snap=True)

    @jax.jit
    def convert(Q):
        A, _, gauge = converter.convert(Q)
        return A, gauge ** 2 * dc.mps_norm2(A), gauge

    tensors, scales, losses = [], [], []
    for Q in Cs:
        A, norm2, gauge = convert(jnp.asarray(Q))
        tensors.append([np.asarray(x) for x in A])  # blocks: the next channel starts on a free device
        norm2 = float(norm2)
        scales.append(float(gauge) / np.sqrt(norm2))
        losses.append(1 - norm2)
        del A
    return tensors, np.array(scales), np.array(losses)


def walker_overlaps(r):
    """(2, BATCH): <walker channel|converted channel>, the converted channel's gauge included."""
    gauges = np.asarray(r["out"][2])
    if REFERENCE_CHI:  # one walker's reference on the device at a time (float() waits for it)
        return np.array([[float(pair_overlap(on_device(walker_refs[spin][0][w]), [A[w] for A in r["out"][spin]]))
                          * walker_refs[spin][1][w] * gauges[spin, w] for w in range(BATCH)] for spin in (0, 1)])
    return np.array([[determinant_mps_transition(Q, *channel_data(r, spin, w), with_density=False)[0]
                      for w, Q in enumerate(channel)] for spin, channel in enumerate((up, dn))])


def trial_transitions(r):
    """<D_i|W> (2, NMEASURE, BATCH) and <D_i|n_s|W> (2, NMEASURE, BATCH, L) for the measurement determinants D_i."""
    gauges = np.asarray(r["out"][2])
    ovs, dens = np.zeros((2, NMEASURE, BATCH)), np.zeros((2, NMEASURE, BATCH, L))
    for spin in (0, 1):
        for i, D in enumerate(measurement_dets[spin]):
            if REFERENCE_CHI:
                tensors, scale = trial_refs[spin][i]
                if r.get("references"):  # the walkers' own references (host): one on the device at a time
                    pairs = [jax.device_get(single_transitions(tensors, on_device(ket))) for ket in r["out"][spin]]
                    o, d = np.array([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
                else:
                    o, d = fixed_bra_transitions(tensors, r["out"][spin])
                ovs[spin, i] = np.asarray(o) * scale * gauges[spin]
                dens[spin, i] = np.asarray(d) * (scale * gauges[spin])[:, None]
            else:
                for w in range(BATCH):
                    ovs[spin, i, w], dens[spin, i, w] = determinant_mps_transition(D, *channel_data(r, spin, w))
    return ovs, dens


def local_energies(r):
    """(energies, trial overlaps) of the converted walkers of r, both (BATCH,)."""
    ovs, dens = trial_transitions(r)
    return np.array([cas_energy_from_transitions(ovs[:, :, w], dens[:, :, w]) for w in range(BATCH)]).T


reference_checks = {}
if REFERENCE_CHI:
    start = time.perf_counter()
    walker_refs = [reference_mps(channel, p) for channel, p in zip((up, dn), plan_sets["rank_exact"])]
    trial_refs, trial_ref_losses = [[], []], np.zeros((2, NMEASURE))
    for spin in (0, 1):
        for i, D in enumerate(measurement_dets[spin]):
            tensors, scale, loss = reference_mps(D[None], utils.make_orbital_plan(D, "rank_exact"))
            trial_refs[spin].append((on_device(tensors[0]), float(scale[0])))  # these stay on the device
            trial_ref_losses[spin, i] = loss[0]
    # the references against exact quantities of the original walkers: trial overlap and local energy
    reference_r = dict(out=(walker_refs[0][0], walker_refs[1][0], (walker_refs[0][1], walker_refs[1][1])),
                       references=True)
    reference_energies, reference_overlaps = local_energies(reference_r)
    reference_checks = dict(reference_walker_norm_loss=np.stack([walker_refs[0][2], walker_refs[1][2]]),
                            reference_trial_norm_loss=trial_ref_losses,
                            reference_overlap_error=reference_overlaps / overlaps_exact - 1,
                            reference_energy_error=reference_energies - E_exact)
    print(f"chi {REFERENCE_CHI} references ({time.perf_counter() - start:.0f} s): walker norm loss max "
          f"{reference_checks['reference_walker_norm_loss'].max():.2e}, trial determinant norm loss max "
          f"{trial_ref_losses.max():.2e}; on the original walkers: trial overlap relative error max "
          f"{np.abs(reference_checks['reference_overlap_error']).max():.2e}, local energy error max "
          f"{np.abs(reference_checks['reference_energy_error']).max():.2e}", flush=True)
else:
    # 16-electron determinant / product-state check of the exact contraction
    test_rng = np.random.default_rng(24)
    test_occ = np.zeros(L, dtype=int)
    test_occ[test_rng.choice(L, N, replace=False)] = 1
    test_tensors = [np.eye(2)[o].reshape(1, 2, 1) for o in test_occ]
    test_charges = [np.array([q]) for q in np.r_[0, np.cumsum(test_occ)]]
    test_bra = measurement_dets[0][0]
    test_s, test_n = determinant_mps_transition(test_bra, test_tensors, test_charges)
    expected_s = np.linalg.det(test_bra[np.flatnonzero(test_occ)])
    np.testing.assert_allclose(test_s, expected_s, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(test_n, test_occ * expected_s, rtol=1e-10, atol=1e-14)
    print(f"{N}-electron determinant / product-state overlap and density checks passed", flush=True)


# ============================================================================
# 5. Convert the same walkers with the production engines, and measure
# ============================================================================

def mps_inner(a, b):
    env = np.ones((1, 1))
    for x, y in zip(a, b):
        env = np.einsum("ab,apc,bpd->cd", env, x, y, optimize=True)
    return float(env[0, 0])


def timed_compile(fn, args, repeats=2):
    """Compile fn for args, run it once and time `repeats` more runs: (outputs on the host, compile s, median run s)."""
    start = time.perf_counter()
    run = jax.jit(fn).lower(*args).compile()
    setup_s = time.perf_counter() - start
    out = jax.block_until_ready(run(*args))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(run(*args))
        times.append(time.perf_counter() - start)
    return jax.tree_util.tree_map(np.asarray, out), setup_s, np.median(times)


shape = (len(PLANS), len(METHODS), len(CHIS), BATCH)
results = {name: np.full(shape, np.nan) for name in (
    "norm2", "loss", "bond_loss", "infidelity", "state_error", "energies", "overlaps", "energy_error",
    "normalized_overlap", "fidelity_energy_bound", "energy_direction_factor")}
compile_s, run_ms = np.full(shape[:3], np.nan), np.full(shape[:3], np.nan)
oracle_relative = np.nan


def save():
    np.savez(OUT, side=SIDE, U=U, nelec=N, n_sites=L, chis=np.array(CHIS), plans=np.array(PLANS),
             methods=np.array(METHODS), snapshot=str(WALKER_PATH), block=block, seed=args.seed, revision=revision,
             reference_chi=REFERENCE_CHI, sample_indices=sample_indices, weights=weights,
             population_weights=population_weights, entropy=entropy[sample_indices],
             entropy_population_mean=np.average(entropy, axis=0, weights=population_weights),
             cas_energy=cas_energies[0], trial_mean=trial_mean, trial_variance=trial_variance,
             E_exact=E_exact, overlaps_exact=overlaps_exact, uncapped_plans=np.array(list(plan_sets)),
             plan_loss=np.array([plan_losses[k] for k in plan_sets]),
             uncapped_energy_error=np.array([uncapped_energy_errors[k] for k in plan_sets]),
             compile_s=compile_s, run_ms=run_ms, oracle_relative=oracle_relative, **results, **reference_checks)


for ip, mode in enumerate(PLANS):
    ps = plan_sets[mode]
    for ic, chi in enumerate(CHIS):
        bs = tuple(utils.plan_bonds(R, p, chi_max=chi) for R, p in zip(reference, ps))  # frozen on the trial
        frozen = engine.make_converter(*ps, *bs)
        cs = tuple(dc.make_dense_converter(p, chi, move="gram", snap=True) for p in ps)

        def dense_pair(a, b, cs=cs):
            A, qa, ga = cs[0].convert(a)
            Bt, qb, gb = cs[1].convert(b)
            return A, Bt, (ga, gb), (qa, qb)

        for im, (method, fn) in enumerate((("frozen", frozen.convert), ("dense", dense_pair))):
            out, compile_s[ip, im, ic], run_s = timed_compile(jax.vmap(fn), (up_device, dn_device))
            run_ms[ip, im, ic] = 1000 * run_s
            start = time.perf_counter()
            r = dict(out=out, labels=out[3] if method == "dense" else frozen.charges)

            norms = np.array([[mps_inner([A[w] for A in channel], [A[w] for A in channel]) for w in range(BATCH)]
                              for channel in out[:2]])
            norm2 = np.prod(np.asarray(out[2]) ** 2, axis=0) * np.prod(norms, axis=0)
            joint = np.prod(walker_overlaps(r), axis=0)
            infidelity = 1 - joint ** 2 / norm2
            energies, overlaps = local_energies(r)
            energy_error = energies - E_exact

            # |dE| <= sqrt(1 - F) * residual / |normalized trial overlap|, times a direction factor between 0 and 1
            normalized_overlap = np.abs(overlaps) / np.sqrt(norm2)
            bound = np.sqrt(np.maximum(infidelity, 0)) * residual_at_original / normalized_overlap
            direction = np.divide(np.abs(energy_error), bound, out=np.zeros(BATCH), where=bound > 0)

            values = dict(norm2=norm2, loss=1 - norm2, bond_loss=1 - np.prod(norms, axis=0), infidelity=infidelity,
                          state_error=np.sqrt(np.maximum(0, 1 + norm2 - 2 * joint)), energies=energies,
                          overlaps=overlaps, energy_error=energy_error, normalized_overlap=normalized_overlap,
                          fidelity_energy_bound=bound, energy_direction_factor=direction)
            for name, value in values.items():
                results[name][ip, im, ic] = value

            if mode == "adaptive" and method == "dense" and ic == 0:
                # production dense against the independent NumPy own-counts oracle, one saved channel
                own = utils.plan_bonds(up[0], ps[0], chi_max=chi)
                host, _, gh = engine.channel_mps_host(up[0], ps[0], own)
                oracle = gh * mps_inner(host, host)
                oracle_relative = abs(out[2][0][0] * mps_inner(host, [A[0] for A in out[0]]) - oracle) / abs(oracle)
                print(f"dense versus own-counts oracle relative overlap error: {oracle_relative:.2e}", flush=True)

            print(f"{mode:10s} {method:6s} chi {chi:4d}: weighted infidelity {np.dot(weights, infidelity):.3e}, "
                  f"norm loss {np.dot(weights, 1 - norm2):.3e}, RMS energy error "
                  f"{np.sqrt(np.dot(weights, energy_error ** 2)):.3e}; min infidelity {infidelity.min():.1e}, "
                  f"{int(np.sum(np.abs(energy_error) > bound + 3e-9))} bound violations; compile "
                  f"{compile_s[ip, im, ic]:.0f} s, conversion {run_ms[ip, im, ic]:.0f} ms, measurements "
                  f"{time.perf_counter() - start:.0f} s", flush=True)
            save()

print(f"wrote {OUT}", flush=True)
