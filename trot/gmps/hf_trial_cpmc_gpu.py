"""CPMC for the open Hubbard chain with a UHF or RHF trial on the GPU, through trot's own Hubbard CPMC path.

The trot-native rewrite of uhf_trial_cpmc_gpu.py: same system, parameters and flags, with as little code of
its own as possible. trot does
  - the trial: trot.trial.ghf.GhfTrial with the spin-block-diagonal coefficients [[C_up, 0], [0, C_dn]] (the
    UHF determinant, or the RHF one with C_up = C_dn) and make_ghf_trial_ops, whose calc_green_u,
    calc_overlap_ratio and update_green are the fast updates;
  - the energy: make_ghf_meas_ops_hubbard, straight from the HamHubbard (no Cholesky form of U n_up n_dn);
  - the walkers: unrestricted, from trot's init_prop_state (the trial determinant, or the UHF one with
    --walker-start uhf through its initial_walkers);
  - propagation and blocks: trot.prop.cpmc (fast updates, the default) or cpmc_slow, and trot.prop.blocks.block;
  - the run and its statistics: trot.driver.run_qmc (equilibration, sampling, rejection of outlier blocks
    further than 10 spreads from the median, Gamma-method and blocking errors; the error bar is run_qmc's
    params.error_method, the Gamma method by default, and the blocking one is saved too).
Its own code: the mean-field SCF (uhf_cpmc.uhf_scf / rhf_scf, the notebook's; checked against trot's energy of the
tau = 0 walkers below), the start of the energy estimate for an RHF trial (see run()), saving and the plot.
tests/test_gmps_uhf_fast_update.py checks this path against uhf_trial_cpmc_gpu.py's, which it checks against the
notebook's propagator (trot.prop.cpmc_slow), block by block to 1e-9.

Differences from uhf_trial_cpmc_gpu.py: run_qmc runs everything in one call, so a job that ends early loses the
run (no resume: give --time its margin) and no walker state is saved; the GHF Green's function is (2L, 2L), twice
the data of the UHF (L, L) blocks, so a block can take up to about twice as long (time it first).

At the end the result goes to <out>/<tag>_run.npz (every block) and <tag>_result.json, and only then is the plot
<tag>_evolution.png drawn (a .pdf where the module's matplotlib cannot rasterize text); --plot-only redraws it
from the saved files. With --project-blocks N the log gives the wall time of a run of N blocks from this one's
(compile included, so a slight overestimate) and an --time of about twice that.

    # from trot/gmps; GPU jobs through run_mps_gpu.sh, CPU-only ones (the plot) on genx, never on the login node
    sbatch -J hf_time --time=00:20:00 --cpus-per-task=2 --mem=6G --export=ALL,TARGET=hf_trial_cpmc_gpu.py \
        run_mps_gpu.sh --trial rhf --eql 10 --blocks 20 --out hf_cpmc_test --project-blocks 1200   # timing test
    sbatch -J hf_rhf --time=<from the timing test> --cpus-per-task=2 --mem=6G \
        --export=ALL,TARGET=hf_trial_cpmc_gpu.py run_mps_gpu.sh --trial rhf                     # production
    sbatch -p genx -J hf_plot --time=00:10:00 --cpus-per-task=2 --mem=4G \
        --export=ALL,TARGET=hf_trial_cpmc_gpu.py run_mps_sweep.sh --trial rhf --plot-only        # redo the plot
"""
import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

_T0 = time.perf_counter()

if "--plot-only" in sys.argv:
    # nothing to compute: stay on the CPU with a small XLA thread pool (by default XLA starts about one thread per
    # core in each pool); jaxlib reads PJRT_NPROC for the pool size
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("PJRT_NPROC", "2")

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from trot.core.system import System
from trot.driver import run_qmc
from trot.gmps.uhf_cpmc import rhf_scf, uhf_scf
from trot.ham.hubbard import HamHubbard
from trot.meas.ghf import make_ghf_meas_ops_hubbard
from trot.prop import blocks, cpmc, cpmc_slow
from trot.prop.types import QmcParams
from trot.trial.ghf import GhfTrial, make_ghf_trial_ops

HERE = Path(__file__).resolve().parent
# references for open chains with this script's Hamiltonian: (L, n_up, n_dn, t, U) -> (energy, label)
REFERENCES = {
    (100, 50, 50, 1.0, 8.0): (-32.545774923969844, "DMRG χ = 200"),  # 30 sweeps, cpmc_L100/L100_U8/dmrg_reference.jsonl
}
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0", "#d4a20f", "#d6457a"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
# ---- system: open chain, as in mps_cpmc_new.py and the cpmc_L100 runs
parser.add_argument("--L", type=int, default=100)
parser.add_argument("--n-up", type=int, default=None, help="default: half filling")
parser.add_argument("--n-dn", type=int, default=None, help="default: half filling")
parser.add_argument("--t", type=float, default=1.0, help="hopping")
parser.add_argument("--U", type=float, default=8.0)
# ---- CPMC (trot defaults otherwise: pop-control damping, weight cap, SR after every block, error method)
parser.add_argument("--walkers", type=int, default=400)
parser.add_argument("--dt", type=float, default=0.005)
parser.add_argument("--steps", type=int, default=50, help="propagation steps per block (tau per block = steps * dt)")
parser.add_argument("--eql", type=int, default=200, help="equilibration blocks")
parser.add_argument("--blocks", type=int, default=1000, help="sampling blocks")
parser.add_argument("--weight-floor", type=float, default=1.0e-8,
                    help="overlap ratios at or below it are zeroed; as the cpmc_L100 runs (trot's default is 1e-3)")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--prop", default="fast", choices=["fast", "slow"],
                    help="fast updates (trot.prop.cpmc) or recomputed overlaps (trot.prop.cpmc_slow)")
parser.add_argument("--trial", default="uhf", choices=["uhf", "rhf"], help="trial determinant (walkers are unrestricted)")
parser.add_argument("--walker-start", default="trial", choices=["trial", "uhf"],
                    help="initial walkers: the trial determinant (trot's default) or the UHF one")
# ---- run control
parser.add_argument("--e-ref", type=float, default=None,
                    help="reference energy for the result and the plot (default: REFERENCES, if it has this chain)")
parser.add_argument("--out", default=str(HERE / "hf_cpmc_data"), help="folder for the blocks, result and plot")
parser.add_argument("--plot-only", action="store_true", help="no run: the plot from the saved blocks")
parser.add_argument("--project-blocks", type=int, default=0,
                    help="project this run's wall time to a run of this many blocks (eql + sampling)")
parser.add_argument("--compile-cache", default=os.path.expanduser("~/.cache/trot_jax_compile"),
                    help="JAX persistent compilation cache ('' = off)")
args = parser.parse_args()

L, T, U = args.L, args.t, args.U
N_UP = L // 2 if args.n_up is None else args.n_up
N_DN = L // 2 if args.n_dn is None else args.n_dn
N_WALKERS, DT, N_PROP, N_EQL, N_BLOCKS = args.walkers, args.dt, args.steps, args.eql, args.blocks
WEIGHT_FLOOR, SEED = args.weight_floor, args.seed
E_REF, REF_LABEL = REFERENCES.get((L, N_UP, N_DN, T, U), (None, None))
if args.e_ref is not None:
    E_REF, REF_LABEL = args.e_ref, None
PROP, TRIAL = args.prop, args.trial
WALKER_START = "trial" if TRIAL == "uhf" else args.walker_start       # for a UHF trial the two starts are the same
if TRIAL == "rhf" and N_UP != N_DN:
    raise SystemExit(f"--trial rhf needs n_up = n_dn, not {N_UP}, {N_DN}")

DATA_DIR = Path(args.out).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
TAG = (f"L{L}_N{N_UP}-{N_DN}_U{U:g}_nw{N_WALKERS}_dt{DT:g}_n{N_PROP}_s{SEED}"
       + ("" if TRIAL == "uhf" else f"_{TRIAL}trial") + ("_wuhf" if WALKER_START == "uhf" else "")
       + ("_fast" if PROP == "fast" else ""))
RUN_FILE = DATA_DIR / f"{TAG}_run.npz"
RESULT_FILE = DATA_DIR / f"{TAG}_result.json"
CONFIG = dict(L=L, N_UP=N_UP, N_DN=N_DN, T=T, U=U, N_WALKERS=N_WALKERS, DT=DT, N_PROP=N_PROP, N_EQL=N_EQL,
              N_BLOCKS=N_BLOCKS, WEIGHT_FLOOR=WEIGHT_FLOOR, SEED=SEED, TRIAL=TRIAL, WALKER_START=WALKER_START,
              PROP=PROP)
N_TOTAL = N_EQL + N_BLOCKS
DEVICE =f"{jax.default_backend()}: {jax.devices()[0].device_kind}"


def hms(seconds):
    minutes = max(10, 10 * math.ceil(seconds / 600))  # whole 10 minutes, at least 10
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def finite(x):
    return None if x is None or not np.isfinite(x) else float(x)


def run():
    """trot's run_qmc on the chain; returns what is saved."""
    t_setup = time.perf_counter()
    h1 = np.zeros((L, L))
    i = np.arange(L - 1)
    h1[i, i + 1] = h1[i + 1, i] = -T

    Ca, Cb, e_uhf, it, change = uhf_scf(h1, U, N_UP, N_DN)
    m = np.einsum("ik,ik->i", Ca, Ca) - np.einsum("ik,ik->i", Cb, Cb)
    print(f"UHF: E = {e_uhf:.10f} ({e_uhf / L:.6f} per site), {it} iterations, last density change {change:.1e}")
    print(f"staggered moment (n_up - n_dn), first sites: {np.round(m[:6], 3)}")
    if TRIAL == "rhf":
        C, e_trial, it, change = rhf_scf(h1, U, N_UP)
        print(f"RHF: E = {e_trial:.10f} ({e_trial / L:.6f} per site), {it} iterations, last density change {change:.1e}")
        Ta, Tb = C, C
    else:
        Ta, Tb, e_trial = Ca, Cb, e_uhf

    ham = HamHubbard(h1=jnp.asarray(h1), u=U)        # propagation and energy
    system = System(norb=L, nelec=(N_UP, N_DN), walker_kind="unrestricted")
    trial = GhfTrial(mo_coeff=jnp.asarray(np.block([[Ta, np.zeros((L, N_DN))], [np.zeros((L, N_UP)), Tb]])))
    trial_ops = make_ghf_trial_ops(system)            # overlap, rdm1 and the fast-update ops
    meas_ops = make_ghf_meas_ops_hubbard(system)
    if PROP == "fast":
        prop_ops = cpmc.make_prop_ops(ham, system.walker_kind, trial_ops)
    else:
        prop_ops = cpmc_slow.make_prop_ops(ham, system.walker_kind)
    params = QmcParams(dt=DT, n_walkers=N_WALKERS, n_prop_steps=N_PROP, n_blocks=N_BLOCKS, n_eql_blocks=N_EQL,
                       weight_floor=WEIGHT_FLOOR, seed=SEED)

    initial_walkers = None                  # trot's default: every walker the trial determinant
    if WALKER_START == "uhf":
        initial_walkers = (jnp.broadcast_to(jnp.asarray(Ca), (N_WALKERS, L, N_UP)),
                           jnp.broadcast_to(jnp.asarray(Cb), (N_WALKERS, L, N_DN)))
    state = prop_ops.init_prop_state(sys=system, ham_data=ham, trial_ops=trial_ops, trial_data=trial,
                                     meas_ops=meas_ops, params=params, initial_walkers=initial_walkers)
    e0 = float(state.e_estimate)
    if WALKER_START == "trial":
        # at tau = 0 every walker is the trial: checks the trial, the walker start and trot's Hubbard energy
        print(f"trot energy at tau = 0: {e0:.10f}, minus E_{TRIAL.upper()}: {e0 - e_trial:.1e}")
        assert abs(e0 - e_trial) < 1e-8
    else:
        print(f"trot energy at tau = 0 (mixed, <{TRIAL.upper()}|H|UHF>/<{TRIAL.upper()}|UHF>): {e0:.10f}")
    if TRIAL != "uhf":
        # trot starts its energy estimate (the population-control shift, and the centre of the +-sqrt(2/dt) window
        # outside which a walker's local energy is replaced by the estimate) at the tau = 0 energy. For the RHF trial
        # at U = 8 that is E_RHF = +73 at L = 100, about 100 above the ground state: within a few steps every local
        # energy is outside the window, each block energy then equals the estimate, and the estimate (which follows
        # the block energies) never moves. Start it at E_UHF, a variational energy about 8 above the ground state;
        # it then follows the block energies (moving average, trot's shift_ema), so the sampling phase does not
        # depend on the start.
        state = state._replace(e_estimate=jnp.full_like(state.e_estimate, e_uhf),
                               pop_control_ene_shift=jnp.full_like(state.pop_control_ene_shift, e_uhf))
        print(f"energy estimate and population-control shift start at E_UHF = {e_uhf:.6f}, not at {e0:.6f}")
    setup_s = time.perf_counter() - t_setup

    t_run = time.perf_counter()
    res = run_qmc(sys=system, params=params, ham_data=ham, trial_data=trial, meas_ops=meas_ops, trial_ops=trial_ops,
                  prop_ops=prop_ops, block_fn=blocks.block, state=state)
    run_s = time.perf_counter() - t_run

    # run_qmc's block_energies: its starting estimate, the N_EQL equilibration blocks, then the sampling blocks
    # it kept (outliers removed); the tau = 0 point of the plot is e0, the measured energy, instead of the first
    energies = np.asarray(res.block_energies, float)
    weights = np.asarray(res.block_weights, float)
    kept = len(energies) - 1 - N_EQL
    timing = dict(device=DEVICE, jax=jax.__version__, startup_s=t_setup - _T0, setup_s=setup_s, run_s=run_s,
                  per_block_s=run_s / N_TOTAL, walker_steps_per_s=N_WALKERS * N_PROP * N_TOTAL / run_s,
                  wall_s=time.perf_counter() - _T0)
    print(f"timing on {DEVICE}: startup {t_setup - _T0:.0f} s + setup {setup_s:.0f} s + run_qmc {run_s:.0f} s"
          f" = {run_s / N_TOTAL:.3f} s/block with the compile ({timing['walker_steps_per_s']:.0f} walker-steps/s)")
    if args.project_blocks:
        total = t_setup - _T0 + setup_s + args.project_blocks * run_s / N_TOTAL
        print(f"projected for a run of {args.project_blocks} blocks: {total / 60:.1f} min"
              f" (plus the job's module load and start-up checks); suggested --time {hms(2 * total)}")
    return dict(energies=energies[1:], weights=weights[1:], e_estimate_start=energies[0], kept=kept, e0=e0,
                e_trial=e_trial, e_uhf=e_uhf, mean=float(res.mean_energy), stderr=float(res.stderr_energy),
                stderr_gamma=finite(res.stderr_gamma), stderr_blocking=finite(res.stderr_blocking),
                error_method=res.error_method, error_reliable=bool(res.error_reliable),
                tau_int=finite(res.tau_int), effective_sample_size=finite(res.effective_sample_size),
                gamma_warnings=list(res.gamma_warnings), blocking_B_star=res.blocking_B_star,
                blocking_plateau_found=bool(res.blocking_plateau_found), timing=timing)


def save(out):
    """Every block to RUN_FILE and the result to RESULT_FILE, each atomically."""
    arrays = {k: out[k] for k in ("energies", "weights")}
    meta = {k: v for k, v in out.items() if k not in arrays}
    tmp = RUN_FILE.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays, meta=json.dumps(meta), config=json.dumps(CONFIG))
    os.replace(tmp, RUN_FILE)
    result = dict(tag=TAG, config=CONFIG, e_ref=E_REF, ref_label=REF_LABEL, run_file=str(RUN_FILE), **meta)
    tmp = RESULT_FILE.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(tmp, RESULT_FILE)
    print(f"saved {RUN_FILE} and {RESULT_FILE}")


def load():
    z = np.load(RUN_FILE)
    config = json.loads(str(z["config"]))
    if config != CONFIG:
        raise SystemExit(f"{RUN_FILE} was run with {config}, not {CONFIG}")
    return dict(energies=z["energies"], weights=z["weights"], **json.loads(str(z["meta"])))


def report(out):
    name = TRIAL.upper()
    n_rej = N_BLOCKS - out["kept"]
    if TRIAL != "uhf":
        print(f"UHF                {out['e_uhf']:12.6f}")
    print(f"{name} trial          {out['e_trial']:12.6f}")
    print(f"CPMC, {name} trial    {out['mean']:12.6f} ± {out['stderr']:.6f}   ({out['error_method']} error"
          f"{'' if out['error_reliable'] else ', flagged unreliable'}; blocking {out['stderr_blocking']};"
          f" {out['kept']} sampling blocks kept, {n_rej} rejected as outliers)")
    if E_REF is not None:
        print(f"reference{' (' + REF_LABEL + ')' if REF_LABEL else ''}   {E_REF:12.6f}")
        print(f"CPMC - reference   {out['mean'] - E_REF:+12.6f}   ({(out['mean'] - E_REF) / out['stderr']:+.1f} error bars)")
        print(f"per site: CPMC {out['mean'] / L:.6f} ± {out['stderr'] / L:.6f}, reference {E_REF / L:.6f}")


def plot(out):
    # evolution of the block energy (tau of the sampling blocks shifts if run_qmc rejected outliers)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.linewidth": 1.0, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 14, "axes.labelsize": 10, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round", "legend.frameon": False, "font.size": 10,
    })
    energies, e0, e_trial = out["energies"], out["e0"], out["e_trial"]
    E_MEAN, E_ERR = out["mean"], out["stderr"]
    tau = (np.arange(len(energies)) + 1) * N_PROP * DT            # block b ends at tau = (b + 1) * N_PROP * DT
    tau_eql = N_EQL * N_PROP * DT
    name, start = TRIAL.upper(), (TRIAL.upper() if WALKER_START == "trial" else "UHF")
    # the blocks (and the reference) set the vertical range; an energy far outside it, like E_RHF = +73 at L = 100,
    # U = 8, and the tau = 0 point when the walkers start as that trial, is marked at the edge of the axes instead
    shown = np.r_[energies, [E_REF] if E_REF is not None else []]
    lo, hi = float(np.min(shown)), float(np.max(shown))
    span = hi - lo
    off_scale = lambda e: e > hi + span or e < lo - span
    ylim = (lo - 0.08 * span, hi + 0.08 * span) if off_scale(e0) or off_scale(e_trial) else None

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.plot(np.r_[0.0, tau], np.r_[e0, energies], color=PALETTE[0], lw=1.4, label="block energy")
    if off_scale(e0):
        edge, marker = (ylim[1], "^") if e0 > hi else (ylim[0], "v")
        ax.plot(0.0, edge, marker, color=PALETTE[0], ms=6, zorder=3, clip_on=False,
                label=f"τ = 0 (walkers = {start}): {e0:.4f}, off scale")
    else:
        ax.plot(0.0, e0, "o", color=PALETTE[0], ms=5, zorder=3, label=f"τ = 0 (walkers = {start}): {e0:.4f}")
    ax.hlines(E_MEAN, tau_eql, tau[-1], color=PALETTE[1], ls="--", lw=1.5,
              label=f"CPMC, sampling mean: {E_MEAN:.4f} ± {E_ERR:.4f}")
    if np.isfinite(E_ERR):
        ax.fill_between([tau_eql, tau[-1]], E_MEAN - E_ERR, E_MEAN + E_ERR, color=PALETTE[1], alpha=0.15, lw=0)
    if E_REF is not None:
        ax.axhline(E_REF, color=PALETTE[2], ls="--", lw=1.5,
                   label=f"reference{', ' + REF_LABEL if REF_LABEL else ''}: {E_REF:.6f}")
    if off_scale(e_trial):  # legend entry only
        ax.plot([], [], color=INK2, ls=":", lw=1.5, label=f"{name} trial: {e_trial:.4f}, off scale")
    else:
        ax.axhline(e_trial, color=INK2, ls=":", lw=1.5, label=f"{name} trial: {e_trial:.4f}")
    ax.axvline(tau_eql, color=GRID, lw=3, zorder=0, label="end of equilibration")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("energy")
    ax.set_title(f"{name}-trial CPMC (trot): L = {L} chain, U = {U:g}, {N_WALKERS} walkers, Δτ = {DT:g}"
                 + ("" if WALKER_START == "trial" else ", UHF walker start"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    path = DATA_DIR / f"{TAG}_evolution.png"
    try:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    except RuntimeError as error:
        # the python/3.12.13 module's matplotlib fails rasterizing glyphs ("FT_Render_Glyph ... raster overflow",
        # jobs 7136717/9, as in plot_cpmc_runs.py); a PDF embeds the glyph outlines and never rasterizes them
        print(f"could not draw {path} ({error}); saving a PDF instead")
        path = path.with_suffix(".pdf")
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


print(f"jax {jax.__version__}, device {DEVICE}")
print(f"run {TAG} ({TRIAL.upper()} trial, walkers start as the {'trial' if WALKER_START == 'trial' else 'UHF'}"
      f" determinant, {'fast updates' if PROP == 'fast' else 'recomputed overlaps'}, trot.driver.run_qmc):"
      f" {N_EQL} + {N_BLOCKS} blocks, tau = {N_PROP * DT:g} per block, saved in {DATA_DIR}/", flush=True)
if args.plot_only:
    if not RUN_FILE.exists():
        raise SystemExit(f"--plot-only: no saved run at {RUN_FILE}")
    out = load()
else:
    if RESULT_FILE.exists():
        print(f"note: {RESULT_FILE.name} exists and will be replaced at the end")
    if args.compile_cache:
        jax.config.update("jax_compilation_cache_dir", args.compile_cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    out = run()
    save(out)                        # on disk before the plot is tried
report(out)
try:
    plot(out)
except Exception:
    traceback.print_exc()
    raise SystemExit(f"the plot failed; the result is saved in {RESULT_FILE} and the blocks in {RUN_FILE}:"
                     f" redo the plot with --plot-only (same arguments)")
