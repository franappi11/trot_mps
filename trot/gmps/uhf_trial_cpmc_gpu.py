"""UHF-trial CPMC for the open Hubbard chain on the GPU: uhf_trial_cpmc.ipynb as a script.

By default pure UHF: the trial is the self-consistent UHF determinant (Neel start, so the
antiferromagnetic solution at half filling) and the walkers are unrestricted determinants, with the
notebook's objects, parameters and defaults. --trial rhf takes the RHF determinant as the trial
instead (trot's UHF trial with the same orbitals for both spins); the walkers stay unrestricted and
start as the trial, or as the UHF determinant with --walker-start uhf. They are propagated
with fast updates (--prop fast, the default:
trot.prop.cpmc with the UHF Green's-function ops of uhf_cpmc.py) or, as in the notebook, by
recomputing both proposal overlaps at every site (--prop slow: trot.prop.cpmc_slow). Both use the
same random numbers the same way, so they follow the same trajectory up to rounding
(tests/test_gmps_uhf_fast_update.py checks it block by block); the one difference in the rules is
that the fast step zeroes a site's field ratio at or below --weight-floor where the slow one
zeroes half the ratio below it.

Blocks run --chunk at a time. After each chunk the walker state goes to
<out>/<tag>_state_<blocks done>.npz and the block energies and weights to <out>/<tag>_blocks.npz:
the notebook's files, so its Result and plot cells read a run of this script (set its DATA_DIR to
--out, and add _fast to its TAG for a fast-update run), and a job that ends early (time limit, node
failure) continues where it stopped when it is submitted again with the same arguments. A resumed
run gives the same blocks as an uninterrupted one on the same kind of device: the random key is
part of the saved state.

At the end the result goes to <out>/<tag>_result.json, and only then is the plot
<out>/<tag>_evolution.png drawn (a .pdf where matplotlib cannot rasterize text, as with the
python/3.12.13 module), so a failing plot loses nothing; --plot-only redraws it from the saved
files, without running anything.

The first chunk of a process includes the compile; the later ones give the time per block. With
--project-blocks N the log gives the wall time of a fresh run of N blocks and an --time of about
twice that. A MIG slice runs at its share of the card (1g.20gb 1/7, 2g.20gb 2/7 of an A100).

    # from trot/gmps, in the environment of run_mps_gpu.sh (any A100: MIG slice or whole card);
    # the host only runs the UHF SCF and the saves: job 7141345 (L=100, 400 walkers) kept ~1 core
    # busy and peaked at 3.1 GB, so 2 CPUs (one feeds the GPU) and 6 GB are enough
    sbatch -J uhf_time --time=00:15:00 --cpus-per-task=2 --mem=6G --export=ALL,TARGET=uhf_trial_cpmc_gpu.py \
        run_mps_gpu.sh --eql 10 --blocks 20 --out uhf_cpmc_test --project-blocks 1200   # timing test
    sbatch -J uhf_cpmc --time=<from the timing test> --cpus-per-task=2 --mem=6G \
        --export=ALL,TARGET=uhf_trial_cpmc_gpu.py run_mps_gpu.sh                      # the notebook's run
    sbatch -p genx -J uhf_plot --time=00:10:00 --cpus-per-task=2 --mem=4G \
        --export=ALL,TARGET=uhf_trial_cpmc_gpu.py run_mps_sweep.sh --plot-only       # redo the plot (never on the login node)
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
    # nothing to compute: stay on the CPU with a small XLA thread pool. By default XLA starts about one thread per
    # core in each of its pools, which on a login node (256 tasks per user, shared with VS Code and the shells)
    # fails with "Thread tf_XLAEigen creation via pthread_create() failed"; jaxlib reads PJRT_NPROC for the pool size
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("PJRT_NPROC", "2")

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from trot.core.system import System
from trot.driver import make_run_blocks
from trot.gmps.uhf_cpmc import make_uhf_cpmc_trial_ops, rhf_scf, uhf_scf
from trot.ham.chol import HamChol
from trot.ham.hubbard import HamHubbard
from trot.meas.uhf import make_uhf_meas_ops
from trot.prop import blocks, cpmc, cpmc_slow
from trot.prop.types import QmcParams
from trot.stat_utils import blocking_analysis_ratio
from trot.trial.uhf import UhfTrial, make_uhf_trial_ops

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
# ---- CPMC (trot defaults otherwise: pop-control damping, weight cap, SR after every block)
parser.add_argument("--walkers", type=int, default=400)
parser.add_argument("--dt", type=float, default=0.005)
parser.add_argument("--steps", type=int, default=50, help="propagation steps per block (tau per block = steps * dt)")
parser.add_argument("--eql", type=int, default=200, help="equilibration blocks")
parser.add_argument("--blocks", type=int, default=1000, help="sampling blocks")
parser.add_argument("--weight-floor", type=float, default=1.0e-8,
                    help="overlap ratios at or below it are zeroed; as the cpmc_L100 runs (trot's default is 1e-3)")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--prop", default="fast", choices=["fast", "slow"],
                    help="fast updates (trot.prop.cpmc) or the notebook's recomputed overlaps (trot.prop.cpmc_slow)")
parser.add_argument("--trial", default="uhf", choices=["uhf", "rhf"], help="trial determinant (walkers are unrestricted)")
parser.add_argument("--walker-start", default="trial", choices=["trial", "uhf"],
                    help="initial walkers: the trial determinant (trot's default) or the UHF one")
# ---- run control
parser.add_argument("--chunk", type=int, default=10, help="blocks per jitted call; progress is printed and saved after each")
parser.add_argument("--rerun", action="store_true", help="delete this run's saved files and start again from tau = 0")
parser.add_argument("--e-ref", type=float, default=None,
                    help="reference energy for the result and the plot (default: REFERENCES, if it has this chain)")
parser.add_argument("--out", default=str(HERE / "uhf_cpmc_data"), help="folder for the state, blocks, result and plot")
parser.add_argument("--plot-only", action="store_true", help="no run: the result and the plot from the saved blocks")
parser.add_argument("--project-blocks", type=int, default=0,
                    help="project this process's wall time to a fresh run of this many blocks (eql + sampling)")
parser.add_argument("--compile-cache", default=os.path.expanduser("~/.cache/trot_jax_compile"),
                    help="JAX persistent compilation cache ('' = off)")
args = parser.parse_args()

L, T, U = args.L, args.t, args.U
N_UP = L // 2 if args.n_up is None else args.n_up
N_DN = L // 2 if args.n_dn is None else args.n_dn
N_WALKERS, DT, N_PROP, N_EQL, N_BLOCKS = args.walkers, args.dt, args.steps, args.eql, args.blocks
WEIGHT_FLOOR, SEED, CHUNK = args.weight_floor, args.seed, args.chunk
E_REF, REF_LABEL = REFERENCES.get((L, N_UP, N_DN, T, U), (None, None))
if args.e_ref is not None:
    E_REF, REF_LABEL = args.e_ref, None

DATA_DIR = Path(args.out).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
PROP, TRIAL = args.prop, args.trial
WALKER_START = "trial" if TRIAL == "uhf" else args.walker_start       # for a UHF trial the two starts are the same
if TRIAL == "rhf" and N_UP != N_DN:
    raise SystemExit(f"--trial rhf needs n_up = n_dn, not {N_UP}, {N_DN}")
# the notebook's tag for its UHF run; other trials, walker starts and the fast updates get their own files, so no two
# of them ever resume each other
TAG = (f"L{L}_N{N_UP}-{N_DN}_U{U:g}_nw{N_WALKERS}_dt{DT:g}_n{N_PROP}_s{SEED}"
       + ("" if TRIAL == "uhf" else f"_{TRIAL}trial") + ("_wuhf" if WALKER_START == "uhf" else "")
       + ("_fast" if PROP == "fast" else ""))
BLOCKS_FILE = DATA_DIR / f"{TAG}_blocks.npz"
RESULT_FILE = DATA_DIR / f"{TAG}_result.json"
CONFIG = dict(L=L, N_UP=N_UP, N_DN=N_DN, T=T, U=U, N_WALKERS=N_WALKERS, DT=DT, N_PROP=N_PROP, N_EQL=N_EQL,
              N_BLOCKS=N_BLOCKS, WEIGHT_FLOOR=WEIGHT_FLOOR, SEED=SEED)
if TRIAL != "uhf":  # only when not the default, so the notebook's (and the UHF runs') saved configs still match
    CONFIG.update(TRIAL=TRIAL, WALKER_START=WALKER_START)
N_TOTAL = N_EQL + N_BLOCKS
DEVICE = f"{jax.default_backend()}: {jax.devices()[0].device_kind}"


def state_file(n_done):
    return DATA_DIR / f"{TAG}_state_{n_done:05d}.npz"


def save_progress(state, energies, weights, e_uhf, e_trial, e0):
    # the state is written first under a new name, then the block file (atomic rename) that points to it, then
    # older states are removed: a crash at any point leaves a block file whose state exists
    n_done = len(energies)
    blocks.dump_prop_state_npz(state, state_file(n_done))
    tmp = BLOCKS_FILE.with_suffix(".tmp.npz")
    np.savez(tmp, energy=np.asarray(energies), weight=np.asarray(weights), nodes=int(state.node_encounters),
             e_uhf=e_uhf, e_trial=e_trial, e0=e0, config=json.dumps(CONFIG), prop=PROP, trial=TRIAL,
             walker_start=WALKER_START, device=DEVICE)
    os.replace(tmp, BLOCKS_FILE)
    for old in DATA_DIR.glob(f"{TAG}_state_*.npz"):
        if old != state_file(n_done):
            old.unlink()


def load_blocks():
    # block energies and weights saved so far, the node count at the last save, and the run's UHF energy
    saved = np.load(BLOCKS_FILE)
    config = json.loads(str(saved["config"]))
    if config != CONFIG:
        raise ValueError(f"{BLOCKS_FILE} was run with {config}, not {CONFIG}: pass --rerun or change --out")
    return saved["energy"], saved["weight"], int(saved["nodes"]), float(saved["e_uhf"])


def sampling_mean(energies, weights):
    # weighted mean over the sampling blocks with trot's blocking-analysis error (nan until a plateau exists)
    stats = blocking_analysis_ratio(energies[N_EQL:], weights[N_EQL:], print_q=False)
    return float(stats["mu"]), float("nan") if stats["se_star"] is None else float(stats["se_star"])


def hms(seconds):
    minutes = max(10, 10 * math.ceil(seconds / 600))  # whole 10 minutes, at least 10
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def run():
    """Run (or continue) the blocks up to N_TOTAL, saving after every chunk; returns this process's timing."""
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

    # the propagator is built from the HamHubbard; the ham_data passed through the blocks is the HamChol and
    # reaches only the energy kernel
    ham = HamHubbard(h1=jnp.asarray(h1), u=U)                     # propagation
    onsite = np.zeros((L, L, L))
    onsite[np.arange(L), np.arange(L), np.arange(L)] = np.sqrt(U)
    ham_meas = HamChol(h0=jnp.zeros(()), h1=jnp.asarray(h1), chol=jnp.asarray(onsite))   # U n_up n_dn as L Cholesky vectors

    system = System(norb=L, nelec=(N_UP, N_DN), walker_kind="unrestricted")
    # an RHF trial is trot's UHF trial with the same orbitals for both spins: the same determinant, measured and
    # propagated by the same (tested) UHF code
    trial = UhfTrial(mo_coeff_a=jnp.asarray(Ta), mo_coeff_b=jnp.asarray(Tb))
    meas_ops = make_uhf_meas_ops(system)
    if PROP == "fast":
        trial_ops = make_uhf_cpmc_trial_ops(system)          # trot's UHF ops + calc_green, calc_overlap_ratio, update_green
        prop_ops = cpmc.make_prop_ops(ham, system.walker_kind, trial_ops)
    else:
        trial_ops = make_uhf_trial_ops(system)
        prop_ops = cpmc_slow.make_prop_ops(ham, system.walker_kind)
    params = QmcParams(dt=DT, n_walkers=N_WALKERS, n_prop_steps=N_PROP, n_blocks=N_BLOCKS, n_eql_blocks=N_EQL,
                       weight_floor=WEIGHT_FLOOR, seed=SEED)

    run_blocks = make_run_blocks(block_fn=blocks.block, sys=system, params=params,
                                 trial_ops=trial_ops, meas_ops=meas_ops, prop_ops=prop_ops)
    ctx = dict(ham_data=ham_meas, trial_data=trial, meas_ctx=meas_ops.build_meas_ctx(ham_meas, trial),
               prop_ctx=prop_ops.build_prop_ctx(ham, trial_ops.get_rdm1(trial), params))
    initial_walkers = None                  # trot's default: every walker the trial determinant
    if WALKER_START == "uhf":
        initial_walkers = (jnp.broadcast_to(jnp.asarray(Ca), (N_WALKERS, L, N_UP)),
                           jnp.broadcast_to(jnp.asarray(Cb), (N_WALKERS, L, N_DN)))
    state = prop_ops.init_prop_state(sys=system, ham_data=ham_meas, trial_ops=trial_ops, trial_data=trial,
                                     meas_ops=meas_ops, params=params, initial_walkers=initial_walkers)

    e0 = float(state.e_estimate)
    if WALKER_START == "trial":
        # at tau = 0 every walker is the trial: checks the trial, the walker start and the HamChol energy
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

    if args.rerun:
        # this run's files only: a glob on f"{TAG}_*" would also take the fast-update run of the slow run's tag
        for f in [BLOCKS_FILE, RESULT_FILE, DATA_DIR / f"{TAG}_evolution.png", DATA_DIR / f"{TAG}_evolution.pdf",
                  *DATA_DIR.glob(f"{TAG}_state_*.npz")]:
            f.unlink(missing_ok=True)
    if BLOCKS_FILE.exists():
        energies, weights, _, _ = load_blocks()
        energies, weights = energies.tolist(), weights.tolist()
        state = blocks.load_prop_state_npz(state_file(len(energies)))
        with np.load(BLOCKS_FILE) as saved:
            saved_on = str(saved["device"]) if "device" in saved.files else "an unrecorded device"
        print(f"resuming after block {len(energies)}/{N_TOTAL} (saved on {saved_on})")
    else:
        energies, weights = [], []
    setup_s = time.perf_counter() - t_setup

    start, n_start, chunks = time.perf_counter(), len(energies), []
    while len(energies) < N_TOTAL:
        n = min(CHUNK, N_TOTAL - len(energies))
        t_chunk = time.perf_counter()
        state, scalars, _ = run_blocks(state, **ctx, n_blocks=n)
        energies += np.asarray(scalars["energy"]).tolist()
        weights += np.asarray(scalars["weight"]).tolist()
        save_progress(state, energies, weights, e_uhf, e_trial, e0)
        chunks.append((n, time.perf_counter() - t_chunk))
        b, elapsed = len(energies), time.perf_counter() - start
        eta = elapsed / (b - n_start) * (N_TOTAL - b)
        print(f"[{'eql ' if b <= N_EQL else 'samp'} {b:5d}/{N_TOTAL}]  tau {b * N_PROP * DT:7.2f}"
              f"  E {np.average(energies[-n:], weights=weights[-n:]):12.6f}  W {np.mean(weights[-n:]):9.3f}"
              f"  nodes {int(state.node_encounters):4d}  t {elapsed / 60:7.1f} min  eta {eta / 3600:6.2f} h", flush=True)
    print(f"{len(energies)}/{N_TOTAL} blocks done")

    # the first chunk compiles, and so does a shorter last one: the time per block comes from the other chunks
    steady = [(n, s) for n, s in chunks[1:] if n == CHUNK]
    per_block = sum(s for _, s in steady) / sum(n for n, _ in steady) if steady else None
    compile_s = chunks[0][1] - chunks[0][0] * per_block if chunks and per_block else None
    timing = dict(device=DEVICE, jax=jax.__version__, startup_s=t_setup - _T0, setup_s=setup_s, compile_s=compile_s,
                  per_block_s=per_block, walker_steps_per_s=N_WALKERS * N_PROP / per_block if per_block else None,
                  blocks_run=len(energies) - n_start, wall_s=time.perf_counter() - _T0)
    if per_block:
        print(f"timing on {DEVICE}: startup {t_setup - _T0:.0f} s + setup {setup_s:.0f} s + compile {compile_s:.0f} s"
              f" + {per_block:.3f} s/block (incl. the saves; {timing['walker_steps_per_s']:.0f} walker-steps/s)")
        if args.project_blocks:
            total = t_setup - _T0 + setup_s + compile_s + args.project_blocks * per_block
            print(f"projected for a fresh run of {args.project_blocks} blocks: {total / 60:.1f} min"
                  f" (plus the job's module load and start-up checks); suggested --time {hms(2 * total)}")
    elif chunks:
        print("timing: fewer than two full chunks ran in this process, so no time per block")
    return timing


def finite(x):
    return None if x is None or not np.isfinite(x) else float(x)


def summarize(timing):
    """Print the result and write it to RESULT_FILE (atomically); returns what the plot needs."""
    energies, weights, nodes, e_uhf = load_blocks()
    done = len(energies)
    E_MEAN, E_ERR = sampling_mean(energies, weights) if done - N_EQL >= 20 else (np.nan, np.nan)
    with np.load(BLOCKS_FILE) as saved:
        saved_on = str(saved["device"]) if "device" in saved.files else None
        # trot's energy of the tau = 0 walkers; files saved before it was stored come from UHF-trial runs and fall
        # back to E_UHF, which run() asserts it equals to 1e-8 (the production run 7138041 logged 4.6e-14)
        e0 = float(saved["e0"]) if "e0" in saved.files else e_uhf
        e_trial = float(saved["e_trial"]) if "e_trial" in saved.files else e_uhf
    status = "" if done == N_TOTAL else f"   (run incomplete: {done}/{N_TOTAL} blocks)"
    name = TRIAL.upper()
    if TRIAL != "uhf":
        print(f"UHF                {e_uhf:12.6f}")
    print(f"{name} trial          {e_trial:12.6f}")
    print(f"CPMC, {name} trial    {E_MEAN:12.6f} ± {E_ERR:.6f}   ({max(done - N_EQL, 0)} sampling blocks,"
          f" tau = {max(done - N_EQL, 0) * N_PROP * DT:g}){status}")
    if E_REF is not None:
        print(f"reference{' (' + REF_LABEL + ')' if REF_LABEL else ''}   {E_REF:12.6f}")
        print(f"CPMC - reference   {E_MEAN - E_REF:+12.6f}   ({(E_MEAN - E_REF) / E_ERR:+.1f} error bars)")
        print(f"per site: CPMC {E_MEAN / L:.6f} ± {E_ERR / L:.6f}, reference {E_REF / L:.6f}")
    print(f"node encounters (overlap ratios at or below the floor): {nodes}")

    if not timing and RESULT_FILE.exists():  # --plot-only: keep the timing of the process that ran the blocks
        timing = json.loads(RESULT_FILE.read_text()).get("last_process", {})
    result = dict(tag=TAG, config=CONFIG, prop=PROP, trial=TRIAL, walker_start=WALKER_START, blocks_done=done,
                  complete=done == N_TOTAL, e_uhf=e_uhf, e_trial=e_trial, e_tau0=e0, e_cpmc=finite(E_MEAN),
                  e_cpmc_err=finite(E_ERR), e_ref=E_REF, ref_label=REF_LABEL,
                  node_encounters=nodes, saved_on=saved_on, blocks_file=str(BLOCKS_FILE), last_process=timing)
    tmp = RESULT_FILE.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(tmp, RESULT_FILE)
    print(f"saved {RESULT_FILE}")
    return energies, weights, nodes, e_trial, e0, E_MEAN, E_ERR


def plot(energies, weights, nodes, e_trial, e0, E_MEAN, E_ERR):
    # evolution of the block energy; works on a run that is still going too
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
    done = len(energies)
    tau = (np.arange(done) + 1) * N_PROP * DT            # block b ends at tau = (b + 1) * N_PROP * DT
    tau_eql = N_EQL * N_PROP * DT

    name, start = TRIAL.upper(), (TRIAL.upper() if WALKER_START == "trial" else "UHF")
    # the blocks (and the reference) set the vertical range; an energy far outside it, like E_RHF = +73 at L = 100,
    # U = 8, and the tau = 0 point when the walkers start as that trial, would squash them into a line, so it is
    # marked at the edge of the axes instead
    shown = np.r_[energies, [E_REF] if E_REF is not None else []]
    lo, hi = float(np.min(shown)), float(np.max(shown))
    span = hi - lo
    off_scale = lambda e: e > hi + span or e < lo - span
    ylim = (lo - 0.08 * span, hi + 0.08 * span) if off_scale(e0) or off_scale(e_trial) else None

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    # the curve starts at tau = 0, where every walker is the start determinant
    ax.plot(np.r_[0.0, tau], np.r_[e0, energies], color=PALETTE[0], lw=1.4, label="block energy")
    if off_scale(e0):
        edge, marker = (ylim[1], "^") if e0 > hi else (ylim[0], "v")
        ax.plot(0.0, edge, marker, color=PALETTE[0], ms=6, zorder=3, clip_on=False,
                label=f"τ = 0 (walkers = {start}): {e0:.4f}, off scale")
    else:
        ax.plot(0.0, e0, "o", color=PALETTE[0], ms=5, zorder=3, label=f"τ = 0 (walkers = {start}): {e0:.4f}")
    if np.isfinite(E_MEAN):
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
    ax.set_title(f"{name}-trial CPMC: L = {L} chain, U = {U:g}, {N_WALKERS} walkers, Δτ = {DT:g}"
                 + ("" if WALKER_START == "trial" else ", UHF walker start")
                 + ("" if done == N_TOTAL else f"  ({done}/{N_TOTAL} blocks)"))
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
    print(f"saved {path}; node encounters so far: {nodes}")


print(f"jax {jax.__version__}, device {DEVICE}")
print(f"run {TAG} ({TRIAL.upper()} trial, walkers start as the {'trial' if WALKER_START == 'trial' else 'UHF'}"
      f" determinant, {'fast updates' if PROP == 'fast' else 'recomputed overlaps'}): {N_EQL} + {N_BLOCKS} blocks,"
      f" tau = {N_PROP * DT:g} per block, saved in {DATA_DIR}/", flush=True)
if args.plot_only:
    if not BLOCKS_FILE.exists():
        raise SystemExit(f"--plot-only: no saved blocks at {BLOCKS_FILE}")
    timing = {}
else:
    if args.compile_cache:
        jax.config.update("jax_compilation_cache_dir", args.compile_cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    timing = run()
summary = summarize(timing)          # the result is on disk before the plot is tried
try:
    plot(*summary)
except Exception:
    traceback.print_exc()
    raise SystemExit(f"the plot failed; the result is saved in {RESULT_FILE} and the blocks in {BLOCKS_FILE}:"
                     f" redo the plot with --plot-only (same arguments)")
