#!/bin/bash -l
#SBATCH --partition=genx
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH -J mps_sweep
#SBATCH -o slurm/slurm-%j.out

# Serial SLURM wrapper for run_mps_sweep.py. Every option is optional and
# forwarded as-is; unset ones fall back to the defaults in run_mps_sweep.py.
#
#   mkdir -p slurm     # sbatch will not create the -o directory
#   sbatch run_mps_sweep.sh --L 32 48 --U 4 8 --trial-chi 8 --chi-w 2 4 6 \
#       --walkers 200 --eql 60 --blocks 200 --out sweep_T8
#
# That runs every combination one after another in a single job. To run them
# in parallel, one job each, use submit_mps_sweep.sh with the same options.
#
# SBATCH resources can be overridden on the command line, e.g.
#   sbatch --time=2-00:00:00 --cpus-per-task=8 run_mps_sweep.sh --L 48 ...
#
# Options (see run_mps_sweep.py):
#   --L N [N ...]  --U X [X ...]  --trial-chi N  --chi-w N [N ...]
#   --walkers N  --eql N  --blocks N  --seed N
#   --plan-reference {natural,rhf}  --walker-start {natural,rhf}  --out DIR
#
# Set PYTHON to pick the interpreter:  PYTHON=/path/to/python sbatch run_mps_sweep.sh ...
# Set TARGET to run another script in this folder with the same environment,
# e.g. TARGET=dmrg_reference.py (submit_mps_sweep.sh does this).

# ~/.trot312 is a venv with --system-site-packages on this module, which provides
# jax 0.11.0, numpy and scipy (the code is developed on jax 0.11.1; jax 0.4.28
# compiled the block scan 10-20x slower). The venv adds pyblock3, pyscf and mkl.
module --force purge
module load modules/2.5-beta2 python/3.12.13

PYTHON="${PYTHON:-$HOME/.trot312/bin/python}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
TARGET="${TARGET:-run_mps_sweep.py}"

# Import trot from the git clone, whether or not it is pip-installed.
export PYTHONPATH="/mnt/home/fnappi/trot_mps${PYTHONPATH:+:$PYTHONPATH}"

# Write the run logs as they happen, so progress is visible while the job runs.
export PYTHONUNBUFFERED=1

# Keep JAX/BLAS inside the allocated cores.
NCPU="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$NCPU"
export OPENBLAS_NUM_THREADS="$NCPU"
export MKL_NUM_THREADS="$NCPU"
# The module's jax ships a CUDA plugin that fails to initialise on CPU nodes
# (a harmless cuPTI traceback in every log); use the CPU backend only.
export JAX_PLATFORMS=cpu

echo "job:    ${SLURM_JOB_ID:-local} on $(hostname)"
echo "python: $PYTHON $TARGET"
echo "args:   $*"
echo "start:  $(date)"

# pyblock3's extension loads MKL privately (RTLD_LOCAL), so when MKL picks its
# generic kernel libmkl_def on some nodes, that library cannot resolve its
# symbols in MKL core and crashes ("Intel MKL FATAL ERROR"). Preloading the
# OpenMP, core and thread layers makes them global; none exports the standard
# BLAS names, so numpy/scipy/jax keep their own BLAS. Works for both pyblock3
# layouts: MKL bundled in pyblock3.libs (old wheel) or pip mkl in the venv's lib.
MKL_FIND="
import glob, os, sys, pyblock3
d = os.path.join(os.path.dirname(os.path.dirname(pyblock3.__file__)), 'pyblock3.libs')
lib = os.path.join(sys.prefix, 'lib')
def first(*patterns):
    for p in patterns:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
"
MKL_LIBS=$("$PYTHON" -c "$MKL_FIND
libs = [first(d + '/libgomp-*.so*'), first(d + '/libmkl_core-*.so.1', lib + '/libmkl_core.so.2'),
        first(d + '/libmkl_gnu_thread-*.so.1', lib + '/libmkl_gnu_thread.so.2')]
print(':'.join(x for x in libs if x))")
export LD_PRELOAD="$MKL_LIBS${LD_PRELOAD:+:$LD_PRELOAD}"

# Fail fast if the environment is incomplete, and record the versions used:
# load MKL's generic kernel, then run a tiny DMRG through the same pyblock3/MKL
# path as the real runs (MKL picks its kernel on the first call), and check
# it against the MPO-independent dense energy.
"$PYTHON" -c "$MKL_FIND
import ctypes
sys.path.insert(0, '$SCRIPT_DIR')
import jax, jax.numpy as jnp, trot, mps_cpmc_new as m
mkl_def = first(d + '/libmkl_def.so.1', lib + '/libmkl_def.so.2')
assert mkl_def, 'libmkl_def not found in pyblock3.libs or the venv lib'
ctypes.CDLL(mkl_def)
cfg = m.Config(L=4, n_up=2, n_down=2, interaction=4.0, trial_chi=16, dmrg_sweeps=8)
mps, e = m.run_dmrg(m.build_dmrg_hamiltonian(cfg), cfg)
t, _ = m.densify_with_charges(mps, cfg.L)
Ht = m.compress_mps(m.apply_mpo(m.hubbard_mpo(cfg.L, cfg.hopping, cfg.interaction), t))
t, Ht = tuple(map(jnp.asarray, t)), tuple(map(jnp.asarray, Ht))
e_dense = float(m.contract_real(Ht, t) / m.contract_real(t, t))
assert abs(e - e_dense) < 1e-8, f'DMRG MPO energy {e} != dense {e_dense}'
print('env:', sys.version.split()[0], 'jax', jax.__version__, 'trot', trot.__file__, f'mkl ok, L=4 DMRG {e:.10f}')
" || exit 1

"$PYTHON" "$SCRIPT_DIR/$TARGET" "$@"
rc=$?

echo "finish: $(date)  rc=$rc"
exit "$rc"
