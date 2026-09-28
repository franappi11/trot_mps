#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH -J mps_gpu
#SBATCH -o slurm/slurm-%j.out
# Any A100, which on rusty may be a MIG slice (1g.20gb / 2g.20gb: 1/7 or 2/7 of
# the card's compute, 20 GB); the log says which. For serious runs and timings
# ask for one whole 80 GB SXM4 card (the options override the lines above):
#   sbatch --gpus-per-task=a100-sxm4-80gb:1 --constraint="a100-80gb&sxm4" run_mps_gpu.sh ...

# GPU SLURM wrapper, same environment as run_mps_sweep.sh but on the CUDA backend.
# TARGET picks what runs (default run_mps_sweep_gpu.py); all arguments are forwarded.
#
#   mkdir -p slurm
#   # GPU baseline: the original script, unchanged, on the GPU
#   sbatch run_mps_gpu.sh --module mps_cpmc_new --L 32 --trial-chi 8 --chi-w 4 --walkers 200 --out sweep_gpu_ref
#   # the GPU implementation
#   sbatch run_mps_gpu.sh --L 32 --trial-chi 8 --chi-w 4 --walkers 2048 --out sweep_gpu
#   # linear-algebra microbenchmark, end-to-end benchmark, tests
#   sbatch --export=ALL,TARGET=gpu_linalg_bench.py run_mps_gpu.sh
#   sbatch --export=ALL,TARGET=bench_mps_cpmc_gpu.py run_mps_gpu.sh --L 32 --baseline
#   sbatch --export=ALL,TARGET=pytest run_mps_gpu.sh -x -q /mnt/home/fnappi/trot_mps/tests/test_gmps_mps_cpmc_gpu.py

module --force purge
module load modules/2.5-beta2 python/3.12.13
# The module's jax ships the CUDA 13 plugin; the toolkit supplies the CUDA libraries
# (cuPTI among them) and ptxas. The A100 nodes' R580 driver runs any CUDA 13.x.
module load cuda/13.3.0 || { echo "module load cuda/13.3.0 failed"; exit 1; }
module list 2>&1
# cuPTI lives in extras/CUPTI/lib64 in the toolkit layout, which modules often
# leave off the library path.
for root in "$CUDA_HOME" "$CUDA_PATH" "$CUDA_ROOT"; do
    if [[ -n "$root" && -d "$root/extras/CUPTI/lib64" ]]; then
        export LD_LIBRARY_PATH="$root/extras/CUPTI/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        break
    fi
done

PYTHON="${PYTHON:-$HOME/.trot312/bin/python}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
TARGET="${TARGET:-run_mps_sweep_gpu.py}"

export PYTHONPATH="/mnt/home/fnappi/trot_mps${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

NCPU="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$NCPU"
export OPENBLAS_NUM_THREADS="$NCPU"
export MKL_NUM_THREADS="$NCPU"
# GPU first (the default device; fails loudly if CUDA is missing), and the CPU
# too: io_callback (the per-block JSONL logger) needs a CPU device (job 7107162).
export JAX_PLATFORMS=cuda,cpu
# Without cuPTI the plugin's start-up version check refuses to register the GPU
# backend (job 7107132, before cuda/13.3.0 was loaded). cuPTI is only used by
# jax.profiler, so if it is still not loadable, skip that check instead; the fp64
# matmul check below still proves the backend works.
if ! "$PYTHON" -c "import ctypes; ctypes.CDLL('libcupti.so.13')" 2>/dev/null; then
    echo "warning: libcupti.so.13 not loadable; skipping JAX's CUDA version check (no jax.profiler)"
    export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1
fi
# Let JAX take most of the card; the chunker in mps_cpmc_gpu budgets inside this.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
# float32 matmuls are never used, but keep TF32 off like trot's conftest does.
export NVIDIA_TF32_OVERRIDE=0

echo "job:    ${SLURM_JOB_ID:-local} on $(hostname)"
echo "python: $PYTHON $TARGET"
echo "args:   $*"
echo "start:  $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "nvidia-smi failed"

# pyblock3/MKL preload, as in run_mps_sweep.sh (DMRG runs on the host CPU).
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

# Fail fast: the CUDA backend must come up with float64, and pyblock3 must work.
"$PYTHON" -c "$MKL_FIND
import ctypes
sys.path.insert(0, '$SCRIPT_DIR')
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
dev = jax.devices()[0]
assert jax.default_backend() == 'gpu', f'backend is {jax.default_backend()}, not gpu'
if 'MIG' in dev.device_kind:
    print(f'note: {dev.device_kind} is a MIG slice, not a whole GPU; timings scale with its share')
x = jnp.ones((2048, 2048), jnp.float64)
y = (x @ x).block_until_ready()
assert y.dtype == jnp.float64 and float(y[0, 0]) == 2048.0
mkl_def = first(d + '/libmkl_def.so.1', lib + '/libmkl_def.so.2')
assert mkl_def, 'libmkl_def not found in pyblock3.libs or the venv lib'
ctypes.CDLL(mkl_def)
import mps_cpmc_new as m
cfg = m.Config(L=4, n_up=2, n_down=2, interaction=4.0, trial_chi=16, dmrg_sweeps=8)
mps, e = m.run_dmrg(m.build_dmrg_hamiltonian(cfg), cfg)
print('env:', sys.version.split()[0], 'jax', jax.__version__, 'device', dev.device_kind,
      'memory', dev.memory_stats().get('bytes_limit') if dev.memory_stats() else None, f'L=4 DMRG {e:.10f}')
" || exit 1

# Launch through srun, as the cluster documentation does for GPU jobs, so the
# task is bound to its GPU and CPUs.
if [[ "$TARGET" == "pytest" ]]; then
    srun "$PYTHON" -m pytest "$@"
else
    srun "$PYTHON" "$SCRIPT_DIR/$TARGET" "$@"
fi
rc=$?

echo "finish: $(date)  rc=$rc"
exit "$rc"
