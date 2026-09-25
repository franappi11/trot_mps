
#!/bin/bash -l
#SBATCH --partition=genx
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH -J mps_sweep
#SBATCH -o slurm/slurm-%j.out

# Serial SLURM wrapper for run_mps_sweep.py. Every option is optional and
# forwarded as-is; unset ones fall back to the defaults in run_mps_sweep.py.
#
#   mkdir -p slurm     # sbatch will not create the -o directory
#   sbatch run_mps_sweep.sh --L 32 --trial-chi 8 --chi-w 2 4 6 \
#       --walkers 200 --eql 60 --blocks 200 --out sweep_L32_T8
#
# One job per walker chi (runs in parallel instead of one after another):
#   for w in 2 4 6; do
#       sbatch -J "L32_w$w" run_mps_sweep.sh --L 32 --chi-w "$w" --out sweep_L32_T8
#   done
#
# SBATCH resources can be overridden on the command line, e.g.
#   sbatch --time=2-00:00:00 --cpus-per-task=8 run_mps_sweep.sh --L 48 ...
#
# Options (see run_mps_sweep.py):
#   --L N  --trial-chi N  --chi-w N [N ...]  --walkers N  --eql N  --blocks N
#   --seed N  --U X  --bond-reference {rhf,natural}  --walker-start {natural,rhf}
#   --out DIR
#
# Set PYTHON to pick the interpreter:  PYTHON=/path/to/python sbatch run_mps_sweep.sh ...

module --force purge
module load modules/2.4-20250724
module load python/3.12.9
module spider jax/0.4.28
PYTHON="${PYTHON:-$HOME/.trot/bin/python}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

# Import trot from the git clone, whether or not it is pip-installed.
export PYTHONPATH="/mnt/home/fnappi/trot_mps${PYTHONPATH:+:$PYTHONPATH}"

# Keep JAX/BLAS inside the allocated cores.
NCPU="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$NCPU"
export OPENBLAS_NUM_THREADS="$NCPU"
export MKL_NUM_THREADS="$NCPU"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"

echo "job:    ${SLURM_JOB_ID:-local} on $(hostname)"
echo "python: $PYTHON"
echo "args:   $*"
echo "start:  $(date)"

"$PYTHON" "$SCRIPT_DIR/run_mps_sweep.py" "$@"
rc=$?

echo "finish: $(date)  rc=$rc"
exit "$rc"
