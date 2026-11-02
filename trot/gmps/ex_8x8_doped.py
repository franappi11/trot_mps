"""MPS-CPMC of the doped 8x8 Hubbard model (U = 8, 28 up + 28 down, open boundaries) with a saved DMRG trial.

The same run as ex_6x6_doped_diag.py, whose main does the work (dense walker legs, the MPO energy, szsz, the per-walker
measurements, the chunking, the manifest and the saved walkers; its docstring describes every flag), on the 8x8
lattice with 8x8 defaults:
- the lattice: --side 8 (the trial's h1 must be this lattice's; its nelec is used);
- the trial: the block2 DMRG trial at bond 1024 (slurm/dmrg_trial_8x8_block2.py, job 7315173; variational energy
  -42.6022, not converged: the last sweep still moved it by 2e-3), frozen as dmrg_chi1024_block2_7315173.npz so a
  later block2 run, which writes dmrg_chi1024_block2.npz, cannot change it under a queued job;
- the output: /mnt/ceph/users/fnappi/trot_walkers/sq8x8_doped/<tag> (tag: the trial file's name and the flags).
Any flag given on the command line overrides these. Through submit_ex_2d.sh the log is slurm/ex_8x8_doped-<jobid>.out:

    N_EQL=5 N_BLOCKS=10 ./submit_ex_2d.sh h100 ex_8x8_doped.py --chi-w 16 --measure-chi 16 --dense --mpo --no-szsz

Where a production block's time goes (--profile, trot.gmps.block_profile): the flags and TROT_ variables of the
production run (here job 7343266, chi_T 2048), so its compiled block comes from the persistent cache, and two blocks:

    TROT_SZSZ_GROUP=16 TROT_BLOCKS_PER_CALL=1 N_EQL=1 N_BLOCKS=1 TIME=03:30:00 CPUS=3 MEM=128G \\
        ./submit_ex_2d.sh h100 ex_8x8_doped.py --trial-file <trials>/dmrg_chi2048_block2.npz \\
        --chi-w 16 --measure-chi 64 --measure-chunk 2 --n-chunks 2 --dense --mpo --profile
"""

import sys
from pathlib import Path

from trot.gmps import ex_6x6_doped_diag  # configures JAX (float64) on import, as when it runs as a script

TRIALS = Path("/mnt/ceph/users/fnappi/trot_trials/uhf_dmrg_8x8_U8/state_00")
TRIAL = TRIALS / "dmrg_chi1024_block2_7315173.npz"

if __name__ == "__main__":
    ex_6x6_doped_diag.main(["--side", "8", "--trial-file", str(TRIAL), *sys.argv[1:]])
