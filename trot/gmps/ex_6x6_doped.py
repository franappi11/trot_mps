from trot.config import configure_once
configure_once()  
import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from trot.core.system import System
from trot.ham.hubbard import HamHubbard
from trot.trial.mps import make_mps_trial, rotate_spin, spin_rotation_y
from trot.prop.types import QmcParamsMps
from trot.gmps import trials
from trot.gmps.driver import run_qmc_mps

# persistent compile cache (as run_mps_cpmc.py --compile-cache): a rerun with the same chi, walkers and chunks loads
# the compiled block instead of compiling it again (12 min at chi 256, 20 min per chunk setting at chi 512)
jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/trot_jax_compile"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

U = 8.0
t = 1.0
L = 6
N = L * L
h1 = np.zeros((N, N))
for y in range(L):
    for x in range(L):
        site = y * L + x
        if x + 1 < L:  
            h1[site, site + 1] = h1[site + 1, site] = -t
        if y + 1 < L:  
            h1[site, site + L] = h1[site + L, site] = -t

ham = HamHubbard(jnp.asarray(h1), U)
parser = argparse.ArgumentParser()
parser.add_argument("--chi", type=int, default=256, help="bond dimension of the saved trial (default 256)")
chi_trial = parser.parse_args().chi
cutoff_trial =  1e-9   
trial = Path(f"/mnt/ceph/users/fnappi/trot_trials/uhf_dmrg_6x6_U8/state_00/dmrg_chi{chi_trial}.npz")
trial_path = trial
assert trials.htrial_cache_file(trial, cutoff_trial).exists(), \
    f"no H|trial> cache for chi {chi_trial} at cutoff {cutoff_trial:g}: run prepare_htrial_saved.py first"  
with np.load(trial) as z:
    n_up, n_dn = (int(x) for x in z["nelec"])                 
    assert np.allclose(z["h1"], h1) and float(z["u"]) == U     
    tensors = [z[f"A{i}"] for i in range(N)]                   
    charges = [z[f"q{i}"] for i in range(N + 1)]
    rdm1 = np.asarray(z["rdm1"])                               
    e_trial = float(z["energy"])                              

sys = System(N, (n_up, n_dn), "unrestricted")                 
(h_blocks, h_labels), hinfo = trials.load_or_make_htrial(trial, h1, U, tensors, charges, e_trial,relative_tolerance=cutoff_trial)
g = rdm1   
R = spin_rotation_y(beta_deg=90.0)
trial = make_mps_trial(tensors=rotate_spin(tensors=tensors, R=R), nelec=sys.nelec, rdm1=rdm1)  # N labels
h_blocks, h_labels = trials.rotate_htrial_blocks(h_blocks, h_labels, 90.0)
htrial = (h_blocks, h_labels, hinfo["trial_energy"])

params = QmcParamsMps(
    n_walkers=400,
    n_chunks=1,  # the steps in one batch of 400 walkers (fastest on a whole GPU); the energy and szsz are measured
    auto_n_chunks=True,  # in chunks of <= 20 walkers (engine.measure_chunks); the search raises n_chunks if it does not fit
    n_eql_blocks=int(os.environ.get("N_EQL", 50)),  # a short test: N_EQL=2 N_BLOCKS=4 OUT=...  
    n_blocks=int(os.environ.get("N_BLOCKS", 200)),
    dt=0.005,
    n_prop_steps=50,
    weight_floor=1e-8,
    seed=1234,
    #Adaptive uses the smaller block size that satisfies the tolerance
    occupation_tolerance=1e-10,
    orbital_plan="adaptive",
    #Bond dimension per spin
    walker_channel_chi=16,
    plan_reference="natural",  # determinant that freezes the conversion circuit: the trial's natural orbitals
    walker_start="natural",  # every walker starts from the trial's natural determinant
)

run = run_qmc_mps(
    sys=sys,
    params=params,
    ham_data=ham,
    trial_data=trial,
    htrial=htrial,  # precomputed H|trial>: no H|trial> product on the host
    target_error=None,
    mesh=None,
    observable_names=("szsz",),
)

OUT = Path(os.environ.get("OUT", f"/mnt/ceph/users/fnappi/trot_walkers/sq6x6_doped_T{chi_trial}_w16"))
OUT.mkdir(parents=True, exist_ok=True)
np.savez(OUT / "result.npz",
         energy=float(run.mean_energy), energy_err=float(run.stderr_energy),
         block_energies=np.asarray(run.block_energies), block_weights=np.asarray(run.block_weights),
         szsz=np.asarray(run.observable_means["szsz"]), szsz_err=np.asarray(run.observable_stderrs["szsz"]),
         szsz_blocks=np.asarray(run.block_observables["szsz"]), trial_file=str(trial_path))   
print(f"E = {float(run.mean_energy):.6f} +/- {float(run.stderr_energy):.6f}; saved to {OUT / 'result.npz'}")
