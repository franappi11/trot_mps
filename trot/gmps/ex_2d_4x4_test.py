from trot import config
# set the precision; before importing jax
config.configure_once()  
import os
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from pyblock3.algebra.core import SparseTensor, SubTensor
from pyblock3.algebra.mpe import MPE
from pyblock3.algebra.mps import MPS
from pyblock3.algebra.symmetry import SZ
from pyscf import ao2mo, gto, scf
from trot.core.system import System
from trot.driver import run_qmc
from trot.gmps.dmrg import hubbard_pyblock3_mpo, make_pyblock3_hamiltonian
from trot.gmps.utils import sd_to_gmps
from trot.ham.hubbard import HamHubbard
from trot.meas.mps import make_mps_meas_ops_hubbard
from trot.prop import mps_cpmc
from trot.prop.types import QmcParamsMps
from trot.trial.mps import (make_mps_trial, make_mps_trial_ops, make_walker_plan, mps_trial_from_pyblock3, rotate_spin,
                            spin_rotation_y)

from trot.trial.mps_rotation import rotate_mps_trial, rotated_rdm1



L = 4 
U = 8.0 
t = 1.0
N = L * L

h1 = np.zeros((N, N))
for y in range(L):
    for x in range(L):
        site = y * L + x
        if x + 1 < L:  
            h1[site, site + 1] = h1[site + 1, site] = -t
        if y + 1 < L:  
            h1[site, site + L] = h1[site + L, site] = -t

n_up = n_dn = 7
energy_fci = -10.1218936956 
#unrestricted walkers
sys = System(N, (n_up, n_dn), "unrestricted")
ham = HamHubbard(jnp.asarray(h1), U)  
params = QmcParamsMps(
    n_walkers=400,
    #When using a slice, I was getting oom without chunking
    n_chunks=4,  
    auto_n_chunks=False,  
    n_eql_blocks=50,  
    n_blocks=500,
    dt=0.005,
    n_prop_steps=50,
    weight_floor=1e-8,
    seed=1234,
    trial_chi=128,
    dmrg_sweeps=20,  # DMRG from the UHF MPS below
    dmrg_seed=0,  # DMRG noise
    #Adaptive uses the smaller block size that satisfies the tolerance
    occupation_tolerance=1e-10,
    orbital_plan="adaptive",
    #Bond dimension per spin
    walker_channel_chi=32,
    plan_reference="natural",  # determinant that freezes the conversion circuit: the trial's natural orbitals
    walker_start="natural",  # every walker starts from the trial's natural determinant
)
# (n_up, n_dn) of the local states |0>, |up>, |dn>, |up dn>
PHYSICAL = [(0, 0), (1, 0), (0, 1), (1, 1)]  

#needed to get the right format for pyblock3
def gmps_to_pyblock3(tensors, charges):
    """Dense charge-labelled site tensors (sd_to_gmps) as a flat pyblock3 SZ MPS."""
    q = lambda a, b: SZ(a + b, a - b, 0)
    out = []
    for A, q_left, q_right in zip(tensors, charges[:-1], charges[1:]):
        A, blocks = np.asarray(A), []
        for left in sorted(set(map(tuple, q_left))):
            rows = np.flatnonzero((q_left == left).all(axis=1))
            for p, (a, b) in enumerate(PHYSICAL):
                right = (left[0] + a, left[1] + b)
                cols = np.flatnonzero((q_right == right).all(axis=1))
                block = A[np.ix_(rows, [p], cols)]
                if np.any(block):
                    blocks.append(SubTensor(reduced=block, q_labels=(q(*left), q(a, b), q(*right))))
        out.append(SparseTensor(blocks=blocks))
    return MPS(tensors=out).to_flat()

#Good initial guess for UHF from 1000 random guesses
guess = np.array([
    [
        [0.75076012, 0.27653361, 0.6980688, 0.97107736, 0.77043689, 0.8369394, 0.15812254, 0.6175353,
         0.79330715, 0.57088681, 0.20399021, 0.42549081, 0.89881931, 0.4943242, 0.45270846, 0.79129473],
        [0.78069706, 0.20819978, 0.21511521, 0.98692276, 0.78455723, 0.58258721, 0.05701075, 0.18937558,
         0.23942091, 0.37406062, 0.14553415, 0.10556818, 0.7800008, 0.19908487, 0.38428448, 0.37229893],
        [0.88754148, 0.98661623, 0.65438665, 0.53776023, 0.82958981, 0.41326864, 0.48363917, 0.57636339,
         0.9892867, 0.26327387, 0.27771919, 0.12498157, 0.54516244, 0.03630874, 0.39598281, 0.07177873],
        [0.99921329, 0.21369474, 0.15128791, 0.27887068, 0.96444567, 0.02475795, 0.11635932, 0.84793116,
         0.57228343, 0.84196897, 0.87500146, 0.63912739, 0.25793519, 0.51592524, 0.81387518, 0.52235427],
        [0.97616933, 0.01964004, 0.04025243, 0.08142407, 0.99114232, 0.07187789, 0.63014811, 0.41160792,
         0.28216033, 0.54048377, 0.48380253, 0.01950652, 0.89331896, 0.80248497, 0.86722197, 0.61649518],
        [0.9715288, 0.67432487, 0.99161173, 0.31062219, 0.51452161, 0.47880658, 0.71360043, 0.34361465,
         0.51863457, 0.48911563, 0.27872079, 0.22240251, 0.96696755, 0.37866449, 0.90758556, 0.85299769],
        [0.4957909, 0.95178741, 0.27891893, 0.9075632, 0.96342521, 0.55249114, 0.94335436, 0.47466161,
         0.03013638, 0.27819025, 0.06483842, 0.49727347, 0.00741089, 0.46277917, 0.81123496, 0.47434033],
        [0.13657056, 0.75766829, 0.49139165, 0.34512082, 0.16897428, 0.4261986, 0.75756795, 0.50796457,
         0.49497169, 0.44021553, 0.47605415, 0.82261906, 0.74545421, 0.70256194, 0.52534715, 0.03807541],
        [0.70628896, 0.12026158, 0.22414101, 0.28673373, 0.78480298, 0.19721606, 0.45967185, 0.57699539,
         0.38388576, 0.67991101, 0.77059728, 0.37657445, 0.64504263, 0.95894785, 0.04407314, 0.86233547],
        [0.29357684, 0.11372996, 0.26585715, 0.17580707, 0.87072454, 0.51386508, 0.27607985, 0.83132104,
         0.34924311, 0.54237293, 0.58005198, 0.09687853, 0.93752234, 0.20556139, 0.66403246, 0.63596251],
        [0.36135881, 0.72526821, 0.71689221, 0.65931593, 0.35720631, 0.52851693, 0.16967331, 0.96738802,
         0.01286938, 0.18738756, 0.20993752, 0.48130631, 0.95032016, 0.01362053, 0.60367796, 0.20998394],
        [0.42905651, 0.24397045, 0.35661151, 0.08902126, 0.37393682, 0.07913849, 0.93262882, 0.26460756,
         0.97935314, 0.14303962, 0.40718209, 0.27724404, 0.98249321, 0.56358376, 0.62868767, 0.18953603],
        [0.24028549, 0.37947398, 0.82887453, 0.14998165, 0.4928008, 0.69412689, 0.55958313, 0.36810744,
         0.52231783, 0.32153256, 0.40939316, 0.55284913, 0.49750536, 0.27330227, 0.78181502, 0.37387601],
        [0.83263499, 0.58656411, 0.98482181, 0.81474768, 0.58766468, 0.20276753, 0.00677464, 0.32574174,
         0.94431331, 0.23721579, 0.27671377, 0.90764552, 0.39868546, 0.85282589, 0.89672055, 0.2382185],
        [0.81326954, 0.33858362, 0.35630785, 0.8072706, 0.71923665, 0.56555937, 0.39668511, 0.53810474,
         0.68976348, 0.94470632, 0.94080505, 0.29736956, 0.55262099, 0.46401708, 0.66180925, 0.30968995],
        [0.59659035, 0.46994336, 0.63798735, 0.99553574, 0.09049419, 0.21361627, 0.63103854, 0.79610985,
         0.30732195, 0.0569342, 0.03047431, 0.8591928, 0.97661052, 0.9625009, 0.69733499, 0.581413],
    ],
    [
        [0.16436664, 0.62154043, 0.00995623, 0.60048965, 0.73425145, 0.20879531, 0.41687351, 0.22012745,
         0.27172113, 0.76024265, 0.05735967, 0.12389932, 0.38198934, 0.7652735, 0.02055702, 0.28695025],
        [0.1316369, 0.86341843, 0.26348876, 0.09039339, 0.08920206, 0.37755263, 0.53339648, 0.37236779,
         0.12752738, 0.54547706, 0.10965577, 0.19073205, 0.2480909, 0.33866172, 0.33457506, 0.59569004],
        [0.17667472, 0.50840882, 0.09780392, 0.7821188, 0.24415584, 0.23879575, 0.06158067, 0.73024575,
         0.54349628, 0.26417103, 0.39327854, 0.68847834, 0.31535654, 0.49348348, 0.40664343, 0.59147479],
        [0.23384613, 0.34907866, 0.44923883, 0.13656648, 0.11938385, 0.16189219, 0.80464477, 0.1839937,
         0.9225331, 0.71438484, 0.69654959, 0.4014952, 0.44499181, 0.47469776, 0.38062839, 0.47065685],
        [0.95409905, 0.73755791, 0.77777363, 0.46947319, 0.86907864, 0.7082999, 0.33217912, 0.80771402,
         0.75829661, 0.7073957, 0.68958141, 0.78099293, 0.38186484, 0.36517162, 0.25731457, 0.15774519],
        [0.71162823, 0.73862706, 0.40014244, 0.33350543, 0.79810301, 0.05301272, 0.76171565, 0.42668717,
         0.69671208, 0.42277904, 0.42606743, 0.73037188, 0.23621359, 0.12330348, 0.30745391, 0.70617495],
        [0.51050606, 0.19603413, 0.89274725, 0.52586436, 0.1575693, 0.16459398, 0.19804642, 0.37482335,
         0.3777602, 0.85490373, 0.72309903, 0.06474041, 0.69771403, 0.28506845, 0.34749016, 0.0741271],
        [0.53240807, 0.17090797, 0.46866417, 0.97125566, 0.79426909, 0.11863844, 0.86641367, 0.07404193,
         0.21844544, 0.21694581, 0.01131456, 0.92012094, 0.37430381, 0.31784452, 0.06787438, 0.98743957],
        [0.61841098, 0.29619861, 0.47959338, 0.56709808, 0.62862886, 0.22123276, 0.82394853, 0.93734225,
         0.07320207, 0.75326384, 0.75807238, 0.23996092, 0.96131012, 0.68219653, 0.00104766, 0.29254256],
        [0.9764697, 0.62155454, 0.45707704, 0.53488345, 0.27876563, 0.89974823, 0.58377781, 0.080501,
         0.39453364, 0.17153285, 0.8727495, 0.42653552, 0.46799817, 0.42245806, 0.11336747, 0.79821096],
        [0.57448616, 0.9004649, 0.99103502, 0.13635225, 0.90742023, 0.27917544, 0.09125311, 0.17668075,
         0.39528365, 0.83329359, 0.0234404, 0.4079667, 0.29669353, 0.16223869, 0.85764146, 0.71359242],
        [0.03303084, 0.98472551, 0.75681916, 0.90370931, 0.83241238, 0.03068654, 0.15052535, 0.31740688,
         0.59855773, 0.2927867, 0.139432, 0.04455445, 0.28989224, 0.15719747, 0.42922803, 0.13675193],
        [0.85160951, 0.00781551, 0.69261986, 0.07119657, 0.37827026, 0.78608237, 0.95707726, 0.12924759,
         0.71746855, 0.75925543, 0.06907793, 0.63140495, 0.12952271, 0.79181269, 0.2967364, 0.47408798],
        [0.89098831, 0.93072596, 0.21548937, 0.69098913, 0.37101226, 0.54844022, 0.30878554, 0.906508,
         0.66059659, 0.46869519, 0.21729489, 0.23392569, 0.89228108, 0.30944771, 0.94090598, 0.14437027],
        [0.98873245, 0.5530111, 0.07984195, 0.88424915, 0.02163562, 0.01757612, 0.95632441, 0.37626658,
         0.32748192, 0.84891738, 0.20821406, 0.64187347, 0.64061216, 0.26059112, 0.40123856, 0.03319099],
        [0.65654745, 0.60787289, 0.0713427, 0.29191574, 0.01290367, 0.08534728, 0.71645442, 0.47772829,
         0.53493122, 0.79214499, 0.06529363, 0.05014535, 0.11688542, 0.28131812, 0.69212041, 0.02059348],
    ],
])  
#UHF calculation
eri = np.zeros((N,) * 4)
for i in range(N):
    eri[i, i, i, i] = U
mol = gto.M(verbose=0)
mol.incore_anyway = True
mol.nelec = (n_up, n_dn)
mf = scf.UHF(mol)
mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: np.eye(N)
mf._eri = ao2mo.restore(8, eri, N)
mf.kernel(guess)
print(f"\nUHF energy {mf.e_tot:.10f} (converged: {mf.converged})")
mo1 = mf.stability()[0]
dm1 = mf.make_rdm1(mo1, mf.mo_occ)
mf = mf.run(dm1)
mf.stability()
Ca, Cb = mf.mo_coeff[0][:, :n_up], mf.mo_coeff[1][:, :n_dn]
rdm_up, rdm_dn = (np.diag(dm) for dm in mf.make_rdm1())
print("n_up - n_dn:\n", np.round((rdm_up - rdm_dn).reshape(L, L), 2))
print("n_up + n_dn:\n", np.round((rdm_up + rdm_dn).reshape(L, L), 2))
#Convert UHF into mps
chi_channel = 16
gmps = sd_to_gmps(Ca, Cb, chi=chi_channel)
mps = gmps_to_pyblock3(gmps.tensors, gmps.charges)
#Convert Hamiltonian into MPO
hamiltonian = make_pyblock3_hamiltonian(h1, (n_up, n_dn))
mpo = hubbard_pyblock3_mpo(hamiltonian, h1, U)
energy = lambda psi: float(MPE(psi, mpo, psi)[0:2].expectation) / float(psi @ psi)
print(f"MPS bonds {mps.show_bond_dims()}, discarded weight {gmps.discarded:.1e}")
print(f"<MPS|H|MPS> = {energy(mps):.10f}   UHF {mf.e_tot:.10f}")
np.random.seed(params.dmrg_seed) 
#Run DMRG
MPE(mps, mpo, mps).dmrg(bdims=[params.trial_chi] * 6, noises=[1e-5] * 4 + [0.0] * 2, dav_thrds=[1e-9], iprint=0,
                        n_sweeps=params.dmrg_sweeps)
e_dmrg = energy(mps)
trial = mps_trial_from_pyblock3(mps, nelec=(n_up, n_dn))

print(f"DMRG trial (from UHF): bonds max {max(trial.bond_dims)}, variational energy {e_dmrg:.10f}, "
      f"E - energy_fci {e_dmrg - energy_fci:+.6f}", flush=True)

#Taking 1rdm before rotation
rdm1 = np.asarray(trial.rdm1)
#Spin-projecting the trial  
R = spin_rotation_y(beta_deg=90.0)  # exp(-i beta S^y), 90 degrees
trial = make_mps_trial(tensors=rotate_spin(tensors=trial.tensors, R=R), nelec=sys.nelec, rdm1=rdm1)  # N labels
print(f"rotated trial: bonds max {max(trial.bond_dims)}", flush=True)
#Setting up computation
plan = make_walker_plan(ham_data=ham, trial=trial, sys_=sys, params=params) 
trial_ops = make_mps_trial_ops(plan=plan)
meas_ops = make_mps_meas_ops_hubbard(plan=plan, energy_kernel=params.energy_kernel, htrial=None)
prop_ops = mps_cpmc.make_prop_ops(ham_data=ham, sys=sys, plan=plan)
meas_ctx = meas_ops.build_meas_ctx(ham, trial) 
prop_ctx = prop_ops.build_prop_ctx(ham, trial_ops.get_rdm1(trial), params) 

run = run_qmc(
    sys=sys,
    params=params,
    ham_data=ham,
    trial_data=trial,
    meas_ops=meas_ops,
    trial_ops=trial_ops,
    prop_ops=prop_ops,
    block_fn=mps_cpmc.block, 
    state=None,  
    meas_ctx=meas_ctx,
    prop_ctx=prop_ctx,
    target_error=None,
    mesh=None,
    observable_names=(),
    runtime=None,
)

print(f"E = {float(run.mean_energy):.6f} +/- {float(run.stderr_energy):.6f}   (E - energy_fci {float(run.mean_energy) - energy_fci:+.6f})")
