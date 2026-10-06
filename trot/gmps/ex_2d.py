from trot.trial.mps import MpsTrial
from trot.ham.hubbard import HamHubbard
from trot.meas.mps import make_mps_meas_ops_hubbard
from trot.prop.m
import jax.numpy as jnp
import numpy as np

L = 4
U = 8
N = L*L
t = 1
h1 = np.zeros((N,N))
H = np.zeros((num_sites, num_sites))
for y in range(L):
    for x in range(L):
        site = y * L + x        
        if x + 1 < L:
            right_site = y * L + (x + 1)
            H[site, right_site] = -t
            H[right_site, site] = -t
            
        # Up neighbor
        if y + 1 < L:
            up_site = (y + 1) * L + x
            H[site, up_site] = -t
            H[up_site, site] = -t
eri = np.zeros((L, L, L, L))
for i in range(L):
    eri[i, i, i, i] = U

trial = MpsTrial()
meas_ops = make_mps_meas_ops_hubbard()

ham_hubbard_2D = HamHubbard(jnp.array(h1),U)