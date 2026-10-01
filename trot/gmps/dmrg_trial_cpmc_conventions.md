# `dmrg_trial_cpmc.ipynb` — conventions, and what validated them

Record of the checks that were run in that notebook, kept so the checks
themselves can be deleted without losing what they established. Every number
below was measured at `L=8`, `(4,4)`, `U/t=4`, DMRG `chi=8`, `seed=0`.

## pyscf: what it was for, and why it is gone

It was never *only* validation — two of its three jobs were load-bearing — but
all three turned out to be replaceable by code the notebook already had.

| was | replaced by |
|---|---|
| `fci.direct_spin1.absorb_h1e` + `contract_2e`, building `MH` and the sampled trial's `MHt` — i.e. the MSD energy estimator itself | `apply_H(X) = HOP @ X + X @ HOP.T + U * DOCC * X`, from the ED cell's own hopping matrix. `H` is one-body within each spin channel plus a diagonal, so it is two `NS x NS` products and an elementwise scaling |
| `fci.cistring.make_strings`, the determinant enumeration | `itertools.combinations`, which the ED cell already used. Any consistent order works once `contract_2e` is gone |
| `fci.direct_spin1.FCI()` -> `e_fci` | `e_exact` from the ED cell, which agrees to all 12 printed digits |

Verified before switching: `apply_H` reproduces `contract_2e` to **3.3e-16** on
the full coefficient matrix and **4.4e-16** on a masked one (the case Part 5
needs), and `E_exact` from the ED matches `E_FCI` exactly.

This does not cost the notebook its independence: `apply_H` and the `Dw=6` MPO
are still two implementations that share no code, which is what the two-route
CPMC comparison rests on.

## Conventions — the things that would silently produce a plausible wrong answer

- **Local index** `l = n_alpha + 2 n_beta`, i.e. `|0>, |a>, |b>, |ab>`. Read off
  `hamil.basis[0]`, which maps `{(0,0):0, (1,0):1, (0,1):2, (1,1):3}`. Do not
  assume it; pyblock3 is free to order its physical basis differently.
- **Densification.** A flat pyblock3 MPS becomes dense `(D_l, 4, D_r)` arrays by
  walking every block, reading its `(q_l, q_p, q_r)` labels and placing it at an
  offset from a bond map built from the blocks themselves. Checked against
  pyblock3's own `.amplitude()` on 40 random determinants: **max error ~1e-16**.
  A wrong bond offset does not raise, it just gives a wrong number.
- **The interleaving sign.** The MPS amplitude is for the interleaved operator
  order `a+_{0a} a+_{0b} a+_{1a} ...`; the determinant overlap
  `det(C_a[r_a,:]) det(C_b[r_b,:])` is natural in all-alpha-then-all-beta order.
  They differ by `(-1)^K`, `K = sum_{i in r_a} |{j in r_b : j < i}|`. With that
  sign applied, `M = ISG * AMP` **equals pyscf's FCI civec**, which is how the
  convention was pinned.
- **No parity insertion between spin channels** in the factorised overlap: for a
  bra and ket that are both spin products the interleaving sign appears once on
  each side of the same configuration and squares to 1. `overlap_u` is plainly
  `det(C_a^dag W_a) det(C_b^dag W_b)`.
- **Walker gauge.** The SD -> MPS conversion drops an overall factor. It is
  recovered either from an amplitude ratio or as `det(R_a) det(R_b) g_a g_b`
  from the QR and the plan; the two agree to **3.1e-15** (det gauge) and
  **3.2e-14** (ratio gauge) against the MSD overlap.

## What the deleted checks established

- **Two independent Hamiltonians agree.** `<Psi_T|H|Psi_T>` through pyscf's FCI
  H in the determinant basis and through the hand-built Dw=6 MPO differ by
  **1.5e-14**. They share no code, so this is the real check on Jordan-Wigner
  string placement in the MPO — the thing most likely to be silently wrong.
  (`-4.217221915802`; `E_FCI = -4.235806999130`, trial error `+1.86e-02`.)
- **The charge-blocked contraction is exact**, not an approximation: the padding
  is zero. Max rel difference to the dense MPS contraction **6.7e-16**, to the
  MSD sum **3.2e-14**. Worth 1.79x on the contraction alone, 1.22x end to end
  once the conversion is counted.
- **The two CPMC routes agree**: MPS-MPS and the 4900-determinant MSD sum give
  `-4.232837637575` with mean difference **1.5e-14** and per-block max
  **1.6e-14**, eleven orders below the statistical error (1.37e-03).
- **The fast sweep reproduces the propagation it replaces** bit for bit — same
  field choices, same node counts, same weights — against a reference that did
  a full overlap for each of the 2L field trials.
- **The perfect sampler draws the right distribution.** Against the exact
  4900-determinant probabilities from `M`: **chi^2 = 2791 on 2839 dof**, total
  variation 0.0167 where sampling noise predicts 0.0169, zero draws outside the
  support, amplitudes matching `AMP[a,b]` to 2.5e-16.
- **The DMRG trial is not a spin product**: spin-Schmidt rank 61 of 70, and
  `TV(p(n_a,n_b), p_a p_b) = 0.59`. Channel-wise sampling would be wrong by that
  much — this is why the sampler runs on the d=4 MPS at full bond dimension.

## Knobs measured and deliberately left off

- **Walker compression** (`CHI_WALKER`): at L=8 the walker MPS reaches chi=256
  and the discarded weight on the HF reference is exactly 0, so truncation only
  costs accuracy. Left `None` (exact).
- **Blocking the energy** (`BLOCK_ENERGY`): the H-side environment is 11.3x
  block-sparse but padding wastes 4.1x of that, and the measured local energy is
  *slower* blocked (7.18 ms vs 5.95 ms over 32 walkers). Left `False`. Fixing it
  needs size-bucketed padding, a bad trade at L=8 where these kernels are
  launch-bound. It is also nearly irrelevant: the propagation needs the overlap
  at every one of the 2L+2 = 18 field decisions per step against ONE energy
  evaluation per block — 360:1 at 20 prop steps.
- **`chi=8` for the trial** was chosen deliberately: `chi=100` reproduces the
  ground state to 1e-9, which leaves CPMC nothing to do. The bias study needs a
  trial whose error is resolvable above the noise.

## Sampled-MSD trial (Part 5)

Determinants drawn from `|Psi_T>` and kept with their exact coefficients form a
trial whose fidelity is `sum_{sampled} |c_d|^2`. 4e6 draws keep 2715 of the 2840
determinants with nonzero weight, fidelity 0.999979, and the CPMC energy lands
**3.4e-07** from the full-MSD run — 4000x inside its own error bar. 2000 draws
keep 486 determinants, fidelity 0.878, and miss by 1.8e-02. Unlike the HF case
in `mps_trial_cpmc.ipynb` there is no product structure to exploit, so the
support never completes and the agreement is not to round-off; it does not need
to be, because sampling visits determinants in proportion to `|c_d|^2`, which is
exactly the fidelity.
