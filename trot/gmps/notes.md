# Slater determinants as MPS in block2 — working notes

Companion to `slater_mps.py`, `hubbard_dmrg.py`, `mps_overlap.py`.

## Status: NOTHING HERE HAS BEEN EXECUTED

All of this code was written without being run. Every module has a
`run_tests()`; run them in this order before trusting anything:

python slater_mps.py # algebra, exactness vs 4**n dense, autodiff
python hubbard_dmrg.py --test # MPO vs block2's published L=8 number
python mps_overlap.py # export conventions vs block2's own overlap


The three tests that carry essentially all the risk:

| test | what breaks if it fails |
|---|---|
| `slater_mps.test_exact_state` | gate signs / local basis order |
| `mps_overlap.test_matches_block2` | the block2 export convention |
| `hubbard_dmrg.test_free_fermion_limit` | the whole SD→MPS→block2 chain |

## The algorithm (Fishman & White, PRB 92, 075132 / arXiv:1504.07701)

A determinant is fixed by Λ_ij = ⟨a†_i a_j⟩, whose eigenvalues are exactly 0
or 1. That degeneracy is the resource: you may rotate freely inside the
occupied block and inside the empty block.

1. Diagonalize the leading B×B subblock of Λ; grow B until some eigenvalue is
   within tolerance of 0 or 1. For a low-entangled state this happens at
   small B.
2. That eigenvector is an approximate eigenvector of the *full* Λ, so B−1
   nearest-neighbour Givens rotations move it onto site 1, decoupling that
   site. Repeat on sites 2…B+1.
3. Reinterpret each single-particle rotation as a two-site number-conserving
   gate, start from the product state of the resulting occupations, and apply
   the gates **in reverse of the order they were derived**, SVD-truncating.

Exact contraction gives χ = 2^(B−1); truncation during the sweep usually gives
far less.

### Why the gate order is reversed

Ŵ(V) defined by Ŵ a†_α Ŵ† = Σ_i V_iα a†_i is a homomorphism,
Ŵ(V)Ŵ(V′) = Ŵ(VV′). With V = g₁g₂…g_M in derivation order,
|Ψ⟩ = Ŵ(V)|n⟩ = Ŵ(g₁)…Ŵ(g_M)|n⟩, so g_M acts on the product state first.

## Conventions — this is where the bodies are buried

Everything below is one consistent convention. Break any single line and the
overlaps silently come out wrong rather than erroring.

- **Site basis** (SZ mode, spatial orbitals, d=4):
  `|0⟩, |α⟩, |β⟩, |αβ⟩`, i.e. local index = `n_α + 2·n_β`.
  Matches `driver.basis` and renormalizer's merged ordering.
- **JW operator order**: `a†_{0α} a†_{0β} a†_{1α} a†_{1β} … |vac⟩`.
  So `|αβ⟩ = a†_{iα} a†_{iβ}|0⟩`, in that order.
- **Bond labels**: `(n_α, n_β)` accumulated **from the left**;
  `qn[0] = [[0,0]]`, `qn[n] = [[na, nb]]`.
- **MPS must be left-canonical** before `MPSTools.to_block2`.

### Two-site gates: build, don't derive

The 16×16 gate is built by exponentiating the one-body generator in the Fock
space of two adjacent spatial orbitals with explicit JW strings:

  Ŵ = exp(θ (a†_q a_p − a†_p a_q)),  Ŵ a†_p Ŵ† = cos θ a†_p + sin θ a†_q

This gets the JW string right by construction — for σ=α it passes through
`iβ`, for σ=β through `(i+1)α`. Deriving those signs by hand is how you lose
an afternoon. Since K³ = −K, `exp = I + sin θ·K + (1−cos θ)·K²` exactly, so no
`expm` call is needed.

### α and β must be compressed SEPARATELY

Running Fishman–White on the full 2K×2K spin-orbital Λ produces intermediate
gates that mix α and β. The *final* state still has good quantum numbers, but
the intermediate bonds don't, and `to_block2` needs per-bond `(n_α, n_β)`.
Compressing each spin channel on the spatial chain keeps every gate
Sz-conserving. The α and β generators commute exactly, so the two gate
sequences can be applied in either order or interleaved.

### The block2 export sign trap

block2's quimb tutorial applies a per-block sign

```python
f = lambda qs: -1 if ix != 0 and abs(qs[0].n) % 2 == 1 and abs(qs[1].n) % 2 == 1 else 1
```

**Do not copy this into a dense contraction.** It exists only to cancel signs
that symmray's `FermionicArray` inserts during fermionic contraction. block2
MPS tensors contract *plainly* to the state amplitudes — that's why
`driver.expectation(bra, identity_mpo, ket)` works. `test_matches_block2`
is the guard against getting this wrong.

Also: `MPSTools.from_block2` returns **rank-2 boundary tensors**, not rank-3
with dummy legs. The quimb example reveals this by passing `[]` for the first
tensor's left index.

Export recipe:
```python
driver.align_mps_center(mps, ref=0)
mps = driver.adjust_mps(mps, dot=1)[0]
py = MPSTools.from_block2(mps)   # .tensors[i].blocks[k].{q_labels, reduced}
```

## Differentiability

The algorithm has genuinely discrete decisions: block size B, occupied-vs-empty
choice, retained SVD rank. So it's split in two:

- **Plan** (NumPy): run it once, record every discrete choice.
- **Replay** (JAX): recompute with the plan frozen. jit-able, static shapes.

The plan is locally constant in C, so ∂/∂C with a frozen plan is the true
derivative almost everywhere. Re-plan when C moves.

Two numerical points that are not optional:

1. **Never `eigh` the block correlation matrix in the differentiable path.**
   The extremal eigenvector sits in a cluster with eigenvalues 1−10⁻¹²; the
   `eigh` VJP carries 1/(λᵢ−λⱼ) and blows up. Replaced by a spectral filter
   `M^(2^p)` applied to a *frozen* reference vector from the plan: polynomial
   in Λ_B, clean gradients, and it gauge-fixes the choice inside the cluster.
   Since 0 ≤ eig(M) ≤ 1, repeated squaring can't overflow.
2. **Broadened SVD VJP** (`set_svd_eps`). Degenerate singular values are common
   in free-fermion states at particle–hole symmetric points.

`arctan2(0,0)` also needs guarding to give zero gradient rather than NaN.

**block2 is not differentiable.** Gradients stop at the JAX tensors; the export
and the block2 MPO are constants.

## Overlaps

⟨SD(C)|Ψ_DMRG⟩ where Ψ is fixed: export Ψ once to dense arrays, keep the SD
side live in JAX, contract. O(L·D_bra·D_ket·(D_bra+D_ket)·4).

- Optimize **|⟨SD|Ψ⟩|²**, not the raw overlap. The determinant is invariant
  under rotations inside the occupied space; the squared overlap inherits that
  invariance, the raw one has a gauge manifold of flat directions.
- The overlap is that of the *truncated* SD MPS. `truncation_diagnostic`
  reports the infidelity you're accepting.
- Dense export costs L·D²·32 bytes — D=500 over 32 sites is ~2 GB. Beyond that,
  contract block-sparse using the exported `bond_qns`.

## Hubbard specifics

- MPO via `ExprBuilder`: `"cd"`/`"CD"` for the two spin hoppings, `"cdCD"` with
  `[i,i,i,i]` for U. Avoids the L⁴ `g2e` array.
- Reference: L=8, U/t=2, half filling, OBC → **E = −6.225634144662398**.
  Use this, not Lieb–Wu (which is PBC / thermodynamic limit).
- Guess: `h_σ = hopping + σ(Δ/2)(−1)^(x+y)`. Δ=0 is the Fermi sea; Δ≈U/4 gives
  an AFM determinant that converges far better at U/t ≳ 6. In 1D the true
  ground state has no long-range order, so watch the staggered magnetization
  decay across sweeps — if it doesn't, you have residual spurious order and
  need more noise.
- 2D: snake ordering, short direction in `ny`. χ grows ~exp(ny).
- **Orbital ordering dominates everything.** Fishman–White exploits locality of
  Λ in the site basis. Localized + Fiedler-ordered orbitals give small B and
  small χ; canonical delocalized MOs do not.

## Open items

- SGF / GHF path (spin-orbital sites, d=2) is not implemented. The gate
  derivation is unchanged; needs a d=2 two-site gate and total-N-only bond
  labels, plus a `from_renormalizer_dense_sz` equivalent.
- SU2 conversion via `MPSTools.trans_sz_to_su2` is untested here and only
  meaningful for closed-shell or high-spin determinants.
- An exact determinant–MPS overlap (sweeping C through Ψ, accumulating
  determinants of D×N_occ blocks, O(L·D²·N_occ), no truncation) would remove
  the truncation caveat but isn't an MPS–MPS contraction.