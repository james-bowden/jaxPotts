# jaxPotts conventions

This file fixes the numerical conventions **before** the inference code is written.
Every entry is grounded in the reference source; see `decisions.md` for the
file:line citations and for cases where the two references disagree.

Reference repos (cloned under `.refs/`, gitignored):

| repo | commit | role |
|------|--------|------|
| `soedinglab/CCMpred`  | `2919b9c9ae976f73bc4dbb67908170afc3578da8` | C/CUDA, pseudo-likelihood (`plm`) reference |
| `soedinglab/CCMgen`   | `4540896203260e810b847916390c4e465d04be6b` | Python (CCMpredPy), Boltzmann/PCD (`bm`) reference |
| `hnisonoff/potts`     | `1b325129cc11` | PyTorch, third `plm` cross-check (see D-021) |

## 1. Alphabet and integer encoding

`q = 21`. **Gap is the LAST state, index 20 — not index 0.** The order is the
one hard-coded in `CCMpred/src/sequence.c` (`CHAR_INDICES`) and
`CCMgen/ccmpred/io/alignment.py` (`AMINO_ACIDS`), and the two agree exactly:

```
index:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
char:   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  -
```

`AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV-"`.

- Any character that is not one of the 20 standard amino acids — including
  `B Z J X U O`, the gap symbols `- .`, and `*` — maps to state **20** (gap).
  (C: `aatoi` returns 20 for anything non-alpha or unmapped; Python: unknown → 20.)
- The prompt's §3 assumption ("gap first, `-` = 0") is **overridden by the
  source.** We follow the source. A permuted alphabet would make `J` look wrong
  for no reason.

## 2. Array shapes and dtypes

| object | symbol | shape | dtype |
|--------|--------|-------|-------|
| one-hot MSA        | `X` | `(N, L, q)` | float32 |
| integer MSA        | `A` | `(N, L)`    | int8/int32 |
| sequence weights   | `w` | `(N,)`      | float32 |
| fields             | `h` | `(L, q)`    | float32 |
| couplings          | `J` | `(L, q, L, q)` | float32 |

Indexing: `J[i, a, j, b]` is the coupling between state `a` at site `i` and
state `b` at site `j`.

**Single-field gauge.** CCMpred stores single fields over **20** states only
(`nsingle = L*20`); the gap state has no field and is the fixed reference with
logit 0. Pair couplings are stored over the **full 21×21** (gap included on both
axes). We carry `h` as `(L, 21)` for uniformity but **pin `h[:, 20] = 0`** (the
gap column) so the extra gauge freedom is fixed and matches CCMpred. This is
asserted in code and tested.

Invariants (asserted and tested):

- `J[i, a, j, b] == J[j, b, i, a]` (symmetric)
- `J[i, :, i, :] == 0` (no self-coupling)
- `h[:, 20] == 0` (gap field pinned)

## 3. Energy and probability

```
logits[i, a] = h[i, a] + sum_{j != i} sum_b J[i, a, j, b] * X[·, j, b]
P(x_i = a | x_{-i}) = softmax_a(logits[i, a])
E(x) = - sum_i h[i, x_i] - sum_{i<j} J[i, x_i, j, x_j]
P(x)  ∝ exp(-E(x))
```

**Sign convention: positive.** Both references compute conditional logits as
`+v_i(a) + sum_{j != i} w_ij(a, x_j)`, with the gap logit held at 0, and softmax
over states. This matches the `E = -Σh - ΣJ`, `P ∝ exp(-E)` convention above.
There is no sign flip relative to the references.

## 4. Symmetry parameterization

Parameterize a free tensor `W` of shape `(L, q, L, q)` and symmetrize in the
forward pass:

```
J = 0.5 * (W + transpose(W, (2,3,0,1))) * offdiag_mask
```

where `offdiag_mask[i,·,j,·] = 0` iff `i == j`. Autodiff handles the gradient;
gradients are **not** hand-symmetrized. `fit` output is tested for the §2
invariants to float tolerance.

## 5. Sequence weighting

Both references cluster sequences at a fractional-identity cutoff (default
**0.8**) and weight each sequence by `1 / cluster_size`; `N_eff = sum(w)`.

The identity fraction is `(# columns where the two sequences are equal) / L`,
counting **all L columns including gap–gap matches** (the denominator is the full
alignment length, not the non-gap count). This is true of both references.

**The two references differ on the comparison operator** (this moves `N_eff` by a
few percent, hence the effective regularization):

| reference | rule | threshold |
|-----------|------|-----------|
| CCMpred (C)     | `n_identical  >  ceil(0.8·L)` (strict) | for `plm` parity |
| CCMpredPy (Py)  | `n_identical  >= ceil(0.8·L)`           | for `bm` parity |

`weights.py` exposes both via a flag (`inclusive: bool`), default chosen per the
objective being matched. Cost is `O(N^2 L)`; implemented as a chunked one-hot
matmul over row blocks, reducing inside each block so `(N, N)` is never
materialized.

## 6. Regularization

L2, applied to `h` and `J` separately. **The two references disagree on
`lambda_single` — the prompt's stated default (10) is CCMpredPy's, not CCMpred's:**

| reference | λ_single | λ_pair | single prior | factors |
|-----------|----------|--------|--------------|---------|
| CCMpred (C)    | **0.01** | `0.2·(L-1)` | 0 (v→0)          | obj: `λ_s‖v‖² + ½λ_p‖w‖²`; grad `2λx` both |
| CCMpredPy (Py) | **10**   | `0.2·(L-1)` | log-freq (`v-center`) | obj: `λ_s‖v-μ‖² + ½λ_p‖w‖²` |

Neither scales the pair penalty by `N_eff`. `plm.py` / `bm.py` expose
`lambda_single`, `lambda_pair_factor` (→ `λ_pair = factor·(L-1)`), and a single
prior (`zero` or `log_freq`) as arguments, defaulting to the reference being
matched. The `½` on the pair term and the `2λ` gradient (each edge stored twice)
are reproduced exactly for parity.

## 7. Initialization

- **Fields.** CCMpred: `h[i,a] = log(f_a) - log(f_gap)`, where
  `f_a = (count_a + 1) / (N + 21)` are pseudocounted (flat +1) **unweighted**
  single-site frequencies; `h[i, 20] = 0`. CCMpredPy initializes at the
  `v-center` log-frequency prior. We default to the CCMpred log-odds init for
  `plm` and the log-frequency prior for `bm`.
- **Couplings.** `J = 0` in both. (In `bm`/CD, CCMpredPy additionally **freezes
  the fields** — `fix_v=True` — and learns only `J`; see §9.)

## 8. Gauge, Frobenius norm, APC

`gauge.py` implements the standard **zero-sum gauge (ZSG)** by double-centering,
which is the canonical minimal-norm gauge and the right basis for comparing raw
`J` between implementations:

```
J_zs[i,a,j,b] = J[i,a,j,b] - mean_a[i,j,b] - mean_b[i,j,a] + mean_ab[i,j]
```
(means over the indicated state index of the `(q,q)` block), with the matching
correction applied to `h` so the distribution is unchanged. ZSG is idempotent and
leaves all energy *differences* unchanged; both are tested.

**The references do NOT use full double-centering for their scores** — they
subtract a single scalar mean per `(i,j)` block. We therefore provide, separately,
an exact-replica scoring path (`ccmpred_score`) for §7.1 parity, and record the
divergence in `decisions.md` (D-008). The reference-specific choices we replicate:

| step | CCMpred (C) | CCMpredPy (Py) |
|------|-------------|----------------|
| centering | scalar mean over full 21×21 block | scalar mean over 20×20 non-gap block |
| Frobenius | over 20×20, **gap excluded** | over 21×21, **gap included** |
| APC means | over all entries incl. diagonal (diagonal = 0) | same |

`gauge.py` exposes `frobenius_norm(J, include_gap: bool)` (default matches the
reference in play) and `apc(F)` with means over all entries (diagonal zeroed
beforehand), matching Dunn et al. 2004 and both references.

## 9. Boltzmann / PCD protocol (bm)

Ported from CCMpredPy's `--ofn-cd --persistent` path (there is **no `--pcd`
flag**; see D-014):

- **Chains** initialized from the data (tiled MSA rows), count
  `n_chains = max(N/10, 500)` by default.
- **Gibbs sweep**: `gibbs_steps = 1` by default = one pass over all sites in
  **random order** (Fisher–Yates per sweep). Conditional
  `P(x_i=a) ∝ exp(+v_i(a) + Σ_{j≠i} w_ij(a, x_j))` over the **20 amino acids**;
  **gaps are frozen** — a position that is a gap in the seed stays a gap, and no
  position ever samples into the gap state (D-011).
- **Persistence**: with `--persistent`, chains reinitialize from data each update
  (vanilla CD) until the learning rate decays below `alpha0/10`, then become
  truly persistent (samples written back to the chain buffer).
- **Gradient** (computed directly, not by autodiff):
  `dL/dJ = <xx>_model - <xx>_data` (and likewise `<x>` for `h`), where both sides
  are pseudocounted, weighted two-/one-point counts; gap gradients and the
  diagonal are zeroed. Fields are frozen (`fix_v=True`) by default in CD.
- **Optimizer**: plain gradient descent, `alpha0 = 1e-3` (CLI default; the code
  default is `0.05/sqrt(N_eff)`), "sig" (compounding) decay with `rate = 5e-6`
  that begins once `Δ‖w‖ < 0.1`.
- **Convergence**: relative change of `‖w‖` over the last 5 iterations `< 1e-5`;
  `maxit = 2000`.
- **Diagnostic** (the real convergence signal): Pearson correlation between model
  and empirical **connected** two-point correlations
  `c_ij(a,b) = f_ij(a,b) - f_i(a) f_j(b)`.

## 10. Pseudocounts

CCMpredPy default: **uniform** pseudocount (`1/21` per state), count `n = 1`,
mixing `τ = n / (N_eff + n)`. Singles: `f = (1-τ) f_emp + τ·(1/21)`. Pairs use a
connected-correlation-preserving mix:
`f_ij = (1-τ)² (f_ij^emp - f_i f_j) + f_i^pc f_j^pc`. Frequencies are normalized
over all 21 states (gaps included), then `degap`-renormalized over 20 where
needed.
