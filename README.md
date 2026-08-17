# jaxPotts

> **Built with [Claude Code](https://claude.com/claude-code) (Anthropic).** The
> package, tests, docs, and comparison notebook were written by Claude, working
> from the reference implementations cited at the bottom.

Potts model (21-state Markov Random Field) inference over protein multiple
sequence alignments (MSAs) in [JAX](https://github.com/google/jax), for large
single GPUs. Two inference objectives share **one energy function and one
parameter array**:

- **`plm`** — pseudo-likelihood maximisation (in the style of CCMpred).
- **`bm`** — Boltzmann machine learning by persistent contrastive divergence
  (in the style of CCMpredPy).

The primary output is the raw coupling tensor `J` of shape `(L, q, L, q)`;
contact scores (zero-sum gauge → Frobenius norm → APC) are derived outputs.

## Install

```bash
pip install -e ".[dev,profiling]"
```

Python ≥ 3.11 with `jax`, `optax`, `numpy`, `msgpack`. For GPU, install a matching
`jax[cuda12]` build.

## Usage

```python
import jaxpotts as jp

A = jp.read_msa("family.a3m")            # (N, L) int8; also .fasta / .aln / .sto
w = jp.sequence_weights(A, cutoff=0.8)   # identity reweighting; N_eff = w.sum()

# Pseudo-likelihood
params, info = jp.plm.fit(A, weights=w)
J = jp.couplings(params.W)                        # (L, q, L, q) couplings
scores = jp.gauge.contact_score(params.h, J)      # (L, L) APC contact scores
jp.save_couplings("model.npz", params.h, J)       # compact triangular storage

# Boltzmann machine / PCD
bm_params, bm_info = jp.bm.fit(A, weights=w)
print("two-point correlation fit:", bm_info["final_corr"])
```

## How it compares

jaxPotts **generally recovers the results of established tools** — its couplings
and contact predictions track CCMpred (pseudo-likelihood) and CCMpredPy (PCD)
closely — and, by expressing the whole objective-and-gradient as `jit`-compiled
einsums on the GPU, it **can be considerably faster per iteration** than the
CPU-based references. It is a reimplementation, not a claim to beat these tools;
the goal is one clean codebase that runs both objectives under one set of
conventions. The head-to-head correctness and profiling comparison (against
CCMpred, CCMpredPy, and a PyTorch plmDCA) is in
[`notebooks/comparison.ipynb`](notebooks/comparison.ipynb).

## Conventions & decisions

The numerical conventions (alphabet, gauge, index order, energy sign) were fixed
by reading the reference source and are documented, with file:line citations, in
[`docs/conventions.md`](docs/conventions.md) and
[`docs/decisions.md`](docs/decisions.md).

## Development

```bash
ruff check src tests
JAX_PLATFORMS=cpu pytest -q -m "not gpu"     # CPU test suite (CI)
```

## Attribution

jaxPotts reimplements ideas from, and is validated against, several excellent
open-source projects:

- **[CCMpred](https://github.com/soedinglab/CCMpred)** (Seemayer, Gruber, Söding) —
  the pseudo-likelihood (`plm`) reference; jaxPotts matches its conventions and
  scoring.
- **[CCMgen / CCMpredPy](https://github.com/soedinglab/CCMgen)** (Vorberg, Seemayer,
  Söding) — the Boltzmann-machine / PCD (`bm`) reference and protocol.
- **[hnisonoff/potts](https://github.com/hnisonoff/potts)** (Hunter Nisonoff) — an
  independent PyTorch Potts implementation used as a third cross-check.
- **[evorca](https://github.com/suzuki-2001/evorca)** (Shosuke Suzuki) — a JAX
  plmDCA package; its A3M parsing and compact triangular coupling storage inspired
  the corresponding features here.

## License

AGPL-3.0-or-later — CCMpred and CCMgen are AGPL and jaxPotts reproduces their
numerical conventions and is validated against them (see
[`docs/decisions.md`](docs/decisions.md), D-013).
