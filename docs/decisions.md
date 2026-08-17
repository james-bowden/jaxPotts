# jaxPotts decision log

Append-only. Each entry: context, decision, alternatives rejected, evidence.
Reference commits: CCMpred `2919b9c9`, CCMgen (CCMpredPy) `4540896`.
All line numbers are against those commits under `.refs/`.

---

## D-001: Alphabet order — gap is state 20, not 0
Date: 2026-08-17
Context: The build prompt §3 assumes "gap first (`-` = 0), then the 20 amino
acids". TODO-VERIFY against the source.
Decision: Use `A R N D C Q E G H I L K M F P S T W Y V -`, i.e. gap = **20**,
`A = 0`. Non-standard chars (B Z J X U O), gap symbols (`- .`) and `*` all map to
20.
Alternatives rejected: gap = 0 (the prompt's assumption) — would permute every
`(q,q)` block and make `J` unmatchable to the reference.
Evidence: `CCMpred/src/sequence.c:14-17` `CHAR_INDICES` table (`A=65…V=86, -=45`
in index order 0..20) and `AMINO_INDICES:11-13`; `aatoi:21-27` returns 20 for
non-alpha/unmapped. `CCMgen/ccmpred/io/alignment.py:5`
`AMINO_ACIDS="ARNDCQEGHILKMFPSTWYV-"`; `CCMgen/ccmpred/counts/msacounts.c:56`
unknown→20. NOTE: the docstring at `sequence.c:18-19` ("0 = gap") is **stale and
wrong** — the tables and all downstream code use gap = 20.

## D-002: Parameter shapes — singles over 20, pairs over 21×21
Date: 2026-08-17
Context: TODO-VERIFY whether CCMpred parameterizes singles over 20 or 21 states.
Decision: Store `J` as `(L,21,L,21)` (gap included both axes). Carry `h` as
`(L,21)` for uniformity but **pin `h[:,20]=0`**, matching CCMpred's 20-state
single fields with gap as the fixed reference (logit 0). Assert/test the pin.
Alternatives rejected: 21 free single states — introduces an unfixed gauge dof
that CCMpred pins to 0; would not match their raw `v`.
Evidence: `CCMpred/include/ccmpred.h:3-19` (`N_ALPHA=21`, `nsingle=ncol*20`,
macros `V`, `W`); `CCMpred/src/ccmpred.c:387-390`
(`nsingle=ncol*(N_ALPHA-1)`, `nvar=nsingle+ncol*ncol*21*21`).
`CCMgen/ccmpred/objfun/cd/__init__.py:39-40` (`nsingle=ncol*20`,
`npair=ncol*ncol*21*21`); PLL path pads singles to mult. of 32 and pairs 21→32
(`pll/__init__.py:52-54`) — padding is a memory-layout detail only, not numerics.

## D-003: Energy sign convention — positive logits, gap logit 0
Date: 2026-08-17
Context: TODO-VERIFY the sign; a flip would invert couplings vs the reference.
Decision: `logits[i,a] = +h[i,a] + Σ_{j≠i} J[i,a,j,·]·x_j`, gap logit fixed to 0,
softmax over states; equivalently `E=-Σh-ΣJ`, `P∝exp(-E)`. Matches the prompt.
Evidence: `CCMpred/src/evaluate_cpu.c:56-75` (`PC(a,s)=V(s,a)`, `PC(gap,s)=0`,
`*p++ += *w++`); log-partition and NLL `evaluate_cpu.c:78-100`
(`fx += weight*(-PC(xik,k)+logZ)`). CCMpredPy sampler
`CCMgen/ccmpred/objfun/cd/cext/cd.c:24-58` (`cond_probs[a]=E1(i,a)`,
`+= E2(...)`, `cond_probs[GAP]=0`).

## D-004: Sequence weighting — identity over all L columns, differing operator
Date: 2026-08-17
Context: TODO-VERIFY threshold, `>=` vs `>`, gap handling in the denominator.
Decision: Identity = (# equal columns)/L over **all L columns** (gap–gap counts
as a match); weight = `1/cluster_size`; default cutoff 0.8. Expose an
`inclusive` flag: strict `>` (default for `plm`, matching C) vs `>=` (default for
`bm`, matching Py).
Alternatives rejected: non-gap denominator — neither reference does this.
Evidence: C `CCMpred/src/reweighting.c:9-58`: `idthres=ceil(0.8*ncol)`, compares
all `ncol` columns, `if(ids > idthres)` (**strict**), `w=1/(w-1)`; default 0.8 at
`ccmpred.c:267`. Py `CCMgen/ccmpred/weighting/cext/weighting.c:58-114`:
`idthres=ceil(cutoff*ncol)`, all columns, `if(my_ids >= idthres)` (**inclusive**),
`weights=1/(weights-1)`; default `--wt-cutoff 0.8`.

## D-005: Regularization — λ_single differs 0.01 (C) vs 10 (Py)
Date: 2026-08-17
Context: TODO-VERIFY defaults and factors. The prompt states λ_single=10 as
"CCMpred's default"; the source shows this is CCMpredPy's, not C CCMpred's.
Decision: `plm` (matching C) defaults λ_single=**0.01**, single prior 0;
`bm` (matching Py) defaults λ_single=**10**, single prior = log-freq (`v-center`).
Both: `λ_pair = 0.2·(L-1)`, **not** scaled by N_eff. Objective
`λ_s‖v-μ‖² + ½λ_p‖w‖²`; pair gradient uses `2λ_p·w` (each edge stored twice).
All exposed as arguments; the prompt's "10" is available but not the C default.
Alternatives rejected: a single shared default — would mismatch one reference.
Evidence: C `CCMpred/src/ccmpred.c:225-227` (`lambda_single=0.01`,
`lambda_pair_factor=0.2`), `:500-502` (`lambda_pair=factor*(L-1)`);
penalty/grad `evaluate_cpu.c:151-163` (`λ_s x²` no ½ for singles, `0.5 λ_p x²`
for pairs, grad `2λx` both). Py `CCMgen/ccmpred/scripts/run_ccmpred.py:137-146`
(`--reg-lambda-single 10`, `--reg-lambda-pair-factor 0.2`);
`regularization.py:15-32` (`λ_s‖v-μ‖² + 0.5 λ_p‖w‖²`, single grad `2λ_s(v-μ)`);
`centering.py:3-18` (`v-center` = centered log single-frequencies);
`__init__.py:403-406` (`multiplier=L-1`).

## D-006: Initialization — log-odds fields (C) / log-freq prior (Py), J=0
Date: 2026-08-17
Context: TODO-VERIFY the field init and pseudocount value.
Decision: `plm`: `h[i,a]=log((c_a+1)/(N+21)) - log((c_gap+1)/(N+21))`, unweighted
counts, flat pseudocount 1, `h[i,20]=0`; `J=0`. `bm`: `h` = `v-center` log-freq
prior, `J=0`, and `h` is frozen thereafter (see D-010).
Evidence: C `CCMpred/src/ccmpred.c:90-131` (`init_bias`: `aacounts` start at 1,
`aasum=nrow+21`, `V=log(f_a)-log(f_gap)`, whole vector memset to 0 first).
Py `CCMgen/ccmpred/__init__.py:440` (init at `v-center`), `centering.py:3-18`.

## D-007: Optimizer — C nonlinear CG; Py LBFGS (pll) / GD (cd)
Date: 2026-08-17
Context: TODO-VERIFY the optimizer, max iters, epsilon.
Decision: For `plm` parity offer (a) `optax.lbfgs` (default; convex problem) and
(b) a CG variant matching CCMpred (max 250 iters, ε=0.01 relative-change over a
5-iteration window). For `bm`, gradient descent per CCMpredPy.
Evidence: C is Fletcher–Reeves nonlinear CG in `lib/libconjugrad/src/conjugrad.c`;
defaults overridden at `CCMpred/src/ccmpred.c:528-534` (`max_iterations=250`,
`epsilon=0.01`, window `k=5`, linesearch 5, ftol 1e-4, wolfe 0.2). Help text
"[default: 50]" (`ccmpred.c:194`) is stale — real default 250 (`ccmpred.c:222`).
Py PLL uses LBFGS; CD uses `gradient_descent.py` (see D-010).

## D-008: Scoring gauge — references use scalar centering, not double-centering
Date: 2026-08-17
Context: The prompt §3 specifies full double-centering ZSG and asks to
TODO-VERIFY the Frobenius gap-inclusion and APC diagonal-inclusion.
Decision: `gauge.py` implements the standard double-centering ZSG (canonical
minimal-norm gauge) as the basis for comparing raw `J`. Separately provide an
exact-replica `ccmpred_score` path for §7.1 parity: scalar-mean centering,
Frobenius over 20×20 (C) or 21×21 (Py), APC over all entries incl. diagonal.
Alternatives rejected: using scalar centering everywhere — inferior gauge for raw
`J` comparison; using double-centering for the parity score — would not reproduce
the reference `.mat`.
Evidence: C `CCMpred/src/util.c:16-64` (`sum_submatrices`: subtract one mean over
all 21×21, Frobenius over `a,b∈[0,19]` — **gap excluded**), `:66-108` (`apc`:
row/col/total means over all entries, diagonal pre-zeroed). Py
`CCMgen/ccmpred/io/contactmatrix.py:7-16` (`frobenius_score` sums full 21×21 —
**gap included**), `:18-30` (APC over all entries incl. diagonal);
`sanity_check.py:36-53` (scalar centering over 20×20 block).

## D-009: Raw coupling output formats
Date: 2026-08-17
Context: TODO-VERIFY the flag/format to dump raw MRF parameters.
Decision: Parse both. C text (`-r`): L lines × 20 singles, then per `i<j` a
`# i j` header + 21×21 block (`W(b,j,a,i)`). Py/C msgpack (`-b`, gzip if `.gz`):
`{format:"ccm-1", ncol, x_single:[L*20], x_pair:{"i/j":{i,j,x:[21*21]}}}`.
`reference.py` reads the msgpack (matching CCMgen's shipped `1atzA.braw.gz`) and
the text raw.
Evidence: C `CCMpred/src/io.c:31-60` (text `write_raw`), `:106-197` (msgpack).
Py `CCMgen/ccmpred/raw/ccmraw.py:148-171` (`write_msgpack`), `:58-79`
(`parse_msgpack`). CAVEAT: `parse_msgpack` passes `encoding="utf-8"` (removed in
msgpack≥1.0) — handled in our own reader (D-016).

## D-010: PCD/CD protocol ported from CCMpredPy
Date: 2026-08-17
Context: TODO-VERIFY the full training schedule for `bm` (the prompt's "--pcd"
path). This is the hard part of Boltzmann learning.
Decision: Port exactly (see conventions §9): chains from data,
`n_chains=max(N/10,500)`; `gibbs_steps=1` random-order sweep; gaps frozen;
gradient `<xx>_model-<xx>_data` on pseudocounted weighted counts with gap+diagonal
zeroed; fields frozen (`fix_v=True`); gradient descent `alpha0=1e-3`, "sig"
compounding decay `rate=5e-6` starting when `Δ‖w‖<0.1`; persistence engages only
after `alpha<alpha0/10`; convergence `Δ‖w‖<1e-5` over 5 iters, `maxit=2000`.
Evidence: `CCMgen/ccmpred/objfun/cd/__init__.py:60-183`,
`ccmpred/algorithm/gradient_descent.py:9-157`, sampler
`ccmpred/objfun/cd/cext/cd.c:24-58,122-180`.

## D-011: Gibbs sampler freezes gaps
Date: 2026-08-17
Context: Sampler gap handling (unstated in the prompt).
Decision: A position that is a gap in the seed sequence is never resampled, and
no position ever samples into the gap state (multinomial over 20 aa only). The
seed's gap pattern is preserved. Provide this as the default; a no-gap-freezing
variant is available for the exact-enumeration tests (which have no gaps).
Evidence: `CCMgen/ccmpred/objfun/cd/cext/cd.c:127-137` (`if(seq[...]!=GAP)`),
`pick_random_weighted(pcondcurr, N_ALPHA-1)` samples 20 states.

## D-012: Pseudocounts — uniform 1/21, n=1, τ=n/(Neff+n)
Date: 2026-08-17
Context: TODO-VERIFY CCMpredPy pseudocount default.
Decision: Default uniform pseudocount `1/21`, count 1, `τ=1/(N_eff+1)`; singles
`f=(1-τ)f_emp+τ/21`; pairs connected-correlation-preserving
`f_ij=(1-τ)²(f_ij-f_i f_j)+f_i^pc f_j^pc`.
Evidence: `CCMgen/ccmpred/pseudocounts.py:78,92-93,105-108,135-138`;
`run_ccmpred.py:159-171`.

## D-013: License — AGPL-3.0-or-later (flag to owner)
Date: 2026-08-17
Context: CCMpred and CCMgen are AGPL-3.0-or-later. §2 asks which case applies.
Decision: Default to **AGPL-3.0-or-later**. jaxPotts is an independent JAX
reimplementation — no source lines are copied — but it reproduces numerical
conventions and one factual lookup table (the 21-char alphabet ordering) read
directly from AGPL sources, and the whole package is designed to interoperate
with and be validated against them. The conservative and clearly-compatible
choice is AGPL. **FLAGGED TO REPO OWNER:** if you are confident this is a
clean-room reimplementation and prefer a permissive license (MIT/Apache-2.0),
say so and we will re-evaluate the alphabet table (the only verbatim artifact)
and switch. Until then `LICENSE` is AGPL-3.0-or-later.

## D-014: CCMpredPy CLI — no `--pcd` flag; it is `--ofn-cd --persistent`
Date: 2026-08-17
Context: The prompt refers to "the `--pcd` path". No such flag exists.
Decision: Drive the reference Boltzmann run with
`ccmpred --ofn-cd --persistent [--nr-markov-chains N --gibbs_steps K ...]
-b out.braw.gz`. Recorded so §7.1b is a fair comparison against the real path.
Evidence: `CCMgen/ccmpred/scripts/run_ccmpred.py:62,72-94`; `setup.py:52-60`
(entry point `ccmpred=...run_ccmpred:main`).

## D-015: Data — only 1atzA ships; Pfam authorized for profiling only
Date: 2026-08-17
Context: Ground rule 2 forbids inventing/downloading data for correctness. The
only MSA shipped by either repo is `1atzA` (3068 seqs × 75 cols), present as
`CCMpred/example/1atzA.aln` and `CCMgen/example/1atzA.fas` (same protein).
Reference outputs shipped: `CCMgen/example/1atzA.braw.gz` (learned couplings),
`1atzA.apc.mat`, `1atzA.noapc.mat`, `1atzA.ec.mat`, `1atzA.pdb`.
Decision: §7.1 correctness uses **only** the shipped `1atzA`. For §7.2 profiling,
the repo owner explicitly authorized pulling standard Pfam pre-built alignments
(target 10–100 families spanning the L×N plane). Those are used for **timing
only**, never for correctness claims, and each is labeled with its Pfam accession
and download provenance in the notebook.
Evidence: `find` over both repos (only `1atzA.aln`/`1atzA.fas`); user message
2026-08-17 authorizing Pfam for benchmarking.

## D-016: Reference-tool build/runtime environments
Date: 2026-08-17
Context: CCMpred (C) builds cleanly; CCMpredPy is a Python-2/3-era package with
3.13 hazards (`np.float`, `msgpack encoding=` kwarg, `plotly==3.0.0rc10` pin).
Decision: Build CCMpred CPU+OpenMP (`cmake . -DWITH_CUDA=off && make`) — and,
if the toolchain allows, a CUDA build for the profiling comparison. Run
CCMpredPy in a **separate** mamba env on Python ≤3.10 with pinned/patched
`msgpack`, `numpy`, `plotly`, keeping the main `jaxPotts` env clean (owner
approved separate envs). jaxPotts ships its own msgpack raw reader so we do not
depend on CCMpredPy's broken `parse_msgpack` for comparison.
Evidence: `CCMgen/setup.py:6-15`; agent audit of `np.float`
(`gradient_descent.py:26`), `msgpack.unpackb(encoding=...)` (`ccmraw.py:62`).

## D-017: bm gradient in count units; pseudocounts stabilise training
Date: 2026-08-17
Context: An early bm implementation diverged (``||w||`` -> 450, two-point fit 0.49):
(a) model frequencies were unweighted while data frequencies were weighted, so the
gradient learned the weighting artifact; (b) no pseudocounts, so pairs with
``f=0`` drove couplings to blow up under CD-1.
Decision: Compute model frequencies with per-chain weights (each chain carries the
weight of the data sequence it was seeded from, matching CCMpredPy's re-weighted
sample side); scale the gradient by ``N_eff`` (the true weighted-log-likelihood
gradient is ``N_eff*(f_data - f_model)``, count units, balancing the ``lambda*J``
penalty as in CCMpredPy); apply CCMpredPy's uniform pseudocounts
(``tau = n/(N_eff+n)``, connected-correlation-preserving for pairs) to **both** data
and model frequencies.
Evidence: after the fix on 1atzA, ``||w||`` stabilises at ~53 (was 450) and the
two-point connected-correlation fit rose from 0.49 to 0.87. Gradient sign/scale
validated by the exact-enumeration recovery test (recovers J to <1e-2, corr>0.999).

## D-018: bm defaults to true PCD, not CCMpredPy's vanilla-CD-until-decay
Date: 2026-08-17
Context: CCMpredPy runs vanilla CD (chains reseeded from data each step) until the
learning rate decays below ``alpha0/10``, then switches to persistent chains. With
CCMpredPy's slow "sig" decay this switch happens ~1000 iterations in; within a few
hundred iterations the vanilla-CD fixed point is biased and the two-point fit
plateaus low (~0.70).
Decision: ``bm.fit(persistent=True)`` (default) runs **true PCD from the start** --
chains persist across all updates. ``persistent=False`` gives vanilla CD. Both paths
are available for parity experiments.
Evidence: on 1atzA, true PCD reaches a two-point connected-correlation fit to the
data of **0.865**, versus **0.605** for a converged (500-iter) CCMpredPy PCD run
sampled with the same sampler; jaxPotts bm couplings correlate with CCMpredPy's at
Pearson 0.95 (ZSG). The two reference tools (CCMpredPy PCD vs CCMpred plm) agree
only at coupling Pearson 0.90 / score Spearman 0.86, which sets the cross-method
expectation.

## D-019: CCMpredPy default ``-m`` matrix has no APC
Date: 2026-08-17
Context: Reproducing CCMpredPy's ``.mat`` for parity.
Decision: ``gauge.ccmpredpy_score`` defaults to ``apply_apc=False``. CCMpredPy's
``-m`` output is ``recenter_potentials`` (scalar-centre the 20x20 block) then the
Frobenius norm over the full 21x21 -- **without** APC (APC is written to a separate
``--apc`` file). CCMpred (C)'s ``-m``, by contrast, applies APC by default.
Evidence: ``ccmpredpy_score(braw, apply_apc=False)`` reproduces CCMpredPy's ``.mat``
to max|dS|=2e-7 (Pearson 1.0); with APC it drops to Pearson 0.475.
Source: ``CCMgen/ccmpred/scripts/run_ccmpred.py:276``
(``compute_contact_matrix(recenter_potentials=True, frob=True)``);
``ccmpred/__init__.py:514-550``.

## D-020: Reference tool environments and builds
Date: 2026-08-17
Context: Running both references for the notebook comparisons.
Decision: CCMpred (C) built CPU+OpenMP (``cmake . -DWITH_CUDA=off`` -> ``bin/ccmpred``).
CCMpredPy runs in a separate ``ccmpredpy`` micromamba env (Python 3.10, ``numpy<1.24``
for ``np.float``, ``scipy<1.11``); the ``plotly==3.0.0rc10`` setup pin is force-upgraded
to ``plotly==5.24.1`` (the pinned RC is import-broken on Python 3.10:
``collections.MutableSequence``); ``setuptools<81``. jaxPotts uses its own msgpack raw
reader, so it does not depend on CCMpredPy's ``encoding=``-broken ``parse_msgpack``.
Evidence: both tools run and write raw/braw files that jaxPotts parses (symmetric,
zero-diagonal verified).

## D-021: Third benchmark — hnisonoff/potts (PyTorch)
Date: 2026-08-17
Context: Repo owner requested benchmarking against
`github.com/hnisonoff/potts` (commit `1b325129cc11`), an independent PyTorch Potts
implementation, in addition to CCMpred/CCMpredPy.
Decision: Fit it by pseudo-likelihood (its `Potts` model + `pseudolikelihood`
method, LBFGS) in a separate `potts_torch` conda env and compare in §1d/§2. Three
conventions must be reconciled to compare couplings: (1) **alphabet** is
`-ACDEFGHIKLMNPQRSTVWY` (gap=0, alphabetical) vs ours (gap=20) — permute the A-axes;
(2) **energy sign** is opposite — their energy = +Σh+½ΣW = −log P, so their
`W = −J` in our convention — flip the sign; (3) their PL loss is the **N_eff-mean**,
not the sum, so their regularisation must be divided by N_eff to match CCMpred
(`lam = λ_single/N_eff`; their `(L-1)(A-1)` coupling scaling already reproduces
CCMpred's pair/single ratio). Their `to_zs_gauge` is the same double-centering ZSG.
Evidence: with matched regularisation, hnisonoff couplings correlate with jaxPotts
plm at **Pearson 0.9999** and with CCMpred at 0.987 (ZSG) — three independent plm
codebases agree once conventions are reconciled. Note: their default `lam=1e-7`
(from their example notebook) is calibrated for the mean-normalised loss and is
essentially unregularised; naively reusing CCMpred's `lam=0.01` over-regularises by
a factor of N_eff (couplings came out ~90x too small until this was corrected).
Env: torch 2.4.1 (`potts_torch`); the `torch` env (2.1.2) is unusable with numpy 2.x.
Only `torch`+`numpy` needed (import `potts.Potts`, `potts.gauge`; avoid
`potts.structure` which imports `esm`).

## D-022: Matmul precision — float32 default; lower precision is safe but unnecessary
Date: 2026-08-17
Context: §4 asks whether to lower the precision of the big plm einsum
(`jax.default_matmul_precision('tensorfloat32'|'bfloat16')`) and to measure the
effect on the final couplings, not silently lower it.
Decision: Keep the default (float32) precision. Measured on 1atzA (plm, LBFGS to
tol 1e-9), taking `highest` precision as the reference:

| precision      | iters | time (s) | Pearson vs highest | max\|dJ\| |
|----------------|-------|----------|--------------------|-----------|
| highest        | 334   | 17.8     | 1.000000           | 0         |
| float32        | 334   |  9.7     | 1.000000           | 0.0000    |
| tensorfloat32  | 286   | 15.7     | 0.999987           | 0.0063    |
| bfloat16       | 286   | 11.1     | 0.999987           | 0.0063    |

Lowering precision changes the couplings negligibly (Pearson 0.999987, max\|dJ\|
0.006 against a coupling std ~0.04) and can speed the einsum up, but at L=75 the
plm fit is already sub-20s and not precision-bound, so we do not lower it by
default. The knob is available via `jax.default_matmul_precision(...)` around
`plm.fit` for larger problems where the einsum dominates.
Evidence: `scratchpad/precision_test.py` run on an RTX A6000.

## D-023: Reviewed evorca (JAX plmDCA sibling); adopted a3m + compact J I/O
Date: 2026-08-17
Context: Repo owner asked whether `github.com/suzuki-2001/evorca` (commit
`2f99e23e9e25`, "fast and minimal plmDCA in JAX") does anything worth adopting or
comparing against.
Assessment: evorca is a symmetric plmDCA in JAX+Optax (shared `J (L,L,Q,Q)`,
re-symmetrised each step) — same objective family as jaxPotts `plm` and CCMpred.
Its scoring stack matches ours (zero-sum gauge -> gap-excluded Frobenius -> APC),
but its estimator differs: **AdamW minibatch-SGD for a fixed ~10 epochs (not run to
convergence)**, **flat unscaled L2 plus AdamW decoupled weight decay** (double
regularisation), **no pseudocount**, and **gap-ignoring Henikoff position-based
weights** (vs our Hamming-80% `1/n`). Its "sparse I/O for large L" shrinks only the
on-disk file — the training tensor is still a dense `(L,L,Q,Q)` with a `vmap` over
columns and **no L-chunking/scan**, so it OOMs at large L (our headroom, not theirs).
Decisions:
- **Adopt (done):** (1) A3M match-state parsing (`io.read_a3m`, drop lowercase and
  `.`) — a3m is the standard HHblits/AlphaFold MSA format; verified on evorca's
  P02358 example (15982x135). (2) Compact triangular coupling I/O
  (`io.save_couplings`/`load_couplings`, upper-triangle + compressed `.npz`),
  mirroring evorca's `sparse_J.npz` idea.
- **Defer:** RNA/nucleic-acid support (evorca's `--seq-type rna`, Q=5). A genuine
  capability extension (generalise the alphabet/q through io/energy/plm/bm) but out
  of current scope; recorded here as future work.
- **Not adopting:** their AdamW/flat-L2/fixed-epoch training and Henikoff weighting —
  jaxPotts deliberately matches CCMpred's converged, N_eff-consistent conventions.
- **Comparison:** not added to the notebook. A raw-coupling comparison would not be
  apples-to-apples (different reweighting, unscaled L2 + AdamW WD, non-converged SGD);
  only a contact-score/top-L comparison would be fair. Documented here instead per
  owner's choice.
Evidence: agent source audit — `evorca/model.py:16-25,44-63,90,102-122`,
`post.py:6-49`, `io_utils.py:19,30-56`, `alphabet.py:9-26`.
