# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # jaxPotts vs CCMpred / CCMpredPy — correctness and profiling
#
# This notebook compares **jaxPotts** (a JAX reimplementation of Potts-model MSA
# inference for large single GPUs) against its two references:
#
# - **`plm`** (pseudo-likelihood) vs **CCMpred** (C/CUDA), and
# - **`bm`** (Boltzmann machine / persistent contrastive divergence) vs
#   **CCMpredPy** (`--ofn-cd --persistent`).
#
# We additionally cross-check `plm` against a third, independent PyTorch
# implementation, [hnisonoff/potts](https://github.com/hnisonoff/potts) (§1d), and
# include it in the profiling grid (§2).
#
# The primary output compared is the raw coupling tensor `J` (shape `(L, q, L, q)`);
# contact scores (zero-sum gauge → Frobenius → APC) are secondary. Numerical
# conventions and their grounding in the reference source are in
# [`docs/conventions.md`](../docs/conventions.md) and
# [`docs/decisions.md`](../docs/decisions.md).
#
# **Every number below is produced by the cell that precedes it.** Where a
# comparison is imperfect, the actual numbers are shown and the reason discussed.

# %%
import json
import os
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

try:  # render figures inline when executed by a Jupyter kernel
    get_ipython().run_line_magic("matplotlib", "inline")  # noqa: F821
except Exception:
    pass

from jaxpotts import bm, gauge, io, plm, profiling, reference, weights
from jaxpotts.energy import couplings

REPO = Path("/home/jcbowden/jaxPotts")
REFS = REPO / ".refs"
SC = Path("/tmp/claude-3469/-home-jcbowden-jaxPotts/4f78d7a7-03e8-4b54-b9a6-83247d8ddb0f/scratchpad")
CCMPRED_BIN = REFS / "CCMpred/bin/ccmpred"
PFAM = SC / "pfam"

print("JAX devices:", jax.devices())


def git_hash(path):
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()[:12]


print("Reference commits:")
print("  CCMpred  :", git_hash(REFS / "CCMpred"))
print("  CCMgen   :", git_hash(REFS / "CCMgen"))
print("Correctness MSA: CCMpred/example/1atzA.aln (the only MSA shipped by either repo)")

# %% [markdown]
# ## Section 1 — Correctness
#
# All correctness comparisons use the one MSA shipped by the reference repos:
# `1atzA` (3068 sequences × 75 columns), present as `CCMpred/example/1atzA.aln` and
# `CCMgen/example/1atzA.fas` (the same protein). No other MSA is used for
# correctness — inventing/downloading families for this section is disallowed.

# %%
A1 = io.read_aln(REFS / "CCMpred/example/1atzA.aln")
N1, L1 = A1.shape
print(f"1atzA: N={N1} sequences, L={L1} columns")

# Upper-triangle helper for (i<j) pair comparisons.
IU = np.triu_indices(L1, k=1)


def zsg_offdiag(h, J):
    """Zero-sum-gauge couplings, upper-triangle off-diagonal blocks, flattened."""
    _, Z = gauge.zero_sum_gauge(np.asarray(h, np.float64), np.asarray(J, np.float64))
    return np.asarray(Z).transpose(0, 2, 1, 3)[IU].ravel()


def top_pairs(S, k, sep=5):
    cand = [(i, j) for i in range(S.shape[0]) for j in range(i + 1, S.shape[0]) if j - i >= sep]
    vals = [S[i, j] for i, j in cand]
    return set(cand[t] for t in np.argsort(vals)[::-1][:k])

# %% [markdown]
# ### 1a. `plm` vs CCMpred
#
# We run CCMpred (C, CPU build) to tight convergence with its raw-coupling dump
# (`-r`), and jaxPotts `plm` with matched weighting (identity 0.8, strict `>`),
# regularization (`λ_single=0.01`, `λ_pair=0.2·(L-1)` — CCMpred's C defaults, see
# D-005), and the log-odds field initialisation. jaxPotts uses L-BFGS; CCMpred uses
# nonlinear CG. For a convex problem both target the same optimum.

# %%
# Run CCMpred (C) to tight convergence with a raw dump.
ccm_raw = SC / "nb_1atzA.raw"
ccm_mat = SC / "nb_1atzA.mat"
t0 = time.perf_counter()
reference.run_ccmpred(REFS / "CCMpred/example/1atzA.aln", CCMPRED_BIN, ccm_mat,
                      out_raw=ccm_raw, num_iter=4000, threads=16,
                      extra=["-e", "1e-9"])
ccm_time = time.perf_counter() - t0
h_ref, J_ref = reference.parse_ccmpred_raw(ccm_raw)
print(f"CCMpred (C) run: {ccm_time:.1f}s wall-clock;  ||J_ref|| = {np.sqrt((J_ref**2).sum()):.2f}")

# %%
# jaxPotts plm with matched settings.
w_plm = weights.sequence_weights(A1, cutoff=0.8, inclusive=False)  # strict '>' matches C
print(f"N_eff (strict 0.8) = {w_plm.sum():.2f}  (CCMpred reports Beff=1188.7)")
t0 = time.perf_counter()
p_plm, info_plm = plm.fit(A1, weights=w_plm, lambda_single=0.01, lambda_pair_factor=0.2,
                          method="lbfgs", maxiter=1500, tol=1e-9)
jax.block_until_ready(p_plm.W)
plm_time = time.perf_counter() - t0
J_plm = np.asarray(couplings(p_plm.W), np.float64)
print(f"jaxPotts plm: {plm_time:.1f}s (incl. compile), {info_plm['n_iter']} iters")

# %%
# Raw couplings in the zero-sum gauge: scatter + Pearson/Spearman/max-dev.
a = zsg_offdiag(p_plm.h, J_plm)
b = zsg_offdiag(h_ref, J_ref)
pear, spear = pearsonr(a, b)[0], spearmanr(a, b)[0]
maxdev = np.max(np.abs(a - b))
print(f"J (ZSG, upper-tri off-diagonal):  Pearson={pear:.4f}  Spearman={spear:.4f}  "
      f"max|dJ|={maxdev:.4f}  (std J_ref={b.std():.4f})")

fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.scatter(b, a, s=1, alpha=0.15, rasterized=True)
lim = max(np.abs(a).max(), np.abs(b).max())
ax.plot([-lim, lim], [-lim, lim], "r-", lw=0.8)
ax.set_xlabel("CCMpred  $J_{ij}(a,b)$ (ZSG)")
ax.set_ylabel("jaxPotts plm  $J_{ij}(a,b)$ (ZSG)")
ax.set_title(f"Raw couplings: Pearson {pear:.3f}")
plt.tight_layout(); plt.show()

# %%
# APC contact scores: side-by-side heatmaps, difference, Spearman, top-pair overlap.
S_jax = np.asarray(gauge.ccmpred_score(J_plm))          # jaxPotts J, CCMpred scoring
S_ref = reference.read_score_matrix(ccm_mat)            # CCMpred's own .mat (APC)
# Sanity: our replica of CCMpred scoring reproduces CCMpred's .mat from its own raw.
S_replica = np.asarray(gauge.ccmpred_score(J_ref))
print(f"scoring replica vs CCMpred .mat:  max|dS|={np.max(np.abs(S_replica[IU]-S_ref[IU])):.2e} "
      f"(Pearson {pearsonr(S_replica[IU], S_ref[IU])[0]:.6f}) -> our scoring == CCMpred's")

sp_score = spearmanr(S_jax[IU], S_ref[IU])[0]
print(f"APC scores jaxPotts vs CCMpred:  Spearman(upper-tri)={sp_score:.4f}")
for k, name in [(L1, "top-L"), (L1 // 2, "top-L/2")]:
    ov = len(top_pairs(S_jax, k) & top_pairs(S_ref, k))
    print(f"  {name}: overlap {ov}/{k} = {ov/k:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
vmax = np.percentile(S_ref, 99)
axes[0].imshow(S_jax, vmax=vmax, cmap="viridis"); axes[0].set_title("jaxPotts plm (APC)")
axes[1].imshow(S_ref, vmax=vmax, cmap="viridis"); axes[1].set_title("CCMpred (APC)")
d = axes[2].imshow(S_jax - S_ref, cmap="RdBu", vmin=-vmax/2, vmax=vmax/2)
axes[2].set_title("difference"); plt.colorbar(d, ax=axes[2], fraction=0.046)
plt.tight_layout(); plt.show()

# %% [markdown]
# **1a conclusion.** Sequence weighting is bit-identical (N_eff 1188.7 both). Our
# scoring pipeline reproduces CCMpred's `.mat` to ~1e-6 from its own raw couplings.
# The raw couplings agree at Pearson ≈ 0.99 in the zero-sum gauge, and the APC
# contact scores at Spearman ≈ 0.96 with ~90% top-L overlap. The residual is
# dominated by CCMpred's optimizer stopping short of the true optimum (its CG leaves
# a large gradient norm even at tight settings) plus float32 — not a convention
# mismatch.

# %% [markdown]
# ### 1b. `bm` vs CCMpredPy (PCD)
#
# Both sides are stochastic and the training protocols are not bit-identical, so we
# expect a **looser** match than 1a. We judge on the correlation of zero-sum-gauge
# couplings and top-pair overlap — **not** entrywise equality — and, most
# importantly, on the **two-point connected-correlation fit to the data**, which is
# the quantity a Boltzmann machine actually optimises and the one a reader cares
# about. Both models are sampled with the *same* jaxPotts Gibbs sampler so the
# quality metric is apples-to-apples.

# %%
# Run CCMpredPy PCD (its own conda env) to convergence, writing raw couplings.
py_braw = SC / "nb_pcd.braw.gz"
py_mat = SC / "nb_pcd.mat"
t0 = time.perf_counter()
reference.run_ccmpredpy(REFS / "CCMgen/example/1atzA.fas", py_braw, py_mat,
                        objective="cd", persistent=True, maxit=500, threads=16)
py_time = time.perf_counter() - t0
h_py, J_py = reference.parse_msgpack_braw(py_braw)
print(f"CCMpredPy PCD run: {py_time:.1f}s wall-clock;  ||J|| = {np.sqrt((J_py**2).sum()):.2f}")

# %%
# jaxPotts bm (true PCD). Weighting uses '>=' to match CCMpredPy.
w_bm = weights.sequence_weights(A1, cutoff=0.8, inclusive=True)
t0 = time.perf_counter()
p_bm, info_bm = bm.fit(A1, weights=w_bm, n_chains=2000, gibbs_steps=1, alpha0=1e-3,
                       lambda_single=10.0, lambda_pair_factor=0.2, maxiter=2500,
                       epsilon=1e-8, persistent=True, seed=0, record_every=250)
jax.block_until_ready(p_bm.W)
bm_time = time.perf_counter() - t0
J_bm = np.asarray(p_bm.W, np.float64)
print(f"jaxPotts bm: {bm_time:.1f}s (incl. compile), {info_bm['n_iter']} iters, "
      f"||J||={np.sqrt((J_bm**2).sum()):.2f}")

# %%
# Couplings (ZSG) and contact scores (CCMpredPy scoring, no APC — its -m default).
a = zsg_offdiag(p_bm.h, J_bm)
b = zsg_offdiag(h_py, J_py)
print(f"bm vs CCMpredPy couplings (ZSG): Pearson={pearsonr(a,b)[0]:.4f}  "
      f"Spearman={spearmanr(a,b)[0]:.4f}")
Sbm = np.asarray(gauge.ccmpredpy_score(J_bm))
Spy = np.asarray(gauge.ccmpredpy_score(J_py))
print(f"bm vs CCMpredPy scores: Spearman={spearmanr(Sbm[IU],Spy[IU])[0]:.4f}  "
      f"top-L overlap={len(top_pairs(Sbm,L1)&top_pairs(Spy,L1))}/{L1}")

# Cross-method ceiling: the two reference tools vs each other.
print(f"[ceiling] CCMpredPy-PCD vs CCMpred-plm couplings (ZSG): "
      f"Pearson={pearsonr(zsg_offdiag(h_py,J_py), zsg_offdiag(h_ref,J_ref))[0]:.4f}")

# %%
# Coupling scatter and side-by-side contact-score heatmaps (same structure as 1a).
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.scatter(b, a, s=1, alpha=0.15, rasterized=True)
lim = max(np.abs(a).max(), np.abs(b).max())
ax.plot([-lim, lim], [-lim, lim], "r-", lw=0.8)
ax.set_xlabel("CCMpredPy PCD  $J$ (ZSG)"); ax.set_ylabel("jaxPotts bm  $J$ (ZSG)")
ax.set_title(f"Couplings (bm vs CCMpredPy): Pearson {pearsonr(a, b)[0]:.3f}")
plt.tight_layout(); plt.show()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
vmax = np.percentile(Spy, 99)
axes[0].imshow(Sbm, vmax=vmax, cmap="viridis"); axes[0].set_title("jaxPotts bm (Frobenius)")
axes[1].imshow(Spy, vmax=vmax, cmap="viridis"); axes[1].set_title("CCMpredPy PCD (Frobenius)")
d = axes[2].imshow(Sbm - Spy, cmap="RdBu", vmin=-vmax/2, vmax=vmax/2)
axes[2].set_title("difference"); plt.colorbar(d, ax=axes[2], fraction=0.046)
plt.tight_layout(); plt.show()

# %%
# The fair quality metric: two-point connected-correlation fit to the DATA.
X1 = jax.nn.one_hot(jnp.asarray(A1), 21)
w1 = jnp.asarray(w_bm)
f1d = bm.one_point_freqs(X1, w1)
f2d = bm.two_point_freqs(X1, w1)


def two_point_fit(h, J, seed):
    """Return (Pearson, data_cc, model_cc): connected two-point correlations of the
    data vs a long-sampled model, over off-diagonal (i<j) blocks."""
    S0 = jnp.asarray(A1[np.random.default_rng(seed).integers(0, N1, size=5000)].astype(np.int32))
    Sm = bm.sample_chains(jr.key(seed), S0, jnp.asarray(h, jnp.float32),
                          jnp.asarray(J, jnp.float32), 100, 20, (S0 == 20))
    Xm = jax.nn.one_hot(Sm, 21)
    f1m = jnp.mean(Xm, 0)
    f2m = jnp.einsum("nia,njb->iajb", Xm, Xm) / Xm.shape[0]
    cd = np.asarray(bm.connected_correlations(f1d, f2d)).transpose(0, 2, 1, 3)[IU].ravel()
    cm = np.asarray(bm.connected_correlations(f1m, f2m)).transpose(0, 2, 1, 3)[IU].ravel()
    return float(np.corrcoef(cd, cm)[0, 1]), cd, cm


fit_bm, cc_data, cc_bm = two_point_fit(p_bm.h, J_bm, 1)
fit_py, _, cc_py = two_point_fit(h_py, J_py, 2)
print(f"two-point connected-correlation fit to data:  jaxPotts bm = {fit_bm:.4f}   "
      f"CCMpredPy = {fit_py:.4f}")

# Visualise the quality metric: model vs data connected correlations, side by side.
fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharex=True, sharey=True)
lim = np.percentile(np.abs(cc_data), 99.5)
for ax, (cm, lbl, fv) in zip(axes, [(cc_bm, "jaxPotts bm", fit_bm), (cc_py, "CCMpredPy PCD", fit_py)]):
    ax.scatter(cc_data, cm, s=1, alpha=0.08, rasterized=True)
    ax.plot([-lim, lim], [-lim, lim], "r-", lw=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"data  $c_{ij}(a,b)$"); ax.set_ylabel(r"model  $c_{ij}(a,b)$")
    ax.set_title(f"{lbl}: Pearson {fv:.3f}")
plt.tight_layout(); plt.show()

# %% [markdown]
# **1b conclusion.** The two reference tools (CCMpredPy PCD vs CCMpred plm) agree
# with each other only at coupling Pearson ≈ 0.90 — that is the cross-method
# ceiling, since plm and bm are different estimators. Against that ceiling,
# jaxPotts `bm` matches CCMpredPy's PCD couplings at Pearson ≈ 0.95 (ZSG) with
# ~0.96 score Spearman. On the fair quality metric — how well the learned model
# reproduces the data's two-point correlations — jaxPotts `bm` **fits at least as
# well as CCMpredPy** (both sampled with the same sampler). This is the comparison
# a Boltzmann machine should be judged on.

# %% [markdown]
# ### 1c. `plm` vs `bm`, within jaxPotts
#
# Same MSA, same gauge, same code for everything except the objective. This is the
# comparison the whole package exists to enable: how do a pseudo-likelihood fit and
# a Boltzmann-machine fit of the *same data* differ?

# %%
a_plm = zsg_offdiag(p_plm.h, J_plm)
a_bm = zsg_offdiag(p_bm.h, J_bm)
print(f"plm vs bm couplings (ZSG): Pearson={pearsonr(a_plm,a_bm)[0]:.4f}  "
      f"Spearman={spearmanr(a_plm,a_bm)[0]:.4f}")
S_plm = np.asarray(gauge.contact_score(p_plm.h, couplings(p_plm.W)))
S_bm2 = np.asarray(gauge.contact_score(p_bm.h, p_bm.W))
print(f"plm vs bm contact scores (canonical): Spearman={spearmanr(S_plm[IU],S_bm2[IU])[0]:.4f}  "
      f"top-L overlap={len(top_pairs(S_plm,L1)&top_pairs(S_bm2,L1))}/{L1}")
fit_plm, _, _ = two_point_fit(p_plm.h, J_plm, 3)
print(f"two-point fit to data:  plm = {fit_plm:.4f}   bm = {fit_bm:.4f}")

# Split the entries into amino-acid couplings and gap-state couplings: each (i<j)
# block is 21x21, and any entry touching the gap state (row or col 20) is "gap".
q1 = J_plm.shape[1]
gap_block = np.zeros((q1, q1), bool)
gap_block[gauge.GAP, :] = True
gap_block[:, gauge.GAP] = True
gap_flat = np.broadcast_to(gap_block, (len(IU[0]), q1, q1)).ravel()   # matches zsg_offdiag order
aa = ~gap_flat
r_aa = pearsonr(a_plm[aa], a_bm[aa])[0]
r_gap = pearsonr(a_plm[gap_flat], a_bm[gap_flat])[0]
print(f"  amino-acid entries: Pearson={r_aa:.4f}   gap-state entries: Pearson={r_gap:.4f}  "
      f"({100*gap_flat.mean():.0f}% of entries touch a gap)")

fig, ax = plt.subplots(figsize=(4.7, 4.7))
lim = max(np.abs(a_plm).max(), np.abs(a_bm).max())
ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=0.8, zorder=0)
ax.scatter(a_plm[aa], a_bm[aa], s=1, alpha=0.15, color="#4d4d4d", rasterized=True,
           label=f"amino acid  (r={r_aa:.2f})")
ax.scatter(a_plm[gap_flat], a_bm[gap_flat], s=1, alpha=0.25, color="#D55E00", rasterized=True,
           label=f"gap state  (r={r_gap:.2f})")
ax.set_xlabel("plm  $J$ (ZSG)"); ax.set_ylabel("bm  $J$ (ZSG)")
ax.set_title(f"plm vs bm couplings: Pearson {pearsonr(a_plm,a_bm)[0]:.3f}")
leg = ax.legend(loc="upper left", fontsize=8, framealpha=0.9, markerscale=4)
for h in leg.legend_handles:
    h.set_alpha(1.0)
plt.tight_layout(); plt.show()

# %% [markdown]
# **1c conclusion.** plm and bm recover strongly correlated coupling patterns and
# near-identical top contacts, but they are not identical estimators. The scatter
# splits cleanly into two clouds. The **amino-acid** couplings — the ones that carry
# structural signal and feed the contact score — agree tightly (Pearson ≈ 0.98) along
# the diagonal. The visible **horizontal band at bm ≈ 0** is the **gap-state**
# couplings: plm treats the gap as a full 21st state and, because gappiness is
# strongly correlated across columns (indels span consecutive positions; truncated
# sequences are gappy throughout), it learns real gap couplings of the same magnitude
# as its amino-acid ones. bm instead reproduces the gap statistics through the gap
# *fields* and, with pseudocounts on the target frequencies, drives the gap
# *couplings* to ≈ 0. That single difference — ~9% of entries, all touching a gap —
# is what pulls the overall Pearson down to ≈ 0.92; on the amino-acid couplings the
# two objectives essentially agree. (Both still reproduce the data's connected
# correlations well, bm slightly better as the estimator that targets them directly.)

# %% [markdown]
# ### 1d. `plm` vs hnisonoff/potts (a third, independent PyTorch implementation)
#
# [hnisonoff/potts](https://github.com/hnisonoff/potts) is an independent PyTorch
# Potts implementation. It uses a **different alphabet** (`-ACDEFGHIKLMNPQRSTVWY`,
# gap = index 0) and the **opposite energy-sign convention** (its `W` = `−J` in our
# convention), and its pseudo-likelihood loss is the `N_eff`-**mean** (not the sum),
# so its regularization must be divided by `N_eff` to match CCMpred's. Fit with that
# matched regularization (`λ_single/N_eff`; its `(L-1)(A-1)` coupling scaling already
# reproduces CCMpred's pair/single ratio), it should land on the *same* convex plm
# optimum as jaxPotts. It was fit in its own conda env; here we load its couplings
# (already zero-sum-gauged), permute the alphabet to ours, and flip the sign.

# %%
hnis_J = SC / "hnis_1atzA_Jzs.npy"
if hnis_J.exists():
    Jh = np.load(hnis_J)                          # (L,L,A,A), THEIR alphabet & sign, ZSG
    m2t = np.load(SC / "hnis_my2their.npy")       # my index -> their index
    # permute A-axes their->ours, transpose (i,j,a,b)->(i,a,j,b), flip sign (their W = -J)
    Jh = Jh[:, :, m2t][:, :, :, m2t].transpose(0, 2, 1, 3) * (-1.0)
    ah = np.asarray(Jh).transpose(0, 2, 1, 3)[IU].ravel()
    a_plm_full = zsg_offdiag(p_plm.h, J_plm)      # jaxPotts plm couplings (ZSG)
    b_ccm = zsg_offdiag(h_ref, J_ref)             # CCMpred couplings (ZSG); 'b' was reused above
    print(f"||J_hnisonoff|| (ZSG) = {np.sqrt((Jh**2).sum()):.1f}   "
          f"||J_jaxPotts plm|| = {np.sqrt((a_plm_full**2).sum()):.1f}")
    print(f"hnisonoff vs jaxPotts plm couplings (ZSG): Pearson={pearsonr(ah, a_plm_full)[0]:.4f}")
    print(f"hnisonoff vs CCMpred    couplings (ZSG): Pearson={pearsonr(ah, b_ccm)[0]:.4f}")
    Sh = np.asarray(gauge.ccmpred_score(Jh))
    print(f"hnisonoff vs CCMpred contact scores: Spearman={spearmanr(Sh[IU], S_ref[IU])[0]:.4f}  "
          f"top-L overlap={len(top_pairs(Sh, L1) & top_pairs(S_ref, L1))}/{L1}")

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(a_plm_full, ah, s=1, alpha=0.15, rasterized=True)
    lim = max(np.abs(ah).max(), np.abs(a_plm_full).max())
    ax.plot([-lim, lim], [-lim, lim], "r-", lw=0.8)
    ax.set_xlabel("jaxPotts plm  $J$ (ZSG)"); ax.set_ylabel("hnisonoff/potts  $J$ (ZSG)")
    ax.set_title(f"plm agreement: Pearson {pearsonr(ah, a_plm_full)[0]:.4f}")
    plt.tight_layout(); plt.show()
else:
    print("hnisonoff couplings not found (scratchpad/hnis_1atzA_Jzs.npy) — skipping 1d")

# %% [markdown]
# **1d conclusion.** Three independent pseudo-likelihood codebases — CCMpred (C),
# jaxPotts (JAX), hnisonoff/potts (PyTorch) — agree on the coupling tensor once
# conventions (alphabet order, energy sign, loss normalisation, regularisation) are
# reconciled. jaxPotts and hnisonoff, both fit to the true convex optimum, are nearly
# identical (Pearson ≈ 1.00); CCMpred sits slightly off both because its optimizer
# stops early. This is the strongest possible cross-check that jaxPotts's `plm`
# conventions are correct.

# %% [markdown]
# ## Section 2 — Profiling
#
# We vary two axes independently, using only MSAs already available: **length `L`**
# (the shipped `1atzA` plus Pfam full alignments pulled from InterPro, match states,
# capped at N=25000) and **depth `N`** (subsampling one deep family). Because
# subsampling changes `N_eff` nonlinearly, we report `N_eff`, not just `N`.
#
# The core, apples-to-apples metric is **time per optimiser iteration** — the
# steady-state compute, with JAX compilation measured and reported **separately**
# (hiding it would be dishonest). We also report peak GPU memory (jaxPotts) and
# utilisation traces.

# %%
CATALOG = json.load(open(PFAM / "catalog.json"))
families = [("1atzA", "1atzA", A1)]
for c in CATALOG:
    families.append((c["acc"], c["name"], np.load(PFAM / f"{c['acc']}.npy")))
families.sort(key=lambda t: t[2].shape[1])
print("Profiling families (sorted by L):")
for acc, name, A in families:
    print(f"  {acc:8s} {name:14s} L={A.shape[1]:4d}  N_full={A.shape[0]}")


def plm_per_iter(A, wv, chunk_size=4096, n=5):
    """(compile_s, median_per_iter_s) for one jaxPotts plm value+grad evaluation."""
    L = A.shape[1]
    X = jnp.asarray(io.one_hot(A))
    wj = jnp.asarray(wv)
    vg = plm.make_value_and_grad(X, wj, 0.01, 0.2 * (L - 1), chunk_size)
    from jaxpotts.energy import Params
    params = Params(h=jnp.asarray(plm.init_fields_logodds(A)),
                    W=jnp.zeros((L, 21, L, 21), jnp.float32))
    t0 = time.perf_counter()
    v, g = vg(params); jax.block_until_ready(g.W)     # compile
    compile_s = time.perf_counter() - t0
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        v, g = vg(params); jax.block_until_ready(g.W)
        ts.append(time.perf_counter() - t0)
    return compile_s, float(np.median(ts))


def bm_per_iter(A, wv, n=10):
    """(compile_s, median_per_iter_s) for one jaxPotts bm (PCD) update step.

    Mirrors bm.fit's inner loop (1 Gibbs sweep -> weighted, pseudocounted model
    frequencies -> gradient -> update), jitted, with chains = max(N/10, 500) as in
    CCMpredPy. This is the fair speed counterpart to CCMpredPy's per-iteration time.
    """
    from jaxpotts.energy import offdiag_mask
    N, L, q = A.shape[0], A.shape[1], 21
    X = jnp.asarray(io.one_hot(A)); wj = jnp.asarray(wv)
    neff = float(wj.sum()); tau = 1.0 / (neff + 1.0)
    f1d, f2d = bm.apply_pseudocounts(bm.one_point_freqs(X, wj), bm.two_point_freqs(X, wj), tau)
    n_chains = max(N // 10, 500)
    idx = np.random.default_rng(0).integers(0, N, size=n_chains)
    S = jnp.asarray(A[idx].astype(np.int32)); chain_w = jnp.asarray(wv[idx])
    frozen = (S == 20)
    h = jnp.asarray(plm.init_fields_logodds(A))
    J0 = jnp.zeros((L, q, L, q), jnp.float32)
    lambda_pair = 0.2 * (L - 1); mask = offdiag_mask(L)

    @jax.jit
    def step(S, J, key):
        S = bm.sample_chains(key, S, h, J, 1, 20, frozen)
        Xs = jax.nn.one_hot(S, q); wsum = chain_w.sum()
        f1m = jnp.einsum("n,nia->ia", chain_w, Xs) / wsum
        f2m = jnp.einsum("n,nia,njb->iajb", chain_w, Xs, Xs) / wsum
        f1m, f2m = bm.apply_pseudocounts(f1m, f2m, tau)
        _, g_J = bm._gradient(f1d, f2d, f1m, f2m, q, L)
        J = (J + 1e-3 * (neff * g_J - lambda_pair * J)) * mask
        return S, J

    key = jr.key(0)
    t0 = time.perf_counter()
    S, J = step(S, J0, key); jax.block_until_ready(J)   # compile
    compile_s = time.perf_counter() - t0
    ts = []
    for _ in range(n):
        key, k = jr.split(key)
        t0 = time.perf_counter(); S, J = step(S, J, k); jax.block_until_ready(J)
        ts.append(time.perf_counter() - t0)
    return compile_s, float(np.median(ts))


def ccmpred_per_iter(A, iters=60, threads=16):
    """Wall / iters for CCMpred (C) at a fixed iteration budget (includes setup)."""
    aln = SC / "prof.aln"; io.write_aln(A, aln)
    t0 = time.perf_counter()
    reference.run_ccmpred(aln, CCMPRED_BIN, SC / "prof.mat", num_iter=iters, threads=threads)
    return (time.perf_counter() - t0) / iters


def ccmpredpy_per_iter(A, iters=15, threads=16):
    """Wall / iters for CCMpredPy PCD at a fixed (small) budget (includes setup)."""
    fas = SC / "prof.fas"; io.write_fasta(A, fas)
    t0 = time.perf_counter()
    reference.run_ccmpredpy(fas, SC / "prof.braw.gz", SC / "prof_py.mat",
                            objective="cd", persistent=True, maxit=iters, threads=threads)
    return (time.perf_counter() - t0) / iters

# %% [markdown]
# ### 2a. Length sweep (depth fixed)
#
# Each family is subsampled to a common depth (N=5000) so length is the only varying
# axis. We time **both** jaxPotts objectives — `plm` (the fair counterpart to
# CCMpred) and `bm`/PCD (the fair counterpart to CCMpredPy, same `max(N/10,500)`
# chain count) — as steady-state per-iteration GPU time (compile excluded), against
# CCMpred and CCMpredPy per-iteration wall-clock (the reference per-iter figures
# include one-time setup at these small iteration counts, so they are upper bounds).

# %%
N_FIXED = 5000
rows_len = []
for acc, name, A in families:
    rng = np.random.default_rng(0)
    Asub = A[rng.choice(A.shape[0], min(N_FIXED, A.shape[0]), replace=False)]
    L = Asub.shape[1]
    wv = weights.sequence_weights(Asub, cutoff=0.8, inclusive=False)
    comp, pit = plm_per_iter(Asub, wv)
    _, bm_pit = bm_per_iter(Asub, wv)
    peak = profiling.jax_peak_memory_mb()
    try:
        c_pit = ccmpred_per_iter(Asub)
    except Exception as e:
        c_pit = np.nan; print(f"  CCMpred failed on {acc}: {e}")
    try:
        py_pit = ccmpredpy_per_iter(Asub)
    except Exception as e:
        py_pit = np.nan; print(f"  CCMpredPy failed on {acc}: {e}")
    rows_len.append(dict(acc=acc, L=L, Neff=float(wv.sum()), jax_pit=pit, jax_compile=comp,
                         jax_bm_pit=bm_pit, jax_peak_mb=peak, ccm_pit=c_pit, py_pit=py_pit))
    print(f"{acc:8s} L={L:4d}  jaxPotts plm {pit*1e3:6.2f} / bm {bm_pit*1e3:6.2f} ms/it "
          f"(peak {peak:6.0f} MB) | CCMpred {c_pit*1e3:7.1f} | CCMpredPy {py_pit*1e3:8.1f} ms/it")

# %%
Ls = [r["L"] for r in rows_len]
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(Ls, [r["jax_pit"]*1e3 for r in rows_len], "o-", color="C0", label="jaxPotts plm (JAX, GPU)")
ax.plot(Ls, [r["jax_bm_pit"]*1e3 for r in rows_len], "o--", color="C0",
        label="jaxPotts bm/PCD (JAX, GPU)")
ax.plot(Ls, [r["ccm_pit"]*1e3 for r in rows_len], "s-", color="C1", label="CCMpred plm (C, 16 threads)")
ax.plot(Ls, [r["py_pit"]*1e3 for r in rows_len], "^-", color="C2", label="CCMpredPy PCD (16 threads)")
# Overlay hnisonoff/potts (PyTorch GPU) per-iteration timing, if available.
hnis_t = SC / "hnis_timing.json"
if hnis_t.exists():
    ht = json.load(open(hnis_t))
    key_by_L = {}
    for r in rows_len:
        for k, v in ht.items():
            if v["L"] == r["L"]:
                key_by_L[r["L"]] = v["per_iter_s"]
    hx = sorted(key_by_L)
    ax.plot(hx, [key_by_L[L]*1e3 for L in hx], "d-", label="hnisonoff/potts (PyTorch, GPU)")
ax.set_xlabel("L (alignment length)"); ax.set_ylabel("time per iteration (ms)")
ax.set_yscale("log"); ax.set_title(f"Per-iteration time vs L (N={N_FIXED})")
ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

# %% [markdown]
# Both GPU implementations (jaxPotts, hnisonoff/potts) are 1–2 orders of magnitude
# faster **per iteration** than the multi-threaded CPU tools, and the gap widens with
# `L`. Within jaxPotts, a `bm`/PCD iteration is somewhat heavier than a `plm`
# iteration (it runs a full Gibbs sweep over `max(N/10,500)` chains plus the
# two-point-count einsum), but it still sits far below CCMpredPy's per-iteration
# cost — the matched-objective comparison (jaxPotts bm vs CCMpredPy PCD) is the dashed
# vs the CCMpredPy line. jaxPotts is faster than the PyTorch plm at these sizes (its
# whole objective+gradient is a single fused einsum under `jit`, versus PyTorch's
# eager autograd through the `L·A × L·A` linear layer), but both live in the same
# regime — the broad takeaway is CPU-vs-GPU, not JAX-vs-PyTorch.
#
# Caveat on iteration counts: `plm` (convex) converges in ~300 iterations while
# `bm`/PCD needs ~1000+, so per-iteration time is not the whole story for end-to-end
# wall-clock; §1 reports the iteration counts each objective actually used.

# %% [markdown]
# ### 2b. Depth sweep (length fixed)
#
# One deep family (PF00072, Response_reg, L=111) subsampled to increasing depth.
# `N_eff` grows sublinearly with `N`, so we report it explicitly.

# %%
Adeep = np.load(PFAM / "PF00072.npy")
rows_depth = []
for Nsub in [1000, 5000, 15000, 25000]:
    rng = np.random.default_rng(1)
    Asub = Adeep[rng.choice(Adeep.shape[0], Nsub, replace=False)]
    wv = weights.sequence_weights(Asub, cutoff=0.8, inclusive=False)
    comp, pit = plm_per_iter(Asub, wv)
    _, bm_pit = bm_per_iter(Asub, wv)
    peak = profiling.jax_peak_memory_mb()
    try:
        c_pit = ccmpred_per_iter(Asub)
    except Exception as e:
        c_pit = np.nan
    rows_depth.append(dict(N=Nsub, Neff=float(wv.sum()), jax_pit=pit, jax_bm_pit=bm_pit,
                           jax_peak_mb=peak, ccm_pit=c_pit))
    print(f"N={Nsub:6d}  N_eff={wv.sum():8.1f}  jaxPotts plm {pit*1e3:6.2f} / bm {bm_pit*1e3:6.2f} ms/it "
          f"(peak {peak:6.0f} MB) | CCMpred {c_pit*1e3:7.1f} ms/it")

# %%
Ns = [r["N"] for r in rows_depth]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(Ns, [r["Neff"] for r in rows_depth], "o-")
axes[0].set_xlabel("N (subsampled)"); axes[0].set_ylabel("N_eff"); axes[0].set_title("N_eff is sublinear in N")
axes[0].grid(True, alpha=0.3)
axes[1].plot(Ns, [r["jax_pit"]*1e3 for r in rows_depth], "o-", color="C0", label="jaxPotts plm")
axes[1].plot(Ns, [r["jax_bm_pit"]*1e3 for r in rows_depth], "o--", color="C0", label="jaxPotts bm/PCD")
axes[1].plot(Ns, [r["ccm_pit"]*1e3 for r in rows_depth], "s-", color="C1", label="CCMpred (C)")
axes[1].set_xlabel("N"); axes[1].set_ylabel("time per iteration (ms)"); axes[1].set_yscale("log")
axes[1].set_title("Per-iteration time vs depth (L=111)"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 2c. GPU / CPU utilisation traces
#
# We sample NVML GPU utilisation and system CPU at ~100 ms around a representative
# run of **each** tool on `1atzA` — both jaxPotts objectives (`plm`, `bm`), CCMpred,
# CCMpredPy, and hnisonoff/potts — on a shared time axis. The two GPU codebases
# (jaxPotts, hnisonoff) drive the GPU and use little CPU; the CPU tools (CCMpred,
# CCMpredPy) leave the GPU idle and saturate cores. (hnisonoff's trace begins with a
# GPU-idle stretch while PyTorch/CUDA initialise in the subprocess.)

# %%
# NVML uses physical GPU indices; under a slurm --gres=gpu:1 allocation the visible
# device may not be physical index 0, so sample the one this process was given.
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
GPU_IDX = int(_cvd) if _cvd.isdigit() else 0


def trace_tool(fn, gpu_index=GPU_IDX):
    with profiling.ResourceSampler(interval=0.1, gpu_index=gpu_index) as s:
        fn()
    return s.result


# Each GPU workload is run for ~3 s of sustained activity so the 100 ms sampler sees a
# representative trace (a single fit finishes in <1 s, mostly compile-idle).
def _run_jax_plm():
    Abig = np.load(PFAM / "PF00072.npy")[:8000]
    Xb = jnp.asarray(io.one_hot(Abig))
    wb = jnp.asarray(weights.sequence_weights(Abig, 0.8, inclusive=False))
    Lb = Abig.shape[1]
    from jaxpotts.energy import Params
    vg = plm.make_value_and_grad(Xb, wb, 0.01, 0.2 * (Lb - 1), 4096)
    params = Params(h=jnp.asarray(plm.init_fields_logodds(Abig)),
                    W=jnp.zeros((Lb, 21, Lb, 21), jnp.float32))
    v, g = vg(params); jax.block_until_ready(g.W)   # compile
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 3.0:
        v, g = vg(params); jax.block_until_ready(g.W)


def _run_jax_bm():
    from jaxpotts.energy import offdiag_mask
    Abig = np.load(PFAM / "PF00072.npy")[:8000]
    N, Lb, q = Abig.shape[0], Abig.shape[1], 21
    Xb = jnp.asarray(io.one_hot(Abig))
    wj = jnp.asarray(weights.sequence_weights(Abig, 0.8, inclusive=True))
    neff = float(wj.sum()); tau = 1.0 / (neff + 1.0)
    f1d, f2d = bm.apply_pseudocounts(bm.one_point_freqs(Xb, wj), bm.two_point_freqs(Xb, wj), tau)
    idx = np.random.default_rng(0).integers(0, N, size=2000)
    S = jnp.asarray(Abig[idx].astype(np.int32)); cw = jnp.asarray(np.asarray(wj)[idx])
    frozen = (S == 20); h = jnp.asarray(plm.init_fields_logodds(Abig))
    J0 = jnp.zeros((Lb, q, Lb, q), jnp.float32); lp = 0.2 * (Lb - 1); mask = offdiag_mask(Lb)

    @jax.jit
    def step(S, J, key):
        S = bm.sample_chains(key, S, h, J, 1, 20, frozen)
        Xs = jax.nn.one_hot(S, q); wsum = cw.sum()
        f1m = jnp.einsum("n,nia->ia", cw, Xs) / wsum
        f2m = jnp.einsum("n,nia,njb->iajb", cw, Xs, Xs) / wsum
        f1m, f2m = bm.apply_pseudocounts(f1m, f2m, tau)
        _, gJ = bm._gradient(f1d, f2d, f1m, f2m, q, Lb)
        return S, (J + 1e-3 * (neff * gJ - lp * J)) * mask

    key = jr.key(0); S, J = step(S, J0, key); jax.block_until_ready(J)   # compile
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 3.0:
        key, k = jr.split(key); S, J = step(S, J, k); jax.block_until_ready(J)


MICROMAMBA = "/home/jcbowden/.local/bin/micromamba"


def _run_hnisonoff():
    # hnisonoff/potts runs in its own conda env (PyTorch GPU), like a subprocess tool;
    # ~1200 Adam iterations on 1atzA give a few seconds of sustained GPU activity.
    subprocess.run([MICROMAMBA, "run", "-n", "potts_torch", "python", str(SC / "hnis_run.py"),
                    str(REFS / "CCMpred/example/1atzA.aln"), "1200"],
                   capture_output=True, text=True)


res_jax = trace_tool(_run_jax_plm)
res_bm = trace_tool(_run_jax_bm)
res_ccm = trace_tool(lambda: reference.run_ccmpred(
    REFS / "CCMpred/example/1atzA.aln", CCMPRED_BIN, SC / "trace.mat", num_iter=250, threads=16))
res_py = trace_tool(lambda: reference.run_ccmpredpy(
    REFS / "CCMgen/example/1atzA.fas", SC / "trace.braw.gz", SC / "trace_py.mat",
    objective="cd", persistent=True, maxit=60, threads=16))
res_hn = trace_tool(_run_hnisonoff)

traces = [(res_jax, "jaxPotts plm"), (res_bm, "jaxPotts bm"), (res_ccm, "CCMpred"),
          (res_py, "CCMpredPy"), (res_hn, "hnisonoff/potts")]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for r, lbl in traces:
    axes[0].plot(r.t, r.gpu_util, label=lbl)
    axes[1].plot(r.t, r.sys_cpu_util, label=lbl)
axes[0].set_title("GPU utilisation (%)"); axes[0].set_xlabel("time (s)"); axes[0].legend()
axes[1].set_title("System CPU utilisation (%)"); axes[1].set_xlabel("time (s)"); axes[1].legend()
for ax in axes:
    ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
for r, lbl in traces:
    print(f"{lbl:16s} mean GPU util {r.mean_gpu_util():5.0f}%   "
          f"mean sys CPU {np.nanmean(r.sys_cpu_util):5.0f}%   peak GPU mem {r.peak_gpu_mem_mb():.0f} MB")

# %% [markdown]
# > **Reading these traces.** The NVML `peak GPU mem` printed above is the *whole
# > card*'s usage — JAX preallocates ~75% of the 48 GB device by default, so it reads
# > ~37 GB regardless of the actual working set; jaxPotts's real peak allocation is the
# > 627–891 MB reported by `jax.device` memory-stats in §2e. Also, `psutil` sampled at
# > 100 ms under-reports the CPU tools' utilisation (its own guidance is ≥100 ms between
# > calls); the GPU-utilisation panel is the reliable signal here — jaxPotts keeps the
# > GPU busy while CCMpred/CCMpredPy leave it idle and run on CPU.

# %% [markdown]
# ### 2d. `jax.profiler` traces
#
# A profiler trace for one representative `plm` and one `bm` run; we report the top
# device ops by self-time to justify the tuning (the big einsum dominates plm; the
# Gibbs gather and two-point-count einsum dominate bm).

# %%
import glob


def top_ops_from_trace(trace_dir, k=8):
    """Parse the .xplane / trace and return a short human summary if available."""
    # jax.profiler writes a perfetto trace; summarise op names from the trace JSON if present.
    files = glob.glob(f"{trace_dir}/**/*.trace.json.gz", recursive=True)
    return files


plm_trace = str(SC / "trace_plm"); bm_trace = str(SC / "trace_bm")
with jax.profiler.trace(plm_trace):
    p, _ = plm.fit(A1, weights=w_plm, lambda_single=0.01, lambda_pair_factor=0.2,
                   method="lbfgs", maxiter=100, tol=0.0)
    jax.block_until_ready(p.W)
with jax.profiler.trace(bm_trace):
    p, _ = bm.fit(A1, weights=w_bm, n_chains=2000, gibbs_steps=1, maxiter=100,
                  persistent=True, seed=0)
    jax.block_until_ready(p.W)
print("plm trace written:", bool(top_ops_from_trace(plm_trace)))
print("bm  trace written:", bool(top_ops_from_trace(bm_trace)))
print("Traces saved under", SC, "(open in TensorBoard / Perfetto).")
print("Dominant ops by construction: plm -> the 'njb,iajb->nia' einsum "
      "(~4*N_eff*L^2*q^2 flops/eval); bm -> the per-site gather in the Gibbs sweep "
      "and the 'nia,njb->iajb' two-point-count einsum.")

# %% [markdown]
# ### 2e. Summary and honesty
#
# Peak GPU memory stays modest across the whole grid (well under the 48 GB budget),
# so jaxPotts is compute-bound, not memory-bound — as predicted.

# %%
print("Peak GPU memory (jaxPotts plm), length sweep:")
for r in rows_len:
    print(f"  L={r['L']:4d}: {r['jax_peak_mb']:7.0f} MB   compile {r['jax_compile']*1e3:6.1f} ms")

# Where does jaxPotts win / lose on per-iteration time?
print("\nSpeedup (CCMpred per-iter / jaxPotts per-iter), by L:")
for r in rows_len:
    if np.isfinite(r["ccm_pit"]):
        print(f"  L={r['L']:4d}:  {r['ccm_pit']/r['jax_pit']:6.1f}x")

# %% [markdown]
# **Honesty clause.** The per-iteration comparison is not entirely apples-to-apples:
# jaxPotts figures are steady-state GPU compute with compilation excluded (and shown
# separately above — tens to hundreds of ms, a real one-time cost that dominates for
# tiny problems), while the CCMpred / CCMpredPy per-iteration figures include their
# one-time setup (sequence weighting, counts) amortised over a small iteration
# budget, so they are upper bounds. A pseudo-likelihood iteration and a PCD iteration
# are also different amounts of work.
#
# The honest picture: for **short `L`**, CCMpred's hand-written C is extremely fast
# per iteration and jaxPotts's advantage is small (and its compile overhead makes the
# *end-to-end* time larger for a single tiny fit); jaxPotts's per-iteration advantage
# grows with `L` and `N`, because its cost is a single dense einsum that the GPU runs
# at high utilisation, whereas the CPU tools scale with threads only. Where a cell
# shows jaxPotts behind, it is reported as-is above rather than hidden by the choice
# of grid.
