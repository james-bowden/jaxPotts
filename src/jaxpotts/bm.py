"""Boltzmann machine learning by persistent contrastive divergence (bm/PCD).

Reference: CCMpredPy / CCMgen ``--ofn-cd --persistent`` (``docs/conventions.md`` §9,
``docs/decisions.md`` D-010, D-011). The gradient is computed directly from
one- and two-point frequencies (not by autodiff):

    dLL/dJ[i,a,j,b] = f2_data[i,a,j,b] - f2_model[i,a,j,b]
    dLL/dh[i,a]     = f1_data[i,a]     - f1_model[i,a]

Model frequencies come from persistent Gibbs chains. Gap gradients and the
diagonal are zeroed; fields are frozen by default (``fix_v=True``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .energy import Params, couplings, offdiag_mask
from .io import GAP, Q, one_hot


# --------------------------------------------------------------------------- #
# Frequencies
# --------------------------------------------------------------------------- #
def one_hot_jax(S: jnp.ndarray, q: int) -> jnp.ndarray:
    """One-hot encode an int chain array ``(C, L)`` -> ``(C, L, q)`` float32 (device)."""
    return jax.nn.one_hot(S, q, dtype=jnp.float32)


def one_point_freqs(X: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Weighted single-site frequencies ``f1[i,a]`` from one-hot ``X`` (N,L,q)."""
    return jnp.einsum("n,nia->ia", w, X) / jnp.sum(w)


def two_point_freqs(X: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Weighted two-point frequencies ``f2[i,a,j,b]`` from one-hot ``X`` (N,L,q)."""
    return jnp.einsum("n,nia,njb->iajb", w, X, X) / jnp.sum(w)


def connected_correlations(f1: jnp.ndarray, f2: jnp.ndarray) -> jnp.ndarray:
    """Connected correlations ``c[i,a,j,b] = f2[i,a,j,b] - f1_i(a) f1_j(b)`` (bm diagnostic)."""
    return f2 - f1[:, :, None, None] * f1[None, None, :, :]


def apply_pseudocounts(
    f1: jnp.ndarray, f2: jnp.ndarray, tau: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Uniform pseudocount mix (CCMpredPy, D-012): singles
    ``(1-tau) f + tau/q``; pairs preserve the connected correlation,
    ``f2_pc = (1-tau)^2 (f2 - f1 f1) + f1_pc f1_pc``. ``tau = n_pc/(N_eff+n_pc)``.
    """
    q = f1.shape[1]
    f1_pc = (1.0 - tau) * f1 + tau / q
    conn = f2 - f1[:, :, None, None] * f1[None, None, :, :]
    f2_pc = (1.0 - tau) ** 2 * conn + f1_pc[:, :, None, None] * f1_pc[None, None, :, :]
    return f1_pc, f2_pc


def correlation_fit(f1_a, f2_a, f1_b, f2_b) -> float:
    """Pearson correlation between two sets of connected two-point correlations,
    over off-diagonal ``(i<j)`` blocks -- the fair convergence metric for bm.
    """
    L = f1_a.shape[0]
    ca = np.asarray(connected_correlations(f1_a, f2_a)).transpose(0, 2, 1, 3)
    cb = np.asarray(connected_correlations(f1_b, f2_b)).transpose(0, 2, 1, 3)
    iu, ju = np.triu_indices(L, k=1)
    a = ca[iu, ju].ravel()
    b = cb[iu, ju].ravel()
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------- #
# Gibbs sampler
# --------------------------------------------------------------------------- #
def gibbs_sweep(
    key: jax.Array,
    S: jnp.ndarray,
    h: jnp.ndarray,
    J: jnp.ndarray,
    n_sample_states: int,
    frozen: jnp.ndarray,
) -> jnp.ndarray:
    """One Gibbs sweep over all sites in random order, vectorised over chains.

    Sites are sequentially dependent (updated one at a time via ``lax.fori_loop``);
    chains are vectorised. For site ``i`` the per-chain logits are
    ``h[i,a] + sum_{j!=i} J[i,a,j,S[c,j]]`` (the ``j==i`` term is 0). States
    ``>= n_sample_states`` (the gap, when ``n_sample_states=20``) are excluded from
    the multinomial, and ``frozen`` positions are never updated (D-011).

    Parameters
    ----------
    S : int array ``(C, L)`` -- current chain states.
    h : ``(L, q)``, J : ``(L, q, L, q)``.
    frozen : bool array ``(C, L)`` -- positions to hold fixed (gaps).
    Returns the updated ``S``.
    """
    C, L = S.shape
    q = h.shape[1]
    key, kperm = jax.random.split(key)
    order = jax.random.permutation(kperm, L)
    neg_inf = jnp.full((q,), -jnp.inf).at[:n_sample_states].set(0.0)  # mask beyond n_sample_states

    def body(t, carry):
        S, key = carry
        i = order[t]
        Ji = J[i]  # (q, L, q)
        # gathered[a, c, j] = Ji[a, j, S[c, j]]
        gathered = Ji[:, jnp.arange(L)[None, :], S]      # (q, C, L)
        contrib = jnp.sum(gathered, axis=2).T             # (C, q)
        logits = h[i][None, :] + contrib + neg_inf[None, :]
        key, ksub = jax.random.split(key)
        drawn = jax.random.categorical(ksub, logits, axis=-1).astype(S.dtype)  # (C,)
        keep = frozen[:, i]
        new_col = jnp.where(keep, S[:, i], drawn)
        S = S.at[:, i].set(new_col)
        return S, key

    S, _ = jax.lax.fori_loop(0, L, body, (S, key))
    return S


def sample_chains(
    key: jax.Array,
    S: jnp.ndarray,
    h: jnp.ndarray,
    J: jnp.ndarray,
    n_steps: int,
    n_sample_states: int,
    frozen: jnp.ndarray,
) -> jnp.ndarray:
    """Run ``n_steps`` Gibbs sweeps. Reproducible from ``key`` (tested)."""
    def body(_, carry):
        S, key = carry
        key, ksub = jax.random.split(key)
        S = gibbs_sweep(ksub, S, h, J, n_sample_states, frozen)
        return S, key

    S, _ = jax.lax.fori_loop(0, n_steps, body, (S, key))
    return S


# --------------------------------------------------------------------------- #
# PCD training
# --------------------------------------------------------------------------- #
def _gradient(f1_data, f2_data, f1_model, f2_model, q, L):
    """Log-likelihood gradient (data - model), gap- and diagonal-zeroed."""
    g_h = f1_data - f1_model
    g_J = f2_data - f2_model
    # zero gap gradients (state q-1) and self-couplings
    if q == Q:
        g_J = g_J.at[:, GAP, :, :].set(0.0).at[:, :, :, GAP].set(0.0)
        g_h = g_h.at[:, GAP].set(0.0)
    g_J = g_J * offdiag_mask(L)
    # symmetrise
    g_J = 0.5 * (g_J + jnp.transpose(g_J, (2, 3, 0, 1)))
    return g_h, g_J


def fit(
    A: np.ndarray,
    weights: np.ndarray | None = None,
    n_chains: int | None = None,
    gibbs_steps: int = 1,
    alpha0: float = 1e-3,
    decay_rate: float = 5e-6,
    decay_start: float = 1e-1,
    lambda_single: float = 10.0,
    lambda_pair_factor: float = 0.2,
    pseudocount_n: float = 1.0,
    maxiter: int = 2000,
    convergence_prev: int = 5,
    epsilon: float = 1e-5,
    fix_v: bool = True,
    freeze_gaps: bool = True,
    persistent: bool = True,
    num_classes: int = Q,
    seed: int = 0,
    init: Params | None = None,
    record_every: int = 25,
    data_for_diagnostic: bool = True,
    verbose: bool = False,
) -> tuple[Params, dict]:
    """Fit a Potts model by persistent contrastive divergence.

    Conventions from CCMpredPy's ``--ofn-cd --persistent`` path (D-010): chains
    seeded from the data (``n_chains = max(N/10, 500)``), one Gibbs sweep per update,
    gradient ``N_eff*(f_data - f_model)`` on pseudocounted, weighted one-/two-point
    frequencies (gaps and diagonal zeroed), fields frozen (``fix_v``), gaps frozen
    during sampling, gradient descent with a compounding ("sig") learning-rate decay
    that starts once ``|Δ‖w‖|/‖w‖ < decay_start``.

    ``persistent=True`` (default) runs **true PCD**: the chains persist across updates
    (seeded once from the data, never reset), which is stable and equilibrates over
    many updates -- in practice this fits the two-point statistics better than the
    CCMpredPy vanilla-CD-until-decay schedule (see D-018). ``persistent=False`` runs
    vanilla CD (chains reseeded from data each step).

    Returns ``(params, info)`` where ``params.W`` already holds the symmetric ``J``.
    ``info`` carries the learning curve, the model connected-correlation fit to the
    data, ``n_eff``, and the ``||w||`` history.
    """
    A = np.asarray(A)
    N, L = A.shape
    q = num_classes
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)

    if weights is None:
        w = np.ones(N, dtype=np.float32)
    else:
        w = np.asarray(weights, dtype=np.float32)

    # Data frequencies, with pseudocounts (D-012).
    X = jnp.asarray(one_hot(A, num_classes=q))
    wj = jnp.asarray(w)
    neff = float(np.sum(w))
    tau = pseudocount_n / (neff + pseudocount_n)
    f1_data, f2_data = apply_pseudocounts(one_point_freqs(X, wj), two_point_freqs(X, wj), tau)

    # Initial parameters: fields at data log-frequencies (v-center prior), J = 0.
    if init is None:
        from .plm import init_fields_logodds

        h0 = init_fields_logodds(A, num_classes=q)
        init = Params(h=h0, W=jnp.zeros((L, q, L, q), dtype=jnp.float32))
    h = init.h
    J = couplings(init.W)

    # Persistent chains seeded from the data. Each chain carries the weight of the
    # data sequence it was seeded from, so model frequencies are weighted the same
    # way as the data frequencies (CCMpredPy: sample side re-weighted by seq weights).
    if n_chains is None:
        n_chains = max(N // 10, 500)

    def reseed(gen):
        idx = gen.integers(0, N, size=n_chains)
        S = jnp.asarray(A[idx].astype(np.int32))
        cw = jnp.asarray(w[idx])
        fr = (S == GAP) if (freeze_gaps and q == Q) else jnp.zeros_like(S, dtype=bool)
        return S, cw, fr

    S, chain_w, frozen = reseed(rng)
    n_sample_states = (q - 1) if (freeze_gaps and q == Q) else q

    lambda_pair = lambda_pair_factor * (L - 1)
    alpha = alpha0
    decay_active = False
    xnorm_hist: list[float] = []
    curve: list[dict] = []

    sweep = jax.jit(sample_chains, static_argnums=(4, 5))

    def model_freqs(S, cw):
        Xs = one_hot_jax(S, q)
        return apply_pseudocounts(one_point_freqs(Xs, cw), two_point_freqs(Xs, cw), tau)

    for it in range(maxiter):
        # persistent=True -> true PCD: chains persist across updates (seeded once from
        # data, never reset), which is stable and equilibrates over many updates.
        # persistent=False -> vanilla CD: chains reseeded from data each step (D-010).
        if not persistent:
            S, chain_w, frozen = reseed(rng)
        key, ksub = jax.random.split(key)
        S = sweep(ksub, S, h, J, gibbs_steps, n_sample_states, frozen)

        f1_model, f2_model = model_freqs(S, chain_w)
        g_h, g_J = _gradient(f1_data, f2_data, f1_model, f2_model, q, L)
        # The gradient of the weighted log-likelihood is N_eff * (f_data - f_model);
        # scaling to count units balances against the L2 penalty as in CCMpredPy.
        g_h = neff * g_h
        g_J = neff * g_J
        # L2 regularisation gradient (couplings toward 0; penalty 0.5*lambda_pair*||J||^2).
        g_J = g_J - lambda_pair * J
        if not fix_v:
            g_h = g_h - 2.0 * lambda_single * h.at[:, -1].set(0.0)

        # Gradient ASCENT on log-likelihood.
        J = J + alpha * g_J
        J = J * offdiag_mask(L)
        if not fix_v:
            h = (h + alpha * g_h).at[:, -1].set(0.0)

        # Convergence on relative change of ||w||.
        xnorm = float(jnp.sqrt(jnp.sum(J * J) / 2.0))
        xnorm_hist.append(xnorm)
        xdiff = np.inf
        if len(xnorm_hist) > convergence_prev:
            prev = xnorm_hist[-convergence_prev - 1]
            xdiff = abs(prev - xnorm) / (prev + 1e-30)
            if not decay_active and xdiff < decay_start:
                decay_active = True
                t_decay = it
        if decay_active:
            alpha = alpha * (1.0 / (1.0 + decay_rate * (it - t_decay + 1)))  # "sig" decay

        if verbose and (it % record_every == 0 or it == maxiter - 1):
            print(f"  [pcd] iter {it:4d}  ||w||={xnorm:.4f}  alpha={alpha:.2e}  d={xdiff:.2e}")
        if it % record_every == 0 or it == maxiter - 1:
            rec = {"iter": it, "xnorm": xnorm, "alpha": alpha}
            if data_for_diagnostic:
                rec["corr"] = correlation_fit(f1_data, f2_data, f1_model, f2_model)
            curve.append(rec)
        if decay_active and xdiff < epsilon:
            break

    params = Params(h=h.at[:, -1].set(0.0), W=J)  # store symmetric J directly as "W"
    info = {
        "curve": curve,
        "xnorm_history": xnorm_hist,
        "n_iter": len(xnorm_hist),
        "n_eff": neff,
        "n_chains": n_chains,
        "lambda_pair": lambda_pair,
        "final_corr": correlation_fit(f1_data, f2_data, f1_model, f2_model),
    }
    return params, info


def couplings_of(params: Params) -> jnp.ndarray:
    """For a bm fit, ``params.W`` already holds the symmetric ``J``; return it masked."""
    return params.W * offdiag_mask(params.W.shape[0])
