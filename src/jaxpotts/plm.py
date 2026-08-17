"""Pseudo-likelihood maximisation (plm). Reference: CCMpred (C).

The objective is the weighted sum over sequences and sites of the conditional
cross-entropy of the observed state, plus L2 regularisation
(``docs/conventions.md`` §6, D-005):

    loss = sum_n w_n sum_i  -log softmax(logits[n,i,:])[x_ni]
         + lambda_single * ||h[:, :20]||^2
         + 0.5 * lambda_pair * ||J||^2

with ``logits[n,i,a] = h[i,a] + sum_{j!=i} sum_b J[i,a,j,b] X[n,j,b]`` and ``J``
the symmetric, zero-diagonal couplings. The gap field ``h[:,20]`` is pinned to 0.

The gradient is chunked over sequences with :func:`jax.lax.scan` so peak memory is
independent of ``N``; the chunked gradient is exactly the full-batch gradient
(tested), because the loss is a plain sum over sequences.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .energy import Params, couplings, fields
from .io import Q, one_hot
from .optim import minimize


def init_fields_logodds(A: np.ndarray, num_classes: int = Q) -> jnp.ndarray:
    """CCMpred field init: ``h[i,a] = log(f_a) - log(f_ref)`` from pseudocounted,
    **unweighted** single-site frequencies ``f_a = (count_a + 1)/(N + q)``.

    The reference state is the last one (the gap for the standard alphabet); its
    field is 0. Returns ``(L, q)`` (D-006).
    """
    A = np.asarray(A)
    N, L = A.shape
    ref = num_classes - 1
    counts = np.ones((L, num_classes), dtype=np.float64)  # flat pseudocount of 1
    for a in range(num_classes):
        counts[:, a] += np.sum(A == a, axis=0)
    freqs = counts / (N + num_classes)
    h = np.log(freqs) - np.log(freqs[:, ref:ref + 1])
    h[:, ref] = 0.0
    return jnp.asarray(h, dtype=jnp.float32)


def _chunk_data_loss(params: Params, Xc: jnp.ndarray, wc: jnp.ndarray) -> jnp.ndarray:
    """Weighted pseudo-likelihood cross-entropy over one chunk of sequences."""
    h = fields(params.h)
    J = couplings(params.W)
    logits = h[None] + jnp.einsum("njb,iajb->nia", Xc, J, optimize=True)
    logp = jax.nn.log_softmax(logits, axis=-1)
    ce = -jnp.sum(Xc * logp, axis=-1)          # (chunk, L)
    return jnp.sum(wc * jnp.sum(ce, axis=1))


def _regularization(params: Params, lambda_single: float, lambda_pair: float) -> jnp.ndarray:
    """L2 penalty matching CCMpred's *effective* gradient (D-005).

    CCMpred's reg gradient is ``2*lambda_single*v`` and ``2*lambda_pair*w`` (its
    ``0.5*lambda_pair*w^2`` objective term is an internal inconsistency -- the
    optimizer follows the gradient). The minimum it targets is therefore
    ``lambda_single*||v||^2 + lambda_pair*||J||^2`` (``J`` counted over both
    orientations, matching CCMpred's doubled ``w`` storage).
    """
    h = fields(params.h)
    J = couplings(params.W)
    reg_single = lambda_single * jnp.sum(h[:, :-1] ** 2)  # exclude the pinned ref column
    reg_pair = lambda_pair * jnp.sum(J ** 2)
    return reg_single + reg_pair


def make_value_and_grad(
    X: jnp.ndarray,
    w: jnp.ndarray,
    lambda_single: float,
    lambda_pair: float,
    chunk_size: int,
):
    """Build a jitted ``params -> (loss, grad)`` closure, chunked over sequences.

    ``X`` is ``(N, L, q)`` float32, ``w`` is ``(N,)``. ``N`` is padded up to a
    multiple of ``chunk_size`` with zero-weight sequences (which contribute nothing
    to loss or gradient), so the result is identical for any ``chunk_size``.
    """
    N = X.shape[0]
    n_chunks = int(np.ceil(N / chunk_size))
    pad = n_chunks * chunk_size - N
    if pad:
        X = jnp.concatenate([X, jnp.zeros((pad, *X.shape[1:]), X.dtype)], axis=0)
        w = jnp.concatenate([w, jnp.zeros((pad,), w.dtype)], axis=0)
    Xc = X.reshape(n_chunks, chunk_size, *X.shape[1:])
    wc = w.reshape(n_chunks, chunk_size)

    data_vg = jax.value_and_grad(_chunk_data_loss)
    reg_vg = jax.value_and_grad(partial(_regularization, lambda_single=lambda_single,
                                        lambda_pair=lambda_pair))

    @jax.jit
    def value_and_grad(params: Params):
        def body(carry, chunk):
            val, grad = carry
            xc, wcc = chunk
            v, g = data_vg(params, xc, wcc)
            val = val + v
            grad = jax.tree_util.tree_map(lambda a, b: a + b, grad, g)
            return (val, grad), None

        zero_grad = jax.tree_util.tree_map(jnp.zeros_like, params)
        loss_dtype = jax.eval_shape(_chunk_data_loss, params, Xc[0], wc[0]).dtype
        zero_val = jnp.zeros((), loss_dtype)
        (data_val, data_grad), _ = jax.lax.scan(body, (zero_val, zero_grad), (Xc, wc))
        reg_val, reg_grad = reg_vg(params)
        total = data_val + reg_val
        grad = jax.tree_util.tree_map(lambda a, b: a + b, data_grad, reg_grad)
        return total, grad

    return value_and_grad


def fit(
    A: np.ndarray,
    weights: np.ndarray | None = None,
    lambda_single: float = 0.01,
    lambda_pair_factor: float = 0.2,
    method: str = "lbfgs",
    maxiter: int = 250,
    tol: float = 1e-5,
    chunk_size: int = 4096,
    init: Params | None = None,
    num_classes: int = Q,
    verbose: bool = False,
) -> tuple[Params, dict]:
    """Fit a Potts model by pseudo-likelihood maximisation.

    Parameters
    ----------
    A : int array ``(N, L)``
        Integer-encoded MSA.
    weights : float array ``(N,)`` or None
        Per-sequence weights; uniform if None.
    lambda_single, lambda_pair_factor : float
        Regularisation. ``lambda_pair = lambda_pair_factor * (L - 1)``. Defaults are
        CCMpred's C defaults (``lambda_single=0.01``); see D-005.
    method : {"lbfgs", "cg", "adam"}
        Optimizer (see :mod:`jaxpotts.optim`).
    maxiter, tol : int, float
        Optimizer budget and relative convergence tolerance.
    chunk_size : int
        Sequence chunk for the gradient scan (memory control; result-invariant).
    init : Params or None
        Initial parameters; defaults to log-odds fields and ``W = 0`` (D-006).

    Returns
    -------
    (params, info) : Params, dict
        ``params`` with ``params.h`` (gap-pinned) and free ``params.W``; use
        :func:`jaxpotts.energy.couplings` for ``J``. ``info`` has ``history`` (loss
        per iteration), ``n_eff``, ``lambda_single``, ``lambda_pair``, ``method``,
        and ``n_iter``.
    """
    A = np.asarray(A)
    N, L = A.shape
    lambda_pair = lambda_pair_factor * (L - 1)
    X = jnp.asarray(one_hot(A, num_classes=num_classes))
    if weights is None:
        w = jnp.ones(N, dtype=jnp.float32)
    else:
        w = jnp.asarray(weights, dtype=jnp.float32)

    if init is None:
        h0 = init_fields_logodds(A, num_classes=num_classes)
        init = Params(h=h0, W=jnp.zeros((L, num_classes, L, num_classes), dtype=jnp.float32))

    vg = make_value_and_grad(X, w, lambda_single, lambda_pair, chunk_size)
    params, history = minimize(vg, init, method=method, maxiter=maxiter, tol=tol, verbose=verbose)

    # Enforce invariants on the returned parameters.
    params = Params(h=fields(params.h), W=params.W)
    info = {
        "history": history,
        "n_eff": float(np.sum(np.asarray(w))),
        "lambda_single": lambda_single,
        "lambda_pair": lambda_pair,
        "method": method,
        "n_iter": len(history),
    }
    return params, info


def couplings_of(params: Params) -> jnp.ndarray:
    """Convenience: symmetric zero-diagonal ``J`` for fitted ``params``."""
    return couplings(params.W)


__all__ = ["fit", "make_value_and_grad", "init_fields_logodds", "couplings_of"]
