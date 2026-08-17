"""Shared Potts energy and parameter container used by both ``plm`` and ``bm``.

Conventions (``docs/conventions.md`` §2-4):

- Fields ``h`` shape ``(L, q)``; the gap column ``h[:, 20]`` is pinned to 0.
- Couplings ``J`` shape ``(L, q, L, q)``, symmetric with zero diagonal blocks.
- ``J`` is materialised from a free tensor ``W`` by symmetrising in the forward
  pass: ``J = 0.5 (W + W^T) * offdiag_mask``; autodiff handles the gradient.
- Energy ``E(x) = -sum_i h[i, x_i] - sum_{i<j} J[i, x_i, j, x_j]``, ``P(x) ∝ e^{-E}``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from .io import Q


class Params(NamedTuple):
    """Free parameters of the Potts model (a JAX pytree).

    Attributes
    ----------
    h : ``(L, q)`` float array
        Single-site fields. The gap column is pinned to 0 by :func:`fields`.
    W : ``(L, q, L, q)`` float array
        Free (un-symmetrised) coupling tensor. Use :func:`couplings` to obtain the
        symmetric, zero-diagonal ``J``.
    """

    h: jnp.ndarray
    W: jnp.ndarray


def offdiag_mask(L: int) -> jnp.ndarray:
    """Boolean mask ``(L, 1, L, 1)`` that is ``False`` on the ``i == j`` diagonal."""
    eye = jnp.eye(L, dtype=bool)
    return (~eye)[:, None, :, None]


def fields(h: jnp.ndarray) -> jnp.ndarray:
    """Return the fields with the reference (gap) column pinned to 0.

    The gap is always the last state (index 20 for the standard alphabet, ``q-1``
    for smaller test models), so the last column is pinned.
    """
    return h.at[:, -1].set(0.0)


def couplings(W: jnp.ndarray) -> jnp.ndarray:
    """Materialise symmetric, zero-diagonal couplings ``J`` from the free tensor ``W``.

    ``J[i,a,j,b] = 0.5 (W[i,a,j,b] + W[j,b,i,a])`` for ``i != j``, else 0.
    """
    L = W.shape[0]
    J = 0.5 * (W + jnp.transpose(W, (2, 3, 0, 1)))
    return J * offdiag_mask(L)


def conditional_logits(h: jnp.ndarray, J: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    """Per-site conditional logits, shape ``(N, L, q)``.

    ``logits[n, i, a] = h[i, a] + sum_{j != i} sum_b J[i, a, j, b] X[n, j, b]``.
    The ``j != i`` exclusion is automatic because ``J`` has zero diagonal blocks.
    ``h`` is used as given; callers maintain the gap-field pin ``h[:, GAP] = 0``
    (via :func:`init_params`, the fitters, or :func:`fields`).
    """
    return h[None] + jnp.einsum("njb,iajb->nia", X, J, optimize=True)


def sequence_energy(h: jnp.ndarray, J: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    """Total energy ``E(x)`` per sequence, shape ``(N,)``.

    ``E = -sum_i h[i, x_i] - sum_{i<j} J[i, x_i, j, x_j]``. Uses the symmetric,
    zero-diagonal ``J`` (so the full double sum is twice the ``i<j`` sum). ``h`` is
    used as given (the gap-field pin is a caller invariant, not enforced here).
    """
    e_single = jnp.einsum("nia,ia->n", X, h)
    e_pair = 0.5 * jnp.einsum("nia,iajb,njb->n", X, J, X, optimize=True)
    return -(e_single + e_pair)


def init_params(L: int, h0: jnp.ndarray | None = None) -> Params:
    """Initialise :class:`Params` with ``W = 0`` and optional fields ``h0``.

    If ``h0`` is ``None``, fields start at 0. The gap column is pinned to 0 either
    way (via :func:`fields` at use time).
    """
    h = jnp.zeros((L, Q)) if h0 is None else jnp.asarray(h0)
    W = jnp.zeros((L, Q, L, Q))
    return Params(h=h, W=W)
