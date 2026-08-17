"""Sequence reweighting (Henikoff-style identity clustering).

Convention (``docs/conventions.md`` §5, D-004): cluster sequences at a fractional
identity ``cutoff`` (default 0.8) computed over **all L columns** (gap-gap counts
as a match), and weight each sequence by ``1 / cluster_size``. ``N_eff = sum(w)``.

The two references differ on the comparison operator:

- CCMpred (C): two sequences are in the same cluster iff ``n_identical  >  ceil(cutoff*L)``
- CCMpredPy:   iff ``n_identical  >= ceil(cutoff*L)``

Select with ``inclusive=`` (``False`` => ``>`` (C, plm), ``True`` => ``>=`` (Py, bm)).

Cost is ``O(N^2 L)``; implemented as a chunked one-hot matmul over row blocks so
the ``(N, N)`` identity matrix is never materialized.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from .io import Q, one_hot


def _cluster_sizes(
    X_flat: jnp.ndarray, idthres: float, inclusive: bool, chunk_size: int
) -> np.ndarray:
    """Return cluster_size[i] = #{ j : identity(i, j) meets the threshold }, incl. self."""
    N = X_flat.shape[0]

    @jax.jit
    def block_sizes(block: jnp.ndarray) -> jnp.ndarray:
        # counts[r, j] = number of columns where row (block[r]) and seq j agree.
        counts = block @ X_flat.T
        similar = counts >= idthres if inclusive else counts > idthres
        return jnp.sum(similar, axis=1)

    sizes = np.empty(N, dtype=np.float64)
    for b0 in range(0, N, chunk_size):
        b1 = min(b0 + chunk_size, N)
        sizes[b0:b1] = np.asarray(block_sizes(X_flat[b0:b1]))
    return sizes


def sequence_weights(
    A: np.ndarray,
    cutoff: float = 0.8,
    inclusive: bool = False,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Compute per-sequence weights for an integer MSA ``A`` of shape ``(N, L)``.

    Parameters
    ----------
    A : int array ``(N, L)``
        Integer-encoded MSA.
    cutoff : float
        Fractional identity cutoff (default 0.8). ``cutoff >= 1`` yields uniform
        weights of 1.0, matching both references.
    inclusive : bool
        ``False`` (default) uses the strict ``>`` rule (CCMpred/plm); ``True`` uses
        ``>=`` (CCMpredPy/bm).
    chunk_size : int
        Row-block size for the identity matmul.

    Returns
    -------
    w : float32 array ``(N,)``
        ``w[i] = 1 / cluster_size[i]``. ``N_eff = w.sum()``.
    """
    A = np.asarray(A)
    N, L = A.shape
    if cutoff >= 1.0:
        return np.ones(N, dtype=np.float32)
    idthres = float(math.ceil(cutoff * L))
    X_flat = jnp.asarray(one_hot(A).reshape(N, L * Q))
    sizes = _cluster_sizes(X_flat, idthres, inclusive, chunk_size)
    return (1.0 / sizes).astype(np.float32)


def n_eff(w: np.ndarray) -> float:
    """Effective number of sequences ``N_eff = sum(w)``."""
    return float(np.sum(w))
