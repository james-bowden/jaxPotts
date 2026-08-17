"""Exact enumeration utilities for tiny Potts models (L, q small enough that q**L
sequences fit in memory). Used by the plm and bm correctness tests.

All functions operate on ``(h, J)`` with the conventions of ``jaxpotts.energy``:
``E(x) = -sum_i h[i,x_i] - sum_{i<j} J[i,x_i,j,x_j]``, ``P(x) ∝ exp(-E)``.
"""

from __future__ import annotations

import itertools

import numpy as np


def all_sequences(L: int, q: int) -> np.ndarray:
    """Every sequence of length ``L`` over ``q`` states, shape ``(q**L, L)``."""
    return np.array(list(itertools.product(range(q), repeat=L)), dtype=np.int64)


def exact_distribution(h, J):
    """Return ``(sequences, probs)`` for the full Boltzmann distribution."""
    h = np.asarray(h)
    J = np.asarray(J)
    L, q = h.shape
    seqs = all_sequences(L, q)
    idx = np.arange(L)
    e_single = h[idx, seqs].sum(axis=1)            # (S,)
    # pair energy: sum_{i<j} J[i, x_i, j, x_j]
    e_pair = np.zeros(len(seqs))
    for i in range(L):
        for j in range(i + 1, L):
            e_pair += J[i, seqs[:, i], j, seqs[:, j]]
    E = -(e_single + e_pair)
    logp = -E
    logp -= logp.max()
    p = np.exp(logp)
    p /= p.sum()
    return seqs, p


def exact_marginals(h, J):
    """Exact one- and two-point frequencies ``(f_single (L,q), f_pair (L,q,L,q))``.

    ``f_pair`` is indexed ``[i, a, j, b]`` to match ``J`` and ``jaxpotts.bm``.
    """
    h = np.asarray(h)
    L, q = h.shape
    seqs, p = exact_distribution(h, J)
    f1 = np.zeros((L, q))
    for i in range(L):
        for a in range(q):
            f1[i, a] = p[seqs[:, i] == a].sum()
    f2 = np.zeros((L, q, L, q))
    for i in range(L):
        for j in range(L):
            for a in range(q):
                mask_a = seqs[:, i] == a
                for b in range(q):
                    f2[i, a, j, b] = p[mask_a & (seqs[:, j] == b)].sum()
    return f1, f2


def connected_correlations(f1, f2):
    """Connected two-point correlations ``c[i,a,j,b] = f2[i,a,j,b] - f_i(a) f_j(b)``."""
    return f2 - f1[:, :, None, None] * f1[None, None, :, :]


def sample_from_distribution(h, J, n, seed=0):
    """Draw ``n`` i.i.d. sequences from the exact distribution, shape ``(n, L)``."""
    seqs, p = exact_distribution(h, J)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(seqs), size=n, p=p)
    return seqs[idx].astype(np.int8)
