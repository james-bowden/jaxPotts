"""Tests for sequence reweighting against a slow NumPy reference (D-004)."""

from __future__ import annotations

import math

import numpy as np

from jaxpotts import weights


def _slow_weights(A, cutoff, inclusive):
    """Obviously-correct O(N^2 L) reference."""
    N, L = A.shape
    idthres = math.ceil(cutoff * L)
    sizes = np.zeros(N)
    for i in range(N):
        for j in range(N):
            ids = int(np.sum(A[i] == A[j]))
            hit = ids >= idthres if inclusive else ids > idthres
            if hit:
                sizes[i] += 1
    return 1.0 / sizes


def test_weights_match_slow_reference_strict():
    rng = np.random.default_rng(0)
    A = rng.integers(0, 21, size=(40, 25)).astype(np.int8)
    w_fast = weights.sequence_weights(A, cutoff=0.7, inclusive=False, chunk_size=7)
    w_slow = _slow_weights(A, 0.7, inclusive=False)
    assert np.allclose(w_fast, w_slow, atol=1e-6)


def test_weights_match_slow_reference_inclusive():
    rng = np.random.default_rng(1)
    A = rng.integers(0, 21, size=(50, 30)).astype(np.int8)
    w_fast = weights.sequence_weights(A, cutoff=0.8, inclusive=True, chunk_size=16)
    w_slow = _slow_weights(A, 0.8, inclusive=True)
    assert np.allclose(w_fast, w_slow, atol=1e-6)


def test_strict_vs_inclusive_differ_on_boundary():
    # Two sequences whose identity is exactly ceil(cutoff*L): '>=' clusters them,
    # '>' does not. This is the C-vs-Py divergence (D-004).
    L = 10
    cutoff = 0.8  # idthres = 8
    base = np.zeros(L, dtype=np.int8)
    other = base.copy()
    other[:2] = 1  # 8 identical columns == idthres
    A = np.stack([base, other])
    w_incl = weights.sequence_weights(A, cutoff=cutoff, inclusive=True)
    w_strict = weights.sequence_weights(A, cutoff=cutoff, inclusive=False)
    assert np.allclose(w_incl, [0.5, 0.5])   # clustered together
    assert np.allclose(w_strict, [1.0, 1.0])  # each its own cluster


def test_uniform_when_cutoff_one():
    rng = np.random.default_rng(2)
    A = rng.integers(0, 21, size=(20, 15)).astype(np.int8)
    w = weights.sequence_weights(A, cutoff=1.0)
    assert np.allclose(w, 1.0)


def test_identical_sequences_downweighted():
    A = np.tile(np.arange(10, dtype=np.int8), (5, 1))  # 5 identical sequences
    w = weights.sequence_weights(A, cutoff=0.8)
    assert np.allclose(w, 0.2)  # cluster size 5
    assert np.isclose(weights.n_eff(w), 1.0)


def test_neff_le_n():
    rng = np.random.default_rng(3)
    A = rng.integers(0, 21, size=(60, 20)).astype(np.int8)
    w = weights.sequence_weights(A, cutoff=0.8)
    assert weights.n_eff(w) <= 60 + 1e-6
