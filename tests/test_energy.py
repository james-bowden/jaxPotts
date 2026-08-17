"""Tests for the shared energy function and symmetry invariants (§2-4)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxpotts import energy

jax.config.update("jax_enable_x64", True)


def test_couplings_symmetric_zero_diagonal():
    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.normal(size=(5, 21, 5, 21)))
    J = energy.couplings(W)
    # J[i,a,j,b] == J[j,b,i,a]
    assert jnp.allclose(J, jnp.transpose(J, (2, 3, 0, 1)), atol=1e-12)
    # J[i,:,i,:] == 0
    for i in range(5):
        assert jnp.allclose(J[i, :, i, :], 0.0)


def test_conditional_logits_shape_and_exclusion():
    L, q, N = 4, 21, 3
    rng = np.random.default_rng(1)
    h = jnp.asarray(rng.normal(size=(L, q))).at[:, 20].set(0.0)
    W = jnp.asarray(rng.normal(size=(L, q, L, q)))
    J = energy.couplings(W)
    A = rng.integers(0, q, size=(N, L))
    X = jnp.asarray(np.eye(q)[A])
    logits = energy.conditional_logits(h, J, X)
    assert logits.shape == (N, L, q)
    # Since J has zero diagonal blocks, site i's logits don't depend on X[:, i, :].
    X2 = X.at[:, 0, :].set(0.0).at[:, 0, 5].set(1.0)  # change state at site 0
    logits2 = energy.conditional_logits(h, J, X2)
    assert jnp.allclose(logits[:, 0, :], logits2[:, 0, :], atol=1e-10)


def test_energy_matches_explicit_sum():
    # Compare vectorised energy to an explicit per-pair Python loop on a tiny model.
    L, q = 3, 4
    rng = np.random.default_rng(2)
    h = jnp.asarray(rng.normal(size=(L, q)))
    W = jnp.asarray(rng.normal(size=(L, q, L, q)))
    J = energy.couplings(W)
    x = np.array([1, 3, 0])
    X = jnp.asarray(np.eye(q)[x][None])
    E_vec = float(energy.sequence_energy(h, J, X)[0])
    E_ref = -float(sum(h[i, x[i]] for i in range(L)))
    for i in range(L):
        for j in range(i + 1, L):
            E_ref -= float(J[i, x[i], j, x[j]])
    assert np.isclose(E_vec, E_ref, atol=1e-9)


def test_probability_normalisation_tiny():
    # For L=2, q=3 (9 states) probabilities from E sum to 1.
    L, q = 2, 3
    rng = np.random.default_rng(3)
    h = jnp.asarray(rng.normal(size=(L, q)))
    W = jnp.asarray(rng.normal(size=(L, q, L, q)))
    J = energy.couplings(W)
    states = np.array([[a, b] for a in range(q) for b in range(q)])
    X = jnp.asarray(np.eye(q)[states])
    E = energy.sequence_energy(h, J, X)
    p = jax.nn.softmax(-E)
    assert jnp.isclose(p.sum(), 1.0)
