"""Tests for pseudo-likelihood: gradient correctness, chunk invariance, recovery."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _exact import sample_from_distribution

from jaxpotts import energy, gauge, plm
from jaxpotts.io import one_hot

jax.config.update("jax_enable_x64", True)


def _tiny_params(L, q, seed):
    rng = np.random.default_rng(seed)
    h = jnp.asarray(rng.normal(size=(L, q)) * 0.5).at[:, 20 if q == 21 else q - 1].set(0.0)
    W = jnp.asarray(rng.normal(size=(L, q, L, q)) * 0.5)
    return energy.Params(h=h, W=W)


def _finite_diff_grad(loss, params, eps=1e-5, n_probe=8, seed=0):
    """Directional finite-difference check: analytic vs numerical along random probes."""
    rng = np.random.default_rng(seed)
    _, grad = jax.value_and_grad(loss)(params)
    leaves, _ = jax.tree_util.tree_flatten(params)
    for _ in range(n_probe):
        dirs = [jnp.asarray(rng.normal(size=leaf.shape)) for leaf in leaves]
        d = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(params), dirs)
        plus = jax.tree_util.tree_map(lambda p, dd: p + eps * dd, params, d)
        minus = jax.tree_util.tree_map(lambda p, dd: p - eps * dd, params, d)
        numeric = (loss(plus) - loss(minus)) / (2 * eps)
        analytic = sum(jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(lambda g, dd: jnp.vdot(g, dd), grad, d)))
        assert jnp.allclose(numeric, analytic, atol=1e-4, rtol=1e-4), (numeric, analytic)


def test_plm_gradient_matches_finite_difference():
    L, q, N = 3, 21, 6
    rng = np.random.default_rng(0)
    A = rng.integers(0, q, size=(N, L)).astype(np.int8)
    X = jnp.asarray(one_hot(A).astype(np.float64))
    w = jnp.asarray(rng.uniform(0.5, 1.5, size=N))
    vg = plm.make_value_and_grad(X, w, lambda_single=0.1, lambda_pair=0.3, chunk_size=N)
    params = _tiny_params(L, q, seed=1)
    _finite_diff_grad(lambda p: vg(p)[0], params)


def test_gradient_invariant_to_chunk_size():
    # The single most valuable test: chunked gradient == full-batch gradient.
    L, q, N = 4, 21, 23  # N deliberately not a multiple of the chunk sizes
    rng = np.random.default_rng(2)
    A = rng.integers(0, q, size=(N, L)).astype(np.int8)
    X = jnp.asarray(one_hot(A).astype(np.float64))
    w = jnp.asarray(rng.uniform(0.3, 2.0, size=N))
    params = _tiny_params(L, q, seed=3)

    val_full, grad_full = plm.make_value_and_grad(X, w, 0.05, 0.2, chunk_size=N)(params)
    for cs in (1, 4, 7, N):
        val, grad = plm.make_value_and_grad(X, w, 0.05, 0.2, chunk_size=cs)(params)
        assert jnp.allclose(val, val_full, atol=1e-9)
        assert jnp.allclose(grad.h, grad_full.h, atol=1e-9)
        assert jnp.allclose(grad.W, grad_full.W, atol=1e-9)


def test_fit_invariants():
    # Fitted parameters satisfy symmetry, zero-diagonal, gap-pin.
    rng = np.random.default_rng(4)
    A = rng.integers(0, 21, size=(50, 6)).astype(np.int8)
    params, info = plm.fit(A, maxiter=20, method="lbfgs")
    J = energy.couplings(params.W)
    assert jnp.allclose(J, jnp.transpose(J, (2, 3, 0, 1)), atol=1e-6)
    for i in range(6):
        assert jnp.allclose(J[i, :, i, :], 0.0)
    assert jnp.allclose(params.h[:, 20], 0.0)
    assert len(info["history"]) >= 1


def test_result_invariant_to_sequence_order():
    rng = np.random.default_rng(5)
    A = rng.integers(0, 21, size=(40, 5)).astype(np.int8)
    w = rng.uniform(0.5, 1.5, size=40).astype(np.float32)
    p1, _ = plm.fit(A, weights=w, maxiter=40, method="lbfgs")
    perm = rng.permutation(40)
    p2, _ = plm.fit(A[perm], weights=w[perm], maxiter=40, method="lbfgs")
    J1 = np.asarray(energy.couplings(p1.W))
    J2 = np.asarray(energy.couplings(p2.W))
    assert np.allclose(J1, J2, atol=1e-4)


@pytest.mark.slow
def test_plm_recovers_couplings_from_samples():
    # Draw many samples from a known small J, fit with plm, check correlation in ZSG.
    # plm is only asymptotically consistent, so this is a loose test.
    L, q = 4, 3
    truth = _tiny_params(L, q, seed=7)
    J_true = energy.couplings(truth.W)
    A = sample_from_distribution(truth.h, J_true, n=20000, seed=8)
    params, _ = plm.fit(A, lambda_single=0.01, lambda_pair_factor=0.05,
                        maxiter=200, method="lbfgs", num_classes=q)
    _, J_fit_zs = gauge.zero_sum_gauge(params.h, energy.couplings(params.W))
    _, J_true_zs = gauge.zero_sum_gauge(truth.h, J_true)
    # Off-diagonal (i != j) blocks only; transpose to (L, L, q, q) to mask on (i, j).
    mask = ~np.eye(L, dtype=bool)
    a = np.transpose(np.asarray(J_fit_zs), (0, 2, 1, 3))[mask].ravel()
    b = np.transpose(np.asarray(J_true_zs), (0, 2, 1, 3))[mask].ravel()
    r = np.corrcoef(a, b)[0, 1]
    assert r > 0.9, f"plm coupling recovery correlation too low: {r:.3f}"
