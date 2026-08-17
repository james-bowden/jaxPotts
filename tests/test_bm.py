"""Tests for Boltzmann machine learning: gradient, recovery, sampler validity."""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _exact import exact_marginals

from jaxpotts import bm, energy, gauge

jax.config.update("jax_enable_x64", True)


def _tiny_params(L, q, seed, scale=0.7):
    rng = np.random.default_rng(seed)
    h = jnp.asarray(rng.normal(size=(L, q)) * scale).at[:, q - 1].set(0.0)
    W = jnp.asarray(rng.normal(size=(L, q, L, q)) * scale)
    return energy.Params(h=h, W=W)


def test_bm_gradient_matches_autodiff_of_exact_ll():
    # Analytic bm gradient (data - model two-point freqs) vs autodiff of the exact
    # expected log-likelihood, on a system small enough to enumerate (L=4, q=3 -> 81).
    from _exact import exact_distribution

    L, q = 4, 3
    truth = _tiny_params(L, q, seed=1)
    J_true = energy.couplings(truth.W)
    f1_data, f2_data = exact_marginals(truth.h, J_true)
    f1_data, f2_data = jnp.asarray(f1_data), jnp.asarray(f2_data)
    _, p_data = exact_distribution(truth.h, J_true)
    p_data = jnp.asarray(p_data)

    params0 = _tiny_params(L, q, seed=2)
    J0 = energy.couplings(params0.W)
    f1_model, f2_model = exact_marginals(params0.h, np.asarray(J0))
    _, g_J = bm._gradient(f1_data, f2_data, jnp.asarray(f1_model), jnp.asarray(f2_model), q, L)

    seqs = jnp.asarray(list(itertools.product(range(q), repeat=L)))
    Xseq = jax.nn.one_hot(seqs, q)

    def expected_ll(W):
        J = energy.couplings(W)
        E = energy.sequence_energy(energy.fields(params0.h), J, Xseq)
        logZ = jax.scipy.special.logsumexp(-E)
        return jnp.sum(p_data * (-E - logZ))

    # g_J is the physical log-likelihood gradient in "unique edge" units (what we add
    # to J): g_J = f2_data - f2_model on the off-diagonal. The energy uses
    # 0.5*sum_{i,j} (= sum_{i<j}), so the autodiff gradient wrt the free tensor W is
    # exactly half of g_J. Asserting that exact factor validates formula and sign.
    grad_ad_W = jax.grad(expected_ll)(params0.W)
    assert jnp.allclose(g_J, 2.0 * grad_ad_W, atol=1e-6)


def test_exact_enumeration_recovery():
    # Strongest correctness test: recover a random J from exact marginals up to gauge.
    L, q = 4, 3
    truth = _tiny_params(L, q, seed=3, scale=0.6)
    J_true = energy.couplings(truth.W)
    f1_data, f2_data = exact_marginals(truth.h, J_true)
    f1_data = jnp.asarray(f1_data)
    f2_data = jnp.asarray(f2_data)

    # Gradient ascent with EXACT model marginals (no sampling noise).
    h = jnp.asarray(truth.h)          # fix fields at truth (bm freezes v anyway)
    J = jnp.zeros((L, q, L, q))
    for _ in range(4000):
        f1_m, f2_m = exact_marginals(np.asarray(h), np.asarray(J))
        _, g_J = bm._gradient(f1_data, f2_data, jnp.asarray(f1_m), jnp.asarray(f2_m), q, L)
        J = J + 1.0 * g_J
        J = J * energy.offdiag_mask(L)

    _, Jz = gauge.zero_sum_gauge(h, J)
    _, Jz_true = gauge.zero_sum_gauge(jnp.asarray(truth.h), J_true)
    mask = ~np.eye(L, dtype=bool)
    a = np.asarray(Jz).transpose(0, 2, 1, 3)[mask].ravel()
    b = np.asarray(Jz_true).transpose(0, 2, 1, 3)[mask].ravel()
    assert np.max(np.abs(a - b)) < 1e-2, f"max|dJ|={np.max(np.abs(a-b)):.4f}"
    assert np.corrcoef(a, b)[0, 1] > 0.999


def test_sampler_reproducible():
    L, q, C = 5, 3, 64
    params = _tiny_params(L, q, seed=4)
    J = energy.couplings(params.W)
    S0 = jnp.asarray(np.random.default_rng(0).integers(0, q, size=(C, L)).astype(np.int32))
    frozen = jnp.zeros_like(S0, dtype=bool)
    key = jax.random.key(42)
    S1 = bm.sample_chains(key, S0, energy.fields(params.h), J, 5, q, frozen)
    S2 = bm.sample_chains(key, S0, energy.fields(params.h), J, 5, q, frozen)
    assert jnp.array_equal(S1, S2)
    key2 = jax.random.key(43)
    S3 = bm.sample_chains(key2, S0, energy.fields(params.h), J, 5, q, frozen)
    assert not jnp.array_equal(S1, S3)


@pytest.mark.slow
def test_sampler_validity_vs_exact():
    # Empirical distribution from Gibbs matches the exact Boltzmann distribution.
    L, q, C = 3, 3, 40000
    params = _tiny_params(L, q, seed=6, scale=0.5)
    J = energy.couplings(params.W)
    from _exact import exact_distribution
    seqs, p_exact = exact_distribution(params.h, J)

    S = jnp.asarray(np.random.default_rng(1).integers(0, q, size=(C, L)).astype(np.int32))
    frozen = jnp.zeros_like(S, dtype=bool)
    S = bm.sample_chains(jax.random.key(7), S, energy.fields(params.h), J, 200, q, frozen)
    S = np.asarray(S)

    # Empirical frequency of each of the q**L states.
    emp = np.zeros(len(seqs))
    key_map = {tuple(s): i for i, s in enumerate(seqs)}
    for s in S:
        emp[key_map[tuple(s)]] += 1
    emp /= C
    assert np.max(np.abs(emp - p_exact)) < 0.02, f"max dev {np.max(np.abs(emp-p_exact)):.4f}"
