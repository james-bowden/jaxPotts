"""Tests for zero-sum gauge, Frobenius norm, and APC (docs/conventions.md §8)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxpotts import energy, gauge

jax.config.update("jax_enable_x64", True)


def _random_params(L, q, seed=0):
    rng = np.random.default_rng(seed)
    h = jnp.asarray(rng.normal(size=(L, q)))
    h = h.at[:, gauge.GAP].set(0.0)
    W = jnp.asarray(rng.normal(size=(L, q, L, q)))
    J = energy.couplings(W)
    return h, J


def test_zsg_hand_computed_2x2():
    # One pair, q=2, block = identity [[1,0],[0,1]] -> double-centred [[.5,-.5],[-.5,.5]].
    L, q = 2, 2
    J = jnp.zeros((L, q, L, q))
    M = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    J = J.at[0, :, 1, :].set(M)
    J = J.at[1, :, 0, :].set(M.T)
    h = jnp.zeros((L, q))
    _, J_zs = gauge.zero_sum_gauge(h, J)
    expected = jnp.array([[0.5, -0.5], [-0.5, 0.5]])
    assert jnp.allclose(J_zs[0, :, 1, :], expected)
    # Row and column sums of every block vanish.
    assert jnp.allclose(J_zs.sum(axis=1), 0.0, atol=1e-12)
    assert jnp.allclose(J_zs.sum(axis=3), 0.0, atol=1e-12)


def test_zsg_idempotent():
    h, J = _random_params(6, 21, seed=1)
    h1, J1 = gauge.zero_sum_gauge(h, J)
    h2, J2 = gauge.zero_sum_gauge(h1, J1)
    assert jnp.allclose(J1, J2, atol=1e-10)
    assert jnp.allclose(h1, h2, atol=1e-10)


def test_zsg_preserves_energy_differences():
    L, q = 5, 21
    h, J = _random_params(L, q, seed=2)
    h_zs, J_zs = gauge.zero_sum_gauge(h, J)
    # Random sequences, including gaps, one-hot encoded.
    rng = np.random.default_rng(3)
    A = rng.integers(0, q, size=(20, L))
    X = jnp.asarray(np.eye(q)[A])
    E_old = energy.sequence_energy(h, J, X)
    E_new = energy.sequence_energy(h_zs, J_zs, X)
    diff = E_new - E_old
    # E changes only by a global constant -> differences are preserved.
    assert jnp.allclose(diff - diff[0], 0.0, atol=1e-9)


def test_zsg_pins_gap_field():
    h, J = _random_params(4, 21, seed=4)
    h_zs, _ = gauge.zero_sum_gauge(h, J)
    assert jnp.allclose(h_zs[:, gauge.GAP], 0.0, atol=1e-12)


def test_frobenius_gap_inclusion():
    h, J = _random_params(4, 21, seed=5)
    _, J_zs = gauge.zero_sum_gauge(h, J)
    F_all = gauge.frobenius_norm(J_zs, include_gap=True)
    F_nogap = gauge.frobenius_norm(J_zs, include_gap=False)
    # Excluding a state can only reduce the summed squares.
    assert jnp.all(F_nogap <= F_all + 1e-9)
    assert F_all.shape == (4, 4)


def test_apc_hand_computed_3x3():
    F = jnp.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    S = gauge.apc(F)
    # row means [1, 4/3, 5/3], total 4/3; S[0,1]=1-1=0, S[0,2]=2-5/4=0.75, S[1,2]=3-5/3.
    assert jnp.isclose(S[0, 1], 0.0)
    assert jnp.isclose(S[0, 2], 0.75)
    assert jnp.isclose(S[1, 2], 3.0 - 5.0 / 3.0)
    assert jnp.isclose(S[0, 0], 0.0)  # diagonal stays zero


def test_score_symmetric():
    h, J = _random_params(7, 21, seed=6)
    S = gauge.contact_score(h, J)
    assert jnp.allclose(S, S.T, atol=1e-10)
    assert jnp.allclose(jnp.diag(S), 0.0)


def test_ccmpred_score_reproduces_reference_mat():
    # Our CCMpred-replica scoring must reproduce CCMpred's own .mat exactly from its
    # raw couplings. Requires the reference repo build + a pre-computed run.
    import pathlib

    import numpy as np

    from jaxpotts import reference

    sc = pathlib.Path("/tmp/claude-3469/-home-jcbowden-jaxPotts/"
                      "4f78d7a7-03e8-4b54-b9a6-83247d8ddb0f/scratchpad")
    raw, mat = sc / "1atzA.ccmpred.tight.raw", sc / "1atzA.ccmpred.tight.mat"
    if not raw.exists() or not mat.exists():
        import pytest

        pytest.skip("CCMpred reference run not available")
    _, J = reference.parse_ccmpred_raw(raw)
    S = np.asarray(gauge.ccmpred_score(J))
    M = reference.read_score_matrix(mat)
    iu = np.triu_indices(J.shape[0], k=1)
    assert np.max(np.abs(S[iu] - M[iu])) < 1e-4
