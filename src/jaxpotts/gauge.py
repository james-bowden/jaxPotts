"""Gauge fixing, Frobenius norms, and APC contact scores.

See ``docs/conventions.md`` §8 and ``docs/decisions.md`` D-008.

``jaxPotts`` uses the standard **zero-sum gauge (ZSG)** by double-centering as the
canonical basis for comparing raw ``J`` between implementations. The references
instead subtract a single scalar mean per block for their *scores*; those
reference score pipelines are reproduced exactly in :func:`ccmpred_score` and
:func:`ccmpredpy_score` for parity checks, and must not be confused with the
canonical ZSG.
"""

from __future__ import annotations

import jax.numpy as jnp

GAP = 20


def zero_sum_gauge(h: jnp.ndarray, J: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Transform ``(h, J)`` to the zero-sum gauge, leaving ``P(x)`` unchanged.

    Double-centres each ``(q, q)`` coupling block::

        J_zs[i,a,j,b] = J[i,a,j,b] - mean_a - mean_b + mean_ab

    and applies the compensating field correction so that all energy *differences*
    are preserved (D-008). The transform is idempotent.

    Returns ``(h_zs, J_zs)`` with the same shapes as the inputs.
    """
    rowmean = J.mean(axis=3)            # R[i,a,j]  = mean_b J
    allmean = J.mean(axis=(1, 3))       # T[i,j]    = mean_ab J
    colmean = J.mean(axis=1)            # C[i,j,b]  = mean_a J

    J_zs = (
        J
        - rowmean[:, :, :, None]
        - colmean[:, None, :, :]
        + allmean[:, None, :, None]
    )

    # Field correction: h_zs[i,a] = h[i,a] + sum_{j} (R[i,a,j] - 0.5 T[i,j]).
    # (Diagonal j==i contributes 0 because J has zero diagonal blocks.)
    c = rowmean - 0.5 * allmean[:, None, :]     # c[i,a,j]
    h_zs = h + c.sum(axis=2)
    # Re-pin the gap field to 0 with a per-site constant shift, which leaves P(x)
    # (and all energy differences) unchanged. (Only for the standard q=21 alphabet;
    # smaller-q test models have no designated gap column.)
    if h_zs.shape[1] > GAP:
        h_zs = h_zs - h_zs[:, GAP:GAP + 1]
    return h_zs, J_zs


def frobenius_norm(J: jnp.ndarray, include_gap: bool = True) -> jnp.ndarray:
    """Per-pair Frobenius norm ``F[i,j] = sqrt(sum_{a,b} J[i,a,j,b]^2)``.

    ``include_gap=True`` sums over all 21 states (CCMpredPy convention);
    ``include_gap=False`` excludes the gap state 20 (CCMpred/C convention).
    """
    if not include_gap:
        J = J[:, :GAP, :, :GAP]
    return jnp.sqrt(jnp.sum(J * J, axis=(1, 3)))


def apc(F: jnp.ndarray, include_diagonal: bool = True) -> jnp.ndarray:
    """Average product correction ``S[i,j] = F[i,j] - F_i. F_.j / F..`` (Dunn 2004).

    The diagonal of ``F`` is zeroed first (both references zero the score diagonal).
    ``include_diagonal=True`` (both references) takes the row/column/total means over
    all entries, including the now-zeroed diagonal.
    """
    L = F.shape[0]
    off = ~jnp.eye(L, dtype=bool)
    F = F * off
    if include_diagonal:
        row = F.mean(axis=1)
        col = F.mean(axis=0)
        total = F.mean()
    else:
        row = (F * off).sum(axis=1) / off.sum(axis=1)
        col = (F * off).sum(axis=0) / off.sum(axis=0)
        total = (F * off).sum() / off.sum()
    S = F - jnp.outer(row, col) / total
    return S * off


def contact_score(
    h: jnp.ndarray,
    J: jnp.ndarray,
    include_gap: bool = False,
    apply_apc: bool = True,
) -> jnp.ndarray:
    """Canonical ``jaxPotts`` contact score: ZSG -> Frobenius -> APC.

    Returns an ``(L, L)`` score matrix with zero diagonal. ``include_gap`` and
    ``apply_apc`` select the norm/correction; defaults exclude the gap state.
    """
    _, J_zs = zero_sum_gauge(h, J)
    F = frobenius_norm(J_zs, include_gap=include_gap)
    return apc(F) if apply_apc else F * (~jnp.eye(F.shape[0], dtype=bool))


def ccmpred_score(J: jnp.ndarray, apply_apc: bool = True) -> jnp.ndarray:
    """Exact replica of CCMpred's (C) score pipeline for parity checks (D-008).

    Per pair, subtract a single scalar mean over the full 21x21 block, take the
    Frobenius norm over the 20x20 non-gap block (**gap excluded**), zero the
    diagonal, then APC over all entries.
    """
    block_mean = J.mean(axis=(1, 3), keepdims=True)     # scalar mean over 21x21
    centred = J - block_mean
    F = jnp.sqrt(jnp.sum(centred[:, :GAP, :, :GAP] ** 2, axis=(1, 3)))
    L = F.shape[0]
    off = ~jnp.eye(L, dtype=bool)
    F = F * off
    if not apply_apc:
        return F
    S = apc(F)
    # CCMpred shifts scores by the min upper-triangle off-diagonal, then zeroes the
    # diagonal (util.c:89-105), so all scores are >= 0.
    min_off = jnp.min(jnp.where(jnp.triu(off, k=1), S, jnp.inf))
    return (S - min_off) * off


def ccmpredpy_score(J: jnp.ndarray, apply_apc: bool = False) -> jnp.ndarray:
    """Exact replica of CCMpredPy's score pipeline for parity checks (D-008).

    Scalar-centre the 20x20 non-gap block (``recenter_potentials``), take the
    Frobenius norm over the full 21x21 block (**gap included**, as coded), zero the
    diagonal. CCMpredPy's default ``-m`` matrix is this Frobenius score **without**
    APC (APC is written to a separate ``--apc`` file), so ``apply_apc`` defaults to
    ``False`` and reproduces the ``.mat`` exactly.
    """
    J = jnp.asarray(J)
    block = J[:, :GAP, :, :GAP]
    block_mean = block.mean(axis=(1, 3), keepdims=True)
    J = J.at[:, :GAP, :, :GAP].add(-block_mean)
    F = frobenius_norm(J, include_gap=True)
    L = F.shape[0]
    F = F * (~jnp.eye(L, dtype=bool))
    return apc(F) if apply_apc else F
