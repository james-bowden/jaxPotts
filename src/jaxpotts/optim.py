"""Optimizer wrappers for full-batch convex fitting.

Provides a single :func:`minimize` entry point over a value-and-grad closure that
returns ``(value, grad_pytree)``. Methods:

- ``"lbfgs"``  -- ``optax.lbfgs`` with a zoom line search (default; the plm problem
  is convex and this is fastest).
- ``"cg"``     -- nonlinear conjugate gradient (Fletcher-Reeves) with backtracking,
  a close analogue of CCMpred's optimizer for parity debugging (D-007).
- ``"adam"``   -- plain Adam, for experimentation.

Every method logs the objective per iteration so convergence can be plotted.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import optax


def _tree_vdot(a, b) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x, y: jnp.vdot(x, y), a, b))
    return sum(leaves)


def minimize(
    value_and_grad: Callable,
    params,
    method: str = "lbfgs",
    maxiter: int = 250,
    tol: float = 1e-5,
    learning_rate: float = 1e-3,
    verbose: bool = False,
) -> tuple[object, list[float]]:
    """Minimise ``value_and_grad`` starting from ``params``.

    Returns ``(params, history)`` where ``history`` is the list of objective values,
    one per iteration.
    """
    if method == "lbfgs":
        return _lbfgs(value_and_grad, params, maxiter, tol, verbose)
    if method == "cg":
        return _cg(value_and_grad, params, maxiter, tol, verbose)
    if method == "adam":
        return _adam(value_and_grad, params, maxiter, tol, learning_rate, verbose)
    raise ValueError(f"unknown method {method!r}")


def _run(step, params, state, maxiter, tol, verbose, tag, print_every):
    """Drive a jitted ``step(params, state) -> (params, state, value)`` to convergence.

    Stops on max iterations or when the relative change in the objective falls below
    ``tol``. Returns ``(params, history)`` with the objective value per iteration.
    """
    history: list[float] = []
    prev = jnp.inf
    for it in range(maxiter):
        params, state, value = step(params, state)
        v = float(value)
        history.append(v)
        if verbose and (it % print_every == 0 or it == maxiter - 1):
            print(f"  [{tag}] iter {it:4d}  f = {v:.6f}")
        if jnp.isfinite(prev) and abs(prev - v) <= tol * max(1.0, abs(prev)):
            break
        prev = v
    return params, history


def _lbfgs(value_and_grad, params, maxiter, tol, verbose):
    opt = optax.lbfgs()

    @jax.jit
    def step(params, state):
        value, grad = value_and_grad(params)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=lambda p: value_and_grad(p)[0]
        )
        return optax.apply_updates(params, updates), state, value

    return _run(step, params, opt.init(params), maxiter, tol, verbose, "lbfgs", 10)


def _cg(value_and_grad, params, maxiter, tol, verbose):
    """Fletcher-Reeves nonlinear CG with backtracking Armijo line search."""
    value, grad = value_and_grad(params)
    direction = jax.tree_util.tree_map(lambda g: -g, grad)
    gg = _tree_vdot(grad, grad)
    history = [float(value)]
    prev = float(value)

    @jax.jit
    def line_search(params, direction, value, grad):
        slope = _tree_vdot(grad, direction)

        def f_at(alpha):
            step = jax.tree_util.tree_map(lambda d: alpha * d, direction)
            return value_and_grad(optax.apply_updates(params, step))[0]

        # Backtracking: halve alpha until the Armijo condition (ftol=1e-4) holds.
        def cond(state):
            alpha, fval = state
            return (fval > value + 1e-4 * alpha * slope) & (alpha > 1e-12)

        def shrink(state):
            alpha = state[0] * 0.5
            return alpha, f_at(alpha)

        state = jax.lax.while_loop(cond, shrink, (1.0, f_at(1.0)))
        return state[0]

    for it in range(maxiter):
        alpha = float(line_search(params, direction, value, grad))
        params = optax.apply_updates(params, jax.tree_util.tree_map(lambda d: alpha * d, direction))
        value, new_grad = value_and_grad(params)
        new_gg = _tree_vdot(new_grad, new_grad)
        beta = new_gg / (gg + 1e-30)
        direction = jax.tree_util.tree_map(lambda d, g: -g + beta * d, direction, new_grad)
        grad, gg = new_grad, new_gg
        v = float(value)
        history.append(v)
        if verbose and (it % 10 == 0 or it == maxiter - 1):
            print(f"  [cg] iter {it:4d}  f = {v:.6f}")
        if abs(prev - v) <= tol * max(1.0, abs(prev)):
            break
        prev = v
    return params, history


def _adam(value_and_grad, params, maxiter, tol, learning_rate, verbose):
    opt = optax.adam(learning_rate)

    @jax.jit
    def step(params, state):
        value, grad = value_and_grad(params)
        updates, state = opt.update(grad, state, params)
        return optax.apply_updates(params, updates), state, value

    return _run(step, params, opt.init(params), maxiter, tol, verbose, "adam", 50)
