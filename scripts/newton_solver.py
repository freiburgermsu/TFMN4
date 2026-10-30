"""
newton_solver - converge the repo's JAX fits (s12, s12b, s13) with damped Newton steps.

The fits are first run with L-BFGS, which stops at an iteration cap. On these
softmax-replicator losses the cap can leave |grad| anywhere from 1e-2 to 1e3, and where it
stops depends on rounding: CPU vs GPU, library versions, even two runs on one GPU give
values that differ by ~0.015/h. Newton steps on the exact Hessian -- built from batched
Hessian-vector products, Cholesky-factorised (damped only when not positive definite), with
an Armijo backtracking line search -- reach |grad| < 1e-6 in a handful of steps and land on
the same optimum on any platform.

Memory: the dense n x n Hessian is written in place (donated buffers), so a solve holds about
two n x n float64 matrices plus ~`hvp_bytes` of batched-HVP work space (n = 11,931 -> 8.6 GiB
of GPU memory for the whole process).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

GRAD_TOL = 1e-6        # convergence: Euclidean norm of the gradient


class NewtonSolver:
    """Minimise `loss(x, *args)` over a float64 vector x of length `n`.

    Build one solver per loss: its jitted value / gradient / batched-HVP functions are compiled
    once and reused by every `solve` (s12b refits the same loss ~70 times with different data
    passed through `args`). `rows` = number of data rows the loss touches (sizes HVP batches).
    """

    def __init__(self, loss, n, rows, hvp_bytes=1e9):
        self.n = int(n)
        grad = jax.grad(loss)
        self.f = jax.jit(loss)
        self.g = jax.jit(grad)
        self.hvb = jax.jit(jax.vmap(lambda x, v, a: jax.jvp(lambda q: grad(q, *a), (x,), (v,))[1],
                                    in_axes=(None, 0, None)))
        self.chunk = int(max(32, min(1024, hvp_bytes / (max(int(rows), 1) * 8 * 12))))
        idx = jnp.arange(self.n)
        self._put = jax.jit(lambda H, blk, s0: jax.lax.dynamic_update_slice(H, blk, (s0, 0)),
                            donate_argnums=0)
        self._add_diag = jax.jit(lambda H, d: H.at[idx, idx].add(d), donate_argnums=0)
        self._chol = jax.jit(lambda H: jax.lax.linalg.cholesky(H, symmetrize_input=False))

    def hessian(self, x, args=()):
        """Dense Hessian at x, row block by row block (lower triangle is what Cholesky reads)."""
        x = jnp.asarray(x, dtype=jnp.float64)
        H = jnp.zeros((self.n, self.n))
        for s0 in range(0, self.n, self.chunk):
            k = min(self.chunk, self.n - s0)
            E = jnp.zeros((k, self.n)).at[jnp.arange(k), jnp.arange(s0, s0 + k)].set(1.0)
            H = self._put(H, self.hvb(x, E, tuple(args)), s0)
        return H

    def grad_norm(self, x, args=()):
        return float(jnp.linalg.norm(self.g(jnp.asarray(x, dtype=jnp.float64), *args)))

    def solve(self, x0, args=(), tol=GRAD_TOL, max_iter=60):
        """Newton from x0 until |grad| < tol. Returns (x as numpy array, final |grad|)."""
        x = jnp.asarray(x0, dtype=jnp.float64)
        for _ in range(max_iter):
            gx = self.g(x, *args)
            if float(jnp.linalg.norm(gx)) < tol:
                break
            H = self.hessian(x, args)
            L = None; added = 0.0
            for damp in (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2):   # damp only if not pos. def.
                if damp > added:
                    H = self._add_diag(H, damp - added); added = damp
                Lc = self._chol(H)
                if bool(jnp.all(jnp.isfinite(Lc))):
                    L = Lc
                    break
                del Lc
            del H
            if L is None:
                break
            step = -jax.scipy.linalg.cho_solve((L, True), gx)
            del L
            f0 = float(self.f(x, *args)); slope = float(gx @ step); t = 1.0
            while t > 1e-6 and float(self.f(x + t * step, *args)) > f0 + 1e-4 * t * slope:
                t *= 0.5
            x = x + t * step
        return np.asarray(x), self.grad_norm(x, args)
