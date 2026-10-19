# Copyright (C) 2026  Shichao Wu
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Whether axis-aligned boxes meet an ellipsoid, decided exactly.

The region is ``R = {x : x^T M x <= 1}`` with ``M = A^T A`` positive
semi-definite, and a box is ``B(c, h) = prod_a [c_a - h_a, c_a + h_a]``.

    B meets R   <=>   min_{x in B} x^T M x <= 1.

``support_inside`` is the cheap sufficient test ``|A c| - sum_a |A_a| h_a
<= 1``: it never drops a box that meets ``R`` but keeps many that do not.

The exact minimum. With the other coordinates frozen, ``q(x) = x^T M x`` is
a parabola in ``x_a``, minimised over its interval by

    x_a  <-  clip(-r_a(x) / m_a, c_a - h_a, c_a + h_a),
    m_a = M_aa,  r_a = sum_{b != a} M_ab x_b.                          (1)

For a diagonal ``M`` one pass is the minimum, ``sum_a m_a max(0, |c_a| -
h_a)^2`` (``diagonal_clamp``). For a tilted ellipsoid it is not: the minimum
is the fixed point of (1) over the axes, which is the KKT point of a convex
quadratic over a box and hence global. Iterating gives a feasible point and
so an upper bound, the wrong direction for dropping a box; strong convexity
gives the lower bound

    min_B q  >=  q(x) + sum_a min_{d in [lo_a - x_a, hi_a - x_a]}
                                 (lam d^2 + g_a d),                    (2)

with ``g = 2 M x`` and ``lam`` the smallest eigenvalue of ``M`` on its range
(a linear bound where it is zero). A box is dropped only when (2) exceeds
one, so a dropped box provably misses ``R``; a box the bound leaves
undecided is kept. ``inside_exact`` is that test; ``active_set_min`` the
brute force it is checked against.
"""
import itertools

import numpy as np


# the quadratic form
def metric(A):
    """The quadratic form ``M = A^T A`` of ``{x : |A x| <= 1}``."""
    A = np.asarray(A, float)
    return A.T @ A


def qform(M, x):
    """``x^T M x`` for each row of ``x``."""
    return np.einsum('ni,ij,nj->n', x, M, x)


def support_inside(A, c, half):
    """The sufficient test ``|A c| - sum_a |A_a| h_a <= 1``."""
    Ac = np.atleast_2d(np.asarray(c, float)) @ np.asarray(A, float).T
    An = np.linalg.norm(np.asarray(A, float), axis=0)
    h = np.asarray(half, float)
    return (np.linalg.norm(Ac, axis=1) - np.atleast_2d(h) @ An) <= 1.0


def _half_matrix(half, shape):
    h = np.asarray(half, float)
    if h.ndim == 1:
        h = np.broadcast_to(h, shape)
    return np.ascontiguousarray(h, dtype=float)


def _active(M):
    return np.nonzero(np.diag(M) > 0.0)[0]


# the clamped iteration and the certified bounds
def box_min_q(c, half, M, sweeps=64, x0=None):
    """The clamped coordinate iteration (1) on every box.

    ``c`` is ``(N, n)``, ``half`` is ``(n,)`` or ``(N, n)``; one sweep is
    ``n`` array operations over all boxes. Returns ``(x, lo, hi)``.
    """
    c = np.atleast_2d(np.asarray(c, float))
    h = _half_matrix(half, c.shape)
    lo, hi = c - h, c + h
    x = c.copy() if x0 is None else np.clip(np.asarray(x0, float), lo, hi)
    act = _active(M)
    for _ in range(sweeps):
        for a in act:
            r = x @ M[a] - M[a, a] * x[:, a]
            x[:, a] = np.clip(-r / M[a, a], lo[:, a], hi[:, a])
    return x, lo, hi


def bounds_at(x, lo, hi, M, lam):
    """``q(x)``, an upper bound on the minimum, and the lower bound (2)."""
    act = _active(M)
    q = qform(M, x)
    if act.size == 0:
        return q, q.copy()
    g = 2.0 * (x @ M)
    lb = q.copy()
    for a in act:
        ga = g[:, a]
        if lam > 0:
            d = np.clip(
                -ga / (2.0 * lam), lo[:, a] - x[:, a], hi[:, a] - x[:, a]
            )
        else:
            # Convexity still supplies a linear lower bound when M is singular.
            d = np.where(ga >= 0, lo[:, a] - x[:, a], hi[:, a] - x[:, a])
        lb += lam * d * d + ga * d
    return q, lb


def interval_bounds(c, half, M, sweeps=64, lam=None, x0=None):
    """``q`` at the clamped fixed point ``x`` and the lower bound (2).

    Returns ``(q_ub, q_lb, x)`` with ``q_lb <= min_B q <= q_ub``.
    """
    c = np.atleast_2d(np.asarray(c, float))
    h = _half_matrix(half, c.shape)
    x, lo, hi = box_min_q(c, h, M, sweeps=sweeps, x0=x0)
    if lam is None:
        lam = smallest_eig(M)
    q, lb = bounds_at(x, lo, hi, M, lam)
    return q, lb, x


def smallest_eig(M):
    act = _active(M)
    if act.size == 0:
        return 0.0
    # A roundoff-negative eigenvalue of A.T @ A supplies no positive curvature.
    return max(0.0, float(np.linalg.eigvalsh(M[np.ix_(act, act)])[0]))


def certified(c, half, M, tol=1e-13, block=32, max_sweeps=4096, lam=None):
    """Certified decisions, iterating only the boxes not yet resolved.

    Returns ``dict(keep, keep_value, q_ub, q_lb, unresolved, sweeps,
    n_disputed)``:

    * ``keep``: every False has a proof that the box misses the ellipsoid.
    * ``keep_value``: ``q(x) <= 1`` at the feasible point; it implies
      ``keep``.
    * ``n_disputed``: boxes where the two differ, the bound not yet tight.
    """
    c = np.atleast_2d(np.asarray(c, float))
    h = _half_matrix(half, c.shape)
    M = np.asarray(M, float)
    if M.shape != (c.shape[1], c.shape[1]) or np.any(h < 0):
        raise ValueError("invalid box metric or half width")
    if not np.all(np.isfinite(M)):
        unresolved = np.ones(len(c), dtype=bool)
        return dict(
            keep=unresolved.copy(),
            keep_value=~unresolved,
            q_ub=np.full(len(c), np.nan),
            q_lb=np.full(len(c), np.nan),
            unresolved=unresolved,
            sweeps=0,
            n_disputed=len(c),
        )
    if not np.allclose(M, M.T, rtol=0, atol=tol):
        raise ValueError("the metric must be symmetric")
    eigen_min = float(np.linalg.eigvalsh(M)[0])
    if eigen_min < -tol * max(1.0, np.linalg.norm(M, ord=2)):
        raise ValueError("the metric must be positive semidefinite")
    if lam is None:
        lam = smallest_eig(M)
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("invalid curvature lower bound")
    if lam > smallest_eig(M) + tol:
        raise ValueError("curvature exceeds the smallest metric eigenvalue")
    q_ub, q_lb, x = interval_bounds(c, h, M, sweeps=block, lam=lam)
    used = block
    todo = np.nonzero(q_ub - q_lb > tol)[0]
    while todo.size and used < max_sweeps:
        x[todo], lo, hi = box_min_q(
            c[todo], h[todo], M, sweeps=block, x0=x[todo]
        )
        q_ub[todo], q_lb[todo] = bounds_at(x[todo], lo, hi, M, lam)
        used += block
        todo = np.nonzero(q_ub - q_lb > tol)[0]
    unresolved = ~np.isfinite(q_lb) | ~np.isfinite(q_ub)
    # An unresolved lower bound cannot certify that a box is outside.
    keep = unresolved | (q_lb <= 1.0 + tol)
    keep_value = q_ub <= 1.0
    return dict(
        keep=keep,
        keep_value=keep_value,
        q_ub=q_ub,
        q_lb=q_lb,
        unresolved=unresolved,
        sweeps=int(used),
        n_disputed=int(np.sum(keep != keep_value)),
    )


def inside_exact(A, c, half, **kw):
    """True where the box meets ``{x : |A x| <= 1}``, certified."""
    return certified(c, half, metric(A), **kw)['keep']


# the closed form for a diagonal metric, and an exact solver to check against
def diagonal_clamp(c, half, M):
    """The one-pass clamping minimum, exact only for a diagonal ``M``."""
    c = np.atleast_2d(np.asarray(c, float))
    h = _half_matrix(half, c.shape)
    out = np.maximum(np.abs(c) - h, 0.0)
    return np.einsum('ni,i,ni->n', out, np.diag(M), out)


def active_set_min(c, half, M):
    """The exact minimum by enumerating every active set.

    For each pattern in ``{-1, 0, +1}^n`` (free, on the lower bound, on the
    upper) the KKT system is solved on the free coordinates, and the
    candidate kept when it is feasible with multipliers of the right sign.
    ``3^n`` patterns: for checking on a few thousand boxes.
    """
    c = np.atleast_2d(np.asarray(c, float))
    h = _half_matrix(half, c.shape)
    lo, hi = c - h, c + h
    n = c.shape[1]
    best = np.full(len(c), np.inf)
    for pat in itertools.product((-1, 0, 1), repeat=n):
        pat = np.asarray(pat)
        B = np.nonzero(pat != 0)[0]
        F = np.nonzero(pat == 0)[0]
        x = c.copy()
        x[:, B] = np.where(pat[B] < 0, lo[:, B], hi[:, B])
        if F.size:
            rhs = -(x[:, B] @ M[np.ix_(B, F)])
            sol = np.linalg.solve(M[np.ix_(F, F)], rhs.T).T
            if not np.all(np.isfinite(sol)):
                continue
            x[:, F] = sol
            ok = np.all((sol >= lo[:, F]) & (sol <= hi[:, F]), axis=1)
        else:
            ok = np.ones(len(c), bool)
        g = 2.0 * (x @ M)
        if B.size:
            ok = ok & np.all(
                np.where(pat[B] < 0, g[:, B] >= 0.0, g[:, B] <= 0.0), axis=1
            )
        q = qform(M, x)
        best = np.where(ok & (q < best), q, best)
    return best
