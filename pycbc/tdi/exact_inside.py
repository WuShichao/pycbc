"""Exact box-ellipsoid intersection for the AMR containment test (SSB_EXACT_INSIDE=1).

The registered test
-------------------
``Geometry.inside`` keeps a cell if

    ||A c|| - sum_a |A_a| (w_a / 2) <= 1,                                     (S)

the support function of the cell box evaluated at the direction ``A c``: a lower bound on
``min_{x in B} ||A x||``, hence a *sufficient* condition for the cell to meet the region.
It is safe -- it never drops a cell that meets the region -- and it is loose: on the
registered level-2 grid it keeps 45,064 cells where 10,692 meet the ellipsoid.

The geometry
------------
``eta`` are the coordinates in which the mismatch is the Euclidean norm,
``eta_a = sqrt(lam_a) (E^T xi)_a``, and the region is the ellipsoid

    R = {eta : eta^T M eta <= 1},   M = A^T A,   A = diag(1/s) E diag(1/sqrt(lam)),

so ``A eta = xi / s`` and ``R`` is the affine image of the unit ball.  The cell is the
axis-aligned box ``B(c, h) = prod_a [c_a - h_a, c_a + h_a]`` with ``h = w / 2``, and

    B meets R   <=>   min_{x in B} x^T M x <= 1.

The formula: clamping
---------------------
Write ``m_a = M_aa`` and ``r_a(x) = sum_{b != a} M_ab x_b``.  With the other coordinates
frozen, ``q(x) = x^T M x`` is a parabola in ``x_a`` and its exact minimiser over the
interval is

    x_a  <-  clip( -r_a(x) / m_a ,  c_a - h_a ,  c_a + h_a ),                 (1)

one division and one clip, because ``dq/dx_a = 2 (m_a x_a + r_a)``.

* If ``M`` is diagonal -- an untilted ellipsoid, the case where the ellipsoid's metric is
  the identity in eta up to a rescaling of each axis -- then ``r_a = 0`` and one pass of
  (1) is the exact joint minimiser, so

      min_{x in B} q  =  sum_a m_a * max(0, |c_a| - h_a)^2 ,                   (2)

  which is the closed form by clamping: clamp the origin into the box and evaluate.
  ``diagonal_clamp`` below is (2).

* If ``M`` is not diagonal, (2) is **not** the minimum.  The joint minimiser is the fixed
  point of (1) iterated over the axes, and that fixed point is the exact global minimum:
  ``q`` is strictly convex on its range and the box is convex, so coordinate-wise
  optimality is the KKT system, and KKT is sufficient for a convex quadratic over a box.
  There is no single-pass closed form in that case -- the active set (which coordinates
  sit on a bound) is not known before solving.

  This geometry is strongly coupled: ``max |M_ab| / sqrt(M_aa M_bb) = 0.84`` and the
  ellipsoid is genuinely tilted.  Using (2) here keeps 2,464 cells instead of 10,692 at
  level 2, i.e. it drops 8,228 cells that meet the region, and the cover breaks.  The
  coupled iteration is therefore not a refinement of the test, it *is* the test.

The certificate
---------------
Iterating (1) produces a feasible point ``x`` and hence an *upper* bound ``q(x)`` on the
minimum, which is the wrong direction for deciding to drop a cell.  The lower bound comes
from strong convexity: with ``g = 2 M x`` and ``lam`` the smallest eigenvalue of ``M`` on
its range,

    q(y) >= q(x) + g . (y - x) + lam ||y - x||^2        for every y,            (3)

and the right-hand side is separable over the box, so

    min_{y in B} q  >=  q(x) + sum_a min_{d in [lo_a - x_a, hi_a - x_a]}
                                        ( lam d^2 + g_a d ),                   (4)

where each inner minimum is the same closed form (1): ``d = clip(-g_a / (2 lam), ...)``.
At the exact KKT point every term of the sum is zero and (4) equals the value.  ``keep``
is decided by the *lower* bound, so a dropped cell is certified to lie strictly outside
the ellipsoid:

    drop   <=>   (4) > 1 .

A cell the bound leaves uncertain is kept, which is the safe direction.  In this geometry
no cell is uncertain: the value test and the certified test return the same count, with
zero cells in dispute.

Nothing here needs a GPU.  numpy only.
"""
import itertools

import numpy as np


# ---------------------------------------------------------------------------------------------
# the quadratic form
# ---------------------------------------------------------------------------------------------
def metric(A):
    """The quadratic form ``M = A^T A`` of the region ``{eta : ||A eta|| <= 1}``."""
    A = np.asarray(A, float)
    return A.T @ A


def qform(M, x):
    """``x^T M x`` for each row of ``x``."""
    return np.einsum('ni,ij,nj->n', x, M, x)


def support_inside(A, c, half):
    """The registered test (S), for reference: ``||A c|| - sum_a |A_a| h_a <= 1``."""
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


# ---------------------------------------------------------------------------------------------
# the clamped iteration and the certified bounds
# ---------------------------------------------------------------------------------------------
def box_min_q(c, half, M, sweeps=64, x0=None):
    """The clamped coordinate iteration (1) on every box.

    ``c`` is ``(N, n)``, ``half`` is ``(n,)`` or ``(N, n)``.  Vectorised over boxes: one
    sweep is ``n`` array operations.  Returns ``(x, lo, hi)``.
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
    """``q(x)`` (an upper bound on the minimum) and the rigorous lower bound (4)."""
    act = _active(M)
    q = qform(M, x)
    if act.size == 0:
        return q, q.copy()
    g = 2.0 * (x @ M)
    lb = q.copy()
    for a in act:
        ga = g[:, a]
        if lam > 0:
            d = np.clip(-ga / (2.0 * lam), lo[:, a] - x[:, a], hi[:, a] - x[:, a])
        else:
            # Convexity still supplies a linear lower bound when M is singular.
            d = np.where(ga >= 0, lo[:, a] - x[:, a], hi[:, a] - x[:, a])
        lb += lam * d * d + ga * d
    return q, lb


def interval_bounds(c, half, M, sweeps=64, lam=None, x0=None):
    """``q`` at the clamped fixed point ``x`` and the rigorous lower bound (4).

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
    """Certified containment decision, refining only the boxes that are not yet resolved.

    Returns ``dict(keep, keep_value, q_ub, q_lb, sweeps, n_disputed)``:

    * ``keep``       -- certified: every False has a proof that the box misses the ellipsoid
    * ``keep_value`` -- ``q(x) <= 1`` at the feasible point; ``keep_value`` implies ``keep``
    * ``n_disputed`` -- cells where the two disagree, i.e. where the bound is not yet tight
    """
    c = np.atleast_2d(np.asarray(c, float))
    h = _half_matrix(half, c.shape)
    M = np.asarray(M, float)
    if M.shape != (c.shape[1], c.shape[1]) or np.any(h < 0):
        raise ValueError("invalid box metric or half width")
    if not np.all(np.isfinite(M)):
        unresolved = np.ones(len(c), dtype=bool)
        return dict(keep=unresolved.copy(), keep_value=~unresolved,
                    q_ub=np.full(len(c), np.nan), q_lb=np.full(len(c), np.nan),
                    unresolved=unresolved, sweeps=0, n_disputed=len(c))
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
        x[todo], lo, hi = box_min_q(c[todo], h[todo], M, sweeps=block, x0=x[todo])
        q_ub[todo], q_lb[todo] = bounds_at(x[todo], lo, hi, M, lam)
        used += block
        todo = np.nonzero(q_ub - q_lb > tol)[0]
    unresolved = ~np.isfinite(q_lb) | ~np.isfinite(q_ub)
    # An unresolved lower bound cannot certify that a box is outside.
    keep = unresolved | (q_lb <= 1.0 + tol)
    keep_value = q_ub <= 1.0
    return dict(keep=keep, keep_value=keep_value, q_ub=q_ub, q_lb=q_lb,
                unresolved=unresolved, sweeps=int(used),
                n_disputed=int(np.sum(keep != keep_value)))


def inside_exact(A, c, half, **kw):
    """Exact containment test: True where the cell box meets ``{eta : ||A eta|| <= 1}``."""
    return certified(c, half, metric(A), **kw)['keep']


# ---------------------------------------------------------------------------------------------
# the closed form for a diagonal metric, and an independent exact solver for verification
# ---------------------------------------------------------------------------------------------
def diagonal_clamp(c, half, M):
    """The closed form (2), valid only when ``M`` is diagonal: the one-pass clamping test.

    Included because it is what "the minimum by clamping" gives when the ellipsoid is not
    tilted, and because the difference from the coupled test is the point of this file.
    """
    c = np.atleast_2d(np.asarray(c, float))
    h = _half_matrix(half, c.shape)
    out = np.maximum(np.abs(c) - h, 0.0)
    return np.einsum('ni,i,ni->n', out, np.diag(M), out)


def active_set_min(c, half, M):
    """The exact minimum by enumerating every active set -- an independent solver.

    For each pattern in ``{-1, 0, +1}^n`` (0 free, -1 on the lower bound, +1 on the upper)
    solve the KKT system on the free coordinates, keep the candidate when it is feasible
    and its multipliers have the right sign, and take the smallest value.  ``3^n``
    patterns; intended for verification on a few thousand boxes.
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
            ok = ok & np.all(np.where(pat[B] < 0, g[:, B] >= 0.0, g[:, B] <= 0.0), axis=1)
        q = qform(M, x)
        best = np.where(ok & (q < best), q, best)
    return best
