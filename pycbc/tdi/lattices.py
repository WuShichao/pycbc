"""The D4 lattice and its three-dimensional bracket A3* (bcc): construction, nearest
point, membership, and a brute-force verification.

The mismatch is Euclidean in eta, m = |eta|^2 (notes/2026/10/2026-10-18-isotropic-cells.md
section 1), so the covering problem is a lattice quantizer problem in that chart and the
object to compare against the axis-aligned box is a lattice, not a box.

D4 is the checkerboard lattice (Conway and Sloane, Sphere Packings, Lattices and Groups,
ch. 4 sec. 7 and ch. 21 sec. 3):

    D4 = {x in Z^4 : sum_a x_a is even}.

In the standard integer scaling used throughout this module its invariants are exact:

    covolume            V = 2
    Voronoi cell        the regular 24-cell, 24 octahedral facets
    covering radius     R = 1                (24 vertices: 8 of (+-1,0,0,0),
                                              16 of (+-1/2,+-1/2,+-1/2,+-1/2))
    inradius            rho = sqrt(2)/2      (minimal vectors (+-1,+-1,0,0), norm^2 = 2)
    second moment       U = int_WS |x|^2 dV = 13/15, mean |x|^2 = 13/30
    quantizer constant  G = (1/n) U / V^(1+2/n) = 13/(120 sqrt 2) = 0.076603
    thickness           Theta = V(B_4(R)) / V = pi^2/4 = 2.467401

The nearest-point map is Conway and Sloane's construction: round to Z^4, and if the
coordinate sum is odd, move the coordinate with the largest rounding error by one step
in the direction of that error.  That is exact for D4 (and for every D_n): the rounded
point is the nearest point of Z^4, so if its parity is even it is the nearest point of
D4; if the parity is odd, exactly one coordinate must be moved and the cheapest one is
the one with the largest error.

A3* (bcc) is the three-dimensional bracket.  Standard scaling: cubic cell of side 1 with
two points, so covolume V = 1/2, covering radius sqrt(5)/4 (R^2 = 5/16), mean |x|^2 =
19/128, G = 19/(192 * 2^(1/3)) = 0.078543, Theta = 1.4635.  Its nearest point is the
nearer of the two cubic cosets, which needs no parity correction.

    python results/2026-10-18/d4_bank/lattices.py
"""
import itertools
import json
import math

import numpy as np

# -- D4, standard integer scaling -----------------------------------------------------------
N4 = 4
D4_COVOL = 2.0
D4_R2 = 1.0                       # squared covering radius
D4_R = 1.0
D4_RHO2 = 0.5                     # squared inradius (packing radius)
D4_U = 13.0 / 15.0                # int_WS |x|^2 dV
D4_MEAN2 = D4_U / D4_COVOL        # 13/30
D4_G = 13.0 / (120.0 * math.sqrt(2.0))
D4_THETA = math.pi ** 2 / 4.0     # = V(B_4(1))/2

# the 24 minimal vectors: the Voronoi-relevant vectors of D4
D4_MIN = np.array([v for v in itertools.product((1.0, -1.0, 0.0), repeat=4)
                   if abs(v[0]) + abs(v[1]) + abs(v[2]) + abs(v[3]) == 2.0])
# the 24 Voronoi vertices (deep holes), all at |x| = 1
D4_HOLES = np.array([[1.0, 0, 0, 0], [-1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, -1.0, 0, 0],
                     [0, 0, 1.0, 0], [0, 0, -1.0, 0], [0, 0, 0, 1.0], [0, 0, 0, -1.0]]
                    + list(itertools.product((0.5, -0.5), repeat=4)))

# -- A3* = bcc, standard integer scaling ----------------------------------------------------
N3 = 3
BCC_COVOL = 0.5
BCC_R2 = 5.0 / 16.0
BCC_R = math.sqrt(BCC_R2)
BCC_MEAN2 = 19.0 / 128.0
BCC_U = BCC_COVOL * BCC_MEAN2
BCC_G = 19.0 / (192.0 * 2.0 ** (1.0 / 3.0))
BCC_THETA = 1.4635                 # Allen's Table I, five figures
# the 24 Voronoi vertices of bcc: every permutation of (+-1/2, +-1/4, 0), all at |x| = R
BCC_HOLES = np.array([[0.5 * a, 0.25 * b, 0.0]
                      for a in (1.0, -1.0) for b in (1.0, -1.0)]
                     + [[0.5 * a, 0.0, 0.25 * b]
                        for a in (1.0, -1.0) for b in (1.0, -1.0)]
                     + [[0.0, 0.5 * a, 0.25 * b]
                        for a in (1.0, -1.0) for b in (1.0, -1.0)])
Z3_THETA = (0.75 * math.pi) ** 1.5 / math.gamma(2.5)   # Allen Eq. (4.2), = 2.7207
Z4_THETA = math.pi ** 2 / 2.0      # = 4.9348
A4S_THETA = 1.7655                 # Allen's Table I, five figures
A4S_G = 0.077559                   # Allen's Table I
Z_G = 1.0 / 12.0


# -- D4 -------------------------------------------------------------------------------------
def d4_members(x, tol=1e-9):
    """True where x is a point of D4."""
    x = np.atleast_2d(np.asarray(x, float))
    near = np.all(np.abs(x - np.rint(x)) <= tol, axis=-1)
    even = (np.rint(x.sum(-1)).astype(np.int64) % 2) == 0
    return near & even


def d4_nearest(y, scale=1.0):
    """Nearest point of ``scale * D4`` to each row of y, by Conway and Sloane's rule.

    ``scale`` is the lattice spacing, i.e. the lattice is scale*D4 with covolume
    2*scale^4.  Returns an array with the same shape as y.
    """
    z = np.asarray(y, float) / scale
    f = np.rint(z)
    e = z - f
    s = np.rint(f.sum(-1)).astype(np.int64)
    rows = np.arange(len(e))
    k = np.argmax(np.abs(e), -1)
    step = np.sign(e[rows, k])
    # an exact zero error (an integer point of odd parity) may step either way
    step = np.where(step == 0.0, 1.0, step)
    g = f.copy()
    g[rows, k] += step
    out = np.where(((s % 2) != 0)[:, None], g, f)
    return out * scale


def d4_brute(y, span=2):
    """Brute-force nearest D4 point over the integer window round(y) +- span.

    Returns (point, distance, number of lattice points at that distance).
    """
    y = np.atleast_2d(np.asarray(y, float))
    off = np.array(list(itertools.product(range(-span, span + 1), repeat=4)), float)
    cand = np.rint(y)[:, None, :] + off[None, :, :]
    par = (np.rint(cand.sum(-1)).astype(np.int64) % 2) == 0
    d2 = np.sum((cand - y[:, None, :]) ** 2, axis=-1)
    d2 = np.where(par, d2, np.inf)
    i = np.argmin(d2, axis=-1)
    rows = np.arange(len(y))
    best = d2[rows, i]
    ties = np.sum(np.abs(d2 - best[:, None]) <= 1e-9, axis=-1)
    return cand[rows, i], np.sqrt(np.maximum(best, 0.0)), ties


def d4_in_voronoi(x, tol=1e-12):
    """True where x lies in the Voronoi cell of the origin: x . v <= |v|^2/2 for every
    minimal vector v, i.e. |x . v| <= 1."""
    x = np.atleast_2d(np.asarray(x, float))
    return np.all(np.abs(x @ D4_MIN.T) <= 0.5 * np.sum(D4_MIN ** 2, -1)[None, :] + tol, axis=-1)


# -- bcc ------------------------------------------------------------------------------------
def bcc_members(x, tol=1e-9):
    """True where x is a point of the bcc lattice Z^3 u (Z^3 + (1/2,1/2,1/2))."""
    x = np.atleast_2d(np.asarray(x, float))
    a = np.all(np.abs(x - np.rint(x)) <= tol, axis=-1)
    h = x - 0.5
    b = np.all(np.abs(h - np.rint(h)) <= tol, axis=-1)
    return a | b


def bcc_nearest(y, scale=1.0):
    """Nearest point of ``scale * A3*``: the nearer of the two cubic cosets."""
    z = np.asarray(y, float) / scale
    a = np.rint(z)
    b = np.rint(z - 0.5) + 0.5
    da = np.sum((z - a) ** 2, -1)
    db = np.sum((z - b) ** 2, -1)
    out = np.where((da <= db)[:, None], a, b)
    return out * scale


def bcc_brute(y, span=2):
    """Brute-force nearest bcc point over both cosets in the window round(y) +- span."""
    y = np.atleast_2d(np.asarray(y, float))
    off = np.array(list(itertools.product(range(-span, span + 1), repeat=3)), float)
    f = np.rint(y)
    ca = f[:, None, :] + off[None, :, :]
    cb = ca + 0.5
    cand = np.concatenate([ca, cb], axis=1)
    d2 = np.sum((cand - y[:, None, :]) ** 2, axis=-1)
    i = np.argmin(d2, axis=-1)
    rows = np.arange(len(y))
    best = d2[rows, i]
    ties = np.sum(np.abs(d2 - best[:, None]) <= 1e-9, axis=-1)
    return cand[rows, i], np.sqrt(np.maximum(best, 0.0)), ties


# -- Monte Carlo: the moments and the covering radius, from the Voronoi cell ----------------
def mc_ball(n, dim, seed):
    """n points uniform in the unit ball of dimension dim (radius^dim = U)."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, dim))
    g /= np.linalg.norm(g, axis=1)[:, None]
    r = rng.random(n) ** (1.0 / dim)
    return g * r[:, None]


def mc_d4(n=4_000_000, seed=11):
    """Uniform Monte Carlo over the D4 Voronoi cell: mean |x|^2 and the sup reached."""
    x = mc_ball(n, 4, seed)
    keep = d4_in_voronoi(x)
    x = x[keep]
    m2 = np.sum(x ** 2, -1)
    return dict(n_drawn=int(n), n_kept=int(len(x)), volume=float(D4_COVOL),
                acceptance=float(len(x) / n),
                mean_r2=float(m2.mean()), mean_r2_stderr=float(m2.std(ddof=1) / math.sqrt(len(x))),
                mean_r2_exact=float(D4_MEAN2),
                U_implied=float(D4_COVOL * m2.mean()), U_exact=float(D4_U),
                max_r2=float(m2.max()), quartiles=[float(q) for q in np.quantile(m2, [0.5, 0.9, 0.99, 0.999])])


def bcc_in_voronoi(x, tol=1e-12):
    """Voronoi cell of the origin for bcc: |x_i| <= 1/2 and |sum(+-x)| <= 3/4."""
    x = np.atleast_2d(np.asarray(x, float))
    ok = np.all(np.abs(x) <= 0.5 + tol, axis=-1)
    s = np.abs(np.stack([x.sum(-1), x[:, 0] + x[:, 1] - x[:, 2],
                         x[:, 0] - x[:, 1] + x[:, 2], -x[:, 0] + x[:, 1] + x[:, 2]], -1))
    return ok & np.all(s <= 0.75 + tol, axis=-1)


def mc_bcc(n=4_000_000, seed=12):
    x = mc_ball(n, 3, seed) * BCC_R
    keep = bcc_in_voronoi(x)
    x = x[keep]
    m2 = np.sum(x ** 2, -1)
    return dict(n_drawn=int(n), n_kept=int(len(x)), volume=float(BCC_COVOL),
                acceptance=float(len(x) / n),
                mean_r2=float(m2.mean()), mean_r2_stderr=float(m2.std(ddof=1) / math.sqrt(len(x))),
                mean_r2_exact=float(BCC_MEAN2),
                U_implied=float(BCC_COVOL * m2.mean()), U_exact=float(BCC_U),
                max_r2=float(m2.max()), quartiles=[float(q) for q in np.quantile(m2, [0.5, 0.9, 0.99, 0.999])])


# -- verification ---------------------------------------------------------------------------
def verify(n=20000, seed=7, scale=0.08837):
    """Compare the nearest-point routines with brute force on a random sample.

    Three samples, because the interesting cases are different: uniform in a box (generic
    points), points on the half-integer grid (where ties live), and points uniform in a
    ball of radius R (the covering regime).
    """
    rng = np.random.default_rng(seed)
    out = {}

    def run(name, y, nearest, brute, d):
        p = nearest(y)
        pb, db, ties = brute(y)
        d_alg = np.linalg.norm(y - p, axis=1)      # in the caller's units
        db = db * scale                            # brute force works in lattice units
        err = np.abs(d_alg - db)
        bad = int(np.sum(err > 1e-9))
        out[name] = dict(n=int(len(y)), dim=d,
                         max_abs_distance_error=float(err.max()),
                         mean_abs_distance_error=float(err.mean()),
                         n_disagreeing=int(bad),
                         frac_disagreeing=float(bad / len(y)),
                         frac_tie_brute=float(np.mean(ties > 1)),
                         frac_tie_max=int(np.max(ties)),
                         frac_wrong_if_ties_ignored=float(np.mean(err > 1e-9)),
                         frac_membership_ok=float(np.mean(
                             d4_members(p / scale, tol=1e-7) if d == 4 else bcc_members(p / scale, tol=1e-7))))

    # (a) uniform in a box: generic points, all parities, all error sizes
    y = rng.uniform(-3.0, 3.0, (n, 4)) * scale
    run('uniform_box', y, lambda z: d4_nearest(z, scale), lambda z: d4_brute(z / scale, 2), 4)

    # (b) half-integer grid, i.e. exactly on the Z^4 cell walls: ties are generic here
    y = (rng.integers(-3, 3, (n, 4)) + 0.5) * scale
    run('half_integer_grid', y, lambda z: d4_nearest(z, scale), lambda z: d4_brute(z / scale, 2), 4)

    # (c) uniform in the ball of radius R*scale: the covering regime
    y = mc_ball(n, 4, seed + 1) * scale
    run('covering_ball', y, lambda z: d4_nearest(z, scale), lambda z: d4_brute(z / scale, 2), 4)

    # (d) bcc, same three samples
    y = rng.uniform(-3.0, 3.0, (n, 3)) * scale
    run('bcc_uniform_box', y, lambda z: bcc_nearest(z, scale), lambda z: bcc_brute(z / scale, 2), 3)
    y = (rng.integers(-3, 3, (n, 3)) + 0.5) * scale
    run('bcc_half_integer_grid', y, lambda z: bcc_nearest(z, scale), lambda z: bcc_brute(z / scale, 2), 3)

    # (e) the deep holes: distance exactly R, and the nearest point is the origin
    ph = d4_nearest(D4_HOLES * scale, scale)
    dh = np.linalg.norm(D4_HOLES * scale - ph, axis=1) / scale
    out['holes'] = dict(n=int(len(D4_HOLES)),
                        radius_exact=float(D4_R),
                        max_radius_measured=float(dh.max()),
                        max_radius_error=float(np.abs(dh - D4_R).max()),
                        n_origin=int(np.sum(np.all(np.abs(ph) < 1e-9, axis=1))),
                        all_in_voronoi=bool(np.all(d4_in_voronoi(D4_HOLES))))
    return out


def main():
    v = verify()
    m4 = mc_d4()
    m3 = mc_bcc()
    rec = dict(constants=dict(
        d4=dict(covolume=D4_COVOL, covering_radius=D4_R, covering_radius_sq=D4_R2,
                inradius_sq=D4_RHO2, U=D4_U, mean_r2=D4_MEAN2, G=D4_G, theta=D4_THETA,
                n_minimal_vectors=int(len(D4_MIN)), n_holes=int(len(D4_HOLES))),
        bcc=dict(covolume=BCC_COVOL, covering_radius=BCC_R, covering_radius_sq=BCC_R2,
                 U=BCC_U, mean_r2=BCC_MEAN2, G=BCC_G, theta=BCC_THETA,
                 theta_cubic=Z3_THETA, G_cubic=Z_G)),
        verify=v, mc_d4=m4, mc_bcc=m3)
    json.dump(rec, open('results/2026-10-18/d4_bank/lattices.json', 'w'), indent=1)

    print('D4 constants: covolume %.1f, R %.1f, U %.6f, mean r^2 %.6f, G %.6f, Theta %.6f'
          % (D4_COVOL, D4_R, D4_U, D4_MEAN2, D4_G, D4_THETA))
    print('  check G from U: %.12f' % ((1.0 / 4) * D4_U / D4_COVOL ** 1.5))
    print('  check Theta:    %.12f   pi^2/4 = %.12f' % (D4_THETA, math.pi ** 2 / 4))
    print('bcc constants: covolume %.1f, R^2 %.6f, mean r^2 %.6f, G %.6f, Theta[cubic] %.6f'
          % (BCC_COVOL, BCC_R2, BCC_MEAN2, BCC_G, Z3_THETA))
    for k, r in v.items():
        if k == 'holes':
            print('holes: %d, max |x| %.15f (exact %.1f), all in WS %s'
              % (r['n'], r['max_radius_measured'], r['radius_exact'], r['all_in_voronoi']))
            continue
        print('%-24s n %6d  max|d_alg-d_brute| %.3e  disagree %.4f  ties(>1) %.4f  max ties %d'
              % (k, r['n'], r['max_abs_distance_error'], r['frac_disagreeing'],
                 r['frac_tie_brute'], r['frac_tie_max']))
    print('MC D4 : mean r^2 %.6f +- %.6f (exact %.6f), max %.6f, U %.6f'
          % (m4['mean_r2'], m4['mean_r2_stderr'], m4['mean_r2_exact'], m4['max_r2'], m4['U_implied']))
    print('MC bcc: mean r^2 %.6f +- %.6f (exact %.6f), max %.6f, U %.6f'
          % (m3['mean_r2'], m3['mean_r2_stderr'], m3['mean_r2_exact'], m3['max_r2'], m3['U_implied']))


if __name__ == '__main__':
    main()
