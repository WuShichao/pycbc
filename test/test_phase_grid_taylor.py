"""PhaseGrid's multivariate Taylor terms: one direction gives the old single-direction orders; two give the
multi-indices reached through a term above the tolerance, each with a parent one order below."""
import math

import numpy

from pycbc.filter.nufft_correlation import _taylor_terms


def old_terms(reach, tol):
    terms, term = 1, 1.0
    while term > tol:
        term *= reach / terms
        terms += 1
    return terms


def test_one_direction_matches_the_old_orders():
    for reach in (0.0, 1e-5, 0.01, 0.3, 1.0, 2.7):
        alphas, parents = _taylor_terms([reach], 1e-4)
        assert len(alphas) == old_terms(reach, 1e-4)
        assert alphas == [(q,) for q in range(len(alphas))]


def test_two_directions():
    alphas, parents = _taylor_terms([0.5, 0.01], 1e-4)
    assert alphas[0] == (0, 0)
    for q, a in enumerate(alphas[1:], 1):
        par, d = parents[q]
        child = tuple(v + (j == d) for j, v in enumerate(alphas[par]))
        assert child == a
    # every term whose parent exceeds the tolerance is there; the small variable stops one order past 1e-4
    assert (0, 2) in alphas and (0, 3) not in alphas
    assert (6, 0) in alphas
    sizes = [0.5 ** a / math.factorial(a) * 0.01 ** b / math.factorial(b) for a, b in alphas]
    assert min(sizes) <= 1e-4
