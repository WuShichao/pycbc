"""The rank-classified F-statistic of pycbc.filter.fstatistic."""
import numpy
import pytest

from pycbc.filter.fstatistic import (FULL_RANK, NO_COVERAGE, RANK_ONE,
                                     UNRESOLVED, twof)
from pycbc.scheme import CUPYScheme, array_module

RNG = numpy.random.default_rng(3)
U = numpy.array([0.6, 0.8j])
V = numpy.array([0.8, -0.6j])
A = RNG.standard_normal((2, 2)) + 1j * RNG.standard_normal((2, 2))
Z = numpy.array([1 + 0.5j, -0.3 + 0.2j])
CASES = {
    'full_rank': (A @ A.conj().T, Z, FULL_RANK),
    'rank_one': (2.5 * numpy.outer(U, U.conj()), Z, RANK_ONE),
    'near_singular_full': (2.5 * numpy.outer(U, U.conj())
                           + 2.5e-4 * numpy.outer(V, V.conj()), Z, FULL_RANK),
    'zero': (numpy.zeros((2, 2), complex), Z, NO_COVERAGE),
    'indefinite': (numpy.diag([1.0, -0.5]).astype(complex), Z, UNRESOLVED),
    'non_finite': (numpy.array([[numpy.nan, 0], [0, 1]], complex), Z,
                   UNRESOLVED),
}


def evaluate(gram, z, xp=numpy):
    one = lambda x: xp.asarray(numpy.array([x]))
    return twof(xp.asarray(z[:, None]), one(gram[0, 0].real),
                one(gram[1, 1].real), one(gram[0, 1]), xp=xp)


@pytest.mark.parametrize('name', sorted(CASES))
def test_each_rank_class(name):
    gram, z, expected = CASES[name]
    value, code = evaluate(gram, z)
    assert int(code[0]) == expected
    if expected == FULL_RANK:
        ref = numpy.real(z.conj() @ numpy.linalg.solve(gram, z))
        assert value[0] == pytest.approx(ref, rel=1e-10)
    elif expected == RANK_ONE:
        ref = numpy.real(z.conj() @ numpy.linalg.pinv(gram) @ z)
        assert value[0] == pytest.approx(ref, rel=1e-10)
    elif expected == NO_COVERAGE:
        assert value[0] == -numpy.inf
    else:
        assert numpy.isnan(value[0])


def test_vectorised_full_rank_against_a_solve():
    n = 500
    b = RNG.normal(size=(n, 2, 2)) + 1j * RNG.normal(size=(n, 2, 2))
    gram = b @ numpy.conj(numpy.swapaxes(b, 1, 2)) + 0.1 * numpy.eye(2)
    z = RNG.normal(size=(2, n)) + 1j * RNG.normal(size=(2, n))
    value, code = twof(z, gram[:, 0, 0].real, gram[:, 1, 1].real,
                       gram[:, 0, 1], kappa=2.0)
    ref = numpy.real(numpy.einsum('ni,nij,nj->n', z.T.conj(),
                                  numpy.linalg.inv(gram), z.T))
    assert numpy.all(code == FULL_RANK)
    assert numpy.allclose(value, 2.0 * ref, rtol=1e-10)


def test_cupy_matches_numpy():
    cupy = pytest.importorskip('cupy')
    try:
        scheme = CUPYScheme()
    except Exception:
        pytest.skip('no usable GPU')
    with scheme:
        xp = array_module()
        assert xp is cupy
        for gram, z, _ in CASES.values():
            on_gpu = evaluate(gram, z, xp=xp)
            on_cpu = evaluate(gram, z)
            assert int(on_gpu[1][0]) == int(on_cpu[1][0])
            numpy.testing.assert_allclose(cupy.asnumpy(on_gpu[0]),
                                          on_cpu[0], rtol=1e-12)
