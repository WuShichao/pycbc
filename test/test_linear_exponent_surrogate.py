"""The linear-exponent template family of pycbc.tdi.linear_exponent_surrogate."""
import numpy
import pytest

from pycbc.scheme import CUPYScheme
from pycbc.tdi.linear_exponent_surrogate import LinearExponentSurrogate
from pycbc.tdi.ssb_yearly_harmonics import SSBGrid, harmonic_coefficients

DAY = 86400.0
KMAX, SAMPLES, NCOARSE, NINT = 3, 16, 12, 4


def build(dtype=numpy.complex128):
    rng = numpy.random.default_rng(5)
    grid = SSBGrid(0.0, 30 * DAY, 0.015, 60, 1 / (30 * DAY), periods=1)
    f = grid.frequencies
    u = (f - f[0]) / (f[-1] - f[0])
    source = (1 + 0.3 * u) * numpy.exp(2j * numpy.pi * 40 * u ** 2)
    stationary = 2 * DAY + 20 * DAY * u
    D = (rng.normal(size=(NINT, 1)) + 1j * rng.normal(size=(NINT, 1))) \
        * 0.1 * u
    T = rng.normal(size=(NINT, 1)) * 1e3 * u
    coarse = numpy.linspace(f[0], f[-1], NCOARSE)
    shape = (SAMPLES, 2, 2, NCOARSE)
    R0 = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    L = 0.1 * (rng.normal(size=(2,) + shape)
               + 1j * rng.normal(size=(2,) + shape))
    family = LinearExponentSurrogate(grid, coarse, R0, L, source,
                                     stationary, D, T, kmax=KMAX,
                                     dtype=dtype)
    data = rng.normal(size=(2, grid.size)) + 1j * rng.normal(
        size=(2, grid.size))
    return family, data, (source, stationary, D, T, R0)


def test_pieces_are_linear_exponents_of_the_reference():
    family, _, (source, stationary, D, T, R0) = build()
    b, t, rho = family.pieces(numpy.zeros(NINT + 2))
    assert numpy.allclose(b, source)
    assert numpy.allclose(t, stationary)
    _, expected = harmonic_coefficients(R0, KMAX)
    assert numpy.allclose(rho, expected)
    xi = numpy.array([0.3, -0.2, 0.1, 0.5, 0.0, 0.0])
    b, t, _ = family.pieces(xi)
    assert numpy.allclose(b, source * numpy.exp(xi[:NINT] @ D))
    assert numpy.allclose(t, stationary + xi[:NINT] @ T)


def test_correlation_is_the_inner_product_at_every_lag():
    family, data, _ = build()
    xi = numpy.array([0.2, 0.1, -0.3, 0.05, 0.4, -0.2])
    b, t, rho = family.pieces(xi)
    z = family.correlate(data, b, t, rho)
    for lag in (0, 1, 37, family.size // 2, family.size - 3):
        y = family.template(b, t, rho, lag * family.grid.delta_t)
        direct = numpy.einsum('acf,cf->a', numpy.conj(y), data)
        assert numpy.allclose(z[:, lag], direct, rtol=1e-9, atol=1e-9)


def test_single_precision_on_the_gpu_matches_the_cpu():
    cupy = pytest.importorskip('cupy')
    try:
        scheme = CUPYScheme()
    except Exception:
        pytest.skip('no usable GPU')
    family, data, _ = build()
    xi = numpy.array([0.2, 0.1, -0.3, 0.05, 0.4, -0.2])
    z = family.correlate(data, *family.pieces(xi))
    with scheme:
        gpu, _, _ = build(dtype=numpy.complex64)
        zg = gpu.correlate(cupy.asarray(data, dtype=numpy.complex64),
                           *gpu.pieces(xi))
        zg = cupy.asnumpy(zg)
    assert numpy.abs(zg - z).max() < 1e-4 * numpy.abs(z).max()


def test_wrong_coordinates_are_refused():
    family, _, _ = build()
    with pytest.raises(ValueError):
        family.pieces(numpy.zeros(3))
