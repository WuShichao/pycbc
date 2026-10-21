"""Correlations of many PN-coefficient templates by one NUFFT,
pycbc.filter.nufft_correlation."""
import numpy
import pytest

from pycbc.waveform.taylorf2_jets import taylorf2_basis_coefficients

REFERENCE = numpy.array([31.3, 1.16, 0.0, 0.0])
F_LOWER = 0.0135


def gpu():
    cupy = pytest.importorskip('cupy')
    pytest.importorskip('cufinufft')
    try:
        cupy.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip('no usable GPU')
    return cupy


def host(array):
    return array.get() if hasattr(array, 'get') else numpy.asarray(array)


def correlator(xp=numpy, n=20000, seed=1):
    from pycbc.filter.nufft_correlation import PhaseNUFFTCorrelation
    rng = numpy.random.default_rng(seed)
    f = numpy.linspace(0.0136, 0.0205, n)
    orders = numpy.array([-1, 0, 1])
    strengths = (rng.normal(size=(3, 2, n))
                 + 1j * rng.normal(size=(3, 2, n))) * f ** (-7 / 6)
    A0, B0, la0 = taylorf2_basis_coefficients(*REFERENCE)
    step = n // 1000
    return PhaseNUFFTCorrelation.from_arrays(
        strengths, f, F_LOWER, orders, A0, B0, la0, f[::step],
        numpy.full(len(f[::step]), 1.0 / len(f[::step])), xp=xp, eps=1e-9)


def templates(count, seed=2):
    """Points near the reference along the directions the phase measures,
    and freely along the chirp-mass - mass-ratio valley."""
    rng = numpy.random.default_rng(seed)
    out = REFERENCE + rng.uniform(-1, 1, size=(count, 4)) * numpy.array(
        [2e-4, 0.05, 0.05, 0.05])
    out[:, 1] = numpy.maximum(out[:, 1], 1.0)
    return out


def relative_error(z, exact):
    return float(numpy.abs(z - exact).max() / numpy.abs(exact).max())


def test_grid_reproduces_the_sum():
    c = correlator()
    thetas = templates(8)
    delays = numpy.linspace(-2e4, 2e4, 21)
    exact = c.direct(thetas, delays)
    grid = c.correlate_grid(thetas, delays, taylor_tolerance=1e-9)
    assert relative_error(grid, exact) < 1e-5
    # per-template delays reach the same values
    from pycbc.filter.nufft_correlation import PhaseGrid
    g = PhaseGrid(c, thetas, (delays.min(), delays.max()),
                  taylor_tolerance=1e-9)
    own = numpy.repeat(delays[None], len(thetas), axis=0)
    assert relative_error(g.evaluate(thetas, own), exact) < 1e-5
    assert g.inside(thetas, (delays.min(), delays.max())).all()
    far = thetas.copy()
    far[:, 0] += 0.05
    assert not g.inside(far, (delays.min(), delays.max())).any()


def test_one_grid_serves_any_number_of_templates():
    from pycbc.filter.nufft_correlation import PhaseGrid
    c = correlator()
    region = templates(12)
    delays = numpy.linspace(-2e4, 2e4, 21)
    g = PhaseGrid(c, region, (delays.min(), delays.max()),
                  taylor_tolerance=1e-9)
    rng = numpy.random.default_rng(4)
    i, j = rng.integers(0, len(region), (2, 200))
    w = rng.random((200, 1))
    many = w * region[i] + (1 - w) * region[j]
    z = g.evaluate(many, delays)
    assert z.shape == (2, 200, len(delays))
    assert relative_error(z[:, :6], c.direct(many[:6], delays)) < 1e-5


def test_gpu_grid_matches_the_cpu():
    cupy = gpu()
    thetas = templates(8)
    delays = numpy.linspace(-2e4, 2e4, 21)
    exact = correlator().direct(thetas, delays)
    z = host(correlator(cupy).correlate_grid(thetas, delays,
                                             taylor_tolerance=1e-7))
    # the transform runs in single precision on the device
    assert relative_error(z, exact) < 5e-4


def test_runs_of_bins_gathered_by_their_moments():
    """For a narrow window of arrival times the transform gathers runs of
    bins at their middles through their moments; the correlations stay
    those of the direct sum."""
    from pycbc.filter.nufft_correlation import PhaseGrid
    c = correlator(n=200000)
    thetas = templates(8)
    delays = numpy.linspace(-300, 300, 7)
    g = PhaseGrid(c, thetas, (delays.min(), delays.max()),
                  taylor_tolerance=1e-9)
    assert g.blocks > 1
    exact = c.direct(thetas, delays)
    assert relative_error(g.evaluate(thetas, delays), exact) < 1e-5


def test_sky_basis_errors_are_exact():
    """The sky basis reports each point's relative error of its projected
    response, weighted by the source's power on the coarse frequencies,
    and keeps the training points within the tolerance."""
    from test_linear_exponent_surrogate import build
    from pycbc.filter.nufft_correlation import SkyBasis, source_power_weights
    family, _, _ = build()
    rng = numpy.random.default_rng(3)
    train = rng.uniform(-1, 1, (200, 2))
    basis = SkyBasis(family, train, tolerance=1e-6)
    assert 1 <= basis.rank < 64
    _, err = basis.project(train)
    assert err.max() <= 1e-6 / 3 * 1.001
    test = rng.uniform(-1, 1, (20, 2))
    coords, err = basis.project(test)
    w = source_power_weights(family)
    for d, c, e in zip(test, coords, err):
        exact = family._response(d)
        approx = sum(cj * v for cj, v in zip(c, basis))
        num = ((numpy.abs(approx - exact) ** 2).sum(axis=(0, 1, 2)) * w).sum()
        den = ((numpy.abs(exact) ** 2).sum(axis=(0, 1, 2)) * w).sum()
        assert numpy.sqrt(num / den) == pytest.approx(e, rel=1e-6, abs=1e-12)
