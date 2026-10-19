"""The linear-exponent family of pycbc.tdi.linear_exponent_surrogate."""
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


def chirp(days, periods=3, seed=5):
    """A stationary-phase-consistent chirp crossing both window edges."""
    from pycbc.tdi.linear_exponent_surrogate import DailyGram
    rng = numpy.random.default_rng(seed)
    nbins = int(20 * days)
    grid = SSBGrid(0.0, days * DAY, 0.015, nbins, 1 / (days * DAY),
                   periods=periods)
    f = grid.frequencies
    u = (f - f[0]) / (f[-1] - f[0])
    t0 = (-days / 6 + 1.47 * days * u) * DAY
    phi = numpy.concatenate([[0], numpy.cumsum(
        0.5 * (t0[1:] + t0[:-1]) * numpy.diff(f))])
    amp = numpy.where((u > 0.02) & (u < 0.98), 1 + 0.3 * u, 0.0)
    coarse = numpy.linspace(f[0], f[-1], NCOARSE)
    shape = (SAMPLES, 2, 2, NCOARSE)
    R0 = 1 + 0.3 * (rng.normal(size=shape) + 1j * rng.normal(size=shape))
    zero = numpy.zeros((1, grid.size))
    family = LinearExponentSurrogate(
        grid, coarse, R0, numpy.zeros((2,) + shape),
        amp * numpy.exp(-2j * numpy.pi * phi), t0, zero, zero, kmax=KMAX)
    return family, DailyGram(family)


def gram_error(days):
    family, daily = chirp(days)
    grid = family.grid
    b, t, rho = family.pieces(numpy.zeros(3))
    at_lags = daily.at_lags(daily(b, t, rho))
    worst = 0.0
    for lag in (0, grid.size - int(0.1 * days * DAY / grid.delta_t)):
        delta = (lag if lag < grid.size // 2 else lag - grid.size) \
            * grid.delta_t
        yt = numpy.fft.ifft(family.template(b, t, rho, delta), axis=-1)
        gram = grid.size * numpy.einsum('t,act,bct->ab', grid.in_data,
                                        numpy.conj(yt), yt)
        error = max(abs(at_lags[0][lag] - gram[0, 0]),
                    abs(at_lags[1][lag] - gram[1, 1]),
                    abs(at_lags[2][lag] - gram[0, 1]))
        worst = max(worst, error / numpy.abs(gram).max())
    return worst


def test_daily_gram_approaches_the_windowed_gram():
    # the error is the stationary-phase smearing at the window edges, about
    # the time spread of one frequency over the window length
    short, long = gram_error(30), gram_error(240)
    assert long < 3e-3
    assert long < short / 5


def test_builder_reproduces_the_research_family():
    """Against ssb_family's Setup and Cell (sandbox results/2026-10-19/
    pycbc_port/make_surrogate_fixture.py) on 80 fine and 12 coarse bins."""
    import json
    import os
    from types import SimpleNamespace
    from pycbc.coordinates.space_orbit import LisaEqualArmOrbit
    from pycbc.tdi.inference import channel_terms
    from pycbc.tdi.linear_exponent_surrogate import surrogate_from_source
    ref = numpy.load(os.path.join(os.path.dirname(__file__), 'data',
                                  'ssb_family_reference.npz'))
    base, point = json.loads(str(ref['params'])), json.loads(str(ref['point']))
    sigma = json.loads(str(ref['sigma']))

    def to_source(p):
        # the research coordinates: q = m2 / m1 <= 1, fiducial masses kept
        out = dict(base)
        if (p['mchirp'], p['q']) != (point['mchirp'], point['q']):
            m1 = p['mchirp'] * (1 + p['q']) ** 0.2 / p['q'] ** 0.6
            out['mass1'], out['mass2'] = m1, p['q'] * m1
        out.update(spin1z=p['spin1z'], spin2z=p['spin2z'])
        return out

    freqs = ref['freqs']
    grid = SimpleNamespace(frequencies=freqs, start=float(ref['start']),
                           size=len(freqs), harmonic_shift=5)
    names = ('mchirp', 'q', 'spin1z', 'spin2z')
    family = surrogate_from_source(
        grid, ref['sw'], LisaEqualArmOrbit(armlength=2.5e9, t0=0.0),
        channel_terms(('A', 'E'), generation=2, delta_t=10.0),
        dict(base, **point), {n: sigma[n] for n in names},
        (sigma['eclipticlongitude'], sigma['eclipticlatitude']),
        ref['coarse'][ref['columns']], to_source=to_source,
        step=float(ref['step']))
    scale = numpy.abs(ref['b0']).max()
    assert numpy.abs(family.b0 - ref['b0']).max() < 1e-12 * scale
    assert numpy.allclose(family.t0, ref['t0'], rtol=0, atol=1e-6)
    assert numpy.allclose(family.D.real, ref['D'].real, rtol=0, atol=1e-12)
    # the phase columns agree up to the unwrapping's whole turns
    turns = (family.D.imag - ref['D'].imag) * 2 * float(ref['step'])
    assert numpy.allclose(numpy.exp(1j * turns), 1, atol=1e-12)
    assert numpy.allclose(family.T, ref['Tt'], rtol=0, atol=1e-6)
    # the response times are staggered by 1 ms per coarse bin, of which the
    # reference had 60
    r0 = numpy.abs(ref['R0']).max()
    assert numpy.abs(family.R0 - ref['R0']).max() < 1e-7 * r0
    assert numpy.abs(family.L - ref['L']).max() \
        < 2e-6 * numpy.abs(ref['L']).max()


def test_tangents_are_the_derivatives_of_the_template():
    family, _, _ = build()
    xi = numpy.array([0.2, -0.1, 0.3, 0.05, 0.1, -0.2])
    delta = 7 * family.grid.delta_t
    y, dy = family.tangents(xi, delta)
    assert numpy.allclose(y, family.template(*family.pieces(xi), delta))
    step = 1e-6
    for i in range(len(xi)):
        up, down = xi.copy(), xi.copy()
        up[i] += step
        down[i] -= step
        numeric = (family.template(*family.pieces(up), delta)
                   - family.template(*family.pieces(down), delta)) / (2 * step)
        assert numpy.abs(dy[i] - numeric).max() < 1e-6 * numpy.abs(y).max()
