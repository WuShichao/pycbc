"""The family exact in the TaylorF2 phase,
pycbc.tdi.pn_coefficient_surrogate."""
import numpy
import pytest

from pycbc.scheme import CUPYScheme

REFERENCE = dict(mchirp=31.3, q=1.16, spin1z=0.0, spin2z=0.0)
SCALES = dict(mchirp=1e-3, q=0.05, spin1z=0.05, spin2z=0.05)
NAMES = ['mchirp', 'q', 'spin1z', 'spin2z']
F_LOWER = 0.015


def family():
    from test_linear_exponent_surrogate import build
    from pycbc.tdi.pn_coefficient_surrogate import PNCoefficientSurrogate
    base, _, _ = build()
    return PNCoefficientSurrogate(base, REFERENCE, SCALES, NAMES, F_LOWER)


def test_source_follows_the_taylorf2_phase_and_time():
    from pycbc.waveform.taylorf2_jets import source_jets
    fam = family()
    xi = numpy.array([0.7, -0.4, 0.3, 0.2])
    b, t = fam._source(xi)
    theta = fam.theta0 + fam.units * xi
    f = numpy.asarray(fam.grid.frequencies, dtype=float)
    logamp, phase, time = source_jets(*theta, numpy.zeros(4), f, F_LOWER,
                                      order=1)
    _, phase0, time0 = source_jets(*fam.theta0, numpy.zeros(4), f, F_LOWER,
                                   order=1)
    ratio = b / fam.b0
    assert numpy.allclose(numpy.abs(ratio), numpy.exp(
        logamp.c[0] - 5 / 6 * numpy.log(fam.theta0[0])), rtol=1e-12)
    assert numpy.allclose(numpy.angle(ratio * numpy.exp(-1j * (
        phase.c[0] - phase0.c[0]))), 0, atol=1e-8)
    assert numpy.allclose(t - fam.t0, time.c[0] - time0.c[0], rtol=0,
                          atol=1e-6)


def test_day_sums_on_the_gpu_are_the_templates():
    cupy = pytest.importorskip('cupy')
    try:
        cupy.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip('no usable GPU')
    from pycbc.tdi.linear_exponent_surrogate import DailyGram
    with CUPYScheme():
        fam = family()
        gram = DailyGram(fam)
        rng = numpy.random.default_rng(4)
        xis = rng.uniform(-1, 1, (6, 4))
        thetas = fam.theta0 + fam.units * xis
        pw, fw = (v.get() for v in fam.day_sums(thetas, gram))
        # the bins one by one, as for templates out of day order
        rising = fam._monotone
        fam._monotone = lambda *a, **k: numpy.zeros(len(thetas), bool)
        pw1, fw1 = (v.get() for v in fam.day_sums(thetas, gram))
        fam._monotone = rising
        for j, xi in enumerate(xis):
            b, t = fam._source(xi)
            p, f = (v.get() for v in gram._day_sums(b, t))
            for got in (pw, pw1):
                assert numpy.allclose(got[j], p, rtol=1e-6,
                                      atol=1e-9 * p.max())
            for got in (fw, fw1):
                assert numpy.allclose(got[j], f, rtol=1e-6,
                                      atol=1e-9 * f.max())
