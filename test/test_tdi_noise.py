"""The TDI noise PSD through an orbit's delays, against the analytical PSD."""
import numpy

from pycbc.coordinates.space_orbit import LisaEqualArmOrbit
from pycbc.psd.analytical_space import psd_lisa_acc_noise, psd_lisa_oms_noise
from pycbc.tdi.inference import DEFAULT_ARM, channel_terms
from pycbc.tdi.noise import noise_transfer_delays, orbit_tdi_psd, psd_from_delays

C = 299792458.0
ORBIT = LisaEqualArmOrbit(armlength=DEFAULT_ARM, t0=0.0)
TERMS = channel_terms(('A', 'E'), generation=2, delta_t=5.0)
TIMES = numpy.linspace(0.0, 2 * 3.15581e7, 8)
F = numpy.array([0.004, 0.0134, 0.0299, 0.0301, 0.045, 0.07])
ZERO = C / (4 * DEFAULT_ARM)


def analytical_ae(f, acc_level, oms_level):
    x = 2 * numpy.pi * f * DEFAULT_ARM / C
    acc, oms = psd_lisa_acc_noise(f, acc_level), psd_lisa_oms_noise(f, oms_level)
    return (8 * numpy.sin(x) ** 2 * (4 * (1 + numpy.cos(x) + numpy.cos(x) ** 2) * acc
                                     + (2 + numpy.cos(x)) * oms)
            * 4 * numpy.sin(2 * x) ** 2)


def test_static_equal_arm_delays_give_the_analytical_psd():
    acc, oms = psd_lisa_acc_noise(F, 2.4e-15), psd_lisa_oms_noise(F, 7.9e-12)
    step = DEFAULT_ARM / C
    for name in ('A', 'E'):
        sources = noise_transfer_delays(ORBIT, TERMS[name], TIMES)
        static = {k: (c, numpy.round(d / step) * step)
                  for k, (c, d) in sources.items()}
        numpy.testing.assert_allclose(
            psd_from_delays(static, F, acc, oms),
            analytical_ae(F, 2.4e-15, 7.9e-12), rtol=1e-10)


def test_the_orbit_fills_the_tdi2_zero_and_agrees_elsewhere():
    f = numpy.concatenate([F, [ZERO]])
    psd = orbit_tdi_psd(ORBIT, TERMS, f, TIMES, 2.4e-15, 7.9e-12)
    expected = analytical_ae(F, 2.4e-15, 7.9e-12)
    for name in ('A', 'E'):
        numpy.testing.assert_allclose(psd[name][:-1], expected, rtol=1e-4)
        assert psd[name][-1] > 0
        assert psd[name][-1] < 1e-12 * numpy.median(expected)
