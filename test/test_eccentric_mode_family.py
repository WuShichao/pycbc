"""EccentricModeFamily with a synthetic mode generator: a circular phase
fast in chirp mass, an eccentric correction C(Mc) e^2 and amplitudes smooth
in both, which the family must reproduce far from its reference."""
import numpy

from pycbc.waveform.eccentric_mode_family import EccentricModeFamily

F = numpy.linspace(0.0136, 0.0157, 400)


def generator(p):
    mc = (p['mass1'] * p['mass2']) ** 0.6 / (p['mass1'] + p['mass2']) ** 0.2
    e = p['eccentricity']
    psi22 = 3e3 * (mc / 30.0) ** (-5.0 / 3) * (F / F[0]) ** (-5.0 / 3)
    modes = {(2, 2, 2): (numpy.ones((2, len(F)), complex) * (1 + e),
                         psi22 + 40 * (mc / 30.0) ** (-5.0 / 3) * e ** 2
                         * (F / F[0]) ** (-34.0 / 9), numpy.ones(len(F), bool))}
    if e > 0:
        modes[2, 3, 2] = (0.3 * e * (mc / 30.0) * numpy.ones((2, len(F)),
                                                             complex),
                          1.5 * psi22 + 60 * (mc / 30.0) ** (-5.0 / 3) * e
                          * (F / F[0]) ** (-19.0 / 9),
                          numpy.ones(len(F), bool))
    return modes


def test_the_family_follows_a_correction_quadratic_in_e():
    ref = dict(mass1=36.0, mass2=31.0, eccentricity=0.05)
    fam = EccentricModeFamily(ref, F, F[0], generator=generator)
    assert sorted(fam.keys) == [(2, 2, 2), (2, 3, 2)]
    for d, de in ((0.0, 0.004), (0.02, -0.004), (-0.03, 0.005)):
        p = dict(ref, mass1=ref['mass1'] * numpy.exp(d),
                 mass2=ref['mass2'] * numpy.exp(d), eccentricity=0.05 + de)
        exact = generator(p)
        got = fam.modes(d, de)
        for k in exact:
            # phases to a small fraction of a radian, amplitudes closely
            assert numpy.max(abs(got[k][1] - exact[k][1])) < 1e-2
            numpy.testing.assert_allclose(got[k][0], exact[k][0], rtol=1e-3)


def test_the_polarisations_sum_the_modes():
    ref = dict(mass1=36.0, mass2=31.0, eccentricity=0.05)
    fam = EccentricModeFamily(ref, F, F[0], generator=generator)
    h = fam.polarizations(0.0, 0.0)
    want = sum(a * numpy.exp(-1j * p) for a, p, _ in generator(ref).values())
    # the stencil keeps amplitudes in single precision
    numpy.testing.assert_allclose(h, want, rtol=1e-6, atol=1e-6)
    numpy.testing.assert_allclose(fam.hplus(0.0, 0.0), want[0], rtol=1e-6,
                                  atol=1e-6)


def _quartic_generator(p):
    """A mode whose eccentric phase is quartic in e: the quadratic stencil
    cannot follow it, the quartic one can."""
    e = p['eccentricity']
    psi = 3e3 * (F / F[0]) ** (-5.0 / 3) + 4e4 * (e - 0.1) ** 4 \
        * (F / F[0]) ** (-34.0 / 9)
    return {(2, 2, 2): (numpy.ones((2, len(F)), complex), psi,
                        numpy.ones(len(F), bool))}


def test_the_quartic_stencil_follows_what_the_quadratic_cannot():
    ref = dict(mass1=36.0, mass2=31.0, eccentricity=0.1)
    errs = {}
    for order in (3, 5):
        fam = EccentricModeFamily(ref, F, F[0], step_mass=0.0, step_e=0.01,
                                  step_q=0.0, generator=_quartic_generator,
                                  order=order)
        p = dict(ref, eccentricity=0.115)
        errs[order] = numpy.max(abs(fam.modes(0.0, 0.015)[2, 2, 2][1]
                                    - _quartic_generator(p)[2, 2, 2][1]))
    assert errs[5] < 1e-8 < errs[3]


def test_the_taper_takes_each_mode_to_zero_at_its_support_edges():
    import pytest
    pytest.importorskip('pyEFPEHM')
    from pycbc.waveform.eccentric_mode_family import efpe_modes
    f = numpy.linspace(0.0136, 0.0157, 2000)
    p = dict(mass1=36.0, mass2=31.0, eccentricity=0.05)
    plain = efpe_modes(p, f, f[0])
    tapered = efpe_modes(p, f, f[0], taper=1e6)
    assert plain.keys() == tapered.keys()
    for k, (a, psi, have) in plain.items():
        b = tapered[k][0]
        idx = numpy.flatnonzero(have)
        assert numpy.all(abs(b[:, idx[[0, -1]]]) == 0)
        assert numpy.all(abs(b) <= abs(a) + 1e-30)
        # a mode's time over the band is years: the middle is untouched
        mid = idx[len(idx) // 2]
        numpy.testing.assert_allclose(b[:, mid], a[:, mid])
