"""Exact TaylorF2 phase derivatives by Taylor-mode jets,
pycbc.waveform.taylorf2_jets."""
import math

import numpy
import pytest

from pycbc.waveform.taylorf2_jets import (
    Jet, PhaseCoordinates, basis_phase_and_time,
    log_jacobian_from_phase_coordinates, source_jets,
    taylorf2_basis_coefficients, taylorf2_derivatives, taylorf2_phasing)

POINT = dict(mchirp=31.3, q=1.16, spin1z=0.2, spin2z=-0.1)
SCALES = dict(mchirp=1e-3, q=0.05, spin1z=0.05, spin2z=0.05)
NAMES = ('mchirp', 'q', 'spin1z', 'spin2z')
F_LOWER = 0.0135
FREQS = numpy.linspace(0.0136, 0.0205, 257)


def test_jet_arithmetic_gives_taylor_coefficients():
    x = Jet.path(2.0, 3.0, 5)                      # x(s) = 2 + 3 s
    p = 1.7
    power = (x ** p).c
    expected = [math.gamma(p + 1) / (math.gamma(k + 1)
                                     * math.gamma(p - k + 1))
                * 2 ** (p - k) * 3 ** k for k in range(6)]
    assert numpy.allclose(power, expected, rtol=1e-13)
    log = x.log().c
    expected = [math.log(2)] + [(-1) ** (k + 1) * 1.5 ** k / k
                                for k in range(1, 6)]
    assert numpy.allclose(log, expected, rtol=1e-13)
    # (x * x) / x is x again, and 1 / x times x is one
    assert numpy.allclose(((x * x) / x).c, x.c, rtol=1e-13)
    assert numpy.allclose(((1 / x) * x).c, [1, 0, 0, 0, 0, 0], atol=1e-13)


def test_phasing_coefficients_are_lalsimulation_s():
    lalsimulation = pytest.importorskip('lalsimulation')
    m1, m2, c1, c2 = 38.8, 33.35, 0.3, -0.2
    a, b = taylorf2_phasing(m1, m2, c1, c2)
    pn = lalsimulation.SimInspiralTaylorF2AlignedPhasing(m1, m2, c1, c2,
                                                         None)
    for k in range(8):
        assert float(a[k]) == pytest.approx(pn.v[k], rel=1e-12, abs=1e-14)
        assert float(b[k]) == pytest.approx(pn.vlogv[k], rel=1e-12,
                                            abs=1e-14)


def test_basis_coefficients_carry_the_whole_phase():
    """The phase of the difference of two points' basis coefficients is the
    difference of their TaylorF2 phases, pinned at the lower frequency."""
    theta = [POINT[n] for n in NAMES]
    other = [31.31, 1.2, 0.25, -0.05]

    def direct(t):
        _, phase, _ = source_jets(*t, numpy.zeros(4), FREQS, F_LOWER,
                                  order=1)
        A, B, _ = taylorf2_basis_coefficients(*t)
        return phase.c[0], (A, B)
    p1, (A1, B1) = direct(theta)
    p2, (A2, B2) = direct(other)
    phase, _ = basis_phase_and_time(A2 - A1, B2 - B1, FREQS, F_LOWER)
    assert numpy.allclose(phase, p2 - p1, rtol=0,
                          atol=1e-9 * numpy.abs(p2 - p1).max())


def test_derivatives_are_those_of_finite_differences():
    D, T, D2, T2 = taylorf2_derivatives(POINT, SCALES, FREQS, F_LOWER)
    theta0 = numpy.array([POINT[n] for n in NAMES])
    units = numpy.array([SCALES[n] for n in NAMES])

    def log_b_and_t(xi):
        A0, B0, la0 = taylorf2_basis_coefficients(*theta0)
        A, B, la = taylorf2_basis_coefficients(*(theta0 + units * xi))
        phase, t = basis_phase_and_time(A - A0, B - B0, FREQS, F_LOWER)
        return (la - la0) + 1j * phase, t
    h = 1e-3
    pairs = [(i, j) for i in range(4) for j in range(i, 4)]
    for i in range(4):
        e = numpy.zeros(4)
        e[i] = h
        (lp, tp), (lm, tm) = log_b_and_t(e), log_b_and_t(-e)
        assert numpy.allclose(D[i], (lp - lm) / (2 * h), rtol=1e-5,
                              atol=1e-8 * numpy.abs(D[i]).max())
        assert numpy.allclose(T[i], (tp - tm) / (2 * h), rtol=1e-5,
                              atol=1e-8 * numpy.abs(T[i]).max())
    # the second derivatives against differences of the first, exact ones,
    # at points displaced along each coordinate
    h = 1e-2
    for j in range(4):
        shifted = []
        for sign in (1, -1):
            moved = dict(POINT)
            moved[NAMES[j]] += sign * h * SCALES[NAMES[j]]
            shifted.append(taylorf2_derivatives(moved, SCALES, FREQS,
                                                F_LOWER, order=1))
        for i in range(4):
            p = pairs.index((min(i, j), max(i, j)))
            dD = (shifted[0][0][i] - shifted[1][0][i]) / (2 * h)
            dT = (shifted[0][1][i] - shifted[1][1][i]) / (2 * h)
            assert numpy.abs(D2[p] - dD).max() < 2e-5 * numpy.abs(
                D2[p]).max()
            assert numpy.abs(T2[p] - dT).max() < 2e-5 * numpy.abs(
                T2[p]).max()


def test_phase_coordinates_invert_and_their_jacobian_is_the_numerical():
    theta = numpy.array([[31.3, 1.16, 0.2, -0.1], [31.29, 2.5, -0.4, 0.6]])
    pc = PhaseCoordinates([POINT[n] for n in NAMES])
    y = pc.forward(theta)
    assert numpy.allclose(pc.inverse(y), theta, rtol=1e-10, atol=1e-10)
    for t in theta:
        h = 1e-6 * numpy.maximum(numpy.abs(t), 1)
        jac = numpy.empty((4, 4))
        for i in range(4):
            e = numpy.zeros(4)
            e[i] = h[i]
            jac[:, i] = (pc.forward((t + e)[None])[0]
                         - pc.forward((t - e)[None])[0]) / (2 * h[i])
        # the log Jacobian of theta with respect to the phase coordinates
        expected = -numpy.linalg.slogdet(jac)[1]
        got = float(log_jacobian_from_phase_coordinates(t[:, None])[0])
        # both by differences
        assert got == pytest.approx(expected, abs=1e-5)


def test_measured_phase_coordinates_invert_over_a_prior():
    from pycbc.waveform.taylorf2_jets import MeasuredPhaseCoordinates
    f = numpy.linspace(0.0136, 0.0157, 500)
    coords = MeasuredPhaseCoordinates([POINT[n] for n in NAMES], f,
                                      f ** (-7 / 3), F_LOWER)
    rng = numpy.random.default_rng(1)
    prior = numpy.column_stack([rng.uniform(31.295, 31.300, 2000),
                                rng.uniform(1, 10, 2000),
                                rng.uniform(-0.99, 0.99, 2000),
                                rng.uniform(-0.99, 0.99, 2000)])
    with pytest.raises(RuntimeError):
        coords.forward(prior[:1])
    coords.fit(prior)
    y = coords.forward(prior[:200])
    assert numpy.allclose(y[:, 2], prior[:200, 1])
    assert numpy.allclose(y[:, 3], 0.5 * (prior[:200, 2] - prior[:200, 3]))
    back = coords.inverse(y)
    ok = numpy.all(numpy.isfinite(back), axis=1)
    assert ok.mean() > 0.95
    assert numpy.allclose(back[ok], prior[:200][ok], rtol=1e-6, atol=1e-6)
    t = prior[0]
    h = 1e-6 * numpy.maximum(numpy.abs(t), 1)
    jac = numpy.empty((4, 4))
    for i in range(4):
        e = numpy.zeros(4)
        e[i] = h[i]
        jac[:, i] = (coords.forward((t + e)[None])[0]
                     - coords.forward((t - e)[None])[0]) / (2 * h[i])
    got = float(coords.log_jacobian(t[None])[0])
    assert got == pytest.approx(-numpy.linalg.slogdet(jac)[1], abs=1e-4)
