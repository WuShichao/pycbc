"""PrecessionMainRow: the isotropic spin prior's projected density, and the build recovering a known main-row
correction (shapes, emulator, 2 pi branches, the pin at f_ref)."""
import numpy
import pytest

from pycbc.waveform.precession_main_row import (build_precession_main_row, features, isotropic_log_prior)


def test_isotropic_prior_against_monte_carlo():
    rng = numpy.random.default_rng(0)
    n = 4_000_000
    q = 1.5
    a = rng.uniform(0, .99, (2, n))
    c = rng.uniform(-1, 1, (2, n))
    perp = a * numpy.sqrt(1 - c ** 2)
    z = a * c
    chi_eff = (q * z[0] + z[1]) / (1 + q)
    # density in a few boxes against the formula integrated over them
    for lo in ((-.06, .2, .3), (-.02, .5, .1), (-.1, .6, .6)):
        box = (.04, .1, .1)
        m = ((chi_eff >= lo[0]) & (chi_eff < lo[0] + box[0]) & (perp[0] >= lo[1]) & (perp[0] < lo[1] + box[1])
             & (perp[1] >= lo[2]) & (perp[1] < lo[2] + box[2]))
        mc = m.mean() / numpy.prod(box)
        g = [l + (numpy.arange(8) + .5) / 8 * b for l, b in zip(lo, box)]
        E, S1, S2 = numpy.meshgrid(*g, indexing='ij')
        formula = numpy.exp(isotropic_log_prior(E.ravel(), S1.ravel(), S2.ravel(), q)).mean()
        assert formula == pytest.approx(mc, rel=0.05)


def test_build_recovers_a_known_correction():
    f = numpy.linspace(1., 2., 400)
    f_ref = 1.1
    keys = [(2, 2, 1), (2, 2, 2), (2, 2, 3)]
    b1 = (f - 1.5) ** 2
    b2 = numpy.sin(3 * f)
    rng = numpy.random.default_rng(1)

    def truth_coeffs(lam):
        F = features(lam)[0]
        return numpy.array([F[5] * 0.8 + F[6] * 0.3, 0.2 * F[13] - 0.1 * F[7]])

    def generator(lam, inplane):
        out = {}
        for k in keys:
            have = (f > 1.0 + 0.05 * (k[2] - 1)) & (f < 1.9)
            psi = 100 * f ** (-5. / 3) * k[2]
            if inplane:
                c = truth_coeffs(lam)
                psi = psi + c[0] * b1 + c[1] * b2 + 4.0 * k[2]       # plus absorbed parts and 2 pi branches
            # efpe_modes' convention: the slowly varying amplitude, the waveform amp * exp(-i psi)
            amp = numpy.zeros((2, len(f)), complex)
            amp[0] = numpy.where(have, 1. + 0.1 * f, 0)
            out[k] = (amp, psi, have)
        return out
    draws = numpy.column_stack([30 * numpy.exp(rng.uniform(-.01, .01, 160)), rng.uniform(.045, .055, 160),
                                rng.uniform(-.1, .1, 160), 1.5 * numpy.exp(rng.uniform(-.2, .2, 160)),
                                rng.uniform(0, .9, 160), rng.uniform(0, .9, 160), rng.uniform(0, 2 * numpy.pi, 160)])
    weight = {k: numpy.ones(len(f)) for k in keys}
    model = build_precession_main_row(generator, draws, f, f_ref, weight, nshape=2)
    assert model.diagnostics['emulator_max_rad'] < 1e-6
    # the model's phase agrees with the truth up to the absorbed parts (phase, time, (n - m) l0)
    lam = draws[-5:]
    for k in keys:
        got = model.phase(k, lam)
        c = numpy.array([truth_coeffs(l) for l in lam])
        true = c[:, :1] * b1[None] + c[:, 1:] * b2[None]
        have = (f > 1.0 + 0.05 * (k[2] - 1)) & (f < 1.9)           # the fitted support (shapes extend past it)
        d = (got - true)[:, have]
        A = numpy.column_stack([numpy.ones(have.sum()), f[have]])
        res = d - (A @ numpy.linalg.lstsq(A, d.T, rcond=None)[0]).T
        assert numpy.abs(res).max() < 1e-6
    # pinned: value and slope of every shape on (2,2,2) at f_ref vanish
    s = model.shapes[(2, 2, 2)]
    for row in s:
        assert abs(numpy.interp(f_ref, f, row)) < 1e-9
        assert abs(numpy.interp(f_ref, f[1:-1], (row[2:] - row[:-2]) / (f[2:] - f[:-2]))) < 1e-6
