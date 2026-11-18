"""EccentricPhaseCoordinates on a synthetic 4-d cell: the linear phase model is the cell's own evaluation (less the
marginalised phases and time), and the coordinates invert."""
import numpy as np

from pycbc.filter.eccentric_shared import EccentricPhaseCoordinates
from pycbc.waveform.eccentric_mode_family import EccentricModeFamily

F = np.linspace(.01, .02, 61)


def generator(p):
    q = p['mass1'] / p['mass2']
    s = p.get('spin1z', 0.) + p.get('spin2z', 0.)
    mc = (p['mass1'] * p['mass2']) ** .6 / (p['mass1'] + p['mass2']) ** .2
    out = {}
    for n in (1, 2, 3):
        # phase changes of tens to hundreds of radians over the box, as the real cell's
        phase = (n / 2) * (3e4 * (mc / 30.) ** (-5 / 3) * (F / F[0]) ** (-5 / 3)) \
            + 4e3 * p['eccentricity'] ** 2 * n * (F / F[0]) ** (-34 / 9) + n * s * 3e2 * (F / F[0]) ** (-2 / 3) \
            + q * 40 * (F / F[0]) ** 2
        amp = np.ones((2, len(F)), complex) * (1 + n * p['eccentricity']) * (1 + .1 * s) * (1 + .05 * q)
        out[2, 2, n] = amp, phase, np.ones(len(F), bool)
    return out


def _family():
    ref = dict(mass1=36., mass2=24., eccentricity=.05)
    return EccentricModeFamily(ref, F, .015, step_mass=.01, step_e=.002, step_q=.1, step_chi=.1, generator=generator,
                               circular_phase_basis='taylorf2', phase_scaling='n')


def test_phase_model_is_the_cells_own_and_the_coordinates_invert():
    fam = _family()
    c = EccentricPhaseCoordinates(fam, lambda f: np.ones_like(f), (-1e15, 1e15), coarse=1000)
    rng = np.random.default_rng(3)
    lo = np.array([c.mc0 * np.exp(-.01), 1.5 * np.exp(-.1), .048, -.05])
    hi = np.array([c.mc0 * np.exp(.01), 1.5 * np.exp(.1), .052, .05])
    draws = lo + rng.random((300, 4)) * (hi - lo)
    for mc, q, e, chi in draws[:4]:
        raw = np.zeros(len(c.frequencies))
        modes = {k: (sl, psi) for k, sl, _, psi, _ in fam._windows_uncached(
            np.log(mc / c.mc0), e - c.e0, np.log(q / c.q0), None, None, chi - c.chi0)}
        for m in c._modes:
            sl, psi = modes[m['key']]
            raw[m['block']] = (psi - fam._expansions[m['key']][3])[m['sel']] * c._sw[m['block']]
        want = raw - c._Q @ (c._Q.T @ raw)
        got = c.phases([[mc, q, e, chi]])[0]
        assert np.linalg.norm(got - want) <= 1e-8 * max(np.linalg.norm(want), 1.)
    c.fit(draws)
    # the tensor contraction is the polynomial's own weights
    X = c._offsets(*draws[:20].T)
    s_ref = sum(m['M'] @ (m['weights'](X) - m['W0'][None]).T for m in c._modes)
    from pycbc.waveform.taylorf2_jets import basis_coefficients
    s_ref = s_ref + c._M_pn @ (basis_coefficients(draws[:20, 0], draws[:20, 1], draws[:20, 3], draws[:20, 3])
                               - c._C0[:, None]) - (c._s0 - sum(m['M'] @ m['W0'] for m in c._modes))[:, None]
    np.testing.assert_allclose(c.forward(draws[:20])[:, :3], s_ref.T, rtol=1e-9, atol=1e-9 * np.abs(s_ref).max())
    assert np.all(np.isnan(c.inverse([[0., 0., 0., .9]])))
    y = c.forward(draws[:50])
    np.testing.assert_allclose(y[:, 3], draws[:50, 1])
    back = c.inverse(y)
    assert np.all(np.isfinite(back))
    np.testing.assert_allclose(back, draws[:50], rtol=0, atol=1e-8 * np.max(hi))
    assert np.all(np.isfinite(c.log_jacobian(draws[:10])))


def test_the_family_pickles_and_evaluates_the_same():
    import pickle
    fam = _family()
    back = pickle.loads(pickle.dumps(fam))
    for args in ((0.004, 0.001, 0.05, 0.03), (-0.006, -0.002, -0.08, 0.0)):
        want = fam.modes(*args[:3], dchi=args[3])
        got = back.modes(*args[:3], dchi=args[3])
        for k in want:
            for a, b in zip(want[k], got[k]):
                np.testing.assert_array_equal(a, b)


def test_two_phase_directions_keep_q_and_e_and_invert_one_to_one():
    """ndir=2: the coordinates are (s_1, s_2, q, e); they invert to the drawn parameters, the (mchirp, chi_eff)
    block of the Jacobian at fixed (q, e) is regular everywhere in the box, and the inverse from the default
    guess returns the drawn point (not another root sharing its y)."""
    fam = _family()
    c = EccentricPhaseCoordinates(fam, lambda f: np.ones_like(f), (-1e15, 1e15), coarse=1000, ndir=2)
    rng = np.random.default_rng(5)
    lo = np.array([c.mc0 * np.exp(-.01), 1.5 * np.exp(-.1), .048, -.05])
    hi = np.array([c.mc0 * np.exp(.01), 1.5 * np.exp(.1), .052, .05])
    draws = lo + rng.random((300, 4)) * (hi - lo)
    c.fit(draws)
    y = c.forward(draws[:60])
    np.testing.assert_allclose(y[:, 2], draws[:60, 1])
    np.testing.assert_allclose(y[:, 3], draws[:60, 2])
    back = c.inverse(y)
    assert np.all(np.isfinite(back))
    np.testing.assert_allclose(back, draws[:60], rtol=0, atol=1e-8 * np.max(hi))
    assert np.all(np.isfinite(c.log_jacobian(draws[:10])))
    span = hi - lo
    for t in draws[:20]:
        J = np.empty((2, 2))
        for col, j in enumerate((0, 3)):
            d = np.zeros(4)
            d[j] = 1e-4 * span[j]
            J[:, col] = (c.forward((t + d)[None])[0, :2] - c.forward((t - d)[None])[0, :2]) / 2e-4
        assert abs(np.linalg.det(J)) > 1e-6 * np.abs(J).max() ** 2
    with np.testing.assert_raises(ValueError):
        EccentricPhaseCoordinates(fam, lambda f: np.ones_like(f), (-1e15, 1e15), coarse=1000, ndir=4)
