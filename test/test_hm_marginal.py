"""The higher-mode (several angular patterns) Bayes factor of
pycbc.filter.hm_marginal: its spherical harmonics and projector against LAL
and pyEFPEHM, its reduction to the (2,2)-only marginalisation, and its value
against a brute-force integral over every angle with explicit templates."""
import unittest

import numpy

from pycbc.filter.bstatistic import (DistancePowerLaw, OrientationNodes,
                                     log_amplitude_marginal)
from pycbc.filter.hm_marginal import (spin_weighted_m2, projector,
                                      pattern_weights, hm_log_bayes,
                                      draw_hm_extrinsics)
from pycbc.filter.mean_anomaly import mean_anomaly_log_bayes, mode_offsets


def _templates(keys, n, rng, scale):
    """Unit-plus and unit-cross responses of each mode: random complex
    vectors of length ``n`` (correlated between modes, as real modes are),
    the higher patterns scaled by ``scale``."""
    base = rng.normal(size=(2, n)) + 1j * rng.normal(size=(2, n))
    y = []
    for k in keys:
        own = rng.normal(size=(2, n)) + 1j * rng.normal(size=(2, n))
        amp = 1.0 if k[:2] == (2, 2) else scale
        y.append(amp * (0.6 * base + 0.8 * own) / numpy.sqrt(n))
    return numpy.array(y)                                   # (K, 2, n)


def _signal(y, keys, h, iota, psi, phi, l0):
    s = 0
    for k, yk in zip(keys, y):
        w = pattern_weights(k[0], k[1], iota, psi)
        s = s + numpy.exp(1j * (k[1] * phi + (k[2] - k[1]) * l0)) \
            * (w[0] * yk[0] + w[1] * yk[1])
    return h * s


def _zg(y, d):
    z = numpy.einsum('kan,n->ka', numpy.conj(y), d)
    gram = numpy.einsum('kan,lbn->kalb', numpy.conj(y), y)
    return z, gram


class TestHarmonics(unittest.TestCase):
    def test_against_lal(self):
        try:
            import lal
        except ImportError:
            self.skipTest('no lal')
        worst = 0
        for th in (0., .3, 1.1, 2.5, numpy.pi):
            for ph in (0., .7):
                for l in (2, 3, 4, 5):
                    for m in range(-l, l + 1):
                        worst = max(worst, abs(
                            complex(spin_weighted_m2(l, m, th, ph))
                            - lal.SpinWeightedSphericalHarmonic(th, ph, -2,
                                                                l, m)))
        self.assertLess(worst, 1e-13)

    def test_projector_reproduces_pyefpehm(self):
        """Every mode's polarisations from its hlm and the projector, at
        inclinations either side of pi / 2, with spins and eccentricity."""
        try:
            from pyEFPEHM.waveform.EFPE import pyEFPE
        except ImportError:
            self.skipTest('no pyEFPEHM')
        from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q
        f = numpy.linspace(.004, .03, 2001)
        worst = 0
        for inc in (.4, 1.6, 2.5):
            p = dict(mass1=float(mass1_from_mchirp_q(30, 8.)),
                     mass2=float(mass2_from_mchirp_q(30, 8.)), spin1z=.3,
                     spin2z=-.2, eccentricity=.1, inclination=inc, phase=.4,
                     mean_anomaly=1.3, f22_start=.012, f22_ref=.012,
                     f22_end=.013, distance=100., Amplitude_tol=1e-12,
                     Compute_hlm_Modes=True)
            mdl = pyEFPE(p)
            pol = mdl.generate_modes(f)['modes']
            hl = mdl.generate_hlm_modes(f)['modes']
            for key, md in pol.items():
                l, m, _ = (int(v) for v in key)
                H = numpy.asarray(hl[key]['hlm'])[:, m + l]
                ap, ac = projector(l, m, inc, mdl.phi_JN)
                rec = numpy.array([H * numpy.conj(ap), H * numpy.conj(ac)])
                P = numpy.asarray(md['polarizations'])
                worst = max(worst, numpy.linalg.norm(rec - P)
                            / numpy.linalg.norm(P))
        self.assertLess(worst, 1e-12)

    def test_22_weights_are_orientation_weights(self):
        from pycbc.tmpltbank.fstatistic_metric import orientation_weights
        a = numpy.conj(projector(2, 2, 0.)[0])
        for io in (.2, 1., 2.6):
            for ps in (.1, .9):
                w = pattern_weights(2, 2, io, ps) / a
                numpy.testing.assert_allclose(
                    w, numpy.array(orientation_weights(io, ps)), atol=1e-12)


class TestMarginal(unittest.TestCase):
    def setUp(self):
        self.prior = DistancePowerLaw(.1, 1e5, scale=100.)
        self.nodes = OrientationNodes(24, 12)

    def test_reduces_to_22(self):
        """Modes all of pattern (2,2): equal to mean_anomaly_log_bayes on
        the face-on templates (``y_face = conj(Ap22(0)) y``)."""
        rng = numpy.random.default_rng(3)
        keys = [(2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4)]
        y = _templates(keys, 400, rng, 1.)
        d = _signal(y, keys, 40., 1.0, .3, .8, 2.) \
            + (rng.normal(size=400) + 1j * rng.normal(size=400))
        z, gram = _zg(y, d)
        a = numpy.conj(projector(2, 2, 0.)[0])
        z_face, g_face = numpy.conj(a) * z, abs(a) ** 2 * gram
        old = mean_anomaly_log_bayes(z_face[..., None], g_face,
                                     mode_offsets(keys), self.prior,
                                     self.nodes, n_l0=16)[0]
        new = hm_log_bayes(z[..., None], gram, keys, self.prior, self.nodes,
                           n_l0=16, n_phi=1024)[0]
        self.assertGreater(float(old[0]), 10.)
        self.assertLess(abs(float(old[0]) - float(new[0])), 1e-6)

    def test_pattern_sums_match_explicit(self):
        """X and g from the pattern components against the explicit
        modes' sums at arbitrary angles."""
        from pycbc.filter import hm_marginal as hm
        rng = numpy.random.default_rng(4)
        keys = [(2, 2, 2), (2, 2, 3), (2, 0, 1), (3, 3, 3), (2, 1, 1)]
        y = _templates(keys, 100, rng, .7)
        d = rng.normal(size=100) + 1j * rng.normal(size=100)
        z, gram = _zg(y, d)
        P = hm.PatternSet(keys)
        zg, Gg, ang = hm.hm_components(z[..., None], gram[..., None], P, 8)
        W = P.weights(self.nodes.cos_iota, self.nodes.psi)
        phis = numpy.array([.3, 1.7, 4.4])
        for j in (1, 5):
            x, gam = hm._node_terms(zg[j:j + 1], Gg[j:j + 1], W, numpy)
            for n in (7, 150):
                X1, g1 = hm._xg_at(x[0][:, n:n + 1], gam[0][:, :, n:n + 1],
                                   P.m, phis[:, None, None], numpy)
                X2, g2 = hm._exact_xg(
                    z, gram, P, numpy.full(3, ang[j]),
                    numpy.full(3, numpy.arccos(self.nodes.cos_iota[n])),
                    numpy.full(3, self.nodes.psi[n]), phis)
                numpy.testing.assert_allclose(X1.ravel(), X2, atol=1e-12)
                numpy.testing.assert_allclose(g1.ravel(), g2, atol=1e-12)

    def test_against_brute_force(self):
        """Mixed patterns with strong higher modes: the marginalisation
        against an explicit integral over cos(iota), psi in [0, pi), phi and
        l0, with signed amplitudes and X, g from the template vectors."""
        rng = numpy.random.default_rng(5)
        keys = [(2, 2, 2), (2, 2, 3), (3, 3, 3), (2, 1, 1), (4, 4, 4)]
        n = 120
        y = _templates(keys, n, rng, .5)
        # loud enough that ln B is tens (at ln B ~ 0 a wrong Gram phase went unnoticed)
        truth = dict(h=25., iota=1.1, psi=.4, phi=2.2, l0=1.)
        d = _signal(y, keys, truth['h'], truth['iota'], truth['psi'],
                    truth['phi'], truth['l0']) \
            + (rng.normal(size=n) + 1j * rng.normal(size=n))
        z, gram = _zg(y, d)
        n_l0, n_phi = 8, 256
        new = float(hm_log_bayes(z[..., None], gram, keys, self.prior,
                                 self.nodes, n_l0=n_l0, n_phi=n_phi)[0][0])
        # brute force: Gauss-Legendre cos(iota) (as the nodes), psi uniform
        # over [0, pi), phi and l0 uniform; X and g from the vectors
        t, wt = numpy.polynomial.legendre.leggauss(24)
        psis = (numpy.arange(24) + .5) * numpy.pi / 24
        phis = 2 * numpy.pi * numpy.arange(n_phi) / n_phi
        l0s = 2 * numpy.pi * numpy.arange(n_l0) / n_l0
        terms = []
        for ci, wc in zip(t, wt):
            for ps in psis:
                for ph in phis:
                    for l0 in l0s:
                        s = _signal(y, keys, 1., numpy.arccos(ci), ps, ph, l0)
                        X = float(numpy.real(numpy.vdot(s, d)))
                        g = float(numpy.real(numpy.vdot(s, s)))
                        terms.append((X, g, numpy.log(.5 * wc / 24 / n_phi
                                                      / n_l0)))
        X, g, lw = (numpy.array(v) for v in zip(*terms))
        lj = log_amplitude_marginal(X, g, self.prior, order=96,
                                    kernel='exp') + lw
        top = lj.max()
        brute = float(top + numpy.log(numpy.sum(numpy.exp(lj - top))))
        self.assertGreater(brute, 10.)
        self.assertLess(abs(new - brute), 1e-3)

    def test_peak_rule(self):
        """The peak-centred phi quadrature against the trapezoid rule on
        4096 points, where the higher modes are as strong as (2,2) (six or
        seven peaks, some too narrow for 64 points) and at high SNR."""
        keys = [(2, 2, 2), (2, 2, 3), (3, 3, 3), (2, 1, 1), (4, 4, 4)]
        for seed, scale, h in ((11, 1.0, 15.), (5, .4, 50.), (11, 1.0, 100.)):
            rng = numpy.random.default_rng(seed)
            y = _templates(keys, 200, rng, scale)
            d = _signal(y, keys, h, 1.1, .4, 2.2, 1.) \
                + (rng.normal(size=200) + 1j * rng.normal(size=200))
            z, gram = _zg(y, d)
            ref = hm_log_bayes(z[..., None], gram, keys, self.prior,
                               self.nodes, n_l0=4, n_phi=4096)[0]
            fast = hm_log_bayes(z[..., None], gram, keys, self.prior,
                                self.nodes, n_l0=4, phi_rule='peaks')[0]
            self.assertLess(abs(float(fast[0]) - float(ref[0])), 1e-6)

    def test_draws(self):
        """Draws of hm_draw_batch: ln B equal to hm_log_bayes', and the drawn
        values' log likelihood within a few of its largest (a wrong Gram
        phase drew samples 130 below)."""
        from pycbc.filter.hm_marginal import (hm_draw_batch, PatternSet,
                                              _exact_xg)
        rng = numpy.random.default_rng(3)
        keys = [(2, 2, 2), (2, 2, 3), (2, 0, 1), (3, 3, 3), (2, 1, 1)]
        y = _templates(keys, 200, rng, .5)
        d = _signal(y, keys, 40., 1.1, .4, 2.2, 1.) \
            + (rng.normal(size=200) + 1j * rng.normal(size=200))
        z, gram = _zg(y, d)
        ref = hm_log_bayes(z[..., None], gram, keys, self.prior, self.nodes,
                           n_l0=8, phi_rule='peaks')[0]
        P = PatternSet(keys)
        lnl = []
        for r in range(6):
            lb, dd = hm_draw_batch(z[..., None], gram[..., None], keys,
                                   self.prior, self.nodes,
                                   [numpy.random.default_rng(r)], n_l0=8)
            self.assertLess(abs(float(lb[0]) - float(ref[0])), 1e-9)
            q = dd[0]
            self.assertTrue(0 <= q['polarization'] < numpy.pi)
            X, g = _exact_xg(z, gram, P, numpy.array([q['mean_anomaly']]),
                             numpy.array([q['inclination']]),
                             numpy.array([q['polarization']]),
                             numpy.array([q['coa_phase']]))
            lnl.append(float(q['amplitude'] * X[0]
                             - .5 * q['amplitude'] ** 2 * g[0]))
            self.assertLess(abs(lnl[-1] - q['loglr']), 1e-6)
        # the best fit over the angles and h is near ln B + a few
        self.assertGreater(numpy.median(lnl), float(ref[0]) - 5.)


if __name__ == '__main__':
    unittest.main()
