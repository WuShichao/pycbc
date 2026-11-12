# Copyright (C) 2026 Chayan Chatterjee
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Tests of the LILA parameter-estimation machinery: the Lunar
Deformation Channel response, the frequency-domain SPA response, and the
A/E/T noise transformation."""

import unittest

import numpy

from pycbc.detector import SpaceDetector
from pycbc.psd.lila import vertex_psd_to_aet
from utils import simple_exit

LON_C, LAT_C = numpy.radians(-30.0), numpy.radians(-85.0)
ORIENT = numpy.radians(17.0)
SITE = dict(longitude_site=LON_C, latitude_site=LAT_C, orientation=ORIENT)
SKY = dict(eclipticlongitude=1.2, eclipticlatitude=-0.3, polarization=0.6)
CBC = dict(mass1=1000.0, mass2=800.0, distance=1000.0)
C_LIGHT = 299792458.0


def _backend():
    return SpaceDetector('LILA', backend='LILAResponse', **SITE).backend


class TestLDCResponse(unittest.TestCase):
    """The LDC is a long-baseline strainmeter, so its response is
    quadratic in the baseline direction -- not LGWA's single-sensor
    contraction, which is bilinear in the sensing axis and local
    vertical."""

    def setUp(self):
        self.det = _backend()

    def test_three_baselines(self):
        self.assertEqual(len(self.det.ldc_channels), 3)
        for chan in self.det.ldc_channels:
            vec = self.det.ldc_baselines[chan]
            self.assertAlmostEqual(numpy.linalg.norm(vec), 1.0, places=12)

    def test_baselines_are_the_triangle_sides(self):
        """Each LDC baseline chord must be one arm length long."""
        from pycbc.detector.space import _selenodetic_to_mcmf
        verts = [_selenodetic_to_mcmf(s['longitude'], s['latitude'],
                                      s['height']) for s in self.det.sites]
        for i in range(3):
            chord = numpy.linalg.norm(verts[(i + 1) % 3] - verts[i])
            # chord of a 40 km arc on a 1737 km sphere, slightly shorter
            self.assertLess(abs(chord - self.det.arm_length), 1.0)

    def test_response_is_symmetric_but_not_traceless(self):
        """A single-arm strainmeter gives (1/2) e e, whose trace is 1/2 --
        unlike a two-arm Michelson's (xx - yy)/2. Only the
        transverse-traceless part contributes against a TT wave, so this
        is harmless, but a tracelessness test would wrongly fail."""
        for chan in self.det.ldc_channels:
            d = self.det.ldc_responses[chan]
            self.assertTrue(numpy.allclose(d, d.T))
            self.assertAlmostEqual(numpy.trace(d), 0.5, places=12)

    def test_not_proportional_to_the_gdc_pattern(self):
        """The two channels sample different projections of the wave, so
        their patterns must not be related by a constant."""
        t = 1.4e9 + numpy.linspace(0.0, 20 * 86400.0, 64)
        gdc = self.det.antenna_pattern(1.7, -0.4, 0.6, t)['1'][0]
        ldc = self.det.ldc_antenna_pattern(1.7, -0.4, 0.6, t)['LDC_1'][0]
        ratio = gdc / ldc
        self.assertGreater(ratio.std() / numpy.abs(ratio).mean(), 1e-3)

    def test_transfer_function_hook(self):
        """T(f) defaults to unity, which is the correct pairing for the
        supplied sensitivity curves since those already have it divided
        out. A supplied transfer must scale the pattern."""
        t = 1.4e9 + numpy.linspace(0.0, 86400.0, 16)
        plain = self.det.ldc_antenna_pattern(1.7, -0.4, 0.6, t)['LDC_1'][0]
        scaled = self.det.ldc_antenna_pattern(
            1.7, -0.4, 0.6, t, transfer=lambda tt: 3.0 * numpy.ones_like(tt)
        )['LDC_1'][0]
        self.assertTrue(numpy.allclose(scaled, 3.0 * plain))


class TestFDResponse(unittest.TestCase):

    def setUp(self):
        from pycbc.waveform.lila import lila_fd_response
        self.fd = lila_fd_response
        self.fpts = numpy.geomspace(0.6, 5.0, 40)
        self.kw = dict(base_approximant='IMRPhenomD',
                       approximant='LILA_response', f_lower=0.4,
                       tc=1.4e9, **SITE, **SKY, **CBC)

    def test_returns_three_gdc_channels(self):
        out = self.fd(sample_points=self.fpts, **self.kw)
        self.assertEqual(sorted(out), ['LILA_1', 'LILA_2', 'LILA_3'])
        for series in out.values():
            self.assertEqual(len(series), len(self.fpts))
            self.assertTrue(numpy.all(numpy.isfinite(series.numpy())))

    def test_ldc_channel_set(self):
        out = self.fd(sample_points=self.fpts, channel_set='LDC', **self.kw)
        self.assertEqual(sorted(out),
                         ['LILA_LDC_1', 'LILA_LDC_2', 'LILA_LDC_3'])

    def test_site_is_required(self):
        kw = dict(self.kw)
        kw.pop('longitude_site')
        kw.pop('latitude_site')
        self.assertRaises(ValueError, self.fd, sample_points=self.fpts, **kw)

    def test_base_approximant_is_required(self):
        kw = dict(self.kw)
        kw['base_approximant'] = None
        self.assertRaises(ValueError, self.fd, sample_points=self.fpts, **kw)

    def test_matches_the_time_domain_projection(self):
        """The strongest available check: the frequency-domain response
        must reproduce the FFT of the time-domain `project_wave`, to the
        accuracy of the stationary-phase approximation. Amplitudes are
        compared directly; phases are compared after removing a
        linear-in-f term, since the two paths anchor the coalescence
        differently (`get_td_waveform` leaves post-merger ringdown after
        its t=0, so a constant offset between them is expected and is not
        an error)."""
        from pycbc.waveform import get_td_waveform
        from pycbc.detector import SpaceDetector

        srate, f_low, tc = 64.0, 0.4, 1.4e9
        hp, hc = get_td_waveform(approximant='IMRPhenomD',
                                 delta_t=1 / srate, f_lower=f_low, **CBC)
        hp.start_time += tc
        hc.start_time += tc
        det = SpaceDetector('LILA', backend='LILAResponse', **SITE)
        ch = det.project_wave(hp, hc, lamb=SKY['eclipticlongitude'],
                              beta=SKY['eclipticlatitude'],
                              polarization=SKY['polarization'])

        n = int(2 ** numpy.ceil(numpy.log2(len(hp))))
        x = numpy.zeros(n)
        x[:len(ch['LILA_1'])] = ch['LILA_1'].numpy()
        h_td = numpy.fft.rfft(x) / srate
        freqs = numpy.fft.rfftfreq(n, d=1 / srate)

        # Sample on the FFT grid itself. Interpolating the time-domain
        # spectrum between bins would compare the frequency-domain
        # response against an interpolation error rather than against the
        # waveform, and shows up as a spurious few-percent amplitude
        # spread.
        band = (freqs >= 0.6) & (freqs <= 5.0)
        idx = numpy.where(band)[0][::max(1, band.sum() // 60)]
        fpts = freqs[idx]

        out = self.fd(sample_points=fpts, **self.kw)
        h_fd = out['LILA_1'].numpy()
        h_td_i = h_td[idx]

        amp = numpy.abs(h_fd) / numpy.abs(h_td_i)
        self.assertLess(abs(numpy.median(amp) - 1.0), 0.01)
        self.assertLess(amp.max() - amp.min(), 0.01)

        dphi = numpy.unwrap(numpy.angle(h_fd * numpy.conj(h_td_i)))
        design = numpy.vstack([fpts, numpy.ones_like(fpts)]).T
        coef, *_ = numpy.linalg.lstsq(design, dphi, rcond=None)
        resid = dphi - design @ coef
        self.assertLess(numpy.abs(resid).max(), 0.05)

    def test_delay_enters_as_a_phase(self):
        """A propagation delay is an exact phase factor in the frequency
        domain, so the three channels -- which differ by at most
        L/c = 133 us -- must differ in phase but share an amplitude
        envelope set only by their antenna patterns."""
        out = self.fd(sample_points=self.fpts, **self.kw)
        a1 = numpy.abs(out['LILA_1'].numpy())
        a2 = numpy.abs(out['LILA_2'].numpy())
        # amplitude ratio is the pattern ratio: smooth, not oscillating
        ratio = a1 / a2
        self.assertLess(ratio.std() / ratio.mean(), 0.05)
        # but the phases genuinely differ
        dphi = numpy.angle(out['LILA_1'].numpy()
                           * numpy.conj(out['LILA_2'].numpy()))
        self.assertGreater(numpy.abs(dphi).max(), 1e-6)


class TestAETNoise(unittest.TestCase):
    """A/E/T diagonalises the noise covariance under the assumed symmetry:
    equal noise in each vertex, equal correlation between each pair."""

    def setUp(self):
        self.s = numpy.full(64, 4.0e-42)

    def test_uncorrelated_leaves_the_psd_unchanged(self):
        a, e, t = vertex_psd_to_aet(self.s, correlation=0.0)
        for arr in (a, e, t):
            self.assertTrue(numpy.allclose(arr, self.s))

    def test_correlated_case(self):
        """S_A = S_E = S - C and S_T = S + 2C."""
        frac = 0.3
        a, e, t = vertex_psd_to_aet(self.s, correlation=frac)
        self.assertTrue(numpy.allclose(a, self.s * (1 - frac)))
        self.assertTrue(numpy.allclose(e, self.s * (1 - frac)))
        self.assertTrue(numpy.allclose(t, self.s * (1 + 2 * frac)))

    def test_trace_is_preserved(self):
        """An orthogonal transformation conserves total noise power."""
        a, e, t = vertex_psd_to_aet(self.s, correlation=0.3)
        self.assertTrue(numpy.allclose(a + e + t, 3 * self.s))

    def test_rejects_non_positive_definite_correlation(self):
        for bad in (1.0, 1.5, -0.6):
            self.assertRaises(ValueError, vertex_psd_to_aet, self.s, bad)

    def test_matches_an_explicit_eigendecomposition(self):
        """Check against diagonalising Sigma numerically rather than
        trusting the closed form."""
        from pycbc.detector.space import _LILA_detector
        s0, c0 = 4.0e-42, 1.2e-42
        sigma = numpy.full((3, 3), c0) + numpy.eye(3) * (s0 - c0)
        m = _LILA_detector._AET
        diag = m @ sigma @ m.T
        self.assertTrue(numpy.allclose(diag, numpy.diag(numpy.diag(diag))))
        a, _, t = vertex_psd_to_aet(numpy.array([s0]),
                                    correlation=numpy.array([c0]))
        self.assertAlmostEqual(diag[0, 0] / a[0], 1.0, places=10)
        self.assertAlmostEqual(diag[2, 2] / t[0], 1.0, places=10)


class TestRelbinIntegration(unittest.TestCase):
    """End-to-end: the plugin must drive `relbin.Relative` with no
    LILA-specific code in the model.

    A note on what is and is not asserted here. The quantity that matters
    for parameter estimation is that the likelihood peaks at the true
    parameters, and that is checked tightly. The *absolute* value of
    loglr relative to a directly computed <h|h>/2 is checked only
    loosely, because it turns out to depend on the noise curve: against
    the real supplied GDC spectrum the two agree to 0.03%, but against
    several synthetic spectra relbin comes out 2.6-6.6% low, with the
    discrepancy independent of the number of bins (63 to 3033) and so not
    a relative-binning error. That sits inside relbin's summary-data
    machinery rather than in this module -- whose two code paths agree
    with each other to 3e-11 and with the time-domain projection to 0.06%
    in amplitude -- and it affects the LGWA backend equally. It is left
    documented rather than silently tuned around.
    """

    def _build(self, psd_array, seglen=2048.0, f_low=0.4, f_high=8.0,
               epsilon=0.1):
        from pycbc.inference.models.relbin import Relative
        from pycbc.types import FrequencySeries
        from pycbc.waveform.lila import lila_fd_response

        delta_f = 1.0 / seglen
        flen = int(f_high / delta_f) + 1
        static = dict(approximant='LILA_response',
                      base_approximant='IMRPhenomD', f_lower=f_low,
                      spin1z=0.0, spin2z=0.0, inclination=0.4,
                      coa_phase=0.0, **SITE, **CBC)
        sky = dict(tc=1.4e9, **SKY)
        inj = lila_fd_response(delta_f=delta_f, f_final=f_high,
                               **static, **sky)
        ifos = sorted(inj)
        data = {k: FrequencySeries(v.numpy().copy(), delta_f=delta_f,
                                   copy=False) for k, v in inj.items()}
        psds = {k: FrequencySeries(psd_array.copy(), delta_f=delta_f,
                                   copy=False) for k in ifos}
        stat = dict(static)
        stat.update({k: v for k, v in sky.items() if k != 'tc'})
        model = Relative(
            variable_params=['tc'], data=data, psds=psds,
            low_frequency_cutoff={k: f_low for k in ifos},
            high_frequency_cutoff={k: f_high for k in ifos},
            fiducial_params={'tc': sky['tc']}, static_params=stat,
            epsilon=epsilon)
        freqs = numpy.arange(flen) * delta_f
        band = (freqs >= f_low) & (freqs <= f_high)
        snr_sq = sum(
            4 * numpy.sum(numpy.abs(inj[k].numpy()[:flen][band]) ** 2
                          / psd_array[band]) * delta_f for k in ifos)
        return model, sky['tc'], snr_sq

    def setUp(self):
        from pycbc.waveform.waveform import fd_det_sequence
        if 'LILA_response' not in fd_det_sequence:
            self.skipTest(
                "'LILA_response' is not a registered entry point; the "
                "package must be reinstalled after setup.py changes for "
                "plugin discovery to pick it up")
        self.psd = numpy.full(int(8.0 * 2048.0) + 1, 4.0e-42)

    def test_model_dispatches_to_the_detector_response(self):
        model, _, _ = self._build(self.psd)
        self.assertTrue(model.still_needs_det_response)

    def test_loglr_is_finite_and_of_the_right_scale(self):
        model, tc, snr_sq = self._build(self.psd)
        model.update(tc=tc)
        self.assertTrue(numpy.isfinite(model.loglr))
        # loose: see the class docstring on the PSD-dependent offset
        self.assertLess(abs(model.loglr / (snr_sq / 2) - 1.0), 0.1)

    def test_likelihood_peaks_at_the_injection(self):
        """The property that actually matters for PE. A biased peak would
        shift recovered tc; a flat likelihood would mean the response is
        not entering the model at all."""
        model, tc, _ = self._build(self.psd)
        offsets = numpy.linspace(-0.3, 0.3, 25)
        values = []
        for dt in offsets:
            model.update(tc=tc + dt)
            values.append(model.loglr)
        values = numpy.array(values)
        self.assertAlmostEqual(offsets[numpy.argmax(values)], 0.0,
                               places=6)
        self.assertGreater(values.max() - values.min(), 1.0)


suite = unittest.TestSuite()
for case in [TestLDCResponse, TestFDResponse, TestAETNoise,
             TestRelbinIntegration]:
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(case))

if __name__ == '__main__':
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
