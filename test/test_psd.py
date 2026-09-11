# Copyright (C) 2012  Tito Dal Canton, Josh Willis
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

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
'''
These are the unittests for the pycbc PSD module.
'''

import os
import tempfile
import pycbc
import pycbc.psd
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.fft import ifft
from pycbc.fft.fftw import set_measure_level
import unittest
import numpy
from utils import parse_args_all_schemes, simple_exit
set_measure_level(0)

_scheme, _context = parse_args_all_schemes("PSD")

class TestPSD(unittest.TestCase):
    def setUp(self):
        self.scheme = _scheme
        self.context = _context
        self.psd_len = 1024
        self.psd_delta_f = 0.1
        self.psd_low_freq_cutoff = 10.
        # generate 1/f noise for testing PSD estimation
        noise_size = 524288
        sample_freq = 4096.
        delta_f = sample_freq / noise_size
        numpy.random.seed(132435)
        fd_size = noise_size // 2 + 1
        noise = numpy.random.normal(loc=0, scale=1, size=fd_size) + \
            1j * numpy.random.normal(loc=0, scale=1, size=fd_size)
        noise_model = 1. / numpy.linspace(1., 100., fd_size)
        noise *= noise_model / numpy.sqrt(delta_f) / 2
        noise[0] = noise[0].real
        noise_fs = FrequencySeries(noise, delta_f=delta_f)
        self.noise = TimeSeries(numpy.zeros(noise_size), delta_t=1./sample_freq)
        ifft(noise_fs, self.noise)

    def test_analytical(self):
        """Basic test of lalsimulation's analytical noise PSDs"""
        with self.context:
            psd_list = pycbc.psd.analytical.get_lalsim_psd_list()
            self.assertTrue(psd_list)
            for psd_name in psd_list:
                psd = pycbc.psd.analytical.from_string(psd_name, self.psd_len,
                                    self.psd_delta_f, self.psd_low_freq_cutoff)
                psd_min = psd.min()
                self.assertTrue(psd_min >= 0,
                                          msg=(psd_name + ': negative values'))
                self.assertTrue(psd.min() < 1e-40,
                                msg=(psd_name + ': unreasonably high minimum'))

    def test_read(self):
        """Test reading PSDs from text files"""
        test_data = numpy.zeros((self.psd_len, 2))
        test_data[:, 0] = numpy.linspace(0.,
                           (self.psd_len - 1) * self.psd_delta_f, self.psd_len)
        test_data[:, 1] = numpy.sqrt(test_data[:, 0])
        file_desc, file_name = tempfile.mkstemp()
        os.close(file_desc)
        numpy.savetxt(file_name, test_data)
        test_data[test_data[:, 0] < self.psd_low_freq_cutoff, 1] = 0.
        with self.context:
            psd = pycbc.psd.read.from_txt(file_name, self.psd_len,
                                    self.psd_delta_f, self.psd_low_freq_cutoff, is_asd_file=True)
            self.assertAlmostEqual(abs(psd - test_data[:, 1] ** 2).max(), 0)
        os.unlink(file_name)

    def test_estimate_welch(self):
        """Test estimating PSDs from data using Welch's method"""
        for seg_len in (2048, 4096, 8192):
            noise_model = (numpy.linspace(1., 100., seg_len//2 + 1)) ** (-2)
            for seg_stride in (seg_len, seg_len//2):
                for method in ('mean', 'median', 'median-mean'):
                    with self.context:
                        psd = pycbc.psd.welch(self.noise, seg_len=seg_len, \
                            seg_stride=seg_stride, avg_method=method)
                        error = (psd.numpy() - noise_model) / noise_model
                    err_rms = numpy.sqrt(numpy.mean(error ** 2))
                    self.assertTrue(err_rms < 0.2,
                        msg='seg_len=%d seg_stride=%d method=%s -> rms=%.3f' % \
                        (seg_len, seg_stride, method, err_rms))

    def test_truncation(self):
        """Test inverse PSD truncation"""
        for seg_len in (2048, 4096, 8192):
            noise_model = (numpy.linspace(1., 100., seg_len//2 + 1)) ** (-2)
            for max_len in (1024, 512, 256):
                with self.context:
                    psd = pycbc.psd.welch(self.noise, seg_len=seg_len, \
                                          seg_stride=seg_len//2, avg_method='mean')
                    psd_trunc = pycbc.psd.inverse_spectrum_truncation(
                            psd, max_len,
                            low_frequency_cutoff=self.psd_low_freq_cutoff)
                    freq = psd.sample_frequencies.numpy()
                    error = (psd.numpy() - noise_model) / noise_model
                error = error[freq > self.psd_low_freq_cutoff]
                err_rms = numpy.sqrt(numpy.mean(error ** 2))
                self.assertTrue(err_rms < 0.1,
                                msg='seg_len=%d max_len=%d -> rms=%.3f' \
                                % (seg_len, max_len, err_rms))


class TestSpacePSD(unittest.TestCase):
    """Network-free checks of the space-borne analytical PSD models."""

    def setUp(self):
        self.length = 3001
        self.delta_f = 1e-5
        self.low_freq_cutoff = 1e-4

    def test_extragalactic_dwd_amplitude(self):
        """Each paper's fit must reproduce its own Omega_gw at 1 mHz."""
        from pycbc.cosmology import get_cosmology
        from pycbc.psd.analytical_space import extragalactic_dwd_fit_lisa

        h0 = get_cosmology(None).H0.si.value
        # hofman2024 quotes ~4e-12 at 1 mHz; the boileau_md_default fit
        # evaluates to 2.08e-12 there.
        for model, expect in (('hofman2024', 4.0e-12),
                              ('boileau_md_default', 2.08e-12)):
            sh = extragalactic_dwd_fit_lisa(self.length, self.delta_f,
                                            self.low_freq_cutoff, model)
            freq = sh.sample_frequencies.numpy()
            idx = int(numpy.argmin(numpy.abs(freq - 1e-3)))
            omega = (sh.numpy()[idx] * 4 * numpy.pi**2 * freq[idx]**3
                     / (3 * h0**2))
            self.assertTrue(abs(omega / expect - 1) < 0.05,
                            msg='%s: Omega(1 mHz) = %.3e' % (model, omega))

    def test_omega_gw_to_strain_psd_roundtrip(self):
        from pycbc.cosmology import get_cosmology
        from pycbc.psd.analytical_space import omega_gw_to_strain_psd

        h0 = get_cosmology(None).H0.si.value
        freq = numpy.array([1e-4, 1e-3, 1e-2])
        omega = numpy.array([1e-11, 4e-12, 1e-12])
        sh = omega_gw_to_strain_psd(freq, omega)
        back = sh * 4 * numpy.pi**2 * freq**3 / (3 * h0**2)
        self.assertTrue(numpy.allclose(back, omega, rtol=1e-12))

    def test_extragalactic_dwd_all_models(self):
        """Every registered model must evaluate, and the amplitude must be
        recovered at the break frequency."""
        from pycbc.psd import analytical_space as space

        for name, (form, par) in space.EXTRAGALACTIC_DWD_MODELS.items():
            freq = numpy.logspace(-4, -1.5, 200)
            omega = space.omega_gw_extragalactic_dwd(freq, name)
            self.assertTrue(numpy.all(numpy.isfinite(omega)), msg=name)
            self.assertTrue(numpy.all(omega > 0), msg=name)
            if form == 'smooth':
                # the smoothly broken power law equals A at the break
                at_break = space.omega_gw_extragalactic_dwd(par[1], name)
                self.assertTrue(abs(at_break/par[0] - 1) < 1e-12, msg=name)

    def test_extragalactic_dwd_unknown_model(self):
        from pycbc.psd.analytical_space import omega_gw_extragalactic_dwd

        self.assertRaises(ValueError, omega_gw_extragalactic_dwd,
                          numpy.array([1e-3]), 'not_a_model')

    def test_extragalactic_dwd_amplitude_scaling(self):
        """The strain PSD must be linear in the Omega amplitude."""
        from pycbc.psd.analytical_space import extragalactic_dwd_fit_lisa

        one = extragalactic_dwd_fit_lisa(self.length, self.delta_f,
                                         self.low_freq_cutoff,
                                         amplitude=1.72e-11).numpy()
        two = extragalactic_dwd_fit_lisa(self.length, self.delta_f,
                                         self.low_freq_cutoff,
                                         amplitude=3.44e-11).numpy()
        self.assertTrue(numpy.any(one > 0))
        keep = one > 0
        self.assertTrue(numpy.allclose(two[keep] / one[keep], 2, rtol=1e-10))


suite = unittest.TestSuite()
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestPSD))
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestSpacePSD))

if __name__ == '__main__':
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
