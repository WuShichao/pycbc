# Copyright (C) 2026  Shichao Wu, Sumit Kumar
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
"""Unit tests for pycbc.population.redshift_models and population_models."""

import unittest
import warnings

import astropy.units as u
import numpy as np
from scipy import integrate as scipy_integrate
from scipy.interpolate import PchipInterpolator

from pycbc import population
from pycbc.cosmology import get_cosmology
from pycbc.population import (
    population_models as pm,
    redshift_models as rm,
)
from pycbc.transforms import CustomTransform


class _UniformDelayDistribution(rm.BaseDelayTimeDistribution):
    """Minimal custom delay distribution without analytic ppf (tests fallback)."""

    def __init__(self, td_min=0.1, td_max=3.0):
        self._td_min = float(td_min)
        self._td_max = float(td_max)

    @property
    def min_delay(self):
        return self._td_min

    @property
    def max_delay(self):
        return self._td_max

    def pdf(self, tau):
        tau_arr = np.asarray(tau, dtype=float)
        out = np.zeros_like(tau_arr, dtype=float)
        valid = (tau_arr >= self._td_min) & (tau_arr <= self._td_max)
        out[valid] = 1.0 / (self._td_max - self._td_min)
        return out


class TestRedshiftEvolutionModels(unittest.TestCase):
    """Tests for BaseRedshiftEvolution and concrete SFR / power-law classes."""

    def setUp(self):
        self.cosmo = get_cosmology()

    def test_public_exports(self):
        expected = [
            "BaseDelayTimeDistribution",
            "BaseRedshiftEvolution",
            "GRB2008SFR",
            "GaussianDelayDistribution",
            "InverseDelayDistribution",
            "LogNormalDelayDistribution",
            "MadauDickinson2014SFR",
            "MadauFragos2017SFR",
            "PowerLawDelayDistribution",
            "PowerLawRedshift",
            "SFRTimeDelayRedshift",
            "power_law_redshift",
            "sfr_grb_2008_redshift",
            "sfr_madau_dickinson_2014_redshift",
            "sfr_madau_fragos_2017_redshift",
        ]
        for name in expected:
            self.assertIn(name, rm.__all__)
            self.assertTrue(hasattr(population, name))

    def test_sfr_formulas_and_wrappers(self):
        z = np.array([0.0, 0.5, 1.5, 3.0, 8.0])

        # Madau & Dickinson (2014)
        expected_md14 = 0.015 * (1.0 + z) ** 2.7 / (1.0 + ((1.0 + z) / 2.9) ** 5.6)
        np.testing.assert_allclose(pm.sfr_madau_dickinson_2014(z), expected_md14)
        np.testing.assert_allclose(
            rm.sfr_madau_dickinson_2014_redshift.psi_z(z), expected_md14
        )

        # Madau & Fragos (2017) high and low modes
        expected_mf17_high = (
            0.66 * 0.015 * (1.0 + z) ** 2.7 / (1.0 + ((1.0 + z) / 3.0) ** 5.35)
        )
        expected_mf17_low = (
            0.66 * 0.015 * (1.0 + z) ** 2.6 / (1.0 + ((1.0 + z) / 3.2) ** 6.2)
        )
        np.testing.assert_allclose(
            pm.sfr_madau_fragos_2017(z, mode="high"), expected_mf17_high
        )
        np.testing.assert_allclose(
            pm.sfr_madau_fragos_2017(z, mode="low"), expected_mf17_low
        )
        with self.assertRaises(ValueError):
            pm.sfr_madau_fragos_2017(z, mode="invalid")

        # GRB 2008
        eta = -10.0
        expected_grb = 0.02 * (
            (1.0 + z) ** (3.4 * eta)
            + ((1.0 + z) / 5000.0) ** (-0.3 * eta)
            + ((1.0 + z) / 9.0) ** (-3.5 * eta)
        ) ** (1.0 / eta)
        np.testing.assert_allclose(pm.sfr_grb_2008(z), expected_grb)

    def test_power_law_pdf_normalization_and_shapes(self):
        model = rm.PowerLawRedshift(z_max=2.0, num_zbins=500, cosmology=self.cosmo)
        z_grid = np.linspace(0.0, 2.0, 1000)
        pdf = model(z_grid, k=2.7)
        integral = np.trapezoid(pdf, z_grid)
        self.assertAlmostEqual(integral, 1.0, places=4)

        # Multidimensional array shape preservation
        z_2d = np.array([[0.1, 0.5], [1.0, 1.5]])
        self.assertEqual(model.prob_redshift(z_2d, k=2.7).shape, (2, 2))
        self.assertEqual(model.dVT_dz(z_2d, k=2.7).shape, (2, 2))
        self.assertEqual(model.comoving_volume(z_2d).shape, (2, 2))
        self.assertEqual(model.comoving_spacetime_volume(z_2d).shape, (2, 2))

        # Scalar returns float / scalar Quantity
        p_scalar = model.prob_redshift(0.5, k=2.7)
        self.assertIsInstance(p_scalar, float)
        self.assertGreater(p_scalar, 0.0)

    def test_out_of_bounds_z_max_consistency(self):
        model = rm.PowerLawRedshift(z_max=2.0, num_zbins=200, cosmology=self.cosmo)

        # Both scalar and array inputs beyond z_max must return zero dVT_dz and PDF
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dvt_scalar = model.dVT_dz(5.0, k=2.0)
            dvt_array = model.dVT_dz(np.array([5.0]), k=2.0)
            pdf_scalar = model.prob_redshift(5.0, k=2.0)
            pdf_array = model.prob_redshift(np.array([5.0]), k=2.0)

        self.assertGreaterEqual(len(caught), 4)
        self.assertEqual(dvt_scalar.to_value(u.Gpc**3), 0.0)
        self.assertEqual(dvt_array.to_value(u.Gpc**3)[0], 0.0)
        self.assertEqual(pdf_scalar, 0.0)
        self.assertEqual(pdf_array[0], 0.0)

    def test_dvc_cache_does_not_alias_caller_array(self):
        model = rm.PowerLawRedshift(z_max=2.0, num_zbins=200, cosmology=self.cosmo)
        z_arr = np.array([0.2, 0.5, 1.0])
        dvt_1 = model.dVT_dz(z_arr, k=0.0).to_value(u.Gpc**3)
        # Mutate caller array in place
        z_arr[:] = np.array([0.4, 0.8, 1.5])
        dvt_2 = model.dVT_dz(z_arr, k=0.0).to_value(u.Gpc**3)
        self.assertFalse(np.allclose(dvt_1, dvt_2))

    def test_comoving_spacetime_volume_accuracy_and_units(self):
        model = rm.PowerLawRedshift(z_max=2.0, num_zbins=200, cosmology=self.cosmo)

        for z_val in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
            computed = model.comoving_spacetime_volume(z_val)
            self.assertEqual(computed.unit, u.Gpc**3)

            z_fine = np.linspace(0.0, z_val, 4000)
            dvc_fine = (
                4.0
                * np.pi
                * self.cosmo.differential_comoving_volume(z_fine).to_value(
                    u.Gpc**3 / u.sr
                )
            )
            exact = np.trapezoid(dvc_fine / (1.0 + z_fine), z_fine)
            rel_err = abs(computed.value - exact) / exact
            self.assertLess(rel_err, 2e-3, msg=f"High error at z={z_val}: {rel_err}")

        # Verify custom z_grid_cosmo is honored rather than silently ignored
        custom_cosmo_grid = np.linspace(0.0, 5.0, 101)
        custom_model = rm.PowerLawRedshift(
            z_max=2.0,
            z_max_cosmo=5.0,
            z_grid_cosmo=custom_cosmo_grid,
            cosmology=self.cosmo,
        )
        self.assertEqual(custom_model._z_grid_cosmo.size, 101)

    def test_total_4volume_and_validation(self):
        model = rm.PowerLawRedshift(z_max=2.0, num_zbins=200, cosmology=self.cosmo)
        vt = model.total_4volume(1.0 * u.yr, k=0.0)
        self.assertTrue(vt.unit.is_equivalent(u.Gpc**3 * u.yr))
        self.assertAlmostEqual(
            vt.to_value(u.Gpc**3 * u.yr),
            model.comoving_spacetime_volume(2.0).to_value(u.Gpc**3),
            delta=0.5,
        )

        with self.assertRaises(u.UnitTypeError):
            model.total_4volume(1.0 * u.m, k=0.0)
        with self.assertRaises(u.UnitTypeError):
            model.total_4volume(1.0, k=0.0)

        with self.assertRaises(ValueError):
            model.prob_redshift(-0.1, k=0.0)
        with self.assertRaises(ValueError):
            model.prob_redshift(np.nan, k=0.0)


class TestDelayTimeDistributions(unittest.TestCase):
    """Tests for BaseDelayTimeDistribution and all built-in subclasses."""

    def test_pdf_normalization_cdf_ppf_and_sampling(self):
        distributions = [
            rm.PowerLawDelayDistribution(td_min=0.02, td_max=13.0, alpha=0.81),
            rm.PowerLawDelayDistribution(td_min=0.02, td_max=13.0, alpha=1.0),
            rm.PowerLawDelayDistribution(td_min=0.05, td_max=10.0, alpha=1.5),
            rm.InverseDelayDistribution(td_min=0.02, td_max=13.0),
            rm.LogNormalDelayDistribution(
                td_min=0.02, td_max=13.0, t_peak=2.9, sigma=0.2
            ),
            rm.LogNormalDelayDistribution(
                td_min=0.02, td_max=13.0, t_peak=2.0, sigma=0.05
            ),
            rm.GaussianDelayDistribution(
                td_min=0.02, td_max=13.0, t_mean=2.0, sigma=0.3
            ),
            rm.GaussianDelayDistribution(
                td_min=0.02, td_max=13.0, t_mean=2.0, sigma=0.005
            ),
            _UniformDelayDistribution(td_min=0.1, td_max=3.0),
        ]

        rng = np.random.RandomState(12345)
        for dist in distributions:
            with self.subTest(dist=type(dist).__name__):
                # Support boundaries
                self.assertEqual(float(dist.pdf(dist.min_delay * 0.5)), 0.0)
                self.assertEqual(float(dist.pdf(dist.max_delay * 1.5)), 0.0)
                self.assertAlmostEqual(float(dist.cdf(dist.min_delay)), 0.0, places=10)
                self.assertAlmostEqual(float(dist.cdf(dist.max_delay)), 1.0, places=10)

                # PDF integrates to 1.0 (use quad with midpoint near peak for narrow)
                points = []
                if hasattr(dist, "_t_mean"):
                    points = [dist._t_mean]
                elif hasattr(dist, "_t_peak"):
                    points = [dist._t_peak]
                integral, _ = scipy_integrate.quad(
                    lambda t, d=dist: float(d.pdf(t)),
                    dist.min_delay,
                    dist.max_delay,
                    points=points,
                    limit=500,
                )
                self.assertAlmostEqual(integral, 1.0, places=5)

                # ppf inverts cdf
                quantiles = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
                taus = dist.ppf(quantiles)
                np.testing.assert_allclose(dist.cdf(taus), quantiles, atol=1e-3)

                # Samples stay in bounds with no boundary pile-up at td_min
                samples = dist.sample(size=5000, random_state=rng)
                self.assertEqual(samples.shape, (5000,))
                self.assertTrue(np.all(samples >= dist.min_delay))
                self.assertTrue(np.all(samples <= dist.max_delay))
                self.assertLess(np.mean(samples == dist.min_delay), 0.005)
                # Sample median matches ppf(0.5) within Monte Carlo tolerance
                self.assertAlmostEqual(
                    float(dist.cdf(np.median(samples))), 0.5, delta=0.03
                )

    def test_delay_parameter_validation(self):
        with self.assertRaises(ValueError):
            rm.PowerLawDelayDistribution(td_min=0.0, td_max=10.0)
        with self.assertRaises(ValueError):
            rm.InverseDelayDistribution(td_min=5.0, td_max=2.0)
        with self.assertRaises(ValueError):
            rm.GaussianDelayDistribution(td_min=0.02, td_max=13.0, sigma=-0.1)
        with self.assertRaises(ValueError):
            rm.LogNormalDelayDistribution(td_min=0.02, td_max=13.0, t_peak=-1.0)

    def test_extreme_tail_stability_ndtr(self):
        # 12-sigma right tail (previously underflowed with erf)
        g_right = rm.GaussianDelayDistribution(
            td_min=2.5, td_max=13.0, t_mean=0.1, sigma=0.2
        )
        # 10-sigma left tail
        g_left = rm.GaussianDelayDistribution(
            td_min=0.02, td_max=1.0, t_mean=3.0, sigma=0.2
        )
        # Far right tail for LogNormal
        ln_right = rm.LogNormalDelayDistribution(
            td_min=5.0, td_max=13.0, t_peak=0.2, sigma=0.3
        )
        for dist in (g_right, g_left, ln_right):
            with self.subTest(dist=type(dist).__name__):
                q = np.array([0.05, 0.5, 0.95])
                taus = dist.ppf(q)
                self.assertTrue(np.all(np.isfinite(taus)))
                np.testing.assert_allclose(dist.cdf(taus), q, atol=1e-6)
                integral, _ = scipy_integrate.quad(
                    lambda t, d=dist: float(d.pdf(t)),
                    dist.min_delay,
                    dist.max_delay,
                    limit=200,
                )
                self.assertAlmostEqual(integral, 1.0, places=5)


class TestSFRTimeDelayRedshift(unittest.TestCase):
    """Tests for SFRTimeDelayRedshift convolution accuracy, caching, and API."""

    def setUp(self):
        self.cosmo = get_cosmology()
        self.sfr = rm.MadauDickinson2014SFR(z_max=10.0, cosmology=self.cosmo)

    def _reference_convolution(self, sfr_model, td_model, z_eval, z_fmax=15.0):
        """High-precision reference convolution via scipy.integrate.quad in tau."""
        zf_grid = np.concatenate(([0.0], np.geomspace(1e-5, z_fmax, 4000)))
        tf_grid = self.cosmo.lookback_time(zf_grid).to_value(u.Gyr)
        ln1p_zf = PchipInterpolator(tf_grid, np.log1p(zf_grid))
        t_max_f = float(tf_grid[-1])

        out = []
        for zm in z_eval:
            tm = float(self.cosmo.lookback_time(zm).to_value(u.Gyr))
            tau_hi = min(td_model.max_delay, t_max_f - tm)
            if tau_hi <= td_model.min_delay:
                out.append(0.0)
                continue
            pts = []
            if (
                hasattr(td_model, "_t_mean")
                and td_model.min_delay < td_model._t_mean < tau_hi
            ):
                pts = [td_model._t_mean]
            elif (
                hasattr(td_model, "_t_peak")
                and td_model.min_delay < td_model._t_peak < tau_hi
            ):
                pts = [td_model._t_peak]
            val, _ = scipy_integrate.quad(
                lambda tau, t_m=tm: float(
                    sfr_model.psi_z(np.expm1(ln1p_zf(t_m + tau))) * td_model.pdf(tau)
                ),
                td_model.min_delay,
                tau_hi,
                points=pts,
                epsrel=1e-7,
                limit=500,
            )
            out.append(val)
        return np.array(out)

    def test_convolution_accuracy_all_delay_models(self):
        z_eval = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
        td_models = [
            rm.InverseDelayDistribution(td_min=0.02, td_max=13.0),
            rm.PowerLawDelayDistribution(td_min=0.02, td_max=13.0, alpha=0.81),
            rm.LogNormalDelayDistribution(
                td_min=0.02, td_max=13.0, t_peak=2.9, sigma=0.2
            ),
            rm.GaussianDelayDistribution(
                td_min=0.02, td_max=13.0, t_mean=2.0, sigma=0.05
            ),
            _UniformDelayDistribution(td_min=0.1, td_max=3.0),
        ]

        for td_model in td_models:
            with self.subTest(td_model=type(td_model).__name__):
                model = rm.SFRTimeDelayRedshift(
                    sfr_model=self.sfr,
                    time_delay_model=td_model,
                    z_max=5.0,
                    num_zbins=200,
                    cosmology=self.cosmo,
                    z_formation_max=15.0,
                )
                computed = model.psi_z(z_eval)
                expected = self._reference_convolution(
                    self.sfr, td_model, z_eval, z_fmax=15.0
                )
                np.testing.assert_allclose(computed, expected, rtol=2e-3, atol=1e-7)

                # Normalized redshift PDF integrates to 1
                zg = np.linspace(0.0, 5.0, 500)
                self.assertAlmostEqual(
                    np.trapezoid(model.prob_redshift(zg), zg), 1.0, places=3
                )

    def test_parameter_forwarding_validation_and_bounded_cache(self):
        model = rm.SFRTimeDelayRedshift(
            sfr_model=self.sfr,
            time_delay_model=rm.InverseDelayDistribution(),
            z_max=3.0,
            num_zbins=50,
            cosmology=self.cosmo,
            z_formation_max=10.0,
        )
        v1 = model.psi_z(1.0, gamma=2.7)
        v2 = model.psi_z(1.0, gamma=3.5)
        self.assertNotAlmostEqual(v1, v2)

        with self.assertRaises(ValueError):
            model.psi_z(-0.5)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.psi_z(1.0, nonexistent_param=42.0)
        self.assertTrue(any("Unknown parameter" in str(w.message) for w in caught))

        # Bounded cache does not grow indefinitely
        model._max_cache_size = 3
        for g in (1.0, 1.5, 2.0, 2.5, 3.0):
            model.psi_z(1.0, gamma=g)
        self.assertLessEqual(len(model._psi_z_cache), 3)

    def test_inherits_sfr_model_cosmology(self):
        custom_cosmo = get_cosmology(cosmology="Planck15")
        sfr_custom = rm.MadauDickinson2014SFR(z_max=5.0, cosmology=custom_cosmo)
        model = rm.SFRTimeDelayRedshift(
            sfr_model=sfr_custom,
            time_delay_model=rm.InverseDelayDistribution(),
            z_max=5.0,
            num_zbins=50,
            z_formation_max=10.0,
        )
        self.assertIs(model.cosmology, custom_cosmo)


class TestPopulationModelsBackwardCompatibility(unittest.TestCase):
    """Regression tests for pycbc.population.population_models and .ini usage."""

    def test_merger_rate_density_all_models_and_maxz20(self):
        rho_local = 16.53e-9
        for td_name in ("inverse", "power_law", "log_normal", "gaussian"):
            with self.subTest(td_model=td_name):
                rate_fn = pm.merger_rate_density(
                    pm.sfr_madau_dickinson_2014,
                    td_name,
                    rho_local,
                    maxz=20,
                    npoints=200,
                )
                self.assertAlmostEqual(float(rate_fn(0.0)), rho_local, places=14)
                self.assertGreater(float(rate_fn(1.0)), 0.0)
                self.assertGreaterEqual(float(rate_fn(20.0)), 0.0)

        # Custom callable SFR and custom z_array not starting at 0
        rate_custom = pm.merger_rate_density(
            lambda z: (1.0 + np.asarray(z)) ** 2.0,
            "inverse",
            rho_local,
            z_array=np.linspace(0.5, 5.0, 50),
            td_min=0.05,
        )
        self.assertGreater(float(rate_custom(1.0)), 0.0)

    def test_ini_custom_transform_rate_to_distance_pipeline(self):
        """Regression test matching PyCBC inference .ini custom transform usage."""
        rho_local = 16.53e-9
        rate_den = pm.merger_rate_density(
            pm.sfr_madau_dickinson_2014,
            "inverse",
            rho_local,
            maxz=20,
            npoints=200,
        )
        coal_rate = pm.coalescence_rate(rate_den, maxz=20, npoints=200)
        max_rate = pm.total_rate_upto_redshift(20.0, coal_rate)
        self.assertGreater(max_rate, 0.0)

        # Evaluate via CustomTransform exactly as in an inference .ini file
        expr = (
            "distance_from_rate(total_rate, "
            "coalescence_rate(merger_rate_density(sfr_madau_dickinson_2014, "
            "'inverse', 16.53*10**(-9), maxz=20, npoints=200), "
            "maxz=20, npoints=200), maxz=20, npoints=200)"
        )
        transform = CustomTransform(
            input_args=["total_rate"],
            output_args=["distance"],
            transform_functions={"distance": expr},
        )
        out = transform.transform({"total_rate": 0.5 * max_rate})
        self.assertGreater(float(out["distance"]), 0.0)
        self.assertTrue(np.isfinite(float(out["distance"])))

    def test_deprecated_legacy_functions_still_work(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dt_dz = pm.diff_lookback_time(1.0)
            pt = pm.p_tau(1.0, td_model="inverse")
        self.assertTrue(any(w.category is DeprecationWarning for w in caught))
        self.assertGreater(float(dt_dz), 0.0)
        self.assertGreater(float(pt), 0.0)


if __name__ == "__main__":
    unittest.main()
