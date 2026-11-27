# Copyright (C) 2025  Sumit Kumar, Shichao Wu
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
"""
Redshift evolution models and delay-time distributions for population modeling.
"""

import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict

import astropy.units as u
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtr, ndtri

from pycbc.cosmology import get_cosmology

_SQRT_2PI = np.sqrt(2.0 * np.pi)
_DEFAULT_MAX_CACHE_SIZE = 64
_FLOAT_TINY = np.finfo(float).tiny
_FLOAT_EPS = np.finfo(float).eps


def _truncnorm_bounds(z_min, z_max):
    """Compute tail-safe truncated-normal CDF/survival parameters via ``ndtr``."""
    use_right_tail = z_min > 0.0
    if use_right_tail:
        sf_min = float(ndtr(-z_min))
        sf_max = float(ndtr(-z_max))
        norm = sf_min - sf_max
        return use_right_tail, sf_min, norm
    cdf_min = float(ndtr(z_min))
    cdf_max = float(ndtr(z_max))
    norm = cdf_max - cdf_min
    return use_right_tail, cdf_min, norm


def _truncnorm_cdf_z(z_vals, use_right_tail, edge_prob, norm):
    """Evaluate truncated standard-normal CDF at standardized values ``z_vals``."""
    if use_right_tail:
        return np.clip((edge_prob - ndtr(-z_vals)) / norm, 0.0, 1.0)
    return np.clip((ndtr(z_vals) - edge_prob) / norm, 0.0, 1.0)


def _truncnorm_ppf_z(q_vals, use_right_tail, edge_prob, norm):
    """Evaluate truncated standard-normal inverse CDF at quantiles ``q_vals``."""
    if use_right_tail:
        surv = np.clip(edge_prob - q_vals * norm, _FLOAT_TINY, 1.0 - _FLOAT_EPS)
        return -ndtri(surv)
    prob = np.clip(edge_prob + q_vals * norm, _FLOAT_TINY, 1.0 - _FLOAT_EPS)
    return ndtri(prob)


def _validate_positive_finite(value, name):
    """Validate that a scalar value is finite and strictly positive."""
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _validate_int_at_least(value, minimum, name):
    """Validate that ``value`` is a non-boolean integer >= ``minimum``."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _validate_quantiles(quantile):
    """Convert and validate cumulative probability quantiles in ``[0, 1]``."""
    q = np.asarray(quantile, dtype=float)
    if np.any((q < 0.0) | (q > 1.0)):
        raise ValueError("quantile values must lie in [0, 1]")
    return q


def _as_float_or_symbolic(redshift):
    """Return a float ndarray unless ``redshift`` is a symbolic expression."""
    if hasattr(redshift, "free_symbols"):
        return redshift
    return np.asarray(redshift, dtype=float)


def _validate_delay_bounds(td_min, td_max):
    """Validate minimum and maximum delay-time bounds in Gyr."""
    td_min = _validate_positive_finite(td_min, "td_min")
    if not np.isfinite(td_max) or td_max <= td_min:
        raise ValueError("td_max must be finite and greater than td_min")
    return td_min, float(td_max)


def _validate_redshift_grid(grid, z_max, name):
    """Validate a user-provided 1D redshift grid starting at 0 and ending at z_max."""
    arr = np.asarray(grid, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"{name} must be a 1D array with at least 2 points")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError(f"{name} must contain finite non-negative values")
    if np.any(np.diff(arr) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    if not np.isclose(arr[0], 0.0):
        raise ValueError(f"{name} should start at z=0")
    if not np.isclose(arr[-1], z_max):
        raise ValueError(f"{name} must end at {z_max}")
    return arr


# =============================================================================
# Base redshift evolution class
# =============================================================================


class BaseRedshiftEvolution(ABC):
    """
    Abstract base class for redshift evolution models with Astropy unit support.

    This class provides a framework for modeling the redshift dependence of
    astrophysical event rates, including differential comoving volume,
    spacetime volume, and redshift probability distributions.

    The core functionality includes:

    - Differential comoving volume (3-volume): ``dVc/dz``
    - Differential spacetime volume factor: ``dVT/dz = psi(z) / (1 + z) * dVc/dz``
    - Redshift probability density functions (PDFs)
    - Normalization and total spacetime volume computation

    Notes
    -----
    - Redshift ``z`` is dimensionless.
    - Comoving volumes and differential spacetime volume factors are returned
      in units of ``Gpc^3``. Multiplying by an ``analysis_time`` quantity via
      :meth:`total_4volume` produces a spacetime 4-volume in ``Gpc^3 * time``.
    - Cosmological interpolation grids are computed lazily on first use to keep
      initialization and module import fast.

    Parameters
    ----------
    z_max : float, optional
        Maximum redshift for integration and normalization. Default is 2.0.
    num_zbins : int, optional
        Number of redshift bins for numerical integration. Must be at least 2.
        Default is 1000.
    cosmology : astropy.cosmology.Cosmology, optional
        Cosmological model to use. If None, defaults to
        ``pycbc.cosmology.get_cosmology()``.
    z_grid : array_like, optional
        Custom redshift grid (must be strictly increasing, start at 0, and end
        at ``z_max``). If None, a uniform grid from 0 to ``z_max`` is used.
    z_max_cosmo : float, optional
        Maximum redshift for the cosmological interpolation grid used by
        :meth:`comoving_spacetime_volume`. Must be >= ``z_max``. Default is
        5000.0.
    z_grid_cosmo : array_like, optional
        Custom redshift grid for cosmological functions (must be strictly
        increasing, start at 0, and end at ``z_max_cosmo``). If None, a
        log-spaced grid from 0 to ``z_max_cosmo`` is constructed lazily.

    Attributes
    ----------
    z_max : float
        Maximum redshift used for integration and normalization.
    z_max_cosmo : float
        Maximum redshift used for cosmological volume interpolation.
    num_zbins : int
        Number of redshift bins on the internal grid.
    cosmology : astropy.cosmology.Cosmology
        Cosmological model used for volume calculations.
    _z_grid : numpy.ndarray
        Internal redshift grid (dimensionless).
    _dvc_dz_grid : astropy.units.Quantity
        Full-sky differential comoving volume on the grid (``Gpc^3``).
    _cached_z : numpy.ndarray or None
        Cached redshift values for interpolation.
    _cached_dvc_dz : astropy.units.Quantity or None
        Cached interpolated ``dVc/dz`` values.

    See Also
    --------
    PowerLawRedshift, GRB2008SFR, MadauDickinson2014SFR, MadauFragos2017SFR,
    SFRTimeDelayRedshift : Concrete implementations of redshift evolution
        models.
    """

    def __init__(
        self,
        z_max: float = 2.0,
        num_zbins: int = 1000,
        cosmology=None,
        z_grid=None,
        z_max_cosmo: float = 5000.0,
        z_grid_cosmo=None,
    ):
        """Initialize the redshift evolution model."""
        self.z_max = _validate_positive_finite(z_max, "z_max")
        self.num_zbins = _validate_int_at_least(num_zbins, 2, "num_zbins")

        self.z_max_cosmo = _validate_positive_finite(z_max_cosmo, "z_max_cosmo")
        if self.z_max_cosmo < self.z_max:
            raise ValueError("z_max_cosmo must be >= z_max")

        self.cosmology = cosmology if cosmology is not None else get_cosmology()

        # Redshift grid (dimensionless) validation
        if z_grid is None:
            self._z_grid = np.linspace(0.0, self.z_max, self.num_zbins)
        else:
            self._z_grid = _validate_redshift_grid(z_grid, self.z_max, "z_grid")
            self.num_zbins = int(self._z_grid.size)

        # Optional custom cosmology grid (validated up front, evaluated lazily)
        if z_grid_cosmo is not None:
            self._z_grid_cosmo_cache = _validate_redshift_grid(
                z_grid_cosmo, self.z_max_cosmo, "z_grid_cosmo"
            )
        else:
            self._z_grid_cosmo_cache = None

        # Lazily computed cosmological arrays and caches
        self._dvc_dz_grid_cache = None
        self._dvc_dz_grid_cosmo_cache = None
        self._cum_spacetime_vol_cosmo_cache = None

        # Cache for repeated array evaluations in dVT_dz
        self._cached_z = None
        self._cached_dvc_dz = None

    def __call__(self, redshift, **parameters):
        """Evaluate the normalized redshift PDF ``p(z)``."""
        return self.prob_redshift(redshift, **parameters)

    # ------------------------------------------------------------------
    # Lazy cosmological grids
    # ------------------------------------------------------------------

    def _compute_fullsky_dvc_dz(self, redshift_array):
        """Evaluate full-sky ``dVc/dz`` in ``Gpc^3`` on ``redshift_array``."""
        return (
            4.0
            * np.pi
            * self.cosmology.differential_comoving_volume(redshift_array).to(
                u.Gpc**3 / u.sr
            )
            * u.sr
        )

    def _interp_dvc_dz(self, redshift_array):
        """Interpolate full-sky ``dVc/dz`` from ``_z_grid`` with zero padding."""
        return (
            np.interp(
                redshift_array,
                self._z_grid,
                self._dvc_dz_grid.value,
                left=0.0,
                right=0.0,
            )
            * u.Gpc**3
        )

    @property
    def _dvc_dz_grid(self):
        """Full-sky differential comoving volume on ``_z_grid`` (in Gpc^3)."""
        if self._dvc_dz_grid_cache is None:
            self._dvc_dz_grid_cache = self._compute_fullsky_dvc_dz(self._z_grid)
        return self._dvc_dz_grid_cache

    @property
    def _z_grid_cosmo(self):
        """Redshift grid used for cosmological volume interpolation."""
        if self._z_grid_cosmo_cache is None:
            n_cosmo = max(2000, int(2 * self.num_zbins))
            z_min_pos = min(1e-5, self.z_max_cosmo * 1e-6)
            self._z_grid_cosmo_cache = np.concatenate(
                ([0.0], np.geomspace(z_min_pos, self.z_max_cosmo, n_cosmo - 1))
            )
        return self._z_grid_cosmo_cache

    @property
    def _dvc_dz_grid_cosmo(self):
        """Full-sky differential comoving volume on ``_z_grid_cosmo`` (Gpc^3)."""
        if self._dvc_dz_grid_cosmo_cache is None:
            self._dvc_dz_grid_cosmo_cache = self._compute_fullsky_dvc_dz(
                self._z_grid_cosmo
            )
        return self._dvc_dz_grid_cosmo_cache

    @property
    def _cum_spacetime_vol_cosmo(self):
        """Cached cumulative integral of ``dVc/dz / (1 + z)`` on ``_z_grid_cosmo``."""
        if self._cum_spacetime_vol_cosmo_cache is None:
            dvc_dz = self._dvc_dz_grid_cosmo.to_value(u.Gpc**3)
            integrand = dvc_dz / (1.0 + self._z_grid_cosmo)
            self._cum_spacetime_vol_cosmo_cache = cumulative_trapezoid(
                integrand, self._z_grid_cosmo, initial=0.0
            )
        return self._cum_spacetime_vol_cosmo_cache

    # ------------------------------------------------------------------
    # Utilities for scalar and arbitrary-shape numpy array compatibility
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_redshift(redshift):
        """Validate redshift input and return ``(z_1d, orig_shape, is_scalar)``."""
        if np.isscalar(redshift):
            z = np.array([redshift], dtype=float)
            orig_shape = ()
            is_scalar = True
        else:
            arr = np.asarray(redshift, dtype=float)
            orig_shape = arr.shape
            z = arr.ravel()
            is_scalar = False

        if not np.all(np.isfinite(z)):
            raise ValueError("Redshift must contain only finite values")

        if np.any(z < 0):
            raise ValueError("Redshift must be non-negative")

        return z, orig_shape, is_scalar

    @classmethod
    def _to_1d_array(cls, redshift):
        """
        Convert scalar or array-like redshift to a validated 1D ndarray.

        Parameters
        ----------
        redshift : scalar or array_like
            Dimensionless redshift(s).

        Returns
        -------
        z_array : numpy.ndarray
            1D array of redshift values (float).
        is_scalar : bool
            True if input was a scalar, False otherwise.

        Raises
        ------
        ValueError
            If any redshift value is negative or non-finite.
        """
        z, _, is_scalar = cls._prepare_redshift(redshift)
        return z, is_scalar

    # ------------------------------------------------------------------
    # Cosmology utilities
    # ------------------------------------------------------------------

    def fullsky_differential_comoving_volume(self, redshift):
        r"""
        Differential comoving 3-volume per unit redshift.

        This is the full-sky differential comoving volume:
            ``dVc/dz = 4 * pi * dVc/(dz dOmega)``.

        Parameters
        ----------
        redshift : float or array_like
            Dimensionless redshift(s).

        Returns
        -------
        astropy.units.Quantity
            Differential comoving volume in ``Gpc^3``.
        """
        zz, orig_shape, is_scalar = self._prepare_redshift(redshift)
        dvc = self._compute_fullsky_dvc_dz(zz)
        return dvc[0] if is_scalar else dvc.reshape(orig_shape)

    def fullsky_differential_spacetime_volume(self, redshift):
        r"""
        Time-dilated differential comoving volume per unit redshift.

        Defined as:
            ``dVT/dz = 4 * pi / (1 + z) * dVc/(dz dOmega)``.

        Parameters
        ----------
        redshift : float or array_like
            Dimensionless redshift(s).

        Returns
        -------
        astropy.units.Quantity
            Time-dilated differential comoving volume in ``Gpc^3``.
        """
        zz, orig_shape, is_scalar = self._prepare_redshift(redshift)
        dvt = self._compute_fullsky_dvc_dz(zz) / (1.0 + zz)
        return dvt[0] if is_scalar else dvt.reshape(orig_shape)

    def comoving_volume(self, redshift):
        """
        Comoving volume enclosed within a given redshift.

        Parameters
        ----------
        redshift : float or array_like
            Dimensionless redshift(s).

        Returns
        -------
        astropy.units.Quantity
            Comoving volume in ``Gpc^3``.
        """
        zz, orig_shape, is_scalar = self._prepare_redshift(redshift)
        vol = self.cosmology.comoving_volume(zz).to(u.Gpc**3)

        return vol[0] if is_scalar else vol.reshape(orig_shape)

    def comoving_spacetime_volume(self, redshift):
        """
        Cumulative full-sky time-dilated comoving volume up to redshift ``z``.

        Computes:
            ``V_ct(z) = integral_0^z [dVc/dz' / (1 + z')] dz'``

        Parameters
        ----------
        redshift : float or array_like
            Redshift value(s). Must be finite and non-negative.

        Returns
        -------
        astropy.units.Quantity
            Cumulative time-dilated comoving volume in ``Gpc^3``.
            Multiply by an observation time to obtain a 4-volume in
            ``Gpc^3 * yr``. Returns a scalar Quantity for scalar input and
            an array Quantity matching the input shape otherwise.

        Notes
        -----
        Values within ``[0, self.z_max_cosmo]`` are evaluated by interpolating
        the cached cumulative integral on ``self._z_grid_cosmo``. Redshifts
        above ``self.z_max_cosmo`` saturate at the value at ``z_max_cosmo``.
        """
        z, orig_shape, is_scalar = self._prepare_redshift(redshift)

        cumulative_volume = self._cum_spacetime_vol_cosmo
        result = (
            np.interp(
                z,
                self._z_grid_cosmo,
                cumulative_volume,
                left=0.0,
                right=cumulative_volume[-1],
            )
            * u.Gpc**3
        )

        return result[0] if is_scalar else result.reshape(orig_shape)

    # ------------------------------------------------------------------
    # Redshift evolution model
    # ------------------------------------------------------------------

    @abstractmethod
    def psi_z(self, redshift, **parameters):
        """
        Redshift evolution function ``psi(z)``.

        This function defines the intrinsic redshift dependence of the event
        rate.

        Parameters
        ----------
        redshift : float or array_like
            Dimensionless redshift(s).
        **parameters
            Model-specific parameters (e.g., ``k``, ``gamma``, ``kappa``,
            ``z_peak``, etc.).

        Returns
        -------
        numpy.ndarray or float
            Redshift evolution factor ``psi(z)`` at each redshift.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_dvc_dz(self, redshift):
        """
        Cache interpolated differential comoving volume for faster repeated
        evaluation.

        Parameters
        ----------
        redshift : numpy.ndarray
            Dimensionless redshift array.

        Returns
        -------
        astropy.units.Quantity
            Cached ``dVc/dz`` values in ``Gpc^3``.
        """
        redshift = np.asarray(redshift, dtype=float)
        self._cached_z = redshift.copy()
        self._cached_dvc_dz = self._interp_dvc_dz(redshift)
        return self._cached_dvc_dz

    # ------------------------------------------------------------------
    # 4-volume and normalization
    # ------------------------------------------------------------------

    def dVT_dz(self, redshift, **parameters):  # noqa: N802
        """
        Differential spacetime volume factor per unit redshift.

        Defined as:
            ``dVT/dz = psi(z) / (1 + z) * dVc/dz``

        Parameters
        ----------
        redshift : float or array_like
            Dimensionless redshift(s).
        **parameters
            Parameters passed to :meth:`psi_z`.

        Returns
        -------
        astropy.units.Quantity
            Differential spacetime volume factor with units of ``Gpc^3``.
        """
        z, orig_shape, is_scalar = self._prepare_redshift(redshift)
        psi = self.psi_z(z, **parameters)

        # Warn if any redshift exceeds z_max
        if np.any(z > self.z_max):
            warnings.warn(
                f"Some redshift values exceed z_max={self.z_max}. "
                "These values will be assigned zero spacetime volume.",
                UserWarning,
                stacklevel=2,
            )

        if not is_scalar:
            if self._cached_z is None or not np.array_equal(z, self._cached_z):
                dvc_dz = self._cache_dvc_dz(z)
            else:
                dvc_dz = self._cached_dvc_dz
        else:
            dvc_dz = self._interp_dvc_dz(z)

        result = psi * dvc_dz / (1.0 + z)
        return result[0] if is_scalar else result.reshape(orig_shape)

    def normalize(self, **parameters):
        """
        Normalization constant for the redshift probability distribution.

        Computes the integral of ``dVT/dz`` over ``[0, z_max]``.

        Parameters
        ----------
        **parameters
            Parameters passed to :meth:`psi_z`.

        Returns
        -------
        astropy.units.Quantity
            Normalization constant with units of ``Gpc^3``.
        """
        psi = self.psi_z(self._z_grid, **parameters)
        integrand = psi * self._dvc_dz_grid.to_value(u.Gpc**3) / (1.0 + self._z_grid)
        return np.trapezoid(integrand, self._z_grid) * u.Gpc**3

    def total_4volume(self, analysis_time: u.Quantity, **parameters):
        """
        Total spacetime volume over an observation time.

        This is the product of the normalization constant and the observation
        time. Note that for SFR subclasses whose :meth:`psi_z` carries an
        overall SFR amplitude (e.g. ``0.015 Msun/yr/Mpc^3``), this quantity is
        scaled by ``psi_z(0)`` relative to a dimensionless ``psi(0) = 1``
        convention.

        Parameters
        ----------
        analysis_time : astropy.units.Quantity
            Observation time (e.g. ``1.0 * u.yr``).
        **parameters
            Parameters passed to :meth:`psi_z`.

        Returns
        -------
        astropy.units.Quantity
            Total spacetime volume (``Gpc^3 * time``).

        Raises
        ------
        astropy.units.UnitTypeError
            If ``analysis_time`` is not a time quantity.
        """
        if not isinstance(
            analysis_time, u.Quantity
        ) or not analysis_time.unit.is_equivalent(u.s):
            raise u.UnitTypeError("analysis_time must be a time quantity")

        return self.normalize(**parameters) * analysis_time

    def prob_redshift(self, redshift, **parameters):
        """
        Normalized redshift probability density function ``p(z)``.

        Parameters
        ----------
        redshift : float or array_like
            Dimensionless redshift(s).
        **parameters
            Parameters passed to :meth:`psi_z`.

        Returns
        -------
        numpy.ndarray or float
            Dimensionless probability density ``p(z)``.
        """
        z, orig_shape, is_scalar = self._prepare_redshift(redshift)
        pdf = np.zeros_like(z, dtype=float)

        if np.any(z > self.z_max):
            warnings.warn(
                f"Some redshift values exceed z_max={self.z_max}. "
                "These values will be assigned zero probability.",
                UserWarning,
                stacklevel=2,
            )

        valid = z <= self.z_max
        if np.any(valid):
            norm_val = self.normalize(**parameters).to_value(u.Gpc**3)
            if norm_val <= 0:
                warnings.warn(
                    "Normalization integral is non-positive; returning zero PDF.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                dvt_val = self.dVT_dz(z[valid], **parameters).to_value(u.Gpc**3)
                pdf[valid] = dvt_val / norm_val

        return float(pdf[0]) if is_scalar else pdf.reshape(orig_shape)


class PowerLawRedshift(BaseRedshiftEvolution):
    """
    Power-law redshift distribution model.

    The redshift evolution function is defined as:
        ``psi(z) = (1 + z)^k``

    Parameters
    ----------
    z_max : float, optional
        Maximum redshift. Default is 2.0.
    num_zbins : int, optional
        Number of redshift bins. Default is 1000.
    cosmology : astropy.cosmology.Cosmology, optional
        Cosmological model. Default is ``get_cosmology()``.
    z_grid : array_like, optional
        Custom redshift grid.

    Attributes
    ----------
    name : str
        Model name: ``"power_law"``.
    param_names : tuple
        Parameter names: ``("k",)``.
    """

    name = "power_law"
    param_names = ("k",)

    def psi_z(self, redshift, *, k: float):
        """
        Redshift evolution function ``psi(z) = (1 + z)^k``.

        Parameters
        ----------
        redshift : float or array_like
            Dimensionless redshift.
        k : float
            Power-law index.

        Returns
        -------
        numpy.ndarray or float
            Dimensionless evolution factor.
        """
        return (1.0 + np.asarray(redshift, dtype=float)) ** k


# Default instance (lightweight due to lazy cosmology grids)
power_law_redshift = PowerLawRedshift()


class GRB2008SFR(BaseRedshiftEvolution):
    """
    Star formation rate (SFR) calibrated by high-redshift gamma-ray bursts
    (GRBs; Yuksel et al. 2008, arXiv:0804.4008).

    The SFR is modeled as:
        ``psi(z) = rho_local * [(1+z)^(3.4*eta) + ((1+z)/5000)^(-0.3*eta)
        + ((1+z)/9)^(-3.5*eta)]^(1/eta)``

    with ``eta = -10`` and ``rho_local = 0.02 Msun/yr/Mpc^3``.
    """

    name = "sfr_grb_2008"
    param_names = ()

    def psi_z(self, redshift, **parameters):
        """Evaluate the GRB 2008 star formation rate in ``Msun/yr/Mpc^3``."""
        del parameters
        z = _as_float_or_symbolic(redshift)
        rho_local = 0.02  # Msolar/yr/Mpc^3
        eta = -10.0
        psiz = (
            (1.0 + z) ** (3.4 * eta)
            + ((1.0 + z) / 5000.0) ** (-0.3 * eta)
            + ((1.0 + z) / 9.0) ** (-3.5 * eta)
        ) ** (1.0 / eta)
        return rho_local * psiz


sfr_grb_2008_redshift = GRB2008SFR()


class MadauDickinson2014SFR(BaseRedshiftEvolution):
    """
    Star formation rate (SFR) from Madau & Dickinson (2014, arXiv:1403.0007).

    The model is:
        ``psi(z) = 0.015 * (1+z)^gamma / [1 + ((1+z)/(1+z_peak))^kappa]``
    """

    name = "sfr_madau_dickinson_2014"
    param_names = ("gamma", "kappa", "z_peak")

    def psi_z(self, redshift, *, gamma=2.7, kappa=5.6, z_peak=1.9):
        """Evaluate the Madau & Dickinson (2014) SFR in ``Msun/yr/Mpc^3``."""
        z = _as_float_or_symbolic(redshift)
        return (
            0.015
            * (1.0 + z) ** gamma
            / (1.0 + ((1.0 + z) / (1.0 + z_peak)) ** kappa)
        )


sfr_madau_dickinson_2014_redshift = MadauDickinson2014SFR()


class MadauFragos2017SFR(BaseRedshiftEvolution):
    """
    Star formation rate (SFR) from Madau & Fragos (2017, arXiv:1606.07887).

    The model is:
        ``psi(z) = k_imf * 0.015 * (1+z)^a / [1 + ((1+z)/b)^c]``

    with parameters depending on ``mode`` (``'high'`` or ``'low'``).
    """

    name = "sfr_madau_fragos_2017"
    param_names = ("k_imf", "mode")

    _MODE_PARAMETERS = {
        "low": (2.6, 3.2, 6.2),
        "high": (2.7, 3.0, 5.35),
    }

    def psi_z(self, redshift, *, k_imf=0.66, mode="high"):
        """Evaluate the Madau & Fragos (2017) SFR in ``Msun/yr/Mpc^3``."""
        z = _as_float_or_symbolic(redshift)

        if mode in self._MODE_PARAMETERS:
            factor_a, factor_b, factor_c = self._MODE_PARAMETERS[mode]
        else:
            raise ValueError("'mode' must choose from 'high' or 'low'.")
        return (
            k_imf
            * 0.015
            * (1.0 + z) ** factor_a
            / (1.0 + ((1.0 + z) / factor_b) ** factor_c)
        )


sfr_madau_fragos_2017_redshift = MadauFragos2017SFR()


# =============================================================================
# Delay-time distribution classes
# =============================================================================


class BaseDelayTimeDistribution(ABC):
    """
    Abstract probability distribution of astrophysical delay times.

    All numerical delay times are expressed in Gyr, and the PDF is expressed
    in ``Gyr^-1``.
    """

    _has_analytic_ppf = False

    @property
    @abstractmethod
    def min_delay(self) -> float:
        """Minimum delay time in Gyr."""

    @property
    @abstractmethod
    def max_delay(self) -> float:
        """Maximum delay time in Gyr."""

    @abstractmethod
    def pdf(self, tau: np.ndarray) -> np.ndarray:
        """
        Return the normalized delay-time PDF in ``Gyr^-1``.

        Parameters
        ----------
        tau : float or array_like
            Delay times in Gyr.

        Returns
        -------
        numpy.ndarray
            Probability density at ``tau`` in ``Gyr^-1``.
        """

    def cdf(self, tau: np.ndarray) -> np.ndarray:
        """
        Compute the cumulative distribution function ``F(tau)``.

        Subclasses with closed-form CDFs override this method; the default
        implementation uses trapezoidal integration on a log-spaced grid.

        Parameters
        ----------
        tau : float or array_like
            Delay times in Gyr.

        Returns
        -------
        numpy.ndarray
            Cumulative probability in ``[0, 1]`` at ``tau``.
        """
        tau_arr = np.asarray(tau, dtype=float)
        tau_vals = np.geomspace(self.min_delay, self.max_delay, 2000)
        pdf_vals = np.asarray(self.pdf(tau_vals), dtype=float)
        cdf_vals = cumulative_trapezoid(pdf_vals, tau_vals, initial=0.0)
        if cdf_vals[-1] <= 0:
            raise ValueError(
                "Delay-time PDF integrates to zero on [min_delay, max_delay]"
            )
        cdf_vals = cdf_vals / cdf_vals[-1]
        return np.interp(tau_arr, tau_vals, cdf_vals, left=0.0, right=1.0)

    def ppf(self, quantile: np.ndarray) -> np.ndarray:
        """
        Percent-point function (inverse CDF) ``F^-1(quantile)``.

        Parameters
        ----------
        quantile : float or array_like
            Cumulative probability values in ``[0, 1]``.

        Returns
        -------
        numpy.ndarray
            Delay times in Gyr corresponding to ``quantile``.
        """
        q = _validate_quantiles(quantile)
        tau_vals = np.geomspace(self.min_delay, self.max_delay, 2000)
        cdf_vals = self.cdf(tau_vals)
        return np.interp(q, cdf_vals, tau_vals)

    def sample(
        self, size: int = 1, random_state: np.random.RandomState | None = None
    ) -> np.ndarray:
        """
        Sample delay times from the distribution via inverse-transform sampling.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Number/shape of samples to generate. Default is 1.
        random_state : numpy.random.RandomState or numpy.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        numpy.ndarray
            Sampled delay times in Gyr.
        """
        if random_state is None:
            random_state = np.random.RandomState()
        u_vals = random_state.uniform(0.0, 1.0, size)
        return self.ppf(u_vals)


class PowerLawDelayDistribution(BaseDelayTimeDistribution):
    """
    Power-law delay-time distribution.

    The normalized PDF on ``[td_min, td_max]`` is:
        ``p(tau) = tau^(-alpha) / Z``

    Parameters
    ----------
    td_min : float, optional
        Minimum delay time in Gyr. Must be positive. Default is 0.02.
    td_max : float, optional
        Maximum delay time in Gyr. Must be greater than ``td_min``. Default is 13.0.
    alpha : float, optional
        Power-law index. Default is 0.81.
    """

    _has_analytic_ppf = True

    def __init__(
        self, td_min: float = 0.02, td_max: float = 13.0, alpha: float = 0.81
    ):
        """Initialize the power-law delay-time distribution."""
        self._td_min, self._td_max = _validate_delay_bounds(td_min, td_max)
        if not np.isfinite(alpha):
            raise ValueError("alpha must be finite")
        self._alpha = float(alpha)

        if np.isclose(self._alpha, 1.0, atol=1e-12, rtol=0.0):
            self._is_inverse = True
            self._norm = np.log(self._td_max / self._td_min)
        else:
            self._is_inverse = False
            one_minus_alpha = 1.0 - self._alpha
            self._norm = (
                self._td_max**one_minus_alpha - self._td_min**one_minus_alpha
            ) / one_minus_alpha

    @property
    def min_delay(self) -> float:
        return self._td_min

    @property
    def max_delay(self) -> float:
        return self._td_max

    @property
    def alpha(self) -> float:
        """Power-law index."""
        return self._alpha

    def pdf(self, tau: np.ndarray) -> np.ndarray:
        """Evaluate the normalized power-law PDF in ``Gyr^-1``."""
        tau_arr = np.asarray(tau, dtype=float)
        result = np.zeros_like(tau_arr, dtype=float)
        valid = (tau_arr >= self._td_min) & (tau_arr <= self._td_max)
        if np.any(valid):
            if self._is_inverse:
                result[valid] = 1.0 / (tau_arr[valid] * self._norm)
            else:
                result[valid] = tau_arr[valid] ** (-self._alpha) / self._norm
        return result

    def cdf(self, tau: np.ndarray) -> np.ndarray:
        """Evaluate the exact analytic CDF."""
        tau_arr = np.asarray(tau, dtype=float)
        result = np.zeros_like(tau_arr, dtype=float)
        result[tau_arr >= self._td_max] = 1.0
        valid = (tau_arr > self._td_min) & (tau_arr < self._td_max)
        if np.any(valid):
            if self._is_inverse:
                result[valid] = np.log(tau_arr[valid] / self._td_min) / self._norm
            else:
                one_minus_alpha = 1.0 - self._alpha
                result[valid] = (
                    tau_arr[valid] ** one_minus_alpha - self._td_min**one_minus_alpha
                ) / (one_minus_alpha * self._norm)
        return result

    def ppf(self, quantile: np.ndarray) -> np.ndarray:
        """Evaluate the exact analytic inverse CDF (percent-point function)."""
        q = _validate_quantiles(quantile)
        if self._is_inverse:
            return self._td_min * (self._td_max / self._td_min) ** q
        one_minus_alpha = 1.0 - self._alpha
        tmin_pow = self._td_min**one_minus_alpha
        tmax_pow = self._td_max**one_minus_alpha
        return (tmin_pow + q * (tmax_pow - tmin_pow)) ** (1.0 / one_minus_alpha)


class InverseDelayDistribution(PowerLawDelayDistribution):
    """
    Inverse (``1 / tau``) delay-time distribution.

    Special case of :class:`PowerLawDelayDistribution` with ``alpha = 1.0``:
        ``p(tau) = 1 / [tau * ln(td_max / td_min)]`` for ``td_min <= tau <= td_max``.

    Parameters
    ----------
    td_min : float, optional
        Minimum delay time in Gyr. Must be positive. Default is 0.02.
    td_max : float, optional
        Maximum delay time in Gyr. Must be greater than ``td_min``. Default is 13.0.
    """

    def __init__(self, td_min: float = 0.02, td_max: float = 13.0):
        """Initialize the inverse delay-time distribution."""
        super().__init__(td_min=td_min, td_max=td_max, alpha=1.0)


class _TruncatedNormalDelayBase(BaseDelayTimeDistribution):
    """Shared base for delay distributions normal in ``tau`` or ``ln(tau)``."""

    _has_analytic_ppf = True

    def __init__(self, td_min: float, td_max: float, sigma: float):
        self._td_min, self._td_max = _validate_delay_bounds(td_min, td_max)
        self._sigma = _validate_positive_finite(sigma, "sigma")

    def _init_truncation(self):
        """Compute tail-safe normalization constants after subclass setup."""
        z_min = float(self._to_z(self._td_min))
        z_max = float(self._to_z(self._td_max))
        self._use_right_tail, self._edge_prob, self._norm = _truncnorm_bounds(
            z_min, z_max
        )
        if self._norm <= 0.0:
            raise ValueError(
                f"{type(self).__name__} has negligible probability mass "
                "within [td_min, td_max]"
            )

    @abstractmethod
    def _to_z(self, tau):
        """Map delay time ``tau`` to standardized normal coordinate ``z``."""

    @abstractmethod
    def _from_z(self, z):
        """Map standardized normal coordinate ``z`` back to delay time ``tau``."""

    @property
    def min_delay(self) -> float:
        return self._td_min

    @property
    def max_delay(self) -> float:
        return self._td_max

    @property
    def sigma(self) -> float:
        """Standard deviation parameter."""
        return self._sigma

    def cdf(self, tau: np.ndarray) -> np.ndarray:
        """Evaluate the exact analytic CDF."""
        tau_arr = np.asarray(tau, dtype=float)
        result = np.zeros_like(tau_arr, dtype=float)
        result[tau_arr >= self._td_max] = 1.0
        valid = (tau_arr > self._td_min) & (tau_arr < self._td_max)
        if np.any(valid):
            z = self._to_z(tau_arr[valid])
            result[valid] = _truncnorm_cdf_z(
                z, self._use_right_tail, self._edge_prob, self._norm
            )
        return result

    def ppf(self, quantile: np.ndarray) -> np.ndarray:
        """Evaluate the exact analytic inverse CDF (percent-point function)."""
        q = _validate_quantiles(quantile)
        z = _truncnorm_ppf_z(q, self._use_right_tail, self._edge_prob, self._norm)
        return np.clip(self._from_z(z), self._td_min, self._td_max)


class LogNormalDelayDistribution(_TruncatedNormalDelayBase):
    """
    Truncated log-normal delay-time distribution.

    On ``[td_min, td_max]``, the PDF is proportional to:
        ``p(tau) ~ exp(-(ln(tau) - ln(t_peak))^2 / (2*sigma^2))``
        ``/ (tau * sigma * sqrt(2*pi))``

    normalized analytically over ``[td_min, td_max]``.

    Notes
    -----
    Here ``t_peak`` is the median scale parameter of the untruncated log-normal
    distribution (the peak in ``ln(tau)`` space); the mode in linear ``tau``
    space is ``t_peak * exp(-sigma^2)``.

    Parameters
    ----------
    td_min : float, optional
        Minimum delay time in Gyr. Must be positive. Default is 0.02.
    td_max : float, optional
        Maximum delay time in Gyr. Must be greater than ``td_min``. Default is 13.0.
    t_peak : float, optional
        Scale/median delay time parameter in Gyr. Default is 2.9.
    sigma : float, optional
        Standard deviation of ``ln(tau)``. Default is 0.2.
    """

    def __init__(
        self,
        td_min: float = 0.02,
        td_max: float = 13.0,
        t_peak: float = 2.9,
        sigma: float = 0.2,
    ):
        """Initialize the truncated log-normal delay-time distribution."""
        super().__init__(td_min=td_min, td_max=td_max, sigma=sigma)
        self._t_peak = _validate_positive_finite(t_peak, "t_peak")
        self._ln_t_peak = np.log(self._t_peak)
        self._init_truncation()

    @property
    def t_peak(self) -> float:
        """Scale/median delay-time parameter in Gyr."""
        return self._t_peak

    def _to_z(self, tau):
        return (np.log(tau) - self._ln_t_peak) / self._sigma

    def _from_z(self, z):
        return np.exp(self._ln_t_peak + self._sigma * z)

    def pdf(self, tau: np.ndarray) -> np.ndarray:
        """Evaluate the normalized truncated log-normal PDF in ``Gyr^-1``."""
        tau_arr = np.asarray(tau, dtype=float)
        result = np.zeros_like(tau_arr, dtype=float)
        valid = (tau_arr >= self._td_min) & (tau_arr <= self._td_max)
        if np.any(valid):
            tv = tau_arr[valid]
            z = self._to_z(tv)
            result[valid] = np.exp(-0.5 * z**2) / (
                tv * self._sigma * _SQRT_2PI * self._norm
            )
        return result


class GaussianDelayDistribution(_TruncatedNormalDelayBase):
    """
    Truncated Gaussian delay-time distribution.

    On ``[td_min, td_max]``, the PDF is proportional to:
        ``p(tau) ~ 1 / (sigma * sqrt(2*pi)) * exp(-(tau - t_mean)^2 / (2*sigma^2))``

    normalized analytically over ``[td_min, td_max]``.

    Parameters
    ----------
    td_min : float, optional
        Minimum delay time in Gyr. Must be positive. Default is 0.02.
    td_max : float, optional
        Maximum delay time in Gyr. Must be greater than ``td_min``. Default is 13.0.
    t_mean : float, optional
        Mean of the untruncated Gaussian distribution in Gyr. Default is 2.0.
    sigma : float, optional
        Standard deviation of the untruncated Gaussian in Gyr. Default is 0.3.
    """

    def __init__(
        self,
        td_min: float = 0.02,
        td_max: float = 13.0,
        t_mean: float = 2.0,
        sigma: float = 0.3,
    ):
        """Initialize the truncated Gaussian delay-time distribution."""
        super().__init__(td_min=td_min, td_max=td_max, sigma=sigma)
        self._t_mean = _validate_positive_finite(t_mean, "t_mean")
        self._init_truncation()

    @property
    def t_mean(self) -> float:
        """Mean of the untruncated Gaussian distribution in Gyr."""
        return self._t_mean

    def _to_z(self, tau):
        return (tau - self._t_mean) / self._sigma

    def _from_z(self, z):
        return self._t_mean + self._sigma * z

    def pdf(self, tau: np.ndarray) -> np.ndarray:
        """Evaluate the normalized truncated Gaussian PDF in ``Gyr^-1``."""
        tau_arr = np.asarray(tau, dtype=float)
        result = np.zeros_like(tau_arr, dtype=float)
        valid = (tau_arr >= self._td_min) & (tau_arr <= self._td_max)
        if np.any(valid):
            z = self._to_z(tau_arr[valid])
            result[valid] = np.exp(-0.5 * z**2) / (
                self._sigma * _SQRT_2PI * self._norm
            )
        return result


# =============================================================================
# Convolved SFR + delay-time redshift evolution model
# =============================================================================


class SFRTimeDelayRedshift(BaseRedshiftEvolution):
    r"""
    Redshift evolution from convolving a star formation rate (SFR) with a
    delay-time distribution.

    The merger rate evolution is computed via:

    .. math::

        \psi(z_m) = \int_{z_m}^{z_{f,\max}} \mathrm{SFR}(z_f;\,\theta_{\mathrm{sfr}})
        \, p(t_d)\, \frac{dt_f}{dz_f}\, dz_f
        = \int_{\tau_{\min}}^{\tau_{\max}(z_m)}
        \mathrm{SFR}\!\left(z_f(t_L(z_m) + \tau);\,\theta_{\mathrm{sfr}}\right)
        p(\tau)\, d\tau

    where :math:`t_d = t_L(z_f) - t_L(z_m)` is the merger delay time and
    :math:`\tau_{\max}(z_m) = \min(\tau_{\max},\, t_L(z_{f,\max}) - t_L(z_m))`.

    For distributions with an analytic inverse CDF (:meth:`ppf`), the integral
    is evaluated via vectorized Gauss-Legendre quadrature in quantile space
    (:math:`du = p(\tau)\,d\tau`), which smoothly resolves both power-law
    singularities and arbitrarily narrow Gaussian/log-normal peaks. For custom
    delay-time distributions without an analytic ``ppf``, vectorized
    Gauss-Legendre quadrature in :math:`\ln\tau` is used.

    Parameters
    ----------
    sfr_model : BaseRedshiftEvolution
        SFR model (e.g., :class:`MadauDickinson2014SFR`).
    time_delay_model : BaseDelayTimeDistribution
        Time-delay distribution (e.g., :class:`PowerLawDelayDistribution`).
    z_max : float, optional
        Maximum merger redshift for integration. Default is 10.0.
    num_zbins : int, optional
        Number of redshift bins. Default is 1000.
    cosmology : astropy.cosmology.Cosmology, optional
        Cosmological model. Default is ``get_cosmology()``.
    z_grid : array_like, optional
        Custom redshift grid.
    z_formation_max : float, optional
        Maximum formation redshift for the lookback-time mapping. Must be
        strictly greater than ``z_max``. Default is 20.0.
    z_max_cosmo : float, optional
        Maximum redshift for the cosmological volume grid. Default is 5000.0.
    z_grid_cosmo : array_like, optional
        Custom cosmological redshift grid.
    n_quad : int, optional
        Number of Gauss-Legendre quadrature nodes per redshift bin. Default
        is 128.
    """

    name = "sfr_time_delay"
    param_names = ()

    def __init__(
        self,
        sfr_model,
        time_delay_model,
        z_max: float = 10.0,
        num_zbins: int = 1000,
        cosmology=None,
        z_grid=None,
        z_formation_max: float = 20.0,
        z_max_cosmo: float = 5000.0,
        z_grid_cosmo=None,
        n_quad: int = 128,
    ):
        """Initialize the SFR-time-delay convolved redshift model."""
        if not isinstance(sfr_model, BaseRedshiftEvolution):
            raise TypeError("sfr_model must be an instance of BaseRedshiftEvolution")
        if not isinstance(time_delay_model, BaseDelayTimeDistribution):
            raise TypeError(
                "time_delay_model must be an instance of BaseDelayTimeDistribution"
            )
        n_quad = _validate_int_at_least(n_quad, 8, "n_quad")

        resolved_cosmology = (
            sfr_model.cosmology if cosmology is None else cosmology
        )
        super().__init__(
            z_max=z_max,
            num_zbins=num_zbins,
            cosmology=resolved_cosmology,
            z_grid=z_grid,
            z_max_cosmo=max(float(z_max_cosmo), float(z_max)),
            z_grid_cosmo=z_grid_cosmo,
        )
        self.sfr_model = sfr_model
        self.time_delay_model = time_delay_model
        self._n_quad = n_quad

        z_formation_max = _validate_positive_finite(
            z_formation_max, "z_formation_max"
        )
        if z_formation_max <= self.z_max:
            raise ValueError("z_formation_max must be greater than z_max")
        self.z_formation_max = z_formation_max

        # Log-spaced formation redshift grid for accurate monotonic z_f(t_L) mapping
        z_min_pos = min(1e-5, self.z_formation_max * 1e-6)
        self._zf_grid = np.concatenate(
            ([0.0], np.geomspace(z_min_pos, self.z_formation_max, 2000))
        )
        self._tf_grid = self.cosmology.lookback_time(self._zf_grid).to_value(u.Gyr)
        self._t_formation_max = float(self._tf_grid[-1])

        # Monotonic interpolation of ln(1 + z_f) vs lookback time t_f (in Gyr)
        self._ln1p_zf_of_t = PchipInterpolator(
            self._tf_grid, np.log1p(self._zf_grid), extrapolate=False
        )

        # Interpolate merger lookback times from _zf_grid (avoids second astropy pass)
        tf_of_z = PchipInterpolator(self._zf_grid, self._tf_grid, extrapolate=False)
        self._tm_grid = tf_of_z(self._z_grid)
        self._gl_nodes, self._gl_weights = np.polynomial.legendre.leggauss(
            self._n_quad
        )

        # Lazily precomputed SFR-independent quadrature nodes and weights
        self._quad_plan_ready = False
        self._quad_active_indices = None
        self._quad_zf_grid = None
        self._quad_half_scale = None
        self._quad_weights_2d = None

        # Extract parameter names from sub-models
        self._sfr_param_names = tuple(getattr(sfr_model, "param_names", ()))
        self.param_names = self._sfr_param_names

        # Bounded LRU cache for precomputed psi(z) grids
        self._psi_z_cache: OrderedDict = OrderedDict()
        self._max_cache_size = _DEFAULT_MAX_CACHE_SIZE
        self._last_sfr_params = None
        self._psi_z_grid = None

    def _prepare_quadrature_plan(self):
        """Precompute SFR-independent formation-redshift nodes and weights once."""
        td_model = self.time_delay_model
        tau_min = td_model.min_delay
        tau_max = td_model.max_delay

        tau_hi = np.minimum(tau_max, self._t_formation_max - self._tm_grid)
        active = tau_hi > tau_min
        self._quad_plan_ready = True

        if not np.any(active):
            self._quad_active_indices = np.empty(0, dtype=int)
            return

        tm_active = self._tm_grid[active]
        tau_hi_active = tau_hi[active]

        if getattr(td_model, "_has_analytic_ppf", False):
            # Quantile-space substitution: u = F(tau), du = p(tau) dtau
            u_hi = td_model.cdf(tau_hi_active)
            has_mass = u_hi > 0.0
            if not np.any(has_mass):
                self._quad_active_indices = np.empty(0, dtype=int)
                return

            self._quad_active_indices = np.flatnonzero(active)[has_mass]
            tm_eff = tm_active[has_mass]
            u_eff = u_hi[has_mass]

            half_u = 0.5 * u_eff
            u_grid = half_u[:, None] * (self._gl_nodes[None, :] + 1.0)
            tau_grid = td_model.ppf(u_grid)
            tf_grid = np.clip(
                tm_eff[:, None] + tau_grid, 0.0, self._t_formation_max
            )
            self._quad_zf_grid = np.expm1(self._ln1p_zf_of_t(tf_grid))
            self._quad_half_scale = half_u
            self._quad_weights_2d = None
        else:
            # General fallback for custom distributions: quadrature in ln(tau)
            self._quad_active_indices = np.flatnonzero(active)
            ln_lo = np.log(tau_min)
            ln_hi = np.log(tau_hi_active)
            half_span = 0.5 * (ln_hi - ln_lo)[:, None]
            mid = 0.5 * (ln_hi + ln_lo)[:, None]
            tau_grid = np.exp(mid + half_span * self._gl_nodes[None, :])
            tf_grid = np.clip(
                tm_active[:, None] + tau_grid, 0.0, self._t_formation_max
            )
            self._quad_zf_grid = np.expm1(self._ln1p_zf_of_t(tf_grid))
            self._quad_half_scale = None
            self._quad_weights_2d = (
                half_span
                * td_model.pdf(tau_grid)
                * tau_grid
                * self._gl_weights[None, :]
            )

    def _compute_convolution_grid(self, sfr_params):
        """Evaluate the delay-time convolution across ``self._z_grid``."""
        if not self._quad_plan_ready:
            self._prepare_quadrature_plan()

        psi_grid = np.zeros(self._z_grid.size, dtype=float)
        if self._quad_active_indices.size == 0:
            return psi_grid

        sfr_vals = self.sfr_model.psi_z(self._quad_zf_grid, **sfr_params)
        if self._quad_half_scale is not None:
            psi_grid[self._quad_active_indices] = self._quad_half_scale * (
                sfr_vals @ self._gl_weights
            )
        else:
            psi_grid[self._quad_active_indices] = np.einsum(
                "ij,ij->i", sfr_vals, self._quad_weights_2d
            )
        return psi_grid

    def _update_psi_z_grid(self, sfr_params=None):
        """Evaluate and cache the convolution over ``self._z_grid``."""
        if sfr_params is None:
            sfr_params = {}

        cache_key = tuple(sorted(sfr_params.items())) if sfr_params else ()
        if cache_key in self._psi_z_cache:
            self._psi_z_cache.move_to_end(cache_key)
            self._psi_z_grid = self._psi_z_cache[cache_key]
            return

        psi_grid = self._compute_convolution_grid(sfr_params)
        self._psi_z_grid = psi_grid
        self._psi_z_cache[cache_key] = psi_grid
        if len(self._psi_z_cache) > self._max_cache_size:
            self._psi_z_cache.popitem(last=False)

    def psi_z(self, redshift, **parameters):
        """
        Evaluate the convolved redshift evolution factor ``psi(z)``.

        Parameters
        ----------
        redshift : float or array_like
            Dimensionless redshift(s). Must be finite and non-negative.
        **parameters
            Parameters forwarded to the SFR model based on its ``param_names``.

        Returns
        -------
        numpy.ndarray or float
            Convolved evolution factor at ``redshift``.
        """
        z, orig_shape, is_scalar = self._prepare_redshift(redshift)

        sfr_params = {}
        for key, value in parameters.items():
            if key in self._sfr_param_names:
                sfr_params[key] = value
            else:
                warnings.warn(
                    f"Unknown parameter '{key}' ignored in "
                    "SFRTimeDelayRedshift.psi_z",
                    UserWarning,
                    stacklevel=2,
                )

        current_sfr_params = tuple(sorted(sfr_params.items())) if sfr_params else ()
        if (
            self._psi_z_grid is None
            or current_sfr_params != self._last_sfr_params
        ):
            self._update_psi_z_grid(sfr_params)
            self._last_sfr_params = current_sfr_params

        result = np.interp(z, self._z_grid, self._psi_z_grid, left=0.0, right=0.0)
        return float(result[0]) if is_scalar else result.reshape(orig_shape)


__all__ = [
    "BaseDelayTimeDistribution",
    "BaseRedshiftEvolution",
    "GaussianDelayDistribution",
    "GRB2008SFR",
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
