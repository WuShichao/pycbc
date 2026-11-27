# Copyright (C) 2021  Shichao Wu, Alex Nitz, Collin Capano
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
This module provides functions for star formation rate models, time delay
models, merger rate density, and population models of BBH/BNS/NSBH.
"""

import warnings
from functools import partial

import numpy as np
import scipy.integrate as scipy_integrate
import scipy.interpolate as scipy_interpolate
from astropy import units

from pycbc.cosmology import cosmological_quantity_from_redshift, get_cosmology
from pycbc.population.redshift_models import (
    BaseDelayTimeDistribution,
    BaseRedshiftEvolution,
    GaussianDelayDistribution,
    GRB2008SFR,
    InverseDelayDistribution,
    LogNormalDelayDistribution,
    MadauDickinson2014SFR,
    MadauFragos2017SFR,
    PowerLawDelayDistribution,
    SFRTimeDelayRedshift,
    sfr_grb_2008_redshift,
    sfr_madau_dickinson_2014_redshift,
    sfr_madau_fragos_2017_redshift,
)


class _CallableSFR(BaseRedshiftEvolution):
    """Adapter wrapping a plain callable ``sfr_func(z)`` as a redshift model."""

    name = "callable_sfr"
    param_names = ()

    def __init__(self, sfr_func, z_max=10.0, num_zbins=1000, cosmology=None):
        super().__init__(z_max=z_max, num_zbins=num_zbins, cosmology=cosmology)
        self._sfr_func = sfr_func

    def psi_z(self, redshift, **parameters):
        del parameters
        return np.asarray(self._sfr_func(redshift), dtype=float)


def sfr_grb_2008(z):
    r"""
    The star formation rate (SFR) calibrated by high-z GRBs data.

    Parameters
    ----------
    z : float or numpy.ndarray
        The redshift.

    Returns
    -------
    rho_z : float or numpy.ndarray
            The SFR at redshift z, in unit of "Msolar/Mpc^3/yr".

    Note
    ----
        Please see Eq.(5) in <arXiv:0804.4008> for more details.

    """
    return sfr_grb_2008_redshift.psi_z(z)


def sfr_madau_dickinson_2014(z, gamma=2.7, kappa=5.6, z_peak=1.9):
    r"""
    The madau-dickinson 2014 star formation rate (SFR).

    Parameters
    ----------
    z : float or numpy.ndarray
        The redshift.
    gamma : float, optional
        Low-redshift power-law index. Default is 2.7.
    kappa : float, optional
        High-redshift decline index. Default is 5.6.
    z_peak : float, optional
        Turnover redshift. Default is 1.9.

    Returns
    -------
    rho_z : float or numpy.ndarray
            The SFR at redshift z, in unit of "Msolar/Mpc^3/yr".

    Notes
    -----
         Please see Eq.(15) in <arXiv:1403.0007> for more details.

    """
    return sfr_madau_dickinson_2014_redshift.psi_z(
        z, gamma=gamma, kappa=kappa, z_peak=z_peak
    )


def sfr_madau_fragos_2017(z, k_imf=0.66, mode="high"):
    r"""
    The madau-fragos 2017 star formation rate (SFR),
         which updates madau-dickinson 2014 SFR by better reproducing
         a number of recent 4 < z < 10 results.

    Parameters
    ----------
    z : float or numpy.ndarray
        The redshift.
    k_imf : float
        The correction factor KIMF adjusts the SFR for the assumed IMF,
        for the Salpeter IMF, k_imf=1.0, for the three component broken
        power-law Kroupa IMF, k_imf=0.66, here we choose Kroupa IMF as default.
    mode : str
        The model of SFR, choose from 'high' and 'low'. Default to 'high'.

    Returns
    -------
    rho_z : float or numpy.ndarray
            The SFR at redshift z, in unit of "Msolar/Mpc^3/yr".

    Notes
    -----
         Please see <arXiv:1606.07887> and <arXiv:1706.07053> for more details.

    """
    return sfr_madau_fragos_2017_redshift.psi_z(z, k_imf=k_imf, mode=mode)


def diff_lookback_time(z, **kwargs):
    r"""
    The derivative of lookback time ``t(z)`` with respect to redshift ``z``.

    .. deprecated::
        Use ``cosmology.H(z)`` from :py:func:`get_cosmology` or the classes
        in :py:mod:`pycbc.population.redshift_models` instead.

    Parameters
    ----------
    z : float, numpy.ndarray, or sympy.Basic
        The redshift.
    \**kwargs :
        All other keyword args are passed to :py:func:`get_cosmology` to
        select a cosmology. If none provided, will use
        :py:attr:`DEFAULT_COSMOLOGY`.

    Returns
    -------
    dt_dz : float, numpy.ndarray, or sympy.Expr
        The value of ``dt/dz`` at redshift ``z`` in ``Gyr``.

    Notes
    -----
        Please see Eq.(A3) in <arXiv:2011.02717v3> for more details.

    """
    warnings.warn(
        "diff_lookback_time is deprecated; use astropy cosmology.H(z) or "
        "pycbc.population.redshift_models instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    cosmology = get_cosmology(**kwargs)

    if isinstance(z, (int, float, np.number, np.ndarray)):
        h_z = cosmology.H(z).to_value(1.0 / units.Gyr)
        return 1.0 / (h_z * (1.0 + z))

    # Symbolic fallback for legacy callers (e.g. convolution_trans)
    from sympy import sqrt

    h0_gyr = cosmology.H0.to_value(1.0 / units.Gyr)
    return 1.0 / h0_gyr / (1.0 + z) / sqrt(
        cosmology.Ode0 + cosmology.Om0 * (1.0 + z) ** 3
    )


_DELAY_MODEL_CLASSES = {
    "inverse": InverseDelayDistribution,
    "power_law": PowerLawDelayDistribution,
    "log_normal": LogNormalDelayDistribution,
    "gaussian": GaussianDelayDistribution,
}

_SFR_MODEL_CLASSES = {
    sfr_grb_2008: GRB2008SFR,
    sfr_madau_dickinson_2014: MadauDickinson2014SFR,
    sfr_madau_fragos_2017: MadauFragos2017SFR,
}


def _eval_on_grid(func, z_array):
    """Evaluate ``func`` on ``z_array``, falling back to scalar loop if needed."""
    try:
        return np.asarray(func(z_array), dtype=float)
    except (TypeError, ValueError):
        return np.array([func(z) for z in z_array], dtype=float)


def _build_delay_distribution(td_model, cosmology=None, **td_kwargs):
    """Map a legacy string name or distribution instance to a delay model."""
    if isinstance(td_model, BaseDelayTimeDistribution):
        return td_model

    cls = _DELAY_MODEL_CLASSES.get(td_model)
    if cls is None:
        raise ValueError(
            "'td_model' must be a BaseDelayTimeDistribution or choose from "
            "['log_normal', 'gaussian', 'power_law', 'inverse']."
        )

    if cosmology is None:
        cosmology = get_cosmology()

    default_td_max = float(cosmology.age(0).to_value(units.Gyr))
    td_min = td_kwargs.pop("td_min", 0.02)
    td_max = td_kwargs.pop("td_max", default_td_max)
    return cls(td_min=td_min, td_max=td_max, **td_kwargs)


def _build_sfr_model(sfr_func, maxz, cosmology):
    """Map a legacy SFR callable or redshift model to a BaseRedshiftEvolution."""
    if isinstance(sfr_func, BaseRedshiftEvolution):
        return sfr_func

    cls = _SFR_MODEL_CLASSES.get(sfr_func)
    if cls is not None:
        return cls(z_max=maxz, cosmology=cosmology)

    if callable(sfr_func):
        return _CallableSFR(sfr_func, z_max=maxz, cosmology=cosmology)

    raise TypeError(
        "sfr_func must be a callable or an instance of BaseRedshiftEvolution."
    )


def p_tau(tau, td_model="inverse"):
    r"""
    The probability distribution of the time delay.

    .. deprecated::
        Use the :py:class:`BaseDelayTimeDistribution` subclasses in
        :py:mod:`pycbc.population.redshift_models` instead.

    Parameters
    ----------
    tau : float, numpy.ndarray, or sympy.Basic
        The merger delay time in Gyr.
    td_model : str
        The time delay model: ``'log_normal'``, ``'gaussian'``,
        ``'power_law'``, or ``'inverse'``.

    Returns
    -------
    p_t : float, numpy.ndarray, or sympy.Expr
        The probability density at time delay ``tau``.

    Notes
    -----
        Please see the Appendix in <arXiv:2011.02717v3> for more details.

    """
    warnings.warn(
        "p_tau is deprecated; use BaseDelayTimeDistribution subclasses in "
        "pycbc.population.redshift_models instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if td_model not in _DELAY_MODEL_CLASSES:
        raise ValueError(
            "'td_model' must choose from "
            "['log_normal', 'gaussian', 'power_law', 'inverse']."
        )

    if isinstance(tau, (int, float, np.number, np.ndarray)):
        dist = _build_delay_distribution(td_model)
        val = dist.pdf(tau)
        return float(val) if np.isscalar(tau) else val

    # Symbolic fallback for legacy callers (e.g. convolution_trans)
    from sympy import Piecewise, exp, log, sqrt

    if td_model == "log_normal":
        t_ln, sigma_ln = 2.9, 0.2
        p_t = exp(-((log(tau) - log(t_ln)) ** 2) / (2 * sigma_ln**2)) / (
            sqrt(2 * np.pi) * sigma_ln * tau
        )
    elif td_model == "gaussian":
        t_g, sigma_g = 2.0, 0.3
        p_t = exp(-((tau - t_g) ** 2) / (2 * sigma_g**2)) / (
            sqrt(2 * np.pi) * sigma_g
        )
    elif td_model == "power_law":
        p_t = tau ** (-0.81)
    else:
        td_min = 0.02
        td_max = cosmological_quantity_from_redshift(0, "age")
        norm_const = 1.0 / np.log(td_max / td_min)
        p_t = Piecewise(
            (0, tau < td_min),
            (0, tau > td_max),
            (norm_const / tau, True),
        )
    return p_t


def convolution_trans(sfr, diff_lookback_t, model_td, **kwargs):
    r"""
    Legacy symbolic integrand helper for CBC merger rate convolution.

    .. deprecated::
        Use :py:class:`pycbc.population.redshift_models.SFRTimeDelayRedshift`
        or :py:func:`merger_rate_density` instead.

    Parameters
    ----------
    sfr : callable
        The star formation rate function used in the convolution.
    diff_lookback_t : callable
        The derivative of lookback time ``t(z)`` with respect to redshift ``z``.
    model_td : str
        The name of the time-delay model.
    \**kwargs :
        Keyword arguments forwarded to ``diff_lookback_t``.

    Returns
    -------
    func : sympy.Expr
        The product of ``sfr(z)``, ``p_tau(tau(z))``, and ``dt(z)/dz``.
    """
    warnings.warn(
        "convolution_trans is deprecated; use SFRTimeDelayRedshift in "
        "pycbc.population.redshift_models instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from sympy import integrate, symbols

    if model_td not in ("log_normal", "gaussian", "power_law", "inverse"):
        raise ValueError(
            "'model_td' must choose from "
            "['log_normal', 'gaussian', 'power_law', 'inverse']."
        )

    diff_lookback_time_z = partial(diff_lookback_t, **kwargs)
    z = symbols("z")
    z_0 = symbols("z_0")
    tau = integrate(diff_lookback_time_z(z), (z, z_0, z))
    return sfr(z) * p_tau(tau, model_td) * diff_lookback_time_z(z)


def merger_rate_density(
    sfr_func, td_model, rho_local, maxz=10.0, npoints=10000, z_array=None, **kwargs
):
    r"""
    Calculate the merger rate density of CBC sources via numerical convolution.

    Computes the convolution of the star formation rate ``SFR(z_f)`` and the
    time-delay probability density ``P(tau)``, normalized to ``rho_local`` at
    ``z = 0``.

    Parameters
    ----------
    sfr_func : callable or BaseRedshiftEvolution
        The star formation rate function (e.g. ``sfr_madau_dickinson_2014``)
        or a :py:class:`BaseRedshiftEvolution` instance.
    td_model : str or BaseDelayTimeDistribution
        Name of the time-delay model (``'log_normal'``, ``'gaussian'``,
        ``'power_law'``, or ``'inverse'``) or a
        :py:class:`BaseDelayTimeDistribution` instance.
    rho_local : float
        The local merger rate density at ``z = 0``, in ``Mpc^-3 yr^-1``.
    maxz : float, optional
        The maximum redshift. Default is 10.0.
    npoints : int, optional
        The number of points used in the output interpolation when ``z_array``
        is None. Default is 10000.
    z_array : numpy.ndarray, optional
        Custom redshift grid for the output interpolator. Default is None.
    \**kwargs :
        Optional delay-time parameters (``td_min``, ``td_max``, ``alpha``,
        ``t_peak``, ``t_mean``, ``sigma``, ``z_formation_max``) and cosmology
        keyword arguments forwarded to :py:func:`get_cosmology`.

    Returns
    -------
    rho_z : scipy.interpolate.interp1d
        Interpolator returning the merger rate density as a function of
        redshift.

    Notes
    -----
        Please see Eq.(A1), Eq.(A2) in <arXiv:2011.02717v3> for more details.

    """
    if z_array is None:
        z_array = np.linspace(0.0, maxz, npoints)
    else:
        z_array = np.asarray(z_array, dtype=float)
        maxz = float(max(maxz, np.max(z_array)))

    td_param_keys = ("td_min", "td_max", "alpha", "t_peak", "t_mean", "sigma")
    td_kwargs = {}
    for key in td_param_keys:
        if key in kwargs:
            td_kwargs[key] = kwargs.pop(key)

    z_formation_max = kwargs.pop("z_formation_max", max(20.0, maxz * 2.0))
    if z_formation_max <= maxz:
        raise ValueError("z_formation_max must be greater than maxz")

    cosmology = get_cosmology(**kwargs)
    sfr_model = _build_sfr_model(sfr_func, maxz=maxz, cosmology=cosmology)
    delay_model = _build_delay_distribution(
        td_model, cosmology=cosmology, **td_kwargs
    )

    num_zbins = min(max(int(npoints), 200), 1000)
    model = SFRTimeDelayRedshift(
        sfr_model=sfr_model,
        time_delay_model=delay_model,
        z_max=maxz,
        num_zbins=num_zbins,
        cosmology=cosmology,
        z_formation_max=z_formation_max,
    )

    f_z = model.psi_z(z_array)
    f_0 = float(model.psi_z(0.0))
    if f_0 != 0.0:
        f_z = f_z / f_0 * rho_local

    return scipy_interpolate.interp1d(z_array, f_z)


def coalescence_rate(rate_den, maxz=10.0, npoints=10000, z_array=None, **kwargs):
    r"""
    This function calculates the coalescence(merger) rate at the redshift z.

    Parameters
    ----------
    rate_den : function or scipy.interpolate.interp1d
        The merger rate density as a function of redshift. In the unit of
        "Mpc^-3yr^-1". Use `merger_rate_density` function to calculate.
    maxz : float
            The max redshift. The default value is 10.
    npoints : int
            The number of points used in the interpolation. The default
            value is 10000.
    z_array : numpy.array or list
            The redshift range. The default value is None.
    \**kwargs :
        All other keyword args are passed to :py:func:`get_cosmology` to
        select a cosmology. If none provided, will use
        :py:attr:`DEFAULT_COSMOLOGY`.

    Returns
    -------
    coalescence_rate_interp : scipy.interpolate.interp1d
          The coalescence rate.

    Notes
    -----
         Please see Eq.(1) in <arXiv:2011.02717v3> for more details.

    """
    if z_array is None:
        z_array = np.linspace(0.0, maxz, npoints)
    else:
        z_array = np.asarray(z_array, dtype=float)

    cosmology = get_cosmology(**kwargs)
    dvc_dz = cosmology.differential_comoving_volume(z_array).to_value(
        units.Mpc**3 / units.sr
    )
    rate_vals = _eval_on_grid(rate_den, z_array)
    dr_dz = 4.0 * np.pi * dvc_dz / (1.0 + z_array) * rate_vals
    return scipy_interpolate.interp1d(z_array, dr_dz, fill_value="extrapolate")


def total_rate_upto_redshift(z, merger_rate):
    r"""
    Total rate of occurrences out to some redshift.

    Parameters
    ----------
    z : int, float, tuple, numpy.ndarray or list
            The redshift.
    merger_rate : scipy.interpolate.interp1d or function
        The merger or coalescence rate. Function should take the
        redshift as a single argument. Provided by users or
        calculated by the `coalescence_rate` function.

    Returns
    -------
    rate: float or list
        The total rate of occurrences out to some redshift. In the
        unit of "yr^-1".

    """
    if isinstance(z, (float, int)):
        total_rate = scipy_integrate.quad(
            merger_rate, 0, z, epsabs=2.00e-4, epsrel=2.00e-4, limit=1000
        )[0]
    elif isinstance(z, (tuple, np.ndarray, list)):
        total_rate = [
            scipy_integrate.quad(
                merger_rate, 0, redshift, epsabs=2.00e-4, epsrel=2.00e-4, limit=1000
            )[0]
            for redshift in z
        ]
    else:
        raise ValueError(
            "'z' must be 'int', 'float', 'tuple', 'numpy.ndarray' or 'list'."
        )

    return total_rate


def average_time_between_signals(z_array, merger_rate):
    r"""
    This function calculates the average time interval
        of a certain type of CBC source.

    Parameters
    ----------
    z_array : numpy.array
            The array of redshift.
    merger_rate : scipy.interpolate.interp1d or function
            The coalescence rate. Provided by users or calculated by
            the `coalescence_rate` function.

    Returns
    -------
    average_time : float
            The average time interval (s).

    """
    total_rate = total_rate_upto_redshift(z_array[-1], merger_rate)  # yr^-1
    return 1.0 / total_rate * 365 * 24 * 3600  # s


def norm_redshift_distribution(z_array, merger_rate):
    r"""
    This function calculates the normalized redshift distribution
        of a certain type of CBC source.

    Parameters
    ----------
    z_array : numpy.array
            The array of redshift.
    merger_rate : scipy.interpolate.interp1d or function
            The coalescence rate. Provided by users or calculated by
            the `coalescence_rate` function.

    Returns
    -------
    norm_coalescence_rate : numpy.array
            The normalized redshift distribution.

    Notes
    -----
         This can be used as a population-informed prior for redshift
         and luminosity distance of CBC sources.

    """
    lambd = average_time_between_signals(z_array, merger_rate)
    return lambd / (365 * 24 * 3600) * merger_rate(z_array)


def distance_from_rate(total_rate, merger_rate, maxz=10, npoints=10000, **kwargs):
    r"""
    Returns the luminosity distance from the given total rate value.

    Parameters
    ----------
    total_rate : float
        The total rate.
    merger_rate : scipy.interpolate.interp1d or function
            The coalescence rate. Provided by users or calculated by
            the `coalescence_rate` function.
    maxz : float
        The max redshift in the interpolation, the default value is 10.
        Please use the same `maxz` as in `merger_rate`.
    npoints : int
        The number of points used in the interpolation, the default value
        is 10000.
    \**kwargs :
        All other keyword args are passed to :py:func:`get_cosmology` to
        select a cosmology. If none provided, will use
        :py:attr:`DEFAULT_COSMOLOGY`.

    Returns
    -------
    dl : float
        The luminosity distance at the given total rate value.
        In the unit of "Mpc".

    Notes
    -----
         This can be used in a population-informed prior for redshift
         and luminosity distance of CBC sources. When this used in
         high redshift range, please first use the `total_rate_upto_redshift`
         function to plot the curve and find the point where the curve
         starts to stay almost horizontal, then set `maxz` to the
         corresponding value and change `npoints` to a reasonable value.

    """
    cosmology = get_cosmology(**kwargs)

    if not hasattr(merger_rate, "dist_interp"):
        merger_rate.dist_interp = {}

    if (cosmology.name not in merger_rate.dist_interp) or (
        len(merger_rate.dist_interp[cosmology.name].x) != npoints
    ):
        z_array = np.linspace(0.0, maxz, npoints)
        dists = cosmological_quantity_from_redshift(
            z_array, "luminosity_distance", **kwargs
        )
        rate_vals = _eval_on_grid(merger_rate, z_array)
        total_rates = scipy_integrate.cumulative_trapezoid(
            rate_vals, z_array, initial=0.0
        )
        interp = scipy_interpolate.interp1d(total_rates, dists)
        merger_rate.dist_interp[cosmology.name] = interp

    dl = merger_rate.dist_interp[cosmology.name](total_rate)
    if np.isscalar(dl):
        dl = float(dl)
    return dl


__all__ = [
    "average_time_between_signals",
    "coalescence_rate",
    "convolution_trans",
    "diff_lookback_time",
    "distance_from_rate",
    "merger_rate_density",
    "norm_redshift_distribution",
    "p_tau",
    "sfr_grb_2008",
    "sfr_madau_dickinson_2014",
    "sfr_madau_fragos_2017",
    "total_rate_upto_redshift",
]
