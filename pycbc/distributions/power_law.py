# Copyright (C) 2016 Christopher M. Biwer
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
"""
This modules provides classes for evaluating distributions where the
probability density function is a power law.
"""
import logging
import numpy

from pycbc.distributions import bounded

logger = logging.getLogger('pycbc.distributions.power_law')

class UniformPowerLaw(bounded.BoundedDist):
    r"""
    For a uniform distribution in power law. The parameters are
    independent of each other. Instances of this class can be called like
    a function. By default, logpdf will be called, but this can be changed
    by setting the class's __call__ method to its pdf method.

    The cumulative distribution function (CDF) will be the ratio of volumes:

    .. math::

        F(r) = \frac{V(r)}{V(R)}

    Where :math:`R` is the radius of the sphere. So we can write our
    probability density function (PDF) as:

    .. math::

        f(r) = c r^n

    For generality we use :math:`n` for the dimension of the volume element,
    eg. :math:`n=2` for a 3-dimensional sphere. And use
    :math:`c` as a general constant.

    So now we calculate the CDF in general for this type of PDF:

    .. math::

        F(r) = \int f(r) dr = \int c r^n dr = \frac{1}{n + 1} c r^{n + 1} + k

    Now with the definition of the CDF at radius :math:`r_{l}` is equal to 0
    and at radius :math:`r_{h}` is equal to 1 we find that the constant from
    integration from this system of equations:

    .. math::

        1 = \frac{1}{n + 1} c ((r_{h})^{n + 1} - (r_{l})^{n + 1}) + k

    Can see that :math:`c = (n + 1) / ((r_{h})^{n + 1} - (r_{l})^{n + 1}))`.
    And :math:`k` is:

    .. math::

        k = - \frac{r_{l}^{n + 1}}{(r_{h})^{n + 1} - (r_{l})^{n + 1}}

    Can see that :math:`c= \frac{n + 1}{R^{n + 1}}`. So can see that the CDF is:

    .. math::

        F(r) = \frac{1}{(r_{h})^{n + 1} - (r_{l})^{n + 1}} r^{n + 1} - \frac{r_{l}^{n + 1}}{(r_{h})^{n + 1} - (r_{l})^{n + 1}}

    And the PDF is the derivative of the CDF:

    .. math::

        f(r) = \frac{(n + 1)}{(r_{h})^{n + 1} - (r_{l})^{n + 1}} (r)^n

    Now we use the probabilty integral transform method to get sampling on
    uniform numbers from a continuous random variable. To do this we find
    the inverse of the CDF evaluated for uniform numbers:

    .. math::

        F(r) = u = \frac{1}{(r_{h})^{n + 1} - (r_{l})^{n + 1}} r^{n + 1} - \frac{r_{l}^{n + 1}}{(r_{h})^{n + 1} - (r_{l})^{n + 1}}

    And find :math:`F^{-1}(u)` gives:

    .. math::

        u = \frac{1}{n + 1} \frac{(r_{h})^{n + 1} - (r_{l})^{n + 1}} - \frac{r_{l}^{n + 1}}{(r_{h})^{n + 1} - (r_{l})^{n + 1}}

    And solving for :math:`r` gives:

    .. math::

        r = ( ((r_{h})^{n + 1} - (r_{l})^{n + 1}) u + (r_{l})^{n + 1})^{\frac{1}{n + 1}}

    Therefore the radius can be sampled by taking the n-th root of uniform
    numbers and multiplying by the radius offset by the lower bound radius.

    \**params :
        The keyword arguments should provide the names of parameters and their
        corresponding bounds, as either tuples or a `boundaries.Bounds`
        instance.

    Attributes
    ----------
    dim : int
        The dimension of volume space. In the notation above `dim`
        is :math:`n+1`. For a 3-dimensional sphere this is 3.
    """
    name = "uniform_power_law"
    def __init__(self, dim=None, **params):
        super(UniformPowerLaw, self).__init__(**params)
        self.dim = dim
        self._norm = 1.0
        self._lognorm = 0.0
        for p in self._params:
            self._norm *= self.dim  / \
                                   (self._bounds[p][1]**(self.dim) -
                                    self._bounds[p][0]**(self.dim))
        self._lognorm = numpy.log(self._norm)

    @property
    def norm(self):
        """float: The normalization of the multi-dimensional pdf."""
        return self._norm

    @property
    def lognorm(self):
        """float: The log of the normalization."""
        return self._lognorm

    def _cdfinv_param(self, param, value):
        """Return inverse of cdf to map unit interval to parameter bounds.
        """
        n = self.dim - 1
        r_l = self._bounds[param][0]
        r_h = self._bounds[param][1]
        new_value = ((r_h**(n+1) - r_l**(n+1))*value + r_l**(n+1))**(1./(n+1))
        return new_value

    def _pdf(self, **kwargs):
        """Returns the pdf at the given values. The keyword arguments must
        contain all of parameters in self's params. Unrecognized arguments are
        ignored.
        """
        for p in self._params:
            if p not in kwargs.keys():
                raise ValueError(
                            'Missing parameter {} to construct pdf.'.format(p))
        if kwargs in self:
            pdf = self._norm * \
                  numpy.prod([(kwargs[p])**(self.dim - 1)
                              for p in self._params])
            return float(pdf)
        else:
            return 0.0

    def _logpdf(self, **kwargs):
        """Returns the log of the pdf at the given values. The keyword
        arguments must contain all of parameters in self's params. Unrecognized
        arguments are ignored.
        """
        for p in self._params:
            if p not in kwargs.keys():
                raise ValueError(
                            'Missing parameter {} to construct pdf.'.format(p))
        if kwargs in self:
            log_pdf = self._lognorm + \
                      (self.dim - 1) * \
                      numpy.log([kwargs[p] for p in self._params]).sum()
            return log_pdf
        else:
            return -numpy.inf

    @classmethod
    def from_config(cls, cp, section, variable_args):
        """Returns a distribution based on a configuration file. The parameters
        for the distribution are retrieved from the section titled
        "[`section`-`variable_args`]" in the config file.

        Parameters
        ----------
        cp : pycbc.workflow.WorkflowConfigParser
            A parsed configuration file that contains the distribution
            options.
        section : str
            Name of the section in the configuration file.
        variable_args : str
            The names of the parameters for this distribution, separated by
            `prior.VARARGS_DELIM`. These must appear in the "tag" part
            of the section header.

        Returns
        -------
        Uniform
            A distribution instance from the pycbc.inference.prior module.
        """
        return super(UniformPowerLaw, cls).from_config(cp, section,
                                                       variable_args,
                                                       bounds_required=True)


class UniformRadius(UniformPowerLaw):
    """ For a uniform distribution in volume using spherical coordinates, this
    is the distriubtion to use for the radius.

    For more details see UniformPowerLaw.
    """
    name = "uniform_radius"
    def __init__(self, dim=3, **params):
        super(UniformRadius, self).__init__(dim=3, **params)



class UniformAmplitudePowerLaw(bounded.BoundedDist):
    r"""A power-law prior on an amplitude-like coordinate.

    The pdf of each parameter :math:`x` is

    .. math::

        p(x) = \frac{1-\mathrm{power}}
                    {x_\mathrm{hi}^{1-\mathrm{power}}
                     - x_\mathrm{lo}^{1-\mathrm{power}}}\,
               x^{-\mathrm{power}},

    on :math:`x_\mathrm{lo} \le x \le x_\mathrm{hi}`, and zero outside.
    The logarithmic normalisation is used when ``power == 1``.  ``power = 0``
    is uniform; ``power = 4`` is the volumetric prior

    .. math:: p(u) \propto u^{-4}

    that a uniform-in-distance prior induces on the amplitude
    :math:`u = K / d`.  Unlike :class:`UniformPowerLaw`, the exponent is a
    scalar option rather than a number of dimensions, and it is read from the
    configuration file.

    The optional ``distance_constant`` is stored for provenance only; it is
    the constant :math:`K` in :math:`d = K / u`, so a consumer can recover a
    distance from a drawn amplitude without repeating the convention.

    Parameters
    ----------
    power : float, optional
        The exponent of the power law. Default is 0 (uniform).
    distance_constant : float, optional
        The constant that converts the coordinate to a distance. Default is
        None (not recorded).

    The keyword arguments also provide the parameter names and their bounds,
    as either tuples or a :class:`pycbc.boundaries.Bounds` instance.
    """
    name = "uniform_amplitude_power_law"

    def __init__(self, power=0.0, distance_constant=None, **params):
        self.power = float(power)
        self.distance_constant = (None if distance_constant is None
                                  else float(distance_constant))
        super(UniformAmplitudePowerLaw, self).__init__(**params)
        lognorm = 0.0
        for p in self._params:
            lo, hi = self._bounds[p]
            if lo <= 0:
                raise ValueError("uniform_amplitude_power_law needs a "
                                 "positive lower bound, got %r for %s"
                                 % (lo, p))
            if abs(self.power - 1.0) < 1e-12:
                norm = 1.0 / numpy.log(hi / lo)
            else:
                norm = ((1.0 - self.power)
                        / (hi ** (1.0 - self.power)
                           - lo ** (1.0 - self.power)))
            lognorm += numpy.log(norm)
        self._lognorm = float(lognorm)
        self._norm = float(numpy.exp(lognorm))

    @property
    def norm(self):
        """float: The normalisation of the pdf."""
        return self._norm

    @property
    def lognorm(self):
        """float: The log of the normalisation."""
        return self._lognorm

    def _pdf(self, **kwargs):
        """Return the pdf at the given values."""
        for p in self._params:
            if p not in kwargs:
                raise ValueError(
                    'Missing parameter {} to construct pdf.'.format(p))
        if kwargs in self:
            values = numpy.array([kwargs[p] for p in self._params], dtype=float)
            return float(self._norm
                         * numpy.prod(values ** (-self.power)))
        return 0.0

    def _logpdf(self, **kwargs):
        """Return the log of the pdf at the given values."""
        for p in self._params:
            if p not in kwargs:
                raise ValueError(
                    'Missing parameter {} to construct pdf.'.format(p))
        if kwargs in self:
            values = numpy.array([kwargs[p] for p in self._params], dtype=float)
            return float(self._lognorm
                         - self.power * numpy.log(values).sum())
        return -numpy.inf

    def _cdfinv_param(self, param, value):
        """Map a unit-interval value to the parameter's support."""
        lo, hi = self._bounds[param]
        if abs(self.power - 1.0) < 1e-12:
            return lo * (hi / lo) ** value
        a = lo ** (1.0 - self.power)
        b = hi ** (1.0 - self.power)
        return (a + (b - a) * value) ** (1.0 / (1.0 - self.power))

    @classmethod
    def from_config(cls, cp, section, variable_args):
        """Return a distribution from a section of a configuration file.

        The section is titled ``[section-variable_args]`` and holds
        ``min-{param}``, ``max-{param}`` and ``power``.
        """
        try:
            power = float(cp.get_opt_tag(section, "power", variable_args))
        except Exception:
            power = 0.0
        return bounded.bounded_from_config(
            cls, cp, section, variable_args, bounds_required=True,
            additional_opts={"power": power})


__all__ = ["UniformPowerLaw", "UniformRadius", "UniformAmplitudePowerLaw"]

