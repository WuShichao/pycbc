# Copyright (C) 2026 Shichao Wu
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
"""A population prior as a density over a search's parameters.

Any prior that ``pycbc_inference`` reads (``[variable_params]``,
``[prior-*]``, optionally ``[waveform_transforms]``) is drawn from, mapped to
the search's parameters and estimated there by a Gaussian kernel density, so
a search or a PE run can swap one population for another (a GWTC
posterior-predictive, a flat prior) by its configuration file alone.
"""

import numpy

from pycbc import distributions, transforms


class TemplatePriorWeights(object):
    """The density of a population over the parameters ``params``.

    Parameters
    ----------
    samples : dict
        Draws of the population: an array for each of ``params``.
    params : list of str
        The search's parameters, e.g. ``['mchirp', 'q', 'spin1z',
        'spin2z']``.
    log_params : tuple of str
        Parameters estimated in their logarithm (positive, spanning
        decades); the density is returned in the parameter itself.
    bandwidth : float, optional
        The kernel's bandwidth factor, as ``scipy.stats.gaussian_kde``
        takes it; Scott's rule by default.
    """
    def __init__(self, samples, params, log_params=('mchirp',),
                 bandwidth=None):
        from scipy.stats import gaussian_kde
        self.params = list(params)
        self.log_params = [p for p in self.params if p in log_params]
        y = self._coordinates(numpy.column_stack(
            [numpy.asarray(samples[p], dtype=float) for p in self.params]))
        self.kde = gaussian_kde(y.T, bw_method=bandwidth)
        self.size = len(y)

    def _coordinates(self, x):
        y = numpy.array(x, dtype=float)
        for i, p in enumerate(self.params):
            if p in self.log_params:
                y[:, i] = numpy.log(y[:, i])
        return y

    @classmethod
    def from_config(cls, cp, params, size=100000, seed=0,
                    prior_section='prior',
                    transforms_section='waveform_transforms', **kwargs):
        """Draw ``size`` points from a configuration's prior, apply its
        waveform transforms, and estimate the density over ``params``."""
        variable = cp.options('variable_params')
        dists = distributions.read_distributions_from_config(cp,
                                                             prior_section)
        joint = distributions.JointDistribution(variable, *dists)
        numpy.random.seed(seed)
        draws = joint.rvs(size=size)
        samples = {p: draws[p] for p in variable}
        if cp.has_section(transforms_section):
            samples = transforms.apply_transforms(
                samples, transforms.read_transforms_from_config(
                    cp, transforms_section))
        return cls(samples, params, **kwargs)

    def logpdf(self, points):
        """The natural log of the density at ``points`` ``(M, len(params))``
        (or a dict of arrays), in the parameters themselves."""
        if isinstance(points, dict):
            points = numpy.column_stack([numpy.asarray(points[p], float)
                                         for p in self.params])
        x = numpy.atleast_2d(numpy.asarray(points, dtype=float))
        out = self.kde.logpdf(self._coordinates(x).T)
        # the Jacobian of the logarithms
        for i, p in enumerate(self.params):
            if p in self.log_params:
                out = out - numpy.log(x[:, i])
        return out
