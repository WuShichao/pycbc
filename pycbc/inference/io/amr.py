# Copyright (C) 2026  The PyCBC contributors
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
I/O for the AMR cell-partition adapter.

WHAT THIS FILE CONTAINS
----------------------
This is not a sample chain.  It is a **deterministic grid quadrature on a fixed
partition**: a set of AMR cells, each contributing a centre, a volume and an
amplitude-marginalised Bayes factor.  The posterior is the weighted histogram of
the cell centres with weight proportional to the Bayes factor times the cell
volume.  There is no Markov chain, no stochastic move, no burn-in and **no
evidence error estimate**.

The file therefore stores the partition as a *weighted* sample set in the
format the nested samplers already use: one entry per cell, with ``logwt``,
``loglikelihood`` and ``logprior``.  ``loglikelihood`` is the cell's
amplitude-marginalised Bayes factor (a likelihood ratio, not a normalised
likelihood), ``logprior`` is the log of the cell's prior *mass*
``V_cell / sum(V)`` (not a pointwise density), and ``logwt`` is their sum, so
``logwt`` is the log posterior mass of the cell and ``logsumexp(logwt)`` is the
Riemann-sum evidence of the partition.

When the sampler is given a ``physical-map``, the file also holds the four
physical parameters (``mchirp``, ``q``, ``spin1z``, ``spin2z``) computed from
the ``eta`` columns through the exact inverse of the shape chart.  They are a
linear recombination of the ``eta`` columns, not independent entries, and the
attributes ``amr_shape_matrix``, ``amr_shape_lam``, ``amr_shape_reference``,
``amr_shape_scale`` and ``amr_injection_eta`` let a reader reproduce them from
the ``eta`` columns alone.

Reading the file
----------------
By default ``read_raw_samples`` returns an equal-weight posterior obtained by
systematic resampling of the ``logwt`` weights, exactly as
:py:class:`pycbc.inference.io.dynesty.DynestyFile` does for nested samples.
``--raw-samples`` returns the weighted partition itself (one row per cell).  A
``--nsamples`` option limits the size of the resampled posterior; the default is
one draw per cell.

Because the resampling is a deterministic function of the weights and the seed,
the posterior is reproducible.  It is still not a chain: do not run MCMC
diagnostics (autocorrelation, burn-in, Gelman-Rubin, acceptance) on it.  The
modules and the file attributes say so, and
:py:meth:`pycbc.inference.io.amr.AmrFile.write_sampler_metadata` records the
nature of the object in the file itself.

A column the partition never refined is stored but flat: every cell reports the
same value and a single width, so the posterior has no extent along it.  The
file records which columns those are (``amr_unrefined_columns``,
``amr_degenerate_columns``) and the number of sampled columns that actually
carry refinement (``amr_posterior_rank``), so a six-column file that is rank
four cannot be read as a six-dimensional posterior.
"""

import numpy

from .base_nested_sampler import BaseNestedSamplerFile
from .dynesty import CommonNestedMetadataIO
from .posterior import read_raw_samples_from_file


#: The one-line statement of what an AMR file contains.  Written into every
#: file's attributes so that a reader cannot mistake it for a chain.
AMR_NATURE = (
    "deterministic grid quadrature on a fixed AMR partition: weighted cell "
    "centres, not a stochastic sampler; no evidence error estimate"
)

#: Prefix used for the adapter's metadata attributes.
AMR_ATTR_PREFIX = "amr_"


def _attr_value(value):
    """Coerce a value to something h5py can store in ``attrs``.

    h5py refuses ``None``; store the string instead so the attribute always
    exists.  Numbers, strings, bools and lists pass through unchanged.
    """
    if value is None:
        return "none"
    return value


class AmrFile(CommonNestedMetadataIO, BaseNestedSamplerFile):
    """File handler for the AMR cell-partition adapter.

    The stored object is a weighted partition (one row per cell), not a
    chain.  See the module docstring for the full statement of what it is
    and is not.
    """

    name = "amr_file"

    def read_raw_samples(self, fields, raw_samples=False, seed=0,
                         nsamples=None):
        """Reads samples from an AMR file and constructs a posterior.

        Parameters
        ----------
        fields : list of str
            The names of the parameters to load.  Names must correspond to
            dataset names in the file's ``samples`` group.
        raw_samples : bool, optional
            Return the raw weighted partition (one row per cell) instead of
            the equal-weight posterior.  Default is False.
        seed : int, optional
            Seed for the systematic resampling used to construct the
            posterior.  Default is 0.  Ignored when ``raw_samples`` is True.
        nsamples : int, optional
            Number of equal-weight posterior samples to draw.  Default (None)
            is one draw per cell.

        Returns
        -------
        dict :
            Dictionary of parameter fields -> samples.  ``loglikelihood`` is
            always included, matching the nested-sampler readers.
        """
        samples = read_raw_samples_from_file(self, fields)
        logwt = read_raw_samples_from_file(self, ["logwt"])["logwt"]
        loglikelihood = read_raw_samples_from_file(
            self, ["loglikelihood"]
        )["loglikelihood"]
        if raw_samples:
            return samples
        # normalise the partition's posterior mass; do not use the stored
        # log_evidence, which may be a Riemann sum over a different
        # normalisation (see the module docstring)
        logz = _logsumexp(logwt)
        weights = numpy.exp(logwt - logz)
        N = len(weights) if nsamples is None else int(nsamples)
        rng = numpy.random.default_rng(seed)
        positions = (rng.random() + numpy.arange(N)) / N
        idx = numpy.searchsorted(
            numpy.cumsum(weights) / numpy.sum(weights), positions,
            side="right")
        # the last position can land exactly on 1.0
        idx = numpy.clip(idx, 0, len(weights) - 1)
        post = {"loglikelihood": loglikelihood[idx]}
        for param in fields:
            post[param] = samples[param][idx]
        return post

    def write_sampler_metadata(self, sampler):
        """Writes the sampler's metadata and the nature of the object.

        The ``sampler`` attribute must exist, because
        :py:class:`pycbc.inference.io.base_nested_sampler.BaseNestedSamplerFile`
        uses it for the sampler name; the AMR adapter additionally records
        that what follows is a grid quadrature.
        """
        self.attrs["sampler"] = sampler.name
        if self.sampler_group not in self.keys():
            self.create_group(self.sampler_group)
        self.write_nature(sampler)
        sampler.model.write_metadata(self)

    def write_nature(self, sampler):
        """Writes the nature-of-the-object attributes.

        Called both when the file is created and when results are written, so
        that a resumed or externally created file carries them too.
        """
        self.attrs["posterior_kind"] = AMR_NATURE
        self.attrs["stochastic_sampler"] = False
        self.attrs["evidence_error"] = "none"
        self.attrs["amr_refinement"] = _attr_value(
            getattr(sampler, "refinement", None))
        self.attrs["amr_max_cells"] = _attr_value(
            getattr(sampler, "max_cells", None))
        self.attrs["amr_max_level"] = _attr_value(
            getattr(sampler, "max_level", None))
        self.attrs["amr_eps"] = _attr_value(getattr(sampler, "eps", None))
        self.attrs["amr_evidence"] = _attr_value(
            getattr(sampler, "evidence", None))
        self.attrs["amr_source"] = _attr_value(
            getattr(sampler, "source", None))
        self.attrs["amr_partition_file"] = _attr_value(
            getattr(sampler, "partition_file", None))
        self.attrs["amr_eta_columns"] = numpy.asarray(
            getattr(sampler, "eta_columns", []), dtype=int)
        # the projection from the shape coordinates to the physical parameters,
        # when the sampler has one.  The matrix, the eigenvalues, the reference
        # values, the scales and the injection are recorded so that a reader
        # can reproduce every physical column from the eta columns alone.
        physical = getattr(sampler, "physical_metadata", None)
        if physical is not None:
            for key, value in physical().items():
                self.attrs[key] = value
        level_info = getattr(sampler, "level_info", None) or {}
        for key in ("level", "n_cells", "n_eval", "finest_width",
                    "log_total_mass"):
            if key in level_info:
                self.attrs["amr_" + key] = _attr_value(level_info[key])
        # Which chart columns the partition actually refines.  A sampled column
        # with one centre and one width is not a posterior dimension: every
        # cell reports the same value, the posterior is flat across the cell,
        # and the file's rank is lower than its column count.  Recorded so a
        # reader does not have to infer it from the samples.
        column_info = getattr(sampler, "column_info", None) or {}
        for key in ("refined", "unrefined", "degenerate"):
            if key in column_info:
                self.attrs["amr_%s_columns" % key] = numpy.asarray(
                    column_info[key], dtype=int)
        for key in ("n_centres", "n_widths"):
            if key in column_info:
                self.attrs["amr_column_" + key] = numpy.asarray(
                    column_info[key], dtype=int)
        for key in ("centre_min", "centre_max", "width_min", "width_max"):
            if key in column_info:
                self.attrs["amr_column_" + key] = numpy.asarray(
                    column_info[key], dtype=float)
        if "rank" in column_info:
            self.attrs["amr_posterior_rank"] = int(column_info["rank"])

    @staticmethod
    def extra_args_parser(parser=None, skip_args=None, **kwargs):
        r"""Adds the options for reading an AMR partition.

        The nested-sampler base parser is not reused directly because its
        help text calls the raw samples "unweighted"; an AMR raw read is the
        *weighted* partition, one row per cell.  The three options are
        ``--raw-samples``, ``--seed`` and ``--nsamples``.

        Parameters
        ----------
        parser : argparse.ArgumentParser, optional
            Add arguments to this parser; if None, create one.
        skip_args : list, optional
            Option names (without the leading ``--``) not to add.
        \**kwargs :
            Passed to ``argparse.ArgumentParser`` when a parser is created.

        Returns
        -------
        parser : argparse.ArgumentParser
        actions : list of argparse.Action
        """
        import argparse
        if parser is None:
            parser = argparse.ArgumentParser(**kwargs)
        elif kwargs:
            raise ValueError("No other keyword arguments should be provided "
                             "if a parser is provided.")
        if skip_args is None:
            skip_args = []
        actions = []
        if "raw_samples" not in skip_args:
            actions.append(parser.add_argument(
                "--raw-samples", action="store_true", default=False,
                help="Return the weighted partition itself: one row per AMR "
                     "cell, with logwt, logprior and loglikelihood. This is "
                     "not an equal-weight posterior and it is not a chain."))
        if "seed" not in skip_args:
            actions.append(parser.add_argument(
                "--seed", type=int, default=0,
                help="Seed for the systematic resampling that turns the "
                     "weighted partition into an equal-weight posterior. "
                     "Default is 0. Ignored with --raw-samples."))
        if "nsamples" not in skip_args:
            actions.append(parser.add_argument(
                "--nsamples", type=int, default=None,
                help="Number of equal-weight posterior samples to draw from "
                     "the AMR partition. Default is one draw per cell."))
        return parser, actions



def _logsumexp(values):
    """Log of the sum of exponentials, stable for the log-weight arrays."""
    values = numpy.asarray(values, dtype=float)
    vmax = values.max()
    if not numpy.isfinite(vmax):
        return -numpy.inf
    return float(vmax + numpy.log(numpy.exp(values - vmax).sum()))
