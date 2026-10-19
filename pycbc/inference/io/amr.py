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

When the sampler is given a ``physical-map``, the file also holds the physical
parameters (``mchirp``, ``q``, ``spin1z``, ``spin2z`` and, when the map carries
them, ``eclipticlongitude`` and ``eclipticlatitude``) computed from the ``eta``
columns through the exact inverse of the shape chart.  They are a linear
recombination of the ``eta`` columns, not independent entries, and the
attributes ``amr_shape_matrix``, ``amr_shape_lam``, ``amr_shape_reference``,
``amr_shape_scale`` and ``amr_injection_eta`` let a reader reproduce them from
the ``eta`` columns alone.

When the sampler is given a ``sufficient-file``, the file also holds the
restored columns ``inclination``, ``polarization``, ``delta_tc``,
``amplitude_u``, ``distance`` and ``distance_ratio``.  They are one joint draw
per cell from ``p(lag, orientation, u | cell)``, not extra dimensions:
``inclination`` and ``polarization`` are the drawn fine-grid orientation
point (a sample from the cubic interpolant of the per-node log-likelihood
onto a 256 x 64 uniform grid, dithered inside the node cell so the column is
continuous, not one of the enumerated 16 x 8 nodes),
``delta_tc`` is the drawn lag on the enumerated arrival-time grid, and
``amplitude_u`` is the drawn ``u = A/A_min``.  The amplitude is drawn from its
exact integrand at the selected fine node, so the inclination and the
amplitude stay correlated.  The arrival time is deliberately NOT interpolated:
a local quadratic misses its held-out lag by 7.87 nats median, so the fix is
oversampling the transform, not smoothing the lag grid; ``amr_arrival_time_note``
says so in the file.  ``distance`` is the same draw in megaparsecs,
``d = K / u``, where ``K`` is the measured distance at which a face-on signal
has optimal SNR 4; ``K`` is recorded as ``amr_distance_constant_mpc`` and it is
an option (``distance-constant``), defaulting to the measured 542.1006 Mpc.
The distance is exactly reciprocal to ``amplitude_u``
(``distance * amplitude_u = K``) and is therefore correlated with the
inclination and the amplitude by construction.  Its marginal is controlled by
the amplitude prior, so the attributes ``amr_distance_note`` and
``amr_amplitude_prior`` state that a distance is quotable only under a prior
whose support exceeds ``u`` of about 7 and which is flat over five to nine.
The attributes ``amr_recovered_kind``, ``amr_amplitude_ceiling``,
``amr_amplitude_power``, ``amr_amplitude_units``, ``amr_distance_units`` and
``amr_amplitude_distance_relation`` say what the columns are and what they
leave out.  Every recovered quantity also carries its own resolution
beside it: ``amr_inclination_resolution`` and
``amr_polarization_resolution`` state the fine grid and its node spacing,
``amr_amplitude_resolution`` and ``amr_distance_resolution`` the amplitude
grid, ``amr_delta_tc_resolution`` and ``amr_arrival_time_grid_s`` the lag
grid and that it is enumerated, and
``amr_orientation_interpolation_error_*`` the measured interpolation error
in nats.

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

When the physical map records an injection, the truth is written both as the
``amr_injection_physical`` attribute and as a top-level ``injections`` group
with one dataset per physical parameter.  When the restored columns are
written, their injected values (inclination, polarization, delta_tc) are added
to the same group.  The group is what
``pycbc_inference_plot_posterior --plot-injection-parameters`` and
``pycbc_inference_plot_pp`` read; it holds only the parameters the file
actually carries, and :meth:`AmrFile.read_injections` reads it directly rather
than through the CBC injection reader, which requires ``tc`` because it is
built for waveform generation.
"""

import numpy

from pycbc.io import FieldArray

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
        # the restored conditional columns: the amplitude ceiling, the prior,
        # the seed and the fact that they are conditionals, not dimensions
        recovered = getattr(sampler, "recovered_metadata", None)
        if recovered is not None:
            for key, value in recovered().items():
                self.attrs[key] = _attr_value(value)
        # the recorded injection's orientation, used by the injections group
        for option, key in (("injection_inclination", "amr_injection_inclination"),
                            ("injection_polarization",
                             "amr_injection_polarization"),
                            ("injection_coa_phase", "amr_injection_coa_phase"),
                            ("injection_delta", "amr_injection_delta_tc")):
            if hasattr(sampler, option):
                self.attrs[key] = float(getattr(sampler, option))
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
        # the truth the map recorded is also written as the group the inference
        # plot tools read.  Skipped when the file has no recorded injection or
        # already carries the group; no existing dataset is touched.
        self.write_injections_group()

    def write_injections_group(self):
        """Writes the recorded truth as an ``injections`` group.

        PyCBC's ``pycbc_inference_plot_posterior --plot-injection-parameters``
        and ``pycbc_inference_plot_pp`` obtain the truth through
        :py:meth:`read_injections`, whose convention is a top-level
        ``injections`` group with one dataset per parameter.  The physical map
        records the physical truth in ``attrs['amr_injection_physical']`` and
        the restored columns record theirs in ``amr_injection_inclination``,
        ``amr_injection_polarization`` and ``amr_injection_delta_tc``; this
        copies both sets into that group so the plot tools read the file
        unchanged.

        Only the parameters the file actually carries are written.  A file with
        no recorded injection, or one that already has the group, is left
        untouched.

        Returns
        -------
        bool
            True if the group was created, False if it was skipped.
        """
        if self.injections_group in self:
            return False
        values = {}
        try:
            physical = numpy.asarray(
                self.attrs["amr_injection_physical"], dtype=float)
            names = self.attrs["amr_physical_params"]
        except KeyError:
            physical = None
            names = None
        if names is not None:
            if isinstance(names, bytes):
                names = names.decode()
            names = str(names).split()
            if names and len(names) == physical.size:
                values.update(zip(names, physical.tolist()))
        for name, key in (("inclination", "amr_injection_inclination"),
                          ("polarization", "amr_injection_polarization"),
                          ("delta_tc", "amr_injection_delta_tc")):
            if key in self.attrs:
                values[name] = float(self.attrs[key])
        if not values:
            return False
        group = self.create_group(self.injections_group)
        for name, value in values.items():
            group[name] = numpy.atleast_1d(value)
        # The generic reader takes the injection type from the *file's* root
        # attributes, where its absence already means 'cbc'; the group
        # attribute is written so the table describes itself.
        group.attrs["injtype"] = "cbc"
        return True

    def read_injections(self, group=None):
        """Reads the injection truth recorded in the ``injections`` group.

        The base implementation builds the table with
        :class:`pycbc.inject.InjectionSet`, which defaults to the CBC reader
        and therefore requires a ``tc`` column, because it is meant for
        generating waveforms.  An AMR file's truth is the physical point the
        partition was built around: the parameters of the map and, when the
        restored columns are present, their injected values
        (``inclination``, ``polarization`` and ``delta_tc``).  Reading the
        datasets directly keeps an absolute ``tc`` out of the file instead of
        inventing one, and returns the same :class:`pycbc.io.FieldArray` the
        plot tools expect.

        Parameters
        ----------
        group : str, optional
            Group that the ``injections`` group is in.  Default (None) is the
            top level, matching the base reader.

        Returns
        -------
        FieldArray
            One row per injection, with one field per recorded parameter.
        """
        if group is None or group == "/":
            path = self.injections_group
        else:
            path = group.rstrip("/") + "/" + self.injections_group
        if path not in self:
            raise KeyError(
                "no {} group in {}; the AMR writer records the truth only "
                "when a physical map with an injection is configured".format(
                    path, self.filename))
        table = self[path]
        return FieldArray.from_kwargs(
            **{name: table[name][()] for name in table})

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
