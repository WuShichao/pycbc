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

r"""
The AMR cell partition as a PyCBC inference sampler.

READ THIS FIRST: WHAT THIS OBJECT IS
------------------------------------
This is a **deterministic grid quadrature on a fixed partition**, not a
stochastic sampler.

The research AMR behind ``SSB_AMR_SAMPLER=1`` (the prototype in the companion
repository, see below) produces a set of cells.  Each cell has a centre, a
width and an amplitude-marginalised Bayes factor ``B_c``.  The posterior
estimate is the histogram over those cells with weight

    w_c = B_c V_c / sum_c B_c V_c ,

where ``V_c`` is the cell volume in the four stiff mismatch coordinates.  That
is a Riemann sum over a partition: there is no chain, no proposal, no
acceptance, no burn-in and no random number anywhere in the posterior.

Consequences that a downstream diagnostic must not forget:

* **The posterior is conditional on the partition.**  Refining the grid changes
  the quadrature, not the target; the target is the continuous posterior of the
  amplitude-marginalised Bayes factor over the covered region.
* **There is no evidence error estimate.**  ``log_evidence`` is the Riemann sum
  ``logsumexp(log(B_c V_c)) - log(sum_c V_c)`` over the cells actually in the
  partition.  ``dlog_evidence`` is written as ``nan``.  Do not present it as a
  nested-sampling error bar.
* **The entries are not independent draws.**  They are cells.  Two neighbouring
  cells are adjacent regions of parameter space, and one large cell can carry
  more posterior mass than many small ones.  MCMC diagnostics (autocorrelation,
  burn-in, Gelman-Rubin, acceptance fraction, effective sample size computed
  from a chain) are undefined here.
* **The output is weighted.**  It is written in the weighted format the nested
  samplers already use: a ``logwt`` field per cell, with
  ``logwt = loglikelihood + logprior``.  The readers resample ``logwt`` into an
  equal-weight posterior, exactly as
  :py:class:`pycbc.inference.io.dynesty.DynestyFile` does for nested samples.

The physics is not reimplemented here
-------------------------------------
Every Bayes factor, every cell width and every containment decision comes from
the research implementation.  This module imports it; it does not copy it.  The
companion repository's ``amr_sampler.py`` is loaded by file path (it is a
script, not an installed package) and supplies ``PosteriorAMR``,
``fixed_signal``, the base covering grid and the production ``Run`` object from
``ssb_amr``.  Only the control flow of the refinement loop is written here, and
that loop is needed because the refinement criterion is a configuration option;
the default ``refinement = both`` with ``eps = 3e-4``, ``n-fmax = 100`` and
``max-level = 6`` reproduces the research ``PosteriorAMR.run`` step for step.

The two data sources
--------------------
``enable-research = false`` (the default)
    Read a weighted partition that the research prototype already wrote, i.e.
    ``eta_L<level>``, ``w_L<level>`` and ``lnB_L<level>`` from
    ``posteriors.npz``.  This is the adapter path: the physics already ran, and
    this module maps its output into PyCBC's samples format.  It needs no GPU
    and no `cupy`.

``enable-research = true``
    Run the research AMR live: import the prototype, build its production
    ``Run``, inject the configured signal, evaluate the Bayes factor cell by
    cell and refine.  This needs the research tree, ``cupy``, a CUDA GPU and
    (for this prototype) about five minutes.  The process changes directory to
    the research root for the duration, because the prototype's data paths are
    relative to it; the directory is restored afterwards.

The SNR colour variable
-----------------------
The file stores an explicit ``snr`` field for every cell, because the cell's
amplitude-marginalised Bayes factor is already an SNR-squared statistic and a
reader should not have to reconstruct it from ``logwt`` and ``logprior``.

The adapter writes ``loglikelihood = ln B_c`` for the cell Bayes factor
``B_c``, and ``logwt = ln B_c + logprior`` with
``logprior = ln(V_c / sum_c V_c)``.  Hence

    ln B_c = logwt - logprior = loglikelihood .

For a fixed template in Gaussian noise, the likelihood ratio for a signal of
amplitude ``A`` is a parabola in ``A``,

    -2 ln L(A) = (rho - A)^2 + const ,

where ``rho`` is the optimal SNR of the template.  Maximising over ``A`` gives
``2 ln B_c = rho_c^2``.  The production statistic marginalises over the
amplitude and orientation instead of maximising, which multiplies ``B_c`` by
the amplitude-prior normalisation but leaves the scaling with ``rho^2`` at the
peak.  The adapter therefore writes

    snr_c = sqrt(2 ln B_c) = sqrt(2 (logwt_c - logprior_c)) ,

clipping ``ln B_c`` at zero so that a cell with ``B_c < 1`` reports
``snr = 0`` rather than a complex number.  This is the detected, effective SNR
of the cell, not the injected optimal SNR of the noiseless signal: the
production injection has optimal network SNR 16 and the level-6 partition's
peak cell comes out at 15.39, 3.8 per cent below, because the amplitude prior,
the orientation marginalisation, the noise realisation and the finite cell
width all act on the recovered value.  That is the sanity check, not an
equality.

``snr`` is written into the samples group so that
``pycbc_inference_plot_posterior --z-arg snr`` colours the cells by the stored
number.  The file does not store a noise-only ``lognl``; the stored ``snr`` is
the authoritative colour variable.

The physical parameters
-----------------------
The chart ``eta`` is a rotation and a rescaling of the four physical offsets,
so its axes are not physical parameters and renaming them would be wrong: only
a projection back recovers the parameters.  ``physical-map`` names a JSON file
that carries the geometry (``lam``, ``E``), the four reference values, the four
shape scales and the recorded injection, and the adapter then writes the four
physical columns, under PyCBC's own parameter names, through the exact inverse
of the chart::

    xi = E diag(1/sqrt(lam)) eta ,     p = p_ref + sigma_p xi_p .

The ``eta`` columns stay in the file, because ``eta`` is the chart in which the
covering, the cell widths and the model error are defined; the physical columns
are a linear recombination of them and are documented as such.  See
:py:class:`ShapeToPhysicalMap` for the transformation, the convention on
``q``, and the check against the recorded injection that a map must pass.

The PyCBC model and the prior
-----------------------------
The adapter does **not** call ``model.loglikelihood``.  The likelihood of this
problem is the amplitude-marginalised Bayes factor evaluated by the research
code, which is not a PyCBC model.  The model supplies the variable-parameter
names that become the columns of the samples file, and its metadata.  This is
stated plainly because it means the posterior in the file is the posterior of
the research likelihood, not of the PyCBC model named in the config.

What maps onto :py:mod:`pycbc.distributions`, and what does not:

* **The four stiff eta coordinates.**  The research prior is flat over the
  region ``sum_a (xi_a / s_a)^2 <= 1`` in shape units, which is the ellipsoid
  ``eta^T M eta <= 1`` in the mismatch coordinates, ``M = A^T A``.  That is a
  uniform density over an ellipsoid: no built-in :py:mod:`pycbc.distributions`
  class targets an ellipsoid, but a product of uniform priors over the bounding
  box plus a ``[constraint-*]`` section whose ``constraint_arg`` is the
  quadratic form represents it exactly.  The file's ``logprior`` field
  carries the log prior mass of each cell under the flat prior over the
  union of the cells, ``V_c / sum_c V_c``.  That is the discrete prior the
  quadrature actually uses; at the boundary it is not the same as the
  continuous ellipsoid prior, because a boundary cell is kept when any part
  of it meets the region and its whole volume is counted.
* **The two sky coordinates.**  In the research chart the sky is not a pair
  of sampled coordinates; it is carried analytically by the linear response
  exponent ``R ~ R0 exp(xi_4 L_0 + xi_5 L_1)`` with ``xi = M eta``.  The
  partition refines the four stiff directions, along which the sky components
  of ``xi`` actually move, and keeps one cell across the two soft directions.
  Those two directions are **not** "the sky": ``E``'s sky rows are spread over
  all six columns, and columns 4 and 5 are the intrinsic-dominated soft
  directions.  A six-column file is therefore rank four, and the sky has to be
  read from the image of the refined directions, not from the two static
  columns.

Which dimensions are actually refined
-------------------------------------
The research covering grid subdivides only the directions it calls stiff; the
others get one cell at zero spanning the prior box.  A column with a single
centre and a single width is not a posterior dimension: every cell reports the
same value, the posterior is flat across that cell, and the file's rank is
lower than its column count.  :meth:`AmrSampler.column_report` counts the
distinct centres and widths per column; the adapter records the refined,
unrefined and degenerate columns and the posterior rank in the file, and logs
a warning when a sampled column is unrefined.  This is what a request to "add
the two sky coordinates" runs into: switching on the two soft columns produces
a six-column file of rank four, and the two added columns are constant.
* **The amplitude and orientation.**  Already integrated inside ``B``: the
  amplitude table, the inclination/polarisation/phase marginalisation and the
  amplitude prior are part of the statistic.  They cannot be written as a
  :py:mod:`pycbc.distributions` prior on the samples, because there is no
  sample column for them; the prior is inside the number that weights the cell.
* **The arrival time.**  Marginalised by the production per-segment
  log-sum-exp over lags inside the 30-day segment, again inside ``B``.  There is
  no ``tc`` column, and the prior over arrival time cannot be separated from
  the cell weight without re-running the statistic per lag.

Configuration
-------------
``[sampler]``::

    name = amr

``[sampler-amr]``::

    # where the partition comes from
    enable-research = false
    partition-file = /path/to/posteriors.npz
    partition-level = last          # 'last' or a level integer
    amr-root = /mnt/d/lisa_tests_with_cc

    # the refinement (research path).  Every option here may instead be put
    # in [sampler] for compatibility with PyCBC's convention; [sampler-amr]
    # wins when both are present.
    refinement = both               # mass | fmax | both
    eps = 3e-4                      # mass fraction per cell
    n-fmax = 100                    # top-2F safety net
    max-cells = 45000               # Bayes-factor evaluation budget
    max-level = 6

    # evidence
    evidence = quadrature           # quadrature | none

    # which eta columns are the variable parameters
    eta-columns = 0 1 2 3

    # the exact projection from eta to the physical parameters.  When set, the
    # four physical columns named in the file are written beside the eta ones.
    physical-map = examples/inference/amr-partition/physical-map.json

    # the injection used only by the research path
    injection-xi = 3 -2 1.5 -1
    injection-rho = 16
    injection-seed = 20261018
    injection-delta = 0
    batch = 16

Every option may instead be placed in ``[sampler]``, which is where PyCBC's
own samplers read theirs; ``[sampler-amr]`` wins when both are present.  Any
other option in ``[sampler-amr]`` is an error, so a typo cannot silently fall
back to a default.
"""

import contextlib
import importlib.util
import json
import logging
import math
import os
import re

import numpy

from .base import BaseSampler, setup_output
from ..io import AmrFile, loadfile


__all__ = ["AmrSampler", "AMR_NATURE", "ShapeToPhysicalMap"]

#: One-line statement of the nature of the object, recorded in the file.
AMR_NATURE = (
    "deterministic grid quadrature on a fixed AMR partition: weighted cell "
    "centres, not a stochastic sampler; no evidence error estimate"
)

#: The research implementation, relative to the companion repository root.
_RESEARCH_PATH = os.path.join(
    "results", "2026-10-18", "amr_sampler", "amr_sampler.py")
#: Default root of the companion repository.
_DEFAULT_AMR_ROOT = "/mnt/d/lisa_tests_with_cc"

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")

#: Options accepted in ``[sampler-amr]``.  Anything else is an error.
_ALLOWED_OPTIONS = frozenset([
    "enable-research", "partition-file", "partition-level", "amr-root",
    "refinement", "eps", "n-fmax", "max-cells", "max-level", "evidence",
    "eta-columns", "physical-map", "injection-xi", "injection-rho",
    "injection-seed", "injection-delta", "batch",
])


def _logsumexp(values):
    """Log of the sum of exponentials, stable for the log-mass arrays."""
    values = numpy.asarray(values, dtype=float)
    vmax = values.max()
    if not numpy.isfinite(vmax):
        return -numpy.inf
    return float(vmax + numpy.log(numpy.exp(values - vmax).sum()))


@contextlib.contextmanager
def _working_directory(path):
    """Change into ``path`` for the duration of the block, then restore.

    The research prototype resolves its data files relative to the companion
    repository root and does its own ``os.chdir`` at import time, so the
    research path has to run with that as the process directory.  Restoring it
    afterwards keeps the rest of PyCBC unaffected.
    """
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _as_bool(value, option):
    """Parse a config boolean, with a message that names the option."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(
        "{} must be one of {} or {}, got {!r}".format(
            option, "/".join(_TRUE), "/".join(_FALSE), value))


def _as_list(value):
    """Split a config list on commas and whitespace."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [item for item in re.split(r"[,\s]+", str(value).strip())
            if item]


class ShapeToPhysicalMap(object):
    r"""The exact projection from the AMR shape coordinates to physical parameters.

    THE CHART
    ---------
    The research sampler works in the chart

        ``eta_a = sqrt(lam_a) (E^T xi)_a`` ,

    where ``xi`` is the offset of the four parameters from the reference point,
    in the research's shape units, ``E`` holds the eigenvectors of the search
    cell's Fisher matrix as columns and ``lam`` its eigenvalues.  ``E`` is
    orthogonal (``E^T E = 1``), so the chart inverts without an approximation:

        ``xi = E diag(1/sqrt(lam)) eta`` .

    A physical parameter is an affine function of its own shape offset, because
    the research displacement is ``point[name] += xi[name] * sig[name]``:

        ``p = p_ref + sigma_p xi_p`` .

    So ``p = p_ref + sigma_p (M eta)_p`` with ``M = E diag(1/sqrt(lam))``.  The
    map is **linear in eta**, which is exactly why the coordinates cannot be
    renamed to physical parameters: ``eta_a`` is a linear combination of all
    four of them, and a linear combination of physical parameters is not a
    physical parameter.

    THE CONVENTION ON ``q``
    -----------------------
    The research chart's mass ratio is ``q_proj = m2/m1 <= 1``
    (``experiments/ratio_cells/coordinates.py``), while PyCBC's canonical ``q``
    is ``m1/m2 >= 1`` (``pycbc.waveform.parameters.q``,
    ``pycbc.conversions.q_from_mass1_mass2``).  A parameter entry with
    ``kind = "reciprocal"`` therefore stores ``1 / (p_ref + sigma xi)``, which
    is PyCBC's ``q``.  The transformation is monotone, so percentiles of the
    stored column transform exactly into percentiles of PyCBC's ``q``; only
    linearity is lost, and the linear coordinate is ``1/q``.  ``mchirp`` and
    the two aligned spins are the same quantity in both conventions
    (``mchirp = (m1 m2)^(3/5) / (m1 + m2)^(1/5)``).

    THE CHECK A MAP MUST PASS
    -------------------------
    A map carries the injection that defines the problem, as ``xi`` (the six
    shape components) and as ``eta`` (the six chart components).  On load the
    class recomputes ``M @ eta`` and raises if it does not return ``xi``, so a
    matrix that is not the one the sampler used cannot be applied silently.
    The two non-sampled chart coordinates are static at zero for every cell of
    the partition, but they are **not** zero for the injection: the
    injection's ``xi`` has components along those eigendirections, and
    ``1/sqrt(lam)`` for them is large (order 1e4 and 1e7), so their
    contribution to ``xi`` is of order unity.  That is why the check must use
    all six components: applying the 4x4 block to the first four entries of
    the recorded ``eta`` does **not** return the recorded ``xi``.

    Note also that those two directions are **not** the sky.  ``E`` holds the
    eigendirections of the projected Fisher matrix, and its sky rows are
    spread over all six columns: columns 4 and 5 are the two softest
    directions, dominated by the mass ratio, chirp mass and spins.  The sky
    components of ``xi`` are dominated by the four *stiff* directions, which
    the partition refines.  ``eta_4`` and ``eta_5`` are therefore "the two
    unrefined chart directions", not "the two sky coordinates".
    """

    #: The number of chart coordinates of the research geometry.
    ndim = 6
    #: Accepted transformation kinds.
    kinds = ("identity", "reciprocal")

    def __init__(self, lam, E, eta_columns, parameters, eta_static=None,
                 injection=None, source=None):
        self.lam = numpy.asarray(lam, dtype=float)
        self.E = numpy.asarray(E, dtype=float)
        if self.lam.shape != (self.ndim,):
            raise ValueError(
                "lam must have {} entries, got {}".format(
                    self.ndim, self.lam.shape))
        if self.E.shape != (self.ndim, self.ndim):
            raise ValueError(
                "E must be {}x{}, got {}".format(
                    self.ndim, self.ndim, self.E.shape))
        orth = numpy.max(numpy.abs(self.E.T @ self.E - numpy.eye(self.ndim)))
        if orth > 1e-8:
            raise ValueError(
                "E must be orthogonal for the chart to invert exactly; "
                "max |E^T E - 1| = {:.3e}".format(orth))
        self.eta_columns = [int(column) for column in eta_columns]
        if len(set(self.eta_columns)) != len(self.eta_columns):
            raise ValueError(
                "eta_columns must be unique, got {}".format(self.eta_columns))
        for column in self.eta_columns:
            if column < 0 or column >= self.ndim:
                raise ValueError(
                    "eta_columns must be in 0..{}, got {}".format(
                        self.ndim - 1, self.eta_columns))
            if not self.lam[column] > 0:
                raise ValueError(
                    "eta column {} has lam = {}; the chart needs lam > 0 to "
                    "invert".format(column, self.lam[column]))
        #: The static values of the chart coordinates that are not sampled.
        self.eta_static = numpy.zeros(self.ndim)
        if eta_static is not None:
            for key, value in dict(eta_static).items():
                self.eta_static[int(key)] = float(value)
        self.injection = dict(injection) if injection else {}
        self.source = source

        self.parameters = []
        seen = set()
        for entry in parameters:
            name = str(entry["name"])
            index = int(entry["xi_index"])
            kind = str(entry.get("kind", "identity"))
            if kind not in self.kinds:
                raise ValueError(
                    "parameter {} has kind {!r}, not one of {}".format(
                        name, kind, self.kinds))
            if index < 0 or index >= self.ndim:
                raise ValueError(
                    "parameter {} has xi_index {}, outside 0..{}".format(
                        name, index, self.ndim - 1))
            if index in seen:
                raise ValueError(
                    "two parameters share xi_index {}".format(index))
            seen.add(index)
            self.parameters.append(dict(
                name=name,
                xi_index=index,
                reference=float(entry["reference"]),
                scale=float(entry["scale"]),
                kind=kind,
                units=str(entry.get("units", "")),
                convention=str(entry.get("convention", ""))))

        #: ``M[i][a] = E[i][a] / sqrt(lam_a)``, so ``xi = M @ eta``.
        self.matrix = numpy.zeros((self.ndim, self.ndim))
        for axis in range(self.ndim):
            if self.lam[axis] > 0:
                self.matrix[:, axis] = self.E[:, axis] / math.sqrt(
                    self.lam[axis])
            elif self.eta_static[axis] != 0.0:
                raise ValueError(
                    "eta coordinate {} is nonzero but lam = {}; the chart "
                    "cannot be inverted there".format(axis, self.lam[axis]))

        if self.injection:
            self._check_injection()

    # -- construction ----------------------------------------------------------------------------
    @classmethod
    def from_file(cls, path):
        """Reads and validates a map file, and re-checks its geometry source."""
        with open(path, "r") as handle:
            spec = json.load(handle)
        geometry = spec.get("geometry", {})
        if geometry.get("file"):
            cls._check_geometry(geometry)
        return cls(
            lam=geometry.get("lam", spec.get("lam")),
            E=geometry.get("E", spec.get("E")),
            eta_columns=spec.get("eta_columns", [0, 1, 2, 3]),
            parameters=spec["parameters"],
            eta_static=spec.get("eta_static"),
            injection=spec.get("injection"),
            source=os.path.abspath(path))

    @staticmethod
    def _check_geometry(geometry):
        """Re-reads the geometry the map was built from and compares it.

        A silently stale map is the one failure mode that would relabel the
        axes with a matrix the sampler did not use, so the file that ``lam``
        and ``E`` came from is read again when it is available.  A missing
        file is not an error: the map is self-contained, and the source path
        is recorded in the samples file either way.
        """
        path = geometry["file"]
        if not os.path.isfile(path):
            logging.warning(
                "AMR physical map: geometry file %s is not available, so the "
                "stored lam and E cannot be re-checked against it", path)
            return
        with open(path, "r") as handle:
            stored = json.load(handle)
        lam = numpy.asarray(stored["lam"], dtype=float)
        E = numpy.asarray(stored["E"], dtype=float)
        same_lam = numpy.allclose(lam, numpy.asarray(geometry["lam"],
                                                     dtype=float),
                                  rtol=1e-10, atol=0.0)
        same_E = numpy.allclose(E, numpy.asarray(geometry["E"], dtype=float),
                                rtol=1e-10, atol=1e-12)
        if not (same_lam and same_E):
            raise ValueError(
                "the physical map's geometry does not match {}: "
                "max |dlam|/lam = {:.3e}, max |dE| = {:.3e}".format(
                    path,
                    numpy.max(numpy.abs(lam - numpy.asarray(geometry["lam"]))
                              / numpy.maximum(numpy.abs(lam), 1e-300)),
                    numpy.max(numpy.abs(E - numpy.asarray(geometry["E"])))))

    def _check_injection(self):
        """Checks that the matrix returns the injection's recorded shape offset."""
        xi = numpy.asarray(self.injection["xi"], dtype=float)
        eta = numpy.asarray(self.injection["eta"], dtype=float)
        if xi.shape != (self.ndim,) or eta.shape != (self.ndim,):
            raise ValueError(
                "the injection must be stored as six component xi and eta "
                "vectors, got {} and {}".format(xi.shape, eta.shape))
        back = self.matrix @ eta
        worst = numpy.max(numpy.abs(back - xi))
        if worst > 1e-8 * max(1.0, numpy.max(numpy.abs(xi))):
            raise ValueError(
                "the map does not reproduce the recorded injection: "
                "M @ eta = {}, recorded xi = {} (max difference {:.3e}). The "
                "matrix, the eigenvalues or the injection are not the ones "
                "the sampler used.".format(back.tolist(), xi.tolist(), worst))

    # -- use -------------------------------------------------------------------------------------
    def physical_names(self):
        """The physical parameter names, in the order the columns are written."""
        return [entry["name"] for entry in self.parameters]

    def xi(self, eta):
        """The six shape-unit offsets of the given sampled chart coordinates."""
        eta = numpy.atleast_2d(numpy.asarray(eta, dtype=float))
        if eta.shape[1] != len(self.eta_columns):
            raise ValueError(
                "expected {} eta columns, got {}".format(
                    len(self.eta_columns), eta.shape[1]))
        full = numpy.repeat(self.eta_static[None, :], len(eta), axis=0)
        for position, column in enumerate(self.eta_columns):
            full[:, column] = eta[:, position]
        return full @ self.matrix.T

    def physical(self, eta):
        """The physical parameters of the given chart coordinates.

        Returns a dict in the map's parameter order, so that inserting it into
        the samples dictionary writes the columns in a documented order.
        """
        return self._physical_from_xi(self.xi(eta))

    def _physical_from_xi(self, xi):
        """Applies ``p = p_ref + sigma_p xi_p`` and the q convention to xi."""
        xi = numpy.atleast_2d(numpy.asarray(xi, dtype=float))
        out = {}
        for entry in self.parameters:
            value = (entry["reference"]
                     + entry["scale"] * xi[:, entry["xi_index"]])
            if entry["kind"] == "reciprocal":
                value = 1.0 / value
            out[entry["name"]] = value
        return out

    def truth(self):
        """The physical parameters of the injected signal, from its ``xi``.

        This is the physical point the injection was placed at, ``p_ref +
        sigma xi_int``, and it is what a reader means by "the truth".
        """
        if not self.injection:
            return {}
        xi = numpy.asarray(self.injection["xi"], dtype=float)
        return {name: values[0]
                for name, values in self._physical_from_xi(xi).items()}

    def xi_from_physical(self, values):
        """The shape offsets that reproduce the given physical parameters.

        The inverse of :py:meth:`physical`: ``xi_p = (p - p_ref) / sigma_p``,
        with the reciprocal convention undone first for ``q``.  This is what
        places an injection quoted in physical parameters back into the chart;
        values are scalars, one per parameter name.
        """
        xi = numpy.zeros(self.ndim)
        for entry in self.parameters:
            value = float(numpy.asarray(values[entry["name"]], dtype=float))
            if entry["kind"] == "reciprocal":
                if value == 0.0:
                    raise ValueError(
                        "parameter {} is reciprocal and cannot be zero".format(
                            entry["name"]))
                value = 1.0 / value
            xi[entry["xi_index"]] = (value - entry["reference"]) / entry["scale"]
        return xi

    def check_partition_truth(self, eta, atol=1e-6):
        """Refuses a partition that was produced at a different injection.

        The research ``posteriors.npz`` stores the injection's chart
        coordinates as ``truth``.  Comparing them with the map's own injection
        catches a partition from another run before its cells are relabelled
        with a matrix from this one: the map is only valid for the geometry and
        the injection it was built for.
        """
        if not self.injection:
            return
        eta = numpy.asarray(eta, dtype=float).ravel()
        expected = numpy.asarray(self.injection["eta"], dtype=float)
        if eta.shape != expected.shape or (
                numpy.max(numpy.abs(eta - expected)) > atol):
            raise ValueError(
                "the partition was produced at a different injection from the "
                "one the physical map records: stored truth {}, map eta {}; "
                "this is a different geometry or run".format(
                    eta.tolist(), expected.tolist()))

    def chart_image(self):
        """The truth as the four-coordinate chart represents it.

        The injection also has small components along the two non-sampled
        chart coordinates, and the partition sets those to zero.  Mapping the
        recorded ``eta`` with the sampled columns only therefore lands at a
        slightly different physical point from :py:meth:`truth`; the
        difference is the chart-truncation budget, recorded so that it is
        quoted rather than assumed to be zero.
        """
        if not self.injection:
            return {}
        eta = numpy.asarray(self.injection["eta"], dtype=float)[None, :]
        return {name: values[0]
                for name, values
                in self.physical(eta[:, self.eta_columns]).items()}

    def metadata(self):
        """The attributes the samples file records about the transformation."""
        entry = {"amr_physical_params": " ".join(self.physical_names()),
                 "amr_shape_matrix": self.matrix,
                 "amr_shape_lam": self.lam,
                 "amr_shape_eigenvectors": self.E}
        if self.source is not None:
            entry["amr_physical_map"] = str(self.source)
        entry["amr_shape_reference"] = numpy.asarray(
            [item["reference"] for item in self.parameters], dtype=float)
        entry["amr_shape_scale"] = numpy.asarray(
            [item["scale"] for item in self.parameters], dtype=float)
        entry["amr_physical_convention"] = " ".join(
            "{}={}".format(item["name"], item["kind"]) for item in self.parameters)
        if self.injection:
            entry["amr_injection_xi"] = numpy.asarray(
                self.injection["xi"], dtype=float)
            entry["amr_injection_eta"] = numpy.asarray(
                self.injection["eta"], dtype=float)
            truth = self.truth()
            image = self.chart_image()
            entry["amr_injection_physical"] = numpy.asarray(
                [truth[name] for name in self.physical_names()], dtype=float)
            entry["amr_injection_physical_chart"] = numpy.asarray(
                [image[name] for name in self.physical_names()], dtype=float)
        return entry


class AmrSampler(BaseSampler):
    """The AMR cell partition wrapped as a PyCBC inference sampler.

    This is a deterministic grid quadrature, not a stochastic sampler; see the
    module docstring.  The class implements the :py:class:`BaseSampler`
    interface so that ``name = amr`` in ``[sampler]`` works and
    ``pycbc_inference`` writes a samples file that
    ``pycbc_inference_plot_posterior`` can read.

    Parameters
    ----------
    model : pycbc.inference.models.BaseModel
        The model.  Its ``variable_params`` name the columns of the samples
        file; its likelihood is not evaluated (the research code supplies the
        Bayes factor).
    enable_research : bool, optional
        Run the research AMR live instead of reading a stored partition.
        Default is False.
    partition_file : str, optional
        The ``posteriors.npz`` written by the research prototype.  Required
        when ``enable_research`` is False.
    partition_level : str or int, optional
        Which refinement level to read: ``'last'`` (the default) or an
        integer.
    amr_root : str, optional
        Root of the companion repository.  Default is
        ``/mnt/d/lisa_tests_with_cc``.
    refinement : {'mass', 'fmax', 'both'}, optional
        The refinement criterion of the research path.  ``mass`` refines cells
        whose Bayes factor times volume exceeds ``eps`` of the running total;
        ``fmax`` refines the ``n_fmax`` cells with the largest maximum 2F;
        ``both`` is the research default and is the union of the two.
    eps : float, optional
        The mass fraction for the ``mass`` criterion.  Default is 3e-4.
    n_fmax : int, optional
        Number of top-2F cells kept by the ``fmax`` criterion.  Default 100.
    max_cells : int, optional
        The Bayes-factor evaluation budget of the research path.  Default
        45000.
    max_level : int, optional
        Maximum refinement level.  Default 6.
    evidence : {'quadrature', 'none'}, optional
        How to estimate ``log_evidence``.  ``quadrature`` is the Riemann sum
        over the partition, normalised by the covered volume; ``none`` writes
        ``nan``.  In both cases ``dlog_evidence`` is ``nan``.  Default
        ``quadrature``.
    eta_columns : list of int, optional
        Which columns of the six-dimensional eta vector are the variable
        parameters, in the order of ``model.variable_params``.  Default is to
        infer ``eta_<k>`` from the parameter names, and otherwise the first
        ``len(variable_params)`` columns.
    physical_map : str, optional
        Path to the JSON file that carries the projection from the shape
        coordinates to the physical parameters.  When given, the four physical
        columns it names are written beside the ``eta`` ones.  See
        :py:class:`ShapeToPhysicalMap`.
    injection_xi, injection_rho, injection_seed, injection_delta, batch :
        The injection and batch size of the research path, matching the
        prototype's ``main``.
    """

    name = "amr"
    _io = AmrFile

    def __init__(self, model, enable_research=False, partition_file=None,
                 partition_level="last", amr_root=None, refinement="both",
                 eps=3e-4, n_fmax=100, max_cells=45000, max_level=6,
                 evidence="quadrature", eta_columns=None, physical_map=None,
                 injection_xi=(3.0, -2.0, 1.5, -1.0), injection_rho=16.0,
                 injection_seed=20261018, injection_delta=0.0, batch=16,
                 nprocesses=1, use_mpi=False):
        super().__init__(model)

        self.enable_research = bool(enable_research)
        self.source = "research" if self.enable_research else "partition"
        self.partition_file = partition_file
        self.partition_level = partition_level
        self.amr_root = amr_root or _DEFAULT_AMR_ROOT
        self.refinement = refinement
        self.eps = float(eps)
        self.n_fmax = int(n_fmax)
        self.max_cells = int(max_cells)
        self.max_level = int(max_level)
        self.evidence = evidence
        self.injection_xi = numpy.asarray(injection_xi, dtype=float)
        self.injection_rho = float(injection_rho)
        self.injection_seed = int(injection_seed)
        self.injection_delta = float(injection_delta)
        self.batch = int(batch)
        self.nprocesses = nprocesses
        self.use_mpi = use_mpi
        self.physical_map = (None if physical_map in (None, "", "none")
                             else ShapeToPhysicalMap.from_file(physical_map))

        self.eta_columns = self._resolve_eta_columns(eta_columns)

        self._samples = None
        self._logz = None
        self._dlogz = numpy.nan
        self._research_module = None
        self.level_info = None
        self.column_info = None
        self.checkpoint_file = None
        self.backup_file = None
        self.new_checkpoint = None

        self._validate_options()

    # -- configuration ---------------------------------------------------------------------------
    def _resolve_eta_columns(self, eta_columns):
        """Work out which eta columns the variable parameters correspond to."""
        nparams = len(self.model.variable_params)
        if eta_columns is None:
            inferred = []
            for param in self.model.variable_params:
                match = re.fullmatch(r"eta_?(\d+)", param)
                if match is not None:
                    inferred.append(int(match.group(1)))
            if len(inferred) != nparams:
                inferred = list(range(nparams))
            eta_columns = inferred
        elif isinstance(eta_columns, str):
            eta_columns = [int(item) for item in _as_list(eta_columns)]
        else:
            eta_columns = [int(item) for item in eta_columns]
        if len(eta_columns) != nparams:
            raise ValueError(
                "eta-columns has {} entries but the model has {} variable "
                "parameters {}".format(
                    len(eta_columns), nparams,
                    list(self.model.variable_params)))
        if len(set(eta_columns)) != len(eta_columns):
            raise ValueError("eta-columns must be unique, got {}".format(
                eta_columns))
        if any(column < 0 or column >= 6 for column in eta_columns):
            raise ValueError(
                "eta-columns must be in 0..5, got {}".format(eta_columns))
        return eta_columns

    def _validate_options(self):
        """Reject combinations that cannot produce a meaningful file."""
        if self.refinement not in ("mass", "fmax", "both"):
            raise ValueError(
                "refinement must be mass, fmax or both, got {!r}".format(
                    self.refinement))
        if self.evidence not in ("quadrature", "none"):
            raise ValueError(
                "evidence must be quadrature or none, got {!r}".format(
                    self.evidence))
        if self.max_level < 0:
            raise ValueError("max-level must be non-negative")
        if self.max_cells < 16:
            raise ValueError("max-cells must be at least 16")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if not self.enable_research and self.partition_file is None:
            raise ValueError(
                "enable-research is false, so partition-file must name the "
                "posteriors.npz written by the research prototype; there is "
                "no way to obtain the AMR Bayes factor from the PyCBC model")
        if self.injection_xi.shape != (4,):
            raise ValueError(
                "injection-xi must have four shape-unit components")
        if self.physical_map is not None:
            if list(self.physical_map.eta_columns) != list(self.eta_columns):
                raise ValueError(
                    "physical-map declares eta columns {} but eta-columns is "
                    "{}; the map must act on the coordinates being sampled "
                    "and reported".format(
                        self.physical_map.eta_columns, self.eta_columns))

    @classmethod
    def from_config(cls, cp, model, output_file=None, nprocesses=1,
                    use_mpi=False):
        """Loads the sampler from the given config file.

        Reads ``name`` from ``[sampler]`` (which must be ``amr``) and the
        adapter's options from ``[sampler-amr]``.  Unknown options in
        ``[sampler-amr]`` raise an error rather than being ignored.
        """
        section = "sampler"
        if cp.get(section, "name") != cls.name:
            raise ValueError(
                "name in section [{}] must be {!r}, got {!r}".format(
                    section, cls.name, cp.get(section, "name")))

        sub = "sampler-" + cls.name
        if cp.has_section(sub):
            unknown = set(cp[sub].keys()) - _ALLOWED_OPTIONS
            if unknown:
                raise RuntimeError(
                    "Unknown options in [{}]: {}. Allowed: {}".format(
                        sub, ", ".join(sorted(unknown)),
                        ", ".join(sorted(_ALLOWED_OPTIONS))))
        else:
            cp.add_section(sub)

        def get(option, default=None):
            # [sampler-amr] is where this adapter documents its options, but
            # PyCBC's own samplers read their options from [sampler], so both
            # are accepted and the dedicated section wins.
            if cp.has_option(sub, option):
                return cp.get(sub, option)
            if cp.has_option(section, option):
                return cp.get(section, option)
            return default

        partition_level = get("partition-level", "last")
        if str(partition_level).strip().lower() != "last":
            partition_level = int(partition_level)

        eta_columns = get("eta-columns", None)
        if eta_columns is not None:
            eta_columns = [int(item) for item in _as_list(eta_columns)]

        injection_xi = get("injection-xi", None)
        if injection_xi is None:
            injection_xi = (3.0, -2.0, 1.5, -1.0)
        else:
            injection_xi = [float(item) for item in _as_list(injection_xi)]

        obj = cls(
            model,
            enable_research=_as_bool(
                get("enable-research", "false"), "enable-research"),
            partition_file=get("partition-file", None),
            partition_level=partition_level,
            amr_root=get("amr-root", None),
            refinement=str(get("refinement", "both")).strip().lower(),
            eps=float(get("eps", 3e-4)),
            n_fmax=int(get("n-fmax", 100)),
            max_cells=int(get("max-cells", 45000)),
            max_level=int(get("max-level", 6)),
            evidence=str(get("evidence", "quadrature")).strip().lower(),
            eta_columns=eta_columns,
            physical_map=get("physical-map", None),
            injection_xi=injection_xi,
            injection_rho=float(get("injection-rho", 16.0)),
            injection_seed=int(get("injection-seed", 20261018)),
            injection_delta=float(get("injection-delta", 0.0)),
            batch=int(get("batch", 16)),
            nprocesses=nprocesses,
            use_mpi=use_mpi,
        )

        # a deterministic quadrature has no samples to count, so do not let
        # setup_output require a non-empty samples group
        setup_output(obj, output_file, check_nsamples=False)
        if not obj.new_checkpoint:
            obj.resume_from_checkpoint()
        return obj

    # -- BaseSampler interface -------------------------------------------------------------------
    @property
    def io(self):
        return self._io

    @property
    def samples(self):
        """The weighted partition: one row per cell.

        The dictionary maps each variable parameter to the cell centres, and
        also contains ``loglikelihood`` (the cell Bayes factor), ``logprior``
        (the log prior mass of the cell), ``logwt`` (their sum, the log
        posterior mass) and ``snr`` (the cell's effective SNR,
        ``sqrt(2 loglikelihood)``).  These are cells, not draws.

        When a ``physical-map`` is configured the four physical parameters are
        present as well, computed through :py:class:`ShapeToPhysicalMap`.  They
        are a linear recombination of the ``eta`` columns, not independent
        information: the ``eta`` columns are the chart the covering and the
        model error are defined in, and the physical columns are the same
        posterior expressed in the parameters a reader wants.
        """
        if self._samples is None:
            return {}
        return self._samples

    def physical_metadata(self):
        """The transformation attributes to record in the samples file."""
        if self.physical_map is None:
            return {}
        return self.physical_map.metadata()

    @property
    def model_stats(self):
        """The posterior statistics of the partition, or None if not run."""
        if self._samples is None:
            return None
        return {key: self._samples[key]
                for key in ("loglikelihood", "logprior", "logwt")}

    def run(self):
        """Produces the weighted partition and writes it to the checkpoint.

        Deterministic: if a valid checkpoint is already in memory (from
        :py:meth:`resume_from_checkpoint`) the stored partition is reused,
        because re-running would reproduce it exactly.
        """
        if self._samples is not None:
            logging.info(
                "AMR: reusing the stored deterministic partition "
                "(%d cells); no resampling is performed",
                len(self._samples["logwt"]))
            return
        eta, widths, lnb = self._partition()
        self._set_samples(eta, widths, lnb)
        for filename in [self.checkpoint_file, self.backup_file]:
            self.write_results(filename)

    def _partition(self):
        """Obtains ``(eta, widths, lnB)`` from the configured source."""
        if self.enable_research:
            return self._research_partition()
        return self._load_partition()

    def _load_partition(self):
        """Reads a weighted partition written by the research prototype."""
        if not os.path.isfile(self.partition_file):
            raise OSError(
                "AMR partition file does not exist: {}".format(
                    self.partition_file))
        with numpy.load(self.partition_file) as data:
            levels = sorted(
                int(key.rsplit("_L", 1)[1]) for key in data.files
                if key.startswith("eta_L"))
            if not levels:
                raise ValueError(
                    "{} has no eta_L<level> arrays".format(
                        self.partition_file))
            if str(self.partition_level).lower() == "last":
                level = levels[-1]
            else:
                level = int(self.partition_level)
            if level not in levels:
                raise ValueError(
                    "level {} is not in {} (levels {})".format(
                        level, self.partition_file, levels))
            eta = numpy.asarray(data["eta_L%d" % level], dtype=float)
            widths = numpy.asarray(data["w_L%d" % level], dtype=float)
            lnb = numpy.asarray(data["lnB_L%d" % level], dtype=float)
            # a partition carries the injection it was produced at, so bind it
            # to the physical map before any cell is relabelled
            if self.physical_map is not None and "truth" in data.files:
                self.physical_map.check_partition_truth(
                    numpy.asarray(data["truth"], dtype=float))
        logging.info(
            "AMR: read level %d from %s: %d cells; this is a deterministic "
            "partition, not a chain", level, self.partition_file, len(eta))
        self.level_info = {
            "level": level,
            "n_cells": int(len(eta)),
            "source": "partition",
        }
        return eta, widths, lnb

    def _research_partition(self):
        """Runs the research AMR live and returns its final partition."""
        module = self._import_research()
        with _working_directory(self.amr_root):
            # every physics call below is the research implementation: the
            # production Run, its injection helper, its Bayes factor and its
            # split/containment algebra
            run = module.A.Run(batch=self.batch)
            signal, _ = module.fixed_signal(
                run, self.injection_xi, self.injection_delta)
            rng = numpy.random.default_rng(self.injection_seed)
            spectrum = run.spectrum_of(
                self.injection_rho * signal + run.noise_bins(rng))
            segment = int(math.floor(
                (self.injection_delta + module.A.DELTA_RANGE) /
                (module.A.SEG_DAYS * module.A.H.DAY)))
            posterior = module.PosteriorAMR(
                run, spectrum, segment, numpy.zeros(6), eps=self.eps)
            eta, widths = run.geo.first_level(2)
            for _ in range(module.COVER_LEVELS):
                eta, widths, _ = run.geo.children(eta, widths)
            logging.info(
                "AMR: research path on %d covering cells, segment %d, "
                "refinement %s", len(eta), segment, self.refinement)
            levels = self._refine(posterior, eta, widths, module)
        final = levels[-1]
        self.level_info = {
            "level": final["level"],
            "n_cells": final["n_cells"],
            "n_eval": final["n_eval"],
            "log_total_mass": final["log_total"],
            "source": "research",
        }
        return final["eta"], final["w"], final["lnB"]

    def _import_research(self):
        """Imports the research prototype by path, without copying it.

        The prototype is a script: it inserts its own ``sys.path`` entries and
        changes the process directory to the companion repository root at
        import time.  Both effects are wanted for the research run and both
        are contained by the caller, which restores the directory.
        """
        if self._research_module is not None:
            return self._research_module
        if not os.path.isdir(self.amr_root):
            raise OSError(
                "the research path needs the companion repository; amr-root "
                "{} is not a directory".format(self.amr_root))
        path = os.path.join(self.amr_root, _RESEARCH_PATH)
        if not os.path.isfile(path):
            raise OSError(
                "the research path needs {}, which does not exist; set "
                "amr-root to the companion repository root".format(path))
        spec = importlib.util.spec_from_file_location(
            "_pycbc_amr_research_impl", path)
        if spec is None or spec.loader is None:
            raise ImportError(
                "could not load the research AMR implementation from "
                "{}".format(path))
        module = importlib.util.module_from_spec(spec)
        try:
            with _working_directory(self.amr_root):
                spec.loader.exec_module(module)
        except ImportError as exc:
            raise ImportError(
                "the research AMR implementation at {} could not be imported "
                "({}). The research path needs cupy, a CUDA GPU and the "
                "companion repository's data; nothing in PyCBC can supply "
                "the AMR Bayes factor. Use enable-research = false with a "
                "stored partition-file instead.".format(path, exc)) from exc
        self._research_module = module
        return module

    def _refine(self, posterior, eta, widths, module):
        """The AMR refinement loop, with the criterion as an option.

        Only the control flow lives here.  The cell algebra -- ``lnB``,
        ``logmass``, ``split`` and the containment test -- is the research
        implementation's.  With ``refinement = both``, ``eps = 3e-4``,
        ``n_fmax = 100`` and the same ``max_level`` this reproduces
        ``PosteriorAMR.run`` exactly: the same union, sort order, budget and
        stopping rule.
        """
        logsumexp = getattr(module, "logsumexp", _logsumexp)
        eta = numpy.array(eta, dtype=float)
        widths = numpy.array(widths, dtype=float)
        levels = []
        for level in range(self.max_level + 1):
            keys = [tuple(numpy.round(numpy.r_[eta[i], widths[i]], 12))
                    for i in range(len(eta))]
            lnb, fmax = posterior.lnB(eta, keys)
            logmass = posterior.logmass(lnb, widths)
            total = logsumexp(logmass)
            if self.refinement in ("mass", "both"):
                selected_mass = numpy.nonzero(
                    logmass > total + math.log(self.eps))[0]
            else:
                selected_mass = numpy.array([], dtype=int)
            if self.refinement in ("fmax", "both") and self.n_fmax > 0:
                available = int(numpy.sum(fmax > 0))
                count = min(self.n_fmax, available)
                selected_fmax = (numpy.argsort(-fmax)[:count] if count
                                 else numpy.array([], dtype=int))
            else:
                selected_fmax = numpy.array([], dtype=int)
            selected = numpy.union1d(selected_mass, selected_fmax)
            room = max(0, (self.max_cells - posterior.n_eval) // 16)
            n_refine = int(min(len(selected), room))
            complete = bool(len(selected) <= room)
            levels.append({
                "level": level,
                "eta": eta.copy(),
                "w": widths.copy(),
                "lnB": lnb,
                "fmax": fmax,
                "logmass": logmass,
                "log_total": total,
                "n_cells": int(len(eta)),
                "n_eval": int(posterior.n_eval),
                "n_mass": int(len(selected_mass)),
                "n_fmax": int(len(selected_fmax)),
                "n_refined": n_refine,
                "complete": complete,
            })
            logging.info(
                "AMR level %d: %d cells, mass-refine %d, 2F-top %d, "
                "union %d, refined %d, evaluated %d, log mass %.2f",
                level, len(eta), len(selected_mass), len(selected_fmax),
                len(selected), n_refine, posterior.n_eval, total)
            if level == self.max_level or n_refine == 0:
                break
            selected = selected[numpy.argsort(-logmass[selected])[:n_refine]]
            child_eta, child_w = posterior.split(
                eta[selected], widths[selected])
            keep = numpy.ones(len(eta), dtype=bool)
            keep[selected] = False
            eta = numpy.vstack([eta[keep], child_eta])
            widths = numpy.vstack([widths[keep], child_w])
        return levels

    @staticmethod
    def column_report(eta, widths):
        """Per-column refinement of a partition.

        A column is **refined** when the partition subdivides it: it has more
        than one distinct centre or more than one distinct width.  A column
        with a single centre and a single width is one cell spanning its whole
        prior range; every cell reports the same coordinate value, the
        posterior is flat across that cell, and the file's rank is lower than
        its column count.  That is not a numerical artefact -- it is how the
        research covering grid treats the non-stiff directions.

        Returns a dict with, for every column, ``n_centres``, ``n_widths``,
        ``centre_min``, ``centre_max``, ``width_min`` and ``width_max``, plus
        the lists ``refined`` and ``unrefined``.  ``degenerate`` and ``rank``
        are added by :meth:`_set_samples`, which knows which columns are
        sampled.
        """
        eta = numpy.asarray(eta, dtype=float)
        widths = numpy.asarray(widths, dtype=float)
        if eta.ndim != 2 or widths.shape != eta.shape:
            raise ValueError(
                "eta and widths must be equal-shaped 2-D arrays, got {} and "
                "{}".format(eta.shape, widths.shape))
        ncols = eta.shape[1]
        report = {
            "n_centres": [int(numpy.unique(eta[:, a]).size)
                          for a in range(ncols)],
            "n_widths": [int(numpy.unique(widths[:, a]).size)
                         for a in range(ncols)],
            "centre_min": [float(eta[:, a].min()) for a in range(ncols)],
            "centre_max": [float(eta[:, a].max()) for a in range(ncols)],
            "width_min": [float(widths[:, a].min()) for a in range(ncols)],
            "width_max": [float(widths[:, a].max()) for a in range(ncols)],
        }
        report["refined"] = [
            a for a in range(ncols)
            if report["n_centres"][a] > 1 or report["n_widths"][a] > 1]
        report["unrefined"] = [a for a in range(ncols)
                               if a not in report["refined"]]
        return report

    def _set_samples(self, eta, widths, lnb):
        """Converts the weighted partition into the samples format.

        ``V_c`` is the cell volume in the variables actually being reported,
        i.e. the ``eta_columns`` directions.  ``logprior`` is the log prior
        mass of the cell and ``logwt = loglikelihood + logprior`` is its log
        posterior mass, so the file is internally consistent and the evidence
        is their log-sum.
        """
        eta = numpy.asarray(eta, dtype=float)
        widths = numpy.asarray(widths, dtype=float)
        lnb = numpy.asarray(lnb, dtype=float)
        columns = self.eta_columns
        volume = numpy.prod(widths[:, columns], axis=1)
        if not numpy.all(volume > 0):
            raise ValueError(
                "every cell must have a positive volume in the reported "
                "eta columns {}".format(columns))
        logmass = lnb + numpy.log(volume)
        log_total = _logsumexp(logmass)
        log_cover = math.log(float(volume.sum()))
        logwt = logmass - log_cover
        logprior = numpy.log(volume) - log_cover
        # The cell's effective SNR: 2 ln B = rho^2 for a fixed template, so
        # rho = sqrt(2 ln B).  Clip at zero so B < 1 gives snr = 0 rather
        # than nan.  See the module docstring for the derivation.
        snr = numpy.sqrt(2.0 * numpy.clip(lnb, 0.0, None))

        samples = {"loglikelihood": lnb,
                   "logprior": logprior,
                   "logwt": logwt,
                   "snr": snr}
        for param, column in zip(self.model.variable_params, columns):
            samples[param] = eta[:, column]
        # the physical parameters are the same posterior in the parameters a
        # reader wants; they are computed from the eta columns, not resampled
        physical = {}
        if self.physical_map is not None:
            physical = self.physical_map.physical(eta[:, columns])
            samples.update(physical)
        # make the insertion order deterministic and informative
        ordered = {param: samples[param]
                   for param in self.model.variable_params}
        ordered.update(physical)
        ordered.update(loglikelihood=lnb, logprior=logprior, logwt=logwt,
                       snr=snr)
        self._samples = ordered
        # Which of the reported columns the partition actually refines.  A
        # sampled column with one centre and one width is not a posterior
        # dimension: the file has that column but its rank is lower.
        report = self.column_report(eta, widths)
        report["degenerate"] = [a for a in columns
                                if a in report["unrefined"]]
        report["rank"] = len(columns) - len(report["degenerate"])
        self.column_info = report
        if report["degenerate"]:
            logging.warning(
                "AMR: the sampled column(s) %s are NOT refined by the "
                "partition: every one of the %d cells reports the same "
                "value with a single cell width, so the posterior is flat "
                "across that cell. The file has %d columns but rank %d. Do "
                "not read the unrefined columns as a constraint; see "
                "AmrSampler.column_report.",
                report["degenerate"], len(eta), len(columns), report["rank"])
        if self.physical_map is not None:
            logging.info(
                "AMR: added the physical parameters %s from %s; they are a "
                "linear recombination of the eta columns, not extra "
                "information", ", ".join(self.physical_map.physical_names()),
                self.physical_map.source)

        if self.evidence == "quadrature":
            self._logz = float(_logsumexp(logwt))
        else:
            self._logz = numpy.nan
        self._dlogz = numpy.nan

        if self.level_info is None:
            self.level_info = {}
        self.level_info.update(
            n_cells=int(len(eta)),
            log_total_mass=float(log_total),
            log_evidence=float(self._logz),
            finest_width=float(widths[:, columns].min()),
            refined_columns=list(report["refined"]),
            unrefined_columns=list(report["unrefined"]),
            degenerate_columns=list(report["degenerate"]),
            posterior_rank=int(report["rank"]),
        )
        logging.info(
            "AMR: %d cells, log evidence %s (Riemann sum over the partition; "
            "no error estimate), finest cell width %.6g",
            len(eta), self._logz, float(widths[:, columns].min()))

    # -- checkpointing ---------------------------------------------------------------------------
    def write_results(self, filename):
        """Writes the weighted partition and the evidence to ``filename``."""
        with self.io(filename, "a") as fp:
            fp.write_raw_samples(self.samples)
            fp.write_logevidence(self._logz, self._dlogz)
            fp.write_nature(self)

    def checkpoint(self):
        """Checkpoints the partition to the checkpoint and backup files."""
        if self._samples is None:
            return
        for filename in [self.checkpoint_file, self.backup_file]:
            self.write_results(filename)

    def resume_from_checkpoint(self):
        """Reads a stored partition, which is exact because the run is deterministic."""
        fields = list(self.model.variable_params) + [
            "loglikelihood", "logprior", "logwt", "snr"]
        if self.physical_map is not None:
            fields += self.physical_map.physical_names()
        try:
            with loadfile(self.checkpoint_file, "r") as fp:
                samples = fp.read_raw_samples(fields, raw_samples=True)
                if not len(samples["logwt"]):
                    raise ValueError("empty samples group")
                self._samples = samples
                self._logz = float(fp.attrs["log_evidence"])
                self._dlogz = numpy.nan
            logging.info(
                "AMR: resuming from %s: %d stored cells; the quadrature is "
                "deterministic so this is the same posterior",
                self.checkpoint_file, len(self._samples["logwt"]))
        except Exception as exc:  # a bad checkpoint must fall back to a run
            logging.info(
                "AMR: no usable checkpoint in %s (%s); the partition will be "
                "computed", self.checkpoint_file, exc)

    def finalize(self):
        """Finalizes the samples file and states what the object is."""
        self.checkpoint()
        logging.warning(
            "AMR output: %s. The file holds weighted cell centres from a "
            "deterministic grid quadrature, not independent samples; "
            "dlog_evidence is nan (no evidence error estimate). Do not run "
            "MCMC diagnostics on it.", AMR_NATURE)
