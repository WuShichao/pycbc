# Copyright (C) 2026  Shichao Wu
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
"""Adaptive mesh refinement quadrature of the posterior.

The prior is mapped to the unit cube with the same transform the nested
samplers use, so a box in the cube carries prior mass equal to its volume.
The cube starts as a regular grid and the likelihood is evaluated at each
box centre. A box is bisected along its widest side when either

* its error estimate exceeds ``mass-fraction`` of the evidence. The error of
  a box is how much the midpoint mass of its region changed when its parent
  was split; a box of the starting grid has its own mass as its error. This
  stops refining where the integral has converged.
* it is more than twice as coarse as a face neighbour, and would hold more
  than ``mass-fraction`` of the evidence at that neighbour's likelihood. This
  carries the refinement along a narrow ridge, whose continuation a coarse
  box can miss because its centre falls off the ridge.
* it holds more than ``mass-fraction`` of the evidence and its log
  likelihood differs from a face neighbour's by more than
  ``loglikelihood-step``. The split error alone can miss a ridge narrower
  than the box when the halves' centres fall as far off it as the parent's.

Refinement stops when no box qualifies, the boxes reach ``max-level``
bisections, or ``max-evaluations`` is spent.

Boxes aligned with the cube's axes follow a ridge oblique to them only by
becoming small. With ``pilot-evaluations`` set, a first refinement of that
many evaluations gives the posterior's mean and covariance, and the cube is
partitioned again in the affine frame that whitens them: boxes there are
parallelepipeds in the cube, about one standard deviation wide around the
mean to start with. The frame's boxes cover the cube; their prior mass is
their volume in the cube, and a box whose centre lies outside the cube has
none.

The evidence is the midpoint sum over the boxes; its error estimate is the
summed error of the final boxes. Posterior samples are drawn by choosing
boxes in proportion to their mass and a point uniformly inside each chosen
box, mapped back through the prior; each sample carries the likelihood of
its box centre.

This is a deterministic quadrature, not a stochastic sampler: there is no
chain, so chain diagnostics do not apply. It suits likelihoods of a few
parameters whose marginalisation over the others is done by the model, as
:py:class:`pycbc.inference.models.bstatistic_surrogate.BStatisticSurrogate`
does.
"""

import logging
import time

import numpy
from scipy.special import logsumexp

from pycbc.pool import choose_pool
from pycbc.inference.io import AMRFile

from .base import BaseSampler, setup_output
from .base_cube import setup_calls


class AMRSampler(BaseSampler):
    """Adaptive mesh refinement quadrature over the prior's unit cube.

    Parameters
    ----------
    model : Model
        An instance of a model from ``pycbc.inference.models``.
    initial_divisions : int, optional
        Divisions per parameter of the starting grid. Default 4.
    max_level : int, optional
        Maximum number of bisections of any box. Default 30.
    pilot_evaluations : int, optional
        Evaluations of the pilot refinement that sets the whitened frame.
        Default 0, no pilot: the boxes stay aligned with the cube.
    loglikelihood_step : float, optional
        A box holding more than ``mass_fraction`` of the evidence is refined
        while its log likelihood differs from a face neighbour's by more
        than this. Default 1.
    mass_fraction : float, optional
        A box is refined while its error estimate, or its volume times the
        likelihood of a much finer neighbour, exceeds this fraction of the
        evidence. Default 1e-4.
    max_evaluations : int, optional
        Budget of likelihood evaluations. Default 100000.
    posterior_samples : int, optional
        Number of posterior samples to draw. Default 10000.
    seed : int, optional
        Seed for drawing the posterior samples. Default 0.
    nprocesses : int, optional
        Number of processes for the likelihood evaluations. Default 1.
    use_mpi : bool, optional
        Use MPI for the likelihood evaluations. Default False.
    loglikelihood_function : str, optional
        The model attribute to use as the log likelihood. Default
        ``loglikelihood``.
    """
    name = 'amr'
    _io = AMRFile

    def __init__(self, model, initial_divisions=4, max_level=30,
                 pilot_evaluations=0, loglikelihood_step=1.0,
                 mass_fraction=1e-4, max_evaluations=100000,
                 posterior_samples=10000, seed=0, nprocesses=1,
                 use_mpi=False, loglikelihood_function=None):
        super().__init__(model)
        self.initial_divisions = int(initial_divisions)
        self.max_level = int(max_level)
        self.pilot_evaluations = int(pilot_evaluations)
        self.loglikelihood_step = float(loglikelihood_step)
        self.mass_fraction = float(mass_fraction)
        self.max_evaluations = int(max_evaluations)
        self.posterior_samples = int(posterior_samples)
        self.seed = int(seed)
        if self.initial_divisions < 1 or self.mass_fraction <= 0:
            raise ValueError("initial-divisions must be at least 1 and "
                             "mass-fraction positive")
        self.ndim = len(model.variable_params)
        self._loglikelihood, self._prior_transform = setup_calls(
            model, loglikelihood_function=loglikelihood_function,
            copy_prior=True)
        self.pool = choose_pool(mpi=use_mpi, processes=nprocesses)
        self.nevaluations = 0
        self.history = []
        # every box ever made; the bisected ones keep their two halves in
        # _children, so the final boxes are the leaves of a tree
        self._lower = self._width = self._logl = self._level = None
        self._logerr = self._children = None
        # the box coordinates s map to the cube as origin + matrix @ s; scale
        # is the length of each axis of s in the frame that sizes the boxes
        self._origin = numpy.zeros(self.ndim)
        self._matrix = numpy.eye(self.ndim)
        self._scale = numpy.ones(self.ndim)
        self._logdet = 0.0
        self._logz = self._dlogz = None
        self._samples = {}

    @property
    def io(self):
        return self._io

    @property
    def model_stats(self):
        pass

    @property
    def samples(self):
        """The posterior samples, with equal weights in ``logwt``."""
        return self._samples

    @property
    def niterations(self):
        return len(self.history)

    @property
    def logz(self):
        return self._logz

    @property
    def logz_err(self):
        return self._dlogz

    @property
    def cells(self):
        """The final boxes: lower corner and width in the box coordinates
        ``s``, the log likelihood at the centre, and the number of
        bisections. The point ``s`` of the unit cube is ``frame_origin +
        frame_matrix @ s``."""
        leaf = self._leaves()
        return {'lower': self._lower[leaf], 'width': self._width[leaf],
                'loglikelihood': self._logl[leaf], 'level': self._level[leaf],
                'frame_origin': self._origin, 'frame_matrix': self._matrix}

    def _leaves(self):
        return numpy.nonzero(self._children[:, 0] < 0)[0]

    def _to_cube(self, s):
        return self._origin + s @ self._matrix.T

    def _evaluate(self, index):
        cube = self._to_cube(self._lower[index] + 0.5 * self._width[index])
        inside = numpy.all((cube >= 0) & (cube <= 1), axis=1)
        logl = numpy.full(len(index), -numpy.inf)
        if inside.any():
            points = [self._prior_transform(c) for c in cube[inside]]
            # a large batch in tenths, each logged with an estimate of the rest
            values, step, start = [], max(100, len(points) // 10), time.time()
            for k in range(0, len(points), step):
                values.extend(self.pool.map(self._loglikelihood,
                                            points[k:k + step]))
                if len(points) > step:
                    done, elapsed = len(values), time.time() - start
                    logging.info("AMR: batch %d/%d evaluated, %.0f s, about "
                                 "%.0f s left", done, len(points), elapsed,
                                 elapsed / done * (len(points) - done))
            logl[inside] = values
        self.nevaluations += int(inside.sum())
        self._logl[index] = logl

    def _log_mass(self, index):
        logvol = (numpy.sum(numpy.log(self._width[index]), axis=1)
                  + self._logdet)
        with numpy.errstate(invalid='ignore'):
            return numpy.where(numpy.isfinite(self._logl[index]),
                               self._logl[index] + logvol, -numpy.inf)

    def run(self):
        self._grid()
        if self.pilot_evaluations > 0:
            self._refine(min(self.pilot_evaluations, self.max_evaluations))
            self._whiten()
        leaves, logmass, logz, refine = self._refine(self.max_evaluations)
        self._logz = logz
        logerr = self._logerr[leaves]
        finite = numpy.isfinite(logerr)
        self._dlogz = (float(numpy.exp(logsumexp(logerr[finite]) - logz))
                       if finite.any() else 0.0)
        if len(refine):
            logging.warning("AMR stopped by max-evaluations with %d boxes "
                            "still to refine", len(refine))
        self._draw_posterior(leaves, logmass)

    def _grid(self, evaluate=True):
        """Starts the tree from a regular grid over the box coordinates."""
        n = self.initial_divisions
        axes = [numpy.arange(n) / n] * self.ndim
        self._lower = numpy.array(numpy.meshgrid(*axes, indexing='ij'))
        self._lower = self._lower.reshape(self.ndim, -1).T
        self._width = numpy.full(self._lower.shape, 1.0 / n)
        self._level = numpy.zeros(len(self._lower), dtype=int)
        self._children = numpy.full((len(self._lower), 2), -1)
        self._logl = numpy.full(len(self._lower), numpy.nan)
        self._logerr = numpy.full(len(self._lower), numpy.nan)
        if evaluate:
            self._evaluate_leaves()

    def _evaluate_leaves(self):
        leaves = self._leaves()
        self._evaluate(leaves)
        self._logerr[leaves] = self._log_mass(leaves)

    def _refine(self, budget):
        threshold = numpy.log(self.mass_fraction)
        while True:
            leaves = self._leaves()
            logmass = self._log_mass(leaves)
            logz = logsumexp(logmass)
            self.history.append((len(leaves), self.nevaluations, logz))
            refine = self._logerr[leaves] - logz > threshold
            refine |= self._neighbour_flags(leaves, logmass - logz > threshold,
                                            logz + threshold)
            refine &= self._level[leaves] < self.max_level
            refine = leaves[refine]
            finite = numpy.isfinite(self._logerr[leaves])
            dlogz = (float(numpy.exp(logsumexp(self._logerr[leaves][finite])
                                     - logz)) if finite.any() else 0.0)
            logging.info("AMR: %d boxes, %d evaluations, log Z %.4f, "
                         "dlog Z %.2g, %d boxes to refine", len(leaves),
                         self.nevaluations, logz, dlogz, len(refine))
            room = (budget - self.nevaluations) // 2
            if len(refine) == 0 or room <= 0:
                return leaves, logmass, logz, refine
            if len(refine) > room:
                priority = numpy.maximum(self._logerr[refine],
                                         self._log_mass(refine))
                refine = refine[numpy.argsort(-priority)[:room]]
            self._bisect(refine)

    def _whiten(self, radius=4.0):
        """Partitions the cube again in the frame that whitens the current
        posterior, with boxes about one standard deviation wide out to
        ``radius`` standard deviations from its mean."""
        leaves = self._leaves()
        logmass = self._log_mass(leaves)
        weight = numpy.exp(logmass - logsumexp(logmass))
        centre = self._to_cube(self._lower[leaves]
                               + 0.5 * self._width[leaves])
        width = self._width[leaves] @ numpy.abs(self._matrix.T)
        mean = weight @ centre
        offset = centre - mean
        cov = (offset.T @ (weight[:, None] * offset)
               + numpy.diag(weight @ width ** 2) / 12)
        chol = numpy.linalg.cholesky(cov)
        inverse = numpy.linalg.inv(chol)
        # the bounding box of the cube in the whitened frame
        middle = inverse @ (0.5 - mean)
        half = 0.5 * numpy.abs(inverse).sum(axis=1)
        low, span = middle - half, 2 * half
        self._origin = mean + chol @ low
        self._matrix = chol * span
        self._scale = span
        self._logdet = float(numpy.sum(numpy.log(numpy.diag(chol)))
                             + numpy.sum(numpy.log(span)))
        logging.info("AMR: whitened frame from %d pilot evaluations",
                     self.nevaluations)
        self._grid(evaluate=False)
        # unit boxes near the mean, unless that would take a tenth of the
        # budget
        target = max(1.0, 2 * radius / (0.1 * self.max_evaluations)
                     ** (1.0 / self.ndim))
        while True:
            leaves = self._leaves()
            zlow = low + span * self._lower[leaves]
            zwidth = span * self._width[leaves]
            nearest = numpy.clip(0, zlow, zlow + zwidth)
            near = numpy.linalg.norm(nearest, axis=1) <= radius
            split = leaves[near & (zwidth.max(axis=1) > target)]
            if len(split) == 0:
                break
            self._split(split)
        self._evaluate_leaves()

    def _locate(self, points):
        """The leaf holding each point of the unit cube."""
        n = self.initial_divisions
        cell = numpy.clip((points * n).astype(int), 0, n - 1)
        node = numpy.ravel_multi_index(tuple(cell.T), (n,) * self.ndim)
        rows = numpy.arange(len(points))
        while True:
            inner = self._children[node, 0] >= 0
            if not inner.any():
                return node
            hi = self._children[node[inner], 1]
            axis = numpy.argmax(self._lower[hi] != self._lower[node[inner]],
                                axis=1)
            upper = points[rows[inner], axis] >= self._lower[hi, axis]
            node[inner] = self._children[node[inner], upper.astype(int)]

    def _neighbour_flags(self, leaves, significant, logbound):
        """Leaves to refine because of their face neighbours: those more
        than twice as coarse as a neighbour whose likelihood, times their own
        volume, exceeds ``logbound``, and the significant ones whose log
        likelihood steps by more than ``loglikelihood_step`` to a neighbour
        (with the neighbour, unless it is the finer)."""
        flag = numpy.zeros(len(self._lower), dtype=bool)
        flag_significant = numpy.zeros(len(self._lower), dtype=bool)
        flag_significant[leaves[significant]] = True
        # a neighbour is at most as large as a box of the starting grid
        largest = (-self.ndim * numpy.log(self.initial_divisions)
                   + self._logdet)
        source = leaves[(self._logl[leaves] + largest > logbound)
                        | significant]
        if len(source) == 0:
            return numpy.zeros(len(leaves), dtype=bool)
        centre = self._lower[source] + 0.5 * self._width[source]
        size = (self._width[source] * self._scale).max(axis=1)
        for axis in range(self.ndim):
            for sign in (-1, 1):
                probe = centre.copy()
                probe[:, axis] += sign * 0.5 * self._width[source, axis] * (
                    1 + 1e-9)
                inside = (probe[:, axis] > 0) & (probe[:, axis] < 1)
                found = self._locate(probe[inside])
                logvol = (numpy.sum(numpy.log(self._width[found]), axis=1)
                          + self._logdet)
                origin = source[inside]
                coarseness = (self._width[found] * self._scale).max(axis=1)
                coarse = ((coarseness > 2 * size[inside])
                          & (self._logl[origin] + logvol > logbound))
                flag[found[coarse]] = True
                with numpy.errstate(invalid='ignore'):
                    step = ~(numpy.abs(self._logl[origin] - self._logl[found])
                             <= self.loglikelihood_step)
                # two boxes outside the likelihood's support do not step
                step &= numpy.isfinite(self._logl[origin])
                step &= flag_significant[origin]
                flag[origin[step]] = True
                flag[found[step & (coarseness >= size[inside])]] = True
        return flag[leaves]

    def _split(self, index):
        """Halves boxes along their widest side, without evaluating them;
        returns the indices of the lower and the upper halves."""
        lower, width = self._lower[index], self._width[index]
        axis = numpy.argmax(width * self._scale, axis=1)
        rows = numpy.arange(len(index))
        half = width.copy()
        half[rows, axis] /= 2
        upper = lower.copy()
        upper[rows, axis] += half[rows, axis]
        start = len(self._lower)
        first, second = start + rows, start + len(index) + rows
        self._children[index] = numpy.stack([first, second], axis=1)
        self._lower = numpy.vstack([self._lower, lower, upper])
        self._width = numpy.vstack([self._width, half, half])
        self._level = numpy.concatenate([self._level,
                                         numpy.tile(self._level[index] + 1,
                                                    2)])
        self._logl = numpy.concatenate([self._logl,
                                        numpy.full(2 * len(index), numpy.nan)])
        self._logerr = numpy.concatenate([self._logerr,
                                          numpy.full(2 * len(index),
                                                     numpy.nan)])
        self._children = numpy.vstack([self._children,
                                       numpy.full((2 * len(index), 2), -1)])
        return first, second

    def _bisect(self, index):
        first, second = self._split(index)
        self._evaluate(numpy.concatenate([first, second]))
        # the change in the region's midpoint mass, carried by both halves
        logvol = (numpy.sum(numpy.log(self._width[first]), axis=1)
                  + self._logdet)
        with numpy.errstate(invalid='ignore'):
            parent, a, b = (numpy.where(numpy.isfinite(self._logl[i]),
                                        self._logl[i] + logvol, -numpy.inf)
                            for i in (index, first, second))
        parent = parent + numpy.log(2)
        top = numpy.maximum(numpy.maximum(parent, a), b)
        top = numpy.where(numpy.isfinite(top), top, 0.0)
        change = numpy.abs(numpy.exp(a - top) + numpy.exp(b - top)
                           - numpy.exp(parent - top))
        with numpy.errstate(divide='ignore'):
            logerr = top + numpy.log(change)
        self._logerr[first] = logerr
        self._logerr[second] = logerr

    def _draw_posterior(self, leaves, logmass):
        rng = numpy.random.default_rng(self.seed)
        weights = numpy.exp(logmass - self._logz)
        weights /= weights.sum()
        boxes = leaves[rng.choice(len(weights), size=self.posterior_samples,
                                  p=weights)]
        cube = numpy.empty((len(boxes), self.ndim))
        todo = numpy.arange(len(boxes))
        # a box straddling the cube's surface is drawn from where it is inside
        while len(todo):
            box = boxes[todo]
            cube[todo] = self._to_cube(
                self._lower[box]
                + rng.random((len(box), self.ndim)) * self._width[box])
            todo = todo[~numpy.all((cube[todo] >= 0) & (cube[todo] <= 1),
                                   axis=1)]
        points = numpy.array([self._prior_transform(c) for c in cube])
        samples = {p: points[:, i]
                   for i, p in enumerate(self.model.variable_params)}
        samples['loglikelihood'] = self._logl[boxes]
        samples['logwt'] = numpy.full(len(boxes),
                                      self._logz - numpy.log(len(boxes)))
        self._samples = samples

    def checkpoint(self):
        pass

    def resume_from_checkpoint(self):
        pass

    def set_initial_conditions(self, initial_distribution=None,
                               samples_file=None):
        pass

    def finalize(self):
        logging.info("log Z, dlog Z: %s, %s", self._logz, self._dlogz)
        for fn in [self.checkpoint_file, self.backup_file]:
            with self.io(fn, 'a') as fp:
                fp.write_raw_samples(self._samples)
                fp.write_logevidence(self._logz, self._dlogz)
                fp.write_niterations(self.niterations)
                fp.write_cells(self.cells)

    @classmethod
    def from_config(cls, cp, model, output_file=None, nprocesses=1,
                    use_mpi=False):
        """Loads the sampler from the ``[sampler]`` section.

        * ``name = amr``
        * ``initial-divisions = INT``, ``max-level = INT``,
          ``pilot-evaluations = INT``, ``loglikelihood-step = FLOAT``,
          ``mass-fraction = FLOAT``, ``max-evaluations = INT``,
          ``posterior-samples = INT``, ``seed = INT``: optional, see the
          class docstring.
        * ``loglikelihood-function = STR``: optional, the model attribute to
          use as the log likelihood.
        """
        section = 'sampler'
        assert cp.get(section, 'name') == cls.name, (
            "name in section [sampler] must match mine")
        types = {'initial-divisions': int, 'max-level': int,
                 'pilot-evaluations': int, 'loglikelihood-step': float,
                 'mass-fraction': float, 'max-evaluations': int,
                 'posterior-samples': int, 'seed': int,
                 'loglikelihood-function': str}
        unknown = set(cp.options(section)) - set(types) - {'name'}
        if unknown:
            raise ValueError("unknown option(s) in [sampler] for amr: "
                             + ', '.join(sorted(unknown)))
        kwargs = {key.replace('-', '_'): convert(cp.get(section, key))
                  for key, convert in types.items()
                  if cp.has_option(section, key)}
        obj = cls(model, nprocesses=nprocesses, use_mpi=use_mpi, **kwargs)
        setup_output(obj, output_file, check_nsamples=False)
        return obj
