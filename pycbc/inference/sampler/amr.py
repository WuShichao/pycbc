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
mean to start with. The frame's boxes cover the cube; only their parts
inside the original prior contribute mass, including boxes whose centres
fall outside it.

The evidence is the midpoint sum over the boxes; its error estimate is the
summed error of the final boxes. Posterior points are drawn by choosing
boxes in proportion to their mass. In an affine frame a point is uniform
over the part of its box inside the prior. In a curved frame its importance
weight also accounts for the varying Jacobian and the box's admitted
fraction. Each point carries the likelihood stored for its box;
the usual file reader resamples the raw points with their ``logwt``.

With ``partition-file``, the output of an adaptive cell search, the
quadrature starts from the search's candidate cells instead: boxes in the
metric chart of the search's template family, on the search's lattice, so
the posterior is the prior restricted to those cells. Their union is the
region the search hands on; the evidence is the integral over it.

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
from pycbc.tmpltbank.adaptive_cell_partition import read_partition
from pycbc.tmpltbank.fstatistic_metric import MetricChart

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
    partition_file : str, optional
        An HDF file of an adaptive cell search; its candidate cells are the
        starting boxes, see :func:`read_partition`. Not with
        ``pilot_evaluations``.
    max_level : int, optional
        Maximum number of bisections of any box. Default 30.
    pilot_evaluations : int, optional
        Evaluations of the pilot refinement that sets the whitened frame.
        Default 0, no pilot: the boxes stay aligned with the cube.
    laplace_frame : bool, optional
        Whiten with the Laplace approximation instead of the pilot's
        samples: the likelihood's maximum, found by Newton steps from the
        pilot's best box, and its curvature there, both by finite
        differences, regularised by the prior's own spread. A pilot resolves
        a ridge much thinner than the prior only coarsely and estimates its
        covariance badly; the curvature does not depend on that. Default
        False.
    curvilinear_frame : bool, optional
        After the pilot, partition in the model's ``curvilinear_coordinates``
        instead of the cube: an object with ``names`` (the parameters it
        changes), ``forward`` and ``inverse`` between those parameters and
        coordinates ``y`` (rows of points; ``inverse`` gives NaN where it
        fails), and ``log_jacobian``, ``ln |det d theta / d y|``. A ridge
        that bends in the parameters but is straight in ``y`` is then
        whitened by the Laplace approximation in ``y``. Needs uniform
        priors. Default False.
    shared_region : int, optional
        After the frame is set (or the starting boxes are evaluated, when
        there is no pilot), hand the model this many points drawn
        uniformly from the boxes whose log likelihood is within 20 of the
        largest (15, 10 or 6 when the model cannot hold that region, or
        again at twice the evaluations) through its ``share`` method, and
        evaluate every later batch of boxes through its ``loglikelihoods``:
        computations the points of the region share are then done once
        (see
        :class:`pycbc.inference.models.bstatistic_surrogate.SharedLikelihood`).
        Default 0, off.
    explore_basin : float, optional
        Before the pilot, evaluate a scrambled Sobol sample of the prior
        large enough to land in any region holding this fraction of the
        prior's volume, but for a probability ``explore_miss``; the frame's
        Newton climbs start from its best separate points as well as the
        pilot's boxes. A likelihood with several peaks, of which the pilot's
        refinement finds a lower one first, then has its frame about the
        highest one whose basin is that large. Default 0, off.
    explore_miss : float, optional
        See ``explore_basin``. Default 0.01.
    check_points : int, optional
        Check the quadrature with this many random points once the
        refinement stops: boxes drawn with probabilities half by their mass
        and half by their volume, each evaluated at a uniform point inside
        it, give an unbiased estimate of what the midpoint rule misses and
        a variance that bounds its error (see :meth:`_check`). The evidence
        is corrected by it and its error is ``dlogz``; while that exceeds
        ``check_tolerance``, the boxes the check found wrong are bisected
        and the refinement goes on, half as many evaluations again each
        round, up to ``max_evaluations``. Default 0, off.
    check_tolerance : float, optional
        The error in ``log Z`` the check must reach. Default 0.1.
    whitening_rounds : int, optional
        Pilot refinements, each of ``pilot_evaluations`` more evaluations
        in the frame the previous one whitened. A thin ridge oblique to the
        cube's axes is resolved only coarsely by the first pilot; each round
        aligns the boxes with it better. Default 1.
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
        Number of raw posterior points to draw. Curved-frame points have
        unequal ``logwt`` and must be weighted or resampled. Default 10000.
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
                 use_mpi=False, loglikelihood_function=None,
                 partition_file=None, whitening_rounds=1,
                 laplace_frame=False, curvilinear_frame=False,
                 shared_region=0, explore_basin=0.0, explore_miss=0.01,
                 check_points=0, check_tolerance=0.1):
        super().__init__(model)
        self.initial_divisions = int(initial_divisions)
        self.partition_file = partition_file
        self.max_level = int(max_level)
        self.pilot_evaluations = int(pilot_evaluations)
        self.whitening_rounds = int(whitening_rounds)
        self.laplace_frame = bool(laplace_frame)
        self.curvilinear_frame = bool(curvilinear_frame)
        self.shared_region = int(shared_region)
        self.explore_basin = float(explore_basin)
        self.explore_miss = float(explore_miss)
        if not 0 <= self.explore_basin < 1 or not 0 < self.explore_miss < 1:
            raise ValueError("explore-basin must be in [0, 1) and "
                             "explore-miss in (0, 1)")
        self.check_points = int(check_points)
        self.check_tolerance = float(check_tolerance)
        if self.check_points < 0 or self.check_points == 1 \
                or self.check_tolerance <= 0:
            raise ValueError("check-points must be 0 or at least 2 and "
                             "check-tolerance positive")
        self._explored = None
        self._batched = False
        self._next_share = 0
        self.loglikelihood_step = float(loglikelihood_step)
        self.mass_fraction = float(mass_fraction)
        self.max_evaluations = int(max_evaluations)
        if self.explore_basin > 0 and self._explore_points() >= \
                self.max_evaluations:
            raise ValueError(
                "exploring for a basin of {} of the prior takes {} "
                "evaluations, not fewer than max-evaluations".format(
                    self.explore_basin, self._explore_points()))
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
        # a curvilinear frame, when set, maps s to the cube itself and gives
        # each box its own log Jacobian
        self._curve = None
        self._logz = self._dlogz = None
        self._samples = {}
        self._diagnostics = {}
        # the starting grid: divisions per axis and the cells of it that
        # are roots of the tree, by their index in the grid
        self._divisions = (self.initial_divisions,) * self.ndim
        self._roots = None
        if partition_file is not None:
            if self.pilot_evaluations > 0:
                raise ValueError("a partition sets the frame; it takes no "
                                 "pilot evaluations")
            if self.explore_basin > 0:
                raise ValueError("a partition locates the candidate; it "
                                 "takes no exploration")
            self._start_from_partition(read_partition(partition_file))

    @property
    def io(self):
        return self._io

    @property
    def model_stats(self):
        pass

    @property
    def samples(self):
        """Raw posterior points and their ``logwt`` importance weights."""
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

    def _start_from_partition(self, part):
        """Roots and frame from a search's candidate cells.

        The candidate cells share a width and lie on the search's lattice in
        the metric chart ``eta``; their bounding box is the unit box of the
        coordinates ``s``, divided into the lattice's cells, and the cells
        are the roots. A parameter is ``reference + scale * xi(eta)``, and
        the frame maps it to the unit cube affinely through the prior's
        range, ``u = (p - p(0)) / (p(1) - p(0))``: exactly the prior
        transform for uniform priors, a choice of frame for others, since a
        box's prior mass is its volume in the cube either way.
        """
        names = list(self.model.variable_params)
        if sorted(names) != sorted(part['parameters']):
            raise ValueError("the partition's parameters {} are not the "
                             "model's {}".format(part['parameters'], names))
        order = [part['parameters'].index(n) for n in names]
        keep = part['candidate'].astype(bool)
        if not keep.any():
            raise ValueError("the partition has no candidate cells")
        centres, widths = part['centres'][keep], part['widths'][keep]
        width = widths[0]
        if not numpy.allclose(widths, width, rtol=1e-9, atol=0):
            raise ValueError("the candidate cells must share one width")
        low = (centres - 0.5 * width).min(axis=0)
        high = (centres + 0.5 * width).max(axis=0)
        divisions = numpy.rint((high - low) / width).astype(int)
        span = divisions * width
        cell = numpy.rint((centres - 0.5 * width - low) / width).astype(int)
        self._divisions = tuple(int(n) for n in divisions)
        self._roots = numpy.unique(numpy.ravel_multi_index(
            tuple(cell.T), self._divisions))
        # xi = eta @ chart_matrix, a parameter is reference + scale * xi
        chart = MetricChart(part['metric'])
        chart_matrix = chart.from_chart(numpy.eye(len(width)))
        scales = part['scales']
        p0 = part['reference'] + scales * (low @ chart_matrix)
        dp = (scales[:, None] * chart_matrix.T) * span[None, :]
        p0, dp = p0[order], dp[order]
        bottom = numpy.array(self._prior_transform(numpy.zeros(self.ndim)))
        top = numpy.array(self._prior_transform(numpy.ones(self.ndim)))
        prior_range = top - bottom
        self._origin = (p0 - bottom) / prior_range
        self._matrix = dp / prior_range[:, None]
        self._scale = span
        self._logdet = float(numpy.linalg.slogdet(self._matrix)[1])
        logging.info("AMR: %d candidate cells of a %s lattice from %s",
                     len(self._roots), 'x'.join(map(str, self._divisions)),
                     self.partition_file)

    def _map(self, s):
        """The cube points of box coordinates ``s`` and the log Jacobian
        ``ln |du / ds|`` there."""
        if self._curve is not None:
            return self._curve(s)
        return (self._origin + s @ self._matrix.T,
                numpy.full(len(s), self._logdet))

    def _to_cube(self, s):
        return self._map(s)[0]

    def _unmap(self, cube):
        """The box coordinates of cube points, the inverse of :meth:`_map`
        (``nan`` where a frame's inverse fails)."""
        if self._curve is not None:
            return self._uncurve(cube)
        return numpy.linalg.solve(self._matrix, (cube - self._origin).T).T

    #: points per box at which its part inside the prior, and a curvilinear
    #: frame's Jacobian over it, are averaged
    volume_points = 16

    def _evaluate(self, index):
        """The log likelihood of each box of ``index`` at its centre, or,
        when the centre lies beyond the prior, at the box's point nearest it
        of those :meth:`_volume` finds inside; and the box's mean Jacobian
        over the prior."""
        centre = self._lower[index] + 0.5 * self._width[index]
        cube, logjac = self._map(centre)
        out = ~(numpy.all((cube >= 0) & (cube <= 1), axis=1)
                & numpy.isfinite(logjac))
        mean, inner = self._volume(index)
        # a box straddling the prior's edge: a point inside it, where any
        has = out & numpy.isfinite(mean)
        if has.any():
            cube[has], logjac[has] = self._map(inner[has])
        self._logl[index] = self._cube_loglikelihoods(cube, logjac)
        self._logjac[index] = mean

    def _volume(self, index):
        """``ln`` of each box's mean Jacobian over scrambled Sobol points,
        counting none beyond the prior (``-inf`` when all are), and of the
        points inside the prior the one nearest the box's centre: a box
        straddling the prior's edge holds its part inside only, and a
        curvilinear frame's Jacobian can change many times over a box."""
        if getattr(self, '_volume_offsets', None) is None:
            from scipy.stats import qmc
            self._volume_offsets = qmc.Sobol(
                self.ndim, seed=self.seed + 11).random_base2(
                    int(numpy.log2(self.volume_points)))
        off = self._volume_offsets
        n, k = len(index), len(off)
        s = (self._lower[index][:, None, :] + off[None]
             * self._width[index][:, None, :])
        cube, logjac = self._map(s.reshape(-1, self.ndim))
        inside = (numpy.all((cube >= 0) & (cube <= 1), axis=1)
                  & numpy.isfinite(logjac)).reshape(n, k)
        logjac = numpy.where(inside, logjac.reshape(n, k), -numpy.inf)
        mean = logsumexp(logjac, axis=1) - numpy.log(k)
        far = numpy.where(inside, numpy.sum((off - 0.5) ** 2, axis=1)[None],
                          numpy.inf)
        inner = s[numpy.arange(n), numpy.argmin(far, axis=1)]
        return mean, inner

    def _cube_loglikelihoods(self, cube, logjac=None):
        """The log likelihood at cube points, ``-inf`` outside the cube or
        where a frame's map fails; through the model's batch method once it
        shares a region's computations."""
        inside = numpy.all((cube >= 0) & (cube <= 1), axis=1)
        if logjac is not None:
            inside &= numpy.isfinite(logjac)
        logl = numpy.full(len(cube), -numpy.inf)
        if inside.any():
            points = [self._prior_transform(c) for c in cube[inside]]
            if self._batched:
                names = list(self.model.variable_params)
                logl[inside] = self.model.loglikelihoods(
                    [dict(zip(names, p)) for p in points])
                self.nevaluations += int(inside.sum())
                return logl
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
        return logl

    def _explore(self):
        """A scrambled Sobol sample of the cube, as many points as miss a
        region of the prior holding a fraction ``explore_basin`` of it with
        probability ``explore_miss``: the local maxima the frame climbs
        from include any peak whose basin is that large, wherever the
        pilot's refinement went."""
        from scipy.stats import qmc
        n = self._explore_points()
        cube = qmc.Sobol(self.ndim, seed=self.seed).random_base2(
            int(numpy.log2(n)))
        logl = self._cube_loglikelihoods(cube)
        self._explored = (cube, logl)
        self._diagnostics.update(explore_points=n,
                                 explore_best=float(numpy.max(logl)))
        logging.info("AMR: explored %d points of the prior, largest ln L "
                     "%.3f", n, numpy.max(logl))

    def _explore_points(self):
        """The exploration's size: the power of two at least ``ln(miss) /
        ln(1 - basin)``, which keeps the Sobol sample balanced."""
        need = numpy.log(self.explore_miss) / numpy.log1p(-self.explore_basin)
        return int(2 ** numpy.ceil(numpy.log2(need)))

    def _climb_starts(self, starts, separation=0.05):
        """Cube points to climb from: the pilot's best boxes and the
        exploration's best points, at most ``starts``, in decreasing ln L,
        each farther than ``separation`` (in the cube, per axis) from those
        taken before it."""
        leaves = self._leaves()
        cube = self._to_cube(self._lower[leaves] + 0.5 * self._width[leaves])
        logl = numpy.where(numpy.isfinite(self._logl[leaves]),
                           self._logl[leaves], -numpy.inf)
        if getattr(self, '_explored', None) is not None:
            cube = numpy.vstack([cube, self._explored[0]])
            logl = numpy.concatenate([logl, self._explored[1]])
        taken = []
        for k in numpy.argsort(-logl):
            if not numpy.isfinite(logl[k]) or len(taken) >= starts:
                break
            if all(numpy.abs(cube[k] - cube[j]).max() > separation
                   for j in taken):
                taken.append(k)
        return cube[taken]

    def _log_mass(self, index):
        logvol = (numpy.sum(numpy.log(self._width[index]), axis=1)
                  + self._logjac[index])
        with numpy.errstate(invalid='ignore'):
            return numpy.where(numpy.isfinite(self._logl[index]),
                               self._logl[index] + logvol, -numpy.inf)

    def run(self):
        start = time.time()
        # the model's counts are its own, over all runs: this run's are the
        # change
        before = dict(getattr(self.model, 'shared_statistics', None) or {})
        if 'largest_refused_gb' in before:
            # a largest is this run's own
            self.model.shared_statistics['largest_refused_gb'] = 0.0
            before['largest_refused_gb'] = 0.0
        if self.explore_basin > 0:
            self._explore()
        self._grid()
        if self.pilot_evaluations > 0:
            for _ in range(self.whitening_rounds):
                budget = self.nevaluations + self.pilot_evaluations
                if budget >= self.max_evaluations:
                    break
                self._refine(budget)
                if self.curvilinear_frame:
                    self._curvilinear()
                elif self.laplace_frame:
                    self._whiten(*self._laplace())
                else:
                    self._whiten()
                if self.shared_region and not self._batched:
                    self._share()
        elif self.shared_region and not self._batched:
            # no pilot, as from a search's cells: share the starting boxes'
            # region
            self._share()
        if self.check_points:
            leaves, logmass, logz, refine = self._checked_refinement()
        else:
            leaves, logmass, logz, refine = self._refine(self.max_evaluations)
            self._logz = logz
            logerr = self._logerr[leaves]
            finite = numpy.isfinite(logerr)
            self._dlogz = (float(numpy.exp(logsumexp(logerr[finite]) - logz))
                           if finite.any() else 0.0)
        if len(refine) and not self.check_points:
            logging.warning("AMR stopped by max-evaluations with %d boxes "
                            "still to refine", len(refine))
        self._draw_posterior(leaves, logmass)
        seconds = time.time() - start
        diag = self._diagnostics
        if self.check_points:
            stopped = ('check' if self._dlogz <= self.check_tolerance
                       else 'max-evaluations')
        else:
            stopped = 'max-evaluations' if len(refine) else 'convergence'
        diag.update(stopped_by=stopped,
                    boxes_to_refine=len(refine), boxes=len(leaves),
                    evaluations=self.nevaluations, logz=float(self._logz),
                    dlogz=float(self._dlogz), seconds=seconds,
                    seconds_per_evaluation=seconds / max(self.nevaluations,
                                                         1))
        stats = getattr(self.model, 'shared_statistics', None)
        if stats is not None:
            stats = {k: v - before.get(k, 0) for k, v in stats.items()}
            diag.update(stats)
            used = stats['shared_points'] + stats['single_points']
            diag['shared_fraction'] = (stats['shared_points']
                                       / self.nevaluations)
            diag['batched_fraction'] = used / self.nevaluations
        logging.info("AMR diagnostics: %s", ', '.join(
            '%s %s' % (k, ('%.4g' % v) if isinstance(v, float) else v)
            for k, v in diag.items()))

    @property
    def diagnostics(self):
        """How the run went, for judging its result afterwards: what
        stopped it and how many boxes were still to refine, the
        evaluations and seconds, the frame's peak, and, for a model that
        shares computations, the fraction of evaluations from shared
        regions and the regions set up or refused."""
        return dict(self._diagnostics)

    def _grid(self, evaluate=True):
        """Starts the tree from a regular grid over the box coordinates, or
        from the cells of it that are roots."""
        dims = numpy.array(self._divisions)
        if self._roots is None:
            cells = numpy.indices(self._divisions).reshape(self.ndim, -1).T
            self._root_index = numpy.arange(len(cells))
        else:
            cells = numpy.array(numpy.unravel_index(self._roots,
                                                    self._divisions)).T
            self._root_index = self._roots
        self._lower = cells / dims
        self._width = numpy.repeat((1.0 / dims)[None, :], len(cells), axis=0)
        self._level = numpy.zeros(len(self._lower), dtype=int)
        self._children = numpy.full((len(self._lower), 2), -1)
        self._logl = numpy.full(len(self._lower), numpy.nan)
        self._logerr = numpy.full(len(self._lower), numpy.nan)
        self._logjac = numpy.full(len(self._lower), numpy.nan)
        if evaluate:
            self._evaluate_leaves()

    def _evaluate_leaves(self):
        leaves = self._leaves()
        self._evaluate(leaves)
        self._logerr[leaves] = self._log_mass(leaves)

    def _refine(self, budget):
        threshold = numpy.log(self.mass_fraction)
        while True:
            if (self.shared_region and not self._batched
                    and self._next_share
                    and self.nevaluations >= self._next_share):
                self._share()
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

    def _share(self, spans=(20.0, 15.0, 10.0, 6.0)):
        """Hand the model the region the refinement still works in: points
        drawn uniformly from the boxes whose log likelihood is within a
        span of the largest, the span narrowed while the model cannot hold
        the region's shared computation."""
        leaves = self._leaves()
        logl = self._logl[leaves]
        finite = numpy.isfinite(logl)
        top = logl[finite].max()
        for span in spans:
            kept = leaves[finite & (logl >= top - span)]
            try:
                self._share_boxes(kept, numpy.ones(len(kept)))
                return
            except MemoryError as err:
                logging.info("AMR: shared region within %.0f of the largest "
                             "log likelihood too large (%s)", span, err)
        # the posterior narrows as the refinement goes on: try again later
        self._next_share = 2 * max(self.nevaluations, 1)
        logging.info("AMR: no shared region fits yet; trying again at %d "
                     "evaluations", self._next_share)

    def _share_boxes(self, leaves, weights):
        rng = numpy.random.default_rng(self.seed + 1)
        boxes = leaves[rng.choice(len(leaves), size=self.shared_region,
                                  p=weights / weights.sum())]
        cube = self._to_cube(self._lower[boxes] + rng.random(
            (len(boxes), self.ndim)) * self._width[boxes])
        ok = numpy.all((cube >= 0) & (cube <= 1), axis=1)
        names = list(self.model.variable_params)
        region = [dict(zip(names, self._prior_transform(c)))
                  for c in cube[ok]]
        full = []
        for p in region:
            q = dict(self.model.static_params)
            q.update(p)
            full.append(q)
        tick = time.time()
        self.model.share(full)
        self._batched = True
        logging.info("AMR: shared region of %d points set up in %.0f s",
                     len(full), time.time() - tick)

    def _curvilinear(self, radius=4.0, prior_draws=20000, iterations=8,
                     starts=5):
        """Partition in the model's curvilinear coordinates ``y``, whitened
        by the Laplace approximation about the likelihood's maximum there:
        the highest of Newton climbs from the pilot's ``starts`` best boxes,
        each step halved while it lowers ln L. Coordinates with a ``fit``
        method take their directions from the prior's draws first.

        A box of ``s`` maps to ``y = mean + chol (low + span s)``, then to
        the parameters and, through the uniform priors' ranges, to the cube;
        its mass is its volume times ``|det chol| prod(span) |d theta / d
        y| / prod(range)`` at its centre."""
        coords = self.model.curvilinear_coordinates
        names = list(self.model.variable_params)
        idx = [names.index(n) for n in coords.names]
        bottom = numpy.array(self._prior_transform(numpy.zeros(self.ndim)))
        top = numpy.array(self._prior_transform(numpy.ones(self.ndim)))
        prior_range = top - bottom

        def to_y(theta):
            y = numpy.array(theta, dtype=float, copy=True)
            y[:, idx] = coords.forward(theta[:, idx])
            return y

        def from_y(y):
            theta = numpy.array(y, dtype=float, copy=True)
            theta[:, idx] = coords.inverse(y[:, idx])
            return theta
        rng = numpy.random.default_rng(self.seed)
        draws = bottom + rng.random((prior_draws, self.ndim)) * prior_range
        if hasattr(coords, 'fit'):
            # coordinates that take their directions from the prior
            coords.fit(draws[:, idx])
        image = to_y(draws)
        image = image[numpy.all(numpy.isfinite(image), axis=1)]
        prior_cov = numpy.cov(image.T)
        unit = numpy.sqrt(numpy.diag(prior_cov))
        prior_precision = numpy.linalg.inv(prior_cov / numpy.outer(unit,
                                                                   unit))

        def climb(y0):
            """Newton steps of ln L in ``z``, ``y = y0 + unit z``, each
            halved while it lowers ln L: the end point, the precision
            there and ln L."""
            def logl_zs(z):
                """ln L at the rows y = y0 + unit z, the map inverted and
                the likelihood evaluated for all rows together."""
                theta = from_y(y0 + unit * numpy.atleast_2d(z))
                u = (theta - bottom) / prior_range
                u[~numpy.all(numpy.isfinite(u), axis=1)] = -1.0
                return self._cube_loglikelihoods(u)

            def logl_z(z):
                return float(logl_zs(z)[0])
            x = numpy.zeros(self.ndim)
            value = logl_z(x)
            step = numpy.full(self.ndim, 1e-3)
            precision = prior_precision
            for _ in range(iterations):
                grad, hess = _finite_differences(logl_zs, x, step)
                values, vectors = numpy.linalg.eigh(-0.5 * (hess + hess.T))
                precision = (vectors * numpy.maximum(values, 0)) @ vectors.T \
                    + prior_precision
                move = numpy.linalg.solve(precision, grad)
                size = numpy.sqrt(move @ precision @ move)
                if size > 3:
                    move *= 3 / size
                for _ in range(6):
                    trial = logl_z(x + move)
                    if trial >= value:
                        break
                    move = 0.5 * move
                else:
                    break
                x, value = x + move, trial
                step = numpy.clip(0.5 / numpy.sqrt(numpy.diag(precision)),
                                  1e-7, 1e-1)
                if size < 0.05:
                    break
            return y0 + unit * x, precision, value
        # from the pilot's and the exploration's best, separate points, the
        # highest maximum
        best = None
        for point in self._climb_starts(starts):
            start = numpy.array(self._prior_transform(point))
            y0 = to_y(start[None])[0]
            if not numpy.all(numpy.isfinite(y0)):
                continue
            found = climb(y0)
            if best is None or found[2] > best[2]:
                best = found
        mean, precision, peak = best
        logging.info("AMR: frame about ln L %.3f", peak)
        self._diagnostics.update(frame_peak=float(peak),
                                 frame_starts=int(starts),
                                 frame_evaluations=self.nevaluations)
        chol = numpy.linalg.cholesky(numpy.linalg.inv(precision)) \
            * unit[:, None]
        inverse = numpy.linalg.inv(chol)
        whitened = (image - mean) @ inverse.T
        low = whitened.min(axis=0)
        span = whitened.max(axis=0) - low
        low, span = low - 0.01 * span, 1.02 * span
        logdet = float(numpy.linalg.slogdet(chol)[1] + numpy.sum(
            numpy.log(span)) - numpy.sum(numpy.log(prior_range)))

        def curve(s):
            y = mean + (low + span * s) @ chol.T
            theta = from_y(y)
            cube = (theta - bottom) / prior_range
            logjac = logdet + coords.log_jacobian(theta[:, idx])
            bad = ~numpy.all(numpy.isfinite(theta), axis=1)
            cube[bad] = numpy.nan
            logjac = numpy.where(bad, -numpy.inf, logjac)
            return cube, logjac
        def uncurve(cube):
            with numpy.errstate(invalid='ignore'):
                y = to_y(bottom + cube * prior_range)
            return ((y - mean) @ inverse.T - low) / span
        self._curve = curve
        self._uncurve = uncurve
        self._scale = span
        logging.info("AMR: curvilinear frame after %d evaluations",
                     self.nevaluations)
        self._start_frame(low, span, radius)

    def _laplace(self, iterations=8, starts=5):
        """Mean and covariance, in the cube, of the Laplace approximation
        about the likelihood's highest maximum the Newton climbs from the
        pilot's (and the exploration's) best separate points reach, each
        step halved while it lowers ln L."""
        # a uniform prior over the cube has variance 1 / 12 per axis
        prior = 12 * numpy.eye(self.ndim)

        def logl_us(u):
            return self._cube_loglikelihoods(numpy.atleast_2d(u))

        def logl_at(u):
            return float(logl_us(u)[0])

        def climb(x):
            value = logl_at(x)
            step = numpy.full(self.ndim, 1e-3)
            precision = prior
            for _ in range(iterations):
                grad, hess = _finite_differences(logl_us, x, step)
                values, vectors = numpy.linalg.eigh(-0.5 * (hess + hess.T))
                precision = (vectors * numpy.maximum(values, 0)) @ vectors.T \
                    + prior
                move = numpy.linalg.solve(precision, grad)
                # at most three standard deviations, and inside the cube
                size = numpy.sqrt(move @ precision @ move)
                if size > 3:
                    move *= 3 / size
                for _ in range(6):
                    trial_x = numpy.clip(x + move, 1e-9, 1 - 1e-9)
                    trial = logl_at(trial_x)
                    if trial >= value:
                        break
                    move = 0.5 * move
                else:
                    break
                x, value = trial_x, trial
                step = numpy.clip(0.5 / numpy.sqrt(numpy.diag(precision)),
                                  1e-7, 1e-2)
                if size < 0.05:
                    break
            return x, precision, value
        best = None
        for point in self._climb_starts(starts):
            found = climb(numpy.array(point, dtype=float))
            if best is None or found[2] > best[2]:
                best = found
        x, precision, peak = best
        self._diagnostics.update(frame_peak=float(peak),
                                 frame_starts=int(starts),
                                 frame_evaluations=self.nevaluations)
        logging.info("AMR: Laplace frame about ln L %.3f after %d "
                     "evaluations", peak, self.nevaluations)
        return x, numpy.linalg.inv(precision)

    def _whiten(self, mean=None, cov=None, radius=4.0):
        """Partitions the cube again in the frame that whitens the current
        posterior, with boxes about one standard deviation wide out to
        ``radius`` standard deviations from its mean: the pilot's mean and
        covariance unless given."""
        if mean is None:
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
        self._start_frame(low, span, radius)

    def _start_frame(self, low, span, radius):
        """A new tree over the frame whose box coordinates ``s`` are
        ``low + span s`` in whitened units: unit boxes out to ``radius``
        of the origin, unless that would take a tenth of the budget."""
        self._grid(evaluate=False)
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
        """The leaf holding each point of the box coordinates, -1 where no
        root does."""
        dims = numpy.array(self._divisions)
        cell = numpy.clip((points * dims).astype(int), 0, dims - 1)
        index = numpy.ravel_multi_index(tuple(cell.T), self._divisions)
        node = numpy.searchsorted(self._root_index, index)
        node = numpy.minimum(node, len(self._root_index) - 1)
        missing = self._root_index[node] != index
        node[missing] = -1
        rows = numpy.nonzero(~missing)[0]
        located = node
        node = node[rows]
        while True:
            inner = self._children[node, 0] >= 0
            if not inner.any():
                located[rows] = node
                return located
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
        jac = self._logjac[leaves]
        largest = (-numpy.sum(numpy.log(self._divisions))
                   + (numpy.nanmax(jac) if numpy.isfinite(jac).any()
                      else self._logdet))
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
                # beyond the roots there is no neighbour
                inside[inside] = found >= 0
                found = found[found >= 0]
                logvol = (numpy.sum(numpy.log(self._width[found]), axis=1)
                          + self._logjac[found])
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
        self._logjac = numpy.concatenate([self._logjac,
                                          numpy.full(2 * len(index),
                                                     numpy.nan)])
        self._children = numpy.vstack([self._children,
                                       numpy.full((2 * len(index), 2), -1)])
        return first, second

    def _bisect(self, index):
        first, second = self._split(index)
        self._evaluate(numpy.concatenate([first, second]))
        # the change in the region's midpoint mass, carried by both halves
        parent, a, b = (self._log_mass(i) for i in (index, first, second))
        top = numpy.maximum(numpy.maximum(parent, a), b)
        top = numpy.where(numpy.isfinite(top), top, 0.0)
        change = numpy.abs(numpy.exp(a - top) + numpy.exp(b - top)
                           - numpy.exp(parent - top))
        with numpy.errstate(divide='ignore'):
            logerr = top + numpy.log(change)
        self._logerr[first] = logerr
        self._logerr[second] = logerr

    def _check(self, leaves, logmass, logz, n):
        """A random check of the quadrature over ``leaves``: importance
        sampling of the prior cube from a mixture, half the prior itself and
        half the tree's posterior (a leaf by its mass, a uniform point in
        its box coordinates).

        With ``q`` the mixture's density in the cube, ``r = L / (Z q)`` at
        ``n`` independent draws has a mean ``1 + D`` unbiased for
        ``Z_true / Z``, whatever the likelihood, the frame or the tree's
        reach; ``V``, the variance of ``r`` over ``n - 1``, is unbiased for
        ``D``'s variance, and ``r <= 2 L / Z`` bounds every draw's weight.

        Returns ``D``, ``V``, ``U``, the part of ``1 + D`` from draws no leaf
        holds, the leaves the draws fell in and, for each, the mass the
        draw gives its box less the box's midpoint mass, over ``Z``."""
        rng = numpy.random.default_rng(self.seed + 7 + len(self.history))
        share = numpy.exp(logmass - logz)
        share /= share.sum()
        ntree = int(rng.binomial(n, 0.5))
        pick = leaves[rng.choice(len(leaves), size=ntree, p=share)]
        prior = rng.random((n - ntree, self.ndim))
        s = numpy.vstack([self._lower[pick] + rng.random((ntree, self.ndim))
                          * self._width[pick], self._unmap(prior)])
        cube, logjac = self._map(s)
        # the prior's draws at their own points, whatever the frame's round
        # trip
        cube[ntree:] = prior
        within = numpy.all(numpy.isfinite(s) & (s >= 0) & (s <= 1), axis=1)
        box = numpy.full(n, -1)
        box[within] = self._locate(s[within])
        box[:ntree] = pick
        held = box >= 0
        # the tree's density at each draw, in the cube
        mass = numpy.full(len(self._lower), -numpy.inf)
        mass[leaves] = logmass - logz
        logq_tree = numpy.full(n, -numpy.inf)
        with numpy.errstate(invalid='ignore'):
            logq_tree[held] = (mass[box[held]]
                               - numpy.sum(numpy.log(self._width[box[held]]),
                                           axis=1) - logjac[held])
        logq = numpy.logaddexp(numpy.log(0.5), numpy.log(0.5) + numpy.where(
            numpy.isfinite(logq_tree), logq_tree, -numpy.inf))
        logl = self._cube_loglikelihoods(cube, numpy.where(
            numpy.arange(n) < ntree, logjac, 0.0))
        r = numpy.exp(numpy.where(numpy.isfinite(logl), logl - logz - logq,
                                  -numpy.inf))
        D = float(numpy.mean(r)) - 1
        V = float(numpy.var(r, ddof=1) / n)
        U = float(numpy.sum(r[~held]) / n)
        with numpy.errstate(invalid='ignore'):
            there = numpy.where(numpy.isfinite(logl[held]), logl[held]
                                + numpy.sum(numpy.log(self._width[box[held]]),
                                            axis=1) + logjac[held] - logz,
                                -numpy.inf)
        d = numpy.exp(there) - numpy.exp(mass[box[held]])
        return D, V, U, box[held], d

    def _checked_refinement(self):
        """Refinement in rounds, each ended by :meth:`_check`, until the
        checked error in ``log Z`` is within ``check_tolerance`` or the
        budget is spent; ``log Z`` is corrected by the last check."""
        n = self.check_points
        budget = min(self.max_evaluations,
                     max(self.nevaluations + 2 * n, self.max_evaluations // 4))
        rounds = []
        while True:
            leaves, logmass, logz, refine = self._refine(budget)
            D, V, U, box, d = self._check(leaves, logmass, logz, n)
            # ln(1 + D) with its error; a correction below -1 is noise of a
            # check far too small
            ratio = max(1 + D, 1e-3)
            err = numpy.sqrt(V) / ratio
            rounds.append((self.nevaluations, float(logz), D, float(err), U))
            logging.info("AMR check: %d evaluations, log Z %.4f, "
                         "correction %+.4f, error %.3f, beyond the tree "
                         "%.3g", self.nevaluations, logz, numpy.log(ratio),
                         err, U)
            if err <= self.check_tolerance or \
                    self.nevaluations + 2 * n >= self.max_evaluations:
                break
            # the boxes whose check point disagreed with their midpoint by
            # more than the mass a box may hold
            wrong = numpy.unique(box[numpy.abs(d) > self.mass_fraction])
            wrong = wrong[self._level[wrong] < self.max_level]
            if len(wrong):
                self._bisect(wrong)
            budget = min(self.max_evaluations,
                         self.nevaluations + max(self.nevaluations // 2, 2 * n))
        self._logz = float(logz + numpy.log(ratio))
        self._dlogz = float(err)
        self._diagnostics.update(check_rounds=len(rounds),
                                 check_uncovered=float(rounds[-1][4]),
                                 check_correction=float(numpy.log(ratio)),
                                 midpoint_logz=float(logz))
        if err > self.check_tolerance:
            logging.warning("AMR check error %.3f above the tolerance %.3f "
                            "at the evaluation budget", err,
                            self.check_tolerance)
        return leaves, logmass, logz, refine

    def _draw_posterior(self, leaves, logmass):
        if self._curve is not None and self.posterior_samples > 0:
            return self._draw_curved_posterior(leaves, logmass)
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

    def _draw_curved_posterior(self, leaves, logmass):
        """Weighted points from the represented curved-frame boxes.

        A box's stored mean Jacobian includes points outside the prior.
        Uniform rejection draws inside it therefore need both their own
        Jacobian and its admitted fraction ``f``. For ``k`` requested points
        in a box, draw one more accepted point: the total number of trials
        ``T`` then gives the unbiased estimate ``k / (T - 1)`` of ``f``.
        The trial count is independent of the accepted positions. The
        extra point is not returned, and no likelihood is evaluated here.
        Failed inverse trials are counted; neither this correction nor the
        weight concentration diagnostic proves that the map covers the prior.
        """
        rng = numpy.random.default_rng(self.seed)
        tree_logz = float(logsumexp(logmass))
        weights = numpy.exp(logmass - tree_logz)
        weights /= weights.sum()
        boxes = leaves[rng.choice(len(weights), size=self.posterior_samples,
                                  p=weights)]
        occupied, inverse, counts = numpy.unique(
            boxes, return_inverse=True, return_counts=True)
        trial_boxes = numpy.concatenate([boxes, occupied])
        group = numpy.concatenate([inverse, numpy.arange(len(occupied))])
        trials = numpy.zeros(len(occupied), dtype=numpy.int64)
        cube = numpy.empty((len(trial_boxes), self.ndim))
        logjac = numpy.empty(len(trial_boxes))
        todo = numpy.arange(len(trial_boxes))
        failed_inverse = 0
        while len(todo):
            chosen = trial_boxes[todo]
            mapped, jacobian = self._map(
                self._lower[chosen]
                + rng.random((len(chosen), self.ndim)) * self._width[chosen])
            if (mapped.shape != (len(todo), self.ndim)
                    or jacobian.shape != (len(todo),)):
                raise ValueError(
                    "the frame returned incompatible posterior points")
            finite = numpy.all(numpy.isfinite(mapped), axis=1)
            failed_inverse += int((~finite).sum())
            inside = finite & numpy.all((mapped >= 0) & (mapped <= 1), axis=1)
            if numpy.any(inside & ~numpy.isfinite(jacobian)):
                raise ArithmeticError(
                    "posterior Jacobian unresolved inside the prior")
            trials += numpy.bincount(group[todo], minlength=len(occupied))
            accepted = todo[inside]
            cube[accepted], logjac[accepted] = mapped[inside], jacobian[inside]
            todo = todo[~inside]
        number = len(boxes)
        log_fraction = numpy.log(counts) - numpy.log(trials - 1)
        correction = (logjac[:number] - self._logjac[boxes]
                      + log_fraction[inverse])
        if not numpy.all(numpy.isfinite(correction)):
            raise ArithmeticError("posterior geometry weights are unresolved")
        norm = float(logsumexp(correction))
        points = numpy.array([self._prior_transform(c) for c in cube[:number]])
        samples = {p: points[:, i]
                   for i, p in enumerate(self.model.variable_params)}
        samples['loglikelihood'] = self._logl[boxes]
        samples['log_geometry_weight'] = correction
        # The file convention scales posterior weights by the reported
        # evidence. Keep the independent geometry estimate in diagnostics.
        samples['logwt'] = self._logz + correction - norm
        self._samples = samples
        self._diagnostics.update(
            posterior_weighting='curved prior geometry',
            posterior_extra_points=int(len(occupied)),
            posterior_trials=int(trials.sum()),
            posterior_inverse_failures=failed_inverse,
            posterior_geometry_logz=tree_logz + norm - numpy.log(number),
            posterior_effective_samples=float(1 / numpy.sum(
                numpy.exp(2 * (correction - norm)))))

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
                fp.write_diagnostics(self.diagnostics)

    @classmethod
    def from_config(cls, cp, model, output_file=None, nprocesses=1,
                    use_mpi=False):
        """Loads the sampler from the ``[sampler]`` section.

        * ``name = amr``
        * ``initial-divisions = INT``, ``max-level = INT``,
          ``pilot-evaluations = INT``, ``whitening-rounds = INT``,
          ``laplace-frame = BOOL``,
          ``loglikelihood-step = FLOAT``,
          ``mass-fraction = FLOAT``, ``max-evaluations = INT``,
          ``posterior-samples = INT``, ``seed = INT``: optional, see the
          class docstring.
        * ``loglikelihood-function = STR``: optional, the model attribute to
          use as the log likelihood.
        * ``partition-file = PATH``: optional, an adaptive cell search's
          output; its candidate cells are the starting boxes.
        """
        section = 'sampler'
        assert cp.get(section, 'name') == cls.name, (
            "name in section [sampler] must match mine")
        types = {'initial-divisions': int, 'max-level': int,
                 'whitening-rounds': int, 'laplace-frame': _boolean,
                 'curvilinear-frame': _boolean, 'shared-region': int,
                 'partition-file': str,
                 'pilot-evaluations': int, 'loglikelihood-step': float,
                 'mass-fraction': float, 'max-evaluations': int,
                 'posterior-samples': int, 'seed': int,
                 'loglikelihood-function': str,
                 'explore-basin': float, 'explore-miss': float,
                 'check-points': int, 'check-tolerance': float}
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


def _boolean(value):
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def _finite_differences(f, x, step, shrink=6):
    """Gradient and Hessian of ``f`` at ``x``: central differences along
    the axes and forward ones for the mixed terms. ``f`` takes the rows of
    an array of points and returns their values, so that all the points
    are evaluated together.

    Next to the edge of the support (``f`` not finite on one side of an
    axis, as at a prior bound) the differences along that axis are one-
    sided, from the points at one and two steps on the other side, and the
    mixed terms go to the sides that are inside; an axis with neither side
    inside has its step halved, up to ``shrink`` times. A peak on a prior
    bound then has the curvature it has inside rather than none."""
    n = len(x)
    step = numpy.array(step, dtype=float)
    eye = numpy.diag(step)
    f0 = float(f(x[None])[0])
    plus, minus = numpy.full(n, numpy.nan), numpy.full(n, numpy.nan)
    todo = numpy.arange(n)
    for _ in range(shrink + 1):
        eye = numpy.diag(step)
        values = numpy.asarray(f(numpy.array(
            [x + eye[i] for i in todo] + [x - eye[i] for i in todo])),
            dtype=float)
        plus[todo], minus[todo] = values[:len(todo)], values[len(todo):]
        lost = ~numpy.isfinite(plus[todo]) & ~numpy.isfinite(minus[todo])
        todo = todo[lost]
        if not len(todo):
            break
        step[todo] /= 2
    # the side each axis is differenced on: 0 both, +1 or -1 one
    side = numpy.where(numpy.isfinite(plus) & numpy.isfinite(minus), 0,
                       numpy.where(numpy.isfinite(plus), 1, -1))
    single = numpy.nonzero(side != 0)[0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    sign = numpy.where(side == 0, 1, side)
    eye = numpy.diag(step * sign)
    extra = [x + 2 * eye[i] for i in single] \
        + [x + eye[i] + eye[j] for i, j in pairs]
    values = numpy.asarray(f(numpy.array(extra)), dtype=float) \
        if extra else numpy.empty(0)
    two = dict(zip(single, values[:len(single)]))
    near = numpy.where(side >= 0, plus, minus)
    grad = numpy.empty(n)
    hess = numpy.zeros((n, n))
    for i in range(n):
        if side[i] == 0:
            grad[i] = (plus[i] - minus[i]) / (2 * step[i])
            hess[i, i] = (plus[i] - 2 * f0 + minus[i]) / step[i] ** 2
        else:
            # one-sided, second order: f(x), f(x + h), f(x + 2h)
            h = side[i] * step[i]
            grad[i] = (-3 * f0 + 4 * near[i] - two[i]) / (2 * h)
            hess[i, i] = (f0 - 2 * near[i] + two[i]) / h ** 2
    for k, (i, j) in enumerate(pairs):
        hess[i, j] = hess[j, i] = (
            values[len(single) + k] - near[i] - near[j] + f0) / (
            sign[i] * step[i] * sign[j] * step[j])
    grad[~numpy.isfinite(grad)] = 0.0
    hess[~numpy.isfinite(hess)] = 0.0
    return grad, hess

