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


class _ShellCertificate(object):
    """The fractional-moment certificate over shells of the signal's
    overlap (lisa_tests_with_cc, notes/2026/11/
    2026-11-19-fractional-moment-coverage-bound.md; the shells from the
    Gemini review of 2026-11-19).

    For unit templates the marginal likelihood ratio is at most
    ``exp(z^2 / 2)``, ``z^2 = 2F`` noncentral chi^2 with ``k`` degrees of
    freedom and noncentrality ``nu^2``, ``nu`` the true signal's overlap with
    the template. The shells ``S_g = {x : tau_{g-1} < nu_true(x) <= tau_g}``
    depend on the signal alone, not on the noise. For a fixed set cut into
    cells, ``t -> t^beta`` (``beta = 1 - eps`` in (0, 1)) is subadditive,
    ``E exp(beta z^2 / 2) = eps^(-k/2) exp(beta nu^2 / (2 eps))``, and by
    Markov, with probability at least ``1 - delta / G`` for each shell,

        ln Z_{S_g} <= (1/beta) [ln(G/delta) + beta ln pi(S_g) + eps ln N(S_g)
                                - (k/2) ln eps + beta tau_g^2 / (2 eps)],

    ``pi`` the prior mass and ``N`` the cells. Where the data's overlaps
    bound the true ones (``nu_hat >= nu_true``, the bound's own event),
    ``S_g`` lies in ``U_{g-1}``, the leaves with ``nu_hat > tau_{g-1}``,
    whatever the tree: ``pi(U_{g-1})`` and ``N(U_{g-1})`` stand for ``pi(S_g)``
    and ``N(S_g)``, and the minimum over ``beta`` is taken after (the bound
    grows with both). The leaves with ``nu_hat <= tau_g`` lie in the union of
    the first ``g`` shells; the largest ``g`` whose shells' bounds sum within
    the budget certifies them. No state: recomputed from the leaves each
    round, so neither the order boxes arrive in nor the tree's shape enters.
    Each cell stands for its continuum (its sup ignored; held in the 1-d and
    2-d toys).
    """

    def __init__(self, delta, k, shells, betas=None):
        self.shells = numpy.sort(numpy.asarray(shells, dtype=float))
        self.k = float(k)
        self.log_inv_delta = float(numpy.log(len(self.shells) / float(delta)))
        self.betas = (numpy.linspace(0.02, 0.995, 400) if betas is None
                      else numpy.asarray(betas, dtype=float))

    def bounds(self, nu, logmass, lncells):
        """``ln`` of the most evidence each shell can hold, from the leaves'
        overlaps, prior masses (upper bounds) and cell counts."""
        b = self.betas
        e = 1. - b
        lower = numpy.concatenate([[-numpy.inf], self.shells[:-1]])
        out = numpy.full(len(self.shells), -numpy.inf)
        for g, (lo, tau) in enumerate(zip(lower, self.shells)):
            u = nu > lo
            if not u.any():
                continue
            lp = min(float(logsumexp(logmass[u])), 0.)
            ln = float(logsumexp(lncells[u]))
            f = (self.log_inv_delta + b * lp + e * ln - 0.5 * self.k * numpy.log(e)
                 + b * tau ** 2 / (2. * e)) / b
            out[g] = float(f.min())
        return out

    def threshold(self, nu, logmass, lncells, budget):
        """The largest shell threshold whose shells' bounds sum within
        ``exp(budget)`` (``-inf`` when not even the first fits), the ``ln``
        of that sum, and the number of shells used."""
        cum = numpy.logaddexp.accumulate(self.bounds(nu, logmass, lncells))
        fit = cum <= budget
        g = len(fit) if fit.all() else int(numpy.argmin(fit))
        if not g:
            return -numpy.inf, -numpy.inf, 0
        return float(self.shells[g - 1]), float(cum[g - 1]), g


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
    frame_starts : int, optional
        The number of Newton climbs the Laplace or curvilinear frame takes
        its maximum from (the pilot's best separate points and the start
        cube). Default 5.
    climb_resolve : float or None, optional
        The largest change of ln L a finite-difference step of those climbs
        may make: an axis whose step changes ln L by more has its step cut
        by eight, up to six times, before the gradient and Hessian are
        taken; where the Hessian has no curvature the Newton step is then
        held within eight such steps. A likelihood far narrower than the
        prior (the Yorsh sobhb6 chirp mass, 1.5e-4 of the frame's prior
        width) was otherwise differenced over seven of its widths, the
        Hessian came out convex, and the step went to the prior's edge.
        None keeps the fixed steps. Default 2.
    climb_precision : array or callable, optional
        The curvilinear frame's climbs: the likelihood's expected precision
        (minus its Hessian) in the frame's coordinates ``y`` (the model's
        parameter order, the curvilinear ones in place of theirs), or a
        callable of the start's ``y`` that returns it. A model whose
        coordinates are the rms phase change (radians) has it as about
        ``2F / 2`` on each phase direction, ``2F`` the start's: the climb's
        first steps are then half its standard deviations, and along a
        direction where the differences find no curvature (a start on the
        convex tail of a peak) the Newton step takes this curvature instead
        of the prior's. The climb's scale is then the measurement's, not the
        prior's: the same for a peak 1e-4 or 1e-1 of the prior wide.
    climb_hessian : {'differences', 'metric'}, optional
        'metric': Fisher scoring, the climbs' curvature the expected
        information ``climb_precision`` gives (a callable of ``y``, or a
        matrix), positive definite, only the gradient of ln L differenced.
        Its dimension is the frame's, whatever the parameters stand for: the
        information of several overlapping sources, with their cross terms,
        is a larger matrix of the same kind. Default 'differences'.
    share_frame : bool, optional
        With ``shared_region`` and a curvilinear frame, hand the model the
        frame's whitened ball before its first boxes are evaluated (see
        :meth:`_share_ball`). Default False.
    frame_moments : bool, optional
        With a curvilinear frame and ``whitening_rounds`` above one, every
        frame after the first is set by the moments of the previous one's
        boxes in ``y`` (their posterior mean and covariance, each box's own
        spread included) instead of a Laplace approximation. Default False.
    edge_points : int, optional
        A box straddling the prior's edge is valued by the mean likelihood
        at up to this many of its inner points (default 1: the inner point
        nearest its centre).
    share_frame_radius : float, optional
        The shared ball's radius over the frame's (default 1.3).
    layout_fraction : float, optional
        A frame's first boxes (unit boxes in its whitened units out to its
        radius) are made coarser where they would take more than this
        fraction of ``max_evaluations`` (default 0.1).
    pilot_layout_fraction : float, optional
        The same for the frames before the last whitening round, whose
        boxes serve only to set the next frame (default
        ``layout_fraction``). The first frame's layout took 10500 of an
        eccentric PE's 23400 evaluations.
    final_share_frame_radius : float, optional
        ``share_frame_radius`` for the last round's frame (default the
        same).
    share_after_pilot : bool, optional
        With ``shared_region``, hand the model the pilot's region already,
        before the frame is set, so that the frame's own evaluations (the
        curvilinear frame's Newton climbs, or the Laplace frame's finite
        differences) go through the model's batch method too. The pilot's
        region is wider than the framed one, so its shared computation
        needs more memory; when it does not fit, the region is offered
        again after the frame as without this option. Default False.
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
    start_cube : array, optional
        Points of the prior's unit cube, ``(n, ndim)``, such as a search's
        loudest templates, evaluated before the pilot and offered to the
        frame's Newton climbs as starts beside the pilot's and the
        exploration's best points. They only seed the climbs: the
        quadrature still covers the whole prior.
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
    reference_logz : float, optional
        The evidence known from elsewhere, ``ln`` of the likelihood's integral
        over the prior cube (as :attr:`logz`): from importance sampling of a
        Laplace approximation, say. The refinement's thresholds are then
        ``mass_fraction`` of the larger of it and the tree's own evidence so
        far; the reported evidence stays the tree's. A tree that starts from
        the whole prior resolves its peak last, and its own evidence lies
        below the true one until then: in the eccentric PE's pp injections it
        was 1.6-28.1 below at the end of a 10000-evaluation budget (cover10,
        2026-11-19). With the reference (cover11) the evidence was unchanged
        and the evaluations were not fewer: the mid-range boxes are flagged by
        the bound's own slack, not by the threshold. An estimate that misses
        mass elsewhere is smaller than the true evidence and only makes the
        refinement more cautious. Default None.
    certificate_delta : float, optional
        With a ``box_snr_bound`` that has an ``overlap`` method (``overlap(
        centre, probes) -> (nu, ln_cells)``: each box's largest overlap of the
        true signal with its templates, in SNR, an upper bound, and the log of
        the number of its independent templates) and a ``dof`` (the degrees
        of freedom of ``2F`` at a fixed template), boxes are certified by the
        fractional-moment bound over shells of the overlap (see
        :class:`_ShellCertificate`) instead of one by one by the sup bound:
        the probability that the certified boxes hold more than
        ``mass_fraction`` of the evidence is at most this, on the event that
        the overlaps bound the true ones. Needs the prior cube or an affine
        chart (the boxes' prior masses bounded by their volumes). Default
        1e-3.
    certificate_shells : tuple of float, optional
        The shells' upper overlaps ``tau_g`` (SNR), fixed before the data.
        Default (0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10).
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
                 check_points=0, check_tolerance=0.1, start_cube=None,
                 share_after_pilot=False, frame_starts=5, share_frame=False,
                 climb_resolve=2.0, climb_precision=None,
                 climb_hessian='differences',
                 share_frame_radius=1.3, edge_points=1, frame_moments=False,
                 layout_fraction=0.1, pilot_layout_fraction=None,
                 final_share_frame_radius=None, linear_posterior=False,
                 linear_cap=10.0, box_distance=None, coverage_kappa=0.0,
                 coverage_resolution=0.5, coverage_probes=4,
                 box_snr_bound=None, coverage_chart=False,
                 reference_logz=None, certificate_delta=1e-3,
                 certificate_shells=(0.5, 1., 1.5, 2., 3., 4., 5., 6., 8.,
                                     10.)):
        super().__init__(model)
        self.reference_logz = (None if reference_logz is None
                               else float(reference_logz))
        # the coverage bound (see _coverage_bound): a box is refined while
        # the most posterior mass its template distance allows exceeds
        # mass_fraction of Z, until that distance is below the resolution
        self.box_distance = box_distance
        # or the bound itself: box_snr_bound(centre, probes, rho_centre) the
        # largest SNR anywhere in each box and its template distance
        self.box_snr_bound = box_snr_bound
        if box_distance is not None and box_snr_bound is not None:
            raise ValueError("give box_distance or box_snr_bound, not both")
        self._coverage = box_distance is not None or box_snr_bound is not None
        if self._coverage and (pilot_evaluations or curvilinear_frame
                               or laplace_frame or partition_file
                               or check_points):
            raise ValueError("the coverage bound refines a tree over the "
                             "prior cube itself: no pilot, frame, partition "
                             "or check points")
        self._peak_cube = None
        self._peak_logl = -numpy.inf
        # the evidence so far, for certifying new boxes before evaluating them
        self._current_logz = (-numpy.inf if self.reference_logz is None
                              else self.reference_logz)
        # the fractional-moment certificate, when the bound gives the boxes'
        # overlaps with the signal
        self._shell = (
            _ShellCertificate(certificate_delta,
                              getattr(box_snr_bound, 'dof', 2),
                              certificate_shells)
            if getattr(box_snr_bound, 'overlap', None) is not None else None)
        self._cert_threshold = -numpy.inf
        self._last_thetas = None
        # the coverage tree in the model's curvilinear coordinates (see
        # _chart_frame) rather than the prior cube
        # True: the model's curvilinear coordinates; a matrix A: the affine
        # chart y = A theta (no inverse map to solve)
        if isinstance(coverage_chart, (bool, numpy.bool_)) or coverage_chart is None:
            self._chart_matrix = None
            self.coverage_chart = bool(coverage_chart)
        else:
            self._chart_matrix = numpy.asarray(coverage_chart, dtype=float)
            if self._chart_matrix.shape != (len(model.variable_params),) * 2:
                raise ValueError("an affine coverage chart is a square matrix "
                                 "over the variable parameters")
            self.coverage_chart = True
        if self.coverage_chart and not self._coverage:
            raise ValueError("coverage_chart is for a coverage bound's tree")
        if (self._shell is not None and self.coverage_chart
                and self._chart_matrix is None):
            raise ValueError("the certificate bounds a box's prior mass by its "
                             "volume: the prior cube or an affine chart")
        # the Occam offset kappa in ln L = rho^2 / 2 - kappa(rho): a number, or
        # a callable of the SNR rho (see _kappa)
        self.coverage_kappa = (coverage_kappa if callable(coverage_kappa)
                               else float(coverage_kappa))
        self.coverage_resolution = float(coverage_resolution)
        self.coverage_probes = int(coverage_probes)
        if self.coverage_resolution < 0 or self.coverage_probes < 0:
            raise ValueError("coverage-resolution and coverage-probes must "
                             "not be negative")
        self._dist = None
        self.linear_posterior = bool(linear_posterior)
        self.linear_cap = float(linear_cap)
        self.initial_divisions = int(initial_divisions)
        self.partition_file = partition_file
        self.max_level = int(max_level)
        self.pilot_evaluations = int(pilot_evaluations)
        self.whitening_rounds = int(whitening_rounds)
        self.laplace_frame = bool(laplace_frame)
        self.curvilinear_frame = bool(curvilinear_frame)
        self.shared_region = int(shared_region)
        self.share_after_pilot = bool(share_after_pilot)
        self.frame_starts = int(frame_starts)
        self.climb_resolve = None if climb_resolve is None else float(climb_resolve)
        self.climb_precision = climb_precision
        if climb_hessian not in ('differences', 'metric'):
            raise ValueError("climb_hessian is 'differences' or 'metric'")
        if climb_hessian == 'metric' and climb_precision is None:
            raise ValueError("climb_hessian='metric' needs climb_precision")
        self.climb_hessian = climb_hessian
        self.share_frame = bool(share_frame)
        self.share_frame_radius = float(share_frame_radius)
        self.edge_points = int(edge_points)
        self.frame_moments = bool(frame_moments)
        self.layout_fraction = float(layout_fraction)
        self.pilot_layout_fraction = (self.layout_fraction
                                      if pilot_layout_fraction is None
                                      else float(pilot_layout_fraction))
        self.final_share_frame_radius = (
            self.share_frame_radius if final_share_frame_radius is None
            else float(final_share_frame_radius))
        self._layout = self.layout_fraction
        self._ball = self.share_frame_radius
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
        self.start_cube = None if start_cube is None else numpy.atleast_2d(
            numpy.asarray(start_cube, dtype=float))
        if self.start_cube is not None and (
                self.start_cube.shape[1] != len(model.variable_params)
                or numpy.any((self.start_cube < 0) | (self.start_cube > 1))):
            raise ValueError("start-cube points must lie in the unit cube, "
                             "one coordinate per variable parameter")
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
        # the model's batches give its loglikelihood only
        self._default_loglikelihood = loglikelihood_function in (
            None, 'loglikelihood')
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
        over the prior. Under a coverage bound a box the bound alone
        certifies (:meth:`_certify`) is not evaluated."""
        index = numpy.asarray(index)
        if self.box_snr_bound is not None and numpy.isfinite(
                self._current_logz) and len(index):
            index = self._certify(index)
            if not len(index):
                return
        centre = self._lower[index] + 0.5 * self._width[index]
        cube, logjac = self._map(centre)
        out = ~(numpy.all((cube >= 0) & (cube <= 1), axis=1)
                & numpy.isfinite(logjac))
        mean, inner, inside, s = self._volume(index)
        # a box straddling the prior's edge: a point inside it, where any
        has = out & numpy.isfinite(mean)
        if has.any():
            cube[has], logjac[has] = self._map(inner[has])
        logl = self._cube_loglikelihoods(cube, logjac)
        k = inside.shape[1]
        # which boxes reach beyond the prior (some volume point outside it):
        # their posterior draws are uniform, not linear (_draw_curved_posterior)
        self._beyond[index] = inside.sum(axis=1) < k
        straddle = numpy.nonzero((inside.sum(axis=1) < k) & inside.any(axis=1))[0]
        if self.edge_points > 1 and len(straddle):
            # a box straddling the prior's edge, valued at one inner point, can hold a likelihood that falls by tens
            # across its inner part (boxes on a prior bound the posterior presses on): its value the mean of the
            # likelihood at up to edge_points inner points, the one valued above and the next Sobol points inside
            extra, owner = [], []
            for b in straddle:
                pts = numpy.nonzero(inside[b])[0][:self.edge_points - 1]
                extra.extend(s[b, j] for j in pts)
                owner.extend([b] * len(pts))
            if extra:
                c2, j2 = self._map(numpy.array(extra))
                v2 = self._cube_loglikelihoods(c2, j2)
                owner = numpy.array(owner)
                for b in straddle:
                    vals = numpy.concatenate([[logl[b]], v2[owner == b]])
                    vals = vals[numpy.isfinite(vals)]
                    if len(vals):
                        logl[b] = logsumexp(vals) - numpy.log(len(vals))
        self._logl[index] = logl
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
        return mean, inner, inside, s

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
            if self._batched or (self._default_loglikelihood and getattr(
                    self.model, 'batches_points', False)):
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
        if self.start_cube is not None:
            logl = self._cube_loglikelihoods(self.start_cube)
            if self._explored is None:
                self._explored = (self.start_cube, logl)
            else:
                self._explored = (numpy.vstack([self._explored[0],
                                                self.start_cube]),
                                  numpy.concatenate([self._explored[1],
                                                     logl]))
        if self.coverage_chart:
            self._chart_frame()
        if self._coverage and self.start_cube is not None:
            # the start the peak's first estimate (see _update_peak)
            best = int(numpy.argmax(self._explored[1]))
            self._peak_cube = numpy.array(self._explored[0][best], dtype=float)
            self._peak_logl = float(self._explored[1][best])
        self._grid()
        if self.pilot_evaluations > 0:
            for whitening_round in range(self.whitening_rounds):
                budget = self.nevaluations + self.pilot_evaluations
                if budget >= self.max_evaluations:
                    break
                self._refine(budget)
                if (self.share_after_pilot and self.shared_region
                        and not self._batched):
                    # the frame's evaluations batched too; a region that
                    # does not fit is offered again after the frame
                    self._share()
                    self._next_share = 0
                last = whitening_round == self.whitening_rounds - 1
                self._layout = (self.layout_fraction if last
                                else self.pilot_layout_fraction)
                self._ball = (self.final_share_frame_radius if last
                              else self.share_frame_radius)
                if self.curvilinear_frame:
                    self._curvilinear(starts=self.frame_starts, moments=(
                        self.frame_moments and whitening_round > 0))
                elif self.laplace_frame:
                    self._whiten(*self._laplace(starts=self.frame_starts))
                else:
                    self._whiten()
                if self.shared_region and not self._batched:
                    self._share()
        elif self.shared_region and not self._batched and self._coverage:
            # the coverage tree starts from the whole prior: its region is
            # shared once the refinement has found where ln L is large
            self._next_share = self.coverage_share_after
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
        # the boxes the budget left unrefined, marked on the samples drawn from them
        self._pending = numpy.asarray(refine, dtype=int)
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
        self._dist = numpy.full(len(self._lower), numpy.nan)
        self._snr = numpy.full(len(self._lower), numpy.nan)
        self._axis_dist = numpy.full((len(self._lower), self.ndim), numpy.nan)
        self._certified = numpy.zeros(len(self._lower), dtype=bool)
        self._beyond = numpy.zeros(len(self._lower), dtype=bool)
        self._nu = numpy.full(len(self._lower), numpy.nan)
        self._lncells = numpy.full(len(self._lower), numpy.nan)
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
            if self._shell is not None:
                # the signal's estimate first: the certificate's overlaps are
                # with it (Gemini review: a move after the shells left this
                # round's threshold the old estimate's)
                self._update_peak(leaves)
            logmass = self._log_mass(leaves)
            logz = logsumexp(logmass)
            # the thresholds' evidence: the tree's own, or a larger one known
            # from elsewhere (reference_logz)
            ref = (logz if self.reference_logz is None
                   else max(logz, self.reference_logz))
            if (self._shell is not None and numpy.isfinite(ref)
                    and self._shell_round(leaves, ref)):
                # released boxes were evaluated: their mass enters
                logmass = self._log_mass(leaves)
                logz = logsumexp(logmass)
                ref = (logz if self.reference_logz is None
                       else max(logz, self.reference_logz))
            self.history.append((len(leaves), self.nevaluations, logz))
            self._current_logz = ref
            refine = self._logerr[leaves] - ref > threshold
            refine |= self._neighbour_flags(leaves, logmass - ref > threshold,
                                            ref + threshold)
            bound = None
            if self._coverage:
                bound, cover = self._coverage_flags(leaves, ref, threshold)
                refine |= cover
            if self._shell is not None:
                # a box the certificate holds is not refined, whatever flagged
                # it
                refine &= ~self._certified[leaves]
            refine &= self._level[leaves] < self.max_level
            if bound is not None:
                bound = bound[refine]
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
            if bound is not None:
                # best first: of the flagged boxes, those whose bound is within
                # a factor 1 / mass_fraction of the largest, at most a batch;
                # the evidence found where the bound is largest then prunes
                # the rest (cover4, 2026-11-19: with every flagged box split
                # each round, Z stayed near 0 and the whole prior was halved
                # uniformly)
                top = numpy.max(bound)
                if numpy.isfinite(top):
                    near = bound >= top + threshold
                    refine, bound = refine[near], bound[near]
                room = min(room, self.coverage_batch)
            if len(refine) > room:
                priority = numpy.maximum(self._logerr[refine],
                                         self._log_mass(refine))
                if bound is not None:
                    priority = numpy.fmax(priority, bound)
                refine = refine[numpy.argsort(-priority)[:room]]
            self._bisect(refine)

    def _coverage_bound(self, logl, distance):
        """The largest log likelihood anywhere in a box whose centre has
        ``logl`` and whose matched-filter SNR ``rho(x) = sqrt(2 F(x))`` rises
        by at most ``distance`` from the centre's (``box_distance`` gives it:
        ``|<d, h(x) - h(c)>|`` bounded through the triangle inequality in the
        noise-weighted inner product, ``h`` unit templates; a box far from
        the signal has only the noise's projection on its template
        differences), and ``ln L = rho^2 / 2 - kappa`` with ``kappa`` the
        marginalisation's Occam offset (``coverage_kappa``, measured at the
        start): ``ln L(x) <= (sqrt(2 (ln L(c) + kappa)) + distance)^2 / 2 -
        kappa``. The same covering argument as a search lattice's: there the
        distance is set so that no signal is missed, here so that no box
        misses posterior mass."""
        with numpy.errstate(invalid='ignore'):
            rho = self._snr_of(logl)
            top = rho + distance
            return numpy.where(numpy.isfinite(logl),
                               0.5 * top ** 2 - self._kappa(top), -numpy.inf)

    def _kappa(self, rho):
        """The Occam offset at SNR ``rho``: ``coverage_kappa`` itself, or its
        value there when it is a callable (the marginalised extrinsics'
        Laplace factor grows as ``k ln rho``; a constant measured at the peak
        overstates it at lower SNR, and the bound there)."""
        rho = numpy.asarray(rho, dtype=float)
        if callable(self.coverage_kappa):
            with numpy.errstate(invalid='ignore'):
                out = numpy.asarray(self.coverage_kappa(
                    numpy.where(numpy.isfinite(rho), rho, 1.0)), dtype=float)
            return numpy.where(numpy.isfinite(rho), out, 0.0)
        return numpy.full(rho.shape, self.coverage_kappa)

    def _snr_of(self, logl, iterations=40):
        """The SNR at which ``rho^2 / 2 - kappa(rho)`` is ``logl`` (a fixed
        point: kappa changes slowly with rho), ``inf`` where ``logl`` is not
        finite."""
        logl = numpy.asarray(logl, dtype=float)
        fin = numpy.isfinite(logl)
        out = numpy.full(logl.shape, numpy.inf)
        if not fin.any():
            return out
        rho = numpy.sqrt(2 * numpy.clip(logl[fin] + self._kappa(
            numpy.full(fin.sum(), 10.0)), 0, None))
        for _ in range(iterations):
            rho = numpy.sqrt(2 * numpy.clip(logl[fin] + self._kappa(rho), 0,
                                            None))
        out[fin] = rho
        return out

    def _probe_offsets(self):
        """Box offsets (in units of its half-widths, about its centre) at
        which the distance from the centre is measured: the centres of its
        ``2 ndim`` faces, its two opposite corners along ``(1, .., 1)`` and
        ``coverage_probes`` further corners from a fixed sample."""
        if getattr(self, '_probes', None) is None:
            n = self.ndim
            off = [numpy.eye(n), -numpy.eye(n), numpy.ones((1, n)),
                   -numpy.ones((1, n))]
            if self.coverage_probes:
                rng = numpy.random.default_rng(self.seed + 13)
                off.append(rng.choice([-1., 1.],
                                      size=(self.coverage_probes, n)))
            self._probes = numpy.vstack(off)
        return self._probes

    def _box_probes(self, index):
        """Each box's probe points in its box coordinates, ``(n, k,
        ndim)``: :meth:`_probe_offsets` and, once a peak is known, the box's
        point nearest it. Nearest in the template metric when the bound
        gives one (``box_snr_bound.metric``): a match with the signal falls
        with the metric distance from its peak, so its largest value over a
        box is there (cover6 toy, 2026-11-19: the Euclidean nearest point of
        a box in a sheared chart missed the peak and the box was certified,
        ln Z 1-2 low); and the Euclidean nearest point too."""
        off = self._probe_offsets()
        centre = self._lower[index] + 0.5 * self._width[index]
        s = centre[:, None, :] + 0.5 * off[None] * self._width[index][:, None, :]
        if self._peak_cube is not None:
            with numpy.errstate(invalid='ignore'):
                sp = self._unmap(numpy.asarray(self._peak_cube)[None])[0]
            if numpy.all(numpy.isfinite(sp)):
                lo = self._lower[index]
                hi = lo + self._width[index]
                near = [numpy.clip(sp[None], lo, hi)]
                metric = self._probe_metric()
                if metric is not None:
                    near.append(_nearest_in_boxes(metric, sp, lo, hi))
                s = numpy.concatenate([s] + [x[:, None, :] for x in near],
                                      axis=1)
        return centre, s

    def _probe_metric(self):
        """``box_snr_bound.metric`` (in the chart's coordinates for a chart
        tree, else in the parameters) in the box coordinates, or None."""
        given = getattr(self.box_snr_bound or self.box_distance, 'metric',
                        None)
        if given is None:
            return None
        if self.coverage_chart:
            scale = numpy.asarray(self.frame['span'], dtype=float)
        else:
            bottom = numpy.array(self._prior_transform(numpy.zeros(self.ndim)))
            top = numpy.array(self._prior_transform(numpy.ones(self.ndim)))
            scale = top - bottom
        g = numpy.asarray(given, dtype=float)
        return scale[:, None] * g * scale[None, :]

    def _box_thetas(self, index):
        """The parameters at each box's centre and probes (beyond the prior
        clipped onto its edge; ``nan`` where a point does not map)."""
        centre, s = self._box_probes(index)
        n, k = s.shape[:2]
        cube_c, _ = self._map(centre)
        cube_p, _ = self._map(s.reshape(-1, self.ndim))
        cube_c = numpy.clip(cube_c, 0, 1)
        cube_p = numpy.clip(cube_p, 0, 1)
        good_c = numpy.all(numpy.isfinite(cube_c), axis=1)
        good_p = numpy.all(numpy.isfinite(cube_p), axis=1).reshape(n, k)
        theta_c = numpy.full((n, self.ndim), numpy.nan)
        theta_p = numpy.full((n * k, self.ndim), numpy.nan)
        if good_c.any():
            theta_c[good_c] = [self._prior_transform(c)
                               for c in cube_c[good_c]]
        gp = good_p.ravel()
        if gp.any():
            theta_p[gp] = [self._prior_transform(c) for c in cube_p[gp]]
        theta_p = theta_p.reshape(n, k, self.ndim)
        both = good_c[:, None] & good_p
        # a probe that does not map is given the centre (distance 0); the
        # box is then refined through ``bad``
        theta_p = numpy.where(both[:, :, None], theta_p, theta_c[:, None, :])
        bad = ~both.all(axis=1)
        return theta_c, theta_p, good_c, bad

    def _box_snr_bounds(self, index, rho):
        """``box_snr_bound`` at boxes ``index`` with centre SNRs ``rho``:
        the largest SNR in each, its template distance and the distance
        along each axis (to the farther of its two faces' centres, from the
        bound's distances to each probe when it gives them); ``inf`` where
        a probe does not map."""
        theta_c, theta_p, good, bad = self._box_thetas(index)
        self._last_thetas = (numpy.array(index), theta_c, theta_p, good, bad)
        n = len(index)
        snr = numpy.full(n, numpy.inf)
        dist = numpy.full(n, numpy.inf)
        axis = numpy.full((n, self.ndim), numpy.nan)
        if good.any():
            out = self.box_snr_bound(theta_c[good], theta_p[good], rho[good])
            snr[good], dist[good] = out[0], out[1]
            if len(out) > 2:
                d = numpy.asarray(out[2], dtype=float)
                k = self.ndim
                axis[good] = numpy.maximum(d[:, :k], d[:, k:2 * k])
        snr[bad] = dist[bad] = numpy.inf
        return snr, dist, axis

    #: the rise in the largest box value that moves the signal's estimate
    coverage_peak_step = 2.0
    #: evaluations after which the coverage tree's region is first shared
    coverage_share_after = 2000
    #: the most boxes the coverage bound splits per round, largest bound first
    coverage_batch = 256

    def _update_peak(self, leaves):
        """Move the signal's estimate to the box with the largest ln L once
        it exceeds the estimate's by ``coverage_peak_step``: the bound's
        match is with the template there (``box_snr_bound.at_peak``), and
        every box's bound is recomputed."""
        logl = numpy.where(numpy.isfinite(self._logl[leaves]),
                           self._logl[leaves], -numpy.inf)
        best = int(numpy.argmax(logl))
        if logl[best] <= self._peak_logl + self.coverage_peak_step:
            return
        box = leaves[best]
        cube, _ = self._map((self._lower[box] + 0.5 * self._width[box])[None])
        self._peak_cube, self._peak_logl = cube[0], float(logl[best])
        at_peak = getattr(self.box_snr_bound, 'at_peak', None)
        if at_peak is not None:
            at_peak(numpy.array(self._prior_transform(cube[0])))
        self._snr[:] = numpy.nan
        self._axis_dist[:] = numpy.nan
        if self._shell is not None:
            # the overlaps were with the old estimate: recomputed, and the
            # certificate with them, before this round's refinement
            self._nu[:] = numpy.nan
            self._cert_threshold = -numpy.inf
        self._diagnostics['coverage_peak_moves'] = self._diagnostics.get(
            'coverage_peak_moves', 0) + 1
        logging.info("AMR coverage: the signal's estimate moved to ln L %.3f",
                     self._peak_logl)

    def _box_distances(self, index):
        """Each box's template distance: the largest of ``box_distance``
        from its centre to its probe points (:meth:`_box_probes`), its
        part inside the prior (points beyond it clipped onto the prior's
        edge); ``inf`` where no point maps, so that the box is refined."""
        centre, s = self._box_probes(index)
        n, k = s.shape[:2]
        cube_c, jc = self._map(centre)
        cube_p, jp = self._map(s.reshape(-1, self.ndim))
        cube_c = numpy.clip(cube_c, 0, 1)
        cube_p = numpy.clip(cube_p, 0, 1)
        good_c = numpy.all(numpy.isfinite(cube_c), axis=1)
        good_p = numpy.all(numpy.isfinite(cube_p), axis=1).reshape(n, k)
        theta_c = numpy.full((n, self.ndim), numpy.nan)
        theta_p = numpy.full((n * k, self.ndim), numpy.nan)
        if good_c.any():
            theta_c[good_c] = [self._prior_transform(c)
                               for c in cube_c[good_c]]
        gp = good_p.ravel()
        if gp.any():
            theta_p[gp] = [self._prior_transform(c) for c in cube_p[gp]]
        theta_p = theta_p.reshape(n, k, self.ndim)
        d = numpy.full((n, k), numpy.inf)
        both = good_c[:, None] & good_p
        rows = numpy.nonzero(both.any(axis=1))[0]
        if len(rows):
            # every row's centre with each of its probes, the probes that do
            # not map given the centre itself (distance 0, overwritten)
            tp = numpy.where(both[rows][:, :, None], theta_p[rows],
                             theta_c[rows][:, None, :])
            d[rows] = numpy.asarray(self.box_distance(theta_c[rows], tp),
                                    dtype=float).reshape(len(rows), k)
            d[rows] = numpy.where(both[rows], d[rows], numpy.inf)
        d = numpy.where(numpy.isnan(d), numpy.inf, d)
        return d.max(axis=1)

    def _tree_weights(self, bound_share):
        """The leaves, each one's probability in the tree proposal and its
        log volume in the cube (see :meth:`tree_proposal`)."""
        if self._curve is not None and not self.coverage_chart:
            raise ValueError("a tree proposal needs the cube's or the "
                             "coverage chart's boxes")
        leaves = self._leaves()
        logvol = (numpy.sum(numpy.log(self._width[leaves]), axis=1)
                  + self._logjac[leaves])
        mid = self._log_mass(leaves)
        logp = mid - logsumexp(mid[numpy.isfinite(mid)])
        if self._coverage and bound_share > 0:
            top = 0.5 * self._snr[leaves] ** 2 - self._kappa(self._snr[leaves])
            bound = numpy.where(numpy.isfinite(top) & numpy.isfinite(logvol),
                                top + logvol, -numpy.inf)
            if numpy.isfinite(bound).any():
                bound = bound - logsumexp(bound[numpy.isfinite(bound)])
                logp = numpy.logaddexp(numpy.log1p(-bound_share) + logp,
                                       numpy.log(bound_share) + bound)
        p = numpy.exp(logp - logsumexp(logp))
        return leaves, p / p.sum(), logvol

    def tree_proposal(self, n, rng=None, bound_share=0.1):
        """``n`` draws from the tree's leaves as a proposal of exact
        density, for importance sampling: a leaf chosen with probability
        ``(1 - bound_share)`` of its midpoint mass and ``bound_share`` of its
        coverage-bound mass (each normalised over the leaves), a point
        uniform inside it. Returns the cube points and ``ln q`` there (the
        density in the cube, piecewise constant); :meth:`tree_log_density`
        gives it at any point, for mixing the tree with other proposals.

        The bound's share bounds the importance weights: in a box b,
        ``L(x) / q(x) <= exp(U_b) V_b / (bound_share exp(U_b) V_b / B) =
        B / bound_share`` with ``B`` the bound's total mass, so no box the
        midpoints undervalue can give one weight that dominates."""
        rng = numpy.random.default_rng(self.seed + 23) if rng is None else rng
        leaves, p, logvol = self._tree_weights(bound_share)
        self._tree = (leaves, p, logvol, bound_share)
        pick = rng.choice(len(leaves), size=int(n), p=p)
        box = leaves[pick]
        s = self._lower[box] + rng.random((len(box), self.ndim)) \
            * self._width[box]
        cube, logjac = self._map(s)
        # uniform in the box coordinates: q(u) = p_b / prod(width_b) / |du/ds|
        # (a draw beyond the prior, possible in a chart's box, comes back
        # outside the cube or nan: weight 0)
        logq = (numpy.log(p[pick])
                - numpy.sum(numpy.log(self._width[box]), axis=1) - logjac)
        self._diagnostics.update(tree_proposal_boxes=int(len(numpy.unique(
            box))), tree_proposal_bound_share=float(bound_share))
        return cube, logq

    def tree_log_density(self, cube, bound_share=None):
        """``ln q`` of the tree proposal (:meth:`tree_proposal`, the last
        one's share unless ``bound_share`` is given) at cube points;
        ``-inf`` outside the tree."""
        if bound_share is None and getattr(self, '_tree', None) is not None:
            leaves, p, logvol, _ = self._tree
        else:
            leaves, p, logvol = self._tree_weights(
                0.1 if bound_share is None else bound_share)
        cube = numpy.atleast_2d(numpy.asarray(cube, dtype=float))
        out = numpy.full(len(cube), -numpy.inf)
        with numpy.errstate(invalid='ignore'):
            s = self._unmap(cube)
        ok = numpy.all(numpy.isfinite(s) & (s >= 0) & (s <= 1), axis=1)
        found = numpy.full(len(cube), -1)
        found[ok] = self._locate(s[ok])
        row = numpy.full(len(self._lower), -1)
        row[leaves] = numpy.arange(len(leaves))
        hit = found >= 0
        k = row[found[hit]]
        at = numpy.flatnonzero(hit)[k >= 0]
        box = leaves[k[k >= 0]]
        _, logjac = self._map(s[at])
        out[at] = (numpy.log(p[k[k >= 0]])
                   - numpy.sum(numpy.log(self._width[box]), axis=1) - logjac)
        return out

    def _certify(self, index):
        """Of new boxes ``index``, those the coverage bound certifies without
        a likelihood value: the bound with no centre value (the global part,
        an infinite centre SNR), times the box's prior mass, below
        ``mass_fraction`` of the evidence so far. They are marked certified,
        given no value (no mass in the evidence; the bound's share of the
        tree proposal keeps them sampled) and not refined. Returns the rest,
        to evaluate. A likelihood is computed only where the certificate
        cannot exclude posterior mass (cover7, 2026-11-19: 1350-2680 s of
        2130-3590 s went on direct evaluations of far boxes outside the
        shared region)."""
        mean, _, _, _ = self._volume(index)
        logvol = numpy.sum(numpy.log(self._width[index]), axis=1) + mean
        if self.coverage_chart:
            missing = ~numpy.isfinite(logvol)
            if missing.any():
                lv, held = self._chart_prior_mass(index[missing])
                logvol[missing] = numpy.where(held, lv, -numpy.inf)
        snr, dist, axis = self._box_snr_bounds(index, numpy.full(len(index),
                                                                 numpy.inf))
        with numpy.errstate(invalid='ignore'):
            top = 0.5 * snr ** 2 - self._kappa(snr)
            bound = numpy.where(numpy.isfinite(top), top, numpy.inf) + logvol
        if self._shell is not None:
            # one certificate: the shells' threshold of the last round (a box
            # the next round no longer holds is evaluated then)
            nu, _ = self._box_overlaps(index)
            ok = nu <= self._cert_threshold
        else:
            ok = (bound - self._current_logz < numpy.log(self.mass_fraction)) \
                | ~numpy.isfinite(logvol)
        done = index[ok]
        self._certified[done] = True
        self._logl[done] = -numpy.inf
        self._logjac[done] = mean[ok]
        self._snr[done], self._dist[done] = snr[ok], dist[ok]
        self._axis_dist[done] = axis[ok]
        self._diagnostics['coverage_certified_unevaluated'] = \
            self._diagnostics.get('coverage_certified_unevaluated', 0) + int(
                ok.sum())
        return index[~ok]

    def _thetas_of(self, index):
        """:meth:`_box_thetas`, reusing the last computed when for the same
        boxes (:meth:`_box_snr_bounds` has just mapped them)."""
        last = self._last_thetas
        if (last is not None and len(last[0]) == len(index)
                and numpy.array_equal(last[0], index)):
            return last[1:]
        return self._box_thetas(index)

    def _box_overlaps(self, index):
        """Each box's overlap bound and log cell count (``box_snr_bound.
        overlap``), computed once per signal estimate; ``inf`` where a probe
        does not map."""
        index = numpy.asarray(index)
        need = index[numpy.isnan(self._nu[index])]
        if len(need):
            theta_c, theta_p, good, bad = self._thetas_of(need)
            ok = good & ~bad
            # a box whose probes do not map: neither its overlap nor its
            # cells are known (not with the cube or an affine chart)
            nu = numpy.full(len(need), numpy.inf)
            ln = numpy.full(len(need), numpy.inf)
            if ok.any():
                a, b = self.box_snr_bound.overlap(theta_c[ok], theta_p[ok])
                nu[ok] = a
                ln[ok] = numpy.maximum(numpy.asarray(b, dtype=float), 0.)
            self._nu[need], self._lncells[need] = nu, ln
        return self._nu[index], self._lncells[index]

    def _log_prior_bound(self, index):
        """An upper bound on each box's prior mass: its volume times the
        chart's constant Jacobian (the cube's or an affine chart's), at most
        1 (not the volume points' estimate, which can miss a box's corner
        inside the prior)."""
        centre = self._lower[index] + 0.5 * self._width[index]
        _, logjac = self._map(centre)
        lv = numpy.sum(numpy.log(self._width[index]), axis=1) + logjac
        return numpy.minimum(numpy.where(numpy.isfinite(lv), lv, 0.), 0.)

    def _shell_round(self, leaves, logz):
        """Recompute the certificate from the leaves (see
        :class:`_ShellCertificate`): the threshold, the leaves it holds, and,
        for a leaf it no longer holds that has no value, its evaluation.
        Returns the number of boxes so evaluated."""
        nu, ln = self._box_overlaps(leaves)
        thr, logsum, used = self._shell.threshold(
            nu, self._log_prior_bound(leaves), ln,
            logz + numpy.log(self.mass_fraction))
        self._cert_threshold = thr
        held = nu <= thr
        released = leaves[self._certified[leaves] & ~held]
        self._certified[leaves] = held
        valueless = released[~numpy.isfinite(self._logl[released])]
        if len(valueless):
            self._logl[valueless] = numpy.nan
            self._evaluate(valueless)
            self._logerr[valueless] = self._log_mass(valueless)
        self._diagnostics.update(
            certificate_threshold=thr, certificate_shells_used=used,
            certificate_log_fraction=logsum - logz,
            certificate_boxes=int(held.sum()),
            certificate_released=self._diagnostics.get(
                'certificate_released', 0) + int(len(released)))
        return len(valueless)

    def _coverage_flags(self, leaves, logz, threshold):
        """The coverage bound of each leaf's mass, ``ln`` of its largest
        likelihood (:meth:`_coverage_bound`) times its volume, and the flag
        to refine it: a bound above ``mass_fraction`` of ``Z`` with the box's
        template distance still above ``coverage_resolution``, below which
        the likelihood is smooth over the box and the midpoint rules
        refine it."""
        logvol = (numpy.sum(numpy.log(self._width[leaves]), axis=1)
                  + self._logjac[leaves])
        if self.box_snr_bound is not None:
            self._update_peak(leaves)
            if self.coverage_chart:
                unknown = ~(numpy.isfinite(logvol)
                            & numpy.isfinite(self._logl[leaves]))
                if unknown.any():
                    lv, held = self._chart_prior_mass(leaves[unknown])
                    logvol = numpy.array(logvol, dtype=float)
                    logvol[unknown] = numpy.where(held, lv, -numpy.inf)
                    self._diagnostics.update(
                        coverage_draw_valued_boxes=int(held.sum()),
                        coverage_empty_boxes=int((~held).sum()),
                        coverage_prior_draws=int(self._chart_ndraw))
            need = leaves[numpy.isnan(self._snr[leaves])
                          & numpy.isfinite(logvol) & ~self._certified[leaves]]
            if len(need):
                rho = self._snr_of(self._logl[need])
                (self._snr[need], self._dist[need],
                 self._axis_dist[need]) = self._box_snr_bounds(need, rho)
            dist = self._dist[leaves]
            with numpy.errstate(invalid='ignore'):
                top = (0.5 * self._snr[leaves] ** 2
                       - self._kappa(self._snr[leaves]))
                bound = numpy.where(numpy.isfinite(logvol),
                                    numpy.where(numpy.isnan(top), numpy.inf,
                                                top) + logvol, -numpy.inf)
        else:
            need = leaves[numpy.isnan(self._dist[leaves])]
            if len(need):
                self._dist[need] = self._box_distances(need)
            dist = self._dist[leaves]
            with numpy.errstate(invalid='ignore'):
                bound = numpy.where(numpy.isfinite(logvol),
                                    self._coverage_bound(self._logl[leaves],
                                                         dist)
                                    + logvol, -numpy.inf)
        unresolved = dist > self.coverage_resolution
        # a box with no value (its centre and volume points outside the prior,
        # its mass counted nowhere but the bound) is refined whatever its
        # distance, until its halves have values: the resolution leaves a box
        # to the midpoint rules, which need its value
        unresolved |= (~numpy.isfinite(self._logl[leaves]) & numpy.isfinite(bound)
                       & ~self._certified[leaves])
        cover = (bound - logz > threshold) & unresolved & ~self._certified[leaves]
        # the most mass the boxes left unresolved could hold, as a fraction
        # of Z: the guarantee the refinement gives on stopping
        if unresolved.any() and numpy.isfinite(bound[unresolved]).any():
            missing = float(logsumexp(bound[unresolved]) - logz)
        else:
            missing = -numpy.inf
        self._diagnostics.update(
            coverage_flagged=int(cover.sum()),
            coverage_unresolved=int(unresolved.sum()),
            coverage_log_bound_fraction=missing,
            coverage_largest_distance=float(numpy.max(dist[numpy.isfinite(
                logvol)])) if numpy.isfinite(logvol).any() else 0.0)
        return bound, cover

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

    def _curvilinear(self, radius=4.0, prior_draws=20000, iterations=50,
                     starts=5, moments=False):
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
        moment_frame = None
        if moments and 'low' in (getattr(self, 'frame', None) or {}):
            # a frame from the current boxes' posterior, in y, the boxes' own spread included: the Laplace
            # approximation at the maximum can miss a direction the posterior is far wider in (108 times the
            # variance along one phase coordinate of the eccentric cell's, so the frame's ball covered a tenth of it)
            fr = self.frame
            leaves = self._leaves()
            lm = self._log_mass(leaves)
            ok = numpy.isfinite(lm)
            w = numpy.exp(lm[ok] - logsumexp(lm[ok]))
            centre = self._lower[leaves][ok] + 0.5 * self._width[leaves][ok]
            yb = fr['mean'] + (fr['low'] + fr['span'] * centre) @ fr['chol'].T
            mu = w @ yb
            d = yb - mu
            cov = d.T @ (d * w[:, None])
            spread = w @ ((fr['span'] * self._width[leaves][ok]) ** 2 / 12)
            cov = cov + fr['chol'] @ numpy.diag(spread) @ fr['chol'].T
            moment_frame = (mu, cov)
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
        # the prior's precision in units of its standard deviations: the
        # identity. The prior is uniform on a box in the parameters, with no
        # Gaussian cross-curvature between the coordinates y; the inverse of
        # its draws' correlation (until 2026-11-19) narrowed q about fivefold
        # at fixed phase coordinates, as it did the eccentric PE's Laplace
        # (cover14; Gemini review)
        prior_precision = numpy.eye(self.ndim)

        def psd(m):
            # the information is positive semidefinite: a negative eigenvalue
            # is the differences' rounding
            m = 0.5 * (m + m.T)
            ev, vec = numpy.linalg.eigh(m)
            return (vec * numpy.clip(ev, 0, None)) @ vec.T

        def climb(y0):
            """The trust-region Newton climb of ln L (:func:`_climb`) in
            ``z``, ``y = y0 + unit z``: the end point, the precision there
            and ln L."""
            def logl_zs(z):
                """ln L at the rows y = y0 + unit z, the map inverted and
                the likelihood evaluated for all rows together."""
                theta = from_y(y0 + unit * numpy.atleast_2d(z))
                u = (theta - bottom) / prior_range
                u[~numpy.all(numpy.isfinite(u), axis=1)] = -1.0
                return self._cube_loglikelihoods(u)
            metric = None
            if self.climb_precision is not None:
                # the expected precision, in z: a callable of y evaluated
                # where the climb is (Fisher scoring re-evaluates it after
                # every accepted step), or a fixed matrix
                given = self.climb_precision
                if callable(given):
                    def metric(z):
                        return psd(unit[:, None] * numpy.asarray(
                            given(y0 + unit * z), dtype=float)
                            * unit[None, :])
                else:
                    metric = psd(unit[:, None] * numpy.asarray(
                        given, dtype=float) * unit[None, :])
            x, precision, value = _climb(
                logl_zs, numpy.zeros(self.ndim), prior_precision,
                metric=metric, resolve=self.climb_resolve, top=1e-1,
                iterations=iterations, hessian=self.climb_hessian)
            return y0 + unit * x, precision, value
        if moment_frame is not None:
            mean, cov = moment_frame
            chol = numpy.linalg.cholesky(cov)
            logging.info("AMR: frame from the boxes' moments after %d "
                         "evaluations", self.nevaluations)
            self._diagnostics.update(frame_moments_evaluations=self.nevaluations)
        else:
            # from the pilot's and the exploration's best, separate points, the
            # highest maximum
            best = None
            climbs = []
            for point in self._climb_starts(starts):
                start = numpy.array(self._prior_transform(point))
                y0 = to_y(start[None])[0]
                if not numpy.all(numpy.isfinite(y0)):
                    continue
                found = climb(y0)
                climbs.append(found)
                if best is None or found[2] > best[2]:
                    best = found
            mean, precision, peak = best
            logging.info("AMR: frame about ln L %.3f", peak)
            # every climb's end: another peak within 5 of the highest and
            # more than 5 of its standard deviations from it is a second mode
            # the frame (one whitening) is not laid out for
            sd = numpy.sqrt(numpy.diag(numpy.linalg.inv(precision)))
            ends = [dict(peak=float(c[2]), distance_sd=float(numpy.max(
                numpy.abs((c[0] - mean) / unit / sd)))) for c in climbs]
            second = [e for e in ends if e['peak'] > peak - 5
                      and e['distance_sd'] > 5]
            if second:
                logging.warning("AMR: %d climbs end at another peak within 5 "
                                "of the highest (%s)", len(second), second)
            self._diagnostics.update(frame_peak=float(peak),
                                     frame_starts=int(starts),
                                     frame_climbs=ends,
                                     frame_second_modes=len(second),
                                     frame_evaluations=self.nevaluations)
            chol = numpy.linalg.cholesky(numpy.linalg.inv(precision)) \
                * unit[:, None]
        inverse = numpy.linalg.inv(chol)
        # the frame, for diagnosis: y = mean + chol w
        self.frame = dict(mean=numpy.array(mean), chol=numpy.array(chol), names=list(names), coords=list(coords.names))
        whitened = (image - mean) @ inverse.T
        low = whitened.min(axis=0)
        span = whitened.max(axis=0) - low
        low, span = low - 0.01 * span, 1.02 * span
        self.frame.update(low=numpy.array(low), span=numpy.array(span))
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
        if self.share_frame and self.shared_region:
            # the boxes reach past the radius by up to half a box (those whose nearest point lies within it), and the
            # refinement past them
            self._share_ball(low, span, self._ball * radius)
        self._start_frame(low, span, radius)

    def _chart_frame(self, prior_draws=20000):
        """The coverage tree's boxes in the model's curvilinear coordinates
        ``y`` (``model.curvilinear_coordinates``: for a template family its
        phase coordinates, the chart a search lattice is laid in), over the
        bound of the prior's image there: ``y = low + span s``, no climb and
        no whitening. Where the template's phase is nearly linear in ``y``
        the posterior's thin directions are nearly along its axes; in the
        prior cube they are oblique (cover5 pp 63, 2026-11-19: Fisher
        eigenvalues 0 to 3e9, the posterior a sheet in (mchirp, e, chi_eff)
        that 10000 axis-aligned boxes did not resolve, ESS 15). Unlike a
        Laplace-whitened frame, whose bound was 1e7 posterior widths along a
        zero-information direction, the image fills much of this bound."""
        names = list(self.model.variable_params)
        bottom = numpy.array(self._prior_transform(numpy.zeros(self.ndim)))
        top = numpy.array(self._prior_transform(numpy.ones(self.ndim)))
        prior_range = top - bottom
        from scipy.stats import qmc
        draws = bottom + qmc.Sobol(self.ndim, seed=self.seed + 19).random_base2(
            int(numpy.ceil(numpy.log2(prior_draws)))) * prior_range
        if self._chart_matrix is not None:
            # the affine chart: y = A theta, its inverse a solve, its Jacobian
            # a constant (the tree only orients its boxes; the bound is
            # exact at the parameters)
            amat = self._chart_matrix
            ainv = numpy.linalg.inv(amat)
            lnabs = float(numpy.linalg.slogdet(amat)[1])
            cnames = names

            def to_y(theta):
                return numpy.asarray(theta, dtype=float) @ amat.T

            def from_y(y):
                return numpy.asarray(y, dtype=float) @ ainv.T

            def log_jacobian(theta):
                return numpy.full(len(theta), -lnabs)
        else:
            coords = self.model.curvilinear_coordinates
            idx = [names.index(n) for n in coords.names]
            cnames = list(coords.names)

            def to_y(theta):
                y = numpy.array(theta, dtype=float, copy=True)
                y[:, idx] = coords.forward(theta[:, idx])
                return y

            def from_y(y):
                theta = numpy.array(y, dtype=float, copy=True)
                theta[:, idx] = coords.inverse(y[:, idx])
                return theta

            def log_jacobian(theta):
                return coords.log_jacobian(theta[:, idx])
            if hasattr(coords, 'fit'):
                coords.fit(draws[:, idx])
        image = to_y(draws)
        image = image[numpy.all(numpy.isfinite(image), axis=1)]
        low, high = image.min(axis=0), image.max(axis=0)
        span = high - low
        low, span = low - 0.01 * span, 1.02 * span
        logdet = float(numpy.sum(numpy.log(span))
                       - numpy.sum(numpy.log(prior_range)))

        def curve(s):
            theta = from_y(low + span * s)
            cube = (theta - bottom) / prior_range
            bad = ~numpy.all(numpy.isfinite(theta), axis=1)
            logjac = numpy.full(len(theta), -numpy.inf)
            if (~bad).any():
                logjac[~bad] = logdet + log_jacobian(theta[~bad])
            cube[bad] = numpy.nan
            return cube, logjac

        def uncurve(cube):
            with numpy.errstate(invalid='ignore'):
                y = to_y(bottom + cube * prior_range)
            return (y - low) / span
        self._curve, self._uncurve = curve, uncurve
        self._scale = numpy.ones(self.ndim)
        # the prior's draws in the tree's coordinates: a box's prior mass where
        # its volume points miss the prior's image (see _chart_prior_mass)
        sp = (image - low) / span
        self._chart_draws = sp[numpy.all((sp >= 0) & (sp <= 1), axis=1)]
        self._chart_ndraw = len(draws)
        self.frame = dict(mean=numpy.zeros(self.ndim), chol=numpy.eye(self.ndim),
                          low=low, span=span, names=names, coords=cnames,
                          chart='affine' if self._chart_matrix is not None
                          else 'curvilinear')
        self._diagnostics.update(chart_log_volume=float(numpy.sum(numpy.log(
            span))))
        logging.info("AMR coverage: the tree in the %s chart over the prior's "
                     "image's bound", self.frame['chart'])

    def _chart_prior_mass(self, leaves):
        """``ln`` of the prior mass of chart boxes from the prior's draws they
        hold, ``(count + 3) / N``, and which hold any. The prior's image is a
        curved part of the chart's bound: a box's 16 volume points
        (:meth:`_volume`) can all miss it although the box holds some of it
        (cover6 pp 5, 2026-11-19: the root box, ln Z -inf). A box holding no
        draw is taken as empty; together those hold less than ``3 / N`` of
        the prior (95%)."""
        found = self._locate(self._chart_draws)
        counts = numpy.bincount(found[found >= 0], minlength=len(self._lower))
        c = counts[leaves]
        return numpy.log((c + 3.0) / self._chart_ndraw), c > 0

    def _share_ball(self, low, span, radius):
        """Hand the model, before the frame's first boxes are evaluated, the
        region they cover: ``shared_region`` points drawn uniformly in the
        frame's whitened ball of ``radius`` (the boxes :meth:`_start_frame`
        lays out), inside the prior. Without it the first batch, which holds
        boxes far from the maximum, is evaluated before any region covers
        it."""
        rng = numpy.random.default_rng(self.seed + 2)
        z = rng.normal(size=(self.shared_region, self.ndim))
        z *= (radius * rng.random(len(z)) ** (1.0 / self.ndim)
              / numpy.linalg.norm(z, axis=1))[:, None]
        cube, logjac = self._curve((z - low) / span)
        ok = (numpy.all((cube >= 0) & (cube <= 1), axis=1)
              & numpy.isfinite(logjac))
        names = list(self.model.variable_params)
        full = []
        for c in cube[ok]:
            q = dict(self.model.static_params)
            q.update(dict(zip(names, self._prior_transform(c))))
            full.append(q)
        tick = time.time()
        try:
            self.model.share(full)
        except MemoryError as err:
            logging.info("AMR: the frame's ball does not fit a shared "
                         "region (%s)", err)
            return
        self._batched = True
        logging.info("AMR: shared the frame's ball, %d points, in %.0f s",
                     len(full), time.time() - tick)

    def _laplace(self, iterations=50, starts=5):
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
            return _climb(logl_us, numpy.asarray(x, dtype=float), prior,
                          resolve=self.climb_resolve, top=1e-2,
                          iterations=iterations,
                          project=lambda v: numpy.clip(v, 1e-9, 1 - 1e-9))
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
        of the origin, unless that would take more than ``_layout`` (a
        tenth by default) of the budget."""
        self._grid(evaluate=False)
        target = max(1.0, 2 * radius / (self._layout * self.max_evaluations)
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
                # a certified box has no value: it is not refined for being
                # coarse, and there is no step to it
                cert = getattr(self, '_certified', None)
                if cert is not None and len(cert) == len(self._lower):
                    coarse &= ~cert[found]
                flag[found[coarse]] = True
                with numpy.errstate(invalid='ignore'):
                    step = ~(numpy.abs(self._logl[origin] - self._logl[found])
                             <= self.loglikelihood_step)
                # two boxes outside the likelihood's support do not step
                step &= numpy.isfinite(self._logl[origin])
                step &= flag_significant[origin]
                if cert is not None and len(cert) == len(self._lower):
                    step &= ~cert[found]
                flag[origin[step]] = True
                flag[found[step & (coarseness >= size[inside])]] = True
        return flag[leaves]

    def _split(self, index):
        """Halves boxes along their widest side, without evaluating them;
        returns the indices of the lower and the upper halves."""
        lower, width = self._lower[index], self._width[index]
        axis = numpy.argmax(width * self._scale, axis=1)
        if self._coverage:
            # under the coverage bound, along the axis over which the box's
            # template changes most: the box's template diameter falls
            # fastest, and boxes stretch along a ridge's zero-information
            # directions
            dist = self._axis_dist[index]
            known = numpy.all(numpy.isfinite(dist), axis=1)
            axis[known] = numpy.argmax(dist[known], axis=1)
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
        self._dist = numpy.concatenate([self._dist,
                                        numpy.full(2 * len(index), numpy.nan)])
        self._snr = numpy.concatenate([self._snr,
                                       numpy.full(2 * len(index), numpy.nan)])
        self._axis_dist = numpy.vstack([self._axis_dist, numpy.full(
            (2 * len(index), self.ndim), numpy.nan)])
        self._certified = numpy.concatenate([self._certified, numpy.zeros(
            2 * len(index), dtype=bool)])
        self._beyond = numpy.concatenate([self._beyond, numpy.zeros(
            2 * len(index), dtype=bool)])
        self._nu = numpy.concatenate([self._nu, numpy.full(2 * len(index),
                                                           numpy.nan)])
        self._lncells = numpy.concatenate([self._lncells, numpy.full(
            2 * len(index), numpy.nan)])
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

    def _box_slopes(self, index):
        """Each box's slope of ln L along each axis of the box coordinates,
        ``(len(index), ndim)``: the central difference of its two face
        neighbours' centre values (probed across the middle of each face),
        the one-sided difference where one neighbour is missing or outside
        the likelihood's support, zero where both are."""
        n = len(index)
        centre = self._lower[index] + 0.5 * self._width[index]
        value = self._logl[index]
        slope = numpy.zeros((n, self.ndim))
        for axis in range(self.ndim):
            vals, dist = [], []
            for sign in (-1, 1):
                probe = centre.copy()
                probe[:, axis] += sign * 0.5 * self._width[index, axis] * (
                    1 + 1e-9)
                inside = (probe[:, axis] > 0) & (probe[:, axis] < 1)
                found = numpy.full(n, -1)
                found[inside] = self._locate(probe[inside])
                v = numpy.full(n, numpy.nan)
                d = numpy.full(n, numpy.nan)
                has = found >= 0
                v[has] = self._logl[found[has]]
                d[has] = (self._lower[found[has], axis]
                          + 0.5 * self._width[found[has], axis]
                          - centre[has, axis])
                v[~numpy.isfinite(v)] = numpy.nan
                vals.append(v)
                dist.append(d)
            (vm, vp), (dm, dp) = vals, dist
            with numpy.errstate(invalid='ignore', divide='ignore'):
                both = numpy.isfinite(vm) & numpy.isfinite(vp)
                plus = ~both & numpy.isfinite(vp)
                minus = ~both & numpy.isfinite(vm)
                g = numpy.zeros(n)
                g[both] = (vp[both] - vm[both]) / (dp[both] - dm[both])
                g[plus] = (vp[plus] - value[plus]) / dp[plus]
                g[minus] = (vm[minus] - value[minus]) / dm[minus]
            g[~numpy.isfinite(g) | ~numpy.isfinite(value)] = 0.0
            slope[:, axis] = g
        # at most linear_cap across a box per axis: a step to a neighbour
        # far below (the support's edge) is not a slope
        cap = self.linear_cap / self._width[index]
        return numpy.clip(slope, -cap, cap)

    @staticmethod
    def _linear_log_factor(a):
        """ln of the mean of exp(a (t - 1/2)) over t in [0, 1], summed over
        the last axis: a box's mass under the linear density over its
        midpoint mass."""
        a = numpy.asarray(a, dtype=float)
        small = numpy.abs(a) < 1e-8
        safe = numpy.where(small, 1.0, a)
        out = numpy.where(small, 0.0,
                          numpy.log(numpy.expm1(safe) / safe) - 0.5 * safe)
        return out.sum(axis=-1)

    @staticmethod
    def _linear_unit_draw(a, u):
        """t in [0, 1] with density proportional to exp(a t), from uniform
        ``u``."""
        small = numpy.abs(a) < 1e-8
        safe = numpy.where(small, 1.0, a)
        t = numpy.log1p(u * numpy.expm1(safe)) / safe
        return numpy.clip(numpy.where(small, u, t), 0.0, 1.0)

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
        # the box each point came from: its depth and its mass error over Z
        # (for judging importance weights against the boxes afterwards)
        samples['box_level'] = self._level[boxes].astype(float)
        samples['box_logerr'] = self._logerr[boxes] - self._logz
        samples['box_index'] = boxes.astype(float)
        samples['box_pending'] = numpy.isin(
            boxes, getattr(self, '_pending', ())).astype(float)
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
        slopes = None
        if self.linear_posterior:
            # a box's density linear in ln L across it, the slopes from its
            # neighbours: exp(l + g (x - c)) in place of exp(l), its mass the
            # midpoint mass times the mean of that factor over the box (the
            # importance weights against the samples' own ln L then lose the
            # first-order variation across a box)
            slopes = numpy.zeros((len(self._lower), self.ndim))
            slopes[leaves] = self._box_slopes(leaves)
            # a box reaching beyond the prior is drawn uniformly: ln L rises
            # towards a bound the posterior presses on, a slope there points
            # out of the prior, and the rejection of the draws beyond it ran
            # without end (2026-11-19, the eccentric PE's curvilinear frame,
            # 30 minutes on one draw; Gemini review)
            slopes[leaves[self._beyond[leaves]]] = 0.0
            logmass = logmass + self._linear_log_factor(
                slopes[leaves] * self._width[leaves])
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
        offsets = numpy.zeros((len(trial_boxes), self.ndim))
        while len(todo):
            chosen = trial_boxes[todo]
            unit = rng.random((len(chosen), self.ndim))
            if slopes is not None:
                unit = self._linear_unit_draw(
                    slopes[chosen] * self._width[chosen], unit)
            point = self._lower[chosen] + unit * self._width[chosen]
            if slopes is not None:
                offsets[todo] = point - (self._lower[chosen]
                                         + 0.5 * self._width[chosen])
            mapped, jacobian = self._map(point)
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
        if slopes is not None:
            samples['loglikelihood'] = self._logl[boxes] + numpy.sum(
                slopes[boxes] * offsets[:number], axis=1)
        # the box each point came from: its depth and its mass error over Z
        # (for judging importance weights against the boxes afterwards)
        samples['box_level'] = self._level[boxes].astype(float)
        samples['box_logerr'] = self._logerr[boxes] - self._logz
        samples['box_index'] = boxes.astype(float)
        samples['box_pending'] = numpy.isin(
            boxes, getattr(self, '_pending', ())).astype(float)
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
                 'share-after-pilot': _boolean,
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


def _nearest_in_boxes(metric, point, lower, upper, sweeps=200):
    """The point of each box ``[lower, upper]`` (rows) nearest ``point`` in
    the quadratic form ``metric``: the box-constrained least squares
    ``min (x - p)^T G (x - p)``, solved by cyclic coordinate descent (each
    step the exact minimum along one coordinate, clipped to the box; it
    converges for a positive semidefinite G). Directions of no information
    (``G_ii = 0``) keep the point's own coordinate, clipped."""
    g = numpy.asarray(metric, dtype=float)
    g = 0.5 * (g + g.T)
    diag = numpy.diag(g).copy()
    x = numpy.clip(numpy.asarray(point, dtype=float)[None], lower, upper)
    p = numpy.asarray(point, dtype=float)
    free = diag > 0
    for _ in range(int(sweeps)):
        moved = 0.0
        for i in numpy.flatnonzero(free):
            r = (x - p) @ g[:, i] - (x[:, i] - p[i]) * g[i, i]
            new = numpy.clip(p[i] - r / g[i, i], lower[:, i], upper[:, i])
            moved = max(moved, float(numpy.max(numpy.abs(new - x[:, i])
                                               / numpy.maximum(upper[:, i]
                                                               - lower[:, i],
                                                               1e-300))))
            x[:, i] = new
        if moved < 1e-9:
            break
    return x


def _climb(f, x, prior_precision, metric=None, resolve=None, top=1e-1,
           iterations=50, project=None, radius=3.0, tolerance=1e-3,
           hessian='differences'):
    """A trust-region Newton climb of ``f`` (rows to values) from ``x``:
    the end point, the precision there and ``f`` there.

    Each iteration takes the gradient and Hessian by finite differences
    (:func:`_finite_differences`, steps resolved to ``resolve`` and half a
    standard deviation of the last precision); the model's precision is the
    differences' concave part, the ``metric`` (the expected precision) along
    directions where they find no curvature, and ``prior_precision``. The
    step is the Newton step held within ``radius`` in that precision's norm
    (the exact solution of the trust-region problem in that norm) and within
    eight resolved steps where no curvature is known at all. It is taken if
    ``f`` rises by more than a tenth of the model's prediction; the radius
    is quartered when the rise is below a quarter of it and doubled (up to
    ten) when it is above three quarters at the boundary. The climb ends
    when the model predicts less than ``tolerance``, the radius falls below
    1e-4, or after ``iterations`` iterations. The ratio test makes every
    accepted step a sufficient rise, so the climb converges to a stationary
    point (the standard trust-region result); a start within the main lobe
    of the peak then reaches the peak. (Until 2026-11-18: at most eight
    Newton steps, each halved at most six times from three prior standard
    deviations; a peak 1.5e-4 of the prior wide was never reached.)

    ``project`` maps a trial point back into the support (the cube).

    ``hessian='metric'`` (Fisher scoring): the model's precision is the
    ``metric`` alone, with the prior's -- the expected Fisher information,
    positive definite wherever it is taken -- and only the gradient is
    differenced (2 n evaluations an iteration instead of about n^2 / 2).
    ``metric`` may then be a callable of the current point, evaluated again
    after every accepted step. The differences' Hessian of a noisy ln L is
    convex on a peak's tail; the information matrix never is."""
    x = numpy.array(x, dtype=float)
    value = float(f(x[None])[0])
    if hessian not in ('differences', 'metric'):
        raise ValueError("hessian is 'differences' or 'metric'")
    if hessian == 'metric' and metric is None:
        raise ValueError("Fisher scoring needs the metric")
    information = metric if callable(metric) else (lambda at: metric)
    current = None if metric is None else numpy.asarray(information(x), dtype=float)
    precision = prior_precision if current is None else prior_precision + current
    step = numpy.minimum(1e-3, numpy.clip(
        0.5 / numpy.sqrt(numpy.diag(precision)), 1e-9, None))
    fresh = True
    for _ in range(int(iterations)):
        if fresh and hessian == 'metric':
            grad, used = _gradient(f, x, step, resolve=resolve)
            current = numpy.asarray(information(x), dtype=float)
            precision = current + prior_precision
            curved = numpy.ones(len(x), dtype=bool)
            newton = numpy.linalg.solve(precision, grad)
            fresh = False
        elif fresh:
            metric_x = current
            grad, hess, used = _finite_differences(
                f, x, step, resolve=resolve, return_step=True)
            values, vectors = numpy.linalg.eigh(-0.5 * (hess + hess.T))
            found = (vectors * numpy.maximum(values, 0)) @ vectors.T
            curved = values > 0
            if metric_x is not None:
                along = numpy.einsum('ji,jk,ki->i', vectors, metric_x,
                                     vectors)
                flat = ~curved & (along > 0)
                found = found + (vectors[:, flat] * along[flat]) \
                    @ vectors[:, flat].T
                curved = curved | flat
            precision = found + prior_precision
            newton = numpy.linalg.solve(precision, grad)
            fresh = False
        size = numpy.sqrt(newton @ precision @ newton)
        move = newton if size <= radius else newton * (radius / size)
        move = _trust(move, curved.astype(float), used, resolve)
        size = numpy.sqrt(move @ precision @ move)
        predicted = grad @ move - 0.5 * move @ precision @ move
        if not predicted > tolerance:
            break
        trial_x = x + move if project is None else project(x + move)
        trial = float(f(trial_x[None])[0])
        ratio = (trial - value) / predicted if numpy.isfinite(trial) \
            else -numpy.inf
        if ratio > 0.1:
            x, value = trial_x, trial
            step = _next_step(precision, used, resolve, top)
            fresh = True
        if ratio < 0.25:
            radius = 0.25 * size
        elif ratio > 0.75 and size >= 0.99 * radius:
            radius = min(2 * radius, 10.0)
        if radius < 1e-4:
            break
    return x, precision, value


def _gradient(f, x, step, resolve=None, refine=6, shrink=6):
    """The gradient of ``f`` at ``x`` by central differences along the axes
    (one-sided, second order, where ``f`` is not finite on one side, as at a
    prior bound; an axis with neither side finite has its step halved up to
    ``shrink`` times), the steps cut by eight up to ``refine`` times where
    one changes ``f`` by more than ``resolve``: the gradient and the steps
    used."""
    n = len(x)
    step = numpy.array(step, dtype=float)
    f0 = float(f(x[None])[0])
    plus, minus = numpy.full(n, numpy.nan), numpy.full(n, numpy.nan)

    def at(todo):
        eye = numpy.diag(step)
        values = numpy.asarray(f(numpy.array(
            [x + eye[i] for i in todo] + [x - eye[i] for i in todo])),
            dtype=float)
        plus[todo], minus[todo] = values[:len(todo)], values[len(todo):]
    todo = numpy.arange(n)
    for _ in range(shrink + 1):
        at(todo)
        todo = todo[~numpy.isfinite(plus[todo]) & ~numpy.isfinite(minus[todo])]
        if not len(todo):
            break
        step[todo] /= 2
    if resolve is not None and numpy.isfinite(f0):
        for _ in range(refine):
            change = _resolved_change(plus, minus, f0)
            todo = numpy.flatnonzero(change > resolve)
            if not len(todo):
                break
            step[todo] /= 8
            at(todo)
    grad = numpy.zeros(n)
    both = numpy.isfinite(plus) & numpy.isfinite(minus)
    grad[both] = (plus[both] - minus[both]) / (2 * step[both])
    for i in numpy.flatnonzero(~both):
        sign = 1 if numpy.isfinite(plus[i]) else -1
        if not numpy.isfinite(plus[i]) and not numpy.isfinite(minus[i]):
            continue
        h = sign * step[i]
        near = plus[i] if sign > 0 else minus[i]
        two = float(f((x + 2 * h * numpy.eye(n)[i])[None])[0])
        if numpy.isfinite(two):
            grad[i] = (-3 * f0 + 4 * near - two) / (2 * h)
        else:
            grad[i] = (near - f0) / h
    return grad, step


def _trust(move, curvatures, steps, resolve, reach=8.0):
    """A climb's Newton ``move`` held within ``reach`` resolved steps per
    axis where the Hessian (eigenvalues ``curvatures`` of -H) has a
    direction without curvature: there the precision is the prior's alone
    and says nothing of how far the local model holds."""
    if resolve is None or numpy.all(curvatures > 0):
        return move
    limit = reach * numpy.asarray(steps, dtype=float)
    big = numpy.abs(move) > limit
    if not big.any():
        return move
    return move * numpy.min(limit[big] / numpy.abs(move[big]))


def _next_step(precision, steps, resolve, top):
    """The next difference steps: half a standard deviation of the
    precision, clipped; with ``resolve`` at most four times the steps just
    resolved (a precision that is the prior's alone would otherwise put
    them back at the prior's scale)."""
    step = 0.5 / numpy.sqrt(numpy.diag(precision))
    if resolve is not None:
        step = numpy.minimum(step, 4 * numpy.asarray(steps, dtype=float))
    return numpy.clip(step, 1e-7, top)


def _resolved_change(plus, minus, f0):
    """The largest change of ``f`` over an axis's two steps, of those that
    land where ``f`` is finite: a step beyond the prior's edge (``f`` is
    ``-inf`` there) is the one-sided differences' business, not a change to
    resolve. (Until 2026-11-19 ``abs(-inf - f0) = inf`` counted as a change
    larger than any ``resolve``, and an axis at a prior bound had its step cut
    by 8 at each of the ``refine`` rounds: the eccentric PE's q at its upper
    bound got 2.3e-10 of a 6.0e-5 step, and its Fisher information came out
    zero; found by the Gemini review.)"""
    with numpy.errstate(invalid='ignore'):
        dp = numpy.where(numpy.isfinite(plus), numpy.abs(plus - f0), numpy.nan)
        dm = numpy.where(numpy.isfinite(minus), numpy.abs(minus - f0),
                         numpy.nan)
        return numpy.fmax(dp, dm)


def _finite_differences(f, x, step, shrink=6, resolve=None, refine=6,
                        return_step=False):
    """Gradient and Hessian of ``f`` at ``x``: central differences along
    the axes and forward ones for the mixed terms. ``f`` takes the rows of
    an array of points and returns their values, so that all the points
    are evaluated together.

    Next to the edge of the support (``f`` not finite on one side of an
    axis, as at a prior bound) the differences along that axis are one-
    sided, from the points at one and two steps on the other side, and the
    mixed terms go to the sides that are inside; an axis with neither side
    inside has its step halved, up to ``shrink`` times. A peak on a prior
    bound then has the curvature it has inside rather than none.

    With ``resolve``, an axis whose step changes ``f`` by more than it (on
    either side) has its step cut by eight, up to ``refine`` times, so that
    the differences see the function's curvature rather than its shape
    over several of its widths. ``return_step``: the steps used too."""
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
    if resolve is not None:
        for _ in range(refine):
            change = _resolved_change(plus, minus, f0)
            todo = numpy.flatnonzero(change > resolve)
            if not len(todo) or not numpy.isfinite(f0):
                break
            step[todo] /= 8
            eye = numpy.diag(step)
            values = numpy.asarray(f(numpy.array(
                [x + eye[i] for i in todo] + [x - eye[i] for i in todo])),
                dtype=float)
            plus[todo], minus[todo] = values[:len(todo)], values[len(todo):]
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
    if return_step:
        return grad, hess, step
    return grad, hess

