# Copyright (C) 2026 Shichao Wu
# This program is free software under the GNU General Public License,
# version 3 or (at your option) any later version.
"""Shared regions for the eccentric cell: every mode's correlations over a region of templates from one phase grid.

The circular PE's shared regions (:class:`pycbc.filter.nufft_correlation.PhaseGrid` over a
:class:`PhaseNUFFTCorrelation`) rest on the template's phase change being linear in a few coefficients. An
EccentricModeFamily mode ``k`` (phase scale ``r_k``) at ``theta = (Mc, e, chi, q)`` near a region reference
``theta_0`` is

    b_k(theta) ~ b_k(theta_0) a(theta) exp(-i [r_k dpsi_PN(f; dC) + sum_j dtheta_j D_kj(f)]),

where ``dpsi_PN`` is the TaylorF2 basis phase difference (exact: the cell takes the circular phase change from
the same basis), linear in the coefficient changes ``dC = (A - A0, B - B0)``. The cell's eccentric correction is a
tensor-product polynomial, linear in its stencil values, ``corr_k(x) = W(x) . V_k``; its change over the region,
``(W(x) - W(x0)) . V_k``, is spanned to ``basis_tolerance`` by a few leading vectors ``U_k`` of the region's
training templates, with coefficients ``(W(x) - W(x0)) . P_k``. The phase is therefore linear in ``c = (r_k dC,
e_k)`` over 16 + rank basis functions, and the mode's clock ``t_k = t_k0 + dpsi' / 2 pi`` too, so each mode is a
:class:`PhaseNUFFTCorrelation` of its own basis and The amplitude change is the cell's own band-averaged ratio, also linear in ``W(x)`` (a scalar per template and
mode: an eccentric harmonic's amplitude goes as e^|n-2|); what
the linearisation and the fixed amplitude miss is measured against the direct engine, not assumed.
"""
import numpy

from pycbc.filter.nufft_correlation import (PhaseGrid, PhaseNUFFTCorrelation, SkyBasis, source_power_weights,
                                            _basis, _host)
from pycbc.waveform.taylorf2_jets import taylorf2_basis_coefficients


class EccentricModeView(object):
    """One mode of an eccentric cell at a region reference, with the interface
    :class:`SkyBasis` and :class:`PhaseNUFFTCorrelation` read from a family:
    ``b0``/``t0`` (the mode's whitened reference waveform and clock on the
    grid), ``orders``, ``R0`` (for shapes), ``responses``, ``fine``,
    ``_index``/``_weight``/``coarse_frequencies`` and ``f_lower``."""
    def __init__(self, block_model, theta0, k, sky_response, f_lower, coarse_frequencies):
        xp = self.xp = block_model.xp
        self.grid = block_model.grid
        b, t = block_model.source_builder(numpy.asarray(theta0, dtype=float))
        self.b0 = xp.asarray(b[k])
        self.t0 = xp.asarray(t[k], dtype=numpy.float64)
        self.orders = numpy.asarray(block_model.orders)
        self._sky = sky_response
        self.R0 = xp.asarray(sky_response(numpy.zeros(2)))
        self._index, self._weight = block_model._index, block_model._weight
        self.coarse_frequencies = numpy.asarray(coarse_frequencies)
        self.f_lower = float(f_lower)
        self.sky_block = None

    def fine(self, values):
        i, w = self._index, self._weight
        return values[..., i] * (1 - w) + values[..., i + 1] * w

    def responses(self, sky_offsets):
        xp = self.xp
        pts = numpy.atleast_2d(numpy.asarray(sky_offsets, dtype=float))
        return xp.stack([xp.asarray(self._sky(p)) for p in pts])

    def _response(self, sky_offsets):
        return self.xp.asarray(self._sky(numpy.asarray(sky_offsets, dtype=float)))


class EccentricModePhaseCorrelation(PhaseNUFFTCorrelation):
    """:class:`PhaseNUFFTCorrelation` for one eccentric mode: its phase basis is
    the 16 TaylorF2 functions plus the mode's 4 eccentric-correction
    derivatives ``D_kj`` (numerical, on the grid), with coefficients ``c =
    (r_k dC, dtheta)``: the cell's mode phase is psi0 + r_k dpsi_PN + corr, its
    circular change dpsi_PN = +dC . psi(f) less the pins, and the waveform
    amplitude exp(-i psi), as the parent's template b0 exp(-i dC . psi).
    Strengths, sky weights and node rows are the parent's.

    ``family`` is the cell's EccentricModeFamily, ``view`` an
    :class:`EccentricModeView` of mode ``key`` at ``theta0 = (Mc, e, chi, q)``,
    ``data`` the whitened SSB data spectrum ``(nchannels, N)``, ``sky_points``
    the region's sky offsets for the sky basis.
    """
    def __init__(self, family, key, view, theta0, data, sky_points, training, sky_tolerance=1e-4, eps=1e-6,
                 window=None, band_margin=5 * 86400.0, basis_tolerance=1e-3, max_training=400, max_rank=12):
        from scipy.interpolate import CubicSpline
        self.family = view
        xp = self.xp = view.xp
        self.eps = float(eps)
        # band_margin (as PhaseNUFFTCorrelation's): only the frequencies whose reference stationary time lies within
        # the data window (``window``, s, default the block's observation) +- the margin carry the correlation; the
        # rest set the transform's extents (the cell's band spans about -1 to 3 yr of a 2 yr window)
        self._sel = None
        if band_margin is not None:
            lo_t, hi_t = window
            t0 = _host(view.t0)
            self._sel = numpy.nonzero((t0 >= lo_t - band_margin) & (t0 <= hi_t + band_margin)
                                      & (_host(abs(view.b0)) > 0))[0]
        self.f_lower = view.f_lower
        self.orders = view.orders
        self.sky = numpy.zeros(2)
        self._data = xp.asarray(data)
        self._power = source_power_weights(view)
        self.sky_basis = SkyBasis(view, sky_points, sky_tolerance, self._power)
        self._rho = list(self.sky_basis)
        self.sky_sets = len(self._rho)
        self._rho_at = view._response(self.sky)
        f_all = numpy.asarray(view.grid.frequencies, dtype=float)
        f = f_all if self._sel is None else f_all[self._sel]
        self.f = f
        self._logf = xp.asarray(numpy.log(f))
        self.theta0 = numpy.asarray(theta0, dtype=float)
        self.r = float(family._expansions[key][0])
        # the eccentric correction and amplitude are the cell's tensor-product polynomials, linear in the stencil
        # values: corr(x) = W(x) . V. The region's changes (W(x) - W(x0)) . V span a low-rank space: its leading
        # vectors U (weighted by template power over the observed band) are the basis, and a template's coefficients
        # are (W(x) - W(x0)) . P, exact but for the rank truncation (basis_tolerance, rms rad over the training
        # templates); the amplitude ratio and the (2,2,2) clock's slope at f_ref are linear in W(x) too
        from scipy.interpolate import CubicSpline
        self.family_cell = family
        self.key = key
        r_, amp, corr, psi0, have, sl = family._expansions[key]
        fx = family.f[sl][have]
        self._x0 = self._offsets(family, self.theta0)
        self._W0 = corr.weights(*self._x0)
        step_c = max(1, len(f_all) // 4000)
        coarse = slice(0, len(f_all), step_c)
        power = self.template_power()[coarse]
        if self._sel is not None:
            observed = numpy.zeros(len(f_all), dtype=bool)
            observed[self._sel] = True
            power = power * observed[coarse]
        self._fc = f_all[coarse]
        self._wc = power / power.sum()
        wx = numpy.interp(fx, self._fc, self._wc, left=0., right=0.)
        wx = wx / max(wx.sum(), 1e-300)
        V = numpy.asarray(corr.stacked, dtype=float)[:, have]                 # (S, nh)
        training = numpy.atleast_2d(numpy.asarray(training, dtype=float))
        if len(training) > max_training:
            training = training[numpy.random.default_rng(1).choice(len(training), max_training, replace=False)]
        X = corr.weights_many(self._offsets_many(family, training)) - self._W0[None]
        delta = (X @ V) * numpy.sqrt(wx)[None]
        u, sv, vt = numpy.linalg.svd(delta, full_matrices=False)
        # rank: the largest training residual rms below basis_tolerance. The residual after k vectors is the sum of
        # the rest's squares, accumulated from the tail: the total less the leading part cancels to about
        # sqrt(eps) |delta|, which over a full-q region (changes of 1e4 rad) exceeded the tolerance and kept every
        # training vector (rank 400 for (2,2,4), a grid of 1e19 points)
        c = (u * sv[None]) ** 2
        tail = numpy.cumsum(c[:, ::-1], axis=1)[:, ::-1]
        left = numpy.sqrt(numpy.concatenate([tail[:, 1:], numpy.zeros((len(c), 1))], axis=1))
        good = numpy.nonzero(left.max(axis=0) <= basis_tolerance)[0]
        rank = int(good[0]) + 1 if len(good) else len(sv)
        rank = min(rank, int(max_rank))
        self.basis_rank = rank
        self.basis_residual = float(left[:, rank - 1].max()) if rank else 0.
        # the vectors as combinations of the training changes themselves (orthonormal in wx; bounded where the
        # weight is small, which vt / sqrt(wx) is not, and the grid's extents read the basis everywhere)
        U = (u[:, :rank] / sv[None, :rank]).T @ (X @ V)                        # (rank, nh)
        # P: coefficients = (W(x) - W0) . P, the weighted projection of each stencil row onto U
        self._P = V @ (U * wx[None]).T                                          # (S, rank)
        D, Dp = numpy.zeros((rank, len(f))), numpy.zeros((rank, len(f)))
        D_all = numpy.zeros((rank, len(self._fc)))
        inside = (f >= fx[0]) & (f <= fx[-1])
        inside_c = (self._fc >= fx[0]) & (self._fc <= fx[-1])
        if rank:
            spline = CubicSpline(fx, U, axis=1, extrapolate=False)
            D[:, inside] = spline(f[inside])
            Dp[:, inside] = spline(f[inside], 1)
            D_all[:, inside_c] = spline(self._fc[inside_c])
        self._D, self._Dp = xp.asarray(D), xp.asarray(Dp)
        # the clock: every mode is anchored to the (2,2,2) mode's time at f_ref, which moves with that mode's
        # correction slope there, (W(x) - W0) . S
        c222 = family._expansions[(2, 2, 2)]
        f222 = family.f[c222[5]][c222[4]]
        self._S = CubicSpline(f222, numpy.asarray(c222[2].stacked, dtype=float)[:, c222[4]], axis=1)(
            self.f_lower, 1)
        self._W0_222 = c222[2].weights(*self._x0)
        # the amplitude ratio: the template-power weighted mean of a(x) / a(x0) over the band, W(x) . Ra
        A = numpy.asarray(amp.stacked)[:, 0][:, have]
        a0 = amp.weights(*self._x0) @ A
        ok = numpy.abs(a0) > 0
        self._Ra = (A[:, ok] / a0[ok][None]) @ wx[ok] / max(wx[ok].sum(), 1e-300)
        self._amp_weights = amp.weights_many
        self._corr_weights = corr.weights_many
        self._222_weights = c222[2].weights_many
        psi_c, _ = _basis(self._fc)
        # rows: 16 PN, rank eccentric, one zero row carrying the anchor slope's change into the time pin
        self._psic = numpy.vstack([psi_c, D_all, numpy.zeros((1, len(self._fc)))])
        self._A0, self._B0, self._la0 = taylorf2_basis_coefficients(*self._pn_params(self.theta0))

    @staticmethod
    def _offsets_many(family, thetas):
        """The cell's offsets ``(d, de, dq, dchi)`` of rows ``(Mc, e, chi, q)``: (M, 4)."""
        t = numpy.atleast_2d(numpy.asarray(thetas, dtype=float))
        ref = family.reference
        mc_ref = (ref['mass1'] * ref['mass2']) ** .6 / (ref['mass1'] + ref['mass2']) ** .2
        return numpy.column_stack([numpy.log(t[:, 0] / mc_ref), t[:, 1] - ref['eccentricity'],
                                   numpy.log(t[:, 3] / family.q0), t[:, 2] - ref.get('spin1z', 0.)])

    @staticmethod
    def _offsets(family, theta):
        """The cell's offsets ``(d, de, dq, dchi)`` of ``theta = (Mc, e, chi, q)``."""
        ref = family.reference
        mc_ref = (ref['mass1'] * ref['mass2']) ** .6 / (ref['mass1'] + ref['mass2']) ** .2
        return numpy.array([numpy.log(theta[0] / mc_ref), theta[1] - ref['eccentricity'],
                            numpy.log(theta[3] / family.q0), theta[2] - ref.get('spin1z', 0.)])

    @staticmethod
    def _pn_params(theta):
        mc, e, chi, q = theta
        return mc, q, chi, chi

    def _coefficients(self, thetas):
        """``c`` (M, 16 + rank + 1): ``r dC``, the eccentric coefficients, the anchor slope's change; and the
        amplitude ratios."""
        fam = self.family_cell
        thetas = numpy.atleast_2d(numpy.asarray(thetas, dtype=float))
        dC = []
        for theta in thetas:
            A, B, la = taylorf2_basis_coefficients(*self._pn_params(theta))
            dC.append(numpy.concatenate([A - self._A0, B - self._B0]))
        X = self._offsets_many(fam, thetas)
        ecc = (self._corr_weights(X) - self._W0[None]) @ self._P
        slope = (self._222_weights(X) - self._W0_222[None]) @ self._S
        amp = self._amp_weights(X) @ self._Ra
        return numpy.column_stack([self.r * numpy.array(dC), ecc, slope]), amp

    def _pins(self, c):
        psi_l, dpsi_l = _basis([self.f_lower])
        n = c.shape[1] - 17
        # the anchor slope's change moves the clock: a time shift pinned at f_ref, phase slope (f - f_ref), so its
        # value pin is f_ref times it (without it the region's correlations had a phase 2 pi f_ref dt common to
        # every mode, dt the clock's shift from the region's reference: up to 2 rad at one posterior width in q
        # or chi; ln B did not see it, the drawn coa_phase did)
        p0 = numpy.concatenate([psi_l[:, 0], numpy.zeros(n), [self.f_lower]])
        p1 = numpy.concatenate([dpsi_l[:, 0], numpy.zeros(n), [1.]])
        return c @ p0, c @ p1

    def _projected_basis(self, D):
        """``D^T Psi(f)`` and ``D^T Psi'(f)`` on the grid for coefficient directions ``D`` (16 + rank + 1, ndir);
        the last row (the anchor slope) has no phase."""
        x, dx = PhaseNUFFTCorrelation._projected_basis(self, D[:16])
        w = self.xp.asarray(D[16:-1])
        return x + w.T @ self._D, dx + w.T @ self._Dp

    def project(self, thetas, frame):
        thetas = numpy.atleast_2d(numpy.asarray(thetas, dtype=float))
        c, amp = self._coefficients(thetas)
        s = (c @ frame['root']) @ frame['vs'].T
        remainder = c - s @ frame['directions'].T
        line = remainder @ frame['coef'].T
        pin0, pin1 = self._pins(c)
        line_dir = frame['line_dir']
        shift = (line[:, 1] + s @ line_dir[:, 1] - pin1) / (2 * numpy.pi)
        const = numpy.exp(1j * (line[:, 0] + s @ line_dir[:, 0] - pin0)) * amp
        return s, shift, const

    def time_offsets(self, thetas):
        raise NotImplementedError

    def release(self):
        """Drop the per-frequency arrays only a grid's build needs (data, strengths, derivatives, the view's
        reference waveform); projections, sky weights and errors keep working."""
        for name in ('_data', '_D', '_Dp', '_logf', '_strengths', '_template_power', '_rho', '_rho_at', '_power'):
            if hasattr(self, name):
                setattr(self, name, None)
        view = self.family
        view.b0 = view.t0 = None
        self.f = None

    def direct(self, thetas, delays):
        """The correlations term by term on the full grid, without a transform or a projection (for tests)."""
        from pycbc.tdi.ssb_yearly_harmonics import ORBITAL_ANGULAR_FREQUENCY
        xp = self.xp
        thetas = numpy.atleast_2d(numpy.asarray(thetas, dtype=float))
        c, amp = self._coefficients(thetas)
        out = xp.zeros((2, len(thetas), len(delays)), dtype=complex)
        df = xp.asarray(2 * numpy.pi * (self.f - self.f_lower))
        for j in range(len(thetas)):
            pin0, pin1 = self._pins(c[j:j + 1])
            phase, slope = self._projected_basis(c[j][:, None])
            phase, slope = phase[0], slope[0]
            for i, k in enumerate(self.orders):
                w = k * ORBITAL_ANGULAR_FREQUENCY
                arg = phase - w / (2 * numpy.pi) * slope
                strength = self.strength(i, -1)
                for n, delay in enumerate(delays):
                    total = arg + (delay - pin1[0] / (2 * numpy.pi)) * (df - w)
                    out[:, j, n] += (strength * xp.exp(1j * total)).sum(axis=-1)
            out[:, j] *= amp[j] * numpy.exp(-1j * pin0[0])
        return out


class EccentricSharedRegion(object):
    """The marginal log likelihood of
    :class:`pycbc.inference.models.eccentric_bstatistic.EccentricCellPE` for
    many points of a region at once: every mode's correlations from one
    :class:`PhaseGrid` per mode over the region (an
    :class:`EccentricModePhaseCorrelation` about the region's middle), each
    point's Grams from the block model's anchors (no mode spectra), then the
    model's own time quadrature (whole-lag 2F screen, oversampled samples,
    ln B where F is within ``bayes_margin`` of the best) and marginal.

    Parameters
    ----------
    model : EccentricCellPE
    region : list of dict
        Points spanning the region (the variable parameters).
    data_window : tuple
        The data's observation (s), for the correlators' band selection.
    family, keys : EccentricModeFamily, list
        The cell and the block model's mode keys, in its order (default: the
        block model's ``source_builder.family`` and ``.keys``).
    band_margin : float
        See :class:`EccentricModePhaseCorrelation` (s).
    sky_tolerance : float
        The relative response error allowed on the region's sky basis; a
        point beyond it is not :meth:`inside`.
    edge_margin : float
        A point's F must fall this far below its largest at both ends of the
        lags its grid times cover (unless they are the model's own ends), or
        its log likelihood is ``nan`` and the model computes it directly.
    probes : int
        Points of the region whose direct likelihood sets the grid's times.
    """
    def __init__(self, model, region, data_window, family=None, keys=None, band_margin=60 * 86400.0,
                 sky_tolerance=1e-4, edge_margin=30.0, probes=8, grid_options=None, seed=0):
        import logging
        import time
        tick = time.time()
        self.model = model
        block = model.block
        self.xp = block.xp
        self.data_window = tuple(float(v) for v in data_window)
        self.sky_tolerance = float(sky_tolerance)
        self.edge_margin = float(edge_margin)
        thetas, skies = self._coordinates(region)
        # the region's middle: its componentwise median
        theta0 = numpy.median(thetas, axis=0)
        family = block.source_builder.family if family is None else family
        self.keys = list(block.source_builder.keys if keys is None else keys)
        dt = block.grid.delta_t
        self.dt = dt
        lo, hi = model.delay_window
        self.lags = numpy.arange(int(numpy.ceil(lo / dt)), int(numpy.floor(hi / dt)))
        # one response per sky point per region, shared by every mode's view (sky weights and errors ask for it
        # per mode and per call)
        self._responses = {}
        builder = block.response_builder

        def response(sky):
            key = (round(float(sky[0]), 12), round(float(sky[1]), 12))
            if key not in self._responses:
                # about 0.14 MB a point (17 harmonics x 2 x 2 x 256): a chunk's worth, not a region's
                if len(self._responses) >= 2048:
                    self._responses.clear()
                self._responses[key] = numpy.asarray(builder(numpy.asarray(sky, dtype=float)))
            return self._responses[key]
        self._response_function = response
        # the probes first (the direct engine, all modes on the full grid), their device memory released before
        # the modes' correlators take theirs: with seven modes both together overran 6 GB
        rng = numpy.random.default_rng(seed)
        pick = rng.choice(len(region), min(int(probes), len(region)), replace=False)
        curves, values = [], []
        for j in pick:
            ev = model.evaluate(region[j])
            curves.append((ev['delays'], ev['log_bayes']))
            values.append(float(ev['loglikelihood']))
        model._last = None
        model.block._prepared = None
        if self.xp is not numpy:
            self.xp.get_default_memory_pool().free_all_blocks()
        tops = numpy.array([v[1].max() for v in curves])
        # the region's largest log likelihood seen: a point far below it needs no exact arrival-time edges
        self.best_loglikelihood = max(values)
        good = numpy.nonzero(tops >= tops.max() - self.edge_margin)[0]
        logging.info("EccentricSharedRegion: %d probes in %.1f s", len(pick), time.time() - tick)
        # the modes one at a time, the reference (2,2,2) first: its correlator, frame, shift and grid, the
        # correlator's device arrays released before the next mode's (all seven modes' correlators at once took
        # 5 GB, leaving 0.36 GB for a grid that needs 0.46)
        options = dict(grid_options or {})
        keys = [tuple(int(v) for v in key) for key in self.keys]
        ref = keys.index((2, 2, 2)) if (2, 2, 2) in keys else 0
        K = len(self.keys)
        self.views, self.correlators, self.grids = [None] * K, [None] * K, [None] * K
        margin = float(options.get('margin', 0.05))
        shifts = [None] * K
        tlo = thi = base_pad = None
        # the grid times the region needs (per mode, T = delay + shift). The posterior's arrival time is set by the
        # dominant (2,2,2) mode, whose T at the peak hardly moves over a region (170 s over a posterior where its
        # shift moves 6000 s): its window is where the probes' ln B lies within edge_margin of each one's largest,
        # for the probes within it of the best, widened by itself and eight lags. Another mode's T at the same delay
        # is the reference's plus shift_k - shift_ref, which moves by up to 3e4 s over a posterior (each mode's
        # frame splits phase and time its own way): its window is the reference's moved by that difference's range
        # over the region's points, widened by the grid margin of it. From the probes alone (8 points) those
        # windows missed most of a posterior that reached past a frame ball's spread: no lags on the weak modes'
        # grids for 90% of the posterior points (timecheck0, fig10k)
        for k in [ref] + [j for j in range(K) if j != ref]:
            key = self.keys[k]
            view = EccentricModeView(block, theta0, k, response, family.f_ref,
                                     block.response_frequencies)
            corr = EccentricModePhaseCorrelation(family, key, view, theta0, block.data, skies, thetas,
                                                 sky_tolerance=self.sky_tolerance, window=self.data_window,
                                                 band_margin=band_margin)
            frame = corr.frame(thetas, 3)
            self.views[k], self.correlators[k] = view, corr
            shifts[k] = corr.project(thetas, frame)[1]
            if k == ref:
                tlo, thi = numpy.inf, -numpy.inf
                for i in good:
                    d, v = curves[i]
                    T = d[v >= tops[i] - self.edge_margin] + shifts[ref][pick[i]]
                    tlo, thi = min(tlo, T.min()), max(thi, T.max())
                base_pad = (thi - tlo) + 8 * dt
            diff = shifts[k] - shifts[ref]
            dmin, dmax = float(diff.min()), float(diff.max())
            pad = base_pad + margin * (dmax - dmin)
            reach = (self.lags[0] * dt + shifts[k].min() - dt, self.lags[-1] * dt + shifts[k].max() + 2 * dt)
            window = (max(tlo + dmin - pad, reach[0]), min(thi + dmax + pad, reach[1]))
            self.grids[k] = PhaseGrid(corr, thetas, (lo, hi), time_range=window, **options)
            g = self.grids[k]
            logging.info("EccentricSharedRegion: mode %d grid times %.0f to %.0f s (step %.1f s, %d cells); "
                         "the region's whole lag reach %.0f to %.0f s", k, window[0], window[1], g.step[-1],
                         g.size[-1], reach[0], reach[1])
            corr.release()
            if self.xp is not numpy:
                self.xp.get_default_memory_pool().free_all_blocks()
        logging.info("EccentricSharedRegion: %d points, correlators, frames and grids in %.1f s; eccentric ranks %s "
                     "(training residual rms %s rad)", len(region), time.time() - tick,
                     [c.basis_rank for c in self.correlators],
                     ['%.1e' % c.basis_residual for c in self.correlators])
        model.block._prepared = None
        logging.info("EccentricSharedRegion: grids %s in %.1f s", [list(g.size) for g in self.grids],
                     time.time() - tick)

    def _coordinates(self, points):
        pairs = [self.model._theta_sky(p) for p in points]
        return numpy.array([t for t, _ in pairs]), numpy.array([s for _, s in pairs])

    def inside(self, points):
        """Whether each point lies in the region every mode's grid covers, with its response on the sky basis
        (``point_chunk`` points at a time: the sky error forms every point's response)."""
        import logging
        if len(points) > self.point_chunk:
            ok = numpy.concatenate([self._inside(points[c:c + self.point_chunk])
                                    for c in range(0, len(points), self.point_chunk)])
            logging.info("EccentricSharedRegion.inside: %d of %d", ok.sum(), len(points))
            return ok
        return self._inside(points)

    def _inside(self, points):
        import logging
        thetas, skies = self._coordinates(points)
        ok = numpy.ones(len(points), dtype=bool)
        reasons = {}
        # the point's whole lags whose grid times every mode's grid holds, as _evaluate finds them: a point with none
        # passed every other check and then went directly (posterior points whose arrival-time shift lies past the
        # region's)
        L = len(self.lags)
        first = numpy.zeros(len(points), dtype=int)
        last = numpy.full(len(points), L - 1)
        for k, grid in enumerate(self.grids):
            s, shift, _ = grid.correlator.project(thetas, grid.frame)
            t_lo, t_hi = grid.time_limits()
            first = numpy.maximum(first, numpy.ceil((t_lo - shift) / self.dt - self.lags[0]).astype(int))
            last = numpy.minimum(last, numpy.floor((t_hi - self.dt - shift) / self.dt - self.lags[0]).astype(int))
            hw = grid.half_width
            on = numpy.ones(len(points), dtype=bool)
            for d in range(grid.ndir):
                lo = grid.start[d] + hw[d] * grid.step[d]
                hi = grid.start[d] + (grid.size[d] - hw[d] - 1) * grid.step[d]
                on &= (s[:, d] >= lo) & (s[:, d] <= hi)
            taylor = numpy.ones(len(points), dtype=bool)
            if grid.terms > 1:
                taylor = numpy.abs(s[:, grid.ndir] - grid.middle) <= grid.s3_half
            phase = grid.correlator.phase_residual(thetas, grid.frame) <= grid.phase_tolerance
            reasons['off_grid_%d' % k] = int((~on).sum())
            reasons['taylor_%d' % k] = int((~taylor).sum())
            reasons['phase_%d' % k] = int((~phase).sum())
            ok &= on & taylor & phase
        has_lags = last - first >= 2
        reasons['no_lags'] = int((~has_lags).sum())
        if len(points) >= 100 and reasons['no_lags']:
            # where the lost points' lags lie against the grids' times (in lags of L; first > L - 1: the grids'
            # times all lie after the point's window, last < 0: all before; both inside: the modes disagree)
            lost = ~has_lags & ok
            if lost.any():
                logging.info("EccentricSharedRegion.inside: no_lags (otherwise inside) %d; first %s, last %s of %d",
                             int(lost.sum()), numpy.percentile(first[lost], [0, 10, 50, 90, 100]).tolist(),
                             numpy.percentile(last[lost], [0, 10, 50, 90, 100]).tolist(), L)
        ok &= has_lags
        if ok.any():
            sky = self.correlators[0].sky_error(skies[ok]) <= self.sky_tolerance
            reasons['sky'] = int((~sky).sum())
            ok[ok] = sky
        if len(points) >= 100 and not ok.all():
            logging.info("EccentricSharedRegion.inside: %d of %d; outside by %s", ok.sum(), len(points),
                         {k: v for k, v in reasons.items() if v})
        return ok

    def _z(self, thetas, skies, delays):
        """``(M, K, 2, L)`` correlations at ``delays`` ``(M, L)`` (the model's convention: the carrier
        ``exp(2 pi i f_lower tau)`` the transform leaves out restored)."""
        xp = self.xp
        out = []
        for corr, grid in zip(self.correlators, self.grids):
            z = grid.evaluate(thetas, delays, skies)                         # (2, M, L)
            carrier = xp.exp(2j * numpy.pi * corr.f_lower * xp.asarray(delays))
            out.append(xp.moveaxis(xp.asarray(z, dtype=xp.complex128) * carrier[None], 0, 1))
        return xp.stack(out, axis=1)

    def _template_grams(self, thetas, skies):
        """``(M, ..., A)``: each point's complete Grams at the block model's anchor delays, as its
        ``template_gram`` gives them. For Grams anchored in sky, q, chi and ln Mc, one weight matrix over the
        anchors (kept on the device) for all points; otherwise point by point."""
        xp, block = self.xp, self.model.block
        if (block.gram_sky_anchors is None or block.gram_mc_anchors is None or block.gram_chi_anchors is None
                or block.gram_q_anchors is None):
            grams = []
            for th, sk in zip(thetas, skies):
                atimes, g = block.template_gram(th, sk)
                grams.append(g)
            self._anchor_times = numpy.asarray(atimes, dtype=float)
            return xp.stack(grams)
        lo, hi = block.delay_range
        jlo, jhi = int(numpy.ceil(lo / block.grid.delta_t)), int(numpy.floor(hi / block.grid.delta_t))
        anchor_indices = numpy.unique(numpy.linspace(jlo, jhi, block.gram_anchors).round().astype(int))
        self._anchor_times = anchor_indices * block.grid.delta_t
        if not hasattr(self, '_anchor_grams'):
            self._anchor_grams, self._anchor_order = {}, []

        def axis(anchors, v, what):
            if anchors is None:
                return [(None, 1.0)]
            if v < anchors[0] - 1e-9 or v > anchors[-1] + 1e-9:
                raise ValueError('template %s lies outside the Gram anchors' % what)
            i = int(numpy.clip(numpy.searchsorted(anchors, v, side='right') - 1, 0, len(anchors) - 2))
            w = min(max((v - anchors[i]) / (anchors[i + 1] - anchors[i]), 0.), 1.)
            return [(a, b) for a, b in ((anchors[i], 1 - w), (anchors[i + 1], w)) if b != 0]
        weights = []
        for th, sk in zip(thetas, skies):
            row = {}
            q = axis(block.gram_q_anchors, numpy.log(th[3] / block.q_reference) if block.gram_q_anchors is not None
                     else 0., 'mass ratio')
            c = axis(block.gram_chi_anchors, th[2] - block.coordinates.spin if block.gram_chi_anchors is not None
                     else 0., 'spin')
            x = axis(block.gram_mc_anchors, numpy.log(th[0] / block.coordinates.mchirp), 'chirp mass')
            for sx, wx in axis(block.gram_sky_anchors, float(sk[0]), 'sky'):
                for sy, wy in axis(block.gram_sky_anchors, float(sk[1]), 'sky'):
                    for qa, wq in q:
                        for ca, wc in c:
                            for xa, wm in x:
                                key = (float(sx), float(sy), qa, ca, float(xa))
                                row[key] = row.get(key, 0.) + wx * wy * wq * wc * wm
            weights.append(row)
        for row, th in zip(weights, thetas):
            for key in row:
                if key not in self._anchor_grams:
                    sx, sy, qa, ca, xa = key
                    g = block._mc_anchor_gram(xa, th, (sx, sy), anchor_indices, (jlo, jhi),
                                              block._response((sx, sy)), ca, qa)
                    self._anchor_grams[key] = xp.asarray(g)
                    self._anchor_order.append(key)
        index = {k: i for i, k in enumerate(self._anchor_order)}
        W = numpy.zeros((len(thetas), len(index)))
        for j, row in enumerate(weights):
            for key, w in row.items():
                W[j, index[key]] = w
        used = numpy.flatnonzero(W.any(axis=0))
        G = xp.stack([self._anchor_grams[self._anchor_order[i]] for i in used])
        return xp.tensordot(xp.asarray(W[:, used]), G, axes=(1, 0))

    #: samples per ln B call (its orientation quadrature holds n_l0 x nodes values per sample)
    bayes_chunk = 256

    def _log_bayes(self, z, g, which):
        """The model's ln B at samples ``which`` of ``z`` (K, 2, S) and ``g`` (..., S), in chunks."""
        xp = self.xp
        out = []
        for c in range(0, len(which), self.bayes_chunk):
            sel = xp.asarray(which[c:c + self.bayes_chunk])
            # the screen's 2F (mean_anomaly_twof_analytic) has checked these Grams already
            v = self.model._log_bayes(z[..., sel], g[..., sel], validate=False)[0]
            out.append(v.get() if hasattr(v, 'get') else numpy.asarray(v))
        return numpy.concatenate(out) if out else numpy.zeros(0)

    def _gram_at(self, stack, rows, delays):
        """Each sample's Grams: ``stack`` ``(M, ..., A)`` the points' anchor Grams, ``rows`` the sample's point,
        ``delays`` its delay; linear in delay between anchors (as ``WindowedModeFFT.interpolated_gram``)."""
        xp = self.xp
        anchors = self._anchor_times
        delays = numpy.asarray(delays, dtype=float)
        if numpy.any(delays < anchors[0]) or numpy.any(delays > anchors[-1]):
            raise ValueError('Gram interpolation would extrapolate')
        at = numpy.clip(numpy.searchsorted(anchors, delays, side='right') - 1, 0, len(anchors) - 2)
        w = (delays - anchors[at]) / (anchors[at + 1] - anchors[at])
        moved = xp.moveaxis(stack, -1, 1)                                    # (M, A, ...)
        r, a = xp.asarray(rows), xp.asarray(at)
        wv = xp.asarray(w).reshape((-1,) + (1,) * (moved.ndim - 2))
        out = moved[r, a] * (1 - wv) + moved[r, a + 1] * wv                  # (S, ...)
        return xp.moveaxis(out, 0, -1)

    def at_delays(self, points, delays):
        """Every mode's correlations ``(K, 2, M)`` and complete Grams ``(K, 2, K, 2, M)`` of each of ``points``
        (inside the region) at one delay each (s), ``point_chunk`` points at a time."""
        xp = self.xp
        zs, gs = [], []
        delays = numpy.asarray(delays, dtype=float)
        for c in range(0, len(points), self.point_chunk):
            part = points[c:c + self.point_chunk]
            thetas, skies = self._coordinates(part)
            d = delays[c:c + len(part)]
            z = self._z(thetas, skies, d[:, None])[..., 0]                   # (M, K, 2)
            stack = self._template_grams(thetas, skies)
            g = self._gram_at(stack, numpy.arange(len(part)), d)              # (K, 2, K, 2, M)
            zs.append(xp.moveaxis(z, 0, -1))
            gs.append(g)
            del stack
        return xp.concatenate(zs, axis=-1), xp.concatenate(gs, axis=-1)

    def __call__(self, points):
        """The log likelihood of each point; ``nan`` where its F has not fallen by ``edge_margin`` at the ends of
        the lags its grid times cover. The correlations, Grams and statistics of ``point_chunk`` points at a time
        are batched."""
        out = [self._evaluate_halving(points[c:c + self.point_chunk])
               for c in range(0, len(points), self.point_chunk)]
        return numpy.concatenate(out) if out else numpy.zeros(0)

    def _evaluate_halving(self, points):
        """:meth:`_evaluate`, the points halved while the device runs out of memory: the fine samples grow with the
        points' screened lags, which a chunk of points far from the peak can hold many of."""
        try:
            return self._evaluate(points)
        except Exception as err:
            if type(err).__name__ != 'OutOfMemoryError' or len(points) < 2:
                raise
        import gc
        import logging
        gc.collect()
        if self.xp is not numpy:
            self.xp.get_default_memory_pool().free_all_blocks()
        logging.info("EccentricSharedRegion: %d points out of device memory, halved", len(points))
        half = len(points) // 2
        return numpy.concatenate([self._evaluate_halving(points[:half]), self._evaluate_halving(points[half:])])

    #: points evaluated together (device memory grows with points x lags)
    point_chunk = 512

    def evaluate(self, point):
        """For one point, what :meth:`EccentricCellPE.evaluate` returns (``loglikelihood``, the quadrature's
        ``delays``, ``widths``, ``log_bayes``, and the ``z`` and ``gram`` of its active samples), from the region;
        ``None`` where its log likelihood would be ``nan``."""
        out = self._evaluate([point], keep=True)
        return out[0] if out else None

    def _evaluate(self, points, keep=False, rngs=None):
        from pycbc.filter.eccentric_fast_search import mean_anomaly_twof_analytic
        m, xp = self.model, self.xp
        host = lambda a: a.get() if hasattr(a, 'get') else numpy.asarray(a)
        thetas, skies = self._coordinates(points)
        M = len(thetas)
        dt, lags = self.dt, self.lags
        L = len(lags)
        var = m.noise_variance
        offsets = m.block.offsets
        # each point's whole lags whose grid times every mode's grid holds
        first = numpy.zeros(M, dtype=int)
        last = numpy.full(M, L - 1)
        for corr, grid in zip(self.correlators, self.grids):
            _, shift, _ = corr.project(thetas, grid.frame)
            t_lo, t_hi = grid.time_limits()
            first = numpy.maximum(first, numpy.ceil((t_lo - shift) / dt - lags[0]).astype(int))
            last = numpy.minimum(last, numpy.floor((t_hi - dt - shift) / dt - lags[0]).astype(int))
        out = numpy.full(M, numpy.nan)
        if M >= 100 and numpy.any(last - first < 2):
            import logging
            lost = last - first < 2
            logging.info("EccentricSharedRegion: lags first/last of points without lags: first %s, last %s (of %d)",
                         numpy.percentile(first[lost], [0, 50, 100]).tolist(),
                         numpy.percentile(last[lost], [0, 50, 100]).tolist(), L)
        idx = numpy.nonzero(last - first >= 2)[0]
        if not len(idx):
            return [None] * M if keep else out
        th, sk, fi, la = thetas[idx], skies[idx], first[idx], last[idx]
        Mi = len(idx)
        count = la - fi + 1
        W = int(count.max())
        col = numpy.arange(W)
        valid = col[None] < count[:, None]
        K = numpy.minimum(fi[:, None] + col[None], la[:, None])
        delays = lags[K] * dt                                                  # (Mi, W)
        # the points' anchor Grams (cached anchors: no mode spectra)
        stack = self._template_grams(th, sk)
        rows = numpy.repeat(numpy.arange(Mi), W)
        z = self._z(th, sk, delays)                                            # (Mi, K, 2, W)
        nk = z.shape[1]
        zs = xp.moveaxis(z, 0, 2).reshape(nk, 2, -1)
        gs = self._gram_at(stack, rows, delays.ravel())
        twof = host(m._twof(zs / var, gs / var, m.screen_anomalies)[0].max(axis=0)).reshape(Mi, W)
        del z, zs, gs
        F = numpy.where(valid, twof / 2, -numpy.inf)
        top = F.max(axis=1)
        r = numpy.arange(Mi)
        edge = ((F[:, 0] <= top - self.edge_margin) | (fi == 0)) & \
            ((F[r, count - 1] <= top - self.edge_margin) | (la == L - 1))
        screened = F >= top[:, None] - m.screen_margin
        screened = (screened | numpy.roll(screened, 1, axis=1) | numpy.roll(screened, -1, axis=1)) & valid
        # the oversampled samples of the kept lags
        P = m.oversampling
        frac = numpy.arange(P) / P
        fine = [((K[j][screened[j]][None, :] + lags[0] + frac[:, None]) * dt).ravel() for j in range(Mi)]
        nf = numpy.array([len(f) for f in fine])
        Wf = int(nf.max())
        fdel = numpy.array([numpy.concatenate([f, numpy.full(Wf - len(f), f[-1])]) for f in fine])
        fvalid = numpy.arange(Wf)[None] < nf[:, None]
        zf = self._z(th, sk, fdel)                                             # (Mi, K, 2, Wf)
        flat = numpy.flatnonzero(fvalid.ravel())
        frows = numpy.repeat(numpy.arange(Mi), Wf)[flat]
        zf = xp.moveaxis(zf, 0, 2).reshape(nk, 2, -1)[..., xp.asarray(flat)]
        gf = self._gram_at(stack, frows, fdel.ravel()[flat])
        del stack
        ftwof = host(m._twof(zf / var, gf / var, m.n_l0)[0].max(axis=0))
        # each point's best sample: its ln B sets which samples matter (F bounds ln B from above)
        starts = numpy.concatenate([[0], numpy.cumsum(nf)[:-1]])
        best_at = numpy.array([s0 + numpy.argmax(ftwof[s0:s0 + n]) for s0, n in zip(starts, nf)])
        best = self._log_bayes(zf, gf, best_at)
        active = numpy.flatnonzero(ftwof / 2 >= best[frows] - m.bayes_margin)
        lnb = self._log_bayes(zf, gf, active)
        terms = lnb + numpy.log(dt / P)
        owner = frows[active]
        lo, hi = m.delay_window
        tops = numpy.full(Mi, -numpy.inf)
        numpy.maximum.at(tops, owner, terms)
        sums = numpy.zeros(Mi)
        numpy.add.at(sums, owner, numpy.exp(terms - tops[owner]))
        value = tops + numpy.log(sums) - numpy.log(hi - lo)
        # a point whose ln B has not fallen off at the ends of its lags, but which, even with every lag of the
        # model's window as large as its largest, lies edge_margin below the region's best, keeps its value: its
        # posterior weight is below exp(-edge_margin) of the best either way
        bound = top + numpy.log(dt * L) - numpy.log(hi - lo)
        # only where the grid covers the point's whole window: elsewhere top is the largest F over the lags the grid
        # holds, and a peak outside them (a template whose arrival time lies past the grid's times) is no bound at
        # all (two posterior samples got ln B 11 and 20 from a region where the direct engine gives 110 and 106)
        whole = (fi == 0) & (la == L - 1)
        edge = edge | (whole & (bound < self.best_loglikelihood - self.edge_margin))
        out[idx] = numpy.where(edge, value, numpy.nan)
        if M >= 100 and (M > Mi or not edge.all()):
            import logging
            logging.info("EccentricSharedRegion: %d points, %d without lags on the grid, %d failing the edge check",
                         M, M - Mi, int((~edge).sum()))
        if not keep:
            return out
        fd = fdel.ravel()[flat]
        if keep == 'sample':
            # one arrival-time sample per point drawn by its ln B (the point's own generator in rngs), all at once:
            # the cumulative weights of the active samples, which come point by point, searched once; the drawn
            # samples' correlations and Grams gathered together (a record per point with all its samples cost a
            # device gather each)
            records = [None] * M
            ok = numpy.flatnonzero(edge)
            if not len(ok):
                return records
            w = numpy.exp(terms - tops[owner])
            cs = numpy.cumsum(w)
            first = numpy.searchsorted(owner, numpy.arange(Mi))
            last = numpy.searchsorted(owner, numpy.arange(Mi), side='right')
            base = numpy.where(first > 0, cs[numpy.maximum(first - 1, 0)], 0.)
            total = cs[numpy.maximum(last - 1, 0)] - base
            u = numpy.array([rngs[idx[j]].random() for j in ok])
            k = numpy.searchsorted(cs, base[ok] + u * total[ok])
            k = numpy.clip(k, first[ok], last[ok] - 1)
            jitter = numpy.array([rngs[idx[j]].random() for j in ok])
            sel = xp.asarray(active[k])
            zsel = zf[..., sel]
            zsel = zsel.get() if hasattr(zsel, 'get') else numpy.asarray(zsel)
            gsel = gf[..., sel]
            gsel = gsel.get() if hasattr(gsel, 'get') else numpy.asarray(gsel)
            for n, j in enumerate(ok):
                records[idx[j]] = dict(loglikelihood=float(value[j]), sample_delay=float(fd[active[k[n]]]),
                                       delay=float(fd[active[k[n]]] + jitter[n] * dt / P), z=zsel[..., n],
                                       gram=gsel[..., n], theta=th[j], sky=sk[j])
            return records
        # one record per point, in order; None where the region gives no value (no lags, or the edge check)
        records = [None] * M
        for j in range(Mi):
            if not edge[j]:
                continue
            mine = active[owner == j]
            sel = xp.asarray(mine)
            records[idx[j]] = dict(loglikelihood=float(value[j]), delays=fd[mine], widths=numpy.full(len(mine), dt / P),
                                   log_bayes=lnb[owner == j], z=zf[..., sel], gram=gf[..., sel],
                                   theta=th[j], sky=sk[j], shared=True)
        return records


class EccentricPhaseCoordinates(object):
    """Curvilinear coordinates ``(s_1 .. s_ndir, q)`` for the eccentric cell's PE, the eccentric counterpart of
    :class:`pycbc.waveform.taylorf2_jets.MeasuredPhaseCoordinates`: ``s`` are the projections of the cell's whole
    phase change on the ``ndir`` directions along which it changes most over a sample of points (:meth:`fit`).

    The phase change of mode ``k`` (phase scale ``r_k``) from the cell's reference is exactly (the cell's own
    evaluation, :meth:`EccentricModeFamily._windows`)

        dpsi_k(f; theta) = r_k dC(theta) . psi(f) + (W_k(x) - W_k(0)) . V_k(f)  (+ the pins: a constant and f),

    linear in the TaylorF2 coefficient changes ``dC`` and the cell polynomial's weights ``W_k`` at the offsets
    ``x = (ln Mc / Mc0, e - e0, ln q / q0, chi - chi0)``, the circular part pinned at ``f_ref`` before ``r_k``
    scales it. Stacked over the modes, weighted by each mode's power over the noise at the frequencies it sweeps
    while the data last (``sqrt w``), less what the marginalised phases (a constant and one proportional to
    ``r_k``) and a common arrival time absorb, it is a vector ``Phi(theta)`` whose norm is the rms phase change in radians;
    ``s = v . (Phi - mean)`` with ``v`` its leading right singular vectors over the sample. Points of equal ``s``
    have nearly equal phase and differ along ``q``, the fourth coordinate, which the phase barely fixes; the
    likelihood's degenerate valley is then nearly straight in ``y``. At ``e -> 0`` the eccentric part vanishes and
    ``s`` are the circular coordinates' projections of ``dC``.

    Parameters
    ----------
    family : EccentricModeFamily
        The cell (with ``step_q`` and ``step_chi``).
    noise_weight : callable
        ``noise_weight(f)``: the inverse noise power at ``f`` (any constant factor).
    data_window : tuple
        The data's start and end, s, on the clock that puts the ``(2,2,2)`` mode at ``f_ref`` at time 0.
    band_margin : float
        Seconds about the window still counted as observed.
    keys : list, optional
        The modes; all the cell's by default.
    ndir : int
        The number of phase directions; the coordinates are ``ndir + 1`` and the parameters four, so 3.
    coarse : int
        Frequencies kept per mode.
    guess : array_like, optional
        ``(mchirp, q, eccentricity, chi_eff)`` the inverse starts from (at each point's own ``q``); the
        cell's reference by default.
    """
    names = ('mchirp', 'q', 'eccentricity', 'chi_eff')

    def __init__(self, family, noise_weight, data_window, band_margin=60 * 86400., keys=None, ndir=3,
                 coarse=2000, guess=None, max_fit=4000):
        ref = family.reference
        self.family = family
        self.mc0 = (ref['mass1'] * ref['mass2']) ** .6 / (ref['mass1'] + ref['mass2']) ** .2
        self.q0 = float(family.q0)
        self.e0 = float(ref.get('eccentricity', 0.))
        self.chi0 = float(ref.get('spin1z', 0.))
        self.guess = numpy.array([self.mc0, self.q0, self.e0, self.chi0] if guess is None else guess, dtype=float)
        if ndir != 3:
            raise ValueError("four parameters: three phase directions and q")
        self.ndir = int(ndir)
        self.max_fit = int(max_fit)
        self.keys = sorted(family._expansions) if keys is None else list(keys)
        pieces = {k: (idx, t) for k, idx, _, _, t in family.iter_mode_pieces(0., 0., 0., dchi=0.)}
        i222, t222 = pieces[(2, 2, 2)]
        t_anchor = numpy.interp(family.f_ref, family.f[i222], t222)
        lo_t, hi_t = data_window
        self._C0 = numpy.concatenate(taylorf2_basis_coefficients(self.mc0, self.q0, self.chi0, self.chi0)[:2])
        f_all, sw_all, blocks, self._modes = [], [], [], []
        start = 0
        for key in self.keys:
            r, amp, corr, psi0, have, sl = family._expansions[key]
            fk = family.f[sl]
            hidx = numpy.flatnonzero(have)
            idx, tk = pieces[key]
            t_at = numpy.interp(fk[hidx], family.f[idx], tk - t_anchor)
            step = max(1, len(hidx) // coarse)
            sel = hidx[::step]
            t_at = t_at[::step]
            a0 = numpy.abs(numpy.asarray(amp(0., 0., 0., 0.))[0][sel]) ** 2
            w = a0 * noise_weight(fk[sel]) * step * ((t_at >= lo_t - band_margin) & (t_at <= hi_t + band_margin))
            V = numpy.asarray(corr.stacked, dtype=float)[:, sel]                 # (S, n_k)
            f_all.append(fk[sel])
            sw_all.append(w)
            self._modes.append(dict(key=key, r=float(r), block=slice(start, start + len(sel)), V=V, sel=sel,
                                    weights=corr.weights_many, W0=corr.weights(0., 0., 0., 0.)))
            start += len(sel)
        f = numpy.concatenate(f_all)
        w = numpy.concatenate(sw_all)
        w = w / w.sum()
        self.frequencies, self.weights = f, w
        sw = self._sw = numpy.sqrt(w)
        # the circular change is pinned in value and slope at f_ref (EccentricModeFamily.circular_phase_change)
        # before the mode's r_k scales it: the pins differ between modes, and set the modes' relative times
        psi, _ = _basis(f)
        psi_r, dpsi_r = _basis(numpy.array([family.f_ref]))
        psi = psi - psi_r - dpsi_r * (f - family.f_ref)[None]
        r_all = numpy.concatenate([numpy.full(m['block'].stop - m['block'].start, m['r']) for m in self._modes])
        self._G = (psi * (r_all * sw)[None]).T                                    # (nf, 16)
        for m in self._modes:
            m['V'] = m['V'] * sw[m['block']][None]
        # what the marginalised phases absorb, m phi + (n - m) l0 (for the (2,2,n) modes alone a constant and one
        # linear in the harmonic's r_k = n / 2; with other patterns, the span of m_k and n_k), and a common
        # arrival time
        m_all = numpy.concatenate([numpy.full(m['block'].stop - m['block'].start, float(m['key'][1]))
                                   for m in self._modes])
        n_all = numpy.concatenate([numpy.full(m['block'].stop - m['block'].start,
                                              float(m['key'][2]) if len(m['key']) > 2 else float(m['key'][1]))
                                   for m in self._modes])
        cols = [sw * m_all, sw * n_all, sw * f / f[w > 0].mean()]
        if numpy.all(m_all == m_all[0]) and m_all[0] != 0:
            cols = [sw, sw * r_all, sw * f / f[w > 0].mean()]
        self._Q = numpy.linalg.qr(numpy.array(cols).T)[0]
        self._vs = None

    def _offsets(self, mc, q, e, chi):
        return numpy.column_stack([numpy.log(mc / self.mc0), e - self.e0, numpy.log(q / self.q0), chi - self.chi0])

    def phases(self, theta):
        """``Phi`` (M, nf) at rows ``(mchirp, q, eccentricity, chi_eff)``: the weighted phase change, rad."""
        from pycbc.waveform.taylorf2_jets import basis_coefficients
        mc, q, e, chi = numpy.atleast_2d(numpy.asarray(theta, dtype=float)).T
        dC = basis_coefficients(mc, q, chi, chi).T - self._C0[None]
        raw = dC @ self._G.T
        X = self._offsets(mc, q, e, chi)
        for m in self._modes:
            raw[:, m['block']] += (m['weights'](X) - m['W0'][None]) @ m['V']
        return raw - (raw @ self._Q) @ self._Q.T

    def fit(self, theta, refit=False):
        """Set the directions from rows of (mchirp, q, eccentricity, chi_eff) (at most ``max_fit`` of them); a
        second call keeps the first fit unless ``refit`` (a sampler fits to the prior's draws again, which a caller
        that already fitted to the same prior need not pay for)."""
        if self._vs is not None and not refit:
            return self
        theta = numpy.atleast_2d(numpy.asarray(theta, dtype=float))
        theta = theta[numpy.all(numpy.isfinite(theta), axis=1)]
        if len(theta) > self.max_fit:
            theta = theta[numpy.random.default_rng(0).choice(len(theta), self.max_fit, replace=False)]
        P = self.phases(theta)
        mean = P.mean(axis=0)
        _, sv, vt = numpy.linalg.svd(P - mean, full_matrices=False)
        vs = vt[:self.ndir]
        # the rms phase change over the sample along each direction, and what the directions leave, rad
        self.singular_values = sv / numpy.sqrt(len(P))
        self.residual_rms = float(numpy.sqrt(numpy.sum(sv[self.ndir:] ** 2) / len(P)))
        # s = vs . (Phi - mean), with Phi = P_perp G z and vs in the range of P_perp: vs . G z
        self._M_pn = vs @ self._G
        for m in self._modes:
            m['M'] = vs[:, m['block']] @ m['V'].T
        self._s0 = vs @ mean + sum(m['M'] @ m['W0'] for m in self._modes)
        # the modes' polynomial parts summed into one tensor over the stencil's nodes, contracted with the 1-d
        # Lagrange weights one direction at a time (W(x) is their tensor product)
        poly = self.family._expansions[self._modes[0]['key']][2]
        self._nodes, self._steps = poly.nodes, poly.steps
        T = numpy.zeros((self.ndir,) + tuple(len(n) for n in self._nodes))
        for m in self._modes:
            p = self.family._expansions[m['key']][2]
            pos = tuple(numpy.searchsorted(numpy.asarray(n), p.index[:, d]) for d, n in enumerate(self._nodes))
            numpy.add.at(T, (slice(None),) + pos, m['M'])
        self._T = T
        self._vs = vs
        return self

    def _lagrange(self, X):
        out = []
        for d, (h, ks) in enumerate(zip(self._steps, self._nodes)):
            L = numpy.ones((len(X), len(ks)))
            if h > 0:
                t = X[:, d]
                for a, k in enumerate(ks):
                    for j in ks:
                        if j != k:
                            L[:, a] *= (t - j * h) / ((k - j) * h)
            out.append(L)
        return out

    def _forward(self, mchirp, q, eccentricity, chi_eff):
        if self._vs is None:
            raise RuntimeError("fit the coordinates to a sample first")
        from pycbc.waveform.taylorf2_jets import basis_coefficients
        mc, q_, e, chi = (numpy.asarray(v, dtype=float) for v in (mchirp, q, eccentricity, chi_eff))
        shape = numpy.broadcast(mc, q_, e, chi).shape
        mc, q_, e, chi = (numpy.broadcast_to(v, shape).ravel() for v in (mc, q_, e, chi))
        s = numpy.empty((self.ndir, len(mc)))
        # in chunks: the polynomial weights are (points, stencil) per mode, and the inverse asks for five columns a
        # point
        for c in range(0, len(mc), 4096):
            sl = slice(c, c + 4096)
            dC = basis_coefficients(mc[sl], q_[sl], chi[sl], chi[sl]) - self._C0[:, None]
            v = self._M_pn @ dC
            L = self._lagrange(self._offsets(mc[sl], q_[sl], e[sl], chi[sl]))
            R = numpy.einsum('aijkl,ml->maijk', self._T, L[3])
            R = numpy.einsum('maijk,mk->maij', R, L[2])
            R = numpy.einsum('maij,mj->mai', R, L[1])
            v += numpy.einsum('mai,mi->am', R, L[0])
            s[:, sl] = v - self._s0[:, None]
        return numpy.concatenate([s, q_[None]]).reshape((4,) + shape)

    def forward(self, theta):
        return self._forward(*numpy.asarray(theta, dtype=float).T).T

    def inverse(self, y):
        from pycbc.waveform.taylorf2_jets import from_phase_coordinates
        y = numpy.atleast_2d(numpy.asarray(y, dtype=float))
        out = numpy.full(y.shape, numpy.nan)
        # q is the last coordinate itself: rows below q = 1 have no parameters
        ok = numpy.all(numpy.isfinite(y), axis=1) & (y[:, 3] >= 1)
        if ok.any():
            guess = numpy.repeat(self.guess[:, None], ok.sum(), axis=1)
            guess[1] = y[ok, 3]
            # residuals in units of 100 rad: steps stop at 1e-8 rad, above the forward map's rounding (about
            # 2e-10 rad for a one-ulp change of the parameters), which a tolerance of 1e-10 rad never reached
            out[ok] = from_phase_coordinates(y[ok].T, guess, iterations=60, forward=self._forward,
                                             scale=numpy.full(4, 100.),
                                             max_step=[0.05 * self.mc0, numpy.inf, 0.005, 0.2]).T
        return out

    def log_jacobian(self, theta):
        from pycbc.waveform.taylorf2_jets import log_jacobian_from_phase_coordinates
        return log_jacobian_from_phase_coordinates(numpy.asarray(theta, dtype=float).T, forward=self._forward)
