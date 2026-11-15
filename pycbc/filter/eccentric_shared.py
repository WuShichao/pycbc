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
the same basis), linear in the coefficient changes ``dC = (A - A0, B - B0)``, and ``D_kj = d corr_k / d theta_j``
linearises the cell's interpolated eccentric correction in ``dtheta = (ln Mc / Mc0, e - e0, chi - chi0,
ln q / q0)``. The phase is therefore linear in ``c = (r_k dC, dtheta)`` over 20 basis functions, and the mode's
clock ``t_k = t_k0 + dpsi' / 2 pi`` too, so each mode is a :class:`PhaseNUFFTCorrelation` of its own basis and
``PhaseGrid`` applies unchanged. The amplitude change is the cell's own, linearised as one complex log-slope per
direction (a scalar per template and mode: an eccentric harmonic's amplitude goes as e^|n-2|); what
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
    def __init__(self, family, key, view, theta0, data, sky_points, sky_tolerance=1e-4, eps=1e-6,
                 step=(1e-5, 1e-5, 1e-4, 1e-4), window=None, band_margin=5 * 86400.0):
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
        # the reference's family offsets
        ref = family.reference
        mc_ref = (ref['mass1'] * ref['mass2']) ** .6 / (ref['mass1'] + ref['mass2']) ** .2
        x0 = numpy.array([numpy.log(self.theta0[0] / mc_ref), self.theta0[1] - ref['eccentricity'],
                          numpy.log(self.theta0[3] / family.q0), self.theta0[2] - ref.get('spin1z', 0.)])
        step_c = max(1, len(f_all) // 4000)
        coarse = slice(0, len(f_all), step_c)
        D_all, _ = self._derivatives(family, key, x0, f_all[coarse], step)
        D, Dp = self._derivatives(family, key, x0, f, step)
        self._D, self._Dp = xp.asarray(D), xp.asarray(Dp)
        # the cell's pins: its circular change is pinned (value and slope) at f_ref = f_lower, its eccentric
        # correction is not, and every mode's clock is anchored to the (2,2,2) mode's time at f_ref, so the
        # eccentric rows enter the time pin only through that common slope
        _, Dp222 = self._derivatives(family, (2, 2, 2), x0, numpy.array([self.f_lower]), step)
        self._anchor_slope = Dp222[:, 0]
        # the frame's coarse grid and weights: the parent's construction
        power = self.template_power()[coarse]
        if self._sel is not None:
            observed = numpy.zeros(len(f_all), dtype=bool)
            observed[self._sel] = True
            power = power * observed[coarse]
        self._fc = f_all[coarse]
        self._wc = power / power.sum()
        psi_c, _ = _basis(self._fc)
        self._psic = numpy.vstack([psi_c, D_all])
        self._A0, self._B0, self._la0 = taylorf2_basis_coefficients(*self._pn_params(self.theta0))
        # the amplitude change: the cell's own (an eccentric harmonic's amplitude goes as e^|n-2|, which the TaylorF2
        # ratio does not see), as one complex log-slope per direction, the template-power weighted mean of
        # (d a_k / d theta_j) / a_k over the observed band
        self._amp_slopes = self._amplitude_slopes(family, key, x0, self._fc, self._wc, step)

    @staticmethod
    def _pn_params(theta):
        mc, e, chi, q = theta
        return mc, q, chi, chi

    def _coefficients(self, thetas):
        """``c`` (M, 20) and the amplitude ratios."""
        c, amp = [], []
        for theta in numpy.atleast_2d(numpy.asarray(thetas, dtype=float)):
            A, B, la = taylorf2_basis_coefficients(*self._pn_params(theta))
            dC = numpy.concatenate([A - self._A0, B - self._B0])
            dtheta = numpy.array([numpy.log(theta[0] / self.theta0[0]), theta[1] - self.theta0[1],
                                  theta[2] - self.theta0[2], numpy.log(theta[3] / self.theta0[3])])
            c.append(numpy.concatenate([self.r * dC, dtheta]))
            amp.append(numpy.exp(self._amp_slopes @ dtheta))
        return numpy.array(c), numpy.array(amp)

    def _pins(self, c):
        psi_l, dpsi_l = _basis([self.f_lower])
        p0 = numpy.concatenate([psi_l[:, 0], numpy.zeros(4)])
        p1 = numpy.concatenate([dpsi_l[:, 0], self._anchor_slope])
        return c @ p0, c @ p1

    @staticmethod
    def _derivatives(family, key, x0, f, step):
        """``d corr_key / d theta_j`` and its frequency derivative on ``f`` (zero outside the mode's support),
        ``theta = (ln Mc, e, chi, ln q)``, by central differences of the cell's interpolated correction."""
        from scipy.interpolate import CubicSpline
        r, amp, corr, psi0, have, sl = family._expansions[key]
        fx = family.f[sl][have]
        f = numpy.asarray(f, dtype=float)
        D, Dp = numpy.zeros((4, len(f))), numpy.zeros((4, len(f)))
        inside = (f >= fx[0]) & (f <= fx[-1])
        # corr is called (d, de, dq, dchi); theta is ordered (ln Mc, e, chi, ln q)
        for row, (j, h) in enumerate(zip((0, 1, 3, 2), step)):
            if family.steps[j] <= 0:
                continue
            up, dn = x0.copy(), x0.copy()
            up[j] += h
            dn[j] -= h
            spline = CubicSpline(fx, ((corr(*up) - corr(*dn)) / (2 * h))[have], extrapolate=False)
            D[row, inside] = spline(f[inside])
            Dp[row, inside] = spline(f[inside], 1)
        return D, Dp

    @staticmethod
    def _amplitude_slopes(family, key, x0, fc, wc, step):
        """The weighted mean over ``fc`` (weights ``wc``) of ``(d a / d theta_j) / a`` for the mode's face-on
        amplitude, ``theta = (ln Mc, e, chi, ln q)``, by central differences of the cell's interpolated amplitude."""
        r, amp, corr, psi0, have, sl = family._expansions[key]
        fx = family.f[sl][have]
        a0 = amp(*x0)[0][have]
        ok = abs(a0) > 0
        g = numpy.zeros(4, dtype=complex)
        inside = (fc >= fx[0]) & (fc <= fx[-1])
        for row, (j, h) in enumerate(zip((0, 1, 3, 2), step)):
            if family.steps[j] <= 0:
                continue
            up, dn = x0.copy(), x0.copy()
            up[j] += h
            dn[j] -= h
            ratio = numpy.where(ok, (amp(*up)[0][have] - amp(*dn)[0][have]) / (2 * h) / numpy.where(ok, a0, 1), 0)
            on = numpy.interp(fc[inside], fx, ratio.real) + 1j * numpy.interp(fc[inside], fx, ratio.imag)
            g[row] = numpy.sum(wc[inside] * on) / max(numpy.sum(wc[inside]), 1e-300)
        return g

    def _projected_basis(self, D):
        """``D^T Psi(f)`` and ``D^T Psi'(f)`` on the grid for coefficient directions ``D`` (20, ndir)."""
        x, dx = PhaseNUFFTCorrelation._projected_basis(self, D[:16])
        w = self.xp.asarray(D[16:])
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
        self.views, self.correlators, self.grids = [], [], []
        frames, shifts = [], []
        for k, key in enumerate(self.keys):
            view = EccentricModeView(block, theta0, k, block.response_builder, family.f_ref,
                                     block.response_frequencies)
            corr = EccentricModePhaseCorrelation(family, key, view, theta0, block.data, skies,
                                                 sky_tolerance=self.sky_tolerance, window=self.data_window,
                                                 band_margin=band_margin)
            frame = corr.frame(thetas, 3)
            self.views.append(view)
            self.correlators.append(corr)
            frames.append(frame)
            shifts.append(corr.project(thetas, frame)[1])
        # the grid times the region needs (per mode, T = delay + shift): where the probes' ln B lies within
        # edge_margin of each one's largest, for the probes within it of the best, widened by itself and eight lags
        rng = numpy.random.default_rng(seed)
        pick = rng.choice(len(region), min(int(probes), len(region)), replace=False)
        curves = []
        for j in pick:
            ev = model.evaluate(region[j])
            curves.append((ev['delays'], ev['log_bayes']))
        tops = numpy.array([v[1].max() for v in curves])
        good = numpy.nonzero(tops >= tops.max() - self.edge_margin)[0]
        options = dict(grid_options or {})
        for k in range(len(self.keys)):
            tlo, thi = numpy.inf, -numpy.inf
            for i in good:
                d, v = curves[i]
                T = d[v >= tops[i] - self.edge_margin] + shifts[k][pick[i]]
                tlo, thi = min(tlo, T.min()), max(thi, T.max())
            pad = (thi - tlo) + 8 * dt
            reach = (self.lags[0] * dt + shifts[k].min() - dt, self.lags[-1] * dt + shifts[k].max() + 2 * dt)
            window = (max(tlo - pad, reach[0]), min(thi + pad, reach[1]))
            self.grids.append(PhaseGrid(self.correlators[k], thetas, (lo, hi), time_range=window, **options))
        model.block._prepared = None

    def _coordinates(self, points):
        pairs = [self.model._theta_sky(p) for p in points]
        return numpy.array([t for t, _ in pairs]), numpy.array([s for _, s in pairs])

    def inside(self, points):
        """Whether each point lies in the region every mode's grid covers, with its response on the sky basis."""
        thetas, skies = self._coordinates(points)
        ok = numpy.ones(len(points), dtype=bool)
        for grid in self.grids:
            ok &= grid.inside(thetas)
        if ok.any():
            ok[ok] = self.correlators[0].sky_error(skies[ok]) <= self.sky_tolerance
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

    #: samples per ln B call (its orientation quadrature holds n_l0 x nodes values per sample)
    bayes_chunk = 256

    def _log_bayes(self, z, g, which):
        """The model's ln B at samples ``which`` of ``z`` (K, 2, S) and ``g`` (..., S), in chunks."""
        xp = self.xp
        out = []
        for c in range(0, len(which), self.bayes_chunk):
            sel = xp.asarray(which[c:c + self.bayes_chunk])
            v = self.model._log_bayes(z[..., sel], g[..., sel])[0]
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

    def __call__(self, points):
        """The log likelihood of each point; ``nan`` where its F has not fallen by ``edge_margin`` at the ends of
        the lags its grid times cover. All points' correlations, Grams and statistics are batched."""
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
        idx = numpy.nonzero(last - first >= 2)[0]
        if not len(idx):
            return out
        th, sk, fi, la = thetas[idx], skies[idx], first[idx], last[idx]
        Mi = len(idx)
        count = la - fi + 1
        W = int(count.max())
        col = numpy.arange(W)
        valid = col[None] < count[:, None]
        K = numpy.minimum(fi[:, None] + col[None], la[:, None])
        delays = lags[K] * dt                                                  # (Mi, W)
        # the points' anchor Grams (cached anchors: no mode spectra)
        grams = []
        for j in range(Mi):
            atimes, g = m.block.template_gram(th[j], sk[j])
            grams.append(g)
        self._anchor_times = numpy.asarray(atimes, dtype=float)
        stack = xp.stack(grams)
        del grams
        rows = numpy.repeat(numpy.arange(Mi), W)
        z = self._z(th, sk, delays)                                            # (Mi, K, 2, W)
        nk = z.shape[1]
        zs = xp.moveaxis(z, 0, 2).reshape(nk, 2, -1)
        gs = self._gram_at(stack, rows, delays.ravel())
        twof = host(mean_anomaly_twof_analytic(zs / var, gs / var, offsets, m.screen_anomalies,
                                               xp=xp)[0].max(axis=0)).reshape(Mi, W)
        del z, zs, gs
        F = numpy.where(valid, twof / 2, -numpy.inf)
        top = F.max(axis=1)
        r = numpy.arange(Mi)
        edge = ((F[:, 0] <= top - self.edge_margin) | (fi == 0)) & \
            ((F[r, count - 1] <= top - self.edge_margin) | (la == L - 1))
        keep = F >= top[:, None] - m.screen_margin
        keep = (keep | numpy.roll(keep, 1, axis=1) | numpy.roll(keep, -1, axis=1)) & valid
        # the oversampled samples of the kept lags
        P = m.oversampling
        frac = numpy.arange(P) / P
        fine = [((K[j][keep[j]][None, :] + lags[0] + frac[:, None]) * dt).ravel() for j in range(Mi)]
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
        ftwof = host(mean_anomaly_twof_analytic(zf / var, gf / var, offsets, m.n_l0, xp=xp)[0].max(axis=0))
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
        out[idx] = numpy.where(edge, value, numpy.nan)
        return out
