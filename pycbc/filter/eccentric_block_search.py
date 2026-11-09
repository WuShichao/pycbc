# Copyright (C) 2026 Shichao Wu
# This program is free software under the GNU General Public License,
# version 3 or (at your option) any later version.
"""An eccentric-cell backend for the existing coherent multilevel search.

The driver, source builder and metric define the cell. This backend uses
WindowedModeFFT over the complete frequency band at every level and combines
all modes coherently on an anomaly grid. Full windowed Grams are prepared
on delay anchors and interpolated as complete matrices, preserving PSD.
Their interpolation accuracy must be checked with exact withheld delays.
"""
from types import SimpleNamespace
from time import perf_counter
import numpy

from pycbc.filter.eccentric_mode_fft import WindowedModeFFT
from pycbc.filter.mean_anomaly import (mean_anomaly_components,
                                      mean_anomaly_twof,
                                      mean_anomaly_log_bayes)
from pycbc.filter.bstatistic import OrientationNodes


class MassEccentricCoordinates:
    """Local coordinates ln(Mc/Mc0) and e^2-e0^2, each in explicit units."""
    def __init__(self, mchirp, eccentricity, scales):
        self.mchirp, self.eccentricity = float(mchirp), float(eccentricity)
        self.scales = numpy.asarray(scales, dtype=float)
        self.ndim = len(self.scales)
        if self.ndim not in (1, 2) or numpy.any(self.scales <= 0):
            raise ValueError('one or two positive coordinate scales are required')

    def encode(self, theta):
        theta = numpy.atleast_2d(numpy.asarray(theta, dtype=float))
        columns = [numpy.log(theta[:, 0]/self.mchirp)/self.scales[0]]
        if self.ndim == 2:
            columns.append((theta[:, 1]**2-self.eccentricity**2)/self.scales[1])
        return numpy.column_stack(columns)

    def decode(self, coordinates):
        s = numpy.atleast_2d(numpy.asarray(coordinates, dtype=float))
        columns = [self.mchirp*numpy.exp(s[:, 0]*self.scales[0])]
        if self.ndim == 2:
            e2 = self.eccentricity**2+s[:, 1]*self.scales[1]
            if numpy.any(e2 < 0):
                raise ValueError('eccentric lattice extends below e^2=0')
            columns.append(numpy.sqrt(e2))
        return numpy.column_stack(columns)


class EccentricFFTBlockModel:
    """Adapter used directly by sky_block_search.multilevel_block_search.

    source_builder(theta) returns whitened mode FD waveforms and individual
    clocks on grid.frequencies, both (K,N). It must evaluate a prepared cell,
    not generate a native candidate waveform. response_builder(sky_offsets)
    returns yearly harmonics (R,2,C,ncoarse) on response_frequencies. These
    are interpolated in frequency; the response is evaluated separately on
    every mode clock by WindowedModeFFT. metric describes squared-match loss
    in the coordinates supplied here. Its domain must be validated by the
    injection experiment. reference/scales include eclipticlongitude and
    eclipticlatitude, in the same units as the existing sky-block driver.
    """
    def __init__(self, grid, data, source_builder, response_builder,
                 response_frequencies, orders, offsets, coordinates, metric,
                 reference, scales, observation, delay_range, xp=numpy,
                 n_l0=64, dtype=numpy.complex64, max_cache_bytes=4_000_000_000,
                 distance_prior=None, bayes_rank=False, contains=None, gram_anchors=3, reuse_grams=False):
        self.grid, self.xp = grid, xp
        self.surrogate = SimpleNamespace(grid=grid, xp=xp)
        self.source_builder, self.response_builder = source_builder, response_builder
        self.orders, self.offsets = numpy.asarray(orders), numpy.asarray(offsets)
        self.coordinates, self.metric = coordinates, numpy.asarray(metric)
        self.reference, self.scales = dict(reference), dict(scales)
        self.observation, self.delay_range = observation, tuple(delay_range)
        self.n_l0, self.dtype = int(n_l0), numpy.dtype(dtype)
        self.max_cache_bytes = max_cache_bytes
        self.distance_prior, self.bayes_rank = distance_prior, bayes_rank
        self.contains = contains
        self.gram_anchors = int(gram_anchors)
        self.reuse_grams = bool(reuse_grams)
        self._template_grams = {}
        self._source_cache = None
        f = numpy.asarray(grid.frequencies)
        coarse = numpy.asarray(response_frequencies)
        if coarse[0] > f[0] or coarse[-1] < f[-1]:
            raise ValueError('response grid must cover the FFT frequencies')
        index = numpy.clip(numpy.searchsorted(coarse, f)-1, 0, len(coarse)-2)
        self._index = xp.asarray(index, dtype=xp.int32)
        self._weight = xp.asarray((f-coarse[index])/(coarse[index+1]-coarse[index]),dtype=xp.float32)
        self._prepared = None
        self._profiles = {}
        self.work = dict(source_seconds=0., prepare_seconds=0., gram_seconds=0.,
                         prepared_templates=0, searched_points=0, gram_cache_hits=0)
        self.set_data(data)

    def set_data(self, data):
        self._prepared = None
        self._profiles = {}
        self.data = self.xp.asarray(data, dtype=self.dtype)

    def release_search_cache(self):
        self._prepared = None
        self._source_cache = None
        self._profiles = {}

    def clear_template_grams(self):
        self._template_grams = {}

    def phase_frame(self, region, **kwargs):
        s = self.coordinates.encode(region)
        return s, self.metric, {'directions':numpy.eye(s.shape[1])}, None

    def _sync(self):
        if self.xp is not numpy:
            self.xp.cuda.get_current_stream().synchronize()

    def _prepare(self, theta, sky, anchors=None):
        anchors = self.gram_anchors if anchors is None else int(anchors)
        key = tuple(theta)+tuple(sky)+(int(anchors),)
        if self._prepared is not None and self._prepared[0] == key:
            return self._prepared[1:]
        self._prepared = None
        xp = self.xp
        start = perf_counter()
        source_key = tuple(theta)
        if self._source_cache is None or self._source_cache[0] != source_key:
            self._source_cache = None
            b, t = self.source_builder(numpy.asarray(theta))
            self._source_cache = (source_key, b, t)
        else:
            _, b, t = self._source_cache
        self.work['source_seconds'] += perf_counter()-start
        rho = xp.asarray(self.response_builder(numpy.asarray(sky)), dtype=self.dtype)
        def response(r):
            return (rho[r][...,self._index]*(1-self._weight)+
                    rho[r][...,self._index+1]*self._weight)
        lo, hi = self.delay_range
        jlo, jhi = int(numpy.ceil(lo/self.grid.delta_t)),int(numpy.floor(hi/self.grid.delta_t))
        start = perf_counter()
        engine = WindowedModeFFT(self.grid,b,t,response,self.orders,self.data,
            self.observation,(jlo,jhi),xp=xp,dtype=self.dtype,
            max_cache_bytes=self.max_cache_bytes)
        del b, t
        self._sync()
        self.work['prepare_seconds'] += perf_counter()-start
        anchor_indices = numpy.unique(numpy.linspace(jlo,jhi,anchors).round().astype(int))
        start = perf_counter()
        if self.reuse_grams and key in self._template_grams:
            gram = xp.asarray(self._template_grams[key])
            self.work['gram_cache_hits'] += 1
        else:
            gram = engine.gram(anchor_indices)
            if self.reuse_grams:
                self._template_grams[key] = gram.get() if hasattr(gram, 'get') else gram.copy()
        self._sync()
        self.work['gram_seconds'] += perf_counter()-start
        anchor_times = anchor_indices*self.grid.delta_t
        self._prepared = (key,engine,anchor_times,gram)
        self.work['prepared_templates'] += 1
        return engine,anchor_times,gram

    def fft_block_search(self, thetas, sky, delay_range=None, top=20,
                         coefficients=None, decimate=1, threshold=None,
                         anchors=5, n_l0=None, lag_chunk=1024, **kwargs):
        xp = self.xp
        anchors = self.gram_anchors
        if kwargs.get('band') is not None or kwargs.get('sky_set') is not None:
            raise ValueError('eccentric backend needs a full band and one sky set')
        points = (numpy.asarray(thetas) if coefficients is None else
                  self.coordinates.decode(coefficients))
        sky = numpy.atleast_2d(numpy.asarray(sky))
        lo,hi = self.delay_range if delay_range is None else delay_range
        lo,hi = max(lo,self.delay_range[0]),min(hi,self.delay_range[1])
        dt = self.grid.delta_t
        lags = numpy.arange(int(numpy.ceil(lo/(dt*decimate))),
                            int(numpy.floor(hi/(dt*decimate)))+1)*decimate
        if not len(lags):
            raise ValueError('empty eccentric delay search')
        delays = lags*dt
        found,survivors=[],[]
        nangle = self.n_l0 if n_l0 is None else n_l0
        engine=None
        for n,theta in enumerate(points):
            if self.contains is not None and not self.contains(theta):
                continue
            best=None
            for p,offset in enumerate(sky):
                profile_key = tuple(theta)+tuple(offset)+(int(anchors),)
                engine = None
                if profile_key not in self._profiles:
                    engine,atimes,ganchors = self._prepare(theta,offset,anchors)
                    full_lags = numpy.arange(int(numpy.ceil(self.delay_range[0]/dt)),
                                             int(numpy.floor(self.delay_range[1]/dt))+1)
                    full_z = engine.correlations(full_lags)
                    host = lambda a: a.get() if hasattr(a, 'get') else numpy.asarray(a)
                    if len(self._profiles) >= 256:
                        self._profiles.pop(next(iter(self._profiles)))
                    self._profiles[profile_key] = (full_lags[0],host(full_z),atimes,host(ganchors))
                    del full_z
                first_lag,stored_z,atimes,stored_g = self._profiles[profile_key]
                z = xp.asarray(stored_z[...,lags-first_lag])
                ganchors = xp.asarray(stored_g)
                for first in range(0,len(lags),lag_chunk):
                    last=min(first+lag_chunk,len(lags))
                    g=WindowedModeFFT.interpolated_gram(self,delays[first:last],atimes,ganchors)
                    stat,angles=mean_anomaly_twof(z[...,first:last],g,self.offsets,nangle,xp=xp)
                    angle_at=xp.argmax(stat,axis=0)
                    values=stat[angle_at,xp.arange(last-first)]
                    j=int(xp.argmax(values))
                    value=float(values[j])
                    if best is None or value>best['twof']:
                        best=dict(twof=value,theta=[float(v) for v in theta],
                            coefficients=self.coordinates.encode(theta)[0].tolist(),
                            sky=offset.tolist(),time=float(delays[first+j]),
                            mean_anomaly=float(angles[angle_at[j]]),
                            longitude=float(self.reference['eclipticlongitude']+offset[0]*self.scales['eclipticlongitude']),
                            latitude=float(self.reference['eclipticlatitude']+offset[1]*self.scales['eclipticlatitude']))
                    if threshold is not None:
                        for tile in range(0,last-first,128):
                            tail=min(tile+128,last-first)
                            k=tile+int(xp.argmax(values[tile:tail]))
                            v=float(values[k])
                            if v>threshold:
                                survivors.append([n,p,first+k,v,0])
                del z
            found.append(best)
        found.sort(key=lambda row:-row['twof'])
        for candidate in found[:int(kwargs.get('refine', 0))]:
            self.refine_candidate(candidate, subsamples=int(kwargs.get('subsamples', 8)))
        found.sort(key=lambda row:-row['twof'])
        count=len(found)*len(sky)*len(lags)
        self.work['searched_points'] += count
        result=dict(candidates=found[:top],templates=count,
                    anomaly_nodes=nangle,backend='eccentric_mode_fft')
        if threshold is not None:
            result.update(survivors=numpy.asarray(survivors).reshape(-1,5),delays=delays)
        return result

    def refine_candidate(self,candidate,subsamples=8,span=1,**kwargs):
        xp=self.xp
        engine,atimes,ganchors=self._prepare(candidate['theta'],candidate['sky'])
        dt=self.grid.delta_t
        delays=candidate['time']+numpy.arange(-span*subsamples,span*subsamples+1)*dt/subsamples
        keep=(delays>=atimes[0])&(delays<=atimes[-1])
        delays=delays[keep]
        z=engine.fractional_correlations(delays)
        g=engine.interpolated_gram(delays,atimes,ganchors)
        stat,angles=mean_anomaly_twof(z,g,self.offsets,self.n_l0,xp=xp)
        angle,lag=numpy.unravel_index(int(xp.argmax(stat)),stat.shape)
        candidate.update(twof=float(stat[angle,lag]),time=float(delays[lag]),
                         mean_anomaly=float(angles[angle]))
        zl,gl,_=mean_anomaly_components(z[...,lag],g[...,lag],self.offsets,self.n_l0,xp=xp)
        zz,gg=zl[angle],gl[angle]
        candidate['z']=[complex(v) for v in (zz.get() if hasattr(zz,'get') else zz)]
        candidate['gram']=[float(gg[0,0].real),float(gg[1,1].real),complex(gg[0,1])]
        if self.distance_prior is not None:
            logb,_,_=mean_anomaly_log_bayes(z[...,lag],g[...,lag],self.offsets,
                self.distance_prior,OrientationNodes(8,8),self.n_l0,xp=xp)
            candidate['log_bayes']=float(logb)
            if self.bayes_rank:candidate['ranking']=float(logb)
        return candidate
