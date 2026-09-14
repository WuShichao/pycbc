"""Relative binning that keeps a source's harmonics apart.

Relative binning approximates the ratio of a candidate to a fiducial as
linear across a bin. For a single carrier that ratio is a smooth function of
frequency and the approximation is excellent. For an eccentric or precessing
source it is not: several harmonics contribute at one frequency from
different times, so the ratio of the *summed* waveform to a summed fiducial
carries the beat between them, and no bin is wide enough to be useful and
narrow enough to be linear.

Binned against its own fiducial, one harmonic is smooth again. So the sum has
to move outside the approximation:

.. math::

    \\langle d|h\\rangle = \\sum_i \\langle d|h_i\\rangle, \\qquad
    \\langle h|h\\rangle = \\sum_i \\langle h_i|h_i\\rangle
        + 2\\,\\mathrm{Re}\\!\\!\\sum_{i<j} \\langle h_i|h_j\\rangle

The self terms are ordinary relative binning, one summary per harmonic. The
cross terms are what makes this more than N independent models, and PyCBC
already has the kernel for them -- ``likelihood_parts_det_multi``, written for
overlapping *signals*, computes exactly this quantity from a summary built
against a pair of fiducials. What it does not have is a model that can hold
harmonics of one source: ``MultiSignalModel`` namespaces parameters per
sub-model, which would require declaring every parameter once per harmonic
and constraining them equal.

Cost. Ten harmonics is ten self terms and forty-five cross terms per channel.
The summaries are built once; per candidate it is one projection per harmonic
-- measured at 27.6 ms for ten against 26.0 ms for the summed projection,
since the work is the same either way -- plus the kernel calls.
"""
import itertools
import logging

import numpy

from pycbc.inference.models.relbin import Relative, setup_bins
from pycbc.inference.models.relbin_cpu import (likelihood_parts_det,
                                               likelihood_parts_det_multi)
from pycbc.waveform.waveform import get_fd_det_waveform_sequence
from pycbc.types import Array


class HarmonicRelative(Relative):
    """Relative binning with one fiducial and one summary per harmonic.

    Requires a detector-response waveform generator that accepts a
    ``tdi_harmonic`` parameter and returns that harmonic alone, as
    `pycbc.tdi.inference` does.

    Parameters
    ----------
    harmonics : sequence, optional
        Labels to keep apart. Default: whatever the fiducial source carries,
        obtained through ``pycbc.tdi.inference.harmonic_labels``.
    cross_terms : bool, optional
        Include the harmonic cross terms. Default True. Setting it False is a
        diagnostic -- it says how much the harmonics actually overlap -- and
        not a cheaper approximation to be used unmeasured.
    """

    name = "tdi_harmonic_relative"

    def __init__(self, variable_params, data, low_frequency_cutoff,
                 harmonics=None, cross_terms=True,
                 harmonic_epsilon=0.1, **kwargs):
        super().__init__(variable_params, data, low_frequency_cutoff,
                         **kwargs)
        self.harmonic_epsilon = float(harmonic_epsilon)
        if not self.still_needs_det_response:
            raise ValueError(
                "this model only makes sense for a waveform that carries its "
                "own detector response; the approximant is not in "
                "fd_det_sequence")
        self.cross_terms = bool(cross_terms)
        if harmonics is None:
            from pycbc.tdi.inference import harmonic_labels
            harmonics = harmonic_labels(self.fid_params)
        self.harmonics = tuple(harmonics)
        if len(self.harmonics) < 1:
            raise ValueError("the source carries no harmonics")

        self.hfedges, self.hedges, self.hquery = {}, {}, {}
        self.h00_h, self.sdat_h, self.cross = {}, {}, {}
        for ifo in self.data:
            self._prepare_channel(ifo)
        self._build_shared_query()
        logging.info("%s: %d harmonics, %d cross pairs per channel",
                     self.name, len(self.harmonics),
                     len(self.harmonics) * (len(self.harmonics) - 1) // 2)

    def _build_shared_query(self):
        """One frequency set per harmonic, shared by every channel.

        Each channel bins against its own fiducial and so asks for a slightly
        different set of points, but the generator's cost is in building the
        harmonic's response, not in evaluating it -- asking channel by channel
        rebuilt the same band's time series once per channel. Taking the union
        up front makes that one call, and `take` puts each channel back on its
        own points.
        """
        self.uquery, self.utake = {}, {}
        for ifo in self.data:
            self.utake[ifo] = {}
        for harmonic in self.harmonics:
            live = [ifo for ifo in self.data
                    if harmonic in self.hquery.get(ifo, {})]
            if not live:
                continue
            values = numpy.unique(numpy.concatenate(
                [self.f[ifo][self.hquery[ifo][harmonic]] for ifo in live]))
            self.uquery[harmonic] = (tuple(live), values)
            for ifo in live:
                own = self.f[ifo][self.hquery[ifo][harmonic]]
                take = numpy.searchsorted(values, own)
                # `searchsorted` places a value that is not in the union at
                # its insertion point rather than failing, so a channel whose
                # frequency grid differed in the last bit would be served its
                # neighbour's sample and nothing would say so. The channels
                # share one grid here; this is what says so if they stop.
                if not numpy.array_equal(values[take], own):
                    raise ValueError(
                        f"{ifo} asks for frequencies absent from the shared "
                        f"set for harmonic {harmonic}; the channels no "
                        "longer share one grid")
                self.utake[ifo][harmonic] = take

    def _fiducial_harmonic(self, ifo, harmonic, frequencies):
        params = dict(self.fid_params)
        params['tdi_harmonic'] = harmonic
        wave = get_fd_det_waveform_sequence(
            ifos=ifo, sample_points=Array(frequencies.astype(numpy.float64)),
            **params)
        return numpy.asarray(wave[ifo])

    def _prepare_channel(self, ifo):
        """Per-harmonic fiducials, their own bins, and the pairwise summaries.

        A single shared partition does not work. The union of every
        harmonic's edges puts each harmonic's fiducial at zero on the other
        harmonics' points, and the kernel divides by it. `Relative` meets the
        same problem between overlapping signals and answers it the same way:
        each term gets the union of just the edges it involves, with any point
        where either fiducial vanishes removed.
        """
        full = self.f[ifo]
        kmin, kmax = self.kmin[ifo], self.kmax[ifo]
        fiducials, own = {}, {}
        for harmonic in self.harmonics:
            values = numpy.zeros(len(full), dtype=numpy.complex128)
            piece = self._fiducial_harmonic(ifo, harmonic, full[kmin:kmax + 1])
            values[kmin:kmax + 1] = piece
            fiducials[harmonic] = values
            if not numpy.any(piece):
                logging.info("%s: harmonic %s is empty in this band",
                             ifo, harmonic)
                continue
            # `setup_bins` places edges from the standard relative-binning
            # powers of frequency and never looks at the fiducial, so binning
            # every harmonic over the whole band would return identical edges.
            # What differs between harmonics is where they live.
            live = numpy.flatnonzero(piece) + kmin
            edges = setup_bins(f_full=full, f_lo=float(full[live[0]]),
                               f_hi=float(full[live[-1]]),
                               eps=float(self.harmonic_epsilon))
            edges = numpy.asarray(edges, dtype=int)
            own[harmonic] = edges[values[edges] != 0]

        if not own:
            raise ValueError(f"no harmonic has support in {ifo}'s band")

        shifted = self._shifted_data(ifo)
        self.hedges[ifo], self.hfedges[ifo] = {}, {}
        self.h00_h[ifo], self.sdat_h[ifo] = {}, {}
        for harmonic, edges in own.items():
            values = fiducials[harmonic]
            bins = numpy.array([(edges[i], edges[i + 1])
                                for i in range(len(edges) - 1)])
            # `summary_product(x, y)` conjugates its first argument, and
            # `Relative` builds the data summary as (data, fiducial); the
            # kernel returns conj(hd), which is <h|d>. Reversing the order
            # would conjugate the whole filter.
            a0, a1 = self.summary_product(shifted, values, bins, ifo)
            b0, b1 = self.summary_product(values, values, bins, ifo)
            self.hedges[ifo][harmonic] = edges
            self.hfedges[ifo][harmonic] = full[edges]
            self.h00_h[ifo][harmonic] = values[edges]
            self.sdat_h[ifo][harmonic] = {
                'a0': a0, 'a1': a1,
                'b0': numpy.abs(b0), 'b1': numpy.abs(b1)}

        self.cross[ifo] = {}
        if self.cross_terms:
            for first, second in itertools.combinations(sorted(own), 2):
                shared = numpy.unique(numpy.concatenate(
                    (own[first], own[second])))
                keep = ((fiducials[first][shared] != 0)
                        & (fiducials[second][shared] != 0))
                shared = shared[keep]
                if len(shared) < 2:
                    continue          # the two never overlap in frequency
                bins = numpy.array([(shared[i], shared[i + 1])
                                    for i in range(len(shared) - 1)])
                a0, a1 = self.summary_product(
                    fiducials[first], fiducials[second], bins, ifo)
                self.cross[ifo][(first, second)] = dict(
                    edges=shared, freqs=full[shared], a0=a0, a1=a1,
                    h1=fiducials[first][shared],
                    h2=fiducials[second][shared])

        self.hquery[ifo] = {}
        for harmonic in own:
            wanted = [self.hedges[ifo][harmonic]]
            for (first, second), block in self.cross[ifo].items():
                if harmonic in (first, second):
                    wanted.append(block['edges'])
            self.hquery[ifo][harmonic] = numpy.unique(
                numpy.concatenate(wanted))

    def _shifted_data(self, ifo):
        """The data with the same time shift the base class applied."""
        tshift = numpy.exp(
            -2.0j * numpy.pi * self.f[ifo] * self.ta[ifo])
        return numpy.asarray(self.data[ifo] * numpy.conjugate(tshift))

    @property
    def cross_fraction(self):
        """Share of <h|h> carried by the harmonic cross terms, last call.

        This is what decides whether the N(N-1)/2 summaries are worth
        building. Harmonics that overlap in frequency need not overlap in
        *time*, and when they do not the integral averages away: an eccentric
        harmonic n sweeps the band that n-1 swept earlier, so the two share
        frequencies while never being loud together.
        """
        if not getattr(self, '_last_self', 0.0):
            return float('nan')
        return self._last_cross / self._last_self

    def _loglr(self):
        params = self.current_params
        filt, norm = 0j, 0.0
        self._last_self, self._last_cross = 0.0, 0.0
        sampled = {ifo: {} for ifo in self.data}
        for harmonic, (live, values) in self.uquery.items():
            request = dict(params)
            request['tdi_harmonic'] = harmonic
            wave = get_fd_det_waveform_sequence(
                ifos=live, sample_points=Array(values.astype(numpy.float64)),
                **request)
            for ifo in live:
                whole = numpy.asarray(wave[ifo], dtype=numpy.complex128)
                sampled[ifo][harmonic] = whole[self.utake[ifo][harmonic]]

        for ifo in self.data:
            live = self.h00_h[ifo]
            index = {h: self.hquery[ifo][h] for h in live}
            sampled_ifo = sampled[ifo]

            for harmonic in live:
                summary = self.sdat_h[ifo][harmonic]
                edges = self.hedges[ifo][harmonic]
                take = numpy.searchsorted(index[harmonic], edges)
                part_filt, part_norm = likelihood_parts_det(
                    self.hfedges[ifo][harmonic], 0.0,
                    numpy.ascontiguousarray(sampled_ifo[harmonic][take]),
                    numpy.ascontiguousarray(self.h00_h[ifo][harmonic]),
                    summary['a0'], summary['a1'],
                    summary['b0'], summary['b1'])
                filt += part_filt
                norm += part_norm
                self._last_self += part_norm

            for (first, second), block in self.cross[ifo].items():
                take1 = numpy.searchsorted(index[first], block['edges'])
                take2 = numpy.searchsorted(index[second], block['edges'])
                pair = likelihood_parts_det_multi(
                    block['freqs'], 0.0,
                    numpy.ascontiguousarray(sampled_ifo[first][take1]),
                    numpy.ascontiguousarray(block['h1']),
                    0.0,
                    numpy.ascontiguousarray(sampled_ifo[second][take2]),
                    numpy.ascontiguousarray(block['h2']),
                    block['a0'], block['a1'])
                # 2 Re<h_i|h_j> enters <h|h>, which enters the likelihood with
                # a factor of one half.
                norm += 2.0 * pair.real
                self._last_cross += 2.0 * pair.real
        return float(numpy.real(filt) - 0.5 * norm)
