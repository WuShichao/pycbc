"""Multirate time-domain TDI built from pre-response harmonic windows."""

from dataclasses import dataclass

import numpy as np

from pycbc.tdi.onthefly import (
    adaptive_sparse_tdi_response,
    delay_padding,
)
from pycbc.tdi.sources import frequency_partition_sources


@dataclass(frozen=True)
class TDIBandResponse:
    """Sparse response and sampling cadence for one harmonic frequency band."""

    harmonic: object
    index: int
    f_lower: float
    f_upper: float
    lower_overlap: float
    upper_overlap: float
    delta_t: float
    response: object

    @property
    def support_blocks(self):
        """Mission-time blocks on which the delayed band can contribute."""
        blocks = []
        for item in self.response.responses:
            support = item["support"]
            if np.ndim(support) == 1:
                blocks.append(tuple(support))
            else:
                blocks.extend(tuple(block) for block in support)
        return tuple(blocks)


@dataclass(frozen=True)
class TDIMultibandBlock:
    """One uniformly sampled block emitted by a multirate TDI response."""

    band: TDIBandResponse
    block_index: int
    series: dict


class MultibandSparseTDIResponse:
    """Collection of exact pre-response bands with independent cadences.

    Frequency windows are already part of each band source, before any link
    retardation. Summing :meth:`sample` across the bands therefore reconstructs
    the full time-domain response up to the independently controlled sparse
    interpolation tolerance.
    """

    def __init__(self, source, bands, padding):
        self.source = source
        self.bands = tuple(bands)
        if not self.bands:
            raise ValueError("a multiband response needs at least one live band")
        channel_sets = {band.response.channels for band in self.bands}
        if len(channel_sets) != 1:
            raise ValueError("every frequency band must contain the same channels")
        self.channels = next(iter(channel_sets))
        self.padding = float(padding)

    def sample(self, times, channels=None, complex_output=False):
        """Evaluate and sum all windowed TDI bands at arbitrary mission times."""
        times = np.asarray(times, dtype=float)
        if channels is None:
            selected = self.channels
        elif isinstance(channels, str):
            selected = (channels,)
        else:
            selected = tuple(channels)
        dtype = complex if complex_output else float
        output = {name: np.zeros(len(times), dtype=dtype) for name in selected}
        for band in self.bands:
            values = band.response.sample(
                times, channels=selected, complex_output=complex_output)
            for name in selected:
                output[name] += values[name]
        return output

    def iter_timeseries(self, channels=None, chunk_size=262144):
        """Yield independently sampled, bounded-memory time-domain blocks.

        Consumers should transform and discard one block at a time. No array at
        the highest cadence over the full observation is constructed.
        """
        for band in self.bands:
            for block_index, (start, stop) in enumerate(band.support_blocks):
                series = band.response.to_timeseries(
                    band.delta_t,
                    t_start=start,
                    t_end=stop,
                    channels=channels,
                    chunk_size=chunk_size,
                )
                yield TDIMultibandBlock(band, block_index, series)

    @property
    def diagnostics(self):
        """Per-band support, cadence and adaptive-response diagnostics."""
        return tuple({
            "harmonic": band.harmonic,
            "band": (band.f_lower, band.f_upper),
            "overlap": (band.lower_overlap, band.upper_overlap),
            "delta_t": band.delta_t,
            "support_blocks": band.support_blocks,
            "response": band.response.diagnostics,
        } for band in self.bands)


def multiband_sparse_tdi_response(
        source, orbit, channel_terms, lamb, beta, band_edges, overlap=0.0,
        t_start=None, t_end=None, samples_per_cycle=4.0,
        initial_step=86400.0, relative_tolerance=1e-4,
        amplitude_floor=1e-3, max_refinements=24, velocity_order=1,
        links=None, padding=None, harmonics=None):
    """Build independently sampled time-domain TDI frequency bands.

    This function performs no frequency-domain detector projection. Each
    complementary source window is evaluated at the retarded source times
    inside the ordinary link response and TDI delay chains.
    """
    if samples_per_cycle <= 2:
        raise ValueError("samples_per_cycle must exceed the Nyquist minimum")
    if t_start is None:
        t_start = source.t_start
    if t_end is None:
        t_end = source.t_end
    t_start, t_end = float(t_start), float(t_end)
    if not t_end > t_start:
        raise ValueError("t_end must be greater than t_start")

    partitions = frequency_partition_sources(
        source, band_edges, overlap=overlap, harmonics=harmonics)
    all_terms = tuple(
        term for terms in channel_terms.values() for term in terms)
    delay_options = {} if links is None else {"links": links}
    if padding is None:
        padding = delay_padding(
            orbit, np.asarray([t_start, t_end]), all_terms, **delay_options)
    padding = float(padding)
    if padding < 0:
        raise ValueError("padding must be non-negative")

    response_options = dict(
        t_start=t_start,
        t_end=t_end,
        initial_step=initial_step,
        relative_tolerance=relative_tolerance,
        amplitude_floor=amplitude_floor,
        max_refinements=max_refinements,
        velocity_order=velocity_order,
        support_padding=padding,
    )
    if links is not None:
        response_options["links"] = links

    bands = []
    for harmonic, windows in partitions.items():
        for index, windowed in enumerate(windows):
            if not windowed.support_blocks(harmonic):
                continue
            response = adaptive_sparse_tdi_response(
                windowed, orbit, channel_terms, lamb, beta,
                **response_options)
            if not response.responses:
                continue
            highest = windowed.f_upper + 0.5 * windowed.upper_overlap
            bands.append(TDIBandResponse(
                harmonic=harmonic,
                index=index,
                f_lower=windowed.f_lower,
                f_upper=windowed.f_upper,
                lower_overlap=windowed.lower_overlap,
                upper_overlap=windowed.upper_overlap,
                delta_t=1.0 / (float(samples_per_cycle) * highest),
                response=response,
            ))
    return MultibandSparseTDIResponse(source, bands, padding)
