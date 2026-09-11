"""Multirate time-domain TDI built from pre-response harmonic windows."""

from dataclasses import dataclass, replace

import numpy as np

from pycbc.tdi.onthefly import (
    adaptive_sparse_tdi_response,
    delay_padding,
)
from pycbc.tdi.sources import frequency_partition_sources


def _mapping_value(value, name):
    """Return a channel-specific value or a shared scalar value."""
    return value[name] if hasattr(value, "keys") else value


def _frequency_grid_indices(frequencies, delta_f):
    """Validate frequencies on one non-negative, uniform FFT grid."""
    frequencies = np.asarray(frequencies, dtype=float)
    if (frequencies.ndim != 1 or np.any(~np.isfinite(frequencies))
            or np.any(np.diff(frequencies) <= 0)):
        raise ValueError("frequencies must be increasing and finite")
    delta_f = float(delta_f)
    if not np.isfinite(delta_f) or delta_f <= 0:
        raise ValueError("delta_f must be positive and finite")
    indices = np.rint(frequencies / delta_f).astype(np.int64)
    tolerance = 32 * np.finfo(float).eps * np.maximum(1.0, frequencies)
    if np.any(np.abs(indices * delta_f - frequencies) > tolerance):
        raise ValueError("frequencies must lie on the delta_f FFT grid")
    return frequencies, indices


def _direct_frequency_samples(series, frequencies, epoch, chunk_size):
    """Evaluate a few Fourier samples without allocating a dense transform."""
    values = np.asarray(series)
    frequencies = np.asarray(frequencies, dtype=float)
    output = np.zeros(len(frequencies), dtype=complex)
    offset = float(series.start_time) - float(epoch)
    delta_t = float(series.delta_t)
    for first in range(0, len(values), chunk_size):
        stop = min(len(values), first + chunk_size)
        times = offset + np.arange(first, stop, dtype=float) * delta_t
        phase = np.exp(-2j * np.pi * frequencies[:, None] * times)
        output += phase @ values[first:stop]
    return delta_t * output


def _zoom_frequency_samples(series, frequencies, indices, delta_f, epoch,
                            direct_chunk_size):
    """Evaluate selected samples of a short block on a longer FFT grid."""
    from scipy.signal import zoom_fft

    weighted = series.copy()
    if len(weighted):
        weighted[0] *= 0.5
        if len(weighted) > 1:
            weighted[len(weighted) - 1] *= 0.5
    values = np.asarray(weighted)
    if not len(values) or not len(frequencies):
        return np.zeros(len(frequencies), dtype=complex), {
            "direct_points": 0, "zoom_points": 0,
        }
    nyquist = 0.5 / float(series.delta_t)
    if np.max(np.abs(frequencies), initial=0.0) > nyquist * (
            1 + 32 * np.finfo(float).eps):
        raise ValueError(
            "a requested frequency exceeds a band's Nyquist frequency")

    # A large empty gap in the requested FFT indices should not force ZoomFFT
    # to materialise every intervening bin. Splitting at roughly one input
    # length keeps both the Bluestein work arrays and output bounded.
    split = np.flatnonzero(np.diff(indices) > max(len(values), 4096)) + 1
    groups = np.split(np.arange(len(indices)), split)
    output = np.empty(len(indices), dtype=complex)
    direct_points = zoom_points = 0
    for group in groups:
        group_indices = indices[group]
        span = int(group_indices[-1] - group_indices[0] + 1)
        # Direct evaluation wins when very few requested samples span a broad
        # interval (the common case near the short, high-frequency merger
        # band). The factor four is conservative for SciPy's Bluestein FFT on
        # the supported CPU backends and avoids a multi-million-bin work array.
        direct_work = len(group) * len(values)
        zoom_work = 4 * (len(values) + span)
        if len(group) <= 2 or direct_work <= zoom_work:
            output[group] = _direct_frequency_samples(
                weighted, frequencies[group], epoch, direct_chunk_size)
            direct_points += len(group)
            continue

        first_frequency = group_indices[0] * delta_f
        last_frequency = group_indices[-1] * delta_f
        transformed = zoom_fft(
            values, [first_frequency, last_frequency], m=span,
            fs=1.0 / float(series.delta_t), endpoint=True)
        transformed *= float(series.delta_t)
        grid = first_frequency + np.arange(span) * delta_f
        offset = float(series.start_time) - float(epoch)
        transformed *= np.exp(-2j * np.pi * grid * offset)
        output[group] = transformed[group_indices - group_indices[0]]
        zoom_points += span
    return output, {
        "direct_points": direct_points,
        "zoom_points": zoom_points,
    }


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
    samples_per_cycle: float
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

    def __init__(self, source, bands, padding, t_start=None, t_end=None):
        self.source = source
        self.bands = tuple(bands)
        if not self.bands:
            raise ValueError("a multiband response needs at least one live band")
        channel_sets = {band.response.channels for band in self.bands}
        if len(channel_sets) != 1:
            raise ValueError("every frequency band must contain the same channels")
        self.channels = next(iter(channel_sets))
        self.padding = float(padding)
        self.t_start = (float(source.t_start) if t_start is None
                        else float(t_start))
        self.t_end = (float(source.t_end) if t_end is None else float(t_end))

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

    def linear_transform(self, matrix, channels):
        """Apply one constant channel transform independently to every band."""
        transformed = tuple(
            replace(
                band,
                response=band.response.linear_transform(matrix, channels),
            )
            for band in self.bands
        )
        return MultibandSparseTDIResponse(
            self.source, transformed, self.padding,
            t_start=self.t_start, t_end=self.t_end)

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

    def frequency_samples(
            self, frequencies, delta_f, channels=None, epoch=None,
            spectral_padding=0.0, chunk_size=262144,
            direct_chunk_size=262144, time_window=None,
            heterodyne=True, return_diagnostics=False):
        """Transform short time-domain bands at selected global FFT samples.

        This is a pruned transform, not a frequency-domain detector response.
        Link retardation, moving-orbit geometry, velocity corrections and TDI
        delay algebra have already been evaluated in the time domain. Each
        short block is transformed directly onto the common ``delta_f`` grid
        with ZoomFFT, and its epoch phase is referred to ``epoch`` before the
        blocks are summed.

        ``frequencies``, ``delta_f`` and ``epoch`` may be mappings keyed by
        channel. This accepts :attr:`TDIRelativeBinning.required_frequencies`
        without constructing a full observation-length frequency series.
        Frequency bands are stitched over their complete taper support;
        ``spectral_padding`` widens that support to retain modulation leakage.
        ``time_window``, when supplied, is called on absolute mission times
        before each short transform so candidates use the same analysis
        window as the fiducial data.

        By default the analytic response in each band is heterodyned to its
        centre and sampled according to its baseband width. Positive-frequency
        samples of the real TDI channel are one half of this analytic spectrum.
        Set ``heterodyne=False`` for a carrier-rate diagnostic transform.
        """
        if channels is None:
            selected = self.channels
        elif isinstance(channels, str):
            selected = (channels,)
        else:
            selected = tuple(channels)
        unknown = set(selected) - set(self.channels)
        if unknown:
            raise ValueError(f"unknown channels: {sorted(unknown)}")
        if hasattr(frequencies, "keys"):
            missing = set(selected) - set(frequencies)
            if missing:
                raise ValueError(
                    f"frequencies are missing channels: {sorted(missing)}")
        spectral_padding = float(spectral_padding)
        if not np.isfinite(spectral_padding) or spectral_padding < 0:
            raise ValueError("spectral_padding must be finite and non-negative")
        chunk_size = int(chunk_size)
        direct_chunk_size = int(direct_chunk_size)
        if chunk_size < 1 or direct_chunk_size < 1:
            raise ValueError("chunk sizes must be positive")

        grids = {}
        output = {}
        origins = {}
        for name in selected:
            requested = _mapping_value(frequencies, name)
            spacing = _mapping_value(delta_f, name)
            requested, indices = _frequency_grid_indices(requested, spacing)
            if len(requested) and requested[0] < 0:
                raise ValueError("global FFT frequencies must be non-negative")
            grids[name] = (requested, indices, float(spacing))
            output[name] = np.zeros(len(requested), dtype=complex)
            origins[name] = (self.t_start if epoch is None
                             else float(_mapping_value(epoch, name)))

        details = []
        # Transform one channel at a time. This repeats inexpensive spline
        # reconstruction but never holds all A/E/T arrays beside a ZoomFFT
        # work buffer, which is important for multi-year observations.
        for band in self.bands:
            lower = (band.f_lower - 0.5 * band.lower_overlap
                     - spectral_padding)
            upper = (band.f_upper + 0.5 * band.upper_overlap
                     + spectral_padding)
            for block_index, (start, stop) in enumerate(band.support_blocks):
                for name in selected:
                    requested, indices, spacing = grids[name]
                    active = (requested >= lower) & (requested <= upper)
                    if not np.any(active):
                        continue
                    origin = origins[name]
                    transform_frequencies = requested[active]
                    transform_indices = indices[active]
                    desired_delta_t = band.delta_t
                    complex_output = False
                    centre_frequency = 0.0
                    if heterodyne:
                        centre_index = int(np.rint(
                            0.5 * (lower + upper) / spacing))
                        centre_frequency = centre_index * spacing
                        radius = max(centre_frequency - lower,
                                     upper - centre_frequency)
                        if radius <= 0:
                            continue
                        desired_delta_t = 1.0 / (
                            band.samples_per_cycle * radius)
                        transform_frequencies = (
                            transform_frequencies - centre_frequency)
                        transform_indices = transform_indices - centre_index
                        complex_output = True
                    # Include both physical support endpoints. Adjusting the
                    # cadence slightly downward makes the interval an integer
                    # number of steps; trapezoidal endpoint weights in the
                    # transform then remove the leading cropped-block error.
                    interval_count = max(
                        1, int(np.ceil((stop - start) / desired_delta_t)))
                    effective_delta_t = (stop - start) / interval_count
                    series = band.response.to_timeseries(
                        effective_delta_t, t_start=start,
                        t_end=stop + 0.5 * effective_delta_t,
                        channels=name, chunk_size=chunk_size,
                        complex_output=complex_output)[name]
                    if time_window is not None:
                        sample_times = (float(series.start_time)
                                        + np.arange(len(series))
                                        * float(series.delta_t))
                        weights = np.asarray(time_window(sample_times),
                                             dtype=float)
                        if weights.shape != (len(series),):
                            raise ValueError(
                                "time_window must return one weight per time")
                        series *= weights
                    if heterodyne:
                        sample_times = (float(series.start_time)
                                        + np.arange(len(series))
                                        * float(series.delta_t))
                        series *= np.exp(
                            -2j * np.pi * centre_frequency
                            * (sample_times - origin))
                    values, transform_detail = _zoom_frequency_samples(
                        series, transform_frequencies, transform_indices,
                        spacing,
                        origin, direct_chunk_size)
                    if heterodyne:
                        values *= 0.5
                    output[name][active] += values
                    details.append({
                        "harmonic": band.harmonic,
                        "band_index": band.index,
                        "block_index": block_index,
                        "channel": name,
                        "time_samples": len(series),
                        "frequency_samples": int(np.count_nonzero(active)),
                        "heterodyne_frequency": centre_frequency,
                        "delta_t": effective_delta_t,
                        **transform_detail,
                    })
        if return_diagnostics:
            return output, tuple(details)
        return output

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
                samples_per_cycle=float(samples_per_cycle),
                response=response,
            ))
    return MultibandSparseTDIResponse(
        source, bands, padding, t_start=t_start, t_end=t_end)
