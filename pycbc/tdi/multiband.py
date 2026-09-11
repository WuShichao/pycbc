"""Multirate time-domain TDI built from pre-response harmonic windows."""

from dataclasses import dataclass, replace

import numpy as np

from pycbc.tdi.onthefly import (
    PreparedSparseTDI,
    SparseTDIResponse,
    adaptive_sparse_tdi_response,
    delay_padding,
    harmonic_windows,
    sparse_windowed_channels_terms,
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


@dataclass(frozen=True)
class _PreparedBand:
    """Source-independent geometry and coverage for one frequency band."""

    template: TDIBandResponse
    prepared: object
    coverage: tuple


class PreparedMultibandTDI:
    """Prepared orbit and delay geometry for repeated multiband projections.

    Construction is the cold path. :meth:`project` accepts another harmonic
    source and performs only sky projection and waveform evaluation on the
    fixed grids. A candidate whose frequency-band support leaves the prepared
    fiducial coverage is rejected instead of being silently truncated; build
    the preparation from a wider training envelope in that case.
    """

    def __init__(self, response, prepared_bands, band_edges, overlaps,
                 harmonics):
        self.response = response
        self.prepared_bands = tuple(prepared_bands)
        self.band_edges = np.asarray(band_edges, dtype=float)
        self.overlaps = np.asarray(overlaps, dtype=float)
        self.harmonics = tuple(harmonics)
        self.channels = response.channels
        self.padding = response.padding
        self.t_start = response.t_start
        self.t_end = response.t_end

    def project(self, source, lamb, beta, matrix=None, channels=None,
                source_batch_size=1, source_workers=1):
        """Project one candidate on the fixed multiband geometry grids.

        A constant ``matrix`` may form named output ``channels`` before the
        response splines are built. Frequency-window bands sharing a source
        harmonic can be evaluated together by increasing
        ``source_batch_size``. The default reflects the measured pyEFPEHM
        optimum; larger source arrays are not always faster.
        ``source_workers`` optionally evaluates independent batches in a
        thread pool. Keep it at one unless the source implementation supports
        concurrent read-only evaluation.
        """
        if matrix is None:
            if channels is not None:
                raise ValueError("channels requires a channel-transform matrix")
            output_channels = self.channels
        else:
            matrix = np.asarray(matrix, dtype=float)
            if channels is None:
                raise ValueError("matrix requires output channel names")
            output_channels = tuple(channels)
            if matrix.shape != (len(output_channels), len(self.channels)):
                raise ValueError(
                    "matrix shape must be (output channels, native channels)")
            if len(set(output_channels)) != len(output_channels):
                raise ValueError("output channel names must be distinct")
        partitions = frequency_partition_sources(
            source, self.band_edges, overlap=self.overlaps,
            harmonics=self.harmonics)
        windows = {
            (harmonic, index): window
            for harmonic, items in partitions.items()
            for index, window in enumerate(items)
        }
        candidates = []
        scale = max(1.0, self.t_end - self.t_start)
        tolerance = 64 * np.finfo(float).eps * scale
        for item in self.prepared_bands:
            template = item.template
            key = (template.harmonic, template.index)
            if key not in windows:
                raise ValueError(f"candidate has no prepared band {key!r}")
            windowed = windows[key]
            candidate_support = harmonic_windows(
                windowed, template.harmonic, self.t_start, self.t_end,
                padding=self.padding)
            for low, high in candidate_support:
                covered = any(
                    low >= outer_low - tolerance
                    and high <= outer_high + tolerance
                    for outer_low, outer_high in item.coverage)
                if not covered:
                    raise ValueError(
                        f"candidate support {(low, high)} leaves prepared "
                        f"coverage for band {key!r}")
            geometry = item.prepared.geometries[template.harmonic]
            candidates.append((item, windowed, candidate_support, geometry))

        source_batch_size = int(source_batch_size)
        source_workers = int(source_workers)
        if source_batch_size < 1:
            raise ValueError("source_batch_size must be positive")
        if source_workers < 1:
            raise ValueError("source_workers must be positive")
        groups = [
            candidates[first:first + source_batch_size]
            for first in range(0, len(candidates), source_batch_size)
        ]

        def evaluate_group(group):
            return sparse_windowed_channels_terms(
                [windowed for _, windowed, _, _ in group],
                [item.template.harmonic for item, _, _, _ in group],
                [geometry for _, _, _, geometry in group],
                lamb, beta, source_batch_size=len(group))

        if source_workers == 1 or len(groups) < 2:
            grouped_brackets = tuple(evaluate_group(group) for group in groups)
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(
                    max_workers=min(source_workers, len(groups))) as pool:
                grouped_brackets = tuple(pool.map(evaluate_group, groups))
        native_brackets = tuple(
            brackets for group in grouped_brackets for brackets in group)
        bands = []
        for (item, windowed, candidate_support, geometry), brackets in zip(
                candidates, native_brackets, strict=True):
            template = item.template
            if matrix is not None:
                native = np.stack([
                    brackets[name] for name in self.channels])
                transformed = matrix @ native
                brackets = dict(zip(
                    output_channels, transformed, strict=True))
            projected = SparseTDIResponse(windowed, ({
                "harmonic": template.harmonic,
                "grid": geometry.grid,
                "brackets": brackets,
                "support": candidate_support,
            },))
            bands.append(replace(template, response=projected))
        return MultibandSparseTDIResponse(
            source, bands, self.padding,
            t_start=self.t_start, t_end=self.t_end)


def prepare_multiband_tdi_response(
        response, orbit, channel_terms, velocity_order=1, links=None):
    """Prepare fixed geometry from a validated fiducial multiband response."""
    if tuple(response.channels) != tuple(channel_terms):
        raise ValueError(
            "response channels must match channel_terms; prepare the native "
            "channels before applying a linear channel transform")
    grouped = {}
    for band in response.bands:
        grouped.setdefault(band.harmonic, []).append(band)
    harmonics = tuple(grouped)
    if not harmonics:
        raise ValueError("response contains no frequency bands")
    first = sorted(grouped[harmonics[0]], key=lambda item: item.index)
    expected_indices = list(range(len(first)))
    if [item.index for item in first] != expected_indices:
        raise ValueError("fiducial bands must have consecutive indices")
    band_edges = np.asarray(
        [first[0].f_lower] + [item.f_upper for item in first], dtype=float)
    overlaps = np.asarray(
        [item.upper_overlap for item in first[:-1]], dtype=float)
    for harmonic in harmonics[1:]:
        items = sorted(grouped[harmonic], key=lambda item: item.index)
        edges = np.asarray(
            [items[0].f_lower] + [item.f_upper for item in items])
        current_overlaps = np.asarray(
            [item.upper_overlap for item in items[:-1]])
        if (len(items) != len(first) or not np.array_equal(edges, band_edges)
                or not np.array_equal(current_overlaps, overlaps)):
            raise ValueError("every harmonic must use the same partition")

    options = {} if links is None else {"links": links}
    prepared_bands = []
    for band in response.bands:
        records = tuple(band.response.responses)
        if len(records) != 1 or records[0]["harmonic"] != band.harmonic:
            raise ValueError("each prepared band must contain one harmonic")
        prepared = PreparedSparseTDI(
            orbit, channel_terms,
            {band.harmonic: np.asarray(records[0]["grid"])},
            velocity_order=velocity_order, **options)
        prepared_bands.append(_PreparedBand(
            template=band, prepared=prepared,
            coverage=band.support_blocks))
    return PreparedMultibandTDI(
        response, prepared_bands, band_edges, overlaps, harmonics)


def prepare_multiband_tdi(
        source, orbit, channel_terms, lamb, beta, band_edges, overlap=0.0,
        t_start=None, t_end=None, samples_per_cycle=4.0,
        geometry_step=86400.0, minimum_grid_points=16, velocity_order=1,
        links=None, padding=None, harmonics=None):
    """Prepare a fixed narrow-band geometry without adaptive window chasing.

    This cold-path constructor is intended for a likelihood epoch. The band
    windows determine their padded time coverage, while ``geometry_step`` and
    ``minimum_grid_points`` control a fixed spline grid in each live block.
    Validate these two settings against a dense fiducial response before using
    the returned preparation for candidates.
    """
    if samples_per_cycle <= 2:
        raise ValueError("samples_per_cycle must exceed the Nyquist minimum")
    geometry_step = float(geometry_step)
    minimum_grid_points = int(minimum_grid_points)
    if not np.isfinite(geometry_step) or geometry_step <= 0:
        raise ValueError("geometry_step must be positive and finite")
    if minimum_grid_points < 4:
        raise ValueError("minimum_grid_points must be at least four")
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

    prepared_bands = []
    response_bands = []
    prepare_options = {} if links is None else {"links": links}
    for harmonic, windows in partitions.items():
        for index, windowed in enumerate(windows):
            coverage = harmonic_windows(
                windowed, harmonic, t_start, t_end, padding=padding)
            pieces = []
            for low, high in coverage:
                count = max(
                    minimum_grid_points,
                    int(np.ceil((high - low) / geometry_step)) + 1)
                pieces.append(np.linspace(low, high, count))
            if not pieces:
                continue
            grid = np.unique(np.concatenate(pieces))
            prepared = PreparedSparseTDI(
                orbit, channel_terms, {harmonic: grid},
                velocity_order=velocity_order, **prepare_options)
            projected = prepared.project(
                windowed, lamb, beta, support_padding=padding)
            highest = windowed.f_upper + 0.5 * windowed.upper_overlap
            template = TDIBandResponse(
                harmonic=harmonic,
                index=index,
                f_lower=windowed.f_lower,
                f_upper=windowed.f_upper,
                lower_overlap=windowed.lower_overlap,
                upper_overlap=windowed.upper_overlap,
                delta_t=1.0 / (float(samples_per_cycle) * highest),
                samples_per_cycle=float(samples_per_cycle),
                response=projected,
            )
            response_bands.append(template)
            prepared_bands.append(_PreparedBand(
                template=template, prepared=prepared,
                coverage=tuple(coverage)))
    response = MultibandSparseTDIResponse(
        source, response_bands, padding,
        t_start=t_start, t_end=t_end)
    edges = np.asarray(band_edges, dtype=float)
    first_windows = next(iter(partitions.values()))
    overlaps = np.asarray(
        [window.upper_overlap for window in first_windows[:-1]], dtype=float)
    return PreparedMultibandTDI(
        response, prepared_bands, edges, overlaps, tuple(partitions))


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

    def prepare_frequency_sampler(self, frequencies, delta_f, channels=None,
                                  epoch=None, spectral_padding=0.0,
                                  time_window=None,
                                  max_matrix_bytes=268435456,
                                  carrier_batch_size=4):
        """Precompute fixed sparse-Fourier kernels for repeated candidates."""
        return PreparedMultibandFrequencySampler(
            self, frequencies, delta_f, channels=channels, epoch=epoch,
            spectral_padding=spectral_padding, time_window=time_window,
            max_matrix_bytes=max_matrix_bytes,
            carrier_batch_size=carrier_batch_size)

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


class PreparedMultibandFrequencySampler:
    """Fixed Fourier kernels for a relbin grid and multiband partition.

    The expensive complex exponentials are a cold-path object. Candidate
    evaluation reconstructs each analytic, heterodyned-width band on the same
    time nodes and applies small matrix products. A hard memory limit prevents
    an accidentally broad partition from materialising a dense two-year
    Fourier operator.
    """

    def __init__(self, response, frequencies, delta_f, channels=None,
                 epoch=None, spectral_padding=0.0, time_window=None,
                 max_matrix_bytes=268435456, carrier_batch_size=4):
        if channels is None:
            selected = response.channels
        elif isinstance(channels, str):
            selected = (channels,)
        else:
            selected = tuple(channels)
        if not selected:
            raise ValueError("at least one channel is required")
        unknown = set(selected) - set(response.channels)
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
        max_matrix_bytes = int(max_matrix_bytes)
        if max_matrix_bytes < 1:
            raise ValueError("max_matrix_bytes must be positive")
        self.carrier_batch_size = int(carrier_batch_size)
        if self.carrier_batch_size < 1:
            raise ValueError("carrier_batch_size must be positive")

        self.channels = selected
        self.band_signature = tuple(
            (band.harmonic, band.index, band.f_lower, band.f_upper,
             band.lower_overlap, band.upper_overlap,
             band.samples_per_cycle)
            for band in response.bands)
        self.output_sizes = {}
        grids = {}
        origins = {}
        spacings = []
        for name in selected:
            requested, indices = _frequency_grid_indices(
                _mapping_value(frequencies, name),
                _mapping_value(delta_f, name))
            if len(requested) and requested[0] < 0:
                raise ValueError("global FFT frequencies must be non-negative")
            spacing = float(_mapping_value(delta_f, name))
            grids[name] = (requested, indices, spacing)
            spacings.append(spacing)
            origins[name] = (response.t_start if epoch is None
                             else float(_mapping_value(epoch, name)))
            self.output_sizes[name] = len(requested)
        if spacings and not np.allclose(spacings, spacings[0], rtol=0,
                                       atol=32 * np.finfo(float).eps):
            raise ValueError(
                "prepared channels must share one delta_f time grid")
        common_spacing = spacings[0] if spacings else None

        tasks = []
        matrix_bytes = 0
        kernel_count = 0
        for band_position, band in enumerate(response.bands):
            lower = (band.f_lower - 0.5 * band.lower_overlap
                     - spectral_padding)
            upper = (band.f_upper + 0.5 * band.upper_overlap
                     + spectral_padding)
            centre_index = int(np.rint(
                0.5 * (lower + upper) / common_spacing))
            centre = centre_index * common_spacing
            radius = max(centre - lower, upper - centre)
            if radius <= 0:
                continue
            desired_delta_t = 1.0 / (band.samples_per_cycle * radius)
            active_channels = {}
            for name in selected:
                requested, _, _ = grids[name]
                active = np.flatnonzero(
                    (requested >= lower) & (requested <= upper))
                if len(active):
                    active_channels[name] = active
            if not active_channels:
                continue

            for block_index, (start, stop) in enumerate(band.support_blocks):
                interval_count = max(
                    1, int(np.ceil((stop - start) / desired_delta_t)))
                times = np.linspace(start, stop, interval_count + 1)
                weights = np.ones(len(times))
                weights[[0, -1]] = 0.5
                if time_window is not None:
                    window = np.asarray(time_window(times), dtype=float)
                    if window.shape != times.shape:
                        raise ValueError(
                            "time_window must return one weight per time")
                    weights *= window
                channel_tasks = {}
                shared_kernels = {}
                for name, positions in active_channels.items():
                    requested, _, _ = grids[name]
                    relative_times = times - origins[name]
                    key = (
                        float(origins[name]),
                        requested[positions].tobytes(),
                    )
                    kernel = shared_kernels.get(key)
                    if kernel is None:
                        kernel = np.exp(
                            -2j * np.pi
                            * requested[positions, None] * relative_times)
                        kernel *= (0.5 * (times[1] - times[0])
                                   * weights[None, :])
                        matrix_bytes += kernel.nbytes
                        kernel_count += 1
                        if matrix_bytes > max_matrix_bytes:
                            raise MemoryError(
                                "prepared Fourier kernels exceed "
                                f"max_matrix_bytes={max_matrix_bytes}; use "
                                "more frequency bands or a smaller relbin "
                                "grid")
                        shared_kernels[key] = kernel
                    channel_tasks[name] = (positions, kernel)
                tasks.append({
                    "band_position": band_position,
                    "block_index": block_index,
                    "times": times,
                    "channels": channel_tasks,
                })
        self.tasks = tuple(tasks)
        self.diagnostics = {
            "tasks": len(tasks),
            "time_samples": sum(len(task["times"]) for task in tasks),
            "matrix_bytes": matrix_bytes,
            "matrix_megabytes": matrix_bytes / 2 ** 20,
            "kernel_count": kernel_count,
        }

    def evaluate(self, response, workers=1):
        """Return required frequency samples for one compatible candidate.

        ``workers`` may parallelize independent, read-only frequency bands.
        Results are accumulated serially in task order, so worker threads do
        not write shared output arrays or change floating-point reduction
        order.
        """
        workers = int(workers)
        if workers < 1:
            raise ValueError("workers must be positive")
        signature = tuple(
            (band.harmonic, band.index, band.f_lower, band.f_upper,
             band.lower_overlap, band.upper_overlap,
             band.samples_per_cycle)
            for band in response.bands)
        if signature != self.band_signature:
            raise ValueError(
                "candidate bands do not match the prepared Fourier sampler")
        output = {
            name: np.zeros(size, dtype=complex)
            for name, size in self.output_sizes.items()
        }
        carrier_phases = {}
        phase_groups = {}
        for task_index, task in enumerate(self.tasks):
            band = response.bands[task["band_position"]]
            windowed = band.response.source
            base = getattr(windowed, "source", windowed)
            phase_groups.setdefault(
                (id(base), band.harmonic),
                (base, band.harmonic, []))[2].append(task_index)
        phase_jobs = []
        for base, harmonic, task_indices in phase_groups.values():
            for first in range(0, len(task_indices),
                               self.carrier_batch_size):
                batch = task_indices[first:first + self.carrier_batch_size]
                phase_jobs.append((base, harmonic, tuple(batch)))

        def evaluate_phases(job):
            base, harmonic, task_indices = job
            sizes = [len(self.tasks[index]["times"])
                     for index in task_indices]
            joined = np.concatenate([
                self.tasks[index]["times"] for index in task_indices])
            phases = base.carrier_phase(harmonic, joined)
            offset = 0
            output_phases = []
            for task_index, size in zip(task_indices, sizes, strict=True):
                output_phases.append(
                    (task_index, phases[offset:offset + size]))
                offset += size
            return tuple(output_phases)

        def evaluate_task(index_and_task):
            task_index, task = index_and_task
            band = response.bands[task["band_position"]]
            names = tuple(task["channels"])
            records = band.response.sample_brackets(
                task["times"], channels=names)
            if len(records) != 1:
                raise ValueError(
                    "each multiband response must contain one harmonic record")
            carrier = np.exp(1j * carrier_phases[task_index])
            values = {
                name: records[0][name] * carrier for name in names}
            kernel_groups = {}
            for name, (positions, kernel) in task["channels"].items():
                kernel_groups.setdefault(
                    (id(kernel), positions.tobytes()),
                    (positions, kernel, []))[2].append(name)
            contributions = []
            for positions, kernel, group_names in kernel_groups.values():
                samples = np.stack([values[name] for name in group_names])
                transformed = samples @ kernel.T
                for row, name in enumerate(group_names):
                    contributions.append(
                        (name, positions, transformed[row]))
            return tuple(contributions)

        if workers == 1 or len(self.tasks) < 2:
            phase_results = map(evaluate_phases, phase_jobs)
            for result in phase_results:
                carrier_phases.update(result)
            task_results = map(evaluate_task, enumerate(self.tasks))
            for result in task_results:
                for name, positions, transformed in result:
                    output[name][positions] += transformed
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(
                    max_workers=min(workers, len(self.tasks))) as pool:
                for result in pool.map(evaluate_phases, phase_jobs):
                    carrier_phases.update(result)
                task_results = tuple(pool.map(
                    evaluate_task, enumerate(self.tasks)))
            for result in task_results:
                for name, positions, transformed in result:
                    output[name][positions] += transformed
        return output


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
