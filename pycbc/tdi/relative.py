"""Zero-safe relative binning for detector-projected TDI channels.

The ordinary relative-binning approximation interpolates the ratio between a
candidate waveform and a fiducial waveform. A TDI channel contains structural
transfer zeros as well as source- and sky-dependent cancellations, so that
ratio is not defined everywhere. This module keeps the ordinary approximation
where the fiducial is well conditioned, excludes samples whose PSD is invalid,
and evaluates the remaining valid samples directly.
"""

from dataclasses import dataclass

import numpy as np

from pycbc.filter.matchedfilter import get_cutoff_indices


def _cutoff_indices(series, low_frequency_cutoff, high_frequency_cutoff):
    ntime = 2 * (len(series) - 1)
    return get_cutoff_indices(
        low_frequency_cutoff, high_frequency_cutoff,
        float(series.delta_f), ntime)


def _contiguous_runs(mask):
    """Return half-open index runs on which a Boolean mask is true."""
    padded = np.pad(np.asarray(mask, dtype=np.int8), 1)
    changes = np.diff(padded)
    return tuple(zip(np.flatnonzero(changes == 1),
                     np.flatnonzero(changes == -1), strict=True))


def _normalise_edges(edges, kmin, kmax):
    edges = np.asarray(edges, dtype=int)
    if edges.ndim != 1:
        raise ValueError("bin indices must be one-dimensional")
    edges = edges[(edges >= kmin) & (edges <= kmax)]
    edges = np.unique(np.concatenate(([kmin], edges, [kmax])))
    if len(edges) < 2:
        raise ValueError("at least two distinct bin indices are required")
    return edges


def adaptive_tdi_frequency_bins(
        reference, training_waveforms, psd, bin_indices=None,
        low_frequency_cutoff=None, high_frequency_cutoff=None,
        relative_tolerance=1e-3, reference_floor=1e-10,
        valid_mask=None, max_refinements=24, return_diagnostics=False):
    """Refine frequency bins using a noise-weighted waveform-error test.

    The grid is trained before sampling and must then remain fixed, because
    relative-binning summary data depend on its edges. Intervals containing
    invalid PSD samples or a poorly conditioned fiducial are deliberately not
    refined: :class:`TDIRelativeBinning` excludes the former and evaluates the
    latter directly.

    Parameters
    ----------
    reference, psd : FrequencySeries
        Full-resolution fiducial waveform and its channel PSD.
    training_waveforms : sequence of array-like
        Full-resolution detector-projected waveforms spanning the intended
        posterior neighbourhood.
    bin_indices : array-like, optional
        Initial FFT-bin edges. The analysis-band endpoints are added.
    relative_tolerance : float, optional
        Maximum whitened interpolation-error norm divided by the candidate
        norm in every retained interval.
    reference_floor : float, optional
        Fiducial magnitudes below this fraction of the in-band maximum use the
        direct fallback.
    valid_mask : array-like, optional
        Additional mask, for example one excluding guarded TDI transfer zeros.
    return_diagnostics : bool, optional
        Also return refinement counts and the largest final interval error.
    """
    if relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive")
    if not 0 <= reference_floor < 1:
        raise ValueError("reference_floor must lie in [0, 1)")
    if max_refinements < 0:
        raise ValueError("max_refinements must be non-negative")

    reference_values = np.asarray(reference, dtype=complex)
    noise = np.asarray(psd, dtype=float)
    if reference_values.ndim != 1 or noise.shape != reference_values.shape:
        raise ValueError("reference and psd must be matching 1-D arrays")
    training = tuple(np.asarray(item, dtype=complex)
                     for item in training_waveforms)
    if not training:
        raise ValueError("at least one training waveform is required")
    if any(item.shape != reference_values.shape for item in training):
        raise ValueError("training waveforms must match the reference shape")
    if not np.isclose(float(reference.delta_f), float(psd.delta_f)):
        raise ValueError("reference and psd must have the same delta_f")

    kmin, kmax = _cutoff_indices(
        reference, low_frequency_cutoff, high_frequency_cutoff)
    if bin_indices is None:
        bin_indices = (kmin, kmax)
    edges = _normalise_edges(bin_indices, kmin, kmax)

    valid = np.isfinite(noise) & (noise > 0)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != valid.shape:
            raise ValueError("valid_mask must match the reference shape")
        valid &= valid_mask
    in_band = np.zeros(len(valid), dtype=bool)
    in_band[kmin:kmax] = True
    valid &= in_band
    peak = np.max(np.abs(reference_values[valid]), initial=0.0)
    safe = valid & (np.abs(reference_values) > reference_floor * peak)
    boundaries = []
    for start0, stop0 in _contiguous_runs(safe[kmin:kmax]):
        start, stop = start0 + kmin, stop0 + kmin
        if stop - start >= 2:
            boundaries.extend((start, stop - 1))
    edges = np.unique(np.concatenate((edges, boundaries)))

    deepest = 0
    largest_error = 0.0
    for depth in range(int(max_refinements) + 1):
        additions = []
        largest_error = 0.0
        for left, right in zip(edges[:-1], edges[1:], strict=True):
            if right - left < 2 or not np.all(safe[left:right + 1]):
                continue
            index = np.arange(left, right)
            fraction = (index - left) / (right - left)
            href = reference_values[index]
            weight = 1.0 / noise[index]
            worst = 0.0
            for candidate in training:
                ratio_left = candidate[left] / reference_values[left]
                ratio_right = candidate[right] / reference_values[right]
                ratio = ratio_left + fraction * (ratio_right - ratio_left)
                residual = candidate[index] - href * ratio
                error2 = np.sum(np.abs(residual) ** 2 * weight)
                norm2 = np.sum(np.abs(candidate[index]) ** 2 * weight)
                error = np.sqrt(error2 / max(norm2, np.finfo(float).tiny))
                worst = max(worst, float(error))
            largest_error = max(largest_error, worst)
            if worst > relative_tolerance:
                additions.append((left + right) // 2)

        if not additions:
            break
        if depth == max_refinements:
            raise RuntimeError(
                "frequency bins did not reach the requested tolerance after "
                f"{max_refinements} refinements")
        edges = np.unique(np.concatenate((edges, additions)))
        deepest = depth + 1

    diagnostics = {
        "bin_count": len(edges) - 1,
        "deepest_refinement": deepest,
        "largest_interval_error": largest_error,
        "relative_tolerance": float(relative_tolerance),
    }
    if return_diagnostics:
        return edges, diagnostics
    return edges


@dataclass(frozen=True)
class _RelativeRun:
    edges: np.ndarray
    frequencies: np.ndarray
    reference: np.ndarray
    a0: np.ndarray
    a1: np.ndarray
    b0: np.ndarray
    b1: np.ndarray


class _ChannelSummary:
    def __init__(self, data, reference, psd, bin_indices,
                 low_frequency_cutoff, high_frequency_cutoff,
                 reference_floor, valid_mask):
        self.delta_f = float(reference.delta_f)
        if not np.isclose(self.delta_f, float(data.delta_f)) or not np.isclose(
                self.delta_f, float(psd.delta_f)):
            raise ValueError("data, reference and psd must share delta_f")
        if len(data) != len(reference) or len(psd) != len(reference):
            raise ValueError("data, reference and psd must share a length")

        observed = np.asarray(data, dtype=complex)
        fiducial = np.asarray(reference, dtype=complex)
        noise = np.asarray(psd, dtype=float)
        kmin, kmax = _cutoff_indices(
            reference, low_frequency_cutoff, high_frequency_cutoff)
        base_edges = _normalise_edges(bin_indices, kmin, kmax)

        valid = np.isfinite(noise) & (noise > 0)
        if valid_mask is not None:
            valid_mask = np.asarray(valid_mask, dtype=bool)
            if valid_mask.shape != valid.shape:
                raise ValueError("valid_mask must match the channel length")
            valid &= valid_mask
        in_band = np.zeros(len(valid), dtype=bool)
        in_band[kmin:kmax] = True
        valid &= in_band

        peak = np.max(np.abs(fiducial[valid]), initial=0.0)
        ratio_safe = valid & (np.abs(fiducial) > reference_floor * peak)
        direct = set(np.flatnonzero(valid & ~ratio_safe))
        runs = []
        for start0, stop0 in _contiguous_runs(ratio_safe[kmin:kmax]):
            start, stop = start0 + kmin, stop0 + kmin
            # A right-edge ratio is needed to integrate [left, right), so the
            # final safe sample in each run is evaluated directly.
            if stop - start < 2:
                direct.update(range(start, stop))
                continue
            stop_edge = stop - 1
            edges = base_edges[
                (base_edges > start) & (base_edges < stop_edge)]
            edges = np.concatenate(([start], edges, [stop_edge]))
            if len(edges) < 2:
                direct.update(range(start, stop))
                continue
            direct.add(stop_edge)
            runs.append(self._build_run(
                observed, fiducial, noise, edges))

        self.runs = tuple(runs)
        self.direct_indices = np.asarray(sorted(direct), dtype=int)
        self.direct_data = np.conjugate(observed[self.direct_indices]) \
            / noise[self.direct_indices]
        self.direct_weight = 1.0 / noise[self.direct_indices]
        required = [self.direct_indices]
        required.extend(run.edges for run in self.runs)
        self.required_indices = np.unique(np.concatenate(required))
        self.excluded_count = int(np.count_nonzero(in_band & ~valid))
        self.relative_bin_count = sum(len(run.edges) - 1
                                      for run in self.runs)

    def _build_run(self, observed, fiducial, noise, edges):
        a0, a1, b0, b1 = [], [], [], []
        for left, right in zip(edges[:-1], edges[1:], strict=True):
            index = np.arange(left, right)
            offset = index - left
            width = right - left
            cross = (np.conjugate(observed[index]) * fiducial[index]
                     / noise[index])
            power = np.abs(fiducial[index]) ** 2 / noise[index]
            a0.append(4 * self.delta_f * np.sum(cross))
            a1.append(4 * self.delta_f
                      * np.sum(cross * offset / width))
            b0.append(4 * self.delta_f * np.sum(power))
            b1.append(4 * self.delta_f
                      * np.sum(power * offset / width))
        return _RelativeRun(
            edges=np.asarray(edges, dtype=int),
            frequencies=np.asarray(edges * self.delta_f, dtype=float),
            reference=np.asarray(fiducial[edges], dtype=np.complex128),
            a0=np.asarray(a0, dtype=np.complex128),
            a1=np.asarray(a1, dtype=np.complex128),
            b0=np.asarray(b0, dtype=float),
            b1=np.asarray(b1, dtype=float),
        )

    def evaluate_sparse(self, values):
        from pycbc.inference.models.relbin_cpu import likelihood_parts_det

        values = np.asarray(values, dtype=np.complex128)
        if values.shape != self.required_indices.shape:
            raise ValueError(
                "sparse channel values must match required_indices")
        filt, norm = 0j, 0.0
        for run in self.runs:
            positions = np.searchsorted(self.required_indices, run.edges)
            candidate = np.ascontiguousarray(values[positions])
            part_filt, part_norm = likelihood_parts_det(
                run.frequencies, 0.0, candidate, run.reference,
                run.a0, run.a1, run.b0, run.b1)
            filt += part_filt
            norm += part_norm

        if len(self.direct_indices):
            positions = np.searchsorted(
                self.required_indices, self.direct_indices)
            candidate = values[positions]
            filt += np.conjugate(
                4 * self.delta_f * np.sum(self.direct_data * candidate))
            norm += 4 * self.delta_f * np.sum(
                np.abs(candidate) ** 2 * self.direct_weight)
        return complex(filt), float(np.real(norm))


class TDIRelativeBinning:
    """Relative-binning summaries for already-projected TDI channels.

    Candidate generation needs only :attr:`required_frequencies`. The same
    fixed grid must be used throughout a sampling epoch. Call
    :meth:`evaluate_sparse` with channel arrays evaluated in that order to
    obtain ``(<d|h>, <h|h>)`` using the PyCBC detector-frame relbin kernel plus
    an exact fallback around ill-conditioned fiducial zeros.
    """

    def __init__(
            self, data, reference, psds, bin_indices=None,
            low_frequency_cutoff=None, high_frequency_cutoff=None,
            epsilon=0.5, gammas=None, reference_floor=1e-10,
            valid_masks=None):
        if not 0 <= reference_floor < 1:
            raise ValueError("reference_floor must lie in [0, 1)")
        self.channels = tuple(reference)
        if set(data) != set(self.channels) or set(psds) != set(self.channels):
            raise ValueError("data, reference and psds need identical channels")
        valid_masks = {} if valid_masks is None else valid_masks
        self._summaries = {}
        for name in self.channels:
            ref = reference[name]
            flow = self._channel_value(low_frequency_cutoff, name)
            fhigh = self._channel_value(high_frequency_cutoff, name)
            if bin_indices is None:
                from pycbc.inference.models.relbin import setup_bins
                frequency = np.asarray(ref.sample_frequencies)
                kmin, kmax = _cutoff_indices(ref, flow, fhigh)
                edges = setup_bins(
                    frequency, kmin * ref.delta_f, kmax * ref.delta_f,
                    gammas=gammas, eps=float(epsilon))
            elif hasattr(bin_indices, "keys"):
                edges = bin_indices[name]
            else:
                edges = bin_indices
            mask = valid_masks.get(name)
            self._summaries[name] = _ChannelSummary(
                data[name], ref, psds[name], edges, flow, fhigh,
                float(reference_floor), mask)

    @staticmethod
    def _channel_value(value, channel):
        if value is None or not hasattr(value, "keys"):
            return value
        return value[channel]

    @property
    def required_indices(self):
        return {name: summary.required_indices.copy()
                for name, summary in self._summaries.items()}

    @property
    def required_frequencies(self):
        return {
            name: summary.required_indices * summary.delta_f
            for name, summary in self._summaries.items()
        }

    @property
    def diagnostics(self):
        return {
            name: {
                "relative_bins": summary.relative_bin_count,
                "direct_points": len(summary.direct_indices),
                "excluded_points": summary.excluded_count,
                "required_points": len(summary.required_indices),
            }
            for name, summary in self._summaries.items()
        }

    def compress(self, waveforms):
        """Select the fixed required frequency samples from full waveforms."""
        return {
            name: np.asarray(waveforms[name])[summary.required_indices]
            for name, summary in self._summaries.items()
        }

    def evaluate(self, waveforms):
        """Evaluate full-resolution candidate waveforms through the summary."""
        return self.evaluate_sparse(self.compress(waveforms))

    def evaluate_sparse(self, waveforms):
        """Return summed ``(<d|h>, <h|h>)`` from required frequency values."""
        if set(waveforms) != set(self.channels):
            raise ValueError("candidate waveforms must match summary channels")
        filt, norm = 0j, 0.0
        for name, summary in self._summaries.items():
            part_filt, part_norm = summary.evaluate_sparse(waveforms[name])
            filt += part_filt
            norm += part_norm
        return filt, norm

    def loglr(self, waveforms, sparse=False):
        """Return the unmarginalized Gaussian log-likelihood ratio."""
        filt, norm = (self.evaluate_sparse(waveforms) if sparse
                      else self.evaluate(waveforms))
        return float(np.real(filt) - 0.5 * norm)
