"""Tests for zero-safe relative binning of projected TDI channels."""

import numpy as np

from pycbc.tdi.relative import (
    TDIRelativeBinning,
    adaptive_tdi_frequency_bins,
)
from pycbc.types import FrequencySeries


def _series(values, delta_f=1e-3):
    return FrequencySeries(np.asarray(values), delta_f=delta_f, copy=False)


def _exact_parts(data, candidate, psd, valid, delta_f):
    cross = np.conjugate(data[valid]) * candidate[valid] / psd[valid]
    filt = np.conjugate(4 * delta_f * np.sum(cross))
    norm = 4 * delta_f * np.sum(
        np.abs(candidate[valid]) ** 2 / psd[valid])
    return filt, norm


def test_relative_binning_falls_back_at_reference_zeros():
    size, delta_f = 1025, 1e-3
    frequency = np.arange(size) * delta_f
    reference = (1 + 0.2 * frequency) * np.exp(
        1j * (0.4 + 3 * frequency))
    data = (0.7 - 0.1 * frequency) * np.exp(
        1j * (0.2 - 2 * frequency))
    candidate = (1.3 - 0.25j) * reference

    # A source/sky cancellation has valid noise and therefore needs the exact
    # fallback. PSD nulls and a caller-guarded TDI zero are excluded instead.
    reference[121:124] = 0
    candidate[121:124] = (0.04 + 0.02j) * np.arange(1, 4)
    psd = 2 + frequency
    psd[200] = np.inf
    psd[201] = 0
    psd[202] = np.nan
    valid_mask = np.ones(size, dtype=bool)
    valid_mask[300] = False

    flow, fhigh = 0.01, 0.8
    kmin, kmax = int(flow / delta_f), int(fhigh / delta_f)
    edges = np.arange(kmin, kmax + 1, 37)
    summary = TDIRelativeBinning(
        {"A": _series(data, delta_f)},
        {"A": _series(reference, delta_f)},
        {"A": _series(psd, delta_f)},
        bin_indices=edges,
        low_frequency_cutoff=flow,
        high_frequency_cutoff=fhigh,
        reference_floor=1e-8,
        valid_masks={"A": valid_mask},
    )

    valid = np.zeros(size, dtype=bool)
    valid[kmin:kmax] = True
    valid &= np.isfinite(psd) & (psd > 0) & valid_mask
    expected_filt, expected_norm = _exact_parts(
        data, candidate, psd, valid, delta_f)
    filt, norm = summary.evaluate({"A": _series(candidate, delta_f)})

    assert np.allclose(filt, expected_filt, rtol=2e-14, atol=2e-14)
    assert np.allclose(norm, expected_norm, rtol=2e-14, atol=2e-14)
    required = summary.required_indices["A"]
    assert set(range(121, 124)).issubset(required)
    assert not set(range(200, 203)).intersection(required)
    assert 300 not in required
    assert summary.diagnostics["A"]["direct_points"] >= 3
    assert summary.diagnostics["A"]["excluded_points"] == 4


def test_sparse_and_full_relative_binning_interfaces_agree():
    size, delta_f = 513, 2e-3
    frequency = np.arange(size) * delta_f
    reference = (2 + frequency) * np.exp(2j * frequency)
    data = (0.4 - 0.3j) * reference
    candidate = (0.9 + 0.1j) * reference
    psd = np.ones(size)
    summary = TDIRelativeBinning(
        {"A": _series(data, delta_f)},
        {"A": _series(reference, delta_f)},
        {"A": _series(psd, delta_f)},
        bin_indices=np.arange(5, 401, 23),
        low_frequency_cutoff=0.01,
        high_frequency_cutoff=0.8,
    )

    full = summary.evaluate({"A": _series(candidate, delta_f)})
    sparse = summary.evaluate_sparse(summary.compress({
        "A": _series(candidate, delta_f),
    }))
    assert np.allclose(full, sparse, rtol=0, atol=0)


def test_relative_binning_requests_only_required_response_samples():
    size, delta_f = 513, 2e-3
    frequency = np.arange(size) * delta_f
    reference = (2 + frequency) * np.exp(2j * frequency)
    data = (0.4 - 0.3j) * reference
    candidate = (0.9 + 0.1j) * reference
    psd = np.ones(size)
    summary = TDIRelativeBinning(
        {"A": _series(data, delta_f)},
        {"A": _series(reference, delta_f)},
        {"A": _series(psd, delta_f)},
        bin_indices=np.arange(5, 401, 23),
        low_frequency_cutoff=0.01,
        high_frequency_cutoff=0.8,
    )

    class SparseResponse:
        def frequency_samples(self, frequencies, delta_f, epoch, channels,
                              **options):
            assert channels == ("A",)
            assert delta_f == summary.delta_fs
            assert epoch == summary.epochs
            assert options == {"spectral_padding": 1e-4}
            indices = np.rint(frequencies["A"] / delta_f["A"]).astype(int)
            return {"A": candidate[indices]}

    expected = summary.evaluate({"A": _series(candidate, delta_f)})
    actual = summary.evaluate_response(
        SparseResponse(), spectral_padding=1e-4)
    assert np.allclose(actual, expected, rtol=0, atol=0)


def test_adaptive_bins_resolve_curved_ratio_around_unsafe_gap():
    size, delta_f = 2049, 5e-4
    frequency = np.arange(size) * delta_f
    reference = (1 + frequency) * np.exp(3j * frequency)
    candidate = reference * np.exp(35j * frequency ** 2)
    psd = 1 + 0.2 * frequency
    reference[700:705] = 0
    candidate[700:705] = 0.01j

    edges, diagnostics = adaptive_tdi_frequency_bins(
        _series(reference, delta_f),
        [_series(candidate, delta_f)],
        _series(psd, delta_f),
        low_frequency_cutoff=0.01,
        high_frequency_cutoff=0.9,
        relative_tolerance=2e-3,
        reference_floor=1e-8,
        return_diagnostics=True,
    )

    assert len(edges) > 10
    assert diagnostics["largest_interval_error"] <= 2e-3
    assert diagnostics["deepest_refinement"] > 0
    assert 699 in edges
    assert 705 in edges
