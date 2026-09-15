"""The catalogue and mismatch arithmetic behind the Yorsh SOBHB validation.

These were helpers inside an example script that a test loaded through an
import-spec hack. They are small, they carry two mistakes that are easy to
make and silent when made, and they belong next to their assertions.

The end-to-end validation against the official Yorsh channels is not here: it
needs the LDC archive, so it could only ever skip. It lives in the research
repository, which has the data and the memory budget for it.
"""

import numpy as np

from pycbc.conversions import MTSUN_SI


def catalogue_parameters(row):
    """Scalar catalogue values, with masses left in the frame they are in.

    The Yorsh catalogue lists source-frame masses alongside a redshift, and
    the response wants detector-frame ones. Applying the redshift here and
    again downstream is the mistake this guards: it is invisible in any
    self-consistent test, because both sides move together.
    """
    result = {}
    for name in row.dtype.names:
        value = row[name].item()
        if isinstance(value, bytes):
            value = value.decode()
        result[name] = value
    return result


def newtonian_end_frequency(mass1, mass2, f_start, duration):
    """Conservative upper frequency, used only to size the source."""
    total = (mass1 + mass2) * MTSUN_SI
    eta = mass1 * mass2 / (mass1 + mass2) ** 2
    chirp = total * eta ** 0.6
    tau = 5 / 256 * chirp ** (-5 / 3) * (np.pi * f_start) ** (-8 / 3)
    remaining = max(tau - duration, 1e-3 * tau)
    return (256 / 5 * chirp ** (5 / 3) * remaining) ** (-3 / 8) / np.pi


def empty_accumulator():
    return {"official_norm": 0.0, "candidate_norm": 0.0, "cross": 0j,
            "bins": 0}


def add_inner_products(accumulator, official, candidate, psd, mask, delta_f):
    """Accumulate noise-weighted inner products one chunk at a time.

    A two-year segment at 5 s does not fit beside everything else, so the
    mismatch is accumulated rather than formed from whole spectra.
    """
    selected_official = official[mask]
    selected_candidate = candidate[mask]
    selected_psd = psd[mask]
    scale = 4 * delta_f
    accumulator["official_norm"] += float(
        scale * np.sum(np.abs(selected_official) ** 2 / selected_psd))
    accumulator["candidate_norm"] += float(
        scale * np.sum(np.abs(selected_candidate) ** 2 / selected_psd))
    accumulator["cross"] += complex(scale * np.sum(
        np.conj(selected_official) * selected_candidate / selected_psd))
    accumulator["bins"] += int(np.count_nonzero(mask))


def finish_metrics(accumulator):
    """Accumulated inner products to signal-to-noise ratios and mismatch."""
    official_norm = accumulator["official_norm"]
    candidate_norm = accumulator["candidate_norm"]
    if official_norm <= 0 or candidate_norm <= 0:
        return {"official_snr": 0.0, "candidate_snr": 0.0,
                "mismatch": np.nan, "bins": accumulator["bins"]}
    match = abs(accumulator["cross"]) / np.sqrt(
        official_norm * candidate_norm)
    # Roundoff can put an exactly identical pair a few ulps above one.
    match = min(float(match), 1.0)
    return {
        "official_snr": float(np.sqrt(official_norm)),
        "candidate_snr": float(np.sqrt(candidate_norm)),
        "mismatch": 1.0 - match,
        "bins": accumulator["bins"],
    }


def test_catalogue_masses_are_not_redshifted_twice():
    row = np.asarray([(
        b"sobhb-test", 35.0, 21.0, 0.12,
    )], dtype=[("Name", "S16"), ("Mass1", "f8"),
              ("Mass2", "f8"), ("Redshift", "f8")])[0]
    parameters = catalogue_parameters(row)
    assert parameters["Name"] == "sobhb-test"
    assert parameters["Mass1"] == 35.0
    assert parameters["Mass2"] == 21.0
    assert parameters["Redshift"] == 0.12


def test_inner_product_accumulation_reports_exact_match():
    official = np.asarray([1 + 2j, 3 - 1j, -2 + 0.5j])
    psd = np.asarray([2.0, 4.0, 3.0])
    mask = np.asarray([True, False, True])
    accumulator = empty_accumulator()
    add_inner_products(accumulator, official, official.copy(), psd, mask, 0.01)
    result = finish_metrics(accumulator)
    assert result["mismatch"] == 0.0
    assert result["official_snr"] == result["candidate_snr"]
    assert result["bins"] == 2


def test_inner_products_accumulate_across_chunks():
    """Chunking must not change the answer, which is the point of it."""
    rng = np.random.default_rng(20260915)
    official = rng.normal(size=64) + 1j * rng.normal(size=64)
    candidate = official + 0.01 * (rng.normal(size=64)
                                   + 1j * rng.normal(size=64))
    psd = rng.uniform(1.0, 4.0, size=64)
    mask = np.ones(64, dtype=bool)

    whole = empty_accumulator()
    add_inner_products(whole, official, candidate, psd, mask, 0.01)

    chunked = empty_accumulator()
    for first in range(0, 64, 7):
        stop = min(64, first + 7)
        add_inner_products(chunked, official[first:stop],
                           candidate[first:stop], psd[first:stop],
                           mask[first:stop], 0.01)

    assert chunked["bins"] == whole["bins"] == 64
    assert np.isclose(finish_metrics(chunked)["mismatch"],
                      finish_metrics(whole)["mismatch"],
                      rtol=1e-12, atol=0)


def test_empty_selection_reports_no_match_rather_than_dividing_by_zero():
    accumulator = empty_accumulator()
    add_inner_products(accumulator, np.zeros(4, dtype=complex),
                       np.zeros(4, dtype=complex), np.ones(4),
                       np.zeros(4, dtype=bool), 0.01)
    result = finish_metrics(accumulator)
    assert result["bins"] == 0
    assert np.isnan(result["mismatch"])


def test_end_frequency_uses_detector_frame_mass_scale():
    duration = 2 * 365.25 * 86400.0
    raw = newtonian_end_frequency(35.0, 21.0, 0.01, duration)
    wrongly_redshifted = newtonian_end_frequency(
        35.0 * 1.2, 21.0 * 1.2, 0.01, duration)
    assert raw != wrongly_redshifted
    assert raw > 0.01


def test_end_frequency_is_finite_for_a_source_that_outlives_the_mission():
    """A binary far from merger must not return an infinite end frequency."""
    duration = 2 * 365.25 * 86400.0
    value = newtonian_end_frequency(5.0, 5.0, 1e-3, duration)
    assert np.isfinite(value)
    assert value > 1e-3
