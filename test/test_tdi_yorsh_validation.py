"""Unit checks for the memory-bounded Yorsh validation driver."""

import importlib.util
from pathlib import Path

import numpy as np

_PATH = (Path(__file__).parents[1] / "examples" / "tdi"
         / "validate_yorsh_sobhb.py")
_SPEC = importlib.util.spec_from_file_location("yorsh_validation", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_yorsh_catalogue_masses_are_not_redshifted_twice():
    row = np.asarray([(
        b"sobhb-test", 35.0, 21.0, 0.12,
    )], dtype=[("Name", "S16"), ("Mass1", "f8"),
              ("Mass2", "f8"), ("Redshift", "f8")])[0]
    parameters = _MODULE.catalogue_parameters(row)
    assert parameters["Name"] == "sobhb-test"
    assert parameters["Mass1"] == 35.0
    assert parameters["Mass2"] == 21.0
    assert parameters["Redshift"] == 0.12


def test_yorsh_inner_product_accumulation_reports_exact_match():
    official = np.asarray([1 + 2j, 3 - 1j, -2 + 0.5j])
    psd = np.asarray([2.0, 4.0, 3.0])
    mask = np.asarray([True, False, True])
    accumulator = _MODULE._empty_accumulator()
    _MODULE._add_inner_products(
        accumulator, official, official.copy(), psd, mask, 0.01)
    result = _MODULE.finish_metrics(accumulator)
    assert result["mismatch"] == 0.0
    assert result["official_snr"] == result["candidate_snr"]
    assert result["bins"] == 2


def test_yorsh_end_frequency_uses_detector_frame_mass_scale():
    duration = 2 * 365.25 * 86400.0
    raw = _MODULE.newtonian_end_frequency(35.0, 21.0, 0.01, duration)
    wrongly_redshifted = _MODULE.newtonian_end_frequency(
        35.0 * 1.2, 21.0 * 1.2, 0.01, duration)
    assert raw != wrongly_redshifted
    assert raw > 0.01
