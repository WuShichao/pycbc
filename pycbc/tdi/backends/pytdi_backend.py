# Copyright (C) 2026  Shichao Wu, Alex Nitz
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""In-memory adapter from PyCBC single-link responses to PyTDI."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from pycbc.tdi.combination import Term, orthogonal_channels
from pycbc.types import TimeSeries


def _require_pytdi():
    try:
        from pytdi import core, michelson
    except ImportError as exc:
        raise ImportError(
            "PyTDI is required for this backend; install pytdi or select "
            "another TDI combiner"
        ) from exc
    return core, michelson


def _uniform_delta_t(times):
    times = np.asarray(times, dtype=float)
    steps = np.diff(times)
    delta_t = steps[0]
    tolerance = 32 * np.finfo(float).eps * max(
        1.0, abs(delta_t), np.max(np.abs(times))
    )
    if not np.allclose(steps, delta_t, rtol=0.0, atol=tolerance):
        raise ValueError("PyTDI dense evaluation requires a uniform time grid")
    return float(delta_t)


def _eta_mapping(values, links):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != len(links):
        raise ValueError(
            f"link responses must have shape (N, {len(links)}), got "
            f"{values.shape}"
        )
    return {
        f"eta_{receiver}{emitter}": values[:, index]
        for index, (receiver, emitter) in enumerate(links)
    }


def _validate_six_links(sample):
    expected = {(1, 2), (2, 3), (3, 1), (1, 3), (3, 2), (2, 1)}
    if len(sample.links) != 6 or set(sample.links) != expected:
        raise ValueError("PyTDI combinations require all six directed links")


@dataclass(frozen=True)
class PyTDICombinationAdapter:
    """Expose a PyTDI ETA combination through the plan's algebra protocol."""

    name: str
    combination: object
    delta_t: float
    delay_order: int = 5

    def terms(self):
        terms = []
        for measurement, polynomial in self.combination.components.items():
            label, indices = measurement.rsplit("_", 1)
            if label != "eta":
                raise ValueError(
                    f"{self.name} is not an ETA-only combination: {measurement}"
                )
            link = (int(indices[0]), int(indices[1]))
            terms.extend(
                Term(link, float(coefficient), tuple(operators))
                for coefficient, operators in polynomial
            )
        return tuple(terms)

    def net_shifts(self, sample):
        shifts, _ = self.combination.build_shifts(
            sample.delays, 1 / self.delta_t, order=self.delay_order
        )
        used = {
            tuple(operators)
            for polynomial in self.combination.components.values()
            for _, operators in polynomial
        }
        # PyTDI represents a delay by a negative shift passed to its
        # interpolator. The framework convention is positive = delay.
        return {operators: -shifts[operators] for operators in used}

    def data_delay_extent(self, sample):
        shifts = self.net_shifts(sample)
        longest_chain = max(map(len, shifts))
        delay_samples = int(np.ceil(np.max(sample.ltt) / self.delta_t))
        margin = delay_samples * (longest_chain + 2) + self.delay_order
        if 2 * margin >= len(sample.t):
            raise ValueError(
                "time grid is too short to measure the interior delay extent"
            )

        def interior(shift):
            shift = np.asarray(shift)
            if shift.ndim == 0:
                return shift
            return shift[margin:-margin]

        minimum = min(np.min(interior(shift)) for shift in shifts.values())
        maximum = max(np.max(interior(shift)) for shift in shifts.values())
        return float(maximum - minimum)

    def waveform_support(self, sample):
        return self.data_delay_extent(sample) + float(np.max(sample.ltt))


def get_pytdi_michelson(generation=2):
    """Return ETA-only PyTDI Michelson combinations ``(X, Y, Z)``."""
    _, michelson = _require_pytdi()
    if generation == 1:
        return michelson.X1_ETA, michelson.Y1_ETA, michelson.Z1_ETA
    if generation == 2:
        return michelson.X2_ETA, michelson.Y2_ETA, michelson.Z2_ETA
    raise ValueError("generation must be 1 or 2")


@lru_cache(maxsize=None)
def get_pytdi_combination(name):
    """Resolve one verified ETA-only combination from the plan registry.

    ``C12_3`` is absent until its path transcription passes a laser-noise
    cancellation test.
    """
    core, michelson = _require_pytdi()
    standard = {
        "X1": michelson.X1_ETA,
        "Y1": michelson.Y1_ETA,
        "Z1": michelson.Z1_ETA,
        "X2": michelson.X2_ETA,
        "Y2": michelson.Y2_ETA,
        "Z2": michelson.Z2_ETA,
    }
    definitions = {
        "UU": "323121323 -312323213",
        "PD4L-1": "-1232 212 -2321 1323 -313 3231",
        "PD4L-2": "-2313 323 -3132 2131 -121 1312",
        "PD4L-3": "-3121 131 -1213 3212 -232 2123",
    }
    name = name.upper()
    if name in standard:
        return standard[name]
    if name in definitions:
        return core.LISATDICombination.from_string(definitions[name])
    if name == "VV":
        return get_pytdi_combination("UU").rotated()
    if name == "WW":
        return get_pytdi_combination("UU").rotated(2)
    available = tuple(standard) + tuple(definitions) + ("VV", "WW")
    raise ValueError(
        f"unknown or unverified TDI combination {name!r}; available: "
        f"{', '.join(available)}"
    )


def combine_link_combination(link_responses, sample, name, *,
                             interpolation_order=31, delay_order=5,
                             unit="frequency"):
    """Evaluate one registered ETA combination as a PyCBC ``TimeSeries``."""
    values = np.asarray(link_responses)
    if values.ndim != 2 or values.shape[0] != len(sample.t):
        raise ValueError(
            "link responses must be a two-dimensional array with the same "
            "length as sample"
        )
    if unit not in ("frequency", "phase"):
        raise ValueError("unit must be 'frequency' or 'phase'")
    _validate_six_links(sample)
    delta_t = _uniform_delta_t(sample.t)
    etas = _eta_mapping(values, sample.links)
    combination = get_pytdi_combination(name)
    built = combination.build(sample.delays, 1 / delta_t, order=delay_order)
    result = built(etas, order=interpolation_order, unit=unit)
    return TimeSeries(result, delta_t=delta_t, epoch=sample.t[0])


def combine_links(link_responses, sample, *, channels="AET", generation=2,
                  interpolation_order=31, delay_order=5, unit="frequency"):
    """Combine six link responses with PyTDI and return PyCBC TimeSeries.

    The link array is interpreted in ``sample.links`` order and mapped directly
    to PyTDI's intermediary ``eta_ij`` variables. This is valid for a pure GW
    response: the local test-mass and reference channels are zero, so the full
    measurement construction reduces exactly to ``eta_ij = y_ij``.

    Parameters
    ----------
    link_responses : array-like
        Fractional-frequency (default) or phase link data with shape ``(N, 6)``.
    sample : pycbc.tdi.response.ConstellationSample
        Orbit delays on the same uniform reception-time grid.
    channels : {"XYZ", "AET"}, optional
        Channel basis to return.
    generation : {1, 2}, optional
        First- or second-generation Michelson TDI.
    interpolation_order, delay_order : int, optional
        Odd Lagrange interpolation orders passed to PyTDI.
    unit : {"frequency", "phase"}, optional
        Unit convention passed to PyTDI.
    """
    if channels not in ("XYZ", "AET"):
        raise ValueError("channels must be 'XYZ' or 'AET'")
    if unit not in ("frequency", "phase"):
        raise ValueError("unit must be 'frequency' or 'phase'")
    values = np.asarray(link_responses)
    if values.ndim != 2:
        raise ValueError("link responses must be a two-dimensional array")
    if values.shape[0] != len(sample.t):
        raise ValueError("link responses and sample must have the same length")
    _validate_six_links(sample)
    delta_t = _uniform_delta_t(sample.t)
    etas = _eta_mapping(values, sample.links)
    combinations = get_pytdi_michelson(generation)
    xyz = []
    for combination in combinations:
        built = combination.build(
            sample.delays, 1 / delta_t, order=delay_order
        )
        xyz.append(built(etas, order=interpolation_order, unit=unit))
    labels = tuple("XYZ")
    arrays = tuple(xyz)
    if channels == "AET":
        labels = tuple("AET")
        arrays = orthogonal_channels(*xyz)
    return {
        label: TimeSeries(array, delta_t=delta_t, epoch=sample.t[0])
        for label, array in zip(labels, arrays, strict=True)
    }
