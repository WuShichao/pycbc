"""Evaluating TDI in chunks must give the same answer as evaluating it whole.

A mission-length channel does not fit in memory, so it is built a piece at a
time with an overlap wide enough for the combination's delay reach. That is
only sound if the seam is invisible, and a seam is exactly the kind of defect
that a self-consistent signal-to-noise number will not show: both sides of a
bad join are equally wrong.

This was a check inside an example script that nothing ran automatically.
"""

import numpy as np

from pycbc.coordinates.space_orbit import LisaEqualArmOrbit
from pycbc.tdi.backends.pytdi_backend import combine_links
from pycbc.tdi.response import (link_geometry, link_response,
                                sample_constellation)

DT = 5.0
LAMB, BETA = 0.9, -0.25


class _Monochromatic:
    """Evaluated directly at every requested retarded time."""

    def __init__(self, frequency):
        self.frequency = frequency

    def polarizations(self, times):
        phase = 2 * np.pi * self.frequency * np.asarray(times)
        return np.cos(phase), 0.2 * np.sin(phase)


def _aet(source, orbit, first, last, generation=1):
    times = np.arange(first, last, dtype=float) * DT
    sample = sample_constellation(times, orbit)
    links = link_response(
        source, sample, link_geometry(sample, LAMB, BETA, velocity_order=1))
    return combine_links(links, sample, generation=generation,
                         channels="AET", interpolation_order=31,
                         delay_order=5)


def _chunk(source, orbit, start, stop, overlap):
    """One chunk, with the overlap trimmed off both ends."""
    channels = _aet(source, orbit, start - overlap, stop + overlap)
    width = stop - start
    return {name: np.asarray(series)[overlap:overlap + width]
            for name, series in channels.items()}


def test_chunked_tdi_matches_the_whole_evaluation():
    orbit = LisaEqualArmOrbit()
    source = _Monochromatic(0.01)
    size, overlap = 4096, 128

    whole = _aet(source, orbit, -overlap, size + overlap)
    whole = {name: np.asarray(value)[overlap:overlap + size]
             for name, value in whole.items()}

    # A stride that does not divide the size, so the last chunk is short and
    # the seams do not land on round numbers.
    split = {name: np.empty(size) for name in "AET"}
    for start in range(0, size, 777):
        stop = min(size, start + 777)
        piece = _chunk(source, orbit, start, stop, overlap)
        for name in "AET":
            split[name][start:stop] = piece[name]

    for name in "AET":
        scale = max(np.max(np.abs(whole[name])), np.finfo(float).tiny)
        error = np.max(np.abs(split[name] - whole[name])) / scale
        assert error < 2e-10, f"{name} seam error {error:.3e}"


def test_too_narrow_an_overlap_is_visible_at_the_seams():
    """The tolerance above has teeth only if a bad overlap fails it.

    X2 reaches seven arms back, about 58 s, so an overlap of one sample
    cannot carry the combination's delay chain.
    """
    orbit = LisaEqualArmOrbit()
    source = _Monochromatic(0.01)
    size, overlap = 1024, 128

    whole = _aet(source, orbit, -overlap, size + overlap)
    whole = {name: np.asarray(value)[overlap:overlap + size]
             for name, value in whole.items()}

    split = {name: np.empty(size) for name in "AET"}
    for start in range(0, size, 333):
        stop = min(size, start + 333)
        piece = _chunk(source, orbit, start, stop, 1)
        for name in "AET":
            split[name][start:stop] = piece[name]

    worst = max(
        np.max(np.abs(split[name] - whole[name]))
        / max(np.max(np.abs(whole[name])), np.finfo(float).tiny)
        for name in "AET")
    assert worst > 2e-10
