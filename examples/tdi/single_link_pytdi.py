"""Generate velocity-corrected laser links and combine them with PyTDI.

Run this with PyTDI installed. The source object is deliberately small: any
waveform implementation exposing ``polarizations(t)`` can replace it.
"""

import numpy as np

from pycbc.coordinates.space_orbit import LisaEqualArmOrbit
from pycbc.tdi.backends.pytdi_backend import combine_links
from pycbc.tdi.response import (
    link_geometry,
    link_response,
    sample_constellation,
)


class MonochromaticSource:
    """A simple source evaluated directly at every requested retarded time."""

    def __init__(self, frequency):
        self.frequency = frequency

    def polarizations(self, times):
        phase = 2 * np.pi * self.frequency * np.asarray(times)
        return np.cos(phase), 0.2 * np.sin(phase)


times = np.arange(4096, dtype=float) * 0.5
sample = sample_constellation(times, LisaEqualArmOrbit(), ltt_order=1)
geometry = link_geometry(
    sample,
    lamb=0.8,
    beta=-0.3,
    velocity_order=1,
    retard_emitter=True,
)
links = link_response(MonochromaticSource(0.01), sample, geometry)
channels = combine_links(links, sample, channels="AET", generation=2)

for name, strain in channels.items():
    print(name, len(strain), strain.delta_t, np.max(np.abs(strain.numpy())))
