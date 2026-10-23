# Copyright (C) 2026 Shichao Wu
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
"""Original-detector-band likelihood with lazy harmonic responses."""
import numpy

from pycbc.filter.detector_harmonic import DetectorHarmonicLikelihood
from pycbc.tdi.streamed_harmonics import (streamed_correlation_spectrum,
                                          streamed_harmonic_spectrum)


class StreamedDetectorHarmonicLikelihood(DetectorHarmonicLikelihood):
    """CPU implementation retaining only one response harmonic at a time.

    The source, projector, data, physical windows and noise conventions are
    unchanged. Correlation polynomials, direct statistics and output memory
    limits are inherited from the original-band likelihood. No existing
    backend or production inference default is changed.
    """

    def correlation_spectrum(self, source, responses, orders, periods=None):
        return streamed_correlation_spectrum(
            source, responses, orders, self.adjoint_data, self._periods(periods))

    def template(self, source, responses, orders, delay, periods=None):
        if not numpy.isfinite(delay):
            raise ValueError('delay must be finite')
        shifted = numpy.asarray(source) * numpy.exp(
            -2j * numpy.pi * self.grid.frequencies * delay)
        spectrum = streamed_harmonic_spectrum(
            shifted, responses, orders, self._periods(periods))
        return self.projector.project(spectrum) * self.weight
