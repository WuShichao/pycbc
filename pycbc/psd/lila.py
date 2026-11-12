# Copyright (C) 2026  Chayan Chatterjee
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
r"""
Noise handling for the LILA Observatory.

The project supplies tabulated **one-sided** power spectral densities in
:math:`\mathrm{strain}^2/\mathrm{Hz}`, separately for the Geodesic
Displacement Channel (suspended optics, usable above
:math:`\sim0.45\,\mathrm{Hz}`) and the Lunar Deformation Channel
(unsuspended, carrying the band below). A third "combined" curve is also
distributed: it is the pointwise *minimum* of the two, i.e. an envelope
showing whichever channel is better at each frequency, and therefore
describes no actual data stream. It is useful for figures and must not be
used in a likelihood. `read_lila_psd` refuses it by default for that
reason.

These curves are *effective strain* sensitivities: the lunar elastic
transfer function is already divided out, which is directly visible as
resonance/antiresonance pairs through the normal-mode comb. The detector
response must therefore be geometry-only -- see
`pycbc.detector.space._LILA_detector.ldc_antenna_pattern`. The sky and
polarization beam pattern, by contrast, is *not* folded in: the curves
are quoted for optimal orientation, so the antenna patterns are applied
on top as usual.
"""

import numpy as np

from pycbc.psd.read import from_txt

__all__ = ["read_lila_psd", "vertex_psd_to_aet"]


def read_lila_psd(filename, length, delta_f, low_freq_cutoff,
                  allow_combined=False):
    """Read a tabulated LILA one-sided PSD.

    Parameters
    ----------
    filename : str
        Path to a two-column file of frequency and PSD in
        strain^2/Hz. The curves are one-sided, matching PyCBC's
        convention, so no factor-of-two correction is applied.
    length, delta_f, low_freq_cutoff :
        As for `pycbc.psd.from_txt`.
    allow_combined : bool, optional
        The combined curve is an envelope, not the noise of any real
        channel, so loading it is refused unless this is set. Default
        False.

    Returns
    -------
    pycbc.types.FrequencySeries
    """
    if not allow_combined and 'combined' in str(filename).lower():
        raise ValueError(
            "this looks like the 'combined' LILA curve, which is the "
            "pointwise minimum of the GDC and LDC spectra -- an envelope "
            "for figures, not the noise of any real data stream. Use the "
            "per-channel GDC or LDC curve in a likelihood, or pass "
            "allow_combined=True if you really intend the envelope.")
    return from_txt(filename, length, delta_f, low_freq_cutoff,
                    is_asd_file=False)


def vertex_psd_to_aet(psd_vertex, correlation=0.0):
    r"""Transform a per-vertex PSD into the orthogonal A, E, T basis.

    Under the standard symmetry assumption -- equal noise in each vertex
    and equal correlation between each pair, so that the covariance is
    :math:`\Sigma_{ij} = S\delta_{ij} + C(1-\delta_{ij})` -- the A, E, T
    recombination of `_LILA_detector` diagonalises :math:`\Sigma`, with

    .. math::
        S_A = S_E = S - C, \qquad S_T = S + 2C .

    The three channels are then independent and the scalar likelihood
    applies channel by channel, which is the practical reason to work in
    A, E, T rather than in the vertex channels directly. T is the null
    stream: it carries no gravitational-wave signal and constrains the
    noise.

    Whether the symmetry assumption holds for LILA is a facility-design
    question that cannot be settled from the data alone; vertices sharing
    corners make some correlation likely. When measured cross-spectra
    become available this function should be replaced by a full
    :math:`\Sigma^{-1}` treatment rather than extended.

    Parameters
    ----------
    psd_vertex : pycbc.types.FrequencySeries or numpy.array
        The one-sided PSD of a single vertex channel.
    correlation : float or array, optional
        The pairwise noise correlation C, in the same units as
        `psd_vertex`, or as a fraction of it if a float in [-0.5, 1) is
        given. Default 0, i.e. independent channels.

    Returns
    -------
    (psd_a, psd_e, psd_t) : tuple
        Same type as the input.
    """
    s = psd_vertex
    arr = s.numpy() if hasattr(s, 'numpy') else np.asarray(s)

    c = np.asarray(correlation, dtype=float)
    if c.ndim == 0 and -0.5 <= float(c) < 1.0:
        cov = float(c) * arr          # interpret a scalar as a fraction
    else:
        cov = c

    # Positive-definiteness of Sigma requires -S/2 < C < S.
    if np.any(cov >= arr) or np.any(cov <= -0.5 * arr):
        raise ValueError(
            "correlation outside the range that keeps the noise "
            "covariance positive definite: need -S/2 < C < S.")

    s_ae = arr - cov
    s_t = arr + 2.0 * cov

    if hasattr(s, 'numpy'):
        out = []
        for a in (s_ae, s_ae, s_t):
            new = s.copy()
            new.data[:] = a
            out.append(new)
        return tuple(out)
    return s_ae, s_ae.copy(), s_t
