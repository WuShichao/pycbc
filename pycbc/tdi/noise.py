# Copyright (C) 2026  Shichao Wu, Alex Nitz
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""The noise PSD of a TDI channel from the same orbit and terms as the response.

`pycbc.psd.analytical_space` gives the equal-arm, static-delay PSD of A and
E. For TDI 2.0 that has a factor ``sin^2(4 pi f L / c)``, which vanishes at
``f = k c / (4 L)`` (0.02998 Hz for 2.5e9 m). The response of the same
channels through an orbit's light travel times does not vanish there: the
travel times differ by link and change over the year (about 1e-4 of ``L``
for the equal-arm LISA orbit, from the constellation's motion), so the
residual of the factor is small but not zero. Whitening that response with
the analytical PSD puts nearly all of a source's whitened power at the zero.

This module computes the noise PSD through the response's own delays. Each
secondary noise enters the single-link measurement ``y_ij`` (received at
``i`` from ``j``) as::

    y_ij = n_ij + delta_ij + D_ij delta_ji

with ``n_ij`` the optical-metrology noise of the link and ``delta_ij`` the
test-mass noise on spacecraft ``i`` facing ``j``, both in fractional
frequency. Through a channel ``sum_k c_k D_{chain_k} y_{link_k}`` each of the
twelve noises has the transfer function ``T(f, t) = sum c exp(-2 pi i f d(t))``
over its terms, with ``d`` the chain's delay from
`pycbc.tdi.onthefly.chain_delay`, frozen at ``t``. The PSD is the mean over
``times`` of ``sum |T|^2 S``. In the static equal-arm limit it is exactly
`analytical_psd_lisa_tdi_AE` (see the tests).

The cross-spectrum between channels and the non-stationarity are dropped: the
result is the time-averaged diagonal.
"""

import numpy

from pycbc.psd.analytical_space import psd_lisa_acc_noise, psd_lisa_oms_noise
from pycbc.tdi.onthefly import chain_delay
from pycbc.tdi.response import LINK_ORDER


def noise_transfer_delays(orbit, terms, times, links=LINK_ORDER):
    """Coefficients and delays of every secondary noise through ``terms``.

    Parameters
    ----------
    orbit
        The orbit, as for the response.
    terms : sequence of Term
        One channel's terms.
    times : array
        Times at which the delays are frozen [s].

    Returns
    -------
    dict
        ``('oms', link)`` and ``('acc', link)`` to ``(coefficients, delays)``,
        the coefficients of shape ``(K,)`` and the delays ``(K, len(times))``.
    """
    times = numpy.asarray(times, dtype=float)
    cache = {}
    out = {}
    for link in links:
        a, b = link
        oms_c, oms_d, acc_c, acc_d = [], [], [], []
        for term in terms:
            if tuple(term.link) == link:
                d = chain_delay(orbit, times, term.operators, links, cache=cache)
                oms_c.append(term.coefficient)
                oms_d.append(d)
                acc_c.append(term.coefficient)
                acc_d.append(d)
            elif tuple(term.link) == (b, a):
                # y_ba carries delta_ab delayed along the link received at b
                acc_c.append(term.coefficient)
                acc_d.append(chain_delay(orbit, times,
                                         tuple(term.operators) + (f'D_{b}{a}',),
                                         links, cache=cache))
        if oms_c:
            out[('oms', link)] = (numpy.array(oms_c), numpy.array(oms_d))
        if acc_c:
            out[('acc', link)] = (numpy.array(acc_c), numpy.array(acc_d))
    return out


def psd_from_delays(sources, frequencies, acc_psd, oms_psd, xp=numpy,
                    chunk=2048):
    """Time-averaged ``sum |T|^2 S`` over the noises of ``sources``.

    ``sources`` is `noise_transfer_delays`' output; ``acc_psd`` and
    ``oms_psd`` are the test-mass and metrology PSDs at ``frequencies``.
    Computed in float64 on ``xp``: the residual at a zero is the difference of
    phases of about ``2 pi f L / c`` that agree to 1e-4.
    """
    f = numpy.asarray(frequencies, dtype=float)
    spectra = dict(acc=numpy.asarray(acc_psd, float),
                   oms=numpy.asarray(oms_psd, float))
    out = numpy.zeros(len(f))
    staged = [(kind, xp.asarray(c, dtype=xp.float64),
               xp.asarray(d, dtype=xp.float64))
              for (kind, _), (c, d) in sources.items()]
    for lo in range(0, len(f), chunk):
        fc = xp.asarray(f[lo:lo + chunk])
        total = {'acc': 0.0, 'oms': 0.0}
        for kind, c, d in staged:
            phase = xp.exp(-2j * numpy.pi * fc[:, None, None] * d[None])
            transfer = xp.einsum('k,fkt->ft', c.astype(xp.complex128), phase)
            total[kind] = total[kind] + (xp.abs(transfer) ** 2).mean(axis=1)
        part = (total['acc'] * xp.asarray(spectra['acc'][lo:lo + chunk])
                + total['oms'] * xp.asarray(spectra['oms'][lo:lo + chunk]))
        out[lo:lo + chunk] = part.get() if hasattr(part, 'get') else part
    return out


def orbit_tdi_psd(orbit, channel_terms, frequencies, times,
                  acc_noise_level=3e-15, oms_noise_level=15e-12, xp=numpy,
                  chunk=2048):
    """The PSD of each channel through the orbit's delays, averaged over times.

    Parameters
    ----------
    orbit
        The orbit the response uses.
    channel_terms : dict
        Channel name to its terms (`pycbc.tdi.inference.channel_terms`).
    frequencies : array
        Frequencies [Hz], all positive.
    times : array
        Times over which the PSD is averaged [s], e.g. spread over the data.
    acc_noise_level, oms_noise_level : float
        As for `pycbc.psd.analytical_space.analytical_psd_lisa_tdi_AE`.

    Returns
    -------
    dict
        Channel name to its PSD at ``frequencies`` (numpy, float64).
    """
    f = numpy.asarray(frequencies, dtype=float)
    acc = psd_lisa_acc_noise(f, numpy.float64(acc_noise_level))
    oms = psd_lisa_oms_noise(f, numpy.float64(oms_noise_level))
    return {name: psd_from_delays(noise_transfer_delays(orbit, terms, times),
                                  f, acc, oms, xp=xp, chunk=chunk)
            for name, terms in channel_terms.items()}
