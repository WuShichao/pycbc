"""Do the response approximations get amplified where a channel nearly nulls?

A TDI combination has frequencies where its transfer function collapses, and
the worry is that an approximation dropped from the single-link response then
gets divided by a vanishing signal. It does not: the error passes through the
same combination that makes the null, so the two collapse together.

The scope matches how the number was first obtained. The single-link geometry
is one frozen epoch of the analytic, nearly equal-arm LISA orbit; only the six
TDI delays carry an unequal-arm snapshot (dL/L = 4.6e-3). The 'null' is the
minimum of |R| averaged over sky directions, not a per-direction null and not
one found by root-finding. So this shows the off-null ordering surviving at a
null; it does not show single-link errors immune to null amplification, which
an earlier version of the plan claimed.
"""

import numpy as np
import pytest

from pycbc.coordinates.space_orbit import LisaEqualArmOrbit
from pycbc.tdi.backends.pytdi_backend import (PyTDICombinationAdapter,
                                              get_pytdi_combination)
from pycbc.tdi.response import (C_SI, antenna_pattern, doppler_factors,
                                polarization_basis, sample_constellation)

EPOCH = 1.0e7
ARM_SPREAD = 4.6e-3            # Wang's unequal-arm snapshot, dL/L
SKY_COUNT = 64


def _frozen(orbit):
    """One epoch of the constellation, plus per-link unequal-arm delays."""
    times = np.array([EPOCH - 5.0, EPOCH, EPOCH + 5.0])
    sample = sample_constellation(times, orbit)
    index = 1
    delays = {}
    generator = np.random.default_rng(11)
    spread = generator.uniform(-ARM_SPREAD, ARM_SPREAD, len(sample.links))
    for position, link in enumerate(sample.links):
        delays[link] = sample.ltt[index, position] * (1 + spread[position])
    return sample, index, delays


def _chain_delay(chain, delays):
    """Net delay of an operator chain under the frozen snapshot."""
    total = 0.0
    for operator in chain:
        kind, indices = operator.split('_')
        arm = delays[(int(indices[0]), int(indices[1]))]
        total += arm if kind == 'D' else -arm
    return total


def _transfer(terms, sample, index, delays, frequency, sky, emitter='exact'):
    """|R_c(f)| for a unit plus-polarised wave, per sky direction.

    Built here because pycbc.tdi is a time-domain path while a null is a
    statement about the frequency-domain transfer. Each term contributes its
    coefficient times the chain's phase times the link's own two sampling
    phases.
    """
    out = np.zeros(len(sky), dtype=complex)
    link_index = {link: position for position, link in enumerate(sample.links)}
    for count, (lamb, beta) in enumerate(sky):
        u_hat, v_hat, k_hat = polarization_basis(lamb, beta)
        n_hat = sample.n_hat[index]
        xi_plus, _ = antenna_pattern(n_hat, u_hat, v_hat)
        prefactor = xi_plus / (2 * (1 - n_hat @ k_hat))
        receivers = [link[0] - 1 for link in sample.links]
        emitters = [link[1] - 1 for link in sample.links]
        travel = sample.ltt[index]
        if emitter == 'exact':
            r_emit = sample.r_emit[index]
        elif emitter == 'velocity':
            r_emit = (sample.position[index][emitters]
                      - sample.velocity[index][emitters] * travel[:, None])
        elif emitter == 'simultaneous':
            r_emit = sample.position[index][emitters]
        else:
            raise ValueError(emitter)
        tau_emit = travel + r_emit @ k_hat / C_SI
        tau_recv = sample.position[index][receivers] @ k_hat / C_SI
        eps1, eps2 = doppler_factors(
            k_hat, n_hat, sample.velocity[index][emitters],
            sample.velocity[index][receivers])
        total = 0j
        for term in terms:
            position = link_index[tuple(term.link)]
            phase = np.exp(-2j * np.pi * frequency
                           * _chain_delay(term.operators, delays))
            link = prefactor[position] * (
                (1 + eps1[position]) * np.exp(-2j * np.pi * frequency
                                              * tau_emit[position])
                - (1 + eps2[position]) * np.exp(-2j * np.pi * frequency
                                                * tau_recv[position]))
            total += term.coefficient * phase * link
        out[count] = total
    return np.abs(out)


def _sky(count=SKY_COUNT):
    generator = np.random.default_rng(3)
    return list(zip(generator.uniform(0, 2 * np.pi, count),
                    np.arcsin(generator.uniform(-1, 1, count))))


@pytest.mark.parametrize("name,null_at,expected_depth", [("X2", 0.5, 1e3),
                                                         ("PD4L-1", 1.0, 20.0)])
def test_the_error_is_not_amplified_by_the_null(name, null_at,
                                                expected_depth):
    """Across the whole scan, not between two chosen points.

    On this configuration the X2 averaged transfer drops by 2.5e4, while
    dropping the acceleration term costs 7.1e-11 to 1.5e-10 across the scan
    and dropping the retardation costs 9.8e-05 to 2.7e-04. Both errors peak
    at the null, by a factor under two against the 2.5e4 the signal falls.

    The plan compared one off-null point with one near-null point and put the
    movement under 25%. Over the whole scan it reaches a factor of two, at the
    null; the off-null ordering, which is what the number was used for, still
    holds.
    """
    orbit = LisaEqualArmOrbit()
    sample, index, delays = _frozen(orbit)
    arm = np.median(sample.ltt[index])
    terms = PyTDICombinationAdapter(name, get_pytdi_combination(name),
                                    delta_t=1.0).terms()
    sky = _sky()
    scan = np.linspace(null_at - 0.12, null_at + 0.12, 61) / arm

    strength = np.array([np.mean(_transfer(terms, sample, index, delays, f,
                                           sky))
                         for f in scan])
    deepest = int(np.argmin(strength))
    # how deep the null is depends on the arm snapshot, so assert a floor and
    # not the figure one draw produces
    assert np.median(strength) / strength[deepest] > expected_depth

    for emitter, ceiling in (('velocity', 1e-8), ('simultaneous', 1e-2)):
        relative = []
        for frequency in scan:
            exact = _transfer(terms, sample, index, delays, frequency, sky)
            other = _transfer(terms, sample, index, delays, frequency, sky,
                              emitter=emitter)
            relative.append(np.mean(np.abs(other - exact)) / np.mean(exact))
        relative = np.array(relative)
        assert relative.max() < ceiling
        # the whole scan sits inside a factor of three, and so does the null
        # against the median, while the signal falls by orders
        assert relative.max() / relative.min() < 3.0
        assert relative[deepest] / np.median(relative) < 3.0

    def error(frequency, emitter):
        exact = _transfer(terms, sample, index, delays, frequency, sky)
        other = _transfer(terms, sample, index, delays, frequency, sky,
                          emitter=emitter)
        return np.mean(np.abs(other - exact)) / np.mean(exact)

    # and the ordering off the null is not overturned at it: the acceleration
    # term stays far below the retardation it is being compared against
    at_null = scan[deepest]
    assert error(at_null, 'velocity') < 1e-4 * error(at_null, 'simultaneous')
