"""How far the emitter-retardation expansion has to be carried, and why.

`sample_constellation` solves the light cone exactly, so nothing here changes
what the code does.  What these tests protect is the reasoning that let
`link_geometry` ship with no ``acceleration_order`` knob: the acceleration
term of ``r_j(t - L)`` is negligible, and it is negligible for a reason that
survives a longer-armed or faster-orbiting constellation being added later.

Expanding the emission event,

    r_j(t - L) = r_j - v L + a L^2 / 2 - adot L^3 / 6,

successive orders do NOT share one factor.  The consecutive ratios are

    acc / vel  = |k.a| L / (2 |k.v|),      jerk / acc = |k.adot| L / (3 |k.a|)

with different denominators, so a single "cost per order" would hide the case
where they disagree -- and TianQin is that case: its jerk/acc is eight times
its acc/vel, because its velocity is dominated by Earth's heliocentric motion
while its acceleration is dominated by the geocentric orbit.
"""

import numpy as np
import pytest

from pycbc.coordinates.space_orbit import (LisaEqualArmOrbit,
                                           TaijiEqualArmOrbit,
                                           TianQinAnalyticOrbit)
from pycbc.tdi.response import C_SI

YEAR = 3.15581498e7
ORBITS = (("LISA", LisaEqualArmOrbit), ("Taiji", TaijiEqualArmOrbit),
          ("TianQin", TianQinAnalyticOrbit))


def _terms(orbit):
    """Largest sampling-time contribution of each Taylor order, in seconds.

    Maximised over time and over sky direction independently, which is what
    makes these ENVELOPES: |k.x| <= |x|, and the maxima of different orders
    fall at different points.
    """
    times = np.linspace(0.0, YEAR, 733)
    arm = orbit.armlength / C_SI
    speed = np.linalg.norm(
        np.asarray(orbit.compute_velocity(times, (1, 2, 3))), axis=-1)
    acceleration = np.linalg.norm(
        np.asarray(orbit.compute_acceleration(times, (1, 2, 3))), axis=-1)
    step = 64.0
    jerk = np.linalg.norm(
        (np.asarray(orbit.compute_acceleration(times + step, (1, 2, 3)))
         - np.asarray(orbit.compute_acceleration(times - step, (1, 2, 3))))
        / (2 * step), axis=-1)
    return dict(
        arm=arm,
        velocity=np.max(speed) * arm / C_SI,
        acceleration=np.max(acceleration) * arm ** 2 / (2 * C_SI),
        jerk=np.max(jerk) * arm ** 3 / (6 * C_SI),
        speed=np.max(speed), magnitude=np.max(acceleration),
        jerk_magnitude=np.max(jerk))


@pytest.mark.parametrize("name,factory", ORBITS)
def test_each_taylor_order_matches_its_own_envelope(name, factory):
    """The ratios are the two envelopes, not one shared expansion parameter."""
    term = _terms(factory())
    acceleration_over_velocity = term['acceleration'] / term['velocity']
    jerk_over_acceleration = term['jerk'] / term['acceleration']
    predicted_first = (term['magnitude'] * term['arm']
                       / (2 * term['speed']))
    predicted_second = (term['jerk_magnitude'] * term['arm']
                        / (3 * term['magnitude']))
    assert abs(acceleration_over_velocity / predicted_first - 1) < 0.05
    assert abs(jerk_over_acceleration / predicted_second - 1) < 0.05

    # the velocity term is the one that matters: sub-millisecond, and the
    # plan's 0.83 ms for LISA
    assert 1e-5 < term['velocity'] < 2e-3
    # the acceleration term is six orders below it and the jerk far below that
    assert acceleration_over_velocity < 2e-6
    assert jerk_over_acceleration < 1e-5

    # a phase error at the top of the band, which is what actually matters
    assert 2 * np.pi * 1.0 * term['acceleration'] < 1e-8


def test_omega_l_over_two_is_not_the_expansion_parameter():
    """It happens to work for LISA and Taiji, and fails for TianQin.

    Recording this because the wrong parameter is the tempting one: for a
    heliocentric constellation the guiding centre supplies both the speed and
    the acceleration, so |a| L / (2 |v|) collapses to omega L / 2.  TianQin's
    do not come from the same motion -- Earth's orbit sets its speed, its own
    geocentric orbit sets its acceleration -- and the two part company.
    """
    earth_rate = 2 * np.pi / YEAR
    ratios = {}
    for name, factory in ORBITS:
        orbit = factory()
        term = _terms(orbit)
        measured = term['acceleration'] / term['velocity']
        naive = earth_rate * term['arm'] / 2
        ratios[name] = measured / naive
    assert abs(ratios["LISA"] - 1) < 0.05
    assert abs(ratios["Taiji"] - 1) < 0.05
    assert ratios["TianQin"] > 3.0


def test_tianqins_series_is_not_uniformly_geometric():
    """jerk/acc far exceeds acc/vel for TianQin, and not for the others.

    One "cost per order" would have hidden this, which is the reason the two
    ratios are asserted separately above.
    """
    spread = {}
    for name, factory in ORBITS:
        term = _terms(factory())
        spread[name] = ((term['jerk'] / term['acceleration'])
                        / (term['acceleration'] / term['velocity']))
    assert spread["TianQin"] > 4 * max(spread["LISA"], spread["Taiji"])
