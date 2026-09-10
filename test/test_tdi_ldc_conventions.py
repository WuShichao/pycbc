"""pycbc.tdi against the LISA Data Challenge, with no challenge data.

The LDC orbit and single-link response are closed form, so they can be
transcribed and compared term by term.  That makes an EXTERNAL check of the
two response choices the plan quantified only from this project's own
scripts -- the light-cone arm direction and the retarded emitter -- and it
costs nothing to run: no 3 GB download, no LDC install, no network.

Everything below is taken from the LDC v1.2.0 source, not from the manual:
pdftotext mangles the parentheses of the velocity equations, and reading the
y component off the PDF gives a sign error worth 249 m/s.

    orbit and travel time   ldc/lisa/orbits/lib/{orbits.cc, common.cc}
    constants               ldc/common/constants/lisaconstants.hpp
    link response           ldc/lisa/projection/projectedstrain.py
    galactic binary         ldc/waveform/waveform/{gb_fdot.py, hphc.py}

The verification-binary parameters are the Sangria catalogue's own, five of
its 36 sources spanning the band.
"""

import numpy as np

from pycbc.coordinates.space_orbit import (EARTH_ORBIT_ANGULAR_FREQUENCY,
                                           LisaEqualArmOrbit)
from pycbc.tdi.response import (C_SI, ConstellationSample, link_geometry,
                                link_response, polarization_basis,
                                sample_constellation)

# ldc/common/constants/lisaconstants.hpp
LDC_AU = 149597870700.0
LDC_SIDEREAL_YEAR_DAY = 365.256363004
LDC_C = 299792458.0
LDC_OMEGA = 2 * np.pi / (LDC_SIDEREAL_YEAR_DAY * 24 * 60 * 60)

# obs/config of LDC2_sangria_training_v2.h5
ARM = 2.5e9
LDC_LINKS = ((1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2))   # get_pairs()

# sky/vgb/cat, five of the 36 verification binaries across the band
VERIFICATION_BINARIES = (
    dict(name="HD265435", amplitude=3.438962635e-22, frequency=3.363652494e-4,
         fdot=5.756230608e-20, inclination=1.117010721,
         polarization=1.073885953, phase0=4.32029186,
         lamb=1.768626707, beta=0.1770505691),
    dict(name="SDSSJ1630", amplitude=1.15121212e-22, frequency=8.368901163e-4,
         fdot=6.705569598e-19, inclination=1.047197551,
         polarization=0.6688097437, phase0=1.11874003,
         lamb=4.04517729, beta=1.100477312),
    dict(name="ZTFJ0722", amplitude=1.129764608e-22, frequency=1.405927307e-3,
         fdot=2.841354107e-18, inclination=1.564862207,
         polarization=4.584771724, phase0=1.982319225,
         lamb=2.02259538, beta=-0.7027579771),
    dict(name="AMCVn", amplitude=2.829116176e-22, frequency=1.944144722e-3,
         fdot=6.061897141e-18, inclination=0.7504915784,
         polarization=3.567121561, phase0=5.141844768,
         lamb=2.973723199, beta=0.6534962531),
    dict(name="HMCnc", amplitude=6.378340109e-23, frequency=6.220278731e-3,
         fdot=3.57e-16, inclination=0.6632251158,
         polarization=5.840950176, phase0=3.747078154,
         lamb=2.102051469, beta=-0.08209999861),
)


def ldc_position(t, spacecraft):
    """orbits.cc position_x/y/z, init_position = init_rotation = 0."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    eccentricity = ARM / (2 * np.sqrt(3) * LDC_AU)
    alpha = LDC_OMEGA * t
    out = np.empty((len(t), len(spacecraft), 3))
    for index, number in enumerate(spacecraft):
        rotation = (number - 1) * 2 * np.pi / 3.0
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        out[:, index, 0] = LDC_AU * (cos_a + eccentricity * (
            sin_a * cos_a * sin_r - (1 + sin_a ** 2) * cos_r))
        out[:, index, 1] = LDC_AU * (sin_a + eccentricity * (
            sin_a * cos_a * cos_r - (1 + cos_a ** 2) * sin_r))
        out[:, index, 2] = -LDC_AU * eccentricity * np.sqrt(3) * np.cos(
            alpha - rotation)
    return out


def ldc_velocity(t, spacecraft):
    """orbits.cc velocity_x/y/z.  Note the PLUS on the y eccentric term."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    eccentricity = ARM / (2 * np.sqrt(3) * LDC_AU)
    alpha = LDC_OMEGA * t
    out = np.empty((len(t), len(spacecraft), 3))
    for index, number in enumerate(spacecraft):
        rotation = (number - 1) * 2 * np.pi / 3.0
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        out[:, index, 0] = LDC_AU * LDC_OMEGA * (
            -sin_a + eccentricity * ((cos_a ** 2 - sin_a ** 2) * sin_r
                                     - 2 * sin_a * cos_a * cos_r))
        out[:, index, 1] = LDC_AU * LDC_OMEGA * (
            cos_a + eccentricity * ((cos_a ** 2 - sin_a ** 2) * cos_r
                                    + 2 * cos_a * sin_a * sin_r))
        out[:, index, 2] = (LDC_AU * eccentricity * np.sqrt(3)
                            * np.sin(alpha - rotation) * LDC_OMEGA)
    return out


def ldc_link(times, receiver, emitter):
    """common.cc travel_time at order 1.

    Both spacecraft at the RECEPTION time and the velocity correction taken
    from the RECEIVER: LDC does not solve the light cone, which is the whole
    point of comparing against it.
    """
    r_emit = ldc_position(times, (emitter,))[:, 0]
    r_recv = ldc_position(times, (receiver,))[:, 0]
    v_recv = ldc_velocity(times, (receiver,))[:, 0]
    separation = r_recv - r_emit
    distance = np.linalg.norm(separation, axis=-1)
    n_hat = separation / distance[:, None]
    travel = (distance / LDC_C) * (
        1 + np.einsum('na,na->n', n_hat, v_recv) / LDC_C)
    return travel, n_hat


class LdcGalacticBinary(object):
    """gb_fdot.compute_hphc_td followed by hphc.source2SSB."""

    def __init__(self, row):
        self.amplitude = row['amplitude']
        self.frequency = row['frequency']
        self.fdot = row['fdot']
        self.phase0 = row['phase0']
        self.cos_inclination = np.cos(row['inclination'])
        cos2 = np.cos(2 * row['polarization'])
        sin2 = np.sin(2 * row['polarization'])
        self.cos2, self.sin2 = cos2, sin2
        self.lamb, self.beta = row['lamb'], row['beta']

    def polarizations(self, t):
        t = np.asarray(t, dtype=float)
        phase = -self.phase0 + np.pi * (2 * t * self.frequency
                                        + t * t * self.fdot)
        source_plus = -np.cos(phase) * self.amplitude * (
            1 + self.cos_inclination ** 2)
        source_cross = -np.sin(phase) * 2 * self.amplitude * \
            self.cos_inclination
        return (source_plus * self.cos2 - source_cross * self.sin2,
                source_plus * self.sin2 + source_cross * self.cos2)


def ldc_polarization_basis(lamb, beta):
    """hphc.py's projection basis, transcribed.

    Transcribed rather than imported from pycbc: a reference that shares the
    code under test cancels its own errors.  Flipping the sign of v_hat in
    pycbc goes unnoticed if this function is the same object.
    """
    sin_beta, cos_beta = np.sin(beta), np.cos(beta)
    sin_lamb, cos_lamb = np.sin(lamb), np.cos(lamb)
    k_hat = np.array([-cos_beta * cos_lamb, -cos_beta * sin_lamb, -sin_beta])
    v_hat = np.array([-sin_beta * cos_lamb, -sin_beta * sin_lamb, cos_beta])
    u_hat = np.array([sin_lamb, -cos_lamb, 0.0])
    return u_hat, v_hat, k_hat


def ldc_arm_response(source, times, links=LDC_LINKS):
    """ProjectedStrain._arm_response, transcribed."""
    position = ldc_position(times, (1, 2, 3))
    u_hat, v_hat, k_hat = ldc_polarization_basis(source.lamb, source.beta)
    out = np.empty((len(times), len(links)))
    for index, (receiver, emitter) in enumerate(links):
        r_emit = position[:, emitter - 1]
        r_recv = position[:, receiver - 1]
        travel, n_hat = ldc_link(times, receiver, emitter)
        u_n, v_n = n_hat @ u_hat, n_hat @ v_hat
        k_n = n_hat @ k_hat
        xi_plus = 0.5 * (u_n ** 2 - v_n ** 2)
        xi_cross = u_n * v_n
        plus_emit, cross_emit = source.polarizations(
            times - travel - (r_emit @ k_hat) / C_SI)
        plus_recv, cross_recv = source.polarizations(
            times - (r_recv @ k_hat) / C_SI)
        out[:, index] = ((plus_emit - plus_recv) * xi_plus
                         + (cross_emit - cross_recv) * xi_cross) / (1 - k_n)
    return out


def ldc_convention_sample(times, orbit, links=LDC_LINKS):
    """A `ConstellationSample` carrying LDC's simultaneous arm geometry.

    pycbc.tdi keeps the light cone: `velocity_order` and `retard_emitter`
    exist, but there is deliberately no switch for a simultaneous arm
    direction, so LDC's convention is built here rather than in the library.
    """
    position = np.asarray(orbit.compute_position(times, (1, 2, 3)))
    velocity = np.asarray(orbit.compute_velocity(times, (1, 2, 3)))
    n_hat = np.empty((len(times), len(links), 3))
    r_emit = np.empty_like(n_hat)
    travel = np.empty((len(times), len(links)))
    for index, (receiver, emitter) in enumerate(links):
        travel[:, index], n_hat[:, index] = ldc_link(times, receiver, emitter)
        r_emit[:, index] = position[:, emitter - 1]
    return ConstellationSample(t=times, position=position, velocity=velocity,
                               n_hat=n_hat, r_emit=r_emit, ltt=travel,
                               links=links)


def retarded_emitter_sample(times, orbit, links=LDC_LINKS):
    """LDC's arm direction with the emitter at the true emission event.

    Isolates the retardation on its own.  Toggling `retard_emitter` against
    `ldc_convention_sample` is a NO-OP, because that sample holds the
    un-retarded position in both of its slots.
    """
    base = ldc_convention_sample(times, orbit, links)
    r_emit = np.empty_like(base.r_emit)
    for index, (_, emitter) in enumerate(links):
        r_emit[:, index] = ldc_position(times - base.ltt[:, index],
                                        (emitter,))[:, 0]
    return ConstellationSample(t=base.t, position=base.position,
                               velocity=base.velocity, n_hat=base.n_hat,
                               r_emit=r_emit, ltt=base.ltt, links=links)


def test_polarization_basis_matches_ldc():
    """The frame convention, checked against a transcription rather than
    against itself."""
    for row in VERIFICATION_BINARIES:
        got = polarization_basis(row['lamb'], row['beta'])
        want = ldc_polarization_basis(row['lamb'], row['beta'])
        for vector, reference in zip(got, want):
            assert np.max(np.abs(np.asarray(vector) - reference)) < 1e-15


def test_ldc_orbit_is_pycbcs_orbit():
    """Same closed form, to the precision of the constants themselves."""
    from astropy.constants import au

    assert float(au.value) == LDC_AU
    relative = abs(EARTH_ORBIT_ANGULAR_FREQUENCY - LDC_OMEGA) / LDC_OMEGA
    assert relative < 1e-12          # pycbc stores a 12-digit 2*pi/YRSID

    times = np.linspace(0.0, 3.15581498e7, 2001)
    orbit = LisaEqualArmOrbit(armlength=ARM, t0=0.0)
    position_error = np.max(np.linalg.norm(
        np.asarray(orbit.compute_position(times, (1, 2, 3)))
        - ldc_position(times, (1, 2, 3)), axis=-1))
    velocity_error = np.max(np.linalg.norm(
        np.asarray(orbit.compute_velocity(times, (1, 2, 3)))
        - ldc_velocity(times, (1, 2, 3)), axis=-1))
    # the position difference is the omega truncation and nothing else
    expected = LDC_AU * relative * LDC_OMEGA * times[-1]
    assert position_error < 1.0
    assert abs(position_error / expected - 1) < 0.1
    assert velocity_error / (LDC_AU * LDC_OMEGA) < 1e-11

    # t0 is not optional: the default is tuned for BBHx, not for LDC's kappa=0
    default = LisaEqualArmOrbit(armlength=ARM)
    assert np.max(np.linalg.norm(
        np.asarray(default.compute_position(times, (1, 2, 3)))
        - ldc_position(times, (1, 2, 3)), axis=-1)) > 1e11


def test_light_cone_costs_the_aberration_angle():
    """LDC does not aberrate, and the difference is v/c on every link.

    `links=LDC_LINKS` is not optional here: pycbc's default LINK_ORDER is a
    different one, and indexing this sample by LDC's ordering with the
    default silently pairs different arms and reports 2*pi/3.
    """
    times = np.linspace(0.0, 3.15581498e7, 2001)
    orbit = LisaEqualArmOrbit(armlength=ARM, t0=0.0)
    sample = sample_constellation(times, orbit, ltt_order=1, links=LDC_LINKS)
    worst_travel, worst_angle = 0.0, 0.0
    for index, (receiver, emitter) in enumerate(LDC_LINKS):
        travel, n_hat = ldc_link(times, receiver, emitter)
        worst_travel = max(worst_travel, np.max(
            np.abs(sample.ltt[:, index] - travel)) / np.mean(travel))
        worst_angle = max(worst_angle, np.max(np.arccos(np.clip(
            np.einsum('na,na->n', sample.n_hat[:, index], n_hat), -1, 1))))
    assert worst_travel < 1e-8
    speed = LDC_AU * LDC_OMEGA / LDC_C
    assert 0.7 * speed < worst_angle < 1.4 * speed

    # LDC's own order-0 against order-1: the plan's 0.83 ms velocity term
    r_emit = ldc_position(times, (2,))[:, 0]
    r_recv = ldc_position(times, (1,))[:, 0]
    geometric = np.linalg.norm(r_recv - r_emit, axis=-1) / LDC_C
    corrected, _ = ldc_link(times, 1, 2)
    assert 5e-4 < np.max(np.abs(corrected - geometric)) < 2e-3


def test_single_link_reproduces_ldcs_own_response():
    """In LDC's convention pycbc IS LDC, to twelve digits.

    The three ways they differ are each switched on alone afterwards, which
    turns the plan's own numbers -- measured until now only from this
    project's scripts -- into a comparison against an outside implementation.
    """
    times = np.arange(0.0, 3 * 86400.0, 5.0)
    orbit = LisaEqualArmOrbit(armlength=ARM, t0=0.0)
    simultaneous = ldc_convention_sample(times, orbit)
    light_cone = sample_constellation(times, orbit, links=LDC_LINKS)
    retarded = retarded_emitter_sample(times, orbit)

    def worst(source, reference, peak, sample, **kwargs):
        geometry = link_geometry(sample, source.lamb, source.beta, **kwargs)
        response = link_response(source, sample, geometry, links=LDC_LINKS)
        return np.max(np.sqrt(np.mean(
            (response - reference) ** 2, axis=0))) / peak

    frequency, same, doppler, direction, retardation = [], [], [], [], []
    for row in VERIFICATION_BINARIES:
        source = LdcGalacticBinary(row)
        reference = ldc_arm_response(source, times)
        peak = np.max(np.abs(reference))
        frequency.append(row['frequency'])
        same.append(worst(source, reference, peak, simultaneous,
                          velocity_order=0, retard_emitter=False))
        doppler.append(worst(source, reference, peak, simultaneous,
                             velocity_order=1, retard_emitter=False))
        direction.append(worst(source, reference, peak, light_cone,
                               velocity_order=0, retard_emitter=False))
        retardation.append(worst(source, reference, peak, retarded,
                                 velocity_order=0, retard_emitter=True))
    frequency = np.array(frequency)
    same = np.array(same)
    doppler = np.array(doppler)
    direction = np.array(direction)
    retardation = np.array(retardation)

    # velocity_order=0 with a simultaneous arm and no retardation IS LDC
    assert same.max() < 1e-8

    # the arm direction costs ~1.15e-4 and is FLAT in frequency, so across an
    # 18x band it may only move by these sources' sky-position scatter
    assert 5e-5 < direction.min() and direction.max() < 2e-4
    assert direction.max() / direction.min() < 4.0

    # the eps weights instead fall as 1/f: eps*f is far more nearly constant
    assert ((doppler.max() / doppler.min())
            > 3 * (doppler * frequency).max() / (doppler * frequency).min())
    # and they dominate the other two through this band
    assert doppler.mean() > 10 * direction.mean()

    # the retardation is the plan's 4.6e-5, with sky scatter around it
    assert 1e-6 < np.median(retardation) < 1e-3


if __name__ == '__main__':
    import unittest

    # imported here rather than at module scope: `utils` lives in test/ and is
    # only on the path when this file is run the way pycbc's own suite is
    from utils import simple_exit

    class LDCConventions(unittest.TestCase):
        def test_basis(self):
            test_polarization_basis_matches_ldc()

        def test_orbit(self):
            test_ldc_orbit_is_pycbcs_orbit()

        def test_direction(self):
            test_light_cone_costs_the_aberration_angle()

        def test_single_link(self):
            test_single_link_reproduces_ldcs_own_response()

    suite = unittest.TestLoader().loadTestsFromTestCase(LDCConventions)
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
