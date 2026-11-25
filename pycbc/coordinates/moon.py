# Copyright (C) 2026  Shichao Wu, Alex Nitz, Jacopo Tissino, Jan Harms
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

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
"""
This module extends `pycbc.coordinates.space`'s SSB-hub coordinate
transforms to a lunar detector (e.g. LGWA), for arrival time, sky
localization, and polarization angle -- the pieces needed for coherent
multiband parameter estimation combining a lunar detector with LISA-family
and ground-based detectors. It does not implement a lunar antenna-pattern
or single-link response function (the analogue of that work for LISA/Taiji/
TianQin is also still pending); this is purely the coordinate layer.

Design, following Tissino et al. 2026 ("The geometry of lunar gravitational
wave detection", arXiv:2606.04918), which shows (their Eq. 7-9) that a
source's sky position and polarization angle do not need to be re-expressed
between solar-system-barycentric-comoving frames -- "these vectors are
always constant for a given source, as motion within the Solar System is
negligible compared to the source distance". Accordingly, the SSB -> Moon
rotation used here is the identity: `longitude`/`latitude`/`polarization`
come out numerically unchanged between the SSB and Moon frames, and only
the arrival time changes. This is analogous to how `ssb_to_lisa` does not
apply any parallax correction to sky position either -- only to arrival
time.

What *does* need the Moon's precise position (and, for a specific surface
site, its libration) is the arrival time itself (Tissino et al. Eq. 12-13):
`moon_site_position_ssb` supports both the Moon's center (real ephemeris
only, no extra dependency) and a specific surface site (via the optional
`lunarsky` package, which models real lunar libration). The lunar body's
*orientation* (needed for an actual antenna-pattern/response function, as
in Tissino et al. Eq. 13's `n`, `b` vectors) is not modeled here at all --
that is response-function work, not coordinate-layer work.
"""

import numpy as np
from astropy.constants import c
from scipy.optimize import fsolve

from pycbc.coordinates.space import (
    geo_to_ssb,
    lisa_to_ssb,
    localization_to_propagation_vector,
    polarization_newframe,
    propagation_vector_to_localization,
    ssb_to_geo,
    ssb_to_lisa,
)

__all__ = [
    "LUNAR_RADIUS",
    "rotation_matrix_ssb_to_moon",
    "moon_triangle_sites",
    "moon_site_position_ssb",
    "t_moon_from_ssb",
    "t_ssb_from_t_moon",
    "ssb_to_moon",
    "moon_to_ssb",
    "moon_to_geo",
    "geo_to_moon",
    "moon_to_lisa",
    "lisa_to_moon",
]


# `lunarsky`'s MoonLocation uses a perfect sphere (its `ellipsoid` is
# literally "SPHERE") of this radius, so the spherical trigonometry in
# `moon_triangle_sites` is exact rather than a first-order approximation --
# unlike the equivalent construction on Earth's reference ellipsoid.
LUNAR_RADIUS = 1737100.0


def _unit_from_lonlat(longitude, latitude):
    """Selenodetic longitude/latitude (radians) -> body-fixed unit vector."""
    return np.array([
        np.cos(latitude) * np.cos(longitude),
        np.cos(latitude) * np.sin(longitude),
        np.sin(latitude),
    ])


def _enu_basis(longitude, latitude):
    """Local East/North/Up unit vectors at a selenodetic site."""
    east = np.array([-np.sin(longitude), np.cos(longitude), 0.0])
    north = np.array([
        -np.sin(latitude) * np.cos(longitude),
        -np.sin(latitude) * np.sin(longitude),
        np.cos(latitude),
    ])
    return east, north, _unit_from_lonlat(longitude, latitude)


def _bearing(lon1, lat1, lon2, lat2):
    """Initial great-circle bearing from site 1 to site 2, in radians
    measured from local north toward east -- the same azimuth convention
    as `pycbc.detector.add_detector_on_earth`."""
    east, north, up = _enu_basis(lon1, lat1)
    target = _unit_from_lonlat(lon2, lat2)
    tangent = target - up * np.dot(up, target)
    return np.arctan2(np.dot(tangent, east),
                      np.dot(tangent, north)) % (2 * np.pi)


def moon_triangle_sites(longitude_center, latitude_center, arm_length,
                        orientation=0.0, height=0.0):
    """Place the three vertices of an equilateral triangular observatory
    on the lunar surface, and give each vertex's arm azimuths.

    This is the lunar analogue of the Einstein Telescope's layout, and
    follows exactly the same conventions, which are implicit in LAL's
    hardcoded E1/E2/E3 constants rather than written down anywhere:

      * The vertices run counter-clockwise as seen from outside the body.
      * At each vertex the x-arm points along the triangle side toward the
        next vertex and the y-arm toward the previous one, so the two arms
        subtend 60 degrees and `(yangle - xangle) mod 2*pi` is 300 degrees.
      * Azimuths are measured from local north toward east.

    Feeding this function Earth's radius and ET's centroid reproduces
    LAL's E1/E2/E3 vertex coordinates to sub-arcsecond accuracy (the
    residual being LAL's ellipsoidal-Earth site offsets); see
    `test/test_detector_lila.py`.

    Parameters
    ----------
    longitude_center, latitude_center : float
        Selenodetic longitude/latitude of the triangle centroid, in the
        unit of 'radian'.
    arm_length : float
        Triangle side length, i.e. the arm length of each vertex
        interferometer, in the unit of 'm'.
    orientation : float, optional
        Bearing of the first vertex as seen from the centroid, in the unit
        of 'radian'. Rotates the whole triangle in the surface plane.
        Default 0.
    height : float, optional
        Height of the vertices above the reference selenoid, in the unit
        of 'm'. Default 0.

    Returns
    -------
    sites : list of dict
        Three entries, each with keys 'longitude', 'latitude', 'height',
        'xangle', 'yangle' (radians and metres), ready to pass to
        `pycbc.detector.body_fixed_detector_tensor`.
    """
    if arm_length <= 0:
        raise ValueError("arm_length must be positive")

    # Circumradius, as an angle subtended at the Moon's centre, of a
    # *spherical* equilateral triangle of side arc `a`. Writing each vertex
    # as cos(theta) u + sin(theta)(...) about the centroid axis u, with the
    # three azimuths 120 degrees apart, the vertex-vertex dot product is
    #     cos(a) = cos^2(theta) + sin^2(theta) cos(120 deg)
    #            = 1 - 1.5 sin^2(theta),
    # whose cancellation-free solution is sin(theta) = 2 sin(a/2)/sqrt(3).
    # The familiar flat-space circumradius arm_length/sqrt(3) is the
    # small-angle limit of this.
    a = arm_length / LUNAR_RADIUS
    theta = np.arcsin(2.0 / np.sqrt(3.0) * np.sin(a / 2.0))

    # Decreasing bearing from the centroid puts the vertices in
    # counter-clockwise order, matching LAL's E1 -> E2 -> E3 handedness.
    # The opposite handedness flips the sign of the opening angle, and
    # with it the sign of Fx.
    lonlats = []
    for k in range(3):
        bearing = orientation - k * (2 * np.pi / 3)
        east, north, up = _enu_basis(longitude_center, latitude_center)
        tangent = np.cos(bearing) * north + np.sin(bearing) * east
        vec = np.cos(theta) * up + np.sin(theta) * tangent
        lonlats.append((np.arctan2(vec[1], vec[0]),
                        np.arcsin(np.clip(vec[2], -1.0, 1.0))))

    sites = []
    for i in range(3):
        longitude, latitude = lonlats[i]
        sites.append({
            "longitude": longitude,
            "latitude": latitude,
            "height": height,
            "xangle": _bearing(longitude, latitude, *lonlats[(i + 1) % 3]),
            "yangle": _bearing(longitude, latitude, *lonlats[(i - 1) % 3]),
        })
    return sites


def rotation_matrix_ssb_to_moon():
    """The rotation matrix (of frame basis) from the SSB frame to the
    lunar frame used throughout this module.

    This is the identity: the lunar frame here is ecliptic-axis-aligned
    with the SSB frame (translated to the Moon's position, not rotated) --
    the Moon's own body orientation/libration is not modeled at this
    coordinate-unification layer (see module docstring). A future lunar
    antenna-pattern/response function would need a real, libration-aware,
    body-fixed rotation instead; this placeholder is not it.

    Returns
    -------
    r : numpy.array
        A 3x3 identity matrix.
    """
    return np.eye(3)


def moon_site_position_ssb(t_moon, longitude=None, latitude=None, height=0.0):
    """Calculating the position vector of a point on (or above) the Moon
    in the SSB frame, at a given time.

    With `longitude`/`latitude` left at their default of None, this gives
    the Moon's barycenter position, using astropy's real solar-system
    ephemeris (`pycbc.coordinates.space_orbit._real_body_position_velocity`)
    -- no extra dependency beyond astropy. Passing an explicit selenodetic
    `longitude`/`latitude` instead gives the precise position of that
    specific surface site, including real lunar libration, via the
    `lunarsky` package (not a pycbc dependency; only imported if a site is
    requested).

    Parameters
    ----------
    t_moon : float
        The time at this position, in the unit of 's' (GPS time).
    longitude, latitude : float or None, optional
        Selenodetic longitude/latitude of a surface site, in the unit of
        'radian'. Default None (both must be None, or both given), which
        gives the Moon's barycenter.
    height : float, optional
        Height above the reference selenoid, in the unit of 'm'. Only
        used if `longitude`/`latitude` are given. Default 0.0.

    Returns
    -------
    p : numpy.array
        The position vector in the SSB frame, shape (3, 1), in the unit
        of 'm'.
    """
    if longitude is None and latitude is None:
        from pycbc.coordinates.space_orbit import _real_body_position_velocity

        pos, _ = _real_body_position_velocity(t_moon, "moon")
        return pos.reshape(3, 1)

    if longitude is None or latitude is None:
        raise ValueError(
            "longitude and latitude must either both be given (a specific "
            "surface site), or both left as None (the Moon's barycenter)."
        )

    try:
        from lunarsky import MoonLocation
    except ImportError as exc:
        raise ImportError(
            "lunarsky is required for moon_site_position_ssb with an "
            "explicit site location (it models real lunar libration); "
            "pip install lunarsky, or omit longitude/latitude to use the "
            "Moon's barycenter instead (no lunarsky needed)."
        ) from exc
    from astropy import units as apy_units
    from astropy.coordinates import ICRS
    from astropy.time import Time

    from pycbc.coordinates.space_orbit import _icrs_to_ecliptic_rotation_matrix

    site = MoonLocation.from_selenodetic(
        lon=longitude * apy_units.rad,
        lat=latitude * apy_units.rad,
        height=height * apy_units.m,
    )
    icrs = site.get_mcmf(Time(t_moon, format="gps")).transform_to(ICRS())
    pos_icrs = np.array(
        [
            icrs.cartesian.x.to(apy_units.m).value,
            icrs.cartesian.y.to(apy_units.m).value,
            icrs.cartesian.z.to(apy_units.m).value,
        ]
    )
    rotation = _icrs_to_ecliptic_rotation_matrix()
    return (rotation @ pos_icrs).reshape(3, 1)


def t_moon_from_ssb(
    t_ssb, longitude_ssb, latitude_ssb, longitude_site=None, latitude_site=None
):
    """Calculating the time when a GW signal arrives at a point on the
    Moon, by using the time and sky localization in the SSB frame.

    Parameters
    ----------
    t_ssb : float
        The time when a GW signal arrives at the origin of SSB frame.
        In the unit of 's'.
    longitude_ssb : float
        The ecliptic longitude of a GW signal in SSB frame.
        In the unit of 'radian'.
    latitude_ssb : float
        The ecliptic latitude of a GW signal in SSB frame.
        In the unit of 'radian'.
    longitude_site, latitude_site : float or None, optional
        See `moon_site_position_ssb`. Default None (the Moon's
        barycenter).

    Returns
    -------
    t_moon : float
        The time when a GW signal arrives at the given point on the Moon.
    """
    k = localization_to_propagation_vector(
        longitude_ssb, latitude_ssb, use_astropy=False
    )

    def equation(t_moon):
        # the Moon is moving, when GW arrives at the given point,
        # time is t_moon, not t_ssb.
        p = moon_site_position_ssb(t_moon, longitude_site, latitude_site)
        return t_moon - t_ssb - np.vdot(k, p) / c.value

    return fsolve(equation, t_ssb)[0]


def t_ssb_from_t_moon(
    t_moon, longitude_ssb, latitude_ssb, longitude_site=None, latitude_site=None
):
    """Calculating the time when a GW signal arrives at the barycenter
    of SSB, by using the time at a point on the Moon and sky localization
    in the SSB frame.

    Parameters
    ----------
    t_moon : float
        The time when a GW signal arrives at the given point on the Moon.
        In the unit of 's'.
    longitude_ssb : float
        The ecliptic longitude of a GW signal in SSB frame.
        In the unit of 'radian'.
    latitude_ssb : float
        The ecliptic latitude of a GW signal in SSB frame.
        In the unit of 'radian'.
    longitude_site, latitude_site : float or None, optional
        See `moon_site_position_ssb`. Default None (the Moon's
        barycenter).

    Returns
    -------
    t_ssb : float
        The time when a GW signal arrives at the origin of SSB frame.
    """
    k = localization_to_propagation_vector(
        longitude_ssb, latitude_ssb, use_astropy=False
    )

    # the Moon is moving, when GW arrives at the given point,
    # time is t_moon, not t_ssb.
    p = moon_site_position_ssb(t_moon, longitude_site, latitude_site)

    def equation(t_ssb):
        return t_moon - t_ssb - np.vdot(k, p) / c.value

    return fsolve(equation, t_moon)[0]


def ssb_to_moon(
    t_ssb,
    longitude_ssb,
    latitude_ssb,
    polarization_ssb,
    longitude_site=None,
    latitude_site=None,
):
    """Converting the arrive time, the sky localization, and the
    polarization from the SSB frame to a lunar frame.

    Because `rotation_matrix_ssb_to_moon` is the identity (see module
    docstring), `longitude`/`latitude`/`polarization` come out numerically
    identical to the SSB-frame input; only the arrival time changes.

    Parameters
    ----------
    t_ssb : float or numpy.array
        The time when a GW signal arrives at the origin of SSB frame.
        In the unit of 's'.
    longitude_ssb : float or numpy.array
        The ecliptic longitude of a GW signal in SSB frame.
        In the unit of 'radian'.
    latitude_ssb : float or numpy.array
        The ecliptic latitude of a GW signal in SSB frame.
        In the unit of 'radian'.
    polarization_ssb : float or numpy.array
        The polarization angle of a GW signal in SSB frame.
        In the unit of 'radian'.
    longitude_site, latitude_site : float or None, optional
        See `moon_site_position_ssb`. Default None (the Moon's
        barycenter).

    Returns
    -------
    (t_moon, longitude_moon, latitude_moon, polarization_moon) : tuple
    t_moon : float or numpy.array
        The time when a GW signal arrives at the given point on the Moon.
        In the unit of 's'.
    longitude_moon : float or numpy.array
        The longitude of a GW signal in the lunar frame, in the unit of
        'radian'.
    latitude_moon : float or numpy.array
        The latitude of a GW signal in the lunar frame, in the unit of
        'radian'.
    polarization_moon : float or numpy.array
        The polarization angle of a GW signal in the lunar frame.
        In the unit of 'radian'.
    """
    if not isinstance(t_ssb, np.ndarray):
        t_ssb = np.array([t_ssb])
    if not isinstance(longitude_ssb, np.ndarray):
        longitude_ssb = np.array([longitude_ssb])
    if not isinstance(latitude_ssb, np.ndarray):
        latitude_ssb = np.array([latitude_ssb])
    if not isinstance(polarization_ssb, np.ndarray):
        polarization_ssb = np.array([polarization_ssb])
    num = len(t_ssb)
    t_moon = np.full(num, np.nan)
    longitude_moon = np.full(num, np.nan)
    latitude_moon = np.full(num, np.nan)
    polarization_moon = np.full(num, np.nan)

    rotation_matrix = rotation_matrix_ssb_to_moon()

    for i in range(num):
        if longitude_ssb[i] < 0 or longitude_ssb[i] >= 2 * np.pi:
            raise ValueError("Longitude should within [0, 2*pi).")
        if latitude_ssb[i] < -np.pi / 2 or latitude_ssb[i] > np.pi / 2:
            raise ValueError("Latitude should within [-pi/2, pi/2].")
        if polarization_ssb[i] < 0 or polarization_ssb[i] >= 2 * np.pi:
            raise ValueError("Polarization angle should within [0, 2*pi).")
        t_moon[i] = t_moon_from_ssb(
            t_ssb[i], longitude_ssb[i], latitude_ssb[i], longitude_site, latitude_site
        )
        k_ssb = localization_to_propagation_vector(
            longitude_ssb[i], latitude_ssb[i], use_astropy=False
        )
        k_moon = rotation_matrix.T @ k_ssb
        longitude_moon[i], latitude_moon[i] = propagation_vector_to_localization(
            k_moon, use_astropy=False
        )
        polarization_moon[i] = polarization_newframe(
            polarization_ssb[i], k_ssb, rotation_matrix, use_astropy=False
        )

    if num == 1:
        params_moon = (
            t_moon[0],
            longitude_moon[0],
            latitude_moon[0],
            polarization_moon[0],
        )
    else:
        params_moon = (t_moon, longitude_moon, latitude_moon, polarization_moon)

    return params_moon


def moon_to_ssb(
    t_moon,
    longitude_moon,
    latitude_moon,
    polarization_moon,
    longitude_site=None,
    latitude_site=None,
):
    """Converting the arrive time, the sky localization, and the
    polarization from a lunar frame to the SSB frame. Inverse of
    `ssb_to_moon`.

    Parameters
    ----------
    t_moon : float or numpy.array
        The time when a GW signal arrives at the given point on the Moon.
        In the unit of 's'.
    longitude_moon : float or numpy.array
        The longitude of a GW signal in the lunar frame, in the unit of
        'radian'.
    latitude_moon : float or numpy.array
        The latitude of a GW signal in the lunar frame, in the unit of
        'radian'.
    polarization_moon : float or numpy.array
        The polarization angle of a GW signal in the lunar frame.
        In the unit of 'radian'.
    longitude_site, latitude_site : float or None, optional
        See `moon_site_position_ssb`. Default None (the Moon's
        barycenter).

    Returns
    -------
    (t_ssb, longitude_ssb, latitude_ssb, polarization_ssb) : tuple
    t_ssb : float or numpy.array
        The time when a GW signal arrives at the origin of SSB frame.
        In the unit of 's'.
    longitude_ssb : float or numpy.array
        The ecliptic longitude of a GW signal in SSB frame.
        In the unit of 'radian'.
    latitude_ssb : float or numpy.array
        The ecliptic latitude of a GW signal in SSB frame.
        In the unit of 'radian'.
    polarization_ssb : float or numpy.array
        The polarization angle of a GW signal in SSB frame.
        In the unit of 'radian'.
    """
    if not isinstance(t_moon, np.ndarray):
        t_moon = np.array([t_moon])
    if not isinstance(longitude_moon, np.ndarray):
        longitude_moon = np.array([longitude_moon])
    if not isinstance(latitude_moon, np.ndarray):
        latitude_moon = np.array([latitude_moon])
    if not isinstance(polarization_moon, np.ndarray):
        polarization_moon = np.array([polarization_moon])
    num = len(t_moon)
    t_ssb = np.full(num, np.nan)
    longitude_ssb = np.full(num, np.nan)
    latitude_ssb = np.full(num, np.nan)
    polarization_ssb = np.full(num, np.nan)

    rotation_matrix = rotation_matrix_ssb_to_moon()

    for i in range(num):
        if longitude_moon[i] < 0 or longitude_moon[i] >= 2 * np.pi:
            raise ValueError("Longitude should within [0, 2*pi).")
        if latitude_moon[i] < -np.pi / 2 or latitude_moon[i] > np.pi / 2:
            raise ValueError("Latitude should within [-pi/2, pi/2].")
        if polarization_moon[i] < 0 or polarization_moon[i] >= 2 * np.pi:
            raise ValueError("Polarization angle should within [0, 2*pi).")
        k_moon = localization_to_propagation_vector(
            longitude_moon[i], latitude_moon[i], use_astropy=False
        )
        k_ssb = rotation_matrix @ k_moon
        longitude_ssb[i], latitude_ssb[i] = propagation_vector_to_localization(
            k_ssb, use_astropy=False
        )
        polarization_ssb[i] = polarization_newframe(
            polarization_moon[i], k_moon, rotation_matrix.T, use_astropy=False
        )
        t_ssb[i] = t_ssb_from_t_moon(
            t_moon[i], longitude_ssb[i], latitude_ssb[i], longitude_site, latitude_site
        )

    if num == 1:
        params_ssb = (t_ssb[0], longitude_ssb[0], latitude_ssb[0], polarization_ssb[0])
    else:
        params_ssb = (t_ssb, longitude_ssb, latitude_ssb, polarization_ssb)

    return params_ssb


def moon_to_geo(
    t_moon,
    longitude_moon,
    latitude_moon,
    polarization_moon,
    longitude_site=None,
    latitude_site=None,
    use_astropy=True,
    lal_convention=True,
):
    """Converting the arrive time, the sky localization, and the
    polarization from a lunar frame to the geocentric frame.

    Parameters
    ----------
    t_moon, longitude_moon, latitude_moon, polarization_moon :
        See `moon_to_ssb`.
    longitude_site, latitude_site : float or None, optional
        See `moon_site_position_ssb`. Default None (the Moon's
        barycenter).
    use_astropy : bool, optional
        Using Astropy to calculate the sky localization or not.
        Default is True.
    lal_convention : bool, optional
        `space.ssb_to_geo`/`space.geo_to_ssb` unconditionally apply a
        +/-pi polarization flip to match the LDC-vs-LAL convention used
        by LAL ground-detector response codes (see their docstrings/
        source, citing LDC manual Sec 4.1.5). Default True reproduces
        that (unchanged, already-tested) behavior -- appropriate when
        this GEO-frame output will feed a LAL-convention ground-detector
        response (e.g. LGWA+ET coherent multiband PE). Set to False to
        skip the flip, when the GEO-frame ra/dec/polarization instead
        feeds a consumer with its own, unrelated polarization convention
        (e.g. the `lgwa_response` package's antenna-pattern formula).
        Implemented by applying the flip's own inverse (itself, since
        +pi mod 2*pi is self-inverse) once more to cancel it, rather than
        by a separate non-flipping implementation of the rotation.

    Returns
    -------
    (t_geo, longitude_geo, latitude_geo, polarization_geo) : tuple
        See `pycbc.coordinates.space.ssb_to_geo`.
    """
    t_ssb, longitude_ssb, latitude_ssb, polarization_ssb = moon_to_ssb(
        t_moon,
        longitude_moon,
        latitude_moon,
        polarization_moon,
        longitude_site,
        latitude_site,
    )
    t_geo, longitude_geo, latitude_geo, polarization_geo = ssb_to_geo(
        t_ssb, longitude_ssb, latitude_ssb, polarization_ssb, use_astropy
    )
    if not lal_convention:
        polarization_geo = np.mod(polarization_geo - np.pi, 2 * np.pi)
    return t_geo, longitude_geo, latitude_geo, polarization_geo


def geo_to_moon(
    t_geo,
    longitude_geo,
    latitude_geo,
    polarization_geo,
    longitude_site=None,
    latitude_site=None,
    use_astropy=True,
    lal_convention=True,
):
    """Converting the arrive time, the sky localization, and the
    polarization from the geocentric frame to a lunar frame.

    Parameters
    ----------
    t_geo, longitude_geo, latitude_geo, polarization_geo :
        See `pycbc.coordinates.space.geo_to_ssb`.
    longitude_site, latitude_site : float or None, optional
        See `moon_site_position_ssb`. Default None (the Moon's
        barycenter).
    use_astropy : bool, optional
        Using Astropy to calculate the sky localization or not.
        Default is True.
    lal_convention : bool, optional
        Inverse of `moon_to_geo`'s `lal_convention` flag: set to False if
        `polarization_geo` was produced without the LDC/LAL +/-pi flip
        (e.g. it came from a `lgwa_response`-convention polarization
        angle, not from `space.ssb_to_geo`/`moon_to_geo(lal_convention=
        True)`). Default True reproduces the original, already-tested
        behavior unchanged.

    Returns
    -------
    (t_moon, longitude_moon, latitude_moon, polarization_moon) : tuple
        See `ssb_to_moon`.
    """
    if not lal_convention:
        polarization_geo = np.mod(polarization_geo + np.pi, 2 * np.pi)
    t_ssb, longitude_ssb, latitude_ssb, polarization_ssb = geo_to_ssb(
        t_geo, longitude_geo, latitude_geo, polarization_geo, use_astropy
    )
    return ssb_to_moon(
        t_ssb,
        longitude_ssb,
        latitude_ssb,
        polarization_ssb,
        longitude_site,
        latitude_site,
    )


def moon_to_lisa(
    t_moon,
    longitude_moon,
    latitude_moon,
    polarization_moon,
    longitude_site=None,
    latitude_site=None,
    t0=None,
    orbit=None,
    sc=(1, 2, 3),
):
    """Converting the arrive time, the sky localization, and the
    polarization from a lunar frame to the LISA (or, via `orbit`, any
    constellation) frame.

    Parameters
    ----------
    t_moon, longitude_moon, latitude_moon, polarization_moon :
        See `moon_to_ssb`.
    longitude_site, latitude_site : float or None, optional
        See `moon_site_position_ssb`. Default None (the Moon's
        barycenter).
    t0 : float or None, optional
        See `pycbc.coordinates.space.ssb_to_lisa`. Default None, which
        uses that function's own default.
    orbit : OrbitProvider, optional
        See `pycbc.coordinates.space.ssb_to_lisa`. Default None (analytic
        circular LISA orbit); passing a Taiji/TianQin/numeric orbit here
        makes this a Moon<->Taiji or Moon<->TianQin transform.
    sc : tuple, optional
        1-indexed spacecraft labels defining the constellation. Only used
        if `orbit` is given. Default (1, 2, 3).

    Returns
    -------
    (t_lisa, longitude_lisa, latitude_lisa, polarization_lisa) : tuple
        See `pycbc.coordinates.space.ssb_to_lisa`.
    """
    t_ssb, longitude_ssb, latitude_ssb, polarization_ssb = moon_to_ssb(
        t_moon,
        longitude_moon,
        latitude_moon,
        polarization_moon,
        longitude_site,
        latitude_site,
    )
    kwargs = {"orbit": orbit, "sc": sc}
    if t0 is not None:
        kwargs["t0"] = t0
    return ssb_to_lisa(t_ssb, longitude_ssb, latitude_ssb, polarization_ssb, **kwargs)


def lisa_to_moon(
    t_lisa,
    longitude_lisa,
    latitude_lisa,
    polarization_lisa,
    longitude_site=None,
    latitude_site=None,
    t0=None,
    orbit=None,
    sc=(1, 2, 3),
):
    """Converting the arrive time, the sky localization, and the
    polarization from the LISA (or, via `orbit`, any constellation) frame
    to a lunar frame.

    Parameters
    ----------
    t_lisa, longitude_lisa, latitude_lisa, polarization_lisa :
        See `pycbc.coordinates.space.lisa_to_ssb`.
    longitude_site, latitude_site : float or None, optional
        See `moon_site_position_ssb`. Default None (the Moon's
        barycenter).
    t0 : float or None, optional
        See `pycbc.coordinates.space.lisa_to_ssb`. Default None, which
        uses that function's own default.
    orbit : OrbitProvider, optional
        See `pycbc.coordinates.space.lisa_to_ssb`. Default None (analytic
        circular LISA orbit); passing a Taiji/TianQin/numeric orbit here
        makes this a Taiji<->Moon or TianQin<->Moon transform.
    sc : tuple, optional
        1-indexed spacecraft labels defining the constellation. Only used
        if `orbit` is given. Default (1, 2, 3).

    Returns
    -------
    (t_moon, longitude_moon, latitude_moon, polarization_moon) : tuple
        See `ssb_to_moon`.
    """
    kwargs = {"orbit": orbit, "sc": sc}
    if t0 is not None:
        kwargs["t0"] = t0
    t_ssb, longitude_ssb, latitude_ssb, polarization_ssb = lisa_to_ssb(
        t_lisa, longitude_lisa, latitude_lisa, polarization_lisa, **kwargs
    )
    return ssb_to_moon(
        t_ssb,
        longitude_ssb,
        latitude_ssb,
        polarization_ssb,
        longitude_site,
        latitude_site,
    )
