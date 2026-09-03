# Copyright (C) 2026 Chayan Chatterjee
#
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

"""Tests of the LILA Observatory lunar triangle: its geometry, its vertex
response tensors, and its three-channel GDC response."""

import unittest

import numpy

from pycbc.coordinates.moon import (LUNAR_RADIUS, _bearing, _unit_from_lonlat,
                                    moon_triangle_sites)
from pycbc.detector import (Detector, SpaceDetector, body_fixed_detector_tensor,
                            get_available_space_detectors)
from utils import simple_exit

# LAL's hardcoded ET vertices: (longitude, latitude, xazimuth, yazimuth) in
# radians, from lal/lib/tools/LALDetectors.h. Used to pin down the triangle
# conventions, which are implicit in these numbers and written down nowhere.
ET_SITES = {
    'E1': (0.18333805213, 0.76151183984, 0.33916285222, 5.57515060820),
    'E2': (0.18405858870, 0.76299307990, 4.52845115854, 3.48125307555),
    'E3': (0.18192996730, 0.76270463257, 2.43258574281, 1.38538766217),
}
EARTH_RADIUS = 6371.0e3

# A representative south-polar LILA site.
LON_C, LAT_C = numpy.radians(-30.0), numpy.radians(-85.0)
ORIENT = numpy.radians(17.0)
ARM = 4.0e4


def _interior_angle(arm_length, radius=LUNAR_RADIUS):
    """Exact interior angle of a spherical equilateral triangle of side
    arc a: from cos a = cos^2 a + sin^2 a cos A, the spherical law of
    cosines specialised to a = b = c. Exceeds 60 degrees by one third of
    the spherical excess; for 40 km on the Moon, 60.00438532 degrees."""
    a = arm_length / radius
    return numpy.arccos(numpy.cos(a) * (1 - numpy.cos(a)) / numpy.sin(a)**2)


def _arc(lon1, lat1, lon2, lat2, radius=LUNAR_RADIUS):
    a = _unit_from_lonlat(lon1, lat1)
    b = _unit_from_lonlat(lon2, lat2)
    return radius * numpy.arccos(numpy.clip(numpy.dot(a, b), -1.0, 1.0))


class TestLILAGeometry(unittest.TestCase):

    def test_equilateral_closure(self):
        """All three sides equal the requested arm length exactly."""
        sites = moon_triangle_sites(LON_C, LAT_C, ARM, orientation=ORIENT)
        self.assertEqual(len(sites), 3)
        for i in range(3):
            j = (i + 1) % 3
            side = _arc(sites[i]['longitude'], sites[i]['latitude'],
                        sites[j]['longitude'], sites[j]['latitude'])
            self.assertAlmostEqual(side, ARM, places=6)

    def test_opening_angle_is_et_convention(self):
        """(yangle - xangle) is -60 degrees, less one third of the
        spherical excess -- ET's convention, on a sphere."""
        sites = moon_triangle_sites(LON_C, LAT_C, ARM, orientation=ORIENT)
        expected = 2 * numpy.pi - _interior_angle(ARM)
        openings = [(s['yangle'] - s['xangle']) % (2 * numpy.pi)
                    for s in sites]
        for opening in openings:
            self.assertAlmostEqual(opening, expected, places=12)
        # All three vertices must be identical, not merely close.
        self.assertLess(max(openings) - min(openings), 1e-12)
        # Sanity: this is 300 degrees to within the spherical excess only.
        self.assertAlmostEqual(numpy.degrees(expected), 300.0, places=2)

    def test_arms_point_at_the_other_vertices(self):
        """x-arm bears on the next vertex, y-arm on the previous one."""
        sites = moon_triangle_sites(LON_C, LAT_C, ARM, orientation=ORIENT)
        for i, site in enumerate(sites):
            nxt = sites[(i + 1) % 3]
            prv = sites[(i - 1) % 3]
            self.assertAlmostEqual(
                site['xangle'],
                _bearing(site['longitude'], site['latitude'],
                         nxt['longitude'], nxt['latitude']), places=12)
            self.assertAlmostEqual(
                site['yangle'],
                _bearing(site['longitude'], site['latitude'],
                         prv['longitude'], prv['latitude']), places=12)

    def test_reproduces_lal_einstein_telescope(self):
        """Fed Earth's radius and ET's centroid, the same construction
        recovers LAL's E1/E2/E3 to sub-arcsecond accuracy. The residual is
        LAL's ellipsoidal-Earth site offsets; what is being tested is that
        the handedness and arm assignment agree, which a sign error in
        either would break by tens of degrees."""
        import pycbc.coordinates.moon as moon_mod

        centroid = sum(_unit_from_lonlat(v[0], v[1])
                       for v in ET_SITES.values()) / 3.0
        centroid /= numpy.linalg.norm(centroid)
        cen_lon = numpy.arctan2(centroid[1], centroid[0])
        cen_lat = numpy.arcsin(centroid[2])

        original = moon_mod.LUNAR_RADIUS
        try:
            moon_mod.LUNAR_RADIUS = EARTH_RADIUS
            orient = _bearing(cen_lon, cen_lat, *ET_SITES['E1'][:2])
            sites = moon_triangle_sites(cen_lon, cen_lat, 1.0e4,
                                        orientation=orient)
        finally:
            moon_mod.LUNAR_RADIUS = original

        arcsec = numpy.pi / 180.0 / 3600.0
        for site, name in zip(sites, ['E1', 'E2', 'E3']):
            lon, lat, xaz, yaz = ET_SITES[name]
            self.assertLess(abs(site['longitude'] - lon), 1.0 * arcsec)
            self.assertLess(abs(site['latitude'] - lat), 1.0 * arcsec)
            # 0.04 deg: LAL's geodetic-vs-spherical azimuth offset, which
            # is uniform across the three vertices.
            self.assertLess(abs(site['xangle'] - xaz), numpy.radians(0.04))
            self.assertLess(abs(site['yangle'] - yaz), numpy.radians(0.04))

    def test_rejects_bad_arm_length(self):
        self.assertRaises(ValueError, moon_triangle_sites,
                          LON_C, LAT_C, 0.0)


class TestLILAResponseTensors(unittest.TestCase):

    def setUp(self):
        self.det = SpaceDetector(
            'LILA', backend='LILAResponse', longitude_site=LON_C,
            latitude_site=LAT_C, orientation=ORIENT).backend

    def test_tensors_are_traceless(self):
        for chan in self.det.channels:
            self.assertAlmostEqual(
                numpy.trace(self.det.responses[chan]), 0.0, places=14)

    def test_null_stream_is_exact(self):
        """The three vertices' response tensors sum to zero identically.
        This is exact geometry, not a small-triangle approximation."""
        total = sum(self.det.responses[c] for c in self.det.channels)
        scale = max(numpy.abs(self.det.responses[c]).max()
                    for c in self.det.channels)
        self.assertLess(numpy.abs(total).max() / scale, 1e-14)

    def test_null_stream_survives_a_huge_triangle(self):
        """Same identity at L/R = 0.5, where any O(L/R) approximation
        would have failed by tens of percent."""
        det = SpaceDetector(
            'LILA', backend='LILAResponse', longitude_site=LON_C,
            latitude_site=LAT_C, arm_length=0.5 * LUNAR_RADIUS).backend
        total = sum(det.responses[c] for c in det.channels)
        scale = max(numpy.abs(det.responses[c]).max() for c in det.channels)
        self.assertLess(numpy.abs(total).max() / scale, 1e-14)

    def test_sixty_degree_v_is_sin60_of_an_l(self):
        """A 60 degree V has sin(60) the peak response of a 90 degree L,
        for an overhead source."""
        def amplitude(opening):
            resps, _, _ = body_fixed_detector_tensor(
                0.0, 0.0, yangle=0.0, xangle=opening)
            resp = numpy.squeeze(resps[0] - resps[1])
            # local up is +x here, so the wave frame spans y and z
            xp = numpy.array([0.0, 1.0, 0.0])
            yp = numpy.array([0.0, 0.0, 1.0])
            fplus = xp @ resp @ xp - yp @ resp @ yp
            fcross = xp @ resp @ yp + yp @ resp @ xp
            return numpy.hypot(fplus, fcross)

        ratio = amplitude(numpy.pi / 3) / amplitude(numpy.pi / 2)
        self.assertAlmostEqual(ratio, numpy.sin(numpy.pi / 3), places=12)

    def test_arms_subtend_sixty_degrees(self):
        sites = moon_triangle_sites(LON_C, LAT_C, ARM, orientation=ORIENT)
        for site in sites:
            _, vecs, _ = body_fixed_detector_tensor(
                site['longitude'], site['latitude'],
                yangle=site['yangle'], xangle=site['xangle'])
            yvec, xvec = vecs
            self.assertAlmostEqual(numpy.linalg.norm(xvec), 1.0, places=12)
            # Not exactly cos(60 deg) = 0.5: on a sphere the interior
            # angle carries a third of the spherical excess.
            self.assertAlmostEqual(float(xvec @ yvec),
                                   numpy.cos(_interior_angle(ARM)), places=12)


class TestLILARegistration(unittest.TestCase):

    def test_aliases_are_discoverable(self):
        available = get_available_space_detectors()
        for name in ['LILA', 'LILA_1', 'LILA_2', 'LILA_3',
                     'LILA_A', 'LILA_E', 'LILA_T']:
            self.assertIn(name, available)

    def test_site_is_required(self):
        """Antenna-pattern geometry needs a real oriented site; unlike
        arrival time it cannot fall back to the Moon's barycenter."""
        self.assertRaises(ValueError, SpaceDetector, 'LILA',
                          backend='LILAResponse')

    def test_unknown_backend_rejected(self):
        self.assertRaises(ValueError, SpaceDetector, 'LILA',
                          backend='NoSuchBackend', longitude_site=LON_C,
                          latitude_site=LAT_C)

    def test_sky_coords(self):
        det = SpaceDetector('LILA', backend='LILAResponse',
                            longitude_site=LON_C, latitude_site=LAT_C)
        self.assertEqual(det.sky_coords,
                         ('eclipticlongitude', 'eclipticlatitude'))


class TestLILAProjection(unittest.TestCase):

    def setUp(self):
        from pycbc.waveform import get_td_waveform
        self.det = SpaceDetector('LILA', backend='LILAResponse',
                                 longitude_site=LON_C, latitude_site=LAT_C,
                                 orientation=ORIENT)
        hp, hc = get_td_waveform(approximant='IMRPhenomD', mass1=1000.,
                                 mass2=800., delta_t=1 / 64., f_lower=0.5,
                                 distance=5000.)
        hp.start_time += 1.4e9
        hc.start_time += 1.4e9
        self.hp, self.hc = hp, hc
        self.sky = dict(lamb=1.2, beta=-0.3, polarization=0.6)

    def test_returns_three_vertex_channels(self):
        out = self.det.project_wave(self.hp, self.hc, **self.sky)
        self.assertEqual(sorted(out), ['LILA_1', 'LILA_2', 'LILA_3'])
        for ts in out.values():
            self.assertEqual(len(ts), len(self.hp))

    def test_channels_share_a_common_grid(self):
        """A coherent three-channel analysis, and the A/E/T
        recombination in particular, requires one grid. Relabelling three
        epochs instead would silently misalign them."""
        out = self.det.project_wave(self.hp, self.hc, include_aet=True,
                                    **self.sky)
        epochs = {float(ts.start_time) for ts in out.values()}
        deltas = {ts.delta_t for ts in out.values()}
        self.assertEqual(len(epochs), 1)
        self.assertEqual(len(deltas), 1)

    def test_aet_channels_present_on_request(self):
        out = self.det.project_wave(self.hp, self.hc, include_aet=True,
                                    **self.sky)
        self.assertEqual(sorted(out),
                         ['LILA_1', 'LILA_2', 'LILA_3',
                          'LILA_A', 'LILA_E', 'LILA_T'])

    def test_vertex_delays_respect_the_geometric_bound(self):
        """No two vertices can differ by more than one side's light time,
        L/c = 133.4 us."""
        _, offsets = self.det.backend._vertex_delays(
            float(self.hp.start_time), self.sky['lamb'], self.sky['beta'])
        spread = max(offsets) - min(offsets)
        self.assertLess(spread, ARM / 299792458.0)
        self.assertGreater(spread, 0.0)

    def test_null_stream_is_suppressed_but_not_zero(self):
        """T must be far below the vertex channels (the tensors cancel)
        yet nonzero (the inter-vertex delays do not). An exactly zero T
        would mean the differential delays were never applied."""
        out = self.det.project_wave(self.hp, self.hc, include_aet=True,
                                    **self.sky)
        single = numpy.abs(out['LILA_1'].numpy()).max()
        null = numpy.abs(out['LILA_T'].numpy()).max()
        self.assertLess(null / single, 1e-1)
        self.assertGreater(null / single, 1e-8)

    def test_null_leakage_grows_with_frequency(self):
        """Leakage enters as 2 pi f dt, so a higher-mass (lower-frequency)
        system must null better. This is the test that a merely small
        residual cannot pass by accident."""
        from pycbc.waveform import get_td_waveform

        leakage = {}
        for m in [800., 4000.]:
            hp, hc = get_td_waveform(approximant='IMRPhenomD', mass1=m,
                                     mass2=0.8 * m, delta_t=1 / 64.,
                                     f_lower=0.5, distance=5000.)
            hp.start_time += 1.4e9
            hc.start_time += 1.4e9
            out = self.det.project_wave(hp, hc, include_aet=True, **self.sky)
            leakage[m] = (numpy.abs(out['LILA_T'].numpy()).max()
                          / numpy.abs(out['LILA_1'].numpy()).max())
        self.assertLess(leakage[4000.], leakage[800.])


class TestGroundDetectorRegression(unittest.TestCase):
    """`add_detector_on_earth` was refactored onto the body-agnostic
    `body_fixed_detector_tensor` core so the lunar code could share it.
    Every existing detector must be untouched by that."""

    def test_existing_detectors_unchanged(self):
        for name in ['H1', 'L1', 'V1', 'K1', 'I1', 'E1', 'E2', 'E3']:
            det = Detector(name)
            self.assertAlmostEqual(numpy.trace(det.response), 0.0, places=12)
            resps, vecs, xangle = body_fixed_detector_tensor(
                det.longitude, det.latitude,
                yangle=det.info['yangle'], xangle=det.info['xangle'],
                xaltitude=det.info['xaltitude'],
                yaltitude=det.info['yaltitude'])
            rebuilt = numpy.squeeze(resps[0] - resps[1])
            self.assertLess(numpy.abs(rebuilt - det.response).max(), 1e-12)

    def test_opening_angle_sign_convention_is_shared(self):
        """The y-arm trails the x-arm by the opening angle for every
        detector PyCBC knows: 270 degrees for the L-shaped ones, 300 for
        ET's 60 degree vertices. LILA follows the same convention, which
        is what makes the ET geometry transferable to the Moon."""
        for name in ['H1', 'L1', 'V1']:
            det = Detector(name)
            opening = ((det.info['yangle'] - det.info['xangle'])
                       % (2 * numpy.pi))
            self.assertAlmostEqual(numpy.degrees(opening), 270.0, places=3)
        for name in ['E1', 'E2', 'E3']:
            det = Detector(name)
            opening = ((det.info['yangle'] - det.info['xangle'])
                       % (2 * numpy.pi))
            self.assertAlmostEqual(numpy.degrees(opening), 300.0, places=3)


suite = unittest.TestSuite()
for case in [TestLILAGeometry, TestLILAResponseTensors, TestLILARegistration,
             TestLILAProjection, TestGroundDetectorRegression]:
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(case))

if __name__ == '__main__':
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
