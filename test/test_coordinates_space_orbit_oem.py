"""`NumericOrbits.from_oem_files` reads the velocity the file supplies.

ESA's OEM rows carry epoch, position, velocity and often acceleration, and
the reader used to keep only the position and differentiate a spline through
it. On the 1-minute ESA file that route reaches 5.8e-3 relative error against
the file's own acceleration column, against 1.6e-8 for differentiating the
velocity column.

The fixtures are synthetic. A test needing a real orbit file can only skip,
and from the outside a skip looks the same as an untested behaviour change; a
written-out file whose velocity is not the derivative of its own position
column settles it in three lines, since a reader that still differentiates
cannot return what the file says.
"""

import os

import numpy as np
import pytest

from pycbc.coordinates.space_orbit import NumericOrbits

EPOCH = '2034-01-01T00:00:00.000'
CADENCE = 3600.0
COUNT = 64


def _write_oem(path, radius_km, phase, velocity_scale=1.0, columns=6,
               frame='EME2000', start=0.0):
    """A minimal but valid CCSDS OEM 2.0 file for a circular orbit.

    ``velocity_scale`` multiplies the velocity column only, leaving the
    position column alone: the file then states a velocity that no
    differentiation of its own positions can produce.
    """
    from astropy.time import Time

    epochs = Time(EPOCH, format='isot', scale='tcb') + \
        np.arange(COUNT) * CADENCE / 86400.0 + start / 86400.0
    rate = 2 * np.pi / (COUNT * CADENCE)
    angle = rate * (np.arange(COUNT) * CADENCE) + phase
    position = radius_km * np.stack(
        [np.cos(angle), np.sin(angle), 0.1 * np.sin(angle)], axis=-1)
    velocity = radius_km * rate * np.stack(
        [-np.sin(angle), np.cos(angle), 0.1 * np.cos(angle)], axis=-1)
    velocity = velocity * velocity_scale
    with open(path, 'w') as handle:
        handle.write('CCSDS_OEM_VERS = 2.0\n')
        handle.write('CREATION_DATE = 2034-01-01T00:00:00\n')
        handle.write('ORIGINATOR = pycbc-test\n')
        handle.write('META_START\n')
        handle.write('OBJECT_NAME = TEST\n')
        handle.write('OBJECT_ID = TEST\n')
        handle.write('CENTER_NAME = SOLAR SYSTEM BARYCENTER\n')
        handle.write(f'REF_FRAME = {frame}\n')
        handle.write('TIME_SYSTEM = TCB\n')
        handle.write('META_STOP\n')
        for index in range(COUNT):
            row = list(position[index])
            if columns >= 6:
                row += list(velocity[index])
            handle.write(epochs[index].isot + ' '
                         + ' '.join(f'{value:.12e}' for value in row) + '\n')
    return position, velocity


def _epoch_gps():
    """The GPS times the fixtures are written at, recomputed here so the test
    does not lean on the object's internals."""
    from astropy.time import Time
    base = Time(EPOCH, format='isot', scale='tcb')
    return np.asarray([(base + index * CADENCE / 86400.0).gps
                       for index in range(COUNT)])


def _three_files(directory, **kwargs):
    paths, positions, velocities = [], [], []
    for index in range(3):
        path = os.path.join(directory, f'sc{index + 1}.oem')
        position, velocity = _write_oem(
            path, 1.4959787e8, index * 2 * np.pi / 3, **kwargs)
        paths.append(path)
        positions.append(position)
        velocities.append(velocity)
    return paths, np.stack(positions, axis=1), np.stack(velocities, axis=1)


def test_velocity_comes_from_the_file_not_from_a_position_spline(tmp_path):
    """State a velocity the positions cannot produce, and see which wins."""
    scale = 1.25
    paths, _, velocity_km_s = _three_files(str(tmp_path),
                                           velocity_scale=scale)
    orbit = NumericOrbits.from_oem_files(*paths)

    # away from the spline's edges, where any interpolant is at its best
    sample = _epoch_gps()[8:-8]
    got = np.asarray(orbit.compute_velocity(sample, (1, 2, 3)))
    speed = np.linalg.norm(got, axis=-1)
    stated = np.linalg.norm(velocity_km_s[8:-8], axis=-1) * 1e3
    assert np.max(np.abs(speed / stated - 1)) < 1e-6

    # and it is genuinely the file's, not the positions': the same orbit
    # written without a velocity column differs by exactly the scale factor
    plain_directory = tmp_path / 'plain'
    plain_directory.mkdir()
    plain_paths, _, _ = _three_files(str(plain_directory), columns=3)
    plain = NumericOrbits.from_oem_files(*plain_paths)
    plain_speed = np.linalg.norm(
        np.asarray(plain.compute_velocity(sample, (1, 2, 3))), axis=-1)
    assert np.max(np.abs(speed / plain_speed - scale)) < 1e-3


def test_position_only_files_still_load(tmp_path):
    """The column is optional, and its absence must not be silent nonsense."""
    paths, position_km, _ = _three_files(str(tmp_path), columns=3)
    orbit = NumericOrbits.from_oem_files(*paths)
    sample = _epoch_gps()[8:-8]
    got = np.asarray(orbit.compute_position(sample, (1, 2, 3)))
    radius = np.linalg.norm(got, axis=-1)
    stated = np.linalg.norm(position_km[8:-8], axis=-1) * 1e3
    assert np.max(np.abs(radius / stated - 1)) < 1e-9
    velocity = np.asarray(orbit.compute_velocity(sample, (1, 2, 3)))
    assert np.all(np.isfinite(velocity))
    assert np.max(np.linalg.norm(velocity, axis=-1)) > 0


def test_the_reader_refuses_what_it_cannot_interpret(tmp_path):
    """A frame it does not convert, and files that disagree on epochs."""
    paths, _, _ = _three_files(str(tmp_path))
    wrong = str(tmp_path / 'icrf.oem')
    _write_oem(wrong, 1.4959787e8, 0.0, frame='ICRF')
    with pytest.raises(ValueError, match='EME2000'):
        NumericOrbits.from_oem_files(wrong, paths[1], paths[2])

    shifted = str(tmp_path / 'shifted.oem')
    _write_oem(shifted, 1.4959787e8, 0.0, start=5.0)
    with pytest.raises(ValueError, match='identical epochs'):
        NumericOrbits.from_oem_files(paths[0], paths[1], shifted)
