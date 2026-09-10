"""The ESA numeric orbit files, read the way the reader reads them.

These fetch real ephemerides from https://github.com/esa/lisa-orbit-files,
which ESA publishes under CC-BY-4.0.  Two sets are used:

  default   the ~208 kB per-spacecraft files, on a 2.3-day grid.  Enough to
            show that reading the velocity column beats differentiating a
            position spline, and cheap enough to fetch every run.
  full      the 35 MB per-spacecraft 1-minute files, fetched only when
            PYCBC_LISA_ORBIT_FULL is set.  These carry the numbers the
            reader was actually changed for -- the coarse grid flatters the
            spline, so the margin there is 10x where on the 1-minute grid it
            is 1e4 -- and the microsecond-quantised epochs behind them.

Downloads are cached, so the cost is paid once per machine; point
PYCBC_LISA_ORBIT_FILES at an existing checkout to skip them entirely.  A
network failure SKIPS rather than fails: an unreachable third-party host is
not a defect in this code, and a test that goes red for it teaches the reader
to ignore red.
"""

import os
import tempfile

import numpy as np
import pytest

from pycbc.coordinates.space_orbit import (NumericOrbits, _parse_oem_file,
                                           _icrs_to_ecliptic_rotation_matrix)

BASE = ('https://raw.githubusercontent.com/esa/lisa-orbit-files/master/'
        'crema_2p0/tcb_time_scale/mida-20deg/')
STEM = 'trajectory_out_mida-20deg_cw_sg-2nmss_may_launch_lisa%d%s.oem'
COARSE, FINE = '', '_step_1min'
CHECKOUT = os.environ.get('PYCBC_LISA_ORBIT_FILES')
CACHE = os.environ.get(
    'PYCBC_LISA_ORBIT_CACHE',
    os.path.join(tempfile.gettempdir(), 'pycbc-lisa-orbit-files'))
WINDOW = 720                       # twelve hours of the one-minute grid


def _paths(suffix):
    """Local paths for the triplet, fetching them if they are not there.

    Raises ``OSError`` when the files are neither local nor reachable; the
    callers turn that into a skip.
    """
    names = [STEM % (index, suffix) for index in (1, 2, 3)]
    if CHECKOUT:
        local = [os.path.join(CHECKOUT, 'crema_2p0', 'tcb_time_scale',
                              'mida-20deg', name) for name in names]
        if all(os.path.exists(path) for path in local):
            return local
    os.makedirs(CACHE, exist_ok=True)
    out = []
    for name in names:
        path = os.path.join(CACHE, name)
        if not os.path.exists(path):
            from urllib.request import urlopen
            partial = path + '.part'
            with urlopen(BASE + name, timeout=60) as response, \
                    open(partial, 'wb') as handle:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    handle.write(block)
            os.replace(partial, path)     # never leave a truncated cache file
        out.append(path)
    return out


def _load(suffix=COARSE, count=None):
    """Positions, velocities and the file's OWN accelerations, in pycbc's
    frame and units, plus the raw epoch labels."""
    from astropy.time import Time

    position, velocity, acceleration, gps, labels = [], [], [], None, None
    for path in _paths(suffix):
        meta, epochs, rows = _parse_oem_file(path)
        assert meta['REF_FRAME'] == 'EME2000'
        assert rows.shape[1] >= 9, 'this file carries no acceleration column'
        limit = len(epochs) if count is None else min(count, len(epochs))
        if gps is None:
            gps = Time(epochs[:limit], format='isot', scale='tcb').gps
            labels = epochs[:limit]
        position.append(rows[:limit, 0:3])
        velocity.append(rows[:limit, 3:6])
        acceleration.append(rows[:limit, 6:9])

    rotation = _icrs_to_ecliptic_rotation_matrix()

    def to_ecliptic(columns):
        stacked = np.stack(columns, axis=1) * 1e3
        return (stacked.reshape(-1, 3) @ rotation.T).reshape(stacked.shape)

    return (gps, to_ecliptic(position), to_ecliptic(velocity),
            to_ecliptic(acceleration), labels)


def _fetch_or_skip(suffix=COARSE, count=None):
    try:
        return _load(suffix, count)
    except OSError as error:            # unreachable host, not a defect here
        pytest.skip(f'ESA orbit files unavailable: {error}')


needs_full = pytest.mark.skipif(
    not os.environ.get('PYCBC_LISA_ORBIT_FULL'),
    reason='set PYCBC_LISA_ORBIT_FULL to fetch the 35 MB 1-minute files')


def _acceleration_errors(gps, position, velocity, acceleration):
    """Relative error of both routes against the file's own column."""
    interior = slice(len(gps) // 4, 3 * len(gps) // 4)
    truth = acceleration[interior]
    scale = np.max(np.linalg.norm(truth, axis=-1))

    def error(orbit):
        got = np.asarray(orbit.compute_acceleration(gps[interior], (1, 2, 3)))
        return np.max(np.linalg.norm(got - truth, axis=-1)) / scale

    return (error(NumericOrbits(gps, position, velocity)),
            error(NumericOrbits(gps, position, None)))


def test_velocity_column_beats_differentiating_positions():
    """The measurement PR1b was made for, on the cheap file.

    Both orbits are built from the same rows, so nothing differs but whether
    the velocity column is used or a position spline is differentiated twice.
    The margin here is modest because a 2.3-day grid flatters the spline;
    the 1-minute file below is where it is five orders.
    """
    gps, position, velocity, acceleration, _ = _fetch_or_skip()
    from_velocity, from_position = _acceleration_errors(
        gps, position, velocity, acceleration)
    assert from_velocity < 1e-6
    assert from_position / from_velocity > 5.0


def test_the_reader_uses_the_column_when_it_is_there():
    """from_oem_files on a real triplet, against the file's own velocity."""
    gps, _, velocity, _, _ = _fetch_or_skip()
    try:
        orbit = NumericOrbits.from_oem_files(*_paths(COARSE))
    except OSError as error:
        pytest.skip(f'ESA orbit files unavailable: {error}')
    interior = slice(len(gps) // 4, 3 * len(gps) // 4)
    got = np.asarray(orbit.compute_velocity(gps[interior], (1, 2, 3)))
    stated = velocity[interior]
    scale = np.max(np.linalg.norm(stated, axis=-1))
    assert np.max(np.linalg.norm(got - stated, axis=-1)) / scale < 1e-10


@needs_full
def test_the_one_minute_grid_shows_the_full_five_orders():
    """Where the reader's change is worth what it is worth."""
    gps, position, velocity, acceleration, _ = _fetch_or_skip(FINE, WINDOW)
    from_velocity, from_position = _acceleration_errors(
        gps, position, velocity, acceleration)
    assert from_velocity < 1e-7
    assert from_position > 1e-3
    assert from_position / from_velocity > 1e4


@needs_full
def test_the_one_minute_epochs_are_not_a_uniform_grid():
    """The dominant error in the position route is in the timestamps.

    The labels are microsecond-quantised, so a 'one-minute' file is not on a
    60 s grid and a second derivative divides by the square of a spacing that
    wobbles.  Recorded because two earlier explanations were wrong: a
    position text quantisation that does not exist -- the files carry 18
    significant digits -- and an assumption that the epochs are exactly
    uniform, which made the GPS conversion the suspect.  It is the secondary
    term, not the cause.
    """
    _, _, _, _, labels = _fetch_or_skip(FINE, WINDOW)
    endings = [label.split('.')[-1] for label in labels]
    counts = {}
    for ending in endings:
        counts[ending] = counts.get(ending, 0) + 1
    assert len(counts) > 1, 'expected microsecond-quantised labels'
    commonest = max(counts.values())
    assert 0.0 < (len(endings) - commonest) / len(endings) < 0.2
