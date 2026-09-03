"""Tests for the sparse on-the-fly TDI evaluator."""

import numpy as np

from pycbc.coordinates.space_orbit import LisaEqualArmOrbit
from pycbc.tdi.backends.pytdi_backend import (PyTDICombinationAdapter,
                                              combine_links,
                                              get_pytdi_combination)
from pycbc.tdi.onthefly import (adaptive_time_grid, chain_delay,
                                reconstruct, sparse_channel)
from pycbc.tdi.response import (link_geometry, link_response,
                                sample_constellation)
from pycbc.tdi.sources import NewtonianChirp


def _terms(name="X2", delta_t=5.0):
    return PyTDICombinationAdapter(
        name, get_pytdi_combination(name), delta_t=delta_t).terms()


def test_chain_delay_reproduces_wang_table_1():
    """Delays summed from the orbit, without build_shifts or dsp.timeshift."""
    orbit = LisaEqualArmOrbit()
    times = np.linspace(1e5, 2e7, 9)
    arm = float(np.mean(sample_constellation(times, orbit).ltt))
    chains = sorted({term.operators for term in _terms("X2")}, key=len)

    for chain in chains:
        delay = chain_delay(orbit, times, chain)
        # every operator in a Michelson chain is one arm
        assert np.allclose(delay / arm, len(chain), rtol=2e-4)

    extent = max(chain_delay(orbit, times, c).mean() for c in chains) \
        - min(chain_delay(orbit, times, c).mean() for c in chains)
    assert np.isclose(extent / arm, 7.0, rtol=1e-3)   # fact 3 / Wang Table 1


def test_chain_delay_prefix_cache_changes_nothing():
    """The memoisation is an optimisation, not a different calculation."""
    orbit = LisaEqualArmOrbit()
    times = np.linspace(1e5, 5e6, 7)
    chains = sorted({term.operators for term in _terms("X2")}, key=len)
    shared = {}
    for chain in chains:
        cached = chain_delay(orbit, times, chain, cache=shared)
        fresh = chain_delay(orbit, times, chain, cache={})
        assert np.array_equal(cached, fresh)


def test_light_cone_solve_survives_a_two_year_mission_time():
    """A fixed 1e-12 s tolerance is below the float64 spacing of t at 6e7 s.

    Before the tolerance was floored at the representable resolution this
    raised 'light-cone solve did not converge' for any late-mission time.
    """
    orbit = LisaEqualArmOrbit()
    for epoch in (1e5, 6.3e7, 3.2e8):
        sample = sample_constellation(
            np.linspace(epoch, epoch + 1e4, 32), orbit)
        assert np.all(np.isfinite(sample.ltt))
        assert np.allclose(sample.ltt, sample.ltt[0, 0], rtol=1e-3)


def test_sparse_matches_dense():
    """The whole point: same channel, two evaluators, one answer."""
    orbit = LisaEqualArmOrbit()
    delta_t, n = 5.0, 120000
    times = np.arange(n) * delta_t + 1e5
    source = NewtonianChirp(3.0e4, times[-1] + 4e5)

    sample = sample_constellation(times, orbit)
    geometry = link_geometry(sample, 0.9, -0.25, velocity_order=1)
    dense = np.asarray(combine_links(
        link_response(source, sample, geometry), sample, channels="XYZ",
        generation=2, interpolation_order=31, delay_order=5)["X"])

    grid = adaptive_time_grid(source, 2, times[0], times[-1],
                              delta_phi=0.5, growth=1.15, dt_max=1e9,
                              max_step_scale=64)
    sparse = reconstruct(source, 2, grid,
                         sparse_channel(source, 2, grid, _terms(), orbit,
                                        0.9, -0.25), times)

    interior = slice(300, -300)
    scale = np.max(np.abs(dense[interior]))
    assert np.max(np.abs(sparse[interior] - dense[interior])) / scale < 1e-7
    assert len(grid) < len(times) / 100          # it is actually sparse


def test_sparse_grid_shrinks_and_error_grows_monotonically():
    """Guards the accuracy/speed knob: a looser grid must cost accuracy."""
    orbit = LisaEqualArmOrbit()
    delta_t, n = 5.0, 120000
    times = np.arange(n) * delta_t + 1e5
    source = NewtonianChirp(3.0e4, times[-1] + 4e5)
    sample = sample_constellation(times, orbit)
    dense = np.asarray(combine_links(
        link_response(source, sample,
                      link_geometry(sample, 0.9, -0.25, velocity_order=1)),
        sample, channels="XYZ", generation=2,
        interpolation_order=31, delay_order=5)["X"])
    interior = slice(300, -300)
    scale = np.max(np.abs(dense[interior]))

    sizes, errors = [], []
    for cap in (64, 512, 4096):
        grid = adaptive_time_grid(source, 2, times[0], times[-1],
                                  delta_phi=0.5, growth=1.15, dt_max=1e9,
                                  max_step_scale=cap)
        sparse = reconstruct(source, 2, grid,
                             sparse_channel(source, 2, grid, _terms(), orbit,
                                            0.9, -0.25), times)
        sizes.append(len(grid))
        errors.append(np.max(np.abs(sparse[interior] - dense[interior])) / scale)
    assert sizes[0] > sizes[1] > sizes[2]
    assert errors[0] < errors[1] < errors[2]


def test_cached_geometries_agree_with_the_direct_path():
    """The three cached forms are optimisations, not different calculations."""
    from pycbc.tdi.onthefly import (SparseGeometry, StackedGeometry,
                                    TermGeometry, sparse_channel_cached,
                                    sparse_channel_stacked,
                                    sparse_channel_terms)
    orbit = LisaEqualArmOrbit()
    times = np.arange(120000) * 5.0 + 1e5
    source = NewtonianChirp(3.0e4, times[-1] + 4e5)
    terms = _terms()
    grid = adaptive_time_grid(source, 2, times[0], times[-1],
                              delta_phi=0.5, growth=1.15, dt_max=1e9,
                              max_step_scale=512)
    direct = sparse_channel(source, 2, grid, terms, orbit, 0.9, -0.25)
    scale = np.max(np.abs(direct))
    for builder, evaluate, tolerance in (
            (SparseGeometry, sparse_channel_cached, 0.0),
            (StackedGeometry, sparse_channel_stacked, 1e-12),
            (TermGeometry, sparse_channel_terms, 1e-12)):
        got = evaluate(source, 2, builder(orbit, grid, terms), 0.9, -0.25)
        assert np.max(np.abs(got - direct)) / scale <= tolerance


def test_term_geometry_keeps_only_the_pairs_the_channel_uses():
    """The optimisation that mattered: gather (chain, link), do not index."""
    from pycbc.tdi.onthefly import StackedGeometry, TermGeometry
    orbit = LisaEqualArmOrbit()
    grid = np.linspace(1e5, 3e6, 64)
    terms = _terms()
    stacked = StackedGeometry(orbit, grid, terms)
    gathered = TermGeometry(orbit, grid, terms)
    assert gathered.shape == (len(terms), len(grid))
    # the stacked form carries every link of every chain
    assert stacked.n_chain * stacked.n_link > 3 * len(terms)
