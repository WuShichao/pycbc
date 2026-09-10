"""Tests for the sparse on-the-fly TDI evaluator."""

import numpy as np
import pytest

from pycbc.coordinates.space_orbit import LisaEqualArmOrbit
from pycbc.tdi.backends.pytdi_backend import (PyTDICombinationAdapter,
                                              combine_links,
                                              get_pytdi_combination)
from pycbc.tdi.onthefly import (adaptive_time_grid, chain_delay,
                                reconstruct, reconstruct_complex,
                                sparse_channel)
from pycbc.tdi.response import (link_geometry, link_response,
                                sample_constellation)
from pycbc.tdi.sources import (LALFDSource, LALIMRPhenomDSource,
                               NewtonianChirp)


def _lal_waveforms_available():
    """Is ``pycbc.waveform`` importable at all in this environment?

    The LAL-backed source needs it, and the plugin scan at import time will
    raise if any INSTALLED package registers a pycbc.waveform entry point
    whose module is absent from the checked-out PyCBC -- which is what a
    sibling feature branch's plugin looks like from here. That is an
    environment defect rather than a missing optional dependency, so the
    reason is reported instead of being swallowed.
    """
    try:
        import pycbc.waveform                                  # noqa: F401
    except Exception as error:                                 # pragma: no cover
        return f"pycbc.waveform is not importable here: {error!r}"
    return None


_NO_LAL = _lal_waveforms_available()


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
                              delta_phi=0.5, dt_max=1e9, growth=1.15,
                              max_step_scale=64)
    sparse = reconstruct(source, 2, grid,
                         sparse_channel(source, 2, grid, _terms(), orbit,
                                        0.9, -0.25), times)

    interior = slice(300, -300)
    scale = np.max(np.abs(dense[interior]))
    assert np.max(np.abs(sparse[interior] - dense[interior])) / scale < 1e-7
    assert len(grid) < len(times) / 100          # it is actually sparse


def test_sparse_grid_shrinks_and_error_grows_monotonically():
    """Guards the accuracy/speed knob: a looser grid must cost accuracy.

    delta_phi is now the only knob -- the geometric step growth it replaced put
    the dense region at the segment start rather than at the merger.
    """
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
                                  delta_phi=0.5, dt_max=1e9, growth=1.15,
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
                              delta_phi=0.5, dt_max=1e9, growth=1.15,
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


def test_real_reconstruction_is_the_real_part_of_analytic_signal():
    source = NewtonianChirp(3.0e4, 1e6)
    times = np.linspace(1e5, 2e5, 1000)
    grid = np.linspace(times[0], times[-1], 40)
    bracket = np.exp(1j * np.linspace(0.0, 0.2, len(grid)))
    analytic = reconstruct_complex(source, 2, grid, bracket, times)
    real = reconstruct(source, 2, grid, bracket, times)
    assert np.array_equal(real, np.real(analytic))


@pytest.mark.skipif(_NO_LAL is not None, reason=str(_NO_LAL))
def test_lal_imrphenomd_inverse_spa_reproduces_native_fd_waveform():
    """The harmonic adapter is LAL IMRPhenomD, not a surrogate waveform."""
    from pycbc.waveform import get_fd_waveform_sequence

    parameters = dict(
        mass1=35.0, mass2=32.0, spin1z=0.6, spin2z=0.2,
        distance=160.0, inclination=2.7, coa_phase=1.2, f_ref=0.0058,
    )
    source = LALIMRPhenomDSource(
        f_lower=0.0058, duration=3 * 86400.0, n_frequency=512,
        **parameters)
    frequency = source.sample_frequencies
    stationary = source.stationary_times
    relative_time = stationary - stationary[0]
    amp_plus, amp_cross = source.amplitude(2, relative_time)
    phase = source.carrier_phase(2, relative_time)

    # The source clock begins at zero, so compare to the correspondingly
    # time-shifted LAL frequency-domain waveform.
    exponent = phase - 2 * np.pi * frequency * relative_time - np.pi / 4
    scale = 0.5 * amp_plus * np.sqrt(source._dt_df)
    reconstructed = scale * np.exp(1j * exponent)
    native_plus, native_cross = get_fd_waveform_sequence(
        approximant="IMRPhenomD", sample_points=frequency, **parameters)
    time_shift = np.exp(1j * np.remainder(
        2 * np.pi * frequency * stationary[0], 2 * np.pi))
    ratio = reconstructed / (np.asarray(native_plus) * time_shift)

    assert np.max(np.abs(np.abs(ratio) - 1)) < 1e-10
    assert np.max(np.abs(np.angle(ratio))) < 1e-6
    cross_ratio = np.asarray(native_cross) / np.asarray(native_plus)
    assert np.max(np.abs(amp_cross / amp_plus - cross_ratio)) < 1e-12
    # The endpoint is itself obtained from a finite phase difference; at this
    # extremely narrow three-day band its numerical uncertainty is seconds.
    assert abs(source.t_end - 3 * 86400.0) < 60.0


@pytest.mark.skipif(_NO_LAL is not None, reason=str(_NO_LAL))
def test_lal_imrphenomd_sparse_response_matches_dense_native_response():
    """LAL waveform and native TDI agree before and after sparse factoring."""
    orbit = LisaEqualArmOrbit()
    delta_t, n = 5.0, 20000
    times = np.arange(n) * delta_t + 1e5
    source = LALIMRPhenomDSource(
        mass1=35.0, mass2=32.0, spin1z=0.6, spin2z=0.2,
        distance=160.0, inclination=2.7, coa_phase=1.2, f_ref=0.0058,
        f_lower=0.0058, duration=150000.0, t_start=90000.0,
        n_frequency=256)

    sample = sample_constellation(times, orbit)
    geometry = link_geometry(sample, 0.9, -0.25, velocity_order=1)
    dense = np.asarray(combine_links(
        link_response(source, sample, geometry), sample, channels="XYZ",
        generation=2, interpolation_order=31, delay_order=5)["X"])
    grid = adaptive_time_grid(
        source, 2, times[0], times[-1], delta_phi=0.5, dt_max=1e9,
        growth=1.15, max_step_scale=64)
    sparse = reconstruct(
        source, 2, grid,
        sparse_channel(source, 2, grid, _terms(), orbit, 0.9, -0.25),
        times, support=source.support(2))

    interior = slice(300, -300)
    scale = np.max(np.abs(dense[interior]))
    error = np.max(np.abs(sparse[interior] - dense[interior])) / scale
    assert error < 1e-6
    assert len(grid) < len(times) / 100


@pytest.mark.skipif(_NO_LAL is not None, reason=str(_NO_LAL))
def test_lal_source_generalises_beyond_imrphenomd():
    """The inverse-SPA construction is not tied to one approximant.

    Every dominant-mode frequency-domain model must round-trip back to its own
    native LAL waveform; the adapter must not quietly become a surrogate.
    ROM models are skipped when their data files are absent, which is an
    environment property rather than a code one.
    """
    from pycbc.waveform import get_fd_waveform_sequence

    parameters = dict(mass1=35.0, mass2=32.0, spin1z=0.6, spin2z=0.2,
                      distance=160.0, inclination=2.7, coa_phase=1.2,
                      f_ref=0.0058)
    checked = 0
    for approximant in ("IMRPhenomD", "IMRPhenomXAS", "TaylorF2"):
        try:
            source = LALFDSource(f_lower=0.0058, duration=3 * 86400.0,
                                 n_frequency=512, approximant=approximant,
                                 **parameters)
        except RuntimeError:                      # missing ROM data
            continue
        frequency = source.sample_frequencies
        relative_time = source.stationary_times - source.stationary_times[0]
        amp_plus, _ = source.amplitude(2, relative_time)
        exponent = (source.carrier_phase(2, relative_time)
                    - 2 * np.pi * frequency * relative_time - np.pi / 4)
        reconstructed = (0.5 * amp_plus * np.sqrt(source._dt_df)
                         * np.exp(1j * exponent))
        native, _ = get_fd_waveform_sequence(
            approximant=approximant, sample_points=frequency, **parameters)
        shift = np.exp(1j * np.remainder(
            2 * np.pi * frequency * source.stationary_times[0], 2 * np.pi))
        ratio = reconstructed / (np.asarray(native) * shift)
        assert np.max(np.abs(np.abs(ratio) - 1)) < 1e-9, approximant
        assert np.max(np.abs(np.angle(ratio))) < 1e-5, approximant
        checked += 1
    assert checked >= 2


@pytest.mark.skipif(_NO_LAL is not None, reason=str(_NO_LAL))
def test_lal_source_rejects_what_it_cannot_represent():
    """A model with no frequency sequence, and a fixed-approximant subclass."""
    with pytest.raises(ValueError, match="frequency-sequence"):
        LALFDSource(mass1=35.0, mass2=32.0, f_lower=0.0058,
                    duration=3 * 86400.0, approximant="NotAnApproximant")
    with pytest.raises(ValueError, match="sample_points"):
        LALFDSource(mass1=35.0, mass2=32.0, f_lower=0.0058,
                    duration=3 * 86400.0, sample_points=np.zeros(4))
    with pytest.raises(ValueError, match="fixes approximant"):
        LALIMRPhenomDSource(mass1=35.0, mass2=32.0, f_lower=0.0058,
                            duration=3 * 86400.0, approximant="TaylorF2")
    source = LALIMRPhenomDSource(mass1=35.0, mass2=32.0, f_lower=0.0058,
                                 duration=3 * 86400.0, n_frequency=256)
    assert source.approximant == "IMRPhenomD"
    with pytest.raises(ValueError, match="carrier"):
        source.amplitude(3, np.zeros(2))
