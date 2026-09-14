"""The sparse TDI response as a pycbc_inference detector-response waveform."""
import numpy as np
import pytest

from pycbc.tdi.combination import Term
from pycbc.tdi.inference import (channel_terms, clear_cache, register,
                                 sparse_tdi_fd_det_sequence)


def test_orthogonal_channels_are_term_lists_not_a_later_combination():
    """A response asked for "A" has to be able to answer.

    A = (Z - X)/sqrt2 is linear in the channels and hence in the terms, so
    scaling coefficients gives it natively. Building X, Y and Z and combining
    afterwards leaves a multiband response unable to produce A at all, which
    is what `Relative` asks it for.
    """
    michelson = channel_terms(('X', 'Y', 'Z'))
    orthogonal = channel_terms(('A', 'E', 'T'))
    assert len(orthogonal['A']) == len(michelson['X']) + len(michelson['Z'])
    assert len(orthogonal['E']) == sum(
        len(michelson[label]) for label in 'XYZ')

    # Every term keeps its link and operator chain; only the weight moves.
    scale = 2.0 ** -0.5
    by_chain = {}
    for term in orthogonal['A']:
        by_chain.setdefault((term.link, term.operators), 0.0)
        by_chain[(term.link, term.operators)] += term.coefficient
    for label, weight in (('Z', scale), ('X', -scale)):
        for term in michelson[label]:
            assert (term.link, term.operators) in by_chain
    total = sum(abs(term.coefficient) for term in orthogonal['A'])
    expected = scale * sum(
        abs(term.coefficient)
        for label in 'XZ' for term in michelson[label])
    assert total == pytest.approx(expected, rel=1e-12)


def test_an_unknown_channel_is_refused():
    with pytest.raises(ValueError, match="unknown TDI channel"):
        channel_terms(('Q',))


def test_registration_lands_in_the_detector_response_registry():
    """`Relative` switches on membership of `fd_det_sequence`, nothing else."""
    from pycbc.waveform.waveform import fd_det_sequence
    name = register(approximant='TDISparseTest', force=True)
    assert fd_det_sequence[name] is sparse_tdi_fd_det_sequence
    assert 'tdi_source' in sparse_tdi_fd_det_sequence.required


def test_the_prepared_geometry_is_cached_but_not_on_sky_position():
    """The cache key decides whether a sampler can afford this.

    Preparing costs about half a second and projecting about seven
    milliseconds, so a key that included the sky would make every likelihood
    call pay the preparation. `PreparedMultibandTDI.project` takes the sky, so
    it does not have to.
    """
    from pycbc.tdi import inference
    clear_cache()
    assert not inference._PREPARED
    params = dict(tdi_source='lal', tdi_band_edges=[1e-3, 2e-3],
                  t_obs_start=0.0, t_obs_end=1.0e7,
                  eclipticlongitude=0.9, eclipticlatitude=-0.25)
    keys = []
    for lamb in (0.9, 2.5):
        edges = tuple(float(v) for v in np.atleast_1d(params['tdi_band_edges']))
        bounds = params.get('tdi_coverage_bounds') or {}
        keys.append((params['tdi_source'], ('A', 'E'), edges,
                     tuple(sorted((n, float(a), float(b))
                                  for n, (a, b) in bounds.items())),
                     float(params['t_obs_start']),
                     float(params['t_obs_end']), 2.5e9, 2))
    assert keys[0] == keys[1], "sky position must not enter the cache key"


def test_band_edges_are_clipped_to_a_single_harmonic():
    """A band's sampling rate comes from its upper edge, so it must fit.

    `samples_per_cycle` over `f_upper` sets the block length, so a harmonic
    living at 5 to 10 mHz inside a band that reaches 90 mHz materialises an
    order of magnitude more samples than its own carrier needs -- and
    materialising the block is half the cost of a frequency evaluation.
    """
    from pycbc.tdi.inference import _harmonic_band_edges

    class Narrow:
        harmonics = (2,)

        def angular_frequency(self, harmonic, times):
            times = np.asarray(times, dtype=float)
            return 2 * np.pi * (5e-3 + 5e-3 * (times - times[0])
                                / max(times[-1] - times[0], 1.0))

    edges = [4e-3, 2.55e-2, 4.7e-2, 6.85e-2, 9e-2]
    clipped = _harmonic_band_edges(Narrow(), 2, edges, 0.0, 1.0e6)
    assert clipped[0] < 5e-3 and clipped[-1] > 1e-2
    assert clipped[-1] < 1.1e-2, clipped
    # None of the global interior edges lie inside 5-10 mHz, so the clipped
    # band is a single interval and its upper edge is the harmonic's own.
    assert len(clipped) == 2, clipped


def test_the_chirp_z_plan_is_reused():
    """`zoom_fft` rebuilds its plan every call; a likelihood asks repeatedly."""
    from pycbc.tdi.multiband import (_ZOOM_PLANS, _zoom_plan,
                                     clear_zoom_plans)
    clear_zoom_plans()
    first = _zoom_plan(1024, 1e-3, 2e-3, 64, 0.2)
    again = _zoom_plan(1024, 1e-3, 2e-3, 64, 0.2)
    assert again is first
    other = _zoom_plan(2048, 1e-3, 2e-3, 64, 0.2)
    assert other is not first
    assert len(_ZOOM_PLANS) == 2
    values = np.random.default_rng(0).normal(size=1024)
    from scipy.signal import zoom_fft
    assert np.allclose(first(values),
                       zoom_fft(values, [1e-3, 2e-3], m=64, fs=0.2,
                                endpoint=True))
    clear_zoom_plans()
    assert not _ZOOM_PLANS
