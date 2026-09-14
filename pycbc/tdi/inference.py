"""Expose the sparse time-domain TDI response to ``pycbc_inference``.

PyCBC already knows how to sample a likelihood from a waveform that carries
its own detector response: `pycbc.waveform.waveform.fd_det_sequence` is the
registry of such approximants, `BBHX_PhenomD` is in it, and
`pycbc.inference.models.relbin.Relative` switches to
``likelihood_parts_det`` and asks the generator for values at its own bin
edges. A TDI channel is exactly that kind of observable, so nothing in the
likelihood has to change -- the "detectors" are the channels.

What this module adds is the generator. It builds a source from the sampled
parameters, projects it through a *prepared* multiband geometry, and returns
the channels at the requested frequencies.

The preparation is the whole point. Building the geometry costs about half a
second and projecting a candidate through it costs about seven milliseconds,
so a likelihood that rebuilt it would be seventy times slower than it needs
to be. It is cached on everything that defines it -- the observation window,
the band edges, the channels, the combination -- and deliberately *not* on
sky position: `PreparedMultibandTDI.project` takes the sky, and reprojecting
a geometry prepared at one sky position onto another is both correct and
cheap (measured: 7.4 ms, and the response genuinely changes).

The coverage is fail-closed. A candidate whose band support leaves what the
preparation covers is refused rather than extrapolated, and since a chirp-mass
change of one part in 1e5 moves t(f) by weeks, the prior's own boundary has to
be declared through ``tdi_coverage_sources``.
"""
import logging

import numpy as np

from pycbc.coordinates.space_orbit import LisaEqualArmOrbit
from pycbc.tdi.backends.pytdi_backend import (PyTDICombinationAdapter,
                                              get_pytdi_combination)
from pycbc.tdi.combination import Term
from pycbc.tdi.multiband import prepare_multiband_tdi
from pycbc.types import Array

#: Prepared geometries, keyed by everything that defines one. Sky position is
#: not part of the key; see the module docstring.
_PREPARED = {}

#: Source builders, selected by the ``tdi_source`` parameter.
_SOURCES = {}

#: Recently built sources, keyed by builder and parameter values. Preparing
#: one geometry per harmonic asks the builder for the same fiducial once per
#: harmonic, and for a precessing pyEFPEHM configuration each construction
#: integrates the orbit and builds precession interpolants -- thirty of them
#: for ten harmonics with a two-ended coverage boundary, which is both slow
#: and enough allocation to exhaust a 3.5 GiB budget.
_SOURCE_CACHE = {}
_SOURCE_CACHE_LIMIT = 32

DEFAULT_ARM = 2.5e9
DEFAULT_CHANNELS = ('A', 'E')

#: A and E as term lists rather than a post-hoc combination of X, Y and Z.
#: The combination is linear in the channels and hence in the terms, so
#: scaling coefficients gives the orthogonal channels natively -- which a
#: multiband response has to have in order to answer a request for "A".
_ORTHOGONAL = {'A': {'Z': 2 ** -0.5, 'X': -(2 ** -0.5)},
               'E': {'X': 6 ** -0.5, 'Y': -2 * 6 ** -0.5, 'Z': 6 ** -0.5},
               'T': {'X': 3 ** -0.5, 'Y': 3 ** -0.5, 'Z': 3 ** -0.5}}


def channel_terms(channels=DEFAULT_CHANNELS, generation=2, delta_t=5.0):
    """Term lists for the requested channels, Michelson or orthogonal."""
    michelson = {}
    for label in 'XYZ':
        name = f'{label}{generation}'
        michelson[label] = PyTDICombinationAdapter(
            name, get_pytdi_combination(name), delta_t=delta_t).terms()
    terms = {}
    for channel in channels:
        if channel in michelson:
            terms[channel] = michelson[channel]
        elif channel in _ORTHOGONAL:
            terms[channel] = tuple(
                Term(term.link, term.coefficient * weight, term.operators)
                for label, weight in _ORTHOGONAL[channel].items()
                for term in michelson[label])
        else:
            raise ValueError(f"unknown TDI channel {channel!r}")
    return terms


def register_source(name, builder):
    """Make a source class available through the ``tdi_source`` parameter.

    ``builder`` takes the sampled parameters as keyword arguments and returns
    an object satisfying the harmonic-source protocol.
    """
    _SOURCES[str(name)] = builder


def _harmonic_band_edges(source, harmonic, edges, t_start, t_end,
                         probe=256, margin=1.05):
    """Band edges clipped to where this harmonic actually lives.

    A band's time sampling is set by its *upper* edge -- `samples_per_cycle`
    over `f_upper` -- so a harmonic that occupies 5 to 10 mHz pays for the
    90 mHz top of a global band it never reaches. Every candidate then
    materialises several times more samples than its own carrier needs, and
    materialising the block is half the cost of a frequency evaluation.

    Clipping is safe because the band only has to contain the harmonic: the
    partition exists to bound the sampling rate, not to filter.
    """
    times = np.linspace(float(t_start), float(t_end), int(probe))
    rate = np.abs(np.asarray(source.angular_frequency(harmonic, times),
                             dtype=float)) / (2 * np.pi)
    rate = rate[np.isfinite(rate) & (rate > 0)]
    if not len(rate):
        return list(edges)
    low, high = float(rate.min()) / margin, float(rate.max()) * margin
    inner = [float(value) for value in edges if low < value < high]
    return [low] + inner + [high]


def _preparation(params, terms, orbit):
    """Fetch or build the prepared geometry for this observation."""
    channels = tuple(terms)
    edges = tuple(float(value) for value in np.atleast_1d(
        params['tdi_band_edges']))
    bounds = params.get('tdi_coverage_bounds') or {}
    key = (params['tdi_source'], channels, edges,
           params.get('tdi_harmonic'),
           tuple(sorted((name, float(low), float(high))
                        for name, (low, high) in bounds.items())),
           float(params['t_obs_start']), float(params['t_obs_end']),
           float(params.get('tdi_arm_length', DEFAULT_ARM)),
           int(params.get('tdi_generation', 2)))
    if key not in _PREPARED:
        fiducial = _narrow(_build_source(params), params)
        coverage = [_narrow(source, params)
                    for source in _coverage_sources(params)]
        # The live harmonic set is a function of the parameters: pyEFPEHM
        # keeps a harmonic while it carries more than `Amplitude_tol` of the
        # total, so a more eccentric candidate brings harmonics the fiducial
        # never had and loses others. `project` refuses an unprepared harmonic
        # rather than dropping it silently, and `prepare_multiband_tdi`
        # partitions every coverage source against the primary's set, so
        # neither the union nor the richest single source works -- only a set
        # every source in play actually contains.
        #
        # So the sampling epoch gets one fixed set, the intersection over the
        # fiducial and the prior corners, and what that costs is reported
        # rather than absorbed. Widening it is a modelling choice: lower
        # `Amplitude_tol` until the set stops moving across the prior.
        common = set(fiducial.harmonics)
        for source in coverage:
            common &= set(source.harmonics)
        if not common:
            raise ValueError(
                "the fiducial and the prior corners share no harmonic; "
                "lower Amplitude_tol or narrow tdi_coverage_bounds")
        dropped = set(fiducial.harmonics) - common
        if dropped:
            logging.info(
                "tdi: %d harmonic(s) live in the fiducial but not across the "
                "whole prior and are excluded from this epoch: %s",
                len(dropped), sorted(dropped))
        # Clip whenever the preparation covers exactly one harmonic, however
        # it came to -- a dominant-mode source qualifies without anyone asking
        # for `tdi_harmonic`. With several harmonics in one preparation there
        # is no single frequency range to clip to.
        band = list(edges)
        if len(fiducial.harmonics) == 1 and params.get('tdi_clip_bands', True):
            band = _harmonic_band_edges(
                fiducial, fiducial.harmonics[0], edges,
                params['t_obs_start'], params['t_obs_end'])
        _PREPARED[key] = prepare_multiband_tdi(
            fiducial, orbit, terms,
            float(params['eclipticlongitude']),
            float(params['eclipticlatitude']),
            harmonics=tuple(sorted(common)),
            band_edges=band,
            t_start=float(params['t_obs_start']),
            t_end=float(params['t_obs_end']),
            coverage_sources=[RestrictedHarmonicSource(
                source, tuple(sorted(common))) for source in coverage])
    return _PREPARED[key]


def _build_source(params):
    """Build a source, reusing an identical one if it is still cached."""
    builder = params['tdi_source']
    try:
        key = (builder, tuple(sorted(
            (name, value) for name, value in params.items()
            if isinstance(value, (int, float, str, bool)))))
    except TypeError:
        return _SOURCES[builder](**params)
    if key not in _SOURCE_CACHE:
        if len(_SOURCE_CACHE) >= _SOURCE_CACHE_LIMIT:
            _SOURCE_CACHE.pop(next(iter(_SOURCE_CACHE)))
        _SOURCE_CACHE[key] = _SOURCES[builder](**params)
    return _SOURCE_CACHE[key]


def _narrow(source, params):
    """Restrict a source to one harmonic when ``tdi_harmonic`` asks for it.

    A geometry prepared for the whole harmonic set refuses a source that
    carries only one of them -- `project` re-partitions against the harmonics
    the preparation knows about. So a single-harmonic request gets its own
    preparation, which is also the better partition: each harmonic occupies
    its own frequency range and deserves its own band edges. Measured on a
    ten-harmonic eccentric, precessing source, preparing all ten separately
    costs 0.88 s once and projecting them one at a time costs 27.6 ms against
    26.0 ms for the summed projection -- the work is the same either way.
    """
    harmonic = params.get('tdi_harmonic')
    if harmonic is None:
        return source
    return SingleHarmonicSource(source, tuple(harmonic)
                                if isinstance(harmonic, (list, tuple))
                                else harmonic)


def _coverage_sources(params):
    """Sources at the prior's own edges, so the preparation covers them.

    `PreparedMultibandTDI.project` refuses a candidate whose band support
    leaves the prepared coverage rather than extrapolating into it, and a
    chirp-mass change of one part in 1e5 moves t(f) by weeks. Declaring the
    boundary as parameter ranges rather than as pre-built source objects keeps
    it expressible in a configuration file: ``tdi_coverage_bounds`` maps a
    parameter to its (low, high), and each end is built with everything else
    held at its fiducial value. Varying one at a time is enough while t(f) is
    monotone in each of them, which it is for masses and start frequency; a
    parameter that fails that has to be given as an explicit corner.
    """
    bounds = params.get('tdi_coverage_bounds') or {}
    sources = []
    for name, (low, high) in bounds.items():
        for value in (float(low), float(high)):
            edge = dict(params)
            edge[name] = value
            sources.append(_build_source(edge))
    return sources


def clear_cache():
    """Drop every prepared geometry and cached source."""
    _PREPARED.clear()
    _SOURCE_CACHE.clear()


def sparse_tdi_fd_det_sequence(**params):
    """TDI channels at requested frequencies, for ``fd_det_sequence``.

    Returns
    -------
    dict
        Channel name to `FrequencySeries` evaluated at ``sample_points``.
    """
    requested = params['ifos']
    if isinstance(requested, str):
        requested = (requested,)
    requested = tuple(requested)
    sample_points = np.asarray(params['sample_points'], dtype=float)

    orbit = LisaEqualArmOrbit(
        armlength=float(params.get('tdi_arm_length', DEFAULT_ARM)), t0=0.0)
    terms = channel_terms(
        requested, generation=int(params.get('tdi_generation', 2)),
        delta_t=float(params.get('tdi_delta_t', 5.0)))
    try:
        prepared = _preparation(params, terms, orbit)
    except ValueError as error:
        # A harmonic whose whole frequency range sits outside the analysis
        # band has no live band to prepare, and it contributes nothing. That
        # is a fact about the request, not an ambiguity, so it returns zeros
        # rather than refusing -- unlike an unprepared *harmonic*, where
        # keeping or dropping it would change the answer. An eccentric source
        # reaches this as soon as (n/2) f22 passes the cadence's Nyquist: at
        # e = 0.5 pyEFPEHM keeps 22 harmonics and the highest are past 0.1 Hz.
        if 'at least one live band' not in str(error):
            raise
        logging.info("tdi: harmonic %s has no support in the requested band; "
                     "returning zeros", params.get('tdi_harmonic'))
        return {name: Array(np.zeros(len(sample_points), dtype=complex))
                for name in requested}

    source = _narrow(_build_source(params), params)
    # The candidate's own harmonic set moves with its parameters too, so it
    # is held to the set the epoch was prepared for: an extra harmonic would
    # be refused by `project`, and silently keeping it would make one
    # candidate's likelihood incomparable with another's.
    wanted = tuple(h for h in prepared.harmonics if h in set(source.harmonics))
    if len(wanted) != len(prepared.harmonics):
        missing = sorted(set(prepared.harmonics) - set(wanted))
        raise ValueError(
            f"candidate is missing prepared harmonics {missing}; the epoch's "
            "common set was computed from a prior boundary this candidate "
            "falls outside")
    if set(wanted) != set(source.harmonics):
        source = RestrictedHarmonicSource(source, wanted)
    response = prepared.project(source,
                                float(params['eclipticlongitude']),
                                float(params['eclipticlatitude']))
    delta_f = float(params['tdi_delta_f'])
    # Absolute zero on the mission clock, not the start of the observation.
    # `Relative` multiplies the data by exp(-2 pi i f * end_time), and for a
    # series whose epoch is the observation start that is a shift by
    # t_obs_start plus one whole duration -- and a shift by exactly the
    # duration is the identity on the FFT grid. So the data it compares
    # against has its origin at zero, and the generator has to match. Getting
    # this wrong is silent: every candidate comes back at -rho^2/2 because
    # <d|h> has gone to zero. Measured on Yorsh sobhb1, loglr -19.18 with the
    # observation start against +18.97 with zero, where the exact peak is
    # +18.99.
    epoch = float(params.get('tdi_epoch', 0.0))
    samples = response.frequency_samples(
        {name: sample_points for name in requested},
        delta_f={name: delta_f for name in requested},
        epoch={name: epoch for name in requested},
        channels=requested)
    # A pycbc Array, not a FrequencySeries. `Relative.__init__` reverses the
    # fiducial with ``curr_wav[::-1]`` to find trailing zeros, and reversing a
    # FrequencySeries hands its constructor a negative delta_f; it also
    # resizes and rolls it. The non-det branch feeds it Arrays from
    # `get_fd_waveform_sequence`, and the det branch has to match, because
    # ``sample_points`` is an arbitrary set of frequencies rather than a grid.
    return {name: Array(np.asarray(samples[name], dtype=np.complex128))
            for name in requested}


#: `get_fd_det_waveform_sequence` checks these before dispatching.
sparse_tdi_fd_det_sequence.required = (
    'approximant', 'tdi_source', 'tdi_band_edges', 'tdi_delta_f',
    't_obs_start', 't_obs_end', 'eclipticlongitude', 'eclipticlatitude')


def register(approximant='TDISparse', force=False):
    """Add the generator to PyCBC's detector-response sequence registry."""
    from pycbc.waveform.plugin import add_custom_waveform
    add_custom_waveform(approximant, sparse_tdi_fd_det_sequence, 'frequency',
                        sequence=True, has_det_response=True, force=force)
    return approximant


def _lal_dominant(**params):
    """A dominant-mode LAL approximant through the inverse-SPA adapter."""
    from pycbc.tdi.sources import LALFDSource
    return LALFDSource(
        mass1=float(params['mass1']), mass2=float(params['mass2']),
        f_lower=float(params['f_lower']),
        f_upper=float(params['tdi_f_upper']),
        t_start=float(params.get('tdi_t_start', 0.0)),
        polarization=float(params.get('polarization', 0.0)),
        approximant=str(params.get('tdi_approximant', 'IMRPhenomD')),
        spin1z=float(params.get('spin1z', 0.0)),
        spin2z=float(params.get('spin2z', 0.0)),
        distance=float(params['distance']),
        inclination=float(params.get('inclination', 0.0)),
        coa_phase=float(params.get('coa_phase', 0.0)))


def _pyefpehm(**params):
    """pyEFPEHM's co-precessing harmonics, time-shifted onto the mission clock.

    Every harmonic is kept separate here. Summing them before the likelihood
    sees them is what makes plain relative binning fail on an eccentric or
    precessing source: at one frequency several harmonics contribute from
    different times, and their ratio to a fiducial carries the beat between
    them rather than a smooth trend. `pycbc.tdi.relative` has to bin each
    harmonic against its own fiducial and add the cross terms back.
    """
    from pycbc.tdi.sources import PyEFPEHMSource, TimeShiftedHarmonicSource
    keys = ('mass1', 'mass2', 'spin1x', 'spin1y', 'spin1z',
            'spin2x', 'spin2y', 'spin2z', 'distance', 'inclination',
            'eccentricity', 'phase', 'f22_start', 'f22_ref', 'f22_end',
            'Amplitude_tol')
    arguments = {key: params[key] for key in keys if key in params}
    arguments.setdefault('f22_start', params.get('f_lower'))
    arguments.setdefault('f22_ref', arguments.get('f22_start'))
    native = PyEFPEHMSource(arguments)
    return TimeShiftedHarmonicSource(native, offset=native.t_start)


register_source('lal', _lal_dominant)
register_source('pyefpehm', _pyefpehm)


class RestrictedHarmonicSource:
    """A harmonic source narrowed to a subset of its harmonics.

    Two places need this. Relative binning needs each harmonic on its own,
    and `prepare_multiband_tdi` requires every coverage source to be a subset
    of the prepared harmonic set -- a coverage source exists only to widen the
    prepared *time* coverage, so its extra harmonics are beside the point and
    narrowing it is the honest way to say so.
    """

    def __init__(self, source, harmonics):
        self.source = source
        self.harmonics = tuple(harmonics)
        self.t_start = getattr(source, 't_start', None)
        self.t_end = getattr(source, 't_end', None)

    def __getattr__(self, name):
        # Only reached for attributes this class does not define, so the
        # narrowed `harmonics` is never shadowed by the wrapped source's.
        return getattr(self.source, name)


class SingleHarmonicSource(RestrictedHarmonicSource):
    """One harmonic of a harmonic source, presented as a source in its own right.

    Relative binning needs each harmonic separately. A multi-harmonic signal
    has several harmonics contributing at one frequency from different times,
    so the ratio of the summed waveform to a summed fiducial carries the beat
    between them and is not smooth across a bin -- which is the one thing the
    approximation requires. Binned against its own fiducial, a single harmonic
    is smooth again.

    This wrapper delegates everything and narrows ``harmonics`` to one label,
    so the ordinary response machinery projects that harmonic alone.
    """

    def __init__(self, source, harmonic):
        super().__init__(source, (harmonic,))


def harmonic_labels(params):
    """The harmonic labels available across this epoch's whole prior.

    The intersection, not the fiducial's own set, because that is what the
    preparation uses -- see `_preparation`. A model that binned a harmonic the
    preparation cannot project would fail on the first candidate that needs
    it.
    """
    common = set(_build_source(params).harmonics)
    for source in _coverage_sources(params):
        common &= set(source.harmonics)
    return tuple(sorted(common))
