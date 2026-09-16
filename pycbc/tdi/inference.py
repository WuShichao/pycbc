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
from pycbc.tdi.multiband import (prepare_sparse_tdi,
                                 raised_cosine_time_window)
from pycbc.types import Array

#: Prepared geometries, keyed by everything that defines one. Sky position is
#: not part of the key; see the module docstring.
_PREPARED = {}

#: Source builders, selected by the ``tdi_source`` parameter.
_SOURCES = {}

#: Frequency samples, keyed by preparation, parameters and the requested
#: grid. `Relative` asks one channel at a time and each ask re-materialises
#: the band's time series, so a two-channel likelihood transformed everything
#: twice; `frequency_samples` takes all the channels at once.
_SAMPLED = {}
_SAMPLED_LIMIT = 16

#: The last projected response, keyed by preparation and parameters. PyCBC's
#: `Relative` asks the generator for one channel at a time, so a two-channel
#: likelihood over ten harmonics would otherwise project twenty times for ten
#: distinct responses. Projection is the larger half of an evaluation.
_PROJECTED = {}
_PROJECTED_LIMIT = 16

#: Recently built sources, keyed by builder and parameter values. Preparing
#: one geometry per harmonic asks the builder for the same fiducial once per
#: harmonic, and for a precessing pyEFPEHM configuration each construction
#: integrates the orbit and builds precession interpolants -- thirty of them
#: for ten harmonics with a two-ended coverage boundary, which is both slow
#: and enough allocation to exhaust a 3.5 GiB budget.
_SOURCE_CACHE = {}
_SOURCE_CACHE_LIMIT = 16

#: One live candidate signature per inference-model epoch. A source or
#: projected response can own large spline arrays, and floating-point sampler
#: proposals are almost never revisited. Retaining the last 32 or 64 proposals
#: therefore grew memory without helping PE. Keep only the current proposal
#: for each active model while still sharing it across harmonics and channels.
_RUNTIME_SIGNATURES = {}
_RUNTIME_EPOCH_LIMIT = 4

#: The expensive pyEFPEHM evolution below the cheap detector-frame wrappers.
#: One intrinsic state per active inference epoch is enough: an extrinsic-only
#: proposal may reuse it, while a new intrinsic proposal replaces it.  Keeping
#: this separate from ``_SOURCE_CACHE`` matters because ``tc`` and
#: ``polarization`` change the wrappers but not the evolution they wrap.
_PYEFPEHM_NATIVE = {}

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


def _label(harmonic):
    """The hashable form of a harmonic label, or ``None``.

    Configuration files hand a harmonic over as a list, the sources key
    theirs with tuples, and a scalar label stays a scalar. Three call sites
    normalised it in three places; they now share this one.
    """
    if harmonic is None:
        return None
    return (tuple(harmonic) if isinstance(harmonic, (list, tuple))
            else harmonic)


def _tolerance(params):
    """The bracket-interpolation tolerance the prepared geometry is built to.

    ``None`` leaves the uniform grid alone. It belongs in the preparation
    cache key as well as the call: a looser tolerance that silently reused a
    tighter epoch's geometry would report the accuracy of a grid it did not
    ask for.
    """
    value = params.get('tdi_relative_tolerance', 1e-4)
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            'tdi_relative_tolerance must be positive and finite, or None')
    return value


def _preparation(params, terms, orbit):
    """Fetch or build the prepared geometry for this observation."""
    channels = tuple(terms)
    edges = tuple(float(value) for value in np.atleast_1d(
        params['tdi_band_edges']))
    bounds = params.get('tdi_coverage_bounds') or {}
    key = (params.get('tdi_preparation_id', 'default'),
           params['tdi_source'], channels, edges,
           _label(params.get('tdi_harmonic')),
           tuple(sorted((name, float(low), float(high))
                        for name, (low, high) in bounds.items())),
           tuple(tuple(sorted(dict(corner).items()))
                 for corner in params.get('tdi_coverage_corners') or ()),
           float(params['t_obs_start']), float(params['t_obs_end']),
           float(params.get('tdi_arm_length', DEFAULT_ARM)),
           int(params.get('tdi_generation', 2)),
           float(params.get('tdi_delta_t', 5.0)),
           float(params.get('tdi_samples_per_cycle', 4.0)),
           float(params.get('tdi_geometry_step', 86400.0)),
           str(params.get('tdi_harmonic_combine', 'intersection')),
           _tolerance(params),
           params.get('tdi_coverage_tolerance'),
           int(params.get('tdi_minimum_grid_points', 16)),
           float(params.get('tdi_band_overlap', 0.0)),
           bool(params.get('tdi_clip_bands', True)))
    if key not in _PREPARED:
        # The live harmonic set is a function of the parameters: pyEFPEHM
        # keeps a harmonic while it carries more than `Amplitude_tol` of the
        # total, so a more eccentric candidate brings harmonics the fiducial
        # never had and loses others.
        #
        # `'intersection'` gives the epoch one fixed set -- the harmonics
        # every source in play contains -- and reports what that costs rather
        # than absorbing it. `'union'` bins everything the prior carries, and
        # then the source a harmonic is prepared from cannot be the fiducial,
        # which does not have it. `harmonic_references` names the prior
        # corner that stands in.
        combine = str(params.get('tdi_harmonic_combine', 'intersection'))
        label = _label(params.get('tdi_harmonic'))
        native = [(source, set(source.harmonics))
                  for source in _coverage_sources(params)]
        prepare_params = params
        if label is not None and label not in set(
                _build_source(params).harmonics):
            overrides = harmonic_references(params).get(label)
            if overrides is None:
                raise ValueError(
                    f"harmonic {label!r} is carried by neither the fiducial "
                    "nor any prior corner")
            prepare_params = dict(params, **overrides)
        fiducial = _narrow(_build_source(prepare_params), prepare_params)
        common = set(fiducial.harmonics)
        for _, other in native:
            common = common | other if combine == 'union' else common & other
        if label is not None:
            common &= {label}
        elif combine == 'union' and common - set(fiducial.harmonics):
            # One preparation is built from one source, and that source
            # cannot supply a harmonic it does not carry. Asking anyway used
            # to intersect the union back to the fiducial in silence.
            raise ValueError(
                "tdi_harmonic_combine='union' needs the per-harmonic path: "
                f"{sorted(common - set(fiducial.harmonics), key=repr)!r} "
                "live in the prior but not in the fiducial. Ask for one "
                "harmonic at a time with tdi_harmonic, as the harmonic "
                "relative-binning model does.")
        if not common:
            raise ValueError(
                "the fiducial and the prior corners share no harmonic; "
                "lower Amplitude_tol or narrow tdi_coverage_bounds")
        # Only a source that actually carries a harmonic can widen its
        # coverage or train its grid. Narrowing one that does not would have
        # it claim a support it has no samples over.
        coverage = [RestrictedHarmonicSource(
                        source, tuple(sorted(common & other)))
                    for source, other in native if common & other]
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
        _PREPARED[key] = prepare_sparse_tdi(
            fiducial, orbit, terms,
            float(params['eclipticlongitude']),
            float(params['eclipticlatitude']),
            harmonics=tuple(sorted(common)),
            band_edges=band,
            t_start=float(params['t_obs_start']),
            t_end=float(params['t_obs_end']),
            samples_per_cycle=float(params.get('tdi_samples_per_cycle', 4.0)),
            geometry_step=float(params.get('tdi_geometry_step', 86400.0)),
            relative_tolerance=_tolerance(params),
            coverage_tolerance=params.get('tdi_coverage_tolerance'),
            minimum_grid_points=int(params.get(
                'tdi_minimum_grid_points', 16)),
            overlap=float(params.get('tdi_band_overlap', 0.0)),
            coverage_sources=coverage)
    return _PREPARED[key]


def _build_source(params):
    """Build a source, reusing an identical one if it is still cached."""
    builder = params['tdi_source']
    try:
        key = (params.get('tdi_preparation_id', 'default'),
               builder, _signature(params))
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
    harmonic = _label(params.get('tdi_harmonic'))
    if harmonic is None:
        return source
    return SingleHarmonicSource(source, harmonic)


def _coverage_overrides(params):
    """Parameter overrides that place a source at the prior's own edges.

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
    overrides = []
    for name, (low, high) in bounds.items():
        for value in (float(low), float(high)):
            overrides.append({name: value})
    # Axial corners move one parameter and hold the rest, so they cannot
    # reach a corner of a multi-parameter tile -- and a harmonic set is not
    # a monotone function of one parameter at a time. `tdi_coverage_corners`
    # takes explicit joint corners, each a mapping of parameter to value,
    # for the case the docstring above names.
    for corner in params.get('tdi_coverage_corners') or ():
        overrides.append(dict(corner))
    return overrides


def _coverage_sources(params):
    """Build a source at each edge of the prior."""
    return [_build_source(dict(params, **item))
            for item in _coverage_overrides(params)]


def harmonic_references(params):
    """The source each harmonic is binned against, as parameter overrides.

    Relative binning divides by its reference, so a harmonic whose reference
    is identically zero has no bins at all -- it is listed, prepared as
    zeros, and then skipped for having an empty fiducial, which is silent
    truncation wearing a union's clothes. Under ``'union'`` the fiducial is
    by construction missing some of the harmonics the prior carries, so each
    of those is given the *first* prior corner that does carry it. An empty
    mapping means the fiducial itself.

    First, not cheapest or nearest: the order of `_coverage_overrides` is the
    order the configuration declares, so the choice is reproducible from the
    file rather than from whatever the search happened to visit.
    """
    references = {label: {} for label in _build_source(params).harmonics}
    if str(params.get('tdi_harmonic_combine', 'intersection')) != 'union':
        return references
    for overrides in _coverage_overrides(params):
        source = _build_source(dict(params, **overrides))
        for label in source.harmonics:
            references.setdefault(label, dict(overrides))
    return references


def _projection(params, prepared):
    """Project this candidate, reusing the last projection of the same one.

    Keyed on the parameters rather than on time, so asking for a second
    channel of a candidate already projected costs nothing while a genuinely
    new candidate is never served a stale response.
    """
    key = (params.get('tdi_preparation_id', 'default'),
           id(prepared), _signature(params))
    response = _PROJECTED.get(key)
    if response is not None:
        return response
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
    if len(_PROJECTED) >= _PROJECTED_LIMIT:
        _PROJECTED.pop(next(iter(_PROJECTED)))
    _PROJECTED[key] = response
    return response


def _signature(params):
    """Hashable summary of the scalar parameters a source is built from."""
    return tuple(sorted(
        (name, value) for name, value in params.items()
        if isinstance(value, (int, float, str, bool))
        and name not in ('ifos', 'tdi_harmonic')))


def _begin_candidate(params):
    """Discard stale runtime objects when an inference candidate changes."""
    epoch = params.get('tdi_preparation_id', 'default')
    signature = _signature(params)
    previous = _RUNTIME_SIGNATURES.get(epoch)
    if previous == signature:
        return
    if previous is None and len(_RUNTIME_SIGNATURES) >= _RUNTIME_EPOCH_LIMIT:
        expired = next(iter(_RUNTIME_SIGNATURES))
        _RUNTIME_SIGNATURES.pop(expired)
        for cache in (_SOURCE_CACHE, _PROJECTED, _SAMPLED):
            for key in tuple(cache):
                if key[0] == expired:
                    cache.pop(key)
    if previous is None:
        _RUNTIME_SIGNATURES[epoch] = signature
        return
    for cache in (_SOURCE_CACHE, _PROJECTED, _SAMPLED):
        for key in tuple(cache):
            if key[0] == epoch:
                cache.pop(key)
    _RUNTIME_SIGNATURES[epoch] = signature


def _analysis_time_window(params):
    """Return the optional raised-cosine observation-edge taper.

    A finite sampled data segment and a continuous pruned transform are the
    same Fourier object only when aliases from the segment boundary are
    negligible.  A nonzero ``tdi_taper_duration`` applies a sine-squared
    fade at both observation edges.  The data must have been conditioned with
    the identical window; this option is therefore off by default.

    ``tdi_taper_start`` and ``tdi_taper_end`` override it one edge at a time.
    A taper costs nothing only where the signal is already quiet, and the two
    edges are rarely alike: a stellar-origin binary is still chirping at
    ``t_obs_end`` and needs the trailing fade, while a massive binary that
    merges inside the segment must not have one -- the fade would drive its
    merger to zero, which is most of its signal-to-noise.
    """
    duration = float(params.get('tdi_taper_duration', 0.0))
    lead = float(params.get('tdi_taper_start', duration))
    trail = float(params.get('tdi_taper_end', duration))
    for name, value in (('tdi_taper_duration', duration),
                        ('tdi_taper_start', lead), ('tdi_taper_end', trail)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and non-negative')
    if lead == 0 and trail == 0:
        return None
    start = float(params['t_obs_start'])
    stop = float(params['t_obs_end'])
    # Validate before returning a closure that may be called much later.
    raised_cosine_time_window(np.empty(0), start, stop, (lead, trail))

    def window(times):
        return raised_cosine_time_window(times, start, stop, (lead, trail))

    return window


def clear_cache():
    """Drop every prepared geometry and cached source."""
    _PREPARED.clear()
    _SOURCE_CACHE.clear()
    _PROJECTED.clear()
    _SAMPLED.clear()
    _RUNTIME_SIGNATURES.clear()
    _PYEFPEHM_NATIVE.clear()


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

    # Prepare and transform every channel the caller will eventually ask for,
    # not just this one. `Relative` iterates its detectors and calls in here
    # once per detector, and each call would otherwise rebuild the band's time
    # series for the same candidate.
    served = tuple(params.get('tdi_channels', requested))
    if not set(requested) <= set(served):
        served = tuple(dict.fromkeys(served + requested))
    # A harmonic the union bins but this candidate does not carry is worth
    # exactly zero, and saying so is the alternative to dropping it in
    # silence. Checked before anything is prepared, because preparing a
    # harmonic the source lacks is what would raise.
    label = _label(params.get('tdi_harmonic'))
    if label is not None:
        if label not in set(_build_source(params).harmonics):
            logging.info("tdi: candidate does not carry harmonic %s; it "
                         "contributes zero", (label,))
            return {name: Array(np.zeros(len(sample_points), dtype=complex))
                    for name in requested}

    orbit = LisaEqualArmOrbit(
        armlength=float(params.get('tdi_arm_length', DEFAULT_ARM)), t0=0.0)
    terms = channel_terms(
        served, generation=int(params.get('tdi_generation', 2)),
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

    _begin_candidate(params)
    response = _projection(params, prepared)
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
    # Two knobs, both needed, both measured on an isolated single-harmonic
    # source against an exact rfft of the same response. `samples_per_cycle`
    # controls the quadrature error, which falls as its square: at the default
    # 4 the amplitude is 1.3e-04 low, at 16 it is 4.5e-06. `spectral_padding`
    # controls the band-edge truncation, which the cadence cannot touch: a
    # requested bin at a band edge is fed only by that band, whose response was
    # built for [f_lower, f_upper] and is cut there. Together they take
    # 1 - |overlap| from 6.5e-06 to 1.8e-08.
    padding = float(params.get('tdi_spectral_padding', 0.0))
    key = (params.get('tdi_preparation_id', 'default'),
           id(prepared), _signature(params), served,
           sample_points.tobytes(), delta_f, epoch, padding)
    samples = _SAMPLED.get(key)
    if samples is None:
        samples = response.frequency_samples(
            {name: sample_points for name in served},
            delta_f={name: delta_f for name in served},
            epoch={name: epoch for name in served},
            channels=served, spectral_padding=padding,
            time_window=_analysis_time_window(params))
        if len(_SAMPLED) >= _SAMPLED_LIMIT:
            _SAMPLED.pop(next(iter(_SAMPLED)))
        _SAMPLED[key] = samples
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
    from pycbc.tdi.sources import (AmplitudeScaledHarmonicSource,
                                   PolarizationRotatedHarmonicSource,
                                   PyEFPEHMSource,
                                   TimeShiftedHarmonicSource)
    keys = ('mass1', 'mass2', 'spin1x', 'spin1y', 'spin1z',
            'spin2x', 'spin2y', 'spin2z', 'distance', 'inclination',
            'eccentricity', 'phase', 'f22_start', 'f22_ref', 'f22_end',
            'Amplitude_tol', 'Interp_points_per_prec_cycle',
            'Series_Reversion_Order')
    arguments = {key: params[key] for key in keys if key in params}
    arguments.setdefault('f22_start', params.get('f_lower'))
    arguments.setdefault('f22_ref', arguments.get('f22_start'))
    requested_distance = float(arguments.get('distance', 100.0))
    if not np.isfinite(requested_distance) or requested_distance <= 0:
        raise ValueError('distance must be positive and finite')
    # Distance enters pyEFPEHM only through h0_pref proportional to 1/d_L.
    # Build the intrinsic state at a canonical 1 Mpc so distance proposals do
    # not reintegrate the same orbit, then restore the requested amplitude.
    arguments['distance'] = 1.0
    epoch = params.get('tdi_preparation_id', 'default')
    signature = tuple(sorted(arguments.items()))
    cached = _PYEFPEHM_NATIVE.get(epoch)
    if cached is None or cached[0] != signature:
        if cached is None and len(_PYEFPEHM_NATIVE) >= _RUNTIME_EPOCH_LIMIT:
            _PYEFPEHM_NATIVE.pop(next(iter(_PYEFPEHM_NATIVE)))
        native = PyEFPEHMSource(arguments)
        _PYEFPEHM_NATIVE[epoch] = (signature, native)
    else:
        native = cached[1]
    # ``tc`` is a mission-clock translation for a detector-response source.
    # A positive value moves every feature later: h_tc(t) = h_0(t - tc), so
    # the source clock queried at mission time t is t + native.t_start - tc.
    scaled = AmplitudeScaledHarmonicSource(
        native, scale=1.0 / requested_distance)
    shifted = TimeShiftedHarmonicSource(
        scaled, offset=native.t_start - float(params.get('tc', 0.0)))
    return PolarizationRotatedHarmonicSource(
        shifted, float(params.get('polarization', 0.0)))


register_source('lal', _lal_dominant)
register_source('pyefpehm', _pyefpehm)


class RestrictedHarmonicSource:
    """A harmonic source narrowed to a subset of its harmonics.

    Two places need this. Relative binning needs each harmonic on its own,
    and `prepare_sparse_tdi` requires every coverage source to be a subset
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
    """The harmonic labels this epoch bins, across its whole prior.

    Two ways to combine the prior's corners, chosen by
    ``tdi_harmonic_combine``.

    ``'intersection'`` (the default) takes only the harmonics every corner
    has. It is safe in the sense that every candidate can supply every
    harmonic, and unsafe in the sense that matters: a candidate carrying more
    is silently served fewer, so a real mode is dropped without anyone being
    told. Measured on an eccentric, precessing source over a production mass
    and eccentricity tile, the native set ranges from nine to fourteen
    harmonics while the intersection is ten -- half the tile loses modes.

    ``'union'`` bins every harmonic any corner has. A candidate that does not
    carry one contributes zero for it, which is what a missing mode is worth,
    and nothing is dropped unannounced. The fiducial is missing some of them
    too, and those cannot be binned against it -- relative binning divides by
    its reference. `harmonic_references` gives each one the first prior
    corner that carries it, and preparing a harmonic is done from that source
    rather than from the fiducial. It costs the harmonics that only some of
    the prior has, and it works only on the per-harmonic path: one
    preparation is built from one source.
    """
    combine = str(params.get('tdi_harmonic_combine', 'intersection'))
    if combine not in ('intersection', 'union'):
        raise ValueError("tdi_harmonic_combine is 'intersection' or 'union'")
    labels = set(_build_source(params).harmonics)
    for source in _coverage_sources(params):
        other = set(source.harmonics)
        labels = labels | other if combine == 'union' else labels & other
    return tuple(sorted(labels))
