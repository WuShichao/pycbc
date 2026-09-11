# -*- coding: UTF-8 -*-
#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
"""
This module provides utilities for simulating the GW response of space-based
observatories.
"""
import logging
from abc import ABC, abstractmethod

import numpy
from astropy import constants
from numpy import cos, sin

from pycbc.coordinates.space import TIME_OFFSET_20_DEGREES
from pycbc.types import TimeSeries


def get_available_space_detectors():
    """List the available space detectors"""
    dets = list(_space_detectors.keys())
    aliases = []
    for i in dets:
        aliases.extend(_space_detectors[i]['aliases'])
    return dets + aliases

def parse_det_name(detector_name):
    """Parse a string into a detector name and TDI channel.
       The input is assumed to look like '{detector name}_{channel name}.'"""
    out = detector_name.split('_', 1)
    det = out[0]
    try:
        chan = out[1]
    except IndexError:
        # detector_name is just the detector, so save channel name as None
        chan = None
    return det, chan

def apply_polarization(hp, hc, polarization):
    """
    Apply polarization rotation matrix.

    Parameters
    ----------
    hp : array
        The plus polarization of the GW.

    hc : array
        The cross polarization of the GW.

    polarization : float
        The SSB polarization angle of the GW in radians.

    Returns
    -------
    (array, array)
        The plus and cross polarizations of the GW rotated by the
        polarization angle.
    """
    cphi = cos(2*polarization)
    sphi = sin(2*polarization)

    hp_ssb = hp*cphi - hc*sphi
    hc_ssb = hp*sphi + hc*cphi

    return hp_ssb, hc_ssb

def check_signal_times(hp, hc, orbit_start_time, orbit_end_time,
                       offset=TIME_OFFSET_20_DEGREES, pad_data=False, t0=1e4):
    """
    Ensure that input signal lies within the provided orbital window. This
    assumes that the start times of hp and hc are relative to the detector
    mission start time.

    Parameters
    ----------
    hp : pycbc.types.TimeSeries
        The plus polarization of the GW.

    hc : pycbc.types.TimeSeries
        The cross polarization of the GW.

    orbit_start_time : float
        SSB start time in seconds of the orbital data. By convention,
        t = 0 corresponds to the mission start time of the detector.

    orbit_end_time : float
        SSB end time in seconds of the orbital data.

    polarization : float (optional)
        The polarization in radians of the GW. Default 0.

    offset : float (optional)
        Time offset in seconds to apply to SSB times to ensure proper
        orientation of the constellation at t=0. Default 7365189.431698299
        for a 20 degree offset from Earth.

    pad_data : bool (optional)
        Flag whether to pad the input GW data with time length t0
        worth of zeros. Default False.

    t0 : float (optional)
        Time duration in seconds by which to pad the data if pad_data
        is True. Default 1e4.

    Returns
    -------
    (pycbc.types.TimeSeries, pycbc.types.TimeSeries)
        The plus and cross polarizations of the GW in the SSB frame,
        padded as requested and/or truncated to fit in the orbital window.
    """
    dt = hp.delta_t

    # apply offsets to wfs
    hp.start_time += offset
    hc.start_time += offset

    # pad the data with zeros
    if pad_data:
        pad_idx = int(t0/dt)
        hp.prepend_zeros(pad_idx)
        hp.append_zeros(pad_idx)
        hc.prepend_zeros(pad_idx)
        hc.append_zeros(pad_idx)

    # make sure signal lies within orbit length
    if hp.duration + hp.start_time > orbit_end_time:
        logging.warning('Time of signal end is greater than end of orbital ' +
                        f'data. Cutting signal at {orbit_end_time}.')
        # cut off data succeeding orbit end time
        end_idx = numpy.argwhere(hp.sample_times.numpy() <= orbit_end_time)[-1][0]
        hp = hp[:end_idx]
        hc = hc[:end_idx]

    if hp.start_time < orbit_start_time:
        logging.warning('Time of signal start is less than start of orbital ' +
                        f'data. Cutting signal at {orbit_start_time}.')
        # cut off data preceding orbit start time
        start_idx = numpy.argwhere(hp.sample_times.numpy() >= orbit_start_time)[0][0]
        hp = hp[start_idx:]
        hc = hc[start_idx:]

    return hp, hc

def cut_channels(tdi_dict, remove_garbage=False, t0=1e4):
    """
    Cut TDI channels if needed.

    Parameters
    ----------
    tdi_dict : dict
        The TDI channels, formatted as a dictionary of TimeSeries arrays
        keyed by the channel label.

    remove_garbage : bool, str (optional)
        Flag whether to remove data from the edges of the channels. If True,
        time length t0 is cut from the start and end. If 'zero', time length
        t0 is zeroed at the start and end. If False, channels are unmodified.
        Default False.

    t0 : float (optional)
        Time in seconds to cut/zero from data if remove_garbage is True/'zero'.
        Default 1e4.
    """
    for chan in tdi_dict.keys():
        if remove_garbage:
            dt = tdi_dict[chan].delta_t
            pad_idx = int(t0/dt)
            if remove_garbage == 'zero':
                # zero the edge data
                tdi_dict[chan][:pad_idx] = 0
                tdi_dict[chan][-pad_idx:] = 0
            elif type(remove_garbage) == bool:
                # cut the edge data
                slc = slice(pad_idx, -pad_idx)
                tdi_dict[chan] = tdi_dict[chan][slc]
            else:
                raise ValueError('remove_garbage arg must be a bool or "zero"')

    return tdi_dict

_space_detectors = {'LISA': {'armlength': 2.5e9,
                             'aliases': ['LISA_A', 'LISA_E', 'LISA_T',
                                         'LISA_X', 'LISA_Y', 'LISA_Z'],
                            },
                    # Taiji and TianQin are registered so they are
                    # discoverable via `get_available_space_detectors`, but
                    # do not yet have a working response/TDI backend -- see
                    # `_Generic_detector` below. Armlengths are nominal
                    # mission design values; see project docs for details.
                    'Taiji': {'armlength': 3.0e9,
                             'aliases': ['Taiji_A', 'Taiji_E', 'Taiji_T',
                                         'Taiji_X', 'Taiji_Y', 'Taiji_Z'],
                            },
                    # TianQin's theoretical design armlength (not the
                    # 1.7e5 km engineering rounding also seen in the
                    # literature), matching space_orbit.TianQinAnalyticOrbit
                    # and pycbc.psd.analytical_space's TianQin PSD functions.
                    'TianQin': {'armlength': numpy.sqrt(3) * 1e8,
                             'aliases': ['TianQin_A', 'TianQin_E',
                                         'TianQin_T', 'TianQin_X',
                                         'TianQin_Y', 'TianQin_Z'],
                            },
                    # LGWA (Lunar Gravitational Wave Antenna) is a single
                    # lunar-surface instrument, not a multi-spacecraft
                    # constellation like the three above -- 'armlength' is
                    # not a meaningful concept for it and is left as None.
                    # The two aliases are its two horizontal sensing axes
                    # (see `_LGWA_detector`), not TDI channels -- LGWA, as
                    # a single station, has no TDI combination.
                    'LGWA': {'armlength': None,
                             'aliases': ['LGWA_X', 'LGWA_Y']},
                    # LILA Observatory: an equilateral triangle of three
                    # lunar-surface stations, so unlike LGWA it does have
                    # a meaningful armlength -- the 40 km triangle side.
                    # Its aliases are the three vertex interferometers
                    # ('_1'/'_2'/'_3') and their orthogonal recombination
                    # ('_A'/'_E'/'_T', with T the null stream); as with
                    # LGWA these are not TDI channels, since each vertex
                    # is an ordinary Michelson, not a spacecraft link.
                    'LILA': {'armlength': 4.0e4,
                             'aliases': ['LILA_1', 'LILA_2', 'LILA_3',
                                         'LILA_A', 'LILA_E', 'LILA_T']},
                   }

class AbsSpaceDet(ABC):
    """
    Abstract base class to set structure for space detector classes.

    Parameters
    ----------
    detector_name : str
        The name of the detector. Accepts any output from
        `get_available_space_detectors`.
    
    reference_time : float (optional)
        The reference time in seconds of the signal in the SSB frame. This is
        defined such that the detector mission start time corresponds to 0.
        Default None.
    """
    def __init__(self, detector_name, reference_time=None, **kwargs):
        self.det, self.chan = parse_det_name(detector_name)
        if detector_name not in get_available_space_detectors():
            raise NotImplementedError('Unrecognized detector. ',
                                      'Currently accepts: ',
                                      f'{get_available_space_detectors()}')
        self.reference_time = reference_time

    @property
    @abstractmethod
    def sky_coords(self):
        """
        List the sky coordinate names for the detector class.
        """
        return

    @abstractmethod
    def project_wave(self, hp, hc, lamb, beta):
        """
        Placeholder for evaluating the TDI channels from the GW projections.
        """
        return


class _LDC_detector(AbsSpaceDet):
    """
    LISA detector modeled using LDC software. Constellation orbits are
    generated using LISA Orbits (https://pypi.org/project/lisaorbits/).
    Link projections are generated using LISA GW Response
    (10.5281/zenodo.6423435). TDI channels are generated using pyTDI
    (10.5281/zenodo.6351736).

    Parameters
    ----------
    detector_name : str
        The name of the detector. Accepts any output from
        `get_available_space_detectors`.
    
    reference_time : float (optional)
        The reference time in seconds of the signal in the SSB frame. This is
        defined such that the detector mission start time corresponds to 0.
        Default None.

    apply_offset : bool (optional)
        Flag whether to shift the times of the input waveforms by
        a given value. Some backends require this such that the
        detector is oriented correctly at t = 0. Default False.

    offset : float (optional)
        The time in seconds by which to offset the input waveform if
        apply_offset is True. Default 7365189.431698299.
    
    orbits : str (optional)
        The constellation orbital data used for generating projections
        and TDI. See self.orbits_init for accepted inputs. Default
        'EqualArmlength'.
    """
    def __init__(self, detector_name, reference_time=None, apply_offset=False,
                 offset=TIME_OFFSET_20_DEGREES,
                 orbits='EqualArmlength', **kwargs):
        super().__init__(detector_name, reference_time, **kwargs)
        assert self.det == 'LISA', 'LDC backend only works with LISA detector'

        # specify whether to apply offsets to GPS times
        if apply_offset:
            self.offset = offset
        else:
            self.offset = 0.

        # orbits properties
        self.orbits = orbits
        self.orbits_start_time = None
        self.orbits_end_time = None

        # waveform properties
        self.dt = None
        self.sample_times = None
        self.start_time = None

        # pre- and post-processing
        self.pad_data = False
        self.remove_garbage = False
        self.t0 = 1e4

        # class initialization
        self.proj_init = None
        self.tdi_init = None
        self.tdi_chan = 'AET'
        if self.chan is not None and self.chan in 'XYZ':
            self.tdi_chan = 'XYZ'

    @property
    def sky_coords(self):
        return 'eclipticlongitude', 'eclipticlatitude'

    def orbits_init(self, orbits, size=316, dt=100000.0, t_init=0.0):
        """
        Initialize the orbital information for the constellation. Defualt args
        generate roughly 1 year worth of data starting at LISA mission
        start time.

        Parameters
        ----------
        orbits : str
            The type of orbit to read in. If "EqualArmlength" or "Keplerian",
            a file is generating using the corresponding method from LISA
            Orbits. Else, the input is treated as a file path following LISA
            Orbits format. Default "EqualArmlength".

        length : int (optional)
            The number of samples to generate if creating a new orbit file.
            Default 316.

        dt : float (optional)
            The time step in seconds to use if generating a new orbit file.
            Default 100000.

        t_init : float (optional)
            The start time in seconds to use if generating a new orbit file.
            Default 0.
        """
        defaults = ['EqualArmlength', 'Keplerian']
        assert type(orbits) == str, ('Must input either a file path as ',
                                     'str, "EqualArmlength", or "Keplerian"')

        # generate a new file
        if orbits in defaults:
            try:
                import lisaorbits
            except ImportError:
                raise ImportError('lisaorbits not found')
            if orbits == 'EqualArmlength':
                o = lisaorbits.EqualArmlengthOrbits()
            if orbits == 'Keplerian':
                o = lisaorbits.KeplerianOrbits()
            o.write('orbits.h5', dt=dt, size=size, t0=t_init, mode='w')
            ofile = 'orbits.h5'
            self.orbits_start_time = t_init
            self.orbits_end_time = t_init + size*dt
            self.orbits = ofile

        # read in from an existing file path
        else:
            import h5py
            ofile = orbits
            with h5py.File(ofile, 'r') as f:
                self.orbits_start_time = f.attrs['t0']
                self.orbits_end_time = self.orbit_start_time + \
                                       f.attrs['dt']*f.attrs['size']

        # add light travel buffer times
        lisa_arm = _space_detectors['LISA']['armlength']
        ltt_au = constants.au.value / constants.c.value
        ltt_arm = lisa_arm / constants.c.value
        self.orbits_start_time += ltt_arm + ltt_au
        self.orbits_end_time += ltt_au

    def strain_container(self, response, orbits=None):
        """
        Read in the necessary link and orbit information for generating TDI
        channels. Replicates the functionality of `pyTDI.Data.from_gws()`.

        Parameters
        ----------
        response : array
            The laser link projections of the GW. Uses get_links output format.

        orbits : str, optional
            The path to the file containing orbital information for the LISA
            constellation. Default to orbits class attribute.

        Returns
        -------
        dict, array
            The arguments and measurements associated with the link and orbital
            data.
        """
        try:
            from pytdi import Data
        except ImportError:
            raise ImportError('pyTDI required for TDI combinations')

        links = ['12', '23', '31', '13', '32', '21']

        # format the measurements from link data
        measurements = {}
        for i, link in enumerate(links):
            measurements[f'isi_{link}'] = response[:, i]
            measurements[f'isi_sb_{link}'] = response[:, i]
            measurements[f'tmi_{link}'] = 0.
            measurements[f'rfi_{link}'] = 0.
            measurements[f'rfi_sb_{link}'] = 0.

        df = 1/self.dt
        t_init = self.orbits_start_time

        # call in the orbital data using pyTDI
        if orbits is None:
            orbits = self.orbits
        return Data.from_orbits(orbits, df, t_init, 'tcb/ltt', **measurements)

    def get_links(self, hp, hc, lamb, beta, polarization):
        """
        Project a radiation frame waveform to the LISA constellation.

        Parameters
        ----------
        hp : pycbc.types.TimeSeries
            The plus polarization of the GW in the radiation frame.

        hc : pycbc.types.TimeSeries
            The cross polarization of the GW in the radiation frame.

        lamb : float
            The ecliptic longitude of the source in the SSB frame.

        beta : float
            The ecliptic latitude of the source in the SSB frame.

        polarization : float (optional)
            The polarization angle of the GW in radians. Default 0.

        Returns
        -------
        ndarray
            The waveform projected to the LISA laser links. Shape is (6, N)
            for input waveforms with N total samples.
        """
        try:
            from lisagwresponse import ReadStrain
        except ImportError:
            raise ImportError('LISA GW Response not found')

        if self.dt is None:
            self.dt = hp.delta_t

        # configure orbits and signal
        self.orbits_init(orbits=self.orbits)
        hp, hc = check_signal_times(hp, hc, self.orbits_start_time,
                                    self.orbits_end_time, offset=self.offset,
                                    pad_data=self.pad_data, t0=self.t0)
        self.start_time = hp.start_time - self.offset
        self.sample_times = hp.sample_times.numpy()

        # apply polarization
        hp, hc = apply_polarization(hp, hc, polarization)

        if self.proj_init is None:
            # initialize the class
            self.proj_init = ReadStrain(self.sample_times, hp, hc,
                                        gw_beta=beta, gw_lambda=lamb,
                                        orbits=self.orbits)
        else:
            # update params in the initialized class
            self.proj_init.gw_beta = beta
            self.proj_init.gw_lambda = lamb
            self.proj_init.set_strain(self.sample_times, hp, hc)

        # project the signal
        wf_proj = self.proj_init.compute_gw_response(self.sample_times,
                                                     self.proj_init.LINKS)

        return wf_proj

    def project_wave(self, hp, hc, lamb, beta, polarization=0,
                     tdi=1.5, tdi_chan=None, pad_data=False,
                     remove_garbage=False, t0=1e4, **kwargs):
        """
        Evaluate the TDI observables.

        The TDI generation requires some startup time at the start and end of
        the waveform, creating erroneous ringing or "garbage" at the edges of
        the signal. By default, this method will cut off a time length t0 from
        the start and end to remove this garbage, which may delete sensitive
        data at the edges of the input strains (e.g., the late inspiral and
        ringdown of a binary merger). Thus, the default output will be shorter
        than the input by (2*t0) seconds. See pad_data and remove_garbage to
        modify this behavior.

        Parameters
        ----------
        hp : pycbc.types.TimeSeries
            The plus polarization of the GW in the radiation frame.

        hc : pycbc.types.TimeSeries
            The cross polarization of the GW in the radiation frame.

        lamb : float
            The ecliptic longitude in the SSB frame.

        beta : float
            The ecliptic latitude in the SSB frame.

        polarization : float
            The polarization angle of the GW in radians.

        tdi : float (optional)
            TDI channel configuration. Accepts 1.5 for 1st generation TDI or
            2 for 2nd generation TDI. Default 1.5.

        tdi_chan : str (optional)
            The TDI observables to calculate. Accepts 'XYZ', 'AET', or 'AE'.
            Default 'AET'.

        pad_data : bool (optional)
            Flag whether to pad the data with time length t0 of zeros at the
            start and end. Default False.

        remove_garbage : bool, str (optional)
            Flag whether to remove gaps in TDI from start and end. If True,
            time length t0 worth of data at the start and end of the waveform
            will be cut from TDI channels. If 'zero', time length t0 worth of
            edge data will be zeroed. If False, TDI channels will not be
            modified. Default False.

        t0 : float (optional)
            Time length in seconds to pad/cut from the start and end of
            the data if pad_data/remove_garbage is True. Default 1e4.

        Returns
        -------
        dict ({str: pycbc.types.TimeSeries})
            The TDI observables as TimeSeries objects keyed by their
            corresponding TDI channel name.
        """
        try:
            from pytdi import michelson
        except ImportError:
            raise ImportError('pyTDI not found')

        # set TDI generation
        if tdi == 1.5:
            X, Y, Z = michelson.X1, michelson.Y1, michelson.Z1
        elif tdi == 2:
            X, Y, Z = michelson.X2, michelson.Y2, michelson.Z2
        else:
            raise ValueError('Unrecognized TDI generation input. ' +
                             'Please input either 1 or 2.')

        # set TDI channels
        if tdi_chan is None:
            tdi_chan = self.tdi_chan

        # generate the Doppler time series
        self.pad_data = pad_data
        self.remove_garbage = remove_garbage
        self.t0 = t0
        response = self.get_links(hp, hc, lamb, beta,
                                  polarization=polarization)

        # load in data using response measurements
        self.tdi_init = self.strain_container(response, self.orbits)

        # generate the XYZ TDI channels
        chanx = X.build(**self.tdi_init.args)(self.tdi_init.measurements)
        chany = Y.build(**self.tdi_init.args)(self.tdi_init.measurements)
        chanz = Z.build(**self.tdi_init.args)(self.tdi_init.measurements)

        # convert to AET if specified
        if tdi_chan == 'XYZ':
            tdi_dict = {'LISA_X': TimeSeries(chanx, delta_t=self.dt,
                                             epoch=self.start_time),
                        'LISA_Y': TimeSeries(chany, delta_t=self.dt,
                                             epoch=self.start_time),
                        'LISA_Z': TimeSeries(chanz, delta_t=self.dt,
                                             epoch=self.start_time)}
        elif tdi_chan == 'AET':
            chana = (chanz - chanx)/numpy.sqrt(2)
            chane = (chanx - 2*chany + chanz)/numpy.sqrt(6)
            chant = (chanx + chany + chanz)/numpy.sqrt(3)
            tdi_dict = {'LISA_A': TimeSeries(chana, delta_t=self.dt,
                                             epoch=self.start_time),
                        'LISA_E': TimeSeries(chane, delta_t=self.dt,
                                             epoch=self.start_time),
                        'LISA_T': TimeSeries(chant, delta_t=self.dt,
                                             epoch=self.start_time)}
        else:
            raise ValueError('Unrecognized TDI channel input. ' +
                             'Please input either "XYZ" or "AET".')

        # processing
        tdi_dict = cut_channels(tdi_dict, remove_garbage=self.remove_garbage,
                                t0=self.t0)
        return tdi_dict


class _FLR_detector(AbsSpaceDet):
    """
    LISA detector modeled using FastLISAResponse. Constellation orbits are
    generated using LISA Analysis Tools (10.5281/zenodo.10930979). Link
    projections and TDI channels are generated using FastLISAResponse
    (https://arxiv.org/abs/2204.06633).

    Parameters
    ----------
    detector_name : str
        The name of the detector. Accepts any output from
        `get_available_space_detectors`.
    
    reference_time : float (optional)
        The reference time in seconds of the signal in the SSB frame. This is
        defined such that the detector mission start time corresponds to 0.
        Default None.

    apply_offset : bool (optional)
        Flag whether to shift the times of the input waveforms by
        a given value. Some backends require this such that the
        detector is oriented correctly at t = 0. Default False.

    offset : float (optional)
        The time in seconds by which to offset the input waveform if
        apply_offset is True. Default 7365189.431698299.

    use_gpu : bool (optional)
        Specify whether to run class on GPU support via CuPy. Default False.

    orbits : str (optional)
        The constellation orbital data used for generating projections
        and TDI. See self.orbits_init for accepted inputs. Default
        'EqualArmlength'.
    """
    def __init__(self, detector_name, reference_time=None, apply_offset=False,
                 offset=TIME_OFFSET_20_DEGREES,
                 orbits='EqualArmlength', use_gpu=False, **kwargs):
        logging.warning('WARNING: FastLISAResponse TDI implementation is a ',
                        'work in progress. Currently unable to reproduce LDC ',
                        'or BBHx waveforms.')
        self.use_gpu = use_gpu
        super().__init__(detector_name, reference_time, **kwargs)
        assert self.det == 'LISA', 'FLR backend only works with LISA detector'

        # specify whether to apply offsets to GPS times
        if apply_offset:
            self.offset = offset
        else:
            self.offset = 0.

        # orbits properties
        self.orbits = orbits
        self.orbits_start_time = None
        self.orbits_end_time = None

        # waveform properties
        self.dt = None
        self.sample_times = None
        self.start_time = None

        # pre- and post-processing
        self.pad_data = False
        self.remove_garbage = False
        self.t0 = 1e4

        # class initialization
        self.tdi_init = None
        self.tdi_chan = 'AET'
        if 'tdi_chan' in kwargs.keys():
            if kwargs['tdi_chan'] is not None and kwargs['tdi_chan'] in 'XYZ':
                self.tdi_chan = 'XYZ'

    @property
    def sky_coords(self):
        return 'eclipticlongitude', 'eclipticlatitude'

    def orbits_init(self, orbits):
        """
        Initialize the orbital information for the constellation.

        Parameters
        ----------
        orbits : str
            The type of orbit to read in. If "EqualArmlength" or "ESA", the
            corresponding Orbits class from LISA Analysis Tools is called.
            Else, the input is treated as a file path following LISA
            Orbits format.
        """
        # if orbits are already a class instance, skip this
        if type(self.orbits) is not (str or None):
            return

        try:
            from lisatools import detector
        except ImportError:
            raise ImportError("LISA Analysis Tools required for FLR orbits")

        # load an orbit from lisatools
        defaults = ['EqualArmlength', 'ESA']
        if orbits in defaults:
            if orbits == 'EqualArmlength':
                o = detector.EqualArmlengthOrbits()
            if orbits == 'ESA':
                o = detector.ESAOrbits()

        # create a new orbits instance for file input
        else:
            class CustomOrbits(detector.Orbits):
                def __init__(self):
                    super().__init__(orbits)
            o = CustomOrbits()

        self.orbits = o
        self.orbits_start_time = self.orbits.t_base[0]
        self.orbits_end_time = self.orbits.t_base[-1]

    def get_links(self, hp, hc, lamb, beta, polarization=0, use_gpu=None):
        """
        Project a radiation frame waveform to the LISA constellation.

        Parameters
        ----------
        hp : pycbc.types.TimeSeries
            The plus polarization of the GW in the radiation frame.

        hc : pycbc.types.TimeSeries
            The cross polarization of the GW in the radiation frame.

        lamb : float
            The ecliptic longitude of the source in the SSB frame.

        beta : float
            The ecliptic latitude of the source in the SSB frame.

        polarization : float (optional)
            The polarization angle of the GW in radians. Default 0.

        use_gpu : bool (optional)
            Flag whether to use GPU support. Default to class input.
            CuPy is required if use_gpu is True.

        Returns
        -------
        ndarray
            The waveform projected to the LISA laser links. Shape is (6, N)
            for input waveforms with N total samples.
        """
        try:
            from fastlisaresponse import pyResponseTDI
        except ImportError:
            raise ImportError('FastLISAResponse not found')

        if self.dt is None:
            self.dt = hp.delta_t

        # configure the orbit and signal
        self.orbits_init(orbits=self.orbits)
        hp, hc = check_signal_times(hp, hc, self.orbits_start_time,
                                    self.orbits_end_time, offset=self.offset,
                                    pad_data=self.pad_data, t0=self.t0)
        self.start_time = hp.start_time - self.offset
        self.sample_times = hp.sample_times.numpy()
        
        # apply polarization
        hp, hc = apply_polarization(hp, hc, polarization)

        # interpolate orbital data to signal sample times
        self.orbits.configure(t_arr=self.sample_times)

        # format wf to hp + i*hc
        hp = hp.numpy()
        hc = hc.numpy()
        wf = hp + 1j*hc

        if use_gpu is None:
            use_gpu = self.use_gpu

        # convert to cupy if needed
        if use_gpu:
            import cupy
            wf = cupy.asarray(wf)

        if self.tdi_init is None:
            # initialize the class
            self.tdi_init = pyResponseTDI(1/self.dt, len(wf),
                                          orbits=self.orbits,
                                          use_gpu=use_gpu)
        else:
            # update params in the initialized class
            self.tdi_init.sampling_frequency = 1/self.dt
            self.tdi_init.num_pts = len(wf)
            self.tdi_init.orbits = self.orbits
            self.tdi_init.use_gpu = use_gpu

        # project the signal
        self.tdi_init.get_projections(wf, lamb, beta, t0=self.t0)
        wf_proj = self.tdi_init.y_gw

        return wf_proj

    def project_wave(self, hp, hc, lamb, beta, polarization=0,
                     tdi=1.5, tdi_chan=None, use_gpu=None, pad_data=False,
                     remove_garbage=False, t0=1e4, **kwargs):
        """
        Evaluate the TDI observables.

        The TDI generation requires some startup time at the start and end of
        the waveform, creating erroneous ringing or "garbage" at the edges of
        the signal. By default, this method will cut off a time length t0 from
        the start and end to remove this garbage, which may delete sensitive
        data at the edges of the input strains (e.g., the late inspiral and
        ringdown of a binary merger). Thus, the default output will be shorter
        than the input by (2*t0) seconds. See pad_data and remove_garbage to
        modify this behavior.

        Parameters
        ----------
        hp : pycbc.types.TimeSeries
            The plus polarization of the GW in the radiation frame.

        hc : pycbc.types.TimeSeries
            The cross polarization of the GW in the radiation frame.

        lamb : float
            The ecliptic longitude in the SSB frame.

        beta : float
            The ecliptic latitude in the SSB frame.

        polarization : float (optional)
            The polarization angle of the GW in radians.

        tdi : float(optional)
            TDI channel configuration. Accepts 1.5 for 1st generation TDI or
            2 for 2nd generation TDI. Default 1.5.

        tdi_chan : str (optional)
            The TDI observables to calculate. Accepts 'XYZ', 'AET', or 'AE'.
            Default 'AET'.

        use_gpu : bool (optional)
            Flag whether to use GPU support. Default False.

        pad_data : bool (optional)
            Flag whether to pad the data with time length t0 of zeros at the
            start and end. Default False.

        remove_garbage : bool, str (optional)
            Flag whether to remove gaps in TDI from start and end. If True,
            time length t0 worth of data at the start and end of the waveform
            will be cut from TDI channels. If 'zero', time length t0 worth of
            edge data will be zeroed. If False, TDI channels will not be
            modified. Default False.

        t0 : float (optional)
            Time length in seconds to pad/cut from the start and end of
            the data if pad_data/remove_garbage is True. Default 1e4.

        Returns
        -------
        dict ({str: pycbc.types.TimeSeries})
            The TDI observables as TimeSeries objects keyed by their
            corresponding TDI channel name.
        """
        # set use_gpu
        if use_gpu is None:
            use_gpu = self.use_gpu

        # generate the Doppler time series
        self.pad_data = pad_data
        self.remove_garbage = remove_garbage
        self.t0 = t0
        self.get_links(hp, hc, lamb, beta, polarization=polarization,
                       use_gpu=use_gpu)

        # set TDI configuration (let FLR handle if not 1 or 2)
        if tdi == 1.5:
            tdi_opt = '1st generation'
        elif tdi == 2:
            tdi_opt = '2nd generation'
        else:
            tdi_opt = tdi

        if tdi_opt != self.tdi_init.tdi:
            # update TDI in existing tdi_init class
            self.tdi_init.tdi = tdi_opt
            self.tdi_init._init_TDI_delays()

        # set TDI channels
        if tdi_chan is None:
            tdi_chan = self.tdi_chan

        if tdi_chan in ['XYZ', 'AET', 'AE']:
            self.tdi_init.tdi_chan = tdi_chan
        else:
            raise ValueError('TDI channels must be one of: XYZ, AET, AE')

        # generate the TDI channels
        tdi_obs = self.tdi_init.get_tdi_delays()

        # processing
        tdi_dict = {}
        for i, chan in enumerate(tdi_chan):
            # save as TimeSeries
            tdi_dict[f'LISA_{chan}'] = TimeSeries(tdi_obs[i], delta_t=self.dt,
                                           epoch=self.start_time)

        tdi_dict = cut_channels(tdi_dict, remove_garbage=self.remove_garbage, 
                                t0=self.t0)
        return tdi_dict


class _LGWA_detector(AbsSpaceDet):
    """
    LGWA (Lunar Gravitational Wave Antenna) detector modeled using the
    `lgwa_response` package (https://github.com/jacopok/lgwa-response,
    Tissino et al. 2026, arXiv:2606.04918) for its antenna-pattern
    geometry only -- not its built-in waveform/likelihood machinery
    (`lgwa_response.likelihood`/`bilby_interface`, which hard-import
    `bilby` at module level). Only the bilby-free
    `lgwa_response.lunar_coordinates` submodule is used, which wraps
    `lunarsky` to compute the detector's real, libration-aware
    orientation on the Moon.

    LGWA is modeled as a single lunar-surface station with two
    orthogonal horizontal sensing axes (not a multi-spacecraft
    interferometer constellation), so there is no TDI combination here
    -- `project_wave` returns the two horizontal-axis channels directly.

    Not installable from PyPI (GPLv3, no tagged releases yet); install
    with `pip install git+https://github.com/jacopok/lgwa-response.git`.

    Parameters
    ----------
    detector_name : str
        The name of the detector. Accepts any output from
        `get_available_space_detectors`.

    reference_time : float (optional)
        The reference time in seconds of the signal in the SSB frame. This is
        defined such that the detector mission start time corresponds to 0.
        Default None.

    longitude_site, latitude_site : float
        Selenodetic longitude/latitude (radians) of the specific LGWA
        surface site, in the same convention as
        `pycbc.coordinates.moon.moon_site_position_ssb`. Required:
        unlike arrival time (which can fall back to the Moon's
        barycenter), antenna-pattern geometry needs an actual oriented
        site.

    cadence : float (optional)
        Spacing, in seconds, of the coarse time grid on which the
        detector's orientation is evaluated via `lunarsky` (slow) before
        being linearly interpolated onto the input waveform's sample
        times (fast). Lunar libration evolves on a timescale of days, so
        the default of 3600s (1 hour) is generously fine; increase it to
        speed up long-duration signals at the cost of interpolation
        accuracy.
    """
    def __init__(self, detector_name, reference_time=None,
                 longitude_site=None, latitude_site=None,
                 cadence=3600.0, **kwargs):
        super().__init__(detector_name, reference_time, **kwargs)
        assert self.det == 'LGWA', (
            'LGWAResponse backend only works with the LGWA detector')
        if longitude_site is None or latitude_site is None:
            raise ValueError(
                'longitude_site and latitude_site (radians) are required: '
                'antenna-pattern geometry needs an actual oriented site on '
                "the Moon, unlike arrival time (which can default to the "
                "Moon's barycenter, see "
                'coordinates.moon.moon_site_position_ssb).')
        self.longitude_site = longitude_site
        self.latitude_site = latitude_site
        self.cadence = cadence
        # Cache of the coarse lunarsky/astropy orientation grid, keyed by
        # nothing but the last-built [t_start, t_end] span (site position
        # and cadence are fixed for the lifetime of this instance): a
        # single instance is expected to be reused across many
        # project_wave/_detector_frame calls covering overlapping time
        # ranges (e.g. repeated relbin likelihood evaluations for the
        # same analysis segment), and rebuilding this grid from scratch
        # every call is by far the dominant cost of both (~85-90% of a
        # project_wave call, measured directly -- the lunarsky/astropy
        # calls in generate_data_response, not the cheap numpy
        # interpolation/combination that follows).
        self._frame_cache = None  # (t_start, t_end, times, data) or None

    @property
    def sky_coords(self):
        return 'eclipticlongitude', 'eclipticlatitude'

    def _detector_frame(self, t_start, t_end, query_times):
        """
        Detector-orientation unit vectors (n, x, y), each shape (M, 3) in
        the ICRS frame, at the given `query_times` (an array of length
        M, not necessarily uniformly spaced or sorted), obtained by
        evaluating `lunar_coordinates.generate_data_response` on a coarse
        `self.cadence`-spaced grid spanning [t_start, t_end] (lazy, slow
        -- real `lunarsky`/astropy calls) and linearly interpolating each
        of its 6 unwrapped angle columns onto `query_times` (fast),
        mirroring the interpolation approach `lgwa_response` itself uses
        internally (see
        `lunar_coordinates.test_interpolation_error_response`).

        The coarse grid is cached on `self` (see `__init__`): if
        [t_start, t_end] falls entirely within a previously-built grid's
        span, that grid is reused as-is; otherwise a new grid spanning
        the union of the requested range and any existing cached range
        is built (so a sequence of calls with growing-but-overlapping
        ranges only ever extends the cache, it doesn't rebuild the parts
        already covered).

        Only `lgwa_response.lunar_coordinates` is imported here -- this
        submodule does not import `bilby`.
        """
        try:
            from lgwa_response import lunar_coordinates
        except ImportError as exc:
            raise ImportError(
                'lgwa_response is required for the LGWAResponse backend '
                '(only its bilby-free lunar_coordinates submodule is '
                'used); install with `pip install git+'
                'https://github.com/jacopok/lgwa-response.git`.') from exc

        if self._frame_cache is not None:
            cached_start, cached_end, times, data = self._frame_cache
            if t_start >= cached_start and t_end <= cached_end:
                return self._interpolate_frame(
                    lunar_coordinates, times, data, query_times)
            t_start = min(t_start, cached_start)
            t_end = max(t_end, cached_end)

        n_points = max(2, int(numpy.ceil((t_end - t_start) / self.cadence))
                        + 1)
        lgwa_position = {
            'longitude': float(numpy.degrees(self.longitude_site)),
            'latitude': float(numpy.degrees(self.latitude_site))}
        times, data = lunar_coordinates.generate_data_response(
            n_points, lgwa_position, gps_time_start=t_start,
            gps_time_end=t_end)
        self._frame_cache = (t_start, t_end, times, data)

        return self._interpolate_frame(
            lunar_coordinates, times, data, query_times)

    @staticmethod
    def _interpolate_frame(lunar_coordinates, times, data, query_times):
        """Linear-interpolate a `generate_data_response` grid's 6
        unwrapped angle columns onto `query_times`, and convert the
        interpolated (n, x, y) angle pairs to ICRS Cartesian unit
        vectors. Split out of `_detector_frame` so both the cache-hit
        and cache-miss paths share the same interpolation code.
        """
        interp = numpy.empty((len(query_times), 6))
        for j in range(6):
            interp[:, j] = numpy.interp(query_times, times, data[:, j])

        n = lunar_coordinates.spherical_to_cartesian(
            interp[:, 0], interp[:, 1])
        x = lunar_coordinates.spherical_to_cartesian(
            interp[:, 2], interp[:, 3])
        y = lunar_coordinates.spherical_to_cartesian(
            interp[:, 4], interp[:, 5])
        return n, x, y

    @staticmethod
    def _antenna_pattern_factors(n, x, y, ra, dec, psi):
        """
        Antenna-pattern combination, replicated from
        `lgwa_response.likelihood.LunarLikelihood.get_antenna_response`
        (pure numpy, no bilby import needed) -- shared by `project_wave`
        (time domain) and the frequency-domain SPA response in
        `pycbc.waveform.lgwa` (each evaluates it at different query
        times, but the formula itself is identical).

        Parameters
        ----------
        n, x, y : numpy.array
            Detector-orientation unit vectors (ICRS Cartesian), each
            shape (M, 3), as returned by `_detector_frame`.
        ra, dec, psi : float
            The source's ICRS right ascension/declination and
            `lgwa_response`-convention polarization angle (radians), as
            returned by `coordinates.moon.moon_to_geo(...,
            lal_convention=False)`.

        Returns
        -------
        (hpx, hcx, hpy, hcy) : tuple of numpy.array
            Each shape (M,): the F+/Fx antenna-pattern factors for the
            X and Y horizontal sensing axes.
        """
        from lgwa_response import lunar_coordinates
        u, v = lunar_coordinates.wave_frame_basis_cartesian(ra, dec, -psi)

        un, ux, uy = n @ u, x @ u, y @ u
        vn, vx, vy = n @ v, x @ v, y @ v
        hpx = un * ux - vn * vx
        hcx = un * vx + vn * ux
        hpy = un * uy - vn * vy
        hcy = un * vy + vn * uy
        return hpx, hcx, hpy, hcy

    def project_wave(self, hp, hc, lamb, beta, polarization=0, **kwargs):
        """
        Project the plus/cross polarizations onto LGWA's two horizontal
        sensing axes.

        `hp`/`hc` are assumed to already be in the SSB frame (as for
        `_LDC_detector`/`_FLR_detector`); `lamb`/`beta`/`polarization`
        are the SSB-frame `eclipticlongitude`/`eclipticlatitude`/
        polarization of the source.

        NOTE: this does NOT call `apply_polarization(hp, hc,
        polarization)` the way `_LDC_detector`/`_FLR_detector` do.
        Those backends' external packages (`lisagwresponse`/
        `fastlisaresponse`) don't accept a polarization angle, so PyCBC
        pre-rotates hp/hc instead. `lgwa_response`'s antenna-pattern
        formula folds the polarization angle directly into the F+/Fx
        combination (via `wave_frame_basis_cartesian(ra, dec, -psi)`),
        so pre-rotating hp/hc here as well would double-count the
        polarization angle.

        Returns
        -------
        dict of pycbc.types.TimeSeries
            Keyed 'LGWA_X'/'LGWA_Y' for the two horizontal sensing axes.
        """
        from pycbc.coordinates import moon as coord_moon

        # SSB-frame ecliptic lon/lat/pol -> ICRS ra/dec/psi (no LDC/LAL
        # flip -- lgwa_response has its own, unrelated polarization
        # convention). `t_geo` (discarded, `_`) is meaningless here:
        # arrival time at the actual LGWA site is computed separately
        # below via t_moon_from_ssb, not via the geocenter.
        _, ra, dec, psi = coord_moon.moon_to_geo(
            t_moon=0.0, longitude_moon=lamb, latitude_moon=beta,
            polarization_moon=polarization, lal_convention=False)

        # Light-travel-time offset (SSB -> LGWA site), evaluated once at
        # a single reference time and applied as a near-constant shift
        # to the whole waveform -- lunar libration/orbital geometry
        # changes on a timescale of days, far slower than the offset
        # would vary across a single signal's duration. This mirrors
        # _LDC_detector's use of a single constant `offset` (though
        # there the offset is a fixed mission-phase constant; here it is
        # computed per-source from the already-built t_moon_from_ssb).
        t_ref = float(hp.start_time)
        delta_t = coord_moon.t_moon_from_ssb(
            t_ref, lamb, beta, self.longitude_site,
            self.latitude_site) - t_ref

        hp = hp.copy()
        hc = hc.copy()
        hp.start_time += delta_t
        hc.start_time += delta_t

        pad = self.cadence
        n, x, y = self._detector_frame(
            hp.start_time - pad, hp.end_time + pad, hp.sample_times.numpy())

        hpx, hcx, hpy, hcy = self._antenna_pattern_factors(
            n, x, y, ra, dec, psi)

        h_x = hp * hpx + hc * hcx
        h_y = hp * hpy + hc * hcy

        return {'LGWA_X': h_x, 'LGWA_Y': h_y}


class _LILA_detector(AbsSpaceDet):
    """
    LILA (Laser Interferometer Lunar Antenna) Observatory, modeled as an
    equilateral triangle of three lunar-surface interferometer stations
    with 40 km arms -- the lunar analogue of the Einstein Telescope's
    layout, and handled here with exactly ET's geometric conventions
    (see `pycbc.coordinates.moon.moon_triangle_sites`).

    Only the Geodesic Displacement Channel (GDC) is modeled: each vertex
    has suspended test masses, so it is an ordinary long-wavelength
    Michelson interferometer with a 60 degree opening angle and its
    response is the usual two-arm difference tensor. The Lunar
    Deformation Channel (LDC), in which the Moon's own elastic normal
    modes are part of the instrument, is a genuinely different response
    function and is not modeled here.

    Unlike `_LGWA_detector`, this backend needs no external package: the
    lunar orientation comes from `lunarsky` (already an optional
    dependency of `pycbc.coordinates.moon`) via the Moon-centred
    Moon-fixed (MCMF) frame, and the antenna-pattern contraction is
    PyCBC's own. Sky position and polarization are converted on entry
    from this module's SSB-ecliptic interface to LAL-convention ICRS
    ra/dec/psi, which is the convention `Detector.antenna_pattern` and
    the tensor construction below both use -- hence
    `lal_convention=True` here, where `_LGWA_detector` needs False for
    `lgwa_response`'s unrelated convention.

    Because the arms are short (40 km) compared with the mid-band
    wavelengths LILA targets (0.1-10 Hz, against a free spectral range
    c/2L = 3.75 kHz), the long-wavelength approximation is used
    throughout and no `single_arm_frequency_response` correction is
    applied.

    Parameters
    ----------
    detector_name : str
        The name of the detector. Accepts any output from
        `get_available_space_detectors`.

    reference_time : float (optional)
        The reference time in seconds of the signal in the SSB frame.
        This is defined such that the detector mission start time
        corresponds to 0. Default None.

    longitude_site, latitude_site : float
        Selenodetic longitude/latitude (radians) of the triangle
        *centroid*, in the same convention as
        `pycbc.coordinates.moon.moon_site_position_ssb`. Required:
        antenna-pattern geometry needs an actual oriented site, and
        unlike arrival time it cannot fall back to the Moon's barycenter.

    orientation : float (optional)
        Bearing, from the centroid, of the first vertex, in radians.
        Default 0.

    arm_length : float (optional)
        Triangle side length in metres. Default 40 km, the LILA
        Observatory baseline.

    height : float (optional)
        Height of the vertices above the reference selenoid, in metres.
        Default 0.

    cadence : float (optional)
        Spacing, in seconds, of the coarse time grid on which the slow
        `lunarsky`/astropy geometry is evaluated before being
        interpolated onto the waveform's sample times. As for
        `_LGWA_detector`, lunar libration evolves over days, so the
        3600 s default is generously fine: measured against exact
        per-sample evaluation it costs 2e-10 in F+/Fx and 25 ps in the
        propagation delay. Pass None to disable interpolation and
        evaluate every sample exactly (much slower).
    """
    # Orthogonal recombination of the three vertex channels, in the same
    # convention `_LDC_detector` uses for LISA's TDI variables (see its
    # `project_wave`), with the vertex channels 1/2/3 playing the role of
    # X/Y/Z there:
    #     A = (Z - X)/sqrt(2), E = (X - 2Y + Z)/sqrt(6),
    #     T = (X + Y + Z)/sqrt(3).
    # Any orthonormal basis of the two-dimensional signal subspace would
    # do -- these three rows are one particular choice among a rotation's
    # worth -- but matching LISA's keeps 'LILA_A' meaning the same
    # combination as 'LISA_A' for anyone reading across the two backends.
    #
    # T is the triangle's null stream: the three vertices' response
    # tensors sum to zero identically (an exact geometric identity for any
    # three sites whose arms lie along the connecting great circles, not a
    # small-triangle approximation), so T carries no GW signal in the
    # long-wavelength limit and is an instrumental/glitch monitor. Under
    # the usual symmetry assumption -- equal noise at each vertex, equal
    # correlation between each pair -- this basis also diagonalizes the
    # noise covariance, as it does for LISA.
    _AET = numpy.array([[-1.0, 0.0, 1.0],
                        [1.0, -2.0, 1.0],
                        [1.0, 1.0, 1.0]]) / numpy.array([[numpy.sqrt(2.0)],
                                                         [numpy.sqrt(6.0)],
                                                         [numpy.sqrt(3.0)]])

    def __init__(self, detector_name, reference_time=None,
                 longitude_site=None, latitude_site=None, orientation=0.0,
                 arm_length=4.0e4, height=0.0, cadence=3600.0, **kwargs):
        super().__init__(detector_name, reference_time, **kwargs)
        assert self.det == 'LILA', (
            'LILAResponse backend only works with the LILA detector')
        if longitude_site is None or latitude_site is None:
            raise ValueError(
                'longitude_site and latitude_site (radians, the triangle '
                'centroid) are required: antenna-pattern geometry needs an '
                'actual oriented site on the Moon, unlike arrival time '
                '(which can default to the Moon\'s barycenter, see '
                'coordinates.moon.moon_site_position_ssb).')

        from pycbc.coordinates.moon import moon_triangle_sites
        from pycbc.detector.ground import body_fixed_detector_tensor

        self.longitude_site = longitude_site
        self.latitude_site = latitude_site
        self.arm_length = arm_length
        self.cadence = cadence
        self.sites = moon_triangle_sites(
            longitude_site, latitude_site, arm_length,
            orientation=orientation, height=height)
        self.channels = ['1', '2', '3']

        # Constant Moon-fixed response tensors, one per vertex. Built with
        # the same body-agnostic core that add_detector_on_earth uses, so
        # the LILA triangle and the ET triangle are literally the same
        # code path modulo which body the site sits on.
        self.responses = {}
        for chan, site in zip(self.channels, self.sites):
            resps, _, _ = body_fixed_detector_tensor(
                site['longitude'], site['latitude'],
                yangle=site['yangle'], xangle=site['xangle'])
            self.responses[chan] = numpy.squeeze(resps[0] - resps[1])

    @property
    def sky_coords(self):
        return 'eclipticlongitude', 'eclipticlatitude'

    def _source_basis_mcmf(self, ra, dec, psi, times):
        """The GW polarization basis vectors (x, y), each shape (3, M),
        rotated from ICRS into the Moon-fixed MCMF frame at `times` (GPS
        seconds).

        This is site-independent -- the ICRS -> MCMF rotation depends
        only on time -- so it is evaluated once and shared by all three
        vertices, each of which then only needs a cheap contraction with
        its own constant response tensor.
        """
        from astropy import units as apy_units
        from astropy.coordinates import (CartesianRepresentation, ICRS,
                                         SkyCoord)
        from astropy.time import Time
        from lunarsky import MCMF

        obstimes = Time(times, format='gps')
        n_time = len(times)

        ca, sa = numpy.cos(ra), numpy.sin(ra)
        cd, sd = numpy.cos(dec), numpy.sin(dec)
        # ICRS sky basis: e_ra toward increasing RA, e_dec toward
        # increasing Dec.
        basis_icrs = [numpy.array([-sa, ca, 0.0]),
                      numpy.array([-sd * ca, -sd * sa, cd])]

        # Astropy frame transforms translate between frame origins, so a
        # finite-distance point would pick up a parallax term. Transform
        # both a displaced point and the origin, then subtract, to isolate
        # the pure rotation. The baseline length is NOT arbitrary in
        # floating point: both endpoints pass through the Moon's ~1.5e11 m
        # barycentric position, whose ulp is ~3e-5 m, so a 1 m baseline
        # would leave ~3e-5 rad of direction error after the subtraction.
        # 1e6 m puts the residual at astropy's own ~2e-8 rad floor.
        baseline = 1.0e6
        zeros = numpy.zeros(n_time) * apy_units.m
        origin = SkyCoord(CartesianRepresentation(x=zeros, y=zeros, z=zeros),
                          frame=ICRS()).transform_to(MCMF(obstime=obstimes))
        origin_xyz = origin.cartesian.xyz.to_value(apy_units.m)

        rotated = []
        for vec in basis_icrs:
            end = SkyCoord(CartesianRepresentation(
                x=numpy.full(n_time, vec[0]) * baseline * apy_units.m,
                y=numpy.full(n_time, vec[1]) * baseline * apy_units.m,
                z=numpy.full(n_time, vec[2]) * baseline * apy_units.m),
                frame=ICRS()).transform_to(MCMF(obstime=obstimes))
            xyz = end.cartesian.xyz.to_value(apy_units.m) - origin_xyz
            rotated.append(xyz / numpy.linalg.norm(xyz, axis=0))

        e_ra, e_dec = rotated
        # Re-orthonormalize: the two transforms are independent, so their
        # results are only orthogonal to within the transform's own error.
        e_dec = e_dec - e_ra * numpy.sum(e_ra * e_dec, axis=0)
        e_dec = e_dec / numpy.linalg.norm(e_dec, axis=0)

        cpsi, spsi = numpy.cos(psi), numpy.sin(psi)
        return (cpsi * e_ra + spsi * e_dec, -spsi * e_ra + cpsi * e_dec)

    def _grid(self, t_start, t_end):
        """Coarse evaluation grid spanning [t_start, t_end], padded by one
        cadence at each end so the interpolants never extrapolate."""
        if self.cadence is None:
            return None
        n_points = max(
            4, int(numpy.ceil((t_end - t_start) / self.cadence)) + 4)
        return numpy.linspace(t_start - self.cadence, t_end + self.cadence,
                              n_points)

    def antenna_pattern(self, ra, dec, psi, times):
        """F+ and Fx for each of the three vertices, at the given GPS
        `times`.

        Parameters
        ----------
        ra, dec, psi : float
            ICRS right ascension, declination and LAL-convention
            polarization angle of the source, in radians.
        times : numpy.array
            GPS times at which to evaluate the patterns.

        Returns
        -------
        dict
            Channel name ('1', '2', '3') -> (fplus, fcross), each an
            array of len(times).
        """
        from scipy.interpolate import CubicSpline

        times = numpy.asarray(times, dtype=numpy.float64)
        grid = self._grid(float(times[0]), float(times[-1]))
        eval_times = times if grid is None else grid

        x_pol, y_pol = self._source_basis_mcmf(ra, dec, psi, eval_times)

        out = {}
        for chan in self.channels:
            resp = self.responses[chan]
            dx, dy = resp @ x_pol, resp @ y_pol
            fplus = numpy.sum(x_pol * dx - y_pol * dy, axis=0)
            fcross = numpy.sum(x_pol * dy + y_pol * dx, axis=0)
            if grid is not None:
                fplus = CubicSpline(grid, fplus)(times)
                fcross = CubicSpline(grid, fcross)(times)
            out[chan] = (fplus, fcross)
        return out

    def _vertex_positions(self, times):
        """Barycentric-ecliptic positions of the three vertices at the given
        GPS `times`, shape (3 vertices, 3 components, M), in metres.

        A vectorised counterpart to
        `pycbc.coordinates.moon.moon_site_position_ssb`, which is
        scalar-only (it reshapes its result to (3, 1)). The frame matches
        that function's, so these positions may be contracted directly
        with a propagation vector from
        `coordinates.space.localization_to_propagation_vector`.
        """
        from astropy import units as apy_units
        from astropy.coordinates import ICRS
        from astropy.time import Time
        from lunarsky import MoonLocation

        from pycbc.coordinates.space_orbit import (
            _icrs_to_ecliptic_rotation_matrix)

        obstimes = Time(numpy.atleast_1d(times), format='gps')
        rotation = _icrs_to_ecliptic_rotation_matrix()

        out = numpy.empty((3, 3, len(obstimes)))
        for j, site in enumerate(self.sites):
            loc = MoonLocation.from_selenodetic(
                lon=site['longitude'] * apy_units.rad,
                lat=site['latitude'] * apy_units.rad,
                height=site['height'] * apy_units.m)
            icrs = loc.get_mcmf(obstimes).transform_to(ICRS())
            xyz = numpy.vstack([
                icrs.cartesian.x.to_value(apy_units.m),
                icrs.cartesian.y.to_value(apy_units.m),
                icrs.cartesian.z.to_value(apy_units.m)])
            out[j] = rotation @ xyz
        return out

    def vertex_delays(self, lamb, beta, times):
        """Light-travel delay from the SSB to each vertex, in seconds, at the
        given *detector* GPS `times`.

        This is the explicit, detector-time form
        .. math::
            \Delta t_i(t) = \hat{k}\cdot\bm{r}_i(t)/c ,
        in which the position is evaluated at the known detector time, so
        no root-finding is involved. It is the direction in which the
        mapping is explicit: going the other way, from a known SSB time to
        the unknown arrival time, puts the unknown inside
        :math:`\bm{r}(\cdot)` and requires the implicit solve performed by
        `coordinates.moon.t_moon_from_ssb` (which `_LGWA_detector` uses).

        The delay is evaluated on the coarse `self.cadence` grid and cubic
        interpolated, as the antenna patterns are; it varies on orbital
        timescales, and a 3600 s grid reproduces it to well under a
        nanosecond.

        Returns
        -------
        dict
            Channel name -> delay array of len(times), in seconds.
        """
        from scipy.interpolate import CubicSpline

        from astropy.constants import c as speed_of_light
        from pycbc.coordinates.space import (
            localization_to_propagation_vector)

        times = numpy.asarray(times, dtype=numpy.float64)
        grid = self._grid(float(times[0]), float(times[-1]))
        eval_times = times if grid is None else grid

        k = numpy.asarray(
            localization_to_propagation_vector(lamb, beta, use_astropy=False),
            dtype=numpy.float64).reshape(3)
        positions = self._vertex_positions(eval_times)

        out = {}
        for j, chan in enumerate(self.channels):
            delay = (k @ positions[j]) / speed_of_light.value
            if grid is not None:
                delay = CubicSpline(grid, delay)(times)
            out[chan] = delay
        return out

    def _window_epoch(self, lamb, beta, t_ref):
        """Detector time at which to start the output series.

        This only selects *which window* of detector time is emitted; the
        delays applied within it are exact at every sample, so an error
        here costs a sliver of signal at the segment edges and nothing
        else. Two explicit evaluations are therefore ample, and no
        implicit solve is needed: the first uses the Moon's position at
        the SSB epoch, the second corrects it for the Moon's motion during
        the light-travel time (about 11 ms of the 475 s total).
        """
        delay = self.vertex_delays(lamb, beta, numpy.array([t_ref, t_ref]))
        guess = float(numpy.mean([delay[c][0] for c in self.channels]))
        refined = self.vertex_delays(
            lamb, beta, numpy.array([t_ref + guess, t_ref + guess]))
        return float(numpy.mean([refined[c][0] for c in self.channels]))

    def project_wave(self, hp, hc, lamb, beta, polarization=0,
                     include_aet=False, **kwargs):
        """
        Project the plus/cross polarizations onto the three GDC vertex
        interferometers of the LILA triangle.

        `hp`/`hc` are assumed to already be in the SSB frame (as for the
        other backends in this module); `lamb`/`beta`/`polarization` are
        the SSB-frame `eclipticlongitude`/`eclipticlatitude`/polarization
        of the source.

        The output grid is a *detector*-time grid, which is what makes the
        propagation delay explicit (see `vertex_delays`). For each vertex,
        .. math::
            h_i(t) = F^{(i)}_+(t)\,h_+(t - \Delta t_i(t))
                   + F^{(i)}_\times(t)\,h_\times(t - \Delta t_i(t)),
        the minus sign following from the plane-wave form
        :math:`h(t,\bm{x}) = f(t - \hat{k}\cdot\bm{x}/c)`: a vertex
        further along the propagation direction receives later, so at a
        given detector time it displays an earlier barycentric sample.

        Unlike `_LDC_detector`/`_LGWA_detector`, which apply a single
        constant epoch shift, :math:`\Delta t_i` is evaluated at every
        sample. That is not a refinement but a requirement here: the
        Moon's barycentric motion is dominated by the Earth's ~30 km/s
        orbit, not by libration, so the delay drifts by ~44 ms across an
        hour of signal and ~1 s across a day -- 10 cycles at 10 Hz, which
        would destroy phase coherence for exactly the long mid-band
        signals LILA is built to see.

        All three channels share one grid. Per-vertex delays differ by at
        most L/c = 133 us and are applied by interpolation, never by
        relabelling epochs, since a coherent three-channel analysis -- and
        the A/E/T recombination above all -- requires a common grid.

        Parameters
        ----------
        include_aet : bool (optional)
            Also return the orthogonal (A, E, T) recombination of the
            three vertex channels, where T is the null stream. Default
            False.

        Returns
        -------
        dict of pycbc.types.TimeSeries
            Keyed 'LILA_1'/'LILA_2'/'LILA_3', plus 'LILA_A'/'LILA_E'/
            'LILA_T' if `include_aet` is set.
        """
        from scipy.interpolate import CubicSpline

        from pycbc.coordinates import moon as coord_moon

        # SSB-frame ecliptic lon/lat/pol -> LAL-convention ICRS
        # ra/dec/psi, which is what the response tensors expect. `t_geo`
        # is discarded: arrival times come from vertex_delays, not via the
        # geocenter.
        _, ra, dec, psi = coord_moon.moon_to_geo(
            t_moon=0.0, longitude_moon=lamb, latitude_moon=beta,
            polarization_moon=polarization, lal_convention=True)

        t_ref = float(hp.start_time)
        delta_t = float(hp.delta_t)
        n_sample = len(hp)

        epoch_delay = self._window_epoch(lamb, beta, t_ref)
        times = t_ref + epoch_delay + numpy.arange(n_sample) * delta_t

        delays = self.vertex_delays(lamb, beta, times)
        patterns = self.antenna_pattern(ra, dec, psi, times)

        # All time arithmetic below is done in seconds *relative to
        # t_ref*, never on absolute GPS times. At t ~ 1.4e9 s the spacing
        # between doubles is 0.24 us, which would quantise the 133 us
        # inter-vertex structure to 0.2%; the delays themselves are only
        # ~475 s, where the spacing is 6e-14 s.
        tau = numpy.arange(n_sample) * delta_t
        hp_spline = CubicSpline(tau, hp.numpy(), extrapolate=False)
        hc_spline = CubicSpline(tau, hc.numpy(), extrapolate=False)

        out = {}
        for chan in self.channels:
            query = tau + (epoch_delay - delays[chan])
            hp_v = numpy.nan_to_num(hp_spline(query))
            hc_v = numpy.nan_to_num(hc_spline(query))
            fplus, fcross = patterns[chan]
            out['LILA_' + chan] = TimeSeries(
                fplus * hp_v + fcross * hc_v, delta_t=delta_t,
                epoch=times[0], copy=False)

        if include_aet:
            stack = numpy.vstack([out['LILA_' + c].numpy()
                                  for c in self.channels])
            aet = self._AET @ stack
            for name, row in zip(['A', 'E', 'T'], aet):
                out['LILA_' + name] = TimeSeries(
                    row, delta_t=delta_t, epoch=times[0], copy=False)
        return out


class _Generic_detector(AbsSpaceDet):
    """
    Placeholder backend for space-borne detectors that do not yet have a
    native single-link response / TDI implementation in PyCBC (currently
    Taiji and TianQin). This exists so that the detector can be named and
    constructed (e.g. to exercise the orbit/coordinate machinery in
    `pycbc.coordinates.space_orbit`), while making it explicit that
    `project_wave` is not yet implemented, rather than silently reusing the
    LISA-specific LDC/FLR backends.

    Parameters
    ----------
    detector_name : str
        The name of the detector. Accepts any output from
        `get_available_space_detectors`.

    reference_time : float (optional)
        The reference time in seconds of the signal in the SSB frame. This is
        defined such that the detector mission start time corresponds to 0.
        Default None.
    """
    def __init__(self, detector_name, reference_time=None, **kwargs):
        super().__init__(detector_name, reference_time, **kwargs)

    @property
    def sky_coords(self):
        return 'eclipticlongitude', 'eclipticlatitude'

    def project_wave(self, hp, hc, lamb, beta, *args, **kwargs):
        raise NotImplementedError(
            f'Native single-link response and TDI generation for '
            f'{self.det} are not yet implemented in PyCBC. This backend '
            'currently only provides discovery/registration of the '
            'detector; see `pycbc.coordinates.space_orbit` for the '
            'orbit/coordinate machinery this response will build on.')


_backends = {'LISA': {'LDC': _LDC_detector,
                      'FLR': _FLR_detector,
                     },
             'Taiji': {'Generic': _Generic_detector},
             'TianQin': {'Generic': _Generic_detector},
             'LGWA': {'Generic': _Generic_detector,
                      'LGWAResponse': _LGWA_detector},
             'LILA': {'Generic': _Generic_detector,
                      'LILAResponse': _LILA_detector},
            }

class SpaceDetector(AbsSpaceDet):
    """
    Space-based detector.

    Parameters
    ----------
    detector_name : str
        The name of the detector. Accepts any output from
        `get_available_space_detectors`.
    
    reference_time : float (optional)
        The reference time in seconds of the signal in the SSB frame. This is
        defined such that the detector mission start time corresponds to 0.
        Default None.

    backend : str (optional)
        The backend architecture to use for generating TDI. Accepts 'LDC'
        or 'FLR'. Default 'LDC'.
    """
    def __init__(self, detector_name, reference_time=None, backend='LDC',
                 **kwargs):
        super().__init__(detector_name, reference_time, **kwargs)
        if backend in _backends[self.det].keys():
            c = _backends[self.det][backend]
            self.backend = c(detector_name, reference_time, **kwargs)
        else:
            raise ValueError(f'Detector {self.det} does not support backend ',
                             f'{backend}.This detector accepts: '
                             f'{_backends[self.det].keys()}')

    @property
    def sky_coords(self):
        return self.backend.sky_coords

    def get_links(self, hp, hc, lamb, beta, *args, **kwargs):
        return self.backend.get_links(hp, hc, lamb, beta, *args, **kwargs)

    def project_wave(self, hp, hc, lamb, beta, *args, **kwargs):
        return self.backend.project_wave(hp, hc, lamb, beta, *args, **kwargs)


__all__ = ['get_available_space_detectors', 'SpaceDetector',
           '_space_detectors',]