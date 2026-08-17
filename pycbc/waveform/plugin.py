""" Utilities for handling waveform plugins
"""


import logging

logger = logging.getLogger('pycbc.waveform.plugin')

def add_custom_waveform(approximant, function, domain,
                        sequence=False, has_det_response=False,
                        force=False,):
    """ Make custom waveform available to pycbc

    Parameters
    ----------
    approximant : str
        The name of the waveform
    function : function
        The function to generate the waveform
    domain : str
        Either 'frequency' or 'time' to indicate the domain of the waveform.
    sequence : bool, False
        Function evaluates waveform at only chosen points (instead of a
        equal-spaced grid).
    has_det_response : bool, False
        Check if waveform generator has built-in detector response.
    """
    from pycbc.waveform.waveform import (cpu_fd, cpu_td, fd_sequence,
                                         fd_det, fd_det_sequence,
                                         td_fd_waveform_transform)

    used = RuntimeError("Can't load plugin waveform {}, the name is"
                        " already in use.".format(approximant))

    if domain == 'time':
        if not force and (approximant in cpu_td):
            raise used
        cpu_td[approximant] = function
        td_fd_waveform_transform(approximant)
    elif domain == 'frequency':
        if sequence:
            if not has_det_response:
                if not force and (approximant in fd_sequence):
                    raise used
                fd_sequence[approximant] = function
            else:
                if not force and (approximant in fd_det_sequence):
                    raise used
                fd_det_sequence[approximant] = function
        else:
            if not has_det_response:
                if not force and (approximant in cpu_fd):
                    raise used
                cpu_fd[approximant] = function
            else:
                if not force and (approximant in fd_det):
                    raise used
                fd_det[approximant] = function
    else:
        raise ValueError("Invalid domain ({}), should be "
                         "'time' or 'frequency'".format(domain))


def add_length_estimator(approximant, function):
    """ Add length estimator for an approximant

    Parameters
    ----------
    approximant : str
        Name of approximant
    function : function
        A function which takes kwargs and returns the waveform length
    """
    from pycbc.waveform.waveform import _filter_time_lengths
    if approximant in _filter_time_lengths:
        raise RuntimeError("Can't load length estimator {}, the name is"
                           " already in use.".format(approximant))
    _filter_time_lengths[approximant] = function

    from pycbc.waveform.waveform import td_fd_waveform_transform
    td_fd_waveform_transform(approximant)


def add_end_frequency_estimator(approximant, function):
    """ Add end frequency estimator for an approximant

    Parameters
    ----------
    approximant : str
        Name of approximant
    function : function
        A function which takes kwargs and returns the waveform end frequency
    """
    from pycbc.waveform.waveform import _filter_ends
    if approximant in _filter_ends:
        raise RuntimeError("Can't load freqeuncy estimator {}, the name is"
                           " already in use.".format(approximant))

    _filter_ends[approximant] = function

from importlib.metadata import entry_points


def _load_plugin_group(group, register):
    """Register each plugin in an entry-point group, skipping broken ones.

    These come from separately-installed packages; a stale entry point should
    not make importing pycbc.waveform raise and break every command.
    """
    for plugin in entry_points(group=group):
        try:
            loaded = plugin.load()
        except Exception as exc:
            logger.warning("Skipping waveform plugin %r from %r: failed to "
                           "import (%s: %s)",
                           plugin.name, group, type(exc).__name__, exc)
            continue
        register(plugin.name, loaded)


def retrieve_waveform_plugins():
    """ Process external waveform plugins
    """
    _load_plugin_group(
        'pycbc.waveform.fd',
        lambda name, f: add_custom_waveform(name, f, 'frequency'))

    _load_plugin_group(
        'pycbc.waveform.fd_det',
        lambda name, f: add_custom_waveform(name, f, 'frequency',
                                            has_det_response=True))

    _load_plugin_group(
        'pycbc.waveform.fd_sequence',
        lambda name, f: add_custom_waveform(name, f, 'frequency',
                                            sequence=True))

    _load_plugin_group(
        'pycbc.waveform.fd_det_sequence',
        lambda name, f: add_custom_waveform(name, f, 'frequency',
                                            sequence=True,
                                            has_det_response=True))

    _load_plugin_group(
        'pycbc.waveform.td',
        lambda name, f: add_custom_waveform(name, f, 'time'))

    _load_plugin_group('pycbc.waveform.length', add_length_estimator)

    _load_plugin_group('pycbc.waveform.end_freq',
                       add_end_frequency_estimator)
