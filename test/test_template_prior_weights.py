"""TemplatePriorWeights: a prior from a configuration as a density over a
search's parameters, swappable by its configuration."""
import numpy

from pycbc.population.template_prior_weights import TemplatePriorWeights
from pycbc.workflow import WorkflowConfigParser

FLAT = """
[variable_params]
mchirp =
q =
spin1z =
spin2z =

[prior-mchirp]
name = uniform
min-mchirp = 20
max-mchirp = 40

[prior-q]
name = uniform
min-q = 1
max-q = 4

[prior-spin1z+spin2z]
name = uniform
min-spin1z = -0.9
max-spin1z = 0.9
min-spin2z = -0.9
max-spin2z = 0.9
"""

PEAKED = FLAT.replace("""[prior-mchirp]
name = uniform
min-mchirp = 20
max-mchirp = 40""", """[prior-mchirp]
name = gaussian
mchirp_mean = 30
mchirp_var = 4
min-mchirp = 20
max-mchirp = 40""")

PARAMS = ['mchirp', 'q', 'spin1z', 'spin2z']


def weights(text):
    cp = WorkflowConfigParser()
    cp.read_string(text)
    return TemplatePriorWeights.from_config(cp, PARAMS, size=20000)


def test_a_flat_prior_has_its_volume_density_inside():
    w = weights(FLAT)
    inside = numpy.array([[28.0, 2.5, 0.0, 0.0], [32.0, 2.0, 0.2, -0.3]])
    want = -numpy.log(20 * 3 * 1.8 ** 2)
    numpy.testing.assert_allclose(w.logpdf(inside), want, atol=0.15)


def test_swapping_the_configuration_swaps_the_population():
    flat, peaked = weights(FLAT), weights(PEAKED)
    near, far = [30.0, 2.5, 0.0, 0.0], [23.0, 2.5, 0.0, 0.0]
    ratio = numpy.diff(peaked.logpdf(numpy.array([far, near])))[0]
    # the gaussian of variance 4 between 30 and 23: 49 / 8, less the kernel's
    # smoothing
    assert 4.5 < ratio < 49 / 8.0 + 0.3
    assert abs(numpy.diff(flat.logpdf(numpy.array([far, near])))[0]) < 0.3
    d = dict(zip(PARAMS, numpy.array([near]).T))
    numpy.testing.assert_allclose(peaked.logpdf(d), peaked.logpdf([near]))
