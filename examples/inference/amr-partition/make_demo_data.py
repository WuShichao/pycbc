#!/usr/bin/env python
"""Write the synthetic data this example uses.

The example is self-contained: it ships no research archive, no real event and
no absolute path.  Running this script writes

* ``posteriors.npz`` -- a 16-cell partition over four chart coordinates,
* ``physical-map.json`` -- the projection from those chart coordinates to
  ``mchirp`` and ``q``,

into ``amr-demo-data/`` next to this file.  ``amr.ini`` reads both files.

The partition is a 2 x 2 x 2 x 2 grid centred on the origin.  The Bayes factor
of a cell is ``exp(-|eta|^2 / 2)``, so the posterior is a broad Gaussian; it is
synthetic and has no physical meaning beyond exercising the file format.
"""

import argparse
import json
import os

import numpy


def physical_map():
    """The projection from the four chart coordinates to two physical ones.

    ``lam`` are the eigenvalues of the chart and ``E`` its eigenvectors.  The
    top-left block is a rescaling by ``1/sqrt(lam)``; here ``E`` is the identity
    so the projection is diagonal: ``xi_0 = eta_0 / 2`` and ``xi_1 = eta_1``.
    ``mchirp`` is the reference plus ``scale * xi``; ``q`` is reciprocal, so the
    stored column is ``1 / (reference + scale * xi)`` in PyCBC's convention.
    """
    parameters = [
        {"name": "mchirp", "xi_index": 0, "kind": "identity",
         "reference": 10.0, "scale": 0.1, "units": "Msun"},
        {"name": "q", "xi_index": 1, "kind": "reciprocal",
         "reference": 2.0, "scale": 0.01, "units": "dimensionless"},
    ]
    return {
        "eta_columns": [0, 1, 2, 3],
        "eta_static": {"4": 0.0, "5": 0.0},
        "lam": [4.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "E": numpy.eye(6).tolist(),
        "parameters": parameters,
        # the injection the map is checked against on load:
        # M @ eta = xi, with M = diag(1/2, 1, 1, 1, 1, 1)
        "injection": {
            "xi": [0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
            "eta": [0.5, 0.25, 0.0, 0.0, 0.0, 0.0],
        },
    }


def partition():
    """A 16-cell Gaussian partition and the chart point of its injection."""
    axes = (-0.5, 0.5)
    eta = numpy.array([[a, b, c, d, 0.0, 0.0]
                       for a in axes for b in axes
                       for c in axes for d in axes])
    widths = numpy.ones_like(eta)
    lnb = -0.5 * numpy.sum(eta ** 2, axis=1)
    truth = numpy.array([0.5, 0.25, 0.0, 0.0, 0.0, 0.0])
    return eta, widths, lnb, truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "amr-demo-data"),
        help="directory to write the two files into")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    eta, widths, lnb, truth = partition()
    numpy.savez(os.path.join(args.output_dir, "posteriors.npz"),
                eta_L6=eta, w_L6=widths, lnB_L6=lnb, truth=truth)
    with open(os.path.join(args.output_dir, "physical-map.json"), "w") as f:
        json.dump(physical_map(), f, indent=1, sort_keys=True)

    print("wrote {} cells to {}".format(len(eta), args.output_dir))


if __name__ == "__main__":
    main()
