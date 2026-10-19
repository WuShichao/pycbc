# The AMR cell partition as a PyCBC inference sampler

`amr.ini` selects the adapter in `pycbc/inference/sampler/amr.py` with
`name = amr`.  The adapter reads a weighted adaptive-mesh partition, written
here by `make_demo_data.py`, and writes it in PyCBC's samples format.

This is a **deterministic grid quadrature on a fixed partition**, not a
stochastic sampler.  Each entry is a cell centre with a volume and a
Bayes factor, not a draw from a chain.  There is no burn-in, no
autocorrelation and **no evidence error estimate**: `dlog_evidence` is
written as `nan`, and MCMC diagnostics do not apply.  The PyCBC model supplies
the variable-parameter names; its likelihood is not evaluated.

## Run

From the top of the checkout:

```bash
python examples/inference/amr-partition/make_demo_data.py
pycbc_inference \
    --config-file examples/inference/amr-partition/amr.ini \
    --output-file /tmp/amr-demo.hdf --force --verbose
```

The first command writes two files into `amr-demo-data/` beside this README.
The second is quick: the partition has 16 cells and the adapter computes the
cell volumes and posteriors directly, with no sampling.

```bash
pycbc_inference_plot_posterior \
    --input-file /tmp/amr-demo.hdf \
    --output-file /tmp/amr-demo.png \
    --parameters eta_0 eta_1 eta_2 eta_3 mchirp q \
    --plot-marginal --plot-scatter
```

The file holds the four `eta` cells with `loglikelihood`, `logprior`, `logwt`
and `snr`, and, because `physical-map` is set, the same posterior expressed as
`mchirp` and `q`.

## Files

* `make_demo_data.py` writes `amr-demo-data/posteriors.npz` (the partition:
  `eta_L6`, `w_L6`, `lnB_L6` and the injection's `truth` chart point) and
  `amr-demo-data/physical-map.json`.
* `amr.ini` names the sampler, the partition and the map.
* `amr-demo-data/` is generated and is ignored by git.

## The physical map

The four `eta` axes are a rotation and a rescaling of physical offsets, so they
are not physical parameters and cannot simply be relabelled as such.  The map
records the exact projection,

```
xi = E diag(1/sqrt(lam)) eta ,     p = p_ref + sigma_p xi_p ,
```

as a JSON object with these fields:

* `lam` (six eigenvalues) and `E` (six orthogonal eigenvectors, as columns) --
  the chart the partition refines;
* `eta_columns` -- which columns of `eta` are sampled;
* `eta_static` -- the values of the unsampled columns, here zero;
* `parameters` -- one entry per physical parameter, with `name`, `xi_index`,
  `reference`, `scale` and `kind` (`identity`, or `reciprocal` for PyCBC's
  `q = m1/m2 >= 1` when the map's linear coordinate is `q_proj = m2/m1`);
* `injection` -- the recorded `xi` and `eta` of the problem, which the map must
  reproduce on load (`E diag(1/sqrt(lam)) eta == xi`), so a stale or
  mismatched map is refused.

Only the fields this example needs are present; a map for a real problem adds
the same fields with its own geometry.  The two unsampled chart directions are
kept because their `1/sqrt(lam)` is large and they carry order-unity pieces of
`xi`; they are not the sky.

`q` is the one convention that differs: `q = 1/(q_ref + sigma xi)`, which is
monotone, so percentiles transform exactly, but the column is not linear in
`eta`.
