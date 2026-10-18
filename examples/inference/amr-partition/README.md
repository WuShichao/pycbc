# The AMR cell partition as a PyCBC inference sampler

`amr.ini` selects the adapter in `pycbc/inference/sampler/amr.py` with
`name = amr` and reads the weighted cell partition that the research AMR
prototype wrote:

```
results/2026-10-18/amr_sampler/posteriors.npz
```

in the companion research repository.  See also
`pycbc/inference/io/amr.py` for the file format and the statement of what the
object is.

## What this is not

This is a **deterministic grid quadrature on a fixed partition**, not a
stochastic sampler.  Each entry is a cell centre with a volume and an
amplitude-marginalised Bayes factor, not a draw from a chain.  There is no
burn-in, no autocorrelation and **no evidence error estimate**:
`dlog_evidence` is written as `nan`.  MCMC diagnostics do not apply.

The PyCBC model in the config supplies the variable-parameter names; its
likelihood is not evaluated.  The likelihood of this problem is the research
Bayes factor, which is not a PyCBC model.

## Run

The `pycbc` in the environment may be a site-packages copy, so put the checkout
first on `PYTHONPATH`:

```bash
cd /home/nightwing/src/pycbc
PYTHONPATH=/home/nightwing/src/pycbc \
  python bin/inference/pycbc_inference \
    --config-file examples/inference/amr-partition/amr.ini \
    --output-file /tmp/amr.hdf --force --verbose
```

This takes about a second: the physics already ran.  It writes 33,065 weighted
cells (refinement level 6) and `log_evidence = 111.8022`, the Riemann sum over
the covered partition (see the note listed below for the constant offset to
the evidence normalised by the exact prior ellipsoid).

To run the research AMR live instead, set `enable-research = true` and point
`amr-root` at the companion repository.  That path needs `cupy`, a CUDA GPU and
about five minutes, and it loads the research implementation by file path
rather than copying it.

## Plot

```bash
PYTHONPATH=/home/nightwing/src/pycbc \
  python bin/inference/pycbc_inference_plot_posterior \
    --input-file /tmp/amr.hdf \
    --output-file /tmp/amr_posterior.png \
    --parameters eta_0 eta_1 eta_2 eta_3 \
    --plot-marginal --plot-scatter --plot-density \
    --expected-parameters eta_0:-0.07597649 eta_1:0.00436479 \
                          eta_2:0.00474236 eta_3:-0.01198615 \
    --nsamples 10000
```

`--raw-samples` reads the weighted partition itself (one row per cell);
`--nsamples N` sets the size of the equal-weight posterior that the default
read resamples from `logwt`.

## Physical parameters

The four `eta` axes are a rotation and a rescaling of the four physical
offsets, so they are not physical parameters and cannot be relabelled as such.
`physical-map = examples/inference/amr-partition/physical-map.json` gives the
exact projection,

```
xi = E diag(1/sqrt(lam)) eta ,     p = p_ref + sigma_p xi_p ,
```

and the adapter writes `mchirp`, `q`, `spin1z` and `spin2z` beside the `eta`
columns.  The map carries the geometry (`lam`, `E`), the four reference values,
the four shape scales and the recorded injection; on load it is checked by
requiring `E diag(1/sqrt(lam)) eta_injection` to return the recorded
`xi_injection`, and by re-reading the geometry file it names.

`q` is the one convention that differs from the research chart, which uses
`q_proj = m2/m1 <= 1`: PyCBC's `q` is `m1/m2 >= 1`, so the stored column is
`1/(q_ref + sigma xi)`.  Percentiles transform exactly because the map is
monotone; the linear coordinate is `1/q`.

Plotting the physical columns:

```bash
PYTHONPATH=/home/nightwing/src/pycbc \
  python bin/inference/pycbc_inference_plot_posterior \
    --input-file /tmp/amr.hdf \
    --output-file /tmp/amr_posterior_physical.png \
    --parameters mchirp q spin1z spin2z \
    --plot-marginal --plot-scatter --plot-contours \
    --contour-percentiles 68 90 95 --z-arg snr:SNR \
    --expected-parameters mchirp:31.83141195 q:2.08874494 \
                          spin1z:-0.14437378 spin2z:-0.79002668
```

## Prior

The four stiff eta coordinates have a flat prior over the ellipsoid
`eta^T M eta <= 1`; `amr.ini` expresses it as uniform boxes plus the
`[constraint-region]` quadratic.  The amplitude/orientation prior and the
arrival-time prior are integrated inside the Bayes factor and have no sample
column, so they cannot be written as `pycbc.distributions` priors.  The note
`notes/2026/10/2026-10-18-amr-sampler-pycbc-integration.md` in the research
repository has the full mapping.
