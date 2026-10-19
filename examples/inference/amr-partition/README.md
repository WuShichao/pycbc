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

This takes a few minutes: the physics already ran, and the new work is reading
the archived sufficient statistics, rebuilding the per-(cell, lag, node)
statistic and drawing the joint over (lag, node, u).  The pass streams all
2,116,160 (cell, lag) rows in chunks, so its memory stays at a few hundred
megabytes.  It writes 33,065 weighted cells (refinement level 6) and the Riemann
sum evidence over the covered partition (see the note listed below for the
constant offset to the evidence normalised by the exact prior ellipsoid).

The demo config also sets `sufficient-file`, so the file holds one joint draw
per cell in `inclination`, `polarization`, `delta_tc`, `amplitude_u`,
`distance` and `distance_ratio` beside the `eta` and physical columns.  With
`amplitude-ceiling = 5` and `amplitude-power = 4` (the registered pair) the
stored cell Bayes factors are used unchanged; with the default `20` they are
recomputed from the archive at the wider prior ceiling.  The two files are
directly comparable:

```bash
PYTHONPATH=/home/nightwing/src/pycbc \
  python bin/inference/pycbc_inference \
    --config-file examples/inference/amr-partition/amr.ini \
    --config-overrides sampler-amr:amplitude-ceiling:5 \
                        sampler-amr:amplitude-power:4 \
    --output-file /tmp/amr_ceiling5.hdf --force
```

To run the research AMR live instead, set `enable-research = true`, remove
`sufficient-file` (the live path produces no archive to restore from), and point
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

and the adapter writes `mchirp`, `q`, `spin1z`, `spin2z`,
`eclipticlongitude` and `eclipticlatitude` beside the `eta` columns.  The map
carries the geometry (`lam`, `E`), the reference values, the shape scales and
the recorded injection; on load it is checked by requiring
`E diag(1/sqrt(lam)) eta_injection` to return the recorded `xi_injection`, and
by re-reading the geometry file it names.  The two sky columns are the chart
image of the four refined coordinates, not independent sky measurements: the
partition holds one cell across the sky.

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

## Recovered columns

The orientation (16 Gauss-Legendre nodes in `cos(iota)` times 8 in
`polarization`), the arrival time (the lag grid) and the amplitude are reduced
to a scalar inside the cell Bayes factor.  With `sufficient-file`, the adapter
rebuilds the per-(cell, lag, node) statistic and draws **one sample per cell
from the joint** `p(lag, orientation, u | cell)`, writing:

| column | units | meaning | resolution |
|---|---|---|---|
| `inclination` | rad | `arccos(cos iota)` of the drawn fine node | 256 uniform `cos(iota)` nodes, spacing 0.0078125 |
| `polarization` | rad | `psi` of the same fine node, on `[0, pi/2)` | 64 uniform `psi` nodes, spacing 0.02454 rad |
| `delta_tc` | s | the drawn arrival-time lag | enumerated on the 142.678 s lag grid; not interpolated |
| `amplitude_u` | dimensionless | `u = A/A_min` of the drawn amplitude | continuous on the 2001-node `u` grid |
| `distance` | Mpc | `d = K/u`, the same draw as a luminosity distance | `K/u` on the same grid |
| `distance_ratio` | dimensionless | `d/d_best = u_best/u` of the same draw | `u_best/u` on the same grid |

The three axes are treated differently because they are different kinds of
object.  The orientation is **interpolated**: `Lbar` is smooth in
`(cos iota, psi)` and the search already forms the 16 x 8 node values, so a
not-a-knot cubic spline in `cos iota` and a periodic cubic spline of period
`pi/2` in `psi` give the fine 256 x 64 conditional at no extra likelihood cost
(measured interpolation error 1.27e-2 nats median).  The drawn node is
dithered uniformly inside its cell so the column is continuous; the
resolution is still the 256 x 64 grid and its spacing is recorded.  The
amplitude is
**sampled from its own integrand** at the selected fine node, which is what
keeps `inclination` correlated with `amplitude_u`.  The arrival time is
**enumerated, not interpolated**: a local quadratic through the retained lags
misses its held-out lag by 7.87 nats median, so no interpolant in lag is valid
here; the fix is oversampling the transform (an FFT oversampling factor), not
smoothing the lag grid.  Set `orientation-interpolation = false` to restore the
old enumerated orientation draw.

Every recovered quantity states its own resolution in the file metadata:
`amr_inclination_resolution`, `amr_polarization_resolution`,
`amr_amplitude_resolution`, `amr_distance_resolution`,
`amr_delta_tc_resolution`, `amr_arrival_time_grid_s` and the measured
`amr_orientation_interpolation_error_*`.

The draw factorises as `p(lag, orientation | cell) p(u | lag, orientation,
cell)`: the pair is weighted by the orientation prior times the amplitude
marginal on the fine grid, and `u` is drawn from the amplitude integrand at the
chosen fine node.  It is a proper joint sample, the same conditional restoration
`DistMarg.reconstruct` does for a distance, applied to the whole reduced axis;
the amplitude is never integrated out before the orientation is chosen, so
`inclination` stays correlated with `amplitude_u`.  `delta_tc` is now a sample
rather than the cell's peak lag.  The columns are **not sampled dimensions**:
they do not enter the prior, the evidence or the cell weight.  The orientation
names are PyCBC's own, so the plotter labels them without a `--labels` option.

The joint, not the inclination marginal, is the object to quote.  The amplitude
prior is registered as `u^-4` on `[1, 5]`, and that window truncates this
injection (`u = 6.673` from the measured waveform norm, `6.717` from the
analytic inclination factor; likelihood mode near `u = 7.15`).  At the registered
ceiling the `(cos iota, ln u)` correlation is only about -0.36 because the prior
wall stops the ridge before it develops; at the corrected ceiling 20 it is -0.94
and under a flat amplitude prior -0.95.  The marginal inclination is therefore
prior-dominated through the amplitude prior: at ceiling 20 its 68 per cent
interval is about 0.40 to 0.92 rad and excludes the injected 1.0 rad, while
conditional on the amplitude the inclination is determined with a width of about
0.055 rad.

The absolute scale that turns `u` into a distance is measured rather than
assumed.  The base waveform is evaluated at the partition's recorded 100 Mpc
and whitened exactly as the search whitens it: the face-on base waveform has
optimal SNR 21.6840 there, so the distance at which a face-on signal reaches
SNR 4, and therefore `u = 1`, is

    K = 100 * 21.6840 / 4 = 542.1006 Mpc ,

and `distance = K / amplitude_u`.  A second determination from the base
orientation extrapolated with the analytic inclination factor gives 548.9 Mpc,
1.3 per cent higher, so the two agree and the constant is not a fit to the
recovered posterior.  On the controlled injection the drawn 68 per cent
distance interval contains the injected 81.24 Mpc.  `distance` is exactly
reciprocal to `amplitude_u` (`distance * amplitude_u = K`) and inherits its
correlation with the inclination, so it is a conditional draw, not an
independent measurement.

`K` is the `distance-constant` option, in Mpc.  It defaults to the measured
value; the adapter raises with a clear error if it is unset, because an unset
constant would write `u^-1` as if it were megaparsecs.  A different partition
or waveform path must measure it again.  The amplitude prior itself is
configurable: `amplitude-ceiling` sets `U` and `amplitude-power` sets the
exponent, both are reachable at the registered pair, and both are recorded in
the file as `amr_amplitude_ceiling` and `amr_amplitude_power`.  The distance
marginal is controlled by that prior: quote a distance only under a prior whose
support exceeds `u` of about 7 and which is flat over five to nine.  The
default is `amplitude-power = 4`, so a flat prior is not the default; select it
with `amplitude-power = 0`.  `amr_distance_constant_mpc`,
`amr_distance_units`, `amr_amplitude_distance_relation`, `amr_distance_note`
and the drawn correlations record all of this beside the columns.

The full posterior, in two readable corners:

```bash
PYTHONPATH=/home/nightwing/src/pycbc \
  python bin/inference/pycbc_inference_plot_posterior \
    --input-file /tmp/amr.hdf \
    --output-file /tmp/amr_demarg_intrinsic.png \
    --parameters mchirp q spin1z spin2z inclination amplitude_u delta_tc \
    --plot-marginal --plot-scatter --plot-contours \
    --contour-percentiles 68 90 95 --z-arg snr:SNR \
    --expected-parameters mchirp:31.83141195 q:2.08874494 \
                          spin1z:-0.14437378 spin2z:-0.79002668 \
                          inclination:1.0 amplitude_u:6.673 delta_tc:0.0
```

## Prior

The four stiff eta coordinates have a flat prior over the ellipsoid
`eta^T M eta <= 1`; `amr.ini` expresses it as uniform boxes plus the
`[constraint-region]` quadratic.  The amplitude and orientation priors are
inside the Bayes factor, and the arrival-time prior is the lag grid, so none of
them can be written as a `pycbc.distributions` prior on the samples.  What the
restored columns carry is the conditional posterior under those priors, not a
new prior dimension.  The note
`notes/2026/10/2026-10-18-amr-sampler-pycbc-integration.md` in the research
repository has the full mapping.
