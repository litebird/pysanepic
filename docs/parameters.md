# Parameters

All the settings of the map-maker, except the data, the output pixelization
and the parallelization, are in {class}`~pysanepic.GLSParameters`. The same
object is passed by litebird_sim (`Simulation.make_sanepic_gls_map(params=...)`),
so new options need no change on the litebird_sim side.

| Field | Default | Meaning |
|---|---|---|
| `chunk_s` | 3600 | $N^{-1}$ is applied on chunks of this duration [s]; longer chunks capture lower frequencies |
| `pad_s` | 0 | Padding added on both sides of each chunk [s], so that the FFT wrap-around falls outside the data (SANEPIC's "inpaint"); each margin is fitted by an offset not included in the maps |
| `pad_fill` | `"zeros"` | `"zeros"` (no bias) or `"extrapolate"` (SANEPIC's linear extrapolation tapered to the chunk mean) |
| `pol` | `True` | Solve for I, Q, U, or I only |
| `min_pol_rcond` | 1e-2 | Q/U are solved only in pixels whose polarization-angle coverage has a reciprocal condition number ≥ this (and ≥ 4 hits); elsewhere only I. 0 disables the check, as in SANEPIC |
| `preconditioner` | `"block"` | `"block"`: exact 3×3 I/Q/U block of $P^T N^{-1} P$ per pixel; `"jacobi"`: original SANEPIC |
| `tol`, `maxiter` | 1e-12, 2000 | The PCG stops when $\lvert r\rvert^2/\lvert b\rvert^2 <$ `tol` or after `maxiter` iterations |
| `angle_dtype` | `np.float32` | Storage of 2a: float32 (angle error ~1e-7 rad) or `np.float64` (exact, 4 more bytes per sample) |
| `flag_method` | `"marginalize"` | Flagged samples in runs shorter than `split_gap_s`: `"marginalize"` (one unknown per sample, exact), `"offset"` (one offset per run), `"act"` (Legendre coefficients per run), or `"sanepic"` (fixed gap filling, no extra unknowns; approximate, see below) |
| `flag_seed` | 0 | Non-negative integer seed for `"sanepic"` gap filling, with independent streams per MPI rank; repeatable with the same input order, chunking and MPI partition. A non-zero seed without random filling (other methods, or `"conditional_mean"`) warns |
| `act_ncoeff` | `None` | With `"act"` (with another method it raises ValueError): `None` keeps the paper's rule; a positive integer fixes the coefficient count per gap; a callable `(lengths_samples, fs_hz)` returns positive integer counts. Counts are capped at the gap length |
| `gap_fill` | `"local_white"` | With `"sanepic"`: local line + white noise, `"conditional_mean"` (deterministic), or `"constrained"` (constrained realization). Conditional modes require `sky_model` |
| `gap_fill_tol`, `gap_fill_maxiter` | 1e-12, 1000 | Squared relative residual threshold and maximum iterations of each conditional gap solve; nonconvergence raises RuntimeError. Non-default values without a conditional `gap_fill` raise ValueError. `MapResult.gap_fill_iterations` reports the largest count used |
| `gap_fill_preconditioner` | `"sine"` | Conditional fills: independent own-noise sine-transform blocks for long gaps; `None` disables preconditioning. Non-default values without a conditional fill raise ValueError |
| `split_gap_s` | 300 | Runs of flagged samples at least this long [s] split the chunk and are dropped; `inf`: never, 0: every run. Disabled for DetectorGroup |
| `fft_dtype` | `np.float64` | Precision of the FFTs of $N^{-1}$ on the PCG search directions; `np.float32` is faster, see [FFT precision](#fft-precision) |
| `template_dtype` | `np.float32` | Storage of the TODs of the templates fitted with the maps (gains of the calibration, `make_maps(templates=...)`): float32 (4 bytes per sample and template, relative rounding ~6e-8) or `np.float64` (exact, 8 bytes). See [Angle precision](#angle-precision) |
| `coarse_nside` | 0 | Two-level preconditioner: also solve exactly for the I map at a coarse resolution. 0: the I monopole alone (one PCG iteration of setup); N (a power of 2): the I map at nside N, 1 with common modes; `None`: off (the preconditioner of pysanepic 0.4). See [Preconditioner](#preconditioner) |

The sections below explain how to choose them.

## Chunk length

$N^{-1}$ is applied independently on chunks of `chunk_s` seconds of each
`DetectorData` (the last chunk takes the leftover samples): correlations
between different chunks are neglected. A chunk must therefore be long
compared with the time scale of the 1/f noise, $1/f_\text{knee}$.

With litebird_sim the noise of each observation is generated independently,
so **one chunk per observation** (e.g. `chunk_s=86400` with one-day
observations) is the natural choice: it describes the simulated noise exactly,
apart from the edges. A `DetectorData` shorter than `chunk_s` is a single
chunk.

## Padding

An FFT treats each chunk as periodic, so $N^{-1}$ sees a jump between the two
ends of a chunk. With `pad_s > 0` every chunk is extended on both sides and
the wrap-around falls in the padding.

- With the 1/f noise of the LiteBIRD IMO ($f_\text{knee}$ = 20 mHz, α = 1)
  this does not matter: chunks of one hour or more give the same maps with or
  without padding.
- With a steep 1/f noise (e.g. $f_\text{knee}$ = 100 mHz, α = 2) the data
  drift, and padding of about one hour removes most of the edge effects;
  `pad_fill="extrapolate"` works best there, while `"zeros"` is unbiased in all
  cases and is the default.

See the [padding notebook](notebooks/padding) for the numbers.

## Tolerance

The PCG stops when $\lvert r\rvert^2/\lvert b\rvert^2 <$ `tol`. The default,
1e-12, is the SANEPIC one and is conservative: on 7 days of 16 LiteBIRD
detectors (nside 512, IMO 1/f noise) the iterations and the change of the map
with respect to `tol=1e-12` (28 iterations) are

| `tol` | iterations | rms map change I / Q / U [µK] |
|---|---|---|
| 1e-6 | 8 | 0.117 / 0.024 / 0.030 |
| 1e-8 | 10 | 0.023 / 0.015 / 0.019 |
| 1e-10 | 13 | 0.002 / 0.009 / 0.011 |

compared with an rms difference between the GLS and the binned maps of
3.5 / 5.3 / 5.3 µK. **`tol=1e-8` is enough for most purposes**; check
`result.residuals` if in doubt. The saving depends on the data: almost three
times fewer iterations here, but 20 instead of 25 on one year of 128 detectors,
where the rms map change was 0.016 / 0.007 / 0.008 µK (GLS − binned:
0.25 / 0.31 / 0.31 µK).

## Polarization-angle coverage

Q and U can be separated in a pixel only if it is observed with enough
different polarization angles. pysanepic solves Q and U only in pixels with at
least 4 polarized hits (γ > 0.5) and a reciprocal condition number of the
angle-coverage matrix $\sum_t w_t w_t^T$, $w = (1, \gamma\cos 2a, \gamma\sin 2a)$,
of at least `min_pol_rcond`; elsewhere only I is solved. With
`min_pol_rcond=0`, as in the original SANEPIC, Q and U are "solved" also in
pixels where they are degenerate, and are pure noise there (see the
[preconditioner notebook](notebooks/preconditioner)).

## Preconditioner

`"block"` uses the inverse of the exact 3×3 I/Q/U block of $P^T N^{-1} P$ in
each pixel, built from the actual angles, polarization efficiency and noise
correlations; `"jacobi"` is the original SANEPIC choice, which assumes a
uniform coverage of polarization angles. With a good coverage (Q-type and
U-type detectors) they converge equally fast; with a single pair of detectors
the block preconditioner needs less than half the iterations. Both converge to
the same map.

`coarse_nside` adds a coarse correction for the large-scale modes of I (see
[two-level preconditioner](algorithm.md#two-level-preconditioner)): its setup
costs about one iteration per coarse pixel, 12 at `coarse_nside=1`. On G100
(16 days, 8 detectors of a wafer, nside 512, half of the noise variance in a
common mode):

| GLS | `coarse_nside` | iterations | setup | total time |
|---|---|---|---|---|
| common mode modelled | off | 46 | – | 285 s |
| | 1 | 21 | 54 s | 191 s |
| | 2 | 22 | 247 s | 403 s |
| | 4 | 22 | 969 s | 1124 s |
| correlation ignored | off | 20 | – | 79 s |
| | 1 | 13 | 31 s | 91 s |

With common modes `coarse_nside=1` is the best choice (−33% time); higher
resolutions only add setup. Without them the 12 products of the setup do not
pay for the iterations they save. (These runs are without a coarse level,
`None`; the monopole level, `0`, did not exist yet.)

Most of what the coarse level gains comes from one mode: the I monopole,
which 1/f weights constrain poorly and which is the last to converge. With
calibration it is worse, as the monopole is coupled to the gain common to all
the detectors: the residual stalls for ~10 iterations until both converge
together. `coarse_nside=0` (the default) is a coarse level of one pixel, the
whole sky: it solves for the monopole alone, and its setup costs a single
product by $P^T N^{-1} P$. In a local test (4 detectors, 1/f noise, nside 16):

| `coarse_nside` | maps | calibration (1st + 2nd solve) | setup |
|---|---|---|---|
| `None` (off) | 31 | 43 + 34 | – |
| `0` (monopole, default) | 22 | 33 + 25 | 1 product |
| `1` | 25 | 33 + 24 | 12 products |

and on G100 (8 days, 8 detectors, nside 256, calibration with one gain per
hour): maps 21 → 12 iterations (62 → 49 s), calibration 44 + 7 → 30 + 7
(152 → 125 s), calibration with a bandpass template 35 + 40 → 35 + 12
(236 → 182 s). Every level, `N ≥ 1` included, contains the monopole: it
changes only the speed of the PCG, the maps are the same within the
tolerance.

## Angle precision

Twice the polarization angle, 2a, is computed in float64, reduced to
[−π, π) and stored in float32 by default, an error of ~1e-7 rad on the angle,
irrelevant for any application; γ cos 2a and γ sin 2a are recomputed for each
chunk when it is used. `angle_dtype=np.float64` stores 2a exactly at the cost
of 4 more bytes per sample.

The same for the templates fitted with the maps (the gains of the calibration
and `make_maps(templates=...)`): their TODs are stored in float32 by default
(`template_dtype`), rounded to a relative ~6e-8. The same rounded templates
are used in $T$ and $T^T$, so the system stays symmetric and the amplitudes
are exactly those of the rounded templates: e.g. for the calibration
template (dipole, ~3 mK) the rounding is ~2e-10 K per sample, uncorrelated,
orders of magnitude below the noise. `template_dtype=np.float64` keeps them
exact at the cost of 4 more bytes per sample and template.

## Flagged samples

`DetectorData.flags` marks samples that must not enter the maps (glitches,
gaps, manoeuvres, ...). Their original TOD is never used, so it may be NaN,
and they are not hits. They cannot simply be removed, because $N^{-1}$ is
applied with FFTs on contiguous chunks:

| | unknowns | result | memory | PCG |
|---|---|---|---|---|
| `flag_method="marginalize"` | one per flagged sample | exact GLS without them | ~70 bytes per flagged sample | slowed down by long gaps |
| `flag_method="offset"` | one per run of flagged samples, set to 0 | unbiased in the linear problem, generally not exact | ~70 bytes per run | long runs cost fewer unknowns |
| `flag_method="act"` | Legendre coefficients per run, samples set to 0 | unbiased in the linear problem, generally not exact | ~70 bytes per coefficient + 32 bytes per nonconstant run | fewer unknowns than per sample for extended gaps |
| `flag_method="sanepic"` | none; fill selected by `gap_fill`, fixed during the map solve | approximate; a conditional fill is exact only for the supplied residual/model | 8 bytes per flag + mask views, no extra map PCG state (without padding); conditional fill has temporary solver buffers | no extra map unknowns; conditional fill has an inner CG |
| splitting (`split_gap_s`) | none: the run is dropped and the chunk split | noise correlations across the gap neglected | less | not slowed down |

Runs at least `split_gap_s` long are split, the shorter ones are handled by
`flag_method`. DetectorGroup chunks are never split at gaps: the detectors
must retain the same time grid. The benchmark below compares only the
marginalize, offset and splitting methods, not ACT or SANEPIC filling.
Inside a long gap the unknowns of `"marginalize"` are strongly
correlated by the 1/f noise and converge slowly: on two days of 4 detectors
($f_\text{knee}$ = 50 mHz) a one-hour gap took the PCG from 25 to 71
iterations, and splitting it (the default, 300 s) brought it back to 25. In
the same test all the methods gave the same map errors (see the
[flags notebook](notebooks/flags)). Splitting every run (`split_gap_s=0`) makes
many short chunks of any length; their FFTs run on fast lengths (see
[noise model](algorithm.md#noise-model)), so this costs no extra time either.

`"sanepic"` implements the approximate normal matrix of
[Patanchon et al. (2008), Sect. 3.6](https://arxiv.org/abs/0711.3462).
The flagged rows of the pointing and fitted templates are zero, with no
dummy-pixel parameters or Schur-complement correction. With the default
`gap_fill="local_white"`, each gap is filled with a line fitted to valid
samples within 20 samples
on each side, plus independent Gaussian white noise whose rms is estimated
from the fit residual (two fitted parameters). At a chunk edge only the
available side is used; with one neighbor the fill is constant, with none
it is zero. Flagged input values never enter the fit. The synthetic values
affect the RHS through noise correlations, although they are not hits.
This is not a constrained noise realization and can bias the sky, especially
on large scales or for long/dense gaps; validate signal transfer and noise
spectra before using it for science. The default remains `"marginalize"`.
`split_gap_s` applies before filling. See
[random seeds and MPI](#random-seeds-and-mpi) for reproducibility.
Without padding only synthetic gap values and views of the original TOD
and flags are kept; the filled TOD is materialized one chunk per worker.
With padding the filled, padded chunks are stored as for the other methods.

`"offset"` differs from this approximation: it still fits the auxiliary
parameters, but shares one within a run. For isolated flags it is identical
to `"marginalize"`, including its memory cost.

### ACT cut model

`flag_method="act"` implements the compressed cut model of
[Naess et al. (2025), Sect. 3.3.3](https://arxiv.org/abs/2503.14451), with
the sky and fitted templates zero on the flags and a Legendre series fitted
in each contiguous run. A run of length up to 1 / 3 / 6 / 20 samples has
1 / 2 / 3 / 4 coefficients; beyond that the count is the nearest integer to
$5 + T/4$, with $T$ the duration in seconds (sample count / sampling rate).
Ties round upward, and the count is capped at the number of samples. These
are the ACT thresholds, retained literally at LiteBIRD sampling rates, not
an optimization for LiteBIRD. `split_gap_s` still applies first.

`act_ncoeff` overrides this rule, both in `GLSParameters` and low-level `GLS`:

```python
import numpy as np
from pysanepic import GLSParameters

# Published ACT rule (default).
params = GLSParameters(flag_method="act", act_ncoeff=None)

# Fixed coefficient count per gap: 1 = offset, 2 = linear, 3 = quadratic.
params = GLSParameters(flag_method="act", act_ncoeff=3)

# One constant plus one coefficient per two seconds, at most eight.
def cut_counts(lengths_samples, fs_hz):
    return np.minimum(8, 1 + np.ceil(lengths_samples / fs_hz / 2).astype(int))

params = GLSParameters(flag_method="act", act_ncoeff=cut_counts)
```

This is a **coefficient count**, not a polynomial degree: degree = count - 1.
The count is always capped at the number of samples in each gap, including
for custom rules. A callable receives all gap lengths of one chunk as a 1D
integer array and that chunk's sampling rate in Hz; it must return a positive
integer scalar or a positive integer array of the same shape. It should be
pure and deterministic, since calibration reuses the GLS setup with the same
cuts. Invalid counts or shapes raise ValueError. `act_ncoeff=1` is equivalent
to `"offset"`; a rule returning `lengths_samples` gives exact marginalization.
Custom rules tune the approximation and should be validated against it.

The basis is evaluated by recurrence without storing a design matrix.
For high orders (`ncoeff > 4` and `ncoeff**2 > 4 * gap_length_samples`),
sampled Legendre columns are replaced by discrete orthonormal Gram
polynomials, spanning **the same polynomial space of the same dimension**.
Low degrees use the degree recurrence; high degrees use the symmetric Hahn
difference equation on the sample grid, evaluated from the endpoint to the
middle and reflected with parity to avoid endpoint instability. This is a
change of coordinates, not a switch to full sample marginalization.
See the [Hahn difference equations](https://dlmf.nist.gov/18.22.ii).
Coefficients belong to their chunk and MPI process. Their
preconditioner is the exact own-noise diagonal of $C^T N^{-1} C$ (as for the
other local unknowns); long cuts use FFT autocorrelations in setup. If the
number of coefficients equals the sample count, an identity basis avoids
an ill-conditioned polynomial representation and gives exact marginalization.
For isolated flags this mode is identical to `"marginalize"`; `"offset"`
is the constant-only version. Otherwise the restricted cut space changes
the noise weighting and need not give minimum-variance maps.

This reproduces the cut parametrization, not ACT's full preprocessing:
their Sect. 3.1 also predicts glitch values using other detectors before
noise estimation and filtering. Here the flagged input TOD is always ignored
and replaced by zero, with no random fill or dependence on `flag_seed`.

### Gap filling and noise realizations

Gap filling means assigning numbers to missing time samples so that regular
FFTs can be used; it does not necessarily mean adding random noise.
`"marginalize"`, `"offset"` and `"act"` use zeros as a reference and fit free
parameters there. With one free value per sample any fixed fill would give
the same map, since its change can be absorbed by the nuisance parameters.
With a restricted polynomial basis only changes in that basis are absorbed;
adding arbitrary white noise would generally change the map.

With the default `gap_fill="local_white"`, `"sanepic"` generates random
white noise on top of a locally fitted line. That is not a constrained
realization of the full correlated noise. `gap_fill="constrained"` instead
draws the Gaussian conditional distribution for the supplied sky/noise model:

$$n_f\mid n_u \sim \mathcal N\!\left(
N_{fu}N_{uu}^{-1}n_u,\quad
N_{ff}-N_{fu}N_{uu}^{-1}N_{uf}\right),$$

where $u$ are valid samples and $f$ the gaps, using the noise residual after
subtracting the supplied sky and current known templates. Use:

```python
from pysanepic import GLSParameters, make_maps

# data, nside and initial.maps are already available; the model is in the
# output map coordinates ("G" here) and at the output nside, or supplied as alm.
params = GLSParameters(
    flag_method="sanepic",
    gap_fill="constrained",       # or "conditional_mean", without random noise
    gap_fill_tol=1e-12,
    gap_fill_maxiter=1000,
    gap_fill_preconditioner="sine",  # None: ordinary CG for comparison
    flag_seed=42,
)
result = make_maps(data, nside, params=params, sky_model=initial.maps)
```

`sky_model` is required to distinguish noise from the sky signal; for
noise-only input pass an explicit zero sky model. The model is subtracted
and its binned map restored as usual. The residual filling contains noise,
not the sky signal in the gap. All gaps of a chunk are conditioned together,
and a DetectorGroup is conditioned jointly across detectors using the same
common-mode precision as GLS. Padding is unobserved: it is filled jointly
with gaps, not used as a constraint; its initial `pad_fill` is immaterial.
Chunks removed by `split_gap_s` are still removed before filling.

The implementation solves $Q_{ff}x=-Q_{fu}n_u+g_f$, with $Q=N^{-1}$ and
$\operatorname{Cov}(g)=Q$. Dropping $g_f$ gives `"conditional_mean"`.
Products use FFTs without a dense time-domain covariance. For common modes,
a low-rank factor generates $g$ without a dense detector covariance per
frequency. The draw uses the full FFT grid, including an FFT extension,
before restricting to the observed time range, so its covariance matches
the precision used by the map-maker.

Weights must be finite and positive except for an optional zero DC weight.
With zero DC, a fully missing detector over the complete FFT grid has no
proper conditional distribution and raises ValueError; use a physically
defined DC prior to draw such an unconstrained mode. Failure to converge
raises RuntimeError. The fill is held fixed throughout the map PCG.
`gap_fill="conditional_mean"` and `"constrained"` require
`flag_method="sanepic"`; choosing them with another method raises ValueError.
`gap_fill_tol` and `gap_fill_maxiter` control the gap CG, whereas `tol` and
`maxiter` control the map PCG: these are separate convergence tests.
`gap_fill_preconditioner="sine"` (default) uses positive definite blocks
for individual gaps. Runs longer than 32 samples use an orthonormal DST-I
and the detector's own inverse noise spectrum interpolated at frequencies
`j / (2 * (gap_length + 1))` in cycles per sample. Short runs use the
own-noise diagonal; a long, fully missing complete FFT row uses its exact
own-noise inverse. It stores no dense time-domain matrices. The blocks
ignore correlations between gaps and detectors, while the **PCG operator
keeps the full GLS precision**, including common modes and FFT extension.
This follows the [sine-transform preconditioning approach](https://arxiv.org/abs/1404.5332);
effectiveness for arbitrary spectra or common modes is not guaranteed.
`None` disables it for comparisons. Both modes stop on the true,
unpreconditioned squared relative residual; preconditioning does not alter
the conditional distribution. No random numbers are consumed by setup.
`flag_seed` controls the random draw for a fixed MPI partition and input order;
`"conditional_mean"` consumes no random numbers.

The conditional draw is correct for the *supplied residual and noise model*.
A single fill frozen during the approximate SANEPIC solve is not exact GLS
marginalization or a Gibbs map sampler. Updating the model and resampling
the map jointly would require a separate outer algorithm. `"marginalize"`
remains the exact reference for maps without a frozen-fill approximation.

The public {func}`~pysanepic.conditional_gap_fill` also operates directly on
noise residuals, independently of map-making; it returns `(filled, info)`:

```python
from pysanepic import conditional_gap_fill

# residual is noise-only, flags is boolean, and noise/fs/nfft are known.
filled, info = conditional_gap_fill(
    residual, flags, noise.weights(nfft, fs), fft_length=nfft,
    rng=42, draw=True, tol=1e-12, maxiter=1000, preconditioner="sine",
)
```

Residuals may be `(ns,)` or simultaneous `(ndet, ns)` arrays. For the latter,
pass per-detector weights and optionally a coupling array of shape `(ndet, r)`
and mode_psd of shape `(r, nfft//2+1)` for the common modes. Weights are
inverse noise eigenvalues as in GLS, not a one-sided PSD. Good samples are preserved
and inputs are not modified. `info` contains iterations, convergence, and
the squared relative residual of the gap solve.

### Random seeds and MPI

Pass the **same** non-negative `flag_seed` (default 0) on every MPI rank.
Each rank constructs its own stream:

```python
rng = np.random.default_rng(np.random.SeedSequence([flag_seed, rank]))
```

This gives distinct streams across ranks. A stream is consumed sequentially
as the local chunks or groups are filled; it is not keyed by detector or
chunk identity. Consequently:

| Change between runs | Random filling |
|---|---|
| Same inputs, flags, chunking, rank assignment and local order | Repeatable within floating-point precision |
| Only the thread count changes | Same random numbers; small numerical differences may remain |
| Rank count, rank assignment, chunking or input order changes | Generally a different realization |
| `gap_fill="conditional_mean"` | No random draws; independent of `flag_seed` |
| `"marginalize"`, `"offset"` or `"act"` | No random filling; `flag_seed` has no effect |

Only `"local_white"` and `"constrained"` use random draws. The generator
is reset at each calibration pass, reusing the underlying white draws when
the chunk/flag layout is unchanged. The filled values can nevertheless
change because the calibrated residual, fitted baseline or noise scale
changes. Fills are fixed during each individual map PCG solve.

A DetectorGroup resides on one rank and is sampled jointly, retaining its
cross-detector correlations. There is currently no detector/chunk-keyed
seed scheme guaranteeing the same realization under MPI redistribution.

## FFT precision

The FFTs that apply $N^{-1}$ take most of the time of an iteration (~60% in
the one-year benchmark). With `fft_dtype=np.float32` those applied to the PCG
search directions run in single precision: on G100 they are ~45% faster
(relative error ~3e-7), while the right-hand side and the true residual
(recomputed every 10 iterations and at convergence) stay in float64, so the
solution is still controlled in double precision. On 14 days of 8 detectors
the iterations were the same and the GLS took 21% less time at `tol=1e-8`
(15% at 1e-12); the maps differed from the float64 ones by ~1e-6 of their
rms, 10⁴ times less than the difference between `tol=1e-8` and 1e-12.

