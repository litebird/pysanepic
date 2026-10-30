# Changelog

## HEAD

Changes on `main` that are not in a release yet.

## 0.6.1 (2026-10-30)

- `GLSParameters.sample_batch` (default 1): with a sky model or sky
  templates as alm, the number of `DetectorData` interpolated together, one
  `ducc0.sht.synthesis_general` call per field, whose fixed cost (a
  spherical harmonic transform at the lmax of the alm) is then shared. On
  G100 (lmax 1536, 48 threads) 8 `DetectorData` of a day per call take ~3
  times less per day; with `DetectorData` of a week or more the cost is
  amortized already. ~50 bytes per sample of the batch. New docs section
  "Sampling alm models" (performance) and test `test_sample_batch`. (#60)
- Sky model and sky templates given as alm: the polarized part sampled at
  the pointing, gamma (Q cos 2a + U sin 2a), is combined by a numba kernel
  on all the threads (cos and sin from the table of the PCG) instead of ~10
  serial numpy passes with `np.cos` and `np.sin` for each field. On one G100
  node (48 threads, lmax 1536, one day at 37.5 Hz), the 4 fields of the e2e
  run (sky model, bandpass, angle and FWHM templates): 1.14 → 0.44 s per
  day, the rest being `ducc0.sht.synthesis_general`. Once per run with
  `cache_model`, else at each calibration iteration. New test
  `test_sample_alm`. (#59)
- The shared templates of a chunk with one amplitude each (e.g. the sky and
  beam templates, one amplitude per detector) are the columns of one array,
  applied in one pass over the chunk in P and P^T instead of one per
  template; `make_maps` passes them so, and `GLS` takes them as
  `(chunk, values (n, ncol), labels (ncol,))`. On one G100 node (96 chunks
  of a day, 48 threads, 8 templates, the gains, 17% flagged), one PCG
  iteration: 2.57 → 2.14 s (1.85 s without the templates). (#58)
- `CalibrationParameters(covariance_fixed_gains=True)`: `template_covariance`
  with the gains fixed at their fitted values instead of marginalized over.
  The gains couple to the sky templates through the sky signal in the gain
  template and slow down the solves of the covariance (~2x their
  iterations); fixed, the errors were 0.05–0.13% smaller in a test. Default
  `False`. (#58)
- Setup: the coverage of the chunks in parallel (8 private arrays, summed in
  a fixed order); the templates parsed in parallel, without copies when
  their flagged samples are already 0 (e.g. with the mask of the gains). (#58)
- 1-year e2e run on G100 (M1_L, 1 detector per process, 48 threads,
  calibration with a |b| > 10° mask, 3 iterations, 7 templates): GLS 2005 →
  1561 s (PCG 958 → 667 s, setup 392 → 274 s), same iterations, gains and
  amplitudes; with `template_covariance`, `covariance_tol=1e-6` and
  `covariance_fixed_gains=True`, GLS 10683 → 4093 s (the covariance in 324
  PCG iterations instead of 914), the same errors to the digits printed. (#58)
- Steps outside the PCG, measured on a 1-year e2e run on G100 (M1_L, 1
  detector per process, 48 threads, calibration with a |b| > 10° mask, 3
  iterations, 7 templates): GLS 3161 → 2005 s, same iterations, gains and
  amplitudes. Sums over the 4 solves:
  - the setup of the preconditioner sorted the samples of each chunk over
    the labels of all the unknowns of flagged samples of the process (200
    million in a year with the mask), not only those of the chunk: 262 →
    93 s;
  - `GLS.binned` (the binned map of the sky model, at each calibration
    iteration) uses the coverage of the setup, kept (56 bytes per pixel,
    divided among the processes), instead of computing it again serially,
    and one buffer for the flagged samples instead of one per chunk: 427 →
    12 s;
  - the preconditioner of an amplitude spanning more than a quarter of the
    FFT of its chunk (e.g. one per detector) takes one FFT of that length
    instead of two of twice its span, and no sort; exact: 251 → 67 s;
  - the projection of the residual map for the gain template, and the
    known part and calibration of the TOD, are numba kernels on all the
    threads (cos and sin from the table of the PCG) instead of serial
    numpy: 347 → 52 s.
  New tests `test_setup_flagged_exact`, `test_template_diag_exact`
  (the preconditioners against dense matrices) and `test_project_maps`. (#56)
- `GLSParameters.covariance_tol` (default 1e-6): the tolerance of the PCG
  solves of `template_covariance`, one per amplitude, which used `tol`. In a
  test with calibration and bandpass, angle and FWHM templates, sigma to
  4e-5 and correlations to 8e-4 against 1e-10, with 1/3 fewer iterations. (#56)

## 0.6.0 (2026-10-28)

- PCG products and template preconditioner: **one chunk per thread** (FFTs
  single-threaded, $P^T$ in a private map per thread if they fit in 4 GB).
  A single long FFT scales poorly with threads (memory bound: 2–4x on 48
  cores of G100), independent FFTs well (15–25x). On one G100 node (96
  chunks of a day, 48 threads), 10 PCG iterations: 60.4 → 14.6 s with
  float32 FFTs, 89.6 → 16.9 s with float64, same result. Chunks with
  common modes keep the previous scheme. (#53)
- PCG with flagged samples (`flag_method="marginalize"`, e.g. the gain mask
  during the calibration): the chunk threads add $P^T$ to the unknowns of
  the flagged samples in place (each belongs to one chunk) instead of a
  private copy per thread summed serially, and the updates and dot products
  of the PCG vectors run on all the threads (dot products over fixed
  blocks: the same result with any number of threads). With 17% of the
  samples flagged (|b| < 10°), one PCG iteration on one G100 node (96
  chunks of a day, 48 threads, 8 templates and the gains): 7.2 → 2.6 s; in a
  90-day e2e run with 3 calibration iterations, PCG 990 → 370 s, GLS 1470 →
  850 s, same result. (#53)
- With `verbose`, `make_maps` prints the time of each step of each solve:
  TOD and templates, GLS setup, PCG, maps. (#53)
- `_template_diag` no longer fails on a template that is 0 in a whole
  chunk (e.g. a chunk in pixels whose Q/U are not solved). (#53)
- `CalibrationParameters(gain_tol=...)`: stop the calibration iterations
  when the largest gain correction |dg| (all detectors, periods and MPI
  processes) is below it, `n_iter` being the maximum; the number done is in
  `MapResult.calibration_iterations`. With `verbose`, max |dg| is printed
  at each iteration. (#49)
- **Breaking change**: with calibration, a `Template` without a mask has
  that of the gains (`CalibrationParameters.mask`) instead of all pixels,
  so that by default each sample is in all the fits or in none: samples
  out of the gain fit only are calibrated with the gains of the previous
  iteration, which converges much slower (in a 1-year test, gain errors
  up to 1e-3 after 4 iterations, against the noise with equal masks). For
  all pixels, pass a mask of ones. `make_maps` warns when the masks of
  gains and templates differ. Equal masks are kept once in memory, and
  their samples are tested once per mask. (#51)
- Calibration: the samples in pixels whose Q/U are not solved (out of every
  fit) are flagged from the second iteration, like those out of every mask,
  and back in the final maps. They were calibrated with the gains of the
  previous iteration and slowed the iterations down to ~2x each (gain errors
  of 6e-3 after 10 iterations with 3 detectors, 1e-9 after 4 now). One more
  GLS setup, only when there are such pixels. New test
  `test_calibration_unsolved_pol`. (#52)
- **Breaking change**: one interface for the templates fitted with the maps,
  `make_maps(templates={name: Template(...)})`, which replaces
  `bandpass_templates` and `bandpass_common`; `MapResult.templates` replaces
  `MapResult.bandpass`, and `CalibrationParameters.mask` is the mask of the
  gains only (one map, no dict): each template has its own,
  `Template(mask=...)`, also without calibration. The bandpass templates of 0.5 are
  `Template(sky=..., sum_zero=True)` (`sum_zero=False` for
  `bandpass_common=True`), with the same results. (#44)
- `Template`: a sky signal (`sky=`, maps or alm) or a TOD given per detector
  in the new `DetectorData.templates`, with one amplitude per detector
  (default, or summing to zero with `sum_zero`), one common to all
  (`common`), or one per `DetectorData` and period (`period_s`, local
  unknowns like the gains). Fitted in the same PCG as the maps, with
  calibration too. (#44)
- `make_maps(template_covariance=True)`: the covariance of the amplitudes
  per detector and common, marginalized over the maps and the amplitudes
  per period, in `MapResult.template_covariance` (one PCG solve per
  amplitude). Low level: `GLS.shared_covariance`. A Monte Carlo
  (`tools/template_covariance_mc.py`) confirms the errors to ~10%. (#44)
- Docs: polarization angles of the detectors fitted with a sky template,
  $(0, 2U, -2Q)$ from the sky model, with sum-zero amplitudes (the angle
  errors; the common rotation goes into the maps); new test
  `test_pol_angle_template`, also with calibration and with an HWP, where
  the sign is reversed. (#45)
- Docs: beams. Pointing offsets, ellipticities and FWHM of the detectors
  fitted with templates: the FWHM as a sky template (the Laplacian of the
  sky model), offsets and ellipticities as TOD templates from the model
  resampled with the beam moved by ±ε, in a recipe (`docs/usage.md`).
  New test `test_beam_templates`: the sky through elliptical Gaussian beams,
  with and without HWP; the fit is right to second order. (#46)
- Template helpers (`pysanepic.templates`): `pol_angle_template`,
  `fwhm_template`, `beam_templates` / `beam_tods` (pointing offsets and
  ellipticity as TODs, exact derivatives of the sky model from the
  spin-raising and lowering operators with ducc0, with the `sum_zero` the
  HWP needs) and
  `beam_parameters` (offsets and beam axes from the fitted amplitudes, the
  $\rho\rho^T$ of the offsets subtracted); the docs use them instead of the
  recipes. (#47)
- Memory of the templates fitted with the maps (gains included): their TODs
  are stored once, in the new `GLSParameters.template_dtype` (default
  `np.float32`, 4 bytes per sample and template, rounded to a relative
  ~6e-8, far below the noise; `np.float64` for the exact values), with
  int32 labels, or one label per chunk for an amplitude per detector. Before,
  a template took ~28 bytes per sample at the peak of the setup (float64
  values and int64 labels, copied): a one-year e2e run with calibration and 7
  templates ran out of memory on G100 (360 GB per detector). With 4
  templates and calibration, 163 → 47 bytes per sample. Low level:
  `GLS(template_dtype=...)`, and the labels of a template can be one int. (#50)

## 0.5.2 (2026-10-26)

- PCG: two products by $P^T N^{-1} P$ fewer per solve. Without `x0` the
  initial residual is the right-hand side (it was computed as $b - A\,0$),
  and the true residual is recomputed after every `recompute_every`-th step
  instead of also after the first one: ~24 → 21 products in a 20-iteration
  run. Same maps within the tolerance. (#40)
- Common modes: $N^{-1}$ of a `DetectorGroup` keeps the spectra of its
  detectors between the two passes of the Woodbury product instead of
  recomputing them: the same FFTs as uncorrelated noise (1.5 times before),
  for 8 bytes per sample of the group while it is processed. Same maps. (#41)
- The preconditioner of the template amplitudes finds the samples of every
  amplitude in one pass over each chunk (it scanned the chunk once per
  amplitude): `set_templates` 1.49 → 0.43 s with gains every 10 minutes
  (8 detectors × 1 day at 19 Hz), now independent of the gain period. Same
  preconditioner. (#42)

## 0.5.1 (2026-10-24)

- `make_maps` refuses `sky_model` or `bandpass_templates` as alm with a
  `Pointing` in pixels before processing any detector, with a message naming
  them (it failed at the first detector, from `Pointing.angles`). (#37)
- With alm models the angles of each detector are rotated to the coordinates
  of the maps once, for the pixels and for all the models (they were rotated
  again for the sky model and for each bandpass template); same results. (#38)
- Docs: the `coordinates` of a `Pointing` are declared, not inferred;
  litebird_sim passes pixels, or angles with alm models, in the coordinates of
  the maps. (#38)

## 0.5.0 (2026-10-22)

- **Behaviour change**: `GLSParameters.coarse_nside=0` (the default) is now a
  two-level preconditioner on the I monopole alone (one coarse pixel), the
  slowest mode of the PCG, at the cost of one product at setup: 31 → 22
  iterations for maps and 43 → 33 with calibration in a test, 62 → 49 s and
  152 → 125 s on G100. The maps are the same within the tolerance.
  `coarse_nside=None` gives the preconditioner of 0.4 (where 0 meant no coarse
  level); `N ≥ 1` is unchanged. (#35)
- Gain calibration on the dipole: `make_maps(calibration=CalibrationParameters(
  gain_period_s, n_iter, mask, cache_model))` fits one gain per detector and
  period together with the maps, linearizing $d = g\,(P m + D) + n$ around
  the current gains with the template $T = D + P m_\text{prev}$, and returns
  them in `MapResult.gains`. `DetectorData.dipole` is the dipole signal $D$,
  whose orbital part sets the absolute calibration. `n_iter` defaults to 2;
  the GLS setup is reused between the iterations, each of which starts from
  the previous maps; `cache_model` keeps the model TODs in memory between
  them. On G100 (8 days, 8 detectors, nside 256) the calibrated maps equal
  those with the true gains, in 125 s against 49 s for the maps alone.
  (#34, #35)
- `CalibrationParameters.mask` (1 = pixel used, 0 = left out; one map, or
  `{"gain": ..., "bandpass": ...}`): samples left out of every fit are flagged
  in the iterations, then the final maps use all of them. (#34, #35)
- `make_maps(sky_model=...)`: a model of the whole sky, as maps or as alm
  interpolated at the pointing (`ducc0.sht.synthesis_general`), is subtracted
  from the TOD and its binned map put back: with alm its structure below the
  pixel size no longer leaks through the 1/f weights, into the maps and, with
  calibration, into the gains. (#35)
- `make_maps(bandpass_templates={name: template})`: one amplitude per template
  and detector (`DetectorData.name`, shared by its `DetectorData` across MPI
  processes), fitted with the maps (and the gains), in `MapResult.bandpass`;
  with $\partial F/\partial\nu$ as template the amplitudes are the band
  shifts. They sum to zero by default; `bandpass_common=True` (alm templates)
  fits the common part too, which otherwise biases the absolute calibration by
  ~β·(mean shift)/ν (4e-4 on G100, removed in a noiseless run; with noise it
  needs long runs). In the notebook the shifts are recovered within the noise
  and the Q/U error drops from 3.8 to 1.3 µK. (#35)
- Low level: `GLS(templates=..., shared_templates=..., shared_groups=...)`
  (template amplitudes fitted with the map, local to each process or shared
  by all, with sum-zero constraints; `info["amplitudes"]`,
  `info["shared_amplitudes"]`), `GLS.set_templates`, `GLS.binned`,
  `Pointing.angles`. Templates are added and summed with numba kernels, and
  the diagonal of their preconditioner comes from autocorrelations (exact,
  ~2 FFTs per chunk). (#34, #35)
- New notebooks `calibration`, `sky_model` and `validation` (the same noise
  with and without gain variation and bandpass mismatch, each ignored or
  fitted). (#34, #35)

## 0.4.0 (2026-10-17)

- `estimate_noise` estimates common modes: for each `DetectorGroup` the
  cross-spectra of its detectors, averaged over the groups with the same
  detectors (also across MPI processes), are fitted with `n_modes` common
  modes plus the own noise of each detector by expectation-maximization (as
  `sanePS` of the IAS SANEPIC). The result has the own noise under each name
  and the list of `CommonMode` under the tuple of the names. On G100 the
  couplings are found within 0.1% and the map with the estimate equals that
  with the true model within 1% of the noise.
- Two-level preconditioner, `GLSParameters.coarse_nside` (and `GLS(coarse=...)`):
  the PCG also solves exactly for the I map at a coarse resolution, which
  takes care of the slow large-scale modes. With a common mode `coarse_nside=1`
  halves the iterations (46 → 21) and the GLS time drops by 33% on G100. The
  default stays 0 (off): without common modes the setup costs more than the
  iterations it saves.
- Correlated noise: `DetectorGroup(detectors, modes=[CommonMode(coupling,
  noise)])` for detectors observed at the same times that share common
  modes, $N = D + U \Lambda U^T$ per frequency, applied with the Woodbury
  identity one detector at a time. `GLS` has a `common_modes` argument.
- `estimate_noise(data, result)`: the noise spectrum of each detector from the
  residual of a first map, as a `TabulatedNoise` (binned periodogram) or, with
  `fit="toast"`/`"keshner"`, a fitted `OneOverFNoise`. `DetectorData.name`
  groups the `DetectorData` of a detector (also across MPI processes).
- The FFT work buffers of $N^{-1}$ are allocated once and reused (−9% on the
  FFTs on G100).
- `GLSParameters.fft_dtype` (and `GLS(fft_dtype=...)`): `np.float32` runs the
  FFTs on the PCG search directions in single precision (~45% faster FFTs,
  15–21% less GLS time on G100), with the right-hand side and the true
  residual in float64. The default stays `np.float64`.

## 0.3.0 (2026-10-15)

- **Breaking change**: the noise of a detector is an object,
  `DetectorData(noise=...)`, instead of the fields `net_ukrts`, `fknee_hz`,
  `alpha`, `fmin_hz`: `OneOverFNoise(net_ukrts, fknee_hz, alpha, fmin_hz,
  model)` or `TabulatedNoise(freqs_hz, psd)`. Each checks its values and gives
  the inverse PSD of a chunk with `weights(ns, fs)`; one object can serve all
  the time spans of a detector.
- `OneOverFNoise.model="keshner"` (the spectrum of litebird_sim's noise with
  `engine="ducc"`) besides the default `"toast"`; `TabulatedNoise`, a
  tabulated one-sided PSD in K²/Hz. For `GLS`, `one_over_f_weights` has a
  `model` argument and `psd_weights` computes the weights of a tabulated PSD.
- Flagged samples: `DetectorData.flags` (boolean, `True` = excluded). Their TOD
  is not used (it may be NaN) and they are not hits. `GLSParameters.flag_method`
  chooses `"marginalize"` (default: one unknown per sample, the exact GLS
  without them) or `"offset"` (one offset per run, samples set to 0), and runs
  longer than `GLSParameters.split_gap_s` (default 300 s) split the chunk and
  are dropped, so that long gaps do not slow down the PCG. In `GLS`, negative
  pixels mark flagged samples, fitted by unknowns local to each process.
- FFTs on fast lengths: `make_maps` applies $N^{-1}$ to each chunk with an FFT
  of the smallest even length with small prime factors (`fast_fft_length`),
  zero-extending the chunk. Chunks split at gaps, or observations of awkward
  length, were up to 7 times slower; the maps of chunks that already had a
  fast length (e.g. one day at 75 Hz) do not change. `GLS` has a
  `fft_lengths` argument.

## 0.2.1

Packaging for PyPI: metadata, source distribution without the C++ code, and
publication from GitHub releases. No code changes.

## 0.2.0 (2026-10-14)

First release: a standalone Python port of the SANEPIC GLS map-maker for
correlated (1/f) noise, used by litebird_sim (`make_sanepic_maps`).

### Map-making

- GLS solution of $(P^T N^{-1} P)\,m = P^T N^{-1} d$ with a preconditioned
  conjugate gradient; $N^{-1}$ applied per chunk as a Fourier filter (ducc0
  FFTs), I or I/Q/U maps, optional HWP angle.
- Block 3×3 preconditioner (Jacobi still available) and the Q/U
  angle-coverage criterion `min_pol_rcond` (#2).
- Chunk padding, as SANEPIC's `-m`/`-q` options (#4).
- `GLSParameters` collects the solver options (#6).

### Interface

- `Pointing`: either angles (θ, φ, ψ in some coordinates), rotated and
  pixelized by `make_maps` at the output nside and coordinates, or HEALPix
  pixels already computed (RING or NESTED), used as they are; the arrays are
  checked when the object is created. `DetectorData` takes a `Pointing` (#6,
  #7, #12).
- Low-level `GLS` class: `apply_A` and `rhs` work on the pixels of the MPI
  rank, `GLS.full` gathers a per-pixel array (#14).

### Parallelism and memory

- MPI: the PCG state is distributed by pixel; the $P^T$ maps are summed with a
  reduce-scatter and the search direction is rebuilt with an allgather. Maps
  up to nside 4096 with MPI (#14).
- Threads (`nthreads`, default `OMP_NUM_THREADS`) for the FFTs and the numba
  kernels; $P^T$ shares one map among the threads, each owning a range of
  pixels (#3, #5, #15).
- 8 bytes per sample besides the TODs: int32 pixel and 2a in float32;
  γ cos 2a and γ sin 2a recomputed for each chunk (#6, #8, #11).
- Faster setup: counting sort by pixel and one irfft per weights array (#13);
  cos and sin from a table, 10 times faster than libm without SVML (#16).

### Documentation

- Sphinx documentation on Read the Docs, with notebooks, MPI and memory
  notes and benchmarks on G100 (#9, #17).
