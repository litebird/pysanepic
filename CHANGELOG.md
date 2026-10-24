# Changelog

## HEAD

Changes on `main` that are not in a release yet.

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
