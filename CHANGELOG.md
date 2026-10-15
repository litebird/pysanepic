# Changelog

## HEAD

Changes on `main` that are not in a release yet.

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
