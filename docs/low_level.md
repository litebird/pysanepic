# Low-level interface

{class}`~pysanepic.GLS` works directly on pixel indices, polarization angles
and noise weights, for users who already have a pointing matrix in any
pixelization (pixels are integers in $[0, 2^{31})$, not necessarily HEALPix).
`make_maps` is a thin layer on top of it that computes pixels, angles, chunks
and weights from `DetectorData`.

Data come as *chunks*: a 2D array `(nchunk, ns)` for chunks of the same
length, or a list of 1D chunks and/or 2D blocks. `weights` has one row of
length `ns//2 + 1` per chunk, the inverse noise power spectrum on the rfft
grid, for example from {func}`~pysanepic.one_over_f_weights` or, for a
tabulated spectrum, {func}`~pysanepic.psd_weights`; chunks may share the same
array. With `fft_lengths` the FFTs run on longer lengths (zero extension,
see {func}`~pysanepic.fast_fft_length`), and `weights` are on their rfft
grid. Negative pixels mark flagged samples: samples of a chunk with the same
negative value share an unknown that is not in the maps (one value per sample
for the exact GLS without them); `apply_A` and `rhs` are then not available,
use `solve`. With `GLS(..., flag_method="sanepic")`, negative pixels instead
have zero pointing/template rows and no unknowns: their **finite, already
gap-filled TOD** is retained in the RHS (the caller fills it; `make_maps`
does this automatically). This is approximate, not exact marginalization;
`apply_A` and `rhs` are available without templates in this mode.
With `flag_method="act", flag_sampling_rate_hz=fs`, contiguous negative
pixels instead get the local Legendre coefficients of the ACT cut model;
fs is scalar or one sampling rate per chunk, needed for gap durations.
Flagged TOD is ignored and set to zero. The basis is evaluated without
stored template arrays; `nflag` counts the coefficients, not cut samples.
Use `solve`, as with the other local unknowns.
`act_ncoeff` is also accepted here: None keeps the ACT rule, an integer fixes
the count, and a callable `(lengths_samples, fs_hz)` returns a positive integer
scalar or an array of counts (one per gap, capped at its length). For example
`act_ncoeff=3` fits quadratics and `act_ncoeff=1` fits offsets.

`gap_fill`, `gap_fill_tol`, `gap_fill_maxiter` and `flag_seed` are high-level
GLSParameters settings, not arguments of GLS. To prepare a low-level SANEPIC
TOD, use {func}`~pysanepic.conditional_gap_fill` on the noise residual after
subtracting a sky estimate, then pass the filled residual to GLS.solve with
the flagged pointing rows negative. Restore the sky estimate separately
for a full-sky map; make_maps performs that subtraction/restoration itself.
The helper takes `rng` (a seed or Generator), `draw` (False for the mean),
`tol`, `maxiter`, and optionally `fft_length`, `coupling` and `mode_psd`.

`common_modes` correlates chunks observed at the same times, and
`templates`, a list of `(chunk, values, labels)`, fits template amplitudes
with the map (the TOD of the chunk gets `values * a[labels]`, labels -1: none;
amplitudes local to each process, returned in `info["amplitudes"]`;
`shared_templates` have amplitudes shared by all the processes, with
`shared_groups` summing to zero, in `info["shared_amplitudes"]`; several
templates with one amplitude each in a chunk can be given as the columns of
one array, `(chunk, values (n, ncol), labels (ncol,))`, applied in one pass
over the chunk instead of one per template; see
{class}`~pysanepic.GLS`). `apply_A` and `rhs` are not available with templates.
`shared_covariance()` gives the covariance of the shared amplitudes (one PCG
solve each), `set_templates` replaces the templates keeping the rest of the setup, and
`binned(tod)` gives the binned (white-noise) map of timelines laid out like
`pix`.

```python
import numpy as np
from pysanepic import GLS, one_over_f_weights

rng = np.random.default_rng(0)
npix, nchunk, ns, fs = 100, 8, 2**14, 10.0
sigma = 1e-4  # white-noise rms per sample [K]

pix = rng.integers(0, npix, (nchunk, ns))  # pixel of each sample (any pixelization)
psi = rng.uniform(0, np.pi, (nchunk, ns))  # polarization angle of each sample
w = one_over_f_weights(ns, fs, sigma, fknee_hz=0.05, alpha=1.0, fmin_hz=1e-4)
weights = np.tile(w, (nchunk, 1))  # inverse noise PSD of each chunk (rfft grid)

sky = rng.normal(0, 1e-3, (3, npix))
tod = sky[0, pix] + sky[1, pix] * np.cos(2 * psi) + sky[2, pix] * np.sin(2 * psi)

gls = GLS(pix, psi, weights)  # pixels without hits are not solved
m, info = gls.solve(tod, tol=1e-12)  # m: (3, npix) I, Q, U
print(info["iterations"], info["converged"], np.abs(m - sky).max())
```

Noiseless data give back the input sky to numerical precision. Other options
of `GLS`:

- `pol_efficiency` (scalar, per chunk or per sample), or
  `pol_angles=(angle2, gamma)` with 2a already computed and reduced to
  [−π, π) (see `two_angle`; used without copies) and γ per chunk or per
  sample;
- `psi=None` for intensity only;
- `comm` and `nthreads`, as in `make_maps`;
- `preconditioner`, `min_pol_rcond`, `min_pol_hits`, `angle_dtype`.
- `coarse`: a function giving the coarse pixel of each pixel label (−1: none)
  for the two-level preconditioner; make_maps builds it from `coarse_nside`.

`GLS.apply_A` ($P^T N^{-1} P$ on a map) and `GLS.rhs` ($P^T N^{-1} d$) are
public, e.g. to use another solver. With MPI they work on the pixels owned by
the process, `[gls.lo, gls.hi)`, as arrays `(ncomp, gls.nloc)` (without MPI,
the whole map), and must be called by every process. `GLS.full` gathers such
an array into the full map on every process, and `GLS.to_healpix` expands the
solution to full HEALPix maps (after `solve`, also on one process only). `solve`
returns the full map on every process.
