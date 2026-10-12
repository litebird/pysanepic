# Low-level interface

{class}`~pysanepic.GLS` works directly on pixel indices, polarization angles
and noise weights, for users who already have a pointing matrix in any
pixelization (pixels are integers in $[0, 2^{31})$, not necessarily HEALPix).
`make_maps` is a thin layer on top of it that computes pixels, angles, chunks
and weights from `DetectorData`.

Data come as *chunks*: a 2D array `(nchunk, ns)` for chunks of the same
length, or a list of 1D chunks and/or 2D blocks. `weights` has one row of
length `ns//2 + 1` per chunk, the inverse noise power spectrum on the rfft
grid, for example from {func}`~pysanepic.one_over_f_weights`; chunks may share
the same array.

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

`GLS.apply_A` ($P^T N^{-1} P$ on a map) and `GLS.rhs` ($P^T N^{-1} d$) are
public, e.g. to use another solver; `GLS.to_healpix` expands the solution to
full HEALPix maps.
