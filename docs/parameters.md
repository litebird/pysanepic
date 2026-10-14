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

## Angle precision

Twice the polarization angle, 2a, is computed in float64, reduced to
[−π, π) and stored in float32 by default, an error of ~1e-7 rad on the angle,
irrelevant for any application; γ cos 2a and γ sin 2a are recomputed for each
chunk when it is used. `angle_dtype=np.float64` stores 2a exactly at the cost
of 4 more bytes per sample.
