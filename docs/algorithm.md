# Algorithm

## The GLS problem

The data of a detector are modelled as $d = P m + n$, where $m$ are the I, Q,
U maps, $P$ projects them to the timeline,
$(Pm)_t = I_{p_t} + \gamma\,(Q_{p_t}\cos 2a_t + U_{p_t}\sin 2a_t)$, and $n$ is
Gaussian noise with covariance $N$. The generalized least-squares map

$$
\hat m = (P^T N^{-1} P)^{-1} P^T N^{-1} d
$$

is the minimum-variance unbiased estimate of $m$. pysanepic solves
$(P^T N^{-1} P)\,\hat m = P^T N^{-1} d$ with preconditioned conjugate gradients
(PCG), never building $P^T N^{-1} P$: each iteration applies $P$, $N^{-1}$
and $P^T$ to the whole data set once.

## Noise model

The noise of each detector is assumed stationary within a chunk of `chunk_s`
seconds, with the power spectrum
$P(f) = \sigma^2 (f^\alpha + f_\text{knee}^\alpha)/(f^\alpha + f_\text{min}^\alpha)$.
$N^{-1}$ is applied as a circulant filter, in Fourier space with ducc0:
$N^{-1} t = \mathrm{irfft}(\mathrm{rfft}(t)/P(f))$, one chunk at a time.
With $f_\text{min} = 0$ the zero frequency has zero weight and the mean of each
chunk is not used.

Neglected:

- correlations between different chunks, including two `DetectorData` of the
  same detector;
- correlations between detectors (common-mode noise);
- the non-periodicity of the noise within a chunk: an FFT joins the end of a
  chunk to its start. `pad_s` moves this junction outside the data, see
  [padding](parameters.md#padding).

The noise parameters are given by the user (or by litebird_sim, which uses
those of the simulation); they are not estimated from the data.

## Preconditioner

The default preconditioner is the inverse of the exact 3×3 I/Q/U block of
$P^T N^{-1} P$ in each pixel. Since $N^{-1}$ is circulant within a chunk, with
first row $c = \mathrm{irfft}(1/P(f))$, the block of pixel $p$ is

$$
\sum_{\text{chunks}} \sum_{t, t' \in p} c_{(t - t') \bmod n}\; w_t w_{t'}^T,
\qquad w_t = (1, \gamma\cos 2a_t, \gamma\sin 2a_t),
$$

computed at setup over all pairs of samples of a chunk falling in the same
pixel. For white noise it is the inverse covariance of the binned map.
SANEPIC's `"jacobi"` preconditioner keeps the exact I diagonal and uses
$2/\text{diag}_I$ for Q and U.

## Iterations

The PCG starts from $m = 0$ and stops when
$\lvert r\rvert^2/\lvert b\rvert^2 <$ `tol`, with $b = P^T N^{-1} d$ (the
SANEPIC criterion). The true residual $b - P^T N^{-1} P m$ is recomputed every
10 iterations to avoid the drift of the recursive one. Pixels without hits,
and Q, U in pixels with a poor angle coverage, are excluded from the
unknowns.

The slowest modes are the I monopole and the largest-scale temperature
harmonics, which 1/f noise constrains least: with the LiteBIRD IMO noise the
PCG needs a few tens of iterations, while a steep 1/f noise
($f_\text{knee}$ = 100 mHz, α = 2) needs hundreds.

## Padding

With `pad_s > 0`, each chunk is extended by `pad_s` seconds on both sides,
filled with zeros or with an extrapolation of the data. Each margin is
assigned to its own *virtual pixel*, an intensity offset solved together with
the maps and then discarded, so the padding does not enter the maps.

## Differences from the C++ SANEPIC

- 3×3 block preconditioner (the C++ code has the Jacobi one, also available);
- Q/U solved only where the angle coverage allows it (`min_pol_rcond`);
- polarization efficiency, ideal HWP and rotation between coordinate systems;
- noise weights computed from (NET, $f_\text{knee}$, α, $f_\text{min}$)
  rather than read from files;
- zero padding (`pad_fill="zeros"`) besides SANEPIC's extrapolation;
- MPI without a master process: each process handles its own data, and the
  PCG state is distributed over the processes by pixel (see
  [MPI](performance.md#mpi-and-threads)); threads within each process;
- a few bugs of the reference code fixed (see the history of `cpp/`).

## Limitations

- No flags: all samples of a `DetectorData` are used. Split the data around
  gaps into several `DetectorData`.
- No correlations between chunks or between detectors (see above).
- No data splits (half-mission, detector subsets) in one call: run
  `make_maps` once per split.
- Every process still holds two kinds of full-size maps, the search direction
  for $P$ and one private map per thread for $P^T$, so the memory per process
  grows with nside (see [memory](performance.md#memory)).
