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
seconds, with a power spectrum $P(f)$: a 1/f model ("toast" or "keshner", see
[conventions](usage.md#conventions)) or a tabulated spectrum.
$N^{-1}$ is applied as a circulant filter, in Fourier space with ducc0:
$N^{-1} t = \mathrm{irfft}(\mathrm{rfft}(t)/P(f))$, one chunk at a time.
`make_maps` runs the FFT of each chunk on the smallest even length with small
prime factors ({func}`~pysanepic.fast_fft_length`): a chunk of awkward
length, e.g. split at a gap, is zero-extended and the result truncated, so
the weight is the corresponding block of a slightly longer circulant (still
positive definite: no bias). An FFT of 6480001 samples is 7 times slower than
one of 6553600.
With $f_\text{min} = 0$ the zero frequency has zero weight and the mean of each
chunk is not used.

Neglected:

- correlations between different chunks, including two `DetectorData` of the
  same detector;
- the non-periodicity of the noise within a chunk: an FFT joins the end of a
  chunk to its start. `pad_s` moves this junction outside the data, see
  [padding](parameters.md#padding).

The noise parameters or spectra are given by the user (or by litebird_sim,
which uses those of the simulation), or estimated from the residual of a first
map with {func}`~pysanepic.estimate_noise` (see
[estimating the noise](usage.md#estimating-the-noise)).

## Correlated noise

For the chunks of a {class}`~pysanepic.DetectorGroup`, observed at the same
times, $N(f) = D(f) + U\,\Lambda(f)\,U^T$ per frequency, with $D$ the own
noise of the detectors, $U$ the couplings (detectors × modes) and $\Lambda$
the spectra of the $r$ modes. With the Woodbury identity

$$N^{-1} = D^{-1} - D^{-1} U M U^T D^{-1}, \qquad
M = (I + \Lambda\, U^T D^{-1} U)^{-1}\,\Lambda,$$

$N^{-1}$ costs $O(n_\text{det}\, r + r^3)$ per frequency besides the FFTs. It is
applied in two passes over the detectors of the group: the first accumulates
$z = \sum_i U_i D_i^{-1} \tilde t_i$ ($r$ spectra), the second computes
$D_i^{-1}(\tilde t_i - U_i M z)$ for each detector, recomputing its FFT, so
that only one detector is in memory at a time (1.5 times the FFTs of
uncorrelated noise). The block preconditioner uses $D$ only, so the PCG needs
more iterations when the modes are strong, unless the two-level preconditioner
is used (see [below](#two-level-preconditioner)). Q/U signal in pixels where Q/U are not
solved (too few angles) is not modelled, and with correlated noise it leaks
into other pixels.

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

## Two-level preconditioner

A preconditioner local to each pixel cannot fix the slowest modes of the
PCG, the large-scale modes of I: with common modes between detectors there are
a dozen eigenvalues of the preconditioned system much smaller than the others,
and the iterations double. With `coarse_nside` the preconditioner becomes

$$M_2 = M + Z\,E^{-1} Z^T, \qquad E = Z^T P^T N^{-1} P\, Z,$$

with $Z$ the indicators (in I) of the pixels of a coarse HEALPix map: the
coarse I map is solved exactly at every iteration. $E$ is computed at setup,
one product by $P^T N^{-1} P$ per coarse pixel, about one PCG iteration each;
afterwards $M_2$ costs two sums over the coarse pixels. This is the additive
two-level preconditioner of Grigori, Stompor & Szydlarski (2012), with a
coarse space of piecewise-constant maps instead of eigenvectors. The C++
SANEPIC uses the Jacobi preconditioner only.

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

## Flagged samples

Flagged samples point to *local unknowns*: intensity-only values that are
not pixels of the map and belong to the process holding the samples, so they
take no part in the reduce-scatter and allgather of the PCG; their TOD is set
to 0. With one unknown per sample (`flag_method="marginalize"`) the solution
is the exact GLS without the flagged samples: each of them is fitted freely,
which is the same as removing it from the data while keeping the noise
correlations of the others. With one unknown per run (`"offset"`) the zeros
are exactly described by the offset, so there is no bias. The preconditioner
of a local unknown is its diagonal element of $P^T N^{-1} P$. Long runs can
instead split the chunk (`split_gap_s`), see
[flagged samples](parameters.md#flagged-samples).

## Calibration

The TOD is modeled as $d = g\,(P m + D) + n$, with $D$ the dipole and one gain
$g$ per detector and period. Around the current gains,
$d/g - D = P m + T\,\delta g + n$ with $T = D + P m_\text{prev}$ (the
template, piecewise in time: one amplitude per period). The template
amplitudes are local unknowns like those of flagged samples, with $T$ as
their column of the pointing matrix, so the PCG solves for map and
$\delta g$ at once, without MPI communication for the gains; their
preconditioner is the exact diagonal of $T^T N^{-1} T$, computed at setup
from the autocorrelation of each amplitude's template (an FFT of twice its
span) and the first row of the circulant $N^{-1}$. Then
$g \leftarrow g\,(1 + \delta g)$, `n_iter` times; each iteration starts the
PCG from the maps of the previous one.

Templates are not applied to samples in pixels whose Q/U are not solved: their
polarized signal is not in the model and would bias the amplitudes. For the
same reason a `mask` flags its samples during the iterations (with
`flag_method`), and one last solve with the final gains makes the maps
with all the samples. A gain common to all periods and detectors is
degenerate with the scale of the map except through the part of $D$ that is
not fixed on the sky, the orbital dipole: it sets the absolute calibration.
In noiseless tests the iterations converge as
$3\times10^{-4} \to 3\times10^{-8} \to 5\times10^{-15}$.

## Sky model and bandpass templates

A sky model $S$ is subtracted from the TOD, $d - S(t) = P\,m_\text{res} + n$,
and the maps are $m_\text{res} + \bar S$, with $\bar S$ the binned map of
$S(t)$ ($(P^T P)^{-1} P^T S$, white weights). For a model sampled in the pixel
$S(t) = P\,\bar S$ and nothing changes where Q/U are solved; for alm
interpolated at the pointing, $S(t) - P\,\bar S$ is the structure of the
model below the pixel size, which the GLS would otherwise spread through the
1/f weights. If the model is exact, $m_\text{res}$ is noise only and the maps
are the binned maps of the signal.

Bandpass templates $B_k(t)$ enter the model as $\sum_k a_{ik} B_k(t)$, with
one amplitude per template and detector. These amplitudes are *shared*
unknowns: a detector has `DetectorData` on several processes, so they are
kept on the first process, broadcast before each product and summed over the
processes after it (an allreduce of a few numbers per PCG iteration). The
common part $\sum_i a_{ik}$ multiplies a sky signal and is degenerate with
the map: the constraint $\sum_i a_{ik} = 0$ is imposed by projecting the
shared part of the PCG vectors (the operator becomes $\Pi A \Pi$, with $\Pi$
the projector on the constraint). With calibration all of it is linearized
together: $d/g - D - S - \sum a B = P\,m_\text{res} + T\,\delta g +
\sum B\,\delta a + n$, $T = D + S + \sum a B + P\,m_\text{res,prev}$.

## Differences from the C++ SANEPIC

- 3×3 block preconditioner (the C++ code has the Jacobi one, also available);
- Q/U solved only where the angle coverage allows it (`min_pol_rcond`);
- polarization efficiency, ideal HWP and rotation between coordinate systems;
- noise weights computed from (NET, $f_\text{knee}$, α, $f_\text{min}$),
  or from a tabulated spectrum given as arrays rather than read from files;
- zero padding (`pad_fill="zeros"`) besides SANEPIC's extrapolation;
- flagged samples (the C++ code reads the options `-G`, `-g` and `-M` but does
  not use them);
- calibration on the dipole and template fitting;
- MPI without a master process: each process handles its own data, and the
  PCG state is distributed over the processes by pixel (see
  [MPI](performance.md#mpi-and-threads)); threads within each process;
- a few bugs of the reference code fixed (see the history of `cpp/`).

## Limitations

- No correlations between chunks of time; correlations between detectors only
  through common modes of a `DetectorGroup` (see above).
- No data splits (half-mission, detector subsets) in one call: run
  `make_maps` once per split.
- Every process still holds two full-size maps, the search direction for $P$
  and the map accumulating $P^T$, so the memory per process grows with nside
  (see [memory](performance.md#memory)).
