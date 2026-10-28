# Performance

## MPI and threads

With MPI each process passes only its own data (`DetectorData` objects, or
litebird_sim observations) and all processes call `make_maps` with the same
communicator; the result is available on every process.

The PCG state is distributed by pixel: each process owns a contiguous range
of pixels, 1/N of the map, and keeps the PCG vectors, the preconditioner and
the mask only there. In each application of $P^T N^{-1} P$:

1. every process projects the full search direction on its samples ($P$),
   applies $N^{-1}$ and accumulates $P^T$ in a full-size map;
2. a **reduce-scatter** sums these maps over the processes and delivers to
   each one only its own pixels, where the PCG algebra is done;
3. an **allgather** rebuilds the full search direction on every process for
   the next $P$.

Together, the reduce-scatter and the allgather move the same data as one
`Allreduce` of the map, and every PCG operation is done once instead of on
every process. Dot products are local sums plus an `Allreduce` of a number.

Within each process, FFTs (ducc0), pixelization (ducc0) and the projection
kernels (numba) use `nthreads` threads. As in litebird_sim, the default is
`OMP_NUM_THREADS` or, if it is not set, **1 thread**, to avoid
oversubscription when several MPI processes share a node: set
`OMP_NUM_THREADS = cores per node / processes per node`. Set
`NUMBA_NUM_THREADS` to at least the same value.

With several chunks per process, the products of the PCG and the
preconditioner of the templates process **one chunk per thread**: each
thread computes $N^{-1} P v$ of its chunk with single-threaded FFTs and the
serial numba kernels, in its own buffers (~40 bytes per sample of the
longest chunk), and adds $P^T$ to a private map, summed at the end. The
chunks are long and their FFTs memory bound: on G100 (48 cores) one FFT of a
day of data (3.2 million samples) on 48 threads is only 2–4 times faster than
on one, while 48 FFTs on one thread each are 15 (float64) to 25 (float32)
times faster. With 96 chunks of a day, 48 threads and one process, 10 PCG
iterations took 14.6 s instead of 60.4 s (float32 FFTs), 16.9 s instead of
89.6 s (float64), with the same result.

The unknowns of flagged samples (`flag_method="marginalize"`, one per sample)
belong to one chunk each, so the threads add $P^T$ to them in place, and the
updates and dot products of the PCG vectors (pixels, flagged samples,
amplitudes) are numba kernels on all the threads; the dot products are summed
over a fixed number of blocks, with the same result for any number of
threads. With calibration and a Galactic mask (|b| > 10°: 17% of the samples
flagged during the iterations) these vectors are several times longer than
the map. On G100, 96 chunks of a day, 48 threads, 8 templates and the gains,
one PCG iteration took 7.2 s with a private copy of the flagged unknowns per
thread and serial vector updates, 2.6 s now (2.1 s without flagged samples);
in a 90-day e2e run with 3 calibration iterations (M1_L, 1 detector per
process) the PCG went from 990 to 370 s, the GLS from 1470 to 850 s. With
`verbose`, `make_maps` prints the time of each step of each solve (TOD and
templates, GLS setup, PCG, maps).

The private maps are used if all of them fit in 4 GB (nside 512: 75 MB per
thread); otherwise, and with one chunk or one thread, each chunk is
processed by all the threads: in $P^T$ each thread owns a range of pixels,
chosen so that the ranges hold the same number of this process's samples,
reads the pixel indices of the whole chunk but adds only the samples of its
own pixels, so the threads share one map without races. Chunks of a
`DetectorGroup` (common modes) are always processed this way.

## Memory

The TODs are used without copies, and $P$, $N^{-1}$ and $P^T$ are applied one
chunk at a time. Besides the TODs, pysanepic keeps **8 bytes per sample**
(int32 pixel and float32 2a, computed from the float64 angles; 12 bytes with
`angle_dtype=np.float64`), also during the setup, plus a few maps. γ cos 2a
and γ sin 2a are recomputed for each chunk when it is used, from a table of
cos and sin and a Taylor series of the remainder (exact to ~1e-16; numba
without SVML calls a slow scalar libm, 10 times slower). With padding
(`pad_s > 0`), γ is stored per sample (0 in the margins): 4 more bytes.

- Pass the data as a generator of `DetectorData`: then the pointings of only
  one detector are in memory at a time (litebird_sim does this).
- Maps: the PCG vectors and the 3×3 preconditioner (~300 bytes per pixel)
  are divided among the processes. Each process keeps in full only the search
  direction for $P$ and the map accumulating $P^T$ (24 bytes per pixel each,
  I/Q/U in float64, whatever the number of threads), plus during the setup
  the coverage and the preconditioner blocks (56 and 48 bytes per pixel, one
  after the other). These two are reduce-scattered one column at a time:
  with a single call Intel MPI allocates a temporary copy of the whole array.
  At nside 2048 (50 million pixels) this is about 2.4 GB per process during
  the iterations, and 2.8 GB at the peak of the setup. On G100 (nside 2048,
  8 processes × 2 threads, little data), the column-wise setup took the peak
  RSS per process from 7.6 to 5.7 GB, for 3 s more.
  On a laptop (nside 1024, 4 processes × 2 threads, little data),
  distributing the PCG state took the peak per process from 324 to 142
  bytes per pixel, before the private maps of the threads were removed.
- Templates fitted with the maps (the gains of the calibration and
  `make_maps(templates=...)`): 4 bytes per sample and template in the GLS
  (`template_dtype`, float32 by default; 8 with float64), labels int32 or one
  per chunk for an amplitude per detector. The sky model, with calibration,
  8 bytes per sample more (its binned map is put back). Measured with
  calibration and 4 TOD templates: 47 bytes per sample at the peak (163
  before 0.6).
- Flagged samples with `flag_method="marginalize"`: ~70 bytes per flagged
  sample, on the process that holds it (its share of the PCG vectors);
  negligible with `"offset"` or when the run is split (`split_gap_s`).
- Pixel indices are int32, enough up to nside 8192. With MPI, the maps
  exchanged are limited by the int32 counts of MPI to nside 4096.

See the [memory notebook](notebooks/memory).

## Benchmarks

On G100 (CINECA; nodes with 48 cores and 375 GB), LiteBIRD channel M2_L
(75 Hz, IMO 1/f noise: $f_\text{knee}$ = 20 mHz, α = 1), no HWP, simulated
with litebird_sim, one-day observations and `chunk_s=86400`, nside 512,
pysanepic 0.2.0.

| | 7 days, 16 detectors | 1 year, 128 detectors |
|---|---|---|
| samples | 7.3·10⁸ | 3.0·10¹¹ |
| resources | 1 node, 16 processes × 3 threads | 13 nodes, 10 processes × 4 threads per node |
| peak memory per process (with the simulation) | 2.2 GB | 29 GB |
| GLS time, `tol=1e-8` | 57 s (10 iterations) | 50 min (20 iterations) |
| GLS time, `tol=1e-12` | 99 s (28 iterations) | 77 min (25 iterations) |

In the one-year run the setup (pixels, angles, preconditioner) takes about 12
minutes and each application of $P^T N^{-1} P$ about 98 s, of which:

| | share |
|---|---|
| FFTs of $N^{-1}$ (ducc0) | ~60% |
| $P$ and $P^T$ | ~20% |
| γ cos 2a and γ sin 2a | ~15% before the cos/sin table, which cut it by ~5 times |
| reduce-scatter and allgather | < 2% |

The memory is dominated by the TODs (1.1 TB in float32) and the 8 bytes per
sample of pysanepic; the maps take ~0.3 GB per process.
