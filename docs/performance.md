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

In $P^T$ each thread owns a range of pixels, chosen so that the ranges hold
the same number of this process's samples. Every thread reads the pixel
indices of the whole chunk but adds only the samples of its own pixels, so
the threads share one map without races and without private copies. With
4–8 threads this was as fast as one private map per thread on a laptop, and
faster on G100 (3 threads). Since every thread reads all the pixel indices,
prefer several MPI processes per node with a few threads each over one
process with many threads.

## Memory

The TODs are used without copies, and $P$, $N^{-1}$ and $P^T$ are applied one
chunk at a time. Besides the TODs, pysanepic keeps **8 bytes per sample**
(int32 pixel and float32 2a, computed from the float64 angles; 12 bytes with
`angle_dtype=np.float64`), also during the setup, plus a few maps. γ cos 2a
and γ sin 2a are recomputed for each chunk when it is used. With padding
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
- Pixel indices are int32, enough up to nside 8192. With MPI, the maps
  exchanged are limited by the int32 counts of MPI to nside 4096.

See the [memory notebook](notebooks/memory).

## Benchmarks

On G100 (CINECA; nodes with 48 cores and 375 GB), LiteBIRD channel M2_L
(75 Hz, IMO 1/f noise: $f_\text{knee}$ = 20 mHz, α = 1), no HWP, simulated
with litebird_sim, one-day observations and `chunk_s=86400`, nside 512.

| | 7 days, 16 detectors | 1 year, 128 detectors |
|---|---|---|
| samples | 7.3·10⁸ | 3.0·10¹¹ |
| resources | 1 node, 16 processes × 3 threads | 16 nodes, 128 processes × 6 threads |
| peak memory per process (with the simulation) | 2.5 GB | 37.6 GB |
| GLS time, `tol=1e-12` | 108 s (28 iterations) | 62 min (25 iterations) |
| GLS time, `tol=1e-8` | 56 s (10 iterations) | |

In the one-year run each iteration takes about 105 s and the setup (pixels,
angles, preconditioner) about 15 minutes; the memory is dominated by the
TODs (1.1 TB in float32) and the 12 bytes per sample that pysanepic kept
then (3.6 TB; 8 bytes now).
