# Performance

## MPI and threads

With MPI each process passes only its own data (`DetectorData` objects, or
litebird_sim observations) and all processes call `make_maps` with the same
communicator. The maps are replicated on all processes and summed with one
`Allreduce` per iteration; the result is available on every process.

Within each process, FFTs (ducc0), pixelization (ducc0) and the projection
kernels (numba) use `nthreads` threads. As in litebird_sim, the default is
`OMP_NUM_THREADS` or, if it is not set, **1 thread**, to avoid
oversubscription when several MPI processes share a node: set
`OMP_NUM_THREADS = cores per node / processes per node`. Set
`NUMBA_NUM_THREADS` to at least the same value.

$P^T$ keeps one private copy of the I/Q/U map per thread (about 75 MB per
thread at nside 512), so the threads never write to the same memory.

## Memory

The TODs are used without copies, and $P$, $N^{-1}$ and $P^T$ are applied one
chunk at a time. Besides the TODs, pysanepic keeps **12 bytes per sample**
(int32 pixel, float32 γ cos 2a and γ sin 2a, computed from the float64
angles; 20 bytes with `angle_dtype=np.float64`), also during the setup, plus a
few maps.

- Pass the data as a generator of `DetectorData`: then the pointings of only
  one detector are in memory at a time (litebird_sim does this).
- The maps are replicated on every process: I/Q/U at nside 2048 takes 1.2 GB
  per vector, and the PCG needs about ten of them (vectors, 3×3
  preconditioner, one private map per thread), so above nside ~1024 the
  resolution, not the data, sets the memory.
- Pixel indices are int32, enough up to nside 8192.

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
TODs (1.1 TB in float32) and the 12 bytes per sample (3.6 TB).
