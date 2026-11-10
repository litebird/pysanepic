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
serial numba kernels, in its own buffers (~40 bytes per sample of its
longest assigned chunk with float64 FFTs), and adds $P^T$ to a private map,
summed at the end. With `fft_dtype=np.float32`, workers retain both float32
and float64 FFT buffers (the RHS and true residual use float64): about
52 bytes per sample with the default angle precision, rather than a memory
saving. FFT extension can increase these estimates.
Chunks are assigned deterministically to workers, balancing the total FFT
length per worker, with a fixed accumulation order inside each worker.
Repeated calls with the same inputs, thread count and chunk layout give
bitwise-identical sums regardless of task completion order. Different thread
counts can still change floating-point grouping and are compared numerically.
The chunks are long and their FFTs memory bound: on G100 (48 cores) one FFT of a
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

During setup, coverage uses at most 8 workers and a separate budget of
512 MiB per process for their full-sky accumulators (56 bytes per pixel
each). At least one accumulator is required even if it exceeds the budget.
Workers are summed in place into the first accumulator; the others are
released, without allocating another full coverage array. At high nside
this reduces coverage parallelism to limit memory. Different worker counts
can change the grouping of floating-point sums.

### Full maps per process or per node

The full maps of the products above (the map read by $P$, the sum of $P^T$,
at setup the coverage and the preconditioner blocks) and the output maps are
one copy per process with `full_maps="rank"` (the default). With
`full_maps="node"` the processes of a node share one copy, in MPI-3 shared
memory, and add $P^T$ in rounds over slabs of pixels, waiting for each other
at each round. The map read by $P$ is shared per node in both modes.

Which one to use depends on how many processes run on a node more than on
nside. On one G100 node at nside 2048 with I/Q/U (random pixels over the
whole sky, 48 × 4 × 2^20 samples in all, 10 PCG iterations, job 23433568,
`tools/cluster/benchmark_full_maps.slurm`):

| processes × threads | `full_maps` | s per iteration | node memory in the PCG |
|---|---|---|---|
| 48 × 1 | `"rank"` | 7.4–8.1 | 158 GB |
| 48 × 1 | `"node"` | 4.8–4.9 | 44 GB |
| 8 × 6 | `"rank"` | 3.3 | 39 GB |
| 8 × 6 | `"node"` | 4.9 | 30 GB |

- With many single-threaded processes per node, `"node"` is faster and uses
  a quarter of the memory: with `"rank"` the sum of 48 copies of 1.2 GB
  (the reduce-scatter) dominates the time.
- With fewer processes and threads, `"rank"` is faster and needs little
  more memory. The rounds of `"node"` cost ~50% here (and ~25% at nside 512
  with 8 × 6 in a 1-year e2e run, where it saves no memory: the TOD
  dominate).

So prefer a few processes per node with threads and `full_maps="rank"`;
use `"node"` when many single-threaded processes per node are imposed (e.g.
one per core by the calling code) and the copies of the full maps would not
fit, ~5 GB per process at nside 2048 with I/Q/U.

## Sampling alm models

A sky model or sky templates given as alm are interpolated at the pointing
with `ducc0.sht.synthesis_general`, once per run with
`CalibrationParameters(cache_model=True)`, else at each calibration
iteration. Each call has a fixed cost, a spherical harmonic transform at the
lmax of the alm, before the interpolation proper, whose cost grows with the
samples. On one G100 node (48 threads, lmax 1536):

| samples per call | spin 0 | spin 2 |
|---|---|---|
| 1000 | 0.020 s | 0.036 s |
| 1 day at 37.5 Hz | 0.041 s | 0.058 s |
| 8 days, per day | 0.015 s | 0.019 s |
| 32 days, per day | 0.012 s | 0.016 s |

`make_maps` calls it once per `DetectorData` and field. With `DetectorData`
of a day (e.g. litebird_sim observations of a day), `GLSParameters(
sample_batch=8)` interpolates 8 of them in one call per field: ~3 times less
per day, for ~50 bytes per sample of the batch (the batch is read from
`data` before it is processed). With `DetectorData` of a week or more the
fixed cost is amortized already, and longer litebird_sim observations do the
same without the memory of the batch. The default is 1. The epsilon of the
interpolation (1e-10, 1e-6 for complex64 alm) changes the time by ~10% only.
The polarized part, gamma (Q cos 2a + U sin 2a), is combined in a numba
kernel (cos and sin from the table of the PCG).

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
- Maps: the PCG vectors, the 3×3 preconditioner and the coverage, kept for
  the binned map of the sky model (~350 bytes per pixel), are divided among
  the processes. Each process keeps in full only the search
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
  ~70 bytes per run with `"offset"` (the same cost for isolated flags).
  `"act"` has ~70 bytes per Legendre coefficient and 32 bytes per run with
  more than one coefficient; the basis is evaluated without stored columns.
  High orders use discrete orthonormal polynomials in the same cut space,
  with three temporary float64 vectors per active gap (24 bytes per gap
  sample per worker), rather than an allocated samples-by-coefficients matrix.
  Constant cuts use the existing offset kernels without run metadata.
  `GapFill` has no extra PCG unknowns: without padding it stores 8 bytes
  per flag (synthetic values) and views of the TOD and flag masks, filling
  one temporary chunk per worker. With padding the filled, padded TOD is
  stored as for the other methods. Split gaps have no unknowns.
  Conditional filling (`GapFill.mode`) adds one temporary solve per chunk or
  common-mode group: a few gap-sized CG vectors and full-FFT work arrays.
  The default sine preconditioner adds one gap-sized vector of diagonal
  scales and inverse spectra for distinct long-gap lengths per detector
  (at most another gap-sized vector). Its setup and application use no dense
  gap matrices. Use `GapFill(preconditioner=None)` for cost comparisons.
  All detectors of a common-mode group are processed together. It is not
  guaranteed faster than exact marginalization; long gaps can slow this
  inner solve too. No gap unknowns persist in the map PCG.
- Prior means are kept only for template amplitudes during PCG. A full
  shift vector is allocated only for nonzero means, collectively with MPI,
  and released before PCG starts. This saves one float64 vector of the
  whole system during iterations, including 8 bytes per flagged unknown.
- Pixel indices are int32, enough up to nside 8192. With MPI, the maps
  exchanged are limited by the int32 counts of MPI to nside 4096.

See the [memory notebook](notebooks/memory).

## Benchmarks

### Comparing memory implementations

`tools/benchmark_memory.py` compares two source snapshots in separate Linux
processes, with identical seeded inputs and alternating execution order:

```bash
python tools/benchmark_memory.py --baseline-src baseline/src \
    --optimized-src optimized/src --output-dir results --nthreads 48 --repeats 2
```

It isolates high-resolution coverage and runs complete GLS solves with
uniform chunks, heterogeneous chunks (float64 and float32 FFTs), and
flagged samples with a nonzero template prior. These are synthetic memory
and numerical checks, not a mission simulation. Tiny inputs are available
with `--quick`. `tools/cluster/benchmark_memory.slurm` follows the G100 job
environment; launch it from an isolated directory containing the two
snapshots, tests, benchmark tool and checksum manifest.
For longer operator timings, use `--cases uniform heterogeneous64
--operator-products 80 --repeats 3`; the repeated products are timed in five
batches after warming the actual chunk layout. The Slurm script accepts
`timing` as its first argument to run this comparison only.

`results/comparison.json` records timings, convergence, full-array numerical
comparisons, source paths, versions, and both RSS and virtual memory. Phase
peaks are sampled every 20 ms; process peak RSS comes from the kernel's
lifetime high-water mark, including input generation and a small JIT warmup
outside the timed phases. Unused `np.empty` buffer capacity can reduce
virtual allocations without reducing RSS by the same amount. Coverage
arrays are compared in blocks; their output I/O is outside the timed phase.

### G100 memory comparison (2026-11-03)

Job 23186465 completed on `r507c07n03` (`g100_usr_prod`, one node, 48 cores).
The baseline was commit `ece6b95156a5285e7eb10ed3ee3fc534f8f62900`; the optimized
source was a frozen snapshot of the `memory_optimization` working tree.
Both used NumPy 2.5.4, Numba 0.68.0 and ducc0 0.42.0. Two fresh processes
per version and case were run in alternating order. Median process peak RSS
in decimal GB:

| synthetic case | baseline | optimized | RSS reduction |
|---|---:|---:|---:|
| coverage only, nside 1024, 8.4 million samples | 6.70 | 1.03 | 84.7% |
| uniform chunks, float64 FFTs, 25.2 million samples | 1.56 | 1.56 | 0.3% |
| heterogeneous chunks, float64 FFTs, 29.1 million samples | 2.02 | 1.88 | 7.0% |
| heterogeneous chunks, float32 FFTs, 29.1 million samples | 2.32 | 1.95 | 15.9% |
| 20% flags, nonzero shared prior, 25.2 million samples | 2.33 | 2.28 | 1.8% |

The isolated coverage calculation also fell from 1.32 to 0.47 seconds.
For heterogeneous chunks, allocated worker workspaces fell from 8.05 to
0.66 GB (float64 FFTs), or 10.47 to 0.86 GB (float32 FFTs); RSS savings are
smaller because much of the oversized `np.empty` capacity was never touched.
The flagged case saved about 42 MB of peak RSS, consistent with removing
one 41.4 MB vector of 5,180,545 system unknowns during PCG.

All compared coverage arrays, maps and shared amplitudes had zero maximum
absolute difference in these tests. Solves converged at `tol=1e-12` in 4
iterations, or 12 with flags, with identical final residuals. All 19 memory/thread
tests passed, plus the two MPI tests for priors and pixel distribution on
four ranks. These short solves alone do not establish a speedup or regression
of the iterative operator.

A longer timing check, job 23186606 on `r511c02n03` (`g100_usr_dbg`, 48 cores),
used 80 normal products per process and three fresh processes per version
and case. Median times per product were 0.0820 to 0.0790 seconds with uniform
chunks and 0.30886 to 0.30862 seconds with heterogeneous float64 chunks.
The operator timing is therefore comparable in these cases; the apparent
15% slowdown of the first, short uniform solve was not reproduced.
All final map and amplitude comparisons again had zero maximum difference.

### LiteBIRD scaling

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
