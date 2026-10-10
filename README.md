# pysanepic

GLS map-making with correlated (1/f) noise for CMB experiments, in Python.

pysanepic is a Python port of [SANEPIC](https://github.com/patanch/SANEPIC)
([Patanchon et al. 2008, ApJ 681, 708](https://arxiv.org/abs/0711.3462)). It solves the generalized least-squares
problem (PᵀN⁻¹P) m = PᵀN⁻¹d with preconditioned conjugate gradients, applying
N⁻¹ in Fourier space on chunks of data, and returns I, Q, U HEALPix maps.

- Designed for experiments **without HWP**, where 1/f noise affects both
  intensity and polarization. An ideal HWP is supported too: pysanepic only
  needs the TOD and the pointings (and the HWP angle, if any).
- Pure Python with numba, scipy.fft and ducc0; runs with MPI (mpi4py).
- Supports polarization efficiency and any coordinate system.
- Independent of any simulation pipeline: data are passed as plain arrays.
- Validated against the original C++ code, which is kept in `cpp/`.

## Installation

```bash
pip install git+https://github.com/litebird/pysanepic
pip install "pysanepic[mpi] @ git+https://github.com/litebird/pysanepic"   # with MPI
```

## Usage with litebird_sim

The interface lives in [litebird_sim](https://github.com/litebird/litebird_sim):
pointings, HWP angle, TOD and the 1/f noise parameters used to simulate the
noise are passed to pysanepic automatically.

```python
from pysanepic import GLSParameters

result = sim.make_sanepic_gls_map(nside=64, params=GLSParameters(chunk_s=3600.0))
result.maps       # (3, npix) I, Q, U; healpy.UNSEEN where not solved
result.hit_map    # samples per pixel
```

Until [litebird_sim#568](https://github.com/litebird/litebird_sim/pull/568) is
merged, install litebird_sim from the `sanepic_gls` branch.

Example notebooks:

- [`lbs_example.ipynb`](notebooks/lbs_example.ipynb): one day of LiteBIRD
  without HWP and with 1/f noise, compared with the litebird_sim binner and
  destriper;
- [`preconditioner.ipynb`](notebooks/preconditioner.ipynb): block vs SANEPIC's
  Jacobi preconditioner, and the Q/U conditioning criterion;
- [`padding.ipynb`](notebooks/padding.ipynb): chunk length and padding, with
  the IMO 1/f noise and with a steeper 1/f noise.
- [`memory.ipynb`](notebooks/memory.ipynb): generators of `DetectorData`, pixel
  input and memory per sample (no litebird_sim needed).

## Usage without litebird_sim

Any pipeline, or data read from disk, can be used by providing one
`DetectorData` per detector and contiguous stretch of time, as a list or, to
save memory, as a generator (pysanepic then keeps the pointings of only one
detector at a time):

```python
import numpy as np
from pysanepic import DetectorData, GLSParameters, make_maps


def data():
    for name in ["det0", "det1"]:
        f = np.load(f"{name}.npz")  # your own files: one per detector and time span
        yield DetectorData(
            tod=f["tod"],  # K
            theta=f["theta"], phi=f["phi"], psi=f["psi"],  # rad
            coordinates="E",  # pointings in Ecliptic coordinates
            hwp_angle=f["hwp_angle"],  # rad; None without HWP
            pol_angle_rad=float(f["pol_angle"]),
            sampling_rate_hz=19.0,
            net_ukrts=40.0, fknee_hz=0.02, alpha=1.0, fmin_hz=1e-5,
        )


result = make_maps(data(), nside=64, coordinates="G", params=GLSParameters(chunk_s=3600.0))
print(result.converged, result.iterations)
i_map, q_map, u_map = result.maps  # healpy.UNSEEN where not solved
```

### `DetectorData` fields and conventions

| Field | Meaning |
|---|---|
| `tod` | Time-ordered data, in K |
| `theta`, `phi` | Colatitude and longitude of the pointing [rad] |
| `pix`, `nside` | Alternative to `theta`, `phi`: HEALPix RING pixel of each sample, at the `nside` and in the `coordinates` of the output maps |
| `psi` | Orientation of the detector frame [rad] |
| `coordinates` | Coordinate system of the pointings: `"E"` (default), `"G"` or `"C"` |
| `hwp_angle` | HWP angle per sample [rad], or `None` |
| `pol_angle_rad` | Polarization angle of the detector [rad] |
| `pol_efficiency` | Polarization efficiency γ (default 1) |
| `sampling_rate_hz` | Sampling rate [Hz] |
| `net_ukrts`, `fknee_hz`, `alpha`, `fmin_hz` | 1/f noise model, see below |

The conventions are the same as litebird_sim:

- signal model: d = I + γ (Q cos 2a + U sin 2a), with polarization angle
  a = ψ + pol_angle without HWP, a = ψ + 2·hwp_angle − pol_angle with an ideal HWP;
- noise power spectrum: P(f) = σ² (f^α + f_knee^α) / (f^α + f_min^α), with
  σ = NET·√f_s (white-noise rms per sample);
- pointings are rotated to the output `coordinates` by pysanepic, including the
  correction of ψ.

### `make_maps` parameters

| Parameter | Default | Meaning |
|---|---|---|
| `nside` | | HEALPix resolution of the output maps (RING) |
| `coordinates` | `"G"` | Coordinate system of the output maps |
| `params` | `GLSParameters()` | Settings of the map-maker, see below |
| `comm` | `None` | mpi4py communicator; each rank passes only its own data |
| `nthreads` | `None` | Threads per process (FFTs, ducc0, numba); `None`: `OMP_NUM_THREADS` if set, else 1 |

### `GLSParameters` fields

The same object is passed by litebird_sim (`Simulation.make_sanepic_gls_map(params=...)`),
so new options need no change on the litebird_sim side.

| Field | Default | Meaning |
|---|---|---|
| `chunk_s` | 3600 | N⁻¹ is applied on chunks of this duration [s]; longer chunks capture lower frequencies |
| `pad_s` | 0 | Padding added on both sides of each chunk [s], so that the FFT wrap-around falls outside the data (SANEPIC's "inpaint"); each margin is fitted by an offset not included in the maps |
| `pad_fill` | `"zeros"` | `"zeros"` (no bias) or `"extrapolate"` (SANEPIC's linear extrapolation tapered to the chunk mean; better with a steep 1/f noise, slightly biased otherwise) |
| `pol` | `True` | Solve for I, Q, U, or I only |
| `min_pol_rcond` | 1e-2 | Q/U are solved only in pixels whose polarization-angle coverage has a reciprocal condition number ≥ this (and ≥ 4 hits); elsewhere only I. 0 disables the check, as in SANEPIC |
| `preconditioner` | `"block"` | `"block"`: exact 3×3 I/Q/U block of PᵀN⁻¹P per pixel; `"jacobi"`: original SANEPIC (I diagonal, Q/U = 2/diag_I) |
| `tol`, `maxiter` | 1e-12, 2000 | PCG stops when \|r\|²/\|b\|² < `tol` |
| `angle_dtype` | `np.float32` | Storage of γ cos 2a, γ sin 2a: float32 (angle error ~1e-7 rad, i.e. 0.02 mas) or `np.float64` (exact, 8 more bytes per sample) |

### MPI and threads

With MPI each process passes only its own data (`DetectorData` objects, or
litebird_sim observations); the maps are replicated on all processes and
summed with one `Allreduce` per iteration. Within each process, FFTs, ducc0 and
numba kernels use `nthreads` threads. As in litebird_sim, the default is
`OMP_NUM_THREADS` or, if it is not set, **1 thread**, to avoid oversubscription
when several MPI processes share a node: set
`OMP_NUM_THREADS = cores per node / processes per node`.

The projection P and its transpose Pᵀ are parallel numba kernels; Pᵀ keeps one
private copy of the I/Q/U map per thread (about 75 MB per thread at nside 512).

### Memory

The TODs are used without copies, and P, N⁻¹ and Pᵀ are applied one chunk at a
time. Besides the TODs, pysanepic keeps 12 bytes per sample (int32 pixel,
float32 γ cos 2a and γ sin 2a; 20 with `angle_dtype=np.float64`), 16 during the
setup, plus a few maps. The maps are replicated on every process: I/Q/U at
nside 2048 takes 1.2 GB per vector, and the PCG needs about ten of them
(vectors, 3×3 preconditioner, one private map per thread), so the
resolution, not the data, sets the memory above nside ~1024. Pixel indices
are int32, which is enough up to nside 8192. With
litebird_sim the pointings are computed one detector at a time and never stored.
On G100 (16 detectors at 75 Hz for 7 days, nside 512, 16 processes × 3 threads),
the peak memory per process is 2.7 GB, of which 1.2 GB are the simulation.

### Low-level interface

`pysanepic.GLS` works directly on pixel indices, polarization angles and noise
weights (one row per chunk), for users who already have a pointing matrix.

## Tests and validation

```bash
pip install -e ".[test]"
make -C cpp ARCH=linux        # or ARCH=mac: builds the C++ reference
pytest                        # includes the comparison with C++ SANEPIC
mpirun -n 4 python tests/test_mpi.py
```

- `test_vs_sanepic.py`: with SANEPIC's preconditioner, maps agree with the C++
  SANEPIC to ~1e-6; the block preconditioner converges to the same map in a
  fraction of the iterations; noiseless data give back the input sky.
- `test_mapmaker.py`: `make_maps` recovers the input sky to machine precision
  with HWP, polarization efficiency and coordinate rotation.
- `test_mpi.py`: MPI and serial runs agree.

## License

GPL-3.0, like litebird_sim.
