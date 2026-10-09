# pysanepic

GLS map-making with correlated (1/f) noise for CMB experiments, in Python.

pysanepic is a Python port of [SANEPIC](https://github.com/patanch/SANEPIC)
([Patanchon et al. 2008, ApJ 681, 708](https://arxiv.org/abs/0711.3462)). It solves the generalized least-squares
problem (PᵀN⁻¹P) m = PᵀN⁻¹d with preconditioned conjugate gradients, applying
N⁻¹ in Fourier space on chunks of data, and returns I, Q, U HEALPix maps.

- Pure Python with numba, scipy.fft and ducc0; runs with MPI (mpi4py).
- Supports an ideal HWP, polarization efficiency and any coordinate system.
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
result = sim.make_sanepic_gls_map(nside=64, chunk_s=3600.0)
result.maps       # (3, npix) I, Q, U; healpy.UNSEEN where not solved
result.hit_map    # samples per pixel
```

Until [litebird_sim#568](https://github.com/litebird/litebird_sim/pull/568) is
merged, install litebird_sim from the `sanepic_gls` branch. A complete example,
compared with the litebird_sim binner and destriper, is in
[`notebooks/lbs_example.ipynb`](notebooks/lbs_example.ipynb).

## Usage without litebird_sim

Any pipeline, or data read from disk, can be used by filling one
`DetectorData` per detector and contiguous stretch of time:

```python
import numpy as np
from pysanepic import DetectorData, make_maps

data = []
for name in ["det0", "det1"]:
    f = np.load(f"{name}.npz")  # your own files: one per detector and time span
    data.append(
        DetectorData(
            tod=f["tod"],  # K
            theta=f["theta"], phi=f["phi"], psi=f["psi"],  # rad
            coordinates="E",  # pointings in Ecliptic coordinates
            hwp_angle=f["hwp_angle"],  # rad; None without HWP
            pol_angle_rad=float(f["pol_angle"]),
            sampling_rate_hz=19.0,
            net_ukrts=40.0, fknee_hz=0.02, alpha=1.0, fmin_hz=1e-5,
        )
    )

result = make_maps(data, nside=64, coordinates="G", chunk_s=3600.0)
print(result.converged, result.iterations)
i_map, q_map, u_map = result.maps  # healpy.UNSEEN where not solved
```

### `DetectorData` fields and conventions

| Field | Meaning |
|---|---|
| `tod` | Time-ordered data, in K |
| `theta`, `phi` | Colatitude and longitude of the pointing [rad] |
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
| `chunk_s` | 3600 | N⁻¹ is applied on chunks of this duration [s]; longer chunks capture lower frequencies |
| `pol` | `True` | Solve for I, Q, U, or I only |
| `tol`, `maxiter` | 1e-12, 2000 | PCG stops when \|r\|²/\|b\|² < `tol` |
| `comm` | `None` | mpi4py communicator; each rank passes only its own data |

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

- `test_vs_sanepic.py`: maps agree with the C++ SANEPIC to ~1e-6; noiseless
  data give back the input sky.
- `test_mapmaker.py`: `make_maps` recovers the input sky to machine precision
  with HWP, polarization efficiency and coordinate rotation.
- `test_mpi.py`: MPI and serial runs agree.

## License

GPL-3.0, like litebird_sim.
