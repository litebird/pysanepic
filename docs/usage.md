# Usage

## With litebird_sim

The interface lives in [litebird_sim](https://github.com/litebird/litebird_sim)
(on `master`, not yet in a release): pointings, HWP angle, TOD
and the 1/f noise parameters used to simulate the noise are passed to
pysanepic automatically.

```python
from pysanepic import GLSParameters

# sim: a litebird_sim.Simulation with pointings, TODs and 1/f noise
result = sim.make_sanepic_gls_map(nside=512, params=GLSParameters(chunk_s=86400.0, tol=1e-8))
result.maps       # (3, npix) I, Q, U; healpy.UNSEEN where not solved
result.hit_map    # samples per pixel
```

The low-level function `litebird_sim.make_sanepic_gls_map` works on a list of
observations and selects the TOD components to use. See the
[litebird_sim documentation](https://litebird-sim.readthedocs.io/en/master/mapmaking.html)
and the [example notebook](notebooks/lbs_example).

## Without litebird_sim

Any pipeline, or data read from disk, can be used by providing one
{class}`~pysanepic.DetectorData` per detector and contiguous stretch of time,
as a list or, to save memory, as a generator (pysanepic then keeps the
pointings of only one detector at a time):

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
            hwp_angle=None,  # rad, per sample; None without HWP
            pol_angle_rad=float(f["pol_angle"]),
            sampling_rate_hz=19.0,
            net_ukrts=40.0, fknee_hz=0.02, alpha=1.0, fmin_hz=1e-5,
        )


result = make_maps(data(), nside=64, coordinates="G", params=GLSParameters(chunk_s=3600.0))
print(result.converged, result.iterations)
i_map, q_map, u_map = result.maps  # healpy.UNSEEN where not solved
```

With MPI, every process calls `make_maps` with its own data and the same
`comm`; see [MPI and threads](performance.md#mpi-and-threads).

## `DetectorData` fields

| Field | Meaning |
|---|---|
| `tod` | Time-ordered data, in K |
| `theta`, `phi` | Colatitude and longitude of the pointing [rad] |
| `pix`, `nside`, `ordering` | Alternative to `theta`, `phi`: HEALPix pixel of each sample, at the `nside` and in the coordinates of the output maps; `ordering` `"RING"` (default) or `"NESTED"`. Pixels outside the map raise an error |
| `psi` | Orientation of the detector frame [rad] |
| `coordinates` | Coordinate system of the pointings and `psi`: `"E"`, `"G"` or `"C"`. Required with `theta`, `phi`; with `pix`, `None` (default) means the output coordinates |
| `hwp_angle` | HWP angle per sample [rad], or `None` |
| `pol_angle_rad` | Polarization angle of the detector [rad] |
| `pol_efficiency` | Polarization efficiency γ (default 1) |
| `sampling_rate_hz` | Sampling rate [Hz] |
| `net_ukrts`, `fknee_hz`, `alpha`, `fmin_hz` | 1/f noise model, see below |

A detector observed in several stretches of time (e.g. one per day, as the
observations of litebird_sim) is passed as several `DetectorData`. The
stretches are treated as independent: see [chunks](parameters.md#chunk-length).

## Conventions

The conventions are the same as litebird_sim:

- signal model: $d = I + \gamma\,(Q \cos 2a + U \sin 2a)$, with polarization
  angle $a = \psi + \text{pol\_angle}$ without HWP and
  $a = \psi + 2\,\text{hwp\_angle} - \text{pol\_angle}$ with an ideal HWP;
- noise power spectrum:
  $P(f) = \sigma^2 \dfrac{f^\alpha + f_\text{knee}^\alpha}{f^\alpha + f_\text{min}^\alpha}$,
  with $\sigma = \text{NET}\sqrt{f_s}$ the white-noise rms per sample;
- pointings are rotated to the output `coordinates` by pysanepic, including the
  correction of ψ.

## `make_maps` parameters

| Parameter | Default | Meaning |
|---|---|---|
| `data` | | Iterable of `DetectorData` local to this process |
| `nside` | | HEALPix resolution of the output maps (RING) |
| `coordinates` | `"G"` | Coordinate system of the output maps |
| `params` | `GLSParameters()` | Settings of the map-maker, see [Parameters](parameters.md) |
| `comm` | `None` | mpi4py communicator; each rank passes only its own data |
| `nthreads` | `None` | Threads per process (FFTs, ducc0, numba); `None`: `OMP_NUM_THREADS` if set, else 1 |
| `verbose` | `False` | Print the residual at each iteration (rank 0) |

## Output

`make_maps` returns a {class}`~pysanepic.MapResult`:

| Field | Meaning |
|---|---|
| `maps` | `(3, npix)` I, Q, U in K (`(1, npix)` with `pol=False`), RING ordering |
| `hit_map` | Number of samples per pixel |
| `nside`, `coordinates` | Pixelization of the maps |
| `converged` | Whether the PCG reached `tol` within `maxiter` iterations |
| `iterations` | Number of PCG iterations |
| `residuals` | $\lvert r\rvert^2/\lvert b\rvert^2$ at each iteration |

Pixels without hits are `healpy.UNSEEN` in all components. In pixels where Q
and U cannot be separated (too few hits, or a poor coverage of polarization
angles, see [`min_pol_rcond`](parameters.md#polarization-angle-coverage)), only I is
solved and Q, U are `healpy.UNSEEN`.

The I monopole is poorly constrained with 1/f noise (it is almost degenerate with
the offset of each chunk): compare maps after removing it.
