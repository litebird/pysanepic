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
from pysanepic import DetectorData, GLSParameters, OneOverFNoise, Pointing, make_maps


noise = OneOverFNoise(net_ukrts=40.0, fknee_hz=0.02, alpha=1.0, fmin_hz=1e-5)


def data():
    for name in ["det0", "det1"]:
        f = np.load(f"{name}.npz")  # your own files: one per detector and time span
        yield DetectorData(
            tod=f["tod"],  # K
            # angles [rad] in Ecliptic coordinates, rotated and pixelized by pysanepic
            pointing=Pointing(theta=f["theta"], phi=f["phi"], psi=f["psi"], coordinates="E"),
            hwp_angle=None,  # rad, per sample; None without HWP
            pol_angle_rad=float(f["pol_angle"]),
            sampling_rate_hz=19.0,
            noise=noise,  # one object can serve all the time spans of a detector
        )


result = make_maps(data(), nside=64, coordinates="G", params=GLSParameters(chunk_s=3600.0))
print(result.converged, result.iterations)
i_map, q_map, u_map = result.maps  # healpy.UNSEEN where not solved
```

With MPI, every process calls `make_maps` with its own data and the same
`comm`; see [MPI and threads](performance.md#mpi-and-threads).

## Pointing

A {class}`~pysanepic.Pointing` gives where a detector looks, in one of two
forms:

- **angles**: `Pointing(theta=..., phi=..., psi=..., coordinates=...)`. pysanepic
  rotates them to the coordinates of the output maps (correcting ψ) and
  computes the pixels at the output `nside`;
- **pixels already computed**: `Pointing(pix=..., nside=..., psi=..., coordinates=..., ordering="RING")`.
  They are used as they are, so `nside` and `coordinates` must be those of the
  output maps (otherwise `make_maps` raises an error), and ψ must be in the same
  coordinates. Pass `pix` as int32 to avoid a copy. litebird_sim uses this form.

| Field | Meaning |
|---|---|
| `theta`, `phi` | Colatitude and longitude [rad] |
| `pix`, `nside`, `ordering` | HEALPix pixel of each sample, its resolution and ordering (`"RING"` or `"NESTED"`) |
| `psi` | Orientation of the detector frame [rad] |
| `coordinates` | Coordinate system of the angles or pixels and of `psi`: `"E"`, `"G"` or `"C"` |

The arrays are checked when the object is created: one form only, same
lengths, integer pixels within $[0, 12\,n_\text{side}^2)$, valid `nside` and
`ordering`.

## `DetectorData` fields

| Field | Meaning |
|---|---|
| `tod` | Time-ordered data, in K |
| `pointing` | A `Pointing`, with as many samples as `tod` |
| `hwp_angle` | HWP angle per sample [rad], or `None` |
| `pol_angle_rad` | Polarization angle of the detector [rad] |
| `pol_efficiency` | Polarization efficiency γ (default 1) |
| `sampling_rate_hz` | Sampling rate [Hz] |
| `noise` | The noise model: a `OneOverFNoise` or a `TabulatedNoise`, see below |
| `flags` | Boolean array like `tod`, `True` for samples to exclude, or `None`; see [flagged samples](parameters.md#flagged-samples) |
| `name` | The detector, or `None`: {func}`~pysanepic.estimate_noise` averages the spectrum over the `DetectorData` with the same name |

A detector observed in several stretches of time (e.g. one per day, as the
observations of litebird_sim) is passed as several `DetectorData`. The
stretches are treated as independent: see [chunks](parameters.md#chunk-length).

## Conventions

The conventions are the same as litebird_sim:

- signal model: $d = I + \gamma\,(Q \cos 2a + U \sin 2a)$, with polarization
  angle $a = \psi + \text{pol\_angle}$ without HWP and
  $a = \psi + 2\,\text{hwp\_angle} - \text{pol\_angle}$ with an ideal HWP;
- noise power spectrum (`DetectorData.noise`), with
  $\sigma = \text{NET}\sqrt{f_s}$ the white-noise rms per sample, as
  litebird_sim's `add_noise`:
  - `OneOverFNoise(net_ukrts=..., fknee_hz=..., alpha=..., fmin_hz=..., model="toast")`:
    $P(f) = \sigma^2 \dfrac{f^\alpha + f_\text{knee}^\alpha}{f^\alpha + f_\text{min}^\alpha}$;
  - `OneOverFNoise(..., model="keshner")`:
    $P(f) = \sigma^2 \left(\dfrac{f^2 + f_\text{knee}^2}{f^2 + f_\text{min}^2}\right)^{\alpha/2}$,
    the spectrum of litebird_sim's `engine="ducc"` (ducc0's `OofaNoise`);
  - `TabulatedNoise(freqs_hz=..., psd=...)`: any one-sided PSD in K²/Hz (white
    noise is $2\sigma^2/f_s$), interpolated linearly in log f – log P and held
    constant outside the table, e.g. a measured or estimated spectrum;
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

## Estimating the noise

When the noise spectrum is not known, {func}`~pysanepic.estimate_noise`
estimates it from the residual of a first map, $d - P\hat m$:

```python
import dataclasses
from pysanepic import OneOverFNoise, estimate_noise, make_maps

first = make_maps(data, nside)  # with a guess, e.g. white noise
noise = estimate_noise(data, first, chunk_s=86400.0)  # {name: TabulatedNoise}
maps = make_maps([dataclasses.replace(d, noise=noise[d.name]) for d in data], nside)
```

The residual of each `DetectorData` is cut in chunks of `chunk_s` seconds; the
periodogram of each chunk (Hann window; flagged samples set to 0, with their
fraction corrected) is averaged over all the `DetectorData` with the same
`name`, also across MPI processes (`comm=`), in `nbins` logarithmic bins from
`1/chunk_s` to $f_s/2$. The result is a `TabulatedNoise` per detector, or, with
`fit="toast"` or `fit="keshner"`, a `OneOverFNoise` fitted to it ($f_\text{min}$
is not constrained below `1/chunk_s`).

The residual differs from the noise in two ways: the map absorbs part of the
noise, mostly at the harmonics of the scan (on a very repetitive test scan
with 1.4% as many unknowns as samples, up to 30% of the power between the spin
frequency and the pixel-crossing frequency), and signal finer than the pixels
stays in the residual. Both are small when the samples far outnumber the
pixels and the sky is smooth on the scale of the pixels; the second map is
unbiased in any case, at most slightly less optimal.

## Correlated noise

Detectors observed at the same times may share noise, e.g. a common mode of
a wafer. A {class}`~pysanepic.DetectorGroup` gathers their `DetectorData` with
one or more {class}`~pysanepic.CommonMode`: detector $i$ sees
$U_{ir}\,c_r(t)$ besides its own noise, with $c_r$ a mode of spectrum
`noise`. In Fourier space $N(f) = D(f) + U\,\Lambda(f)\,U^T$, with $D$ the
own noise of the detectors (their `noise`) and $\Lambda$ the spectra of the
modes.

```python
from pysanepic import CommonMode, DetectorGroup, OneOverFNoise, make_maps

mode = OneOverFNoise(net_ukrts=20.0, fknee_hz=1.0, alpha=2.0, fmin_hz=1e-3)
group = DetectorGroup(detectors=wafer, modes=[CommonMode(coupling=[1.0, 0.9, ...], noise=mode)])
result = make_maps([group, *other_detectors], nside)
```

The detectors of a group must have the same number of samples and sampling
rate, and a group must be on one MPI process; its chunks are not split at
gaps (`split_gap_s`), flagged samples are handled by `flag_method`.
$N^{-1}$ is applied with the Woodbury identity, one detector at a time
(see [correlated noise](algorithm.md#correlated-noise)). Common modes slow
down the PCG; `GLSParameters(coarse_nside=1)` brings the iterations back to
those of uncorrelated noise (see [preconditioner](parameters.md#preconditioner)).

