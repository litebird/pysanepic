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
| `dipole` | The dipole signal in the TOD [K], like `tod`, or `None`; needed for [calibration](#calibration) |

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
| `gains` | With [calibration](#calibration), one array of gains per `DetectorData` of this process; `None` otherwise |
| `bandpass` | With [bandpass templates](#bandpass-templates), `{template: {detector: amplitude}}`, on every process; `None` otherwise |

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

### Common modes

For a {class}`~pysanepic.DetectorGroup` the cross-spectra of its detectors are
averaged over all the groups with the same detectors (e.g. the days of a
wafer, also across MPI processes) and fitted with `n_modes` common modes
(default 1) plus the own noise of each detector,
$R(f) = A\,P(f)\,A^T + \mathrm{diag}\,N(f)$, by expectation-maximization, the
spectral matching of `sanePS` in the IAS version of SANEPIC. The modes given
in the group are not used. The result has the own noise of each detector
under its name, and the list of `CommonMode` (couplings with rms 1) under the
tuple of the names:

```python
first = make_maps(groups, nside)  # e.g. ignoring the correlation
est = estimate_noise(groups, first, chunk_s=86400.0)
names = tuple(d.name for d in groups[0].detectors)
estimated = [DetectorGroup(detectors=[dataclasses.replace(d, noise=est[d.name]) for d in g.detectors],
                           modes=est[names]) for g in groups]
maps = make_maps(estimated, nside, params=GLSParameters(coarse_nside=1))
```

On G100 (16 days, 8 detectors of a wafer, nside 512, half of the noise
variance in a common mode) the couplings are found within 0.1% and the
spectra within 5% above 1 mHz; the I map with the estimated mode differs from
that with the true one by 1% of its noise. The detectors of a group need a
name; with $n$ detectors at most the $r$ modes with $(n-r)^2 \ge n + r$ can be
estimated (1 mode from 3 detectors).

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

## Calibration

With `calibration=`{class}`~pysanepic.CalibrationParameters`, `make_maps`
fits one gain per detector and period together with the maps, using the
dipole as calibrator. Each `DetectorData` needs `dipole`, the dipole signal
of each sample (in K, the solar plus the orbital dipole), which is
subtracted from the calibrated data.

```python
from pysanepic import CalibrationParameters, make_maps

cal = CalibrationParameters(gain_period_s=1200.0, mask=galactic_mask)  # 0 on the Galactic plane
result = make_maps(data, nside, params=params, calibration=cal)
result.gains  # one array per DetectorData: gain of each 20-minute period
```

| Field | Meaning |
|---|---|
| `gain_period_s` | One gain per detector and period of this duration [s], from the first sample of each `DetectorData`; `inf`: one per `DetectorData` |
| `n_iter` | Calibration iterations, each a GLS solve (default 2: enough when the noise dominates; more for noiseless tests). The setup of the GLS is built once and reused, and each solve starts from the previous maps |
| `cache_model` | Keep the TOD of the sky model and of the bandpass templates in memory between the iterations, instead of computing them again (default `False`; 8 bytes per sample each) |
| `mask` | HEALPix map (output nside and coordinates), 1 for pixels used and 0 for pixels left out of the fits (e.g. the Galactic plane), as the usual masks; or `{"gain": ..., "bandpass": ...}`, one per kind of fit (see [below](#bandpass-templates)). Samples left out of every fit are flagged in the iterations, then one more solve makes the maps with all the samples |

The absolute calibration comes from the orbital dipole: a gain common to all
periods would otherwise just rescale the map. Signal that the map cannot
model, such as the Galaxy below the pixel size, biases the gains through the
1/f weights: model it (`sky_model` as alm, `bandpass_templates`, below) or
mask it. See [calibration](algorithm.md#calibration) and the
[notebook](notebooks/calibration), where the gain period and the mask
are tested.

## Sky model

`make_maps(sky_model=...)` takes a model of the whole sky seen by all the
detectors (e.g. CMB plus foregrounds at the reference frequency of the
channel), I or I/Q/U in K in the coordinates of the maps, as one array:

- **alm** (complex, healpy order, mmax = lmax): interpolated at the exact
  pointing of each sample with `ducc0.sht.synthesis_general` (the `Pointing`
  needs angles);
- **HEALPix maps** at the output nside: sampled in the pixel.

Its TOD is subtracted from the data and its binned map is put back in the
result, so the maps are still those of the whole sky. With alm this removes the
structure of the model below the pixel size, which otherwise leaks through the
1/f weights into the maps (pixelization noise) and, with calibration, into the
gains. With maps the result only changes in pixels whose Q/U are not solved
(their polarization is removed) and in the calibration template. Without
calibration the dipole is not used.

## Bandpass templates

`make_maps(bandpass_templates={name: template, ...})` fits, for each template
(maps or alm, as `sky_model`, but not part of it), one amplitude per detector
(`DetectorData.name`, shared by all its `DetectorData`, also on different MPI
processes), with the amplitudes of a template summing to zero over the
detectors (their common part is the map). For the foregrounds seen through
different bands:

```python
result = make_maps(data, nside, params=params, calibration=cal,
                   sky_model=cmb_alm + fg_alm,                       # at the reference frequency nu0
                   bandpass_templates={"fg_deriv1": dfg_dnu_alm})   # dF/dnu at nu0; or {"fg": fg_alm}
result.bandpass   # {"fg_deriv1": {"d0": shift of the band of d0 [GHz], ...}}
```

With the derivative of the foreground with respect to frequency
($F_i \simeq F + \Delta\nu_i\,\partial F/\partial\nu$) the amplitudes are the
shifts of the bands, and the spatial variation of the spectral index is taken
into account; with `{"fg": ...}` each detector scales the foreground. The
expansion is around the foreground in `sky_model`: the derivative must be at
the same reference frequency, otherwise the difference is a common part that
the amplitudes (summing to zero) cannot absorb. The fit is in the same PCG as
the maps: in one solve, or in each calibration iteration.

More templates can be passed if needed: other components, or higher orders
of the expansion ($\tfrac{1}{2}\Delta\nu_i^2\,\partial^2F/\partial\nu^2$,
`{"fg_deriv1": ..., "fg_deriv2": ...}`, whose amplitudes are then fitted
freely, not as the square of the shift). Templates nearly proportional to
each other are not separable: with a spectral index that varies little,
$\partial^2F/\partial\nu^2 \simeq (\beta - 1)/\nu\;\partial F/\partial\nu$, and the
two amplitudes are meaningless (the maps are not affected). The first
derivative is usually enough: the second order is $\sim\Delta\nu/\nu$ smaller.

### The common part of the band shifts

By default the amplitudes of each template sum to zero: only the differences
between the detectors are fitted. The common part, i.e. how far the mean band
of the detectors is from the frequency of `sky_model`, is left to the maps,
which absorb it pixel by pixel. With calibration this biases the absolute
calibration. If the mean band is at $\nu_0 + \overline{\Delta\nu}$, the dust in
the data is the model scaled by $\kappa = \beta\,\overline{\Delta\nu}/\nu_0$. Below
the pixel size, where the model (as alm) has structure, the only unknown that
can rescale the signal is the gain common to all the detectors, so the
absolute calibration moves by about $\kappa$. On G100 (16 detectors at
118 GHz, $\overline{\Delta\nu} = 0.031$ GHz, $\kappa = 4.1\times10^{-4}$), noiseless:

| band shifts | absolute calibration | I error |
|---|---|---|
| mean 0.031 GHz, sum-zero amplitudes | $+4.0\times10^{-4}$ | 0.47 µK |
| the same, mean removed | $+2.9\times10^{-5}$ | 0.03 µK |
| no mismatch | $+8.5\times10^{-7}$ | 0.00 µK |

Q and U are not affected. `make_maps(bandpass_common=True)` fits the common
part too: the amplitudes are free, and with derivative templates they are
the shifts with respect to the frequency of `sky_model`. It needs the
templates as alm: the common part is constrained only by their structure
below the pixel size (as maps it is exactly the map, and `make_maps` raises
an error). Without calibration it is exact. With calibration it removes most
of the bias, but not all: a common rescaling of the foreground (the common
gain) and a common shift of the band are nearly degenerate when the spectral
index varies little, and are separated only by the dipole and the CMB, which
the gain rescales too. In a local noiseless test (4 detectors, 140 GHz, mean
shift 0.3 GHz, $\kappa = 3.3\times10^{-3}$):

| | absolute calibration | mean shift found |
|---|---|---|
| sum-zero amplitudes | $+3.1\times10^{-3}$ | 0 (by construction) |
| `bandpass_common=True` | $-2.8\times10^{-4}$ | 0.33 GHz (true 0.30) |

On G100 (16 days, 16 detectors at 118 GHz, mean shift 0.031 GHz,
$\kappa = 4.1\times10^{-4}$), the absolute calibration with respect to the run
without mismatch, the same sky and noise:

| | noiseless | with 1/f noise | mean shift found (noise) |
|---|---|---|---|
| sum-zero amplitudes | $+4.0\times10^{-4}$ | $+3.5\times10^{-4}$ | – |
| `bandpass_common=True` | $-1.4\times10^{-6}$ | $+2.2\times10^{-4}$ | 0.013 GHz (true 0.031) |

Without noise the bias goes entirely; with the noise of 16 days of 16
detectors the common part is measured only to ~0.02 GHz, comparable with the
shift itself, and the bias drops by ~40% in this realization. The common
part is constrained only below the pixel size, so it needs much data (or a
bright foreground with structure): it pays off for long runs; for short
ones, give the sky model at the mean band. The PCG needs more iterations
(52 instead of 13 in the noiseless case), as the common part is weakly
constrained.

Use it when the frequency of the sky model is not known to better than
$\sim\nu_0/(\beta\cdot\text{required calibration})$; otherwise give the sky model at
the mean band of the detectors.

A mask per kind of fit, e.g. `mask={"gain": galaxy * sources, "bandpass":
sources}`: the bandpass amplitudes are fitted on the Galaxy, where the
foreground is bright, point sources missing from the model are left out of
everything. Samples left out of some fits only keep the other templates; they
are calibrated with the values of the previous iteration, so the iterations
converge slower. See the [notebook](notebooks/sky_model).
