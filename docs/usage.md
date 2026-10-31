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
  coordinates. Pass `pix` as int32 to avoid a copy. A `sky_model` or
  sky templates given as alm need the angles.

The coordinates are not inferred: `coordinates` must be those of the angles or
pixels and of ψ. litebird_sim (`make_sanepic_gls_map`) computes the pointings in
the coordinates of the maps and passes pixels, or angles when a model is given
as alm (its `pointing_form`), so pysanepic does not rotate them.

| Field | Meaning |
|---|---|
| `theta`, `phi` | Colatitude and longitude [rad] |
| `pix`, `nside`, `ordering` | HEALPix pixel of each sample, its resolution and ordering (`"RING"` or `"NESTED"`) |
| `psi` | Orientation of the detector frame [rad] |
| `coordinates` | Coordinate system of the angles or pixels and of `psi`: a {class}`~pysanepic.Coordinates` (`Ecliptic`, `Galactic`, `Equatorial`) or its value `"E"`, `"G"` or `"C"`; stored as a `Coordinates` |

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

## Flagged samples

Pass boolean `DetectorData.flags`: True means excluded; integer bit fields
must first be converted with the desired bit mask. NaN values in TOD do not
automatically flag a sample. A flagged value may be NaN because its original
value is never used, and flagged samples do not count in the hit map.

Given a DetectorData named `detector`, mark a short interval while retaining
its existing flags:

```python
from dataclasses import replace
import numpy as np
from pysanepic import GLSParameters, make_maps

flags = np.zeros(detector.tod.size, dtype=bool) if detector.flags is None else detector.flags.copy()
flags[100:110] = True
flagged_detector = replace(detector, flags=flags)

params = GLSParameters(flag_method="act", act_ncoeff=3)  # quadratic cut model
result = make_maps([flagged_detector], nside, params=params)
```

Choose `flag_method` independently of the length threshold `split_gap_s`:

| Method | Gap treatment | Specific settings |
|---|---|---|
| `"marginalize"` (default) | One free value per flagged sample, exact for the retained chunk model | None |
| `"offset"` | One constant per contiguous run | None |
| `"act"` | Legendre coefficients per run | `act_ncoeff`: None for the ACT rule, a positive integer, or a callable |
| `"sanepic"` | Fixed synthetic values, no gap unknowns in the map solve | `gap_fill`, `flag_seed`; conditional modes also use `gap_fill_tol` and `gap_fill_maxiter` |

Runs at least `split_gap_s=300` seconds long are dropped and split the chunk
before this treatment; use `np.inf` to retain every run, or 0 to split every
run. This splitting is disabled for DetectorGroup. Conditional filling is
available only with `"sanepic"`:

```python
params = GLSParameters(
    flag_method="sanepic",
    gap_fill="conditional",  # "local_white" is the default; "conditional_mean" is deterministic
    flag_seed=42,
    gap_fill_tol=1e-12,
    gap_fill_maxiter=1000,
)
# sky_estimate is a prior model or preliminary map, in the output coordinates
# and at this nside (or provided as alm). For noise-only TOD use a zero model.
result = make_maps([flagged_detector], nside, params=params, sky_model=sky_estimate)
```

The conditional fill is exact for the supplied residual/noise model, then
fixed during the approximate SANEPIC map solve: it does not make that solve
equivalent to exact marginalization or a Gibbs sampler. `flag_seed` is
reproducible for the same input, chunking and MPI assignment, generally not
after redistributing data. `tol`/`maxiter` control the map solve separately
from `gap_fill_tol`/`gap_fill_maxiter`.

The ACT and SANEPIC filling modes above are unreleased additions; use the
development checkout. See [flagged-sample parameters](parameters.md#flagged-samples),
[seed behavior with MPI](parameters.md#random-seeds-and-mpi), and the
[saved benchmark notebook](notebooks/flags) (marginalize, offset and splitting).

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
| `coordinates` | `"G"` | Coordinate system of the output maps, a `Coordinates` or `"E"`, `"G"`, `"C"` |
| `params` | `GLSParameters()` | Settings of the map-maker, see [Parameters](parameters.md) |
| `comm` | `None` | mpi4py communicator; each rank passes only its own data |
| `nthreads` | `None` | Threads per process (FFTs, ducc0, numba); `None`: `OMP_NUM_THREADS` if set, else 1 |
| `verbose` | `False` | Print the residual at each iteration and the time of each step of each solve (rank 0) |

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
| `templates` | With [templates](#templates), `{name: amplitudes}`: `{detector: amplitude}` (one per detector, on every process), a float (common), or one array per `DetectorData` of this process (per period, as `gains`); `None` otherwise |
| `template_covariance` | With `template_covariance=True`, `{"labels": [(template, detector or None), ...], "matrix": ...}`, the covariance of the amplitudes per detector and common, on every process; `None` otherwise |

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
| `gain_tol` | Stop the iterations when the largest absolute gain correction $\delta g$ (all detectors and periods) is below it; `n_iter` is then the maximum, and `MapResult.calibration_iterations` the number done (default `None`: always `n_iter`). The convergence is geometric, slower for periods with a weak dipole and much slower with samples left out of some fits only (a gain mask but templates on the whole sky) |
| `cache_model` | Keep the TOD of the sky model and of the sky templates in memory between the iterations, instead of computing them again (default `False`; 8 bytes per sample each) |
| `covariance_fixed_gains` | With `template_covariance=True`: the covariance with the gains fixed at their fitted values instead of marginalized over (default `False`). The gains couple to the sky templates through the sky signal in the gain template and slow down the solves of the covariance; fixed, the errors were 0.05–0.13% smaller in a test, and the covariance of a 1-year e2e run took 3 times fewer PCG iterations |
| `mask` | HEALPix map (output nside and coordinates), 1 for pixels used and 0 for pixels left out of the gain fit (e.g. the Galactic plane), as the usual masks; also the mask of the templates without their own (`Template.mask`, see [below](#masks)). Samples left out of every fit are flagged in the iterations, then one more solve makes the maps with all the samples |

The absolute calibration comes from the orbital dipole: a gain common to all
periods would otherwise just rescale the map. Signal that the map cannot
model, such as the Galaxy below the pixel size, biases the gains through the
1/f weights: model it (`sky_model` as alm, [templates](#templates), below) or
mask it. See [calibration](algorithm.md#calibration) and the
[notebook](notebooks/calibration), where the gain period and the mask
are tested.

## Sky model

`make_maps(sky_model=...)` takes a model of the whole sky seen by all the
detectors (e.g. CMB plus foregrounds at the reference frequency of the
channel), I or I/Q/U in K in the coordinates of the maps, as one array:

- **alm** (complex, healpy order, mmax = lmax): interpolated at the exact
  pointing of each sample with `ducc0.sht.synthesis_general` (the `Pointing`
  needs angles), one call per `DetectorData` and field, or per
  `GLSParameters.sample_batch` `DetectorData` (see
  [performance](performance.md#sampling-alm-models));
- **HEALPix maps** at the output nside: sampled in the pixel.

Its TOD is subtracted from the data and its binned map is put back in the
result, so the maps are still those of the whole sky. With alm this removes the
structure of the model below the pixel size, which otherwise leaks through the
1/f weights into the maps (pixelization noise) and, with calibration, into the
gains. With maps the result only changes in pixels whose Q/U are not solved
(their polarization is removed) and in the calibration template. Without
calibration the dipole is not used.

## Templates

`make_maps(templates={name: Template(...), ...})` fits templates of the TOD
together with the maps, in the same PCG: in one solve, or in each calibration
iteration. Each {class}`~pysanepic.Template` is

- a **sky signal**, `Template(sky=...)`: maps or alm, as `sky_model` but not
  part of it, sampled for every detector (e.g. the derivative of the
  foregrounds with respect to frequency, for bandpass mismatch);
- or a **TOD given per detector**, `Template()`: the array in
  `DetectorData.templates[name]` (e.g. a thermometer, a housekeeping signal);
  a `DetectorData` without it has no amplitude for it.

and has amplitudes

| `Template(...)` | Amplitudes |
|---|---|
| default | one per detector (`DetectorData.name`, shared by all its `DetectorData`, also on different MPI processes) |
| `sum_zero=True` | one per detector, summing to zero over the detectors: only the differences are fitted |
| `common=True` | one for all the detectors |
| `period_s=...` | one per `DetectorData` and period, from its first sample, like the gains; local to each process (any number of them); `inf`: one per `DetectorData` |

and optionally a `mask` (pixels left out of its fit, see [below](#masks)).

A sky template given as maps is exactly a map, so its common part is
degenerate with the maps: it needs `sum_zero=True` (`make_maps` raises an
error otherwise). As alm, the common part is constrained by the structure
below the pixel size (see [below](#the-common-part-of-the-band-shifts)).

```python
from pysanepic import Template

data = [DetectorData(..., name="d0", templates={"thermal": bath_temperature_d0}), ...]
result = make_maps(data, nside, params=params, templates={
    "thermal": Template(),                    # one coupling per detector
    "hk": Template(common=True),              # e.g. a housekeeping signal in DetectorData.templates["hk"]
    "drift": Template(period_s=86400.0),      # one amplitude per DetectorData and day
}, template_covariance=True)
result.templates["thermal"]       # {"d0": ..., "d1": ...}
result.template_covariance        # {"labels": [("thermal", "d0"), ..., ("hk", None)], "matrix": ...}
```

With calibration the templates are inside the gain,
$d = g\,(P m + D + S + \sum_k a_k T_k) + n$: the amplitudes are in the units
of the calibrated TOD.

### Covariance of the amplitudes

`template_covariance=True` gives the covariance of the amplitudes per
detector and common, marginalized over the maps and the amplitudes per
period (and the gains): the corresponding block of the inverse of the GLS
operator, for the noise model of the data. It costs one more PCG solve per
amplitude (with calibration, after the last iteration), to
`GLSParameters.covariance_tol` (default 1e-6: the errors to ~1e-4, with ~1/3
fewer iterations than the `tol` of the maps). With calibration,
`CalibrationParameters(covariance_fixed_gains=True)` keeps the gains fixed
instead of marginalizing over them: the errors are ~0.1% smaller, and the
solves much faster (see [calibration](#calibration)). It shows which
templates are degenerate: e.g. with the derivatives of the foregrounds to
first and second order, the two amplitudes of a detector are strongly
correlated and individually poorly measured, while the maps are not
affected. Amplitudes summing to zero have the covariance of the constrained
amplitudes (it is singular along the sum). The amplitudes per period have no
covariance. A Monte Carlo over noise realizations confirms the errors to
~10% ([validation](validation.md#covariance-of-the-template-amplitudes)); the
[sky model notebook](notebooks/sky_model) shows the degeneracy of the first
and second derivatives of the foregrounds.

### Bandpass mismatch

For the foregrounds seen through different bands:

```python
result = make_maps(data, nside, params=params, calibration=cal,
                   sky_model=cmb_alm + fg_alm,                                   # at the reference frequency nu0
                   templates={"fg_deriv1": Template(sky=dfg_dnu_alm, sum_zero=True)})  # dF/dnu at nu0
result.templates["fg_deriv1"]   # {"d0": shift of the band of d0 [GHz], ...}
```

With the derivative of the foreground with respect to frequency
($F_i \simeq F + \Delta\nu_i\,\partial F/\partial\nu$) the amplitudes are the
shifts of the bands, and the spatial variation of the spectral index is taken
into account; with the foreground itself, `Template(sky=fg_alm, sum_zero=True)`,
each detector scales it. The
expansion is around the foreground in `sky_model`: the derivative must be at
the same reference frequency, otherwise the difference is a common part that
the amplitudes (summing to zero) cannot absorb.

More templates can be passed if needed: other components, or higher orders
of the expansion ($\tfrac{1}{2}\Delta\nu_i^2\,\partial^2F/\partial\nu^2$,
`"fg_deriv1"` and `"fg_deriv2"`, whose amplitudes are then fitted freely,
not as the square of the shift). Templates nearly proportional to each other
are not separable: with a spectral index that varies little,
$\partial^2F/\partial\nu^2 \simeq (\beta - 1)/\nu\;\partial F/\partial\nu$, and the
two amplitudes are meaningless (the maps are not affected; see their
[covariance](#covariance-of-the-amplitudes)). The second order is
$\sim\Delta\nu/\nu$ smaller than the first, but it matters where the foreground
is bright and the noise low: in a G100 test (M1_L, one year, 4 detectors,
noise / 100) the first derivative alone left residuals of ~0.5 µK in Q/U on
the Galactic plane, which the second order removes down to the noise.

### The common part of the band shifts

With `sum_zero=True` the amplitudes of each template sum to zero: only the differences
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

Q and U are not affected. `Template(sky=..., sum_zero=False)` fits the common
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
| free amplitudes | $-2.8\times10^{-4}$ | 0.33 GHz (true 0.30) |

On G100 (16 days, 16 detectors at 118 GHz, mean shift 0.031 GHz,
$\kappa = 4.1\times10^{-4}$), the absolute calibration with respect to the run
without mismatch, the same sky and noise:

| | noiseless | with 1/f noise | mean shift found (noise) |
|---|---|---|---|
| sum-zero amplitudes | $+4.0\times10^{-4}$ | $+3.5\times10^{-4}$ | – |
| free amplitudes | $-1.4\times10^{-6}$ | $+2.2\times10^{-4}$ | 0.013 GHz (true 0.031) |

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

### Polarization angles

An error $\delta_i$ on the polarization angle of detector $i$ changes its TOD
by $\delta_i\,\partial d/\partial\delta = \delta_i\,2\gamma\,(-Q \sin 2a + U \cos 2a)$,
which is the signal of the sky $(0,\, 2U,\, -2Q)$: a sky template, made from
the polarization of the sky model, whose amplitudes are the angle errors in
radians. {func}`~pysanepic.pol_angle_template` makes it, as alm
$(0, 2B, -2E)$ or maps $(0, 2U, -2Q)$, with `sum_zero=True`:

```python
from pysanepic import pol_angle_template

# sky_alm: T, E, B of the sky model (or maps I, Q, U)
result = make_maps(data, nside, params=params, sky_model=sky_alm,
                   templates={"pol_angle": pol_angle_template(sky_alm)})
result.templates["pol_angle"]   # {"d0": delta_0 - mean, ...} [rad]
```

- **Common rotation**: an angle error common to all the detectors rotates
  the polarization of the map (E into B), and the maps absorb it: the
  amplitudes sum to zero, and the maps are the sky rotated by the mean
  error. The absolute angle needs an external
  calibrator (e.g. Tau A, or EB with an assumed cosmology). As alm the
  common part is constrained only by the structure below the pixel size,
  which is not a calibration: keep `sum_zero=True`.
- **HWP**: with an ideal HWP the angle seen by the sky is
  $\psi + 2\chi - \theta$, so the amplitudes are $-\delta_i$.
- **Linearization**: the template is the first order in $\delta$; the
  error on the fitted angles is $O(\delta^2)$: in a noiseless test,
  $4\times10^{-9}$ rad for $\delta \sim 10^{-4}$ rad and 0.4% for
  $\delta \sim 1°$, with the maps right to $\sim 2\,\delta^2$ of the signal:
  negligible for errors of a few arcminutes.
- **With the other templates**: angles, gains and bandpass amplitudes can
  be fitted together; `template_covariance=True` shows how much they are
  correlated.

### Beams

A detector sees the sky smoothed by its beam. If the beam differs from the
one of the sky model by a small offset of its centre or a small change of
its shape, the TOD changes, to first order, by the derivatives of the
model at the pointing, weighted by the moments of the difference. In a frame
$(x, y)$ on the sky attached to the detector:

$$
d \simeq s + \rho_x\,\partial_x s + \rho_y\,\partial_y s
+ \tfrac12\,\delta\sigma^2\,\nabla^2 s
+ c_+\,\tfrac12\,(\partial_{xx} - \partial_{yy})\,s + c_\times\,\partial_{xy}\,s
$$

with $s$ the sky model through its (circular) beam, $\rho$ the offset of the
beam centre [rad], and $\delta\sigma^2$, $c_+$, $c_\times$ [rad²] the change of
its second moments $\Sigma$:
$\Sigma_{xx} = \sigma^2 + \delta\sigma^2 + c_+$,
$\Sigma_{yy} = \sigma^2 + \delta\sigma^2 - c_+$, $\Sigma_{xy} = c_\times$. For a
Gaussian beam with $\sigma_\text{maj}$, $\sigma_\text{min}$
($\text{FWHM} = \sigma\sqrt{8 \ln 2}$) and major axis at an angle $\beta$ from
$x$: $\delta\sigma^2 = (\sigma_\text{maj}^2 + \sigma_\text{min}^2)/2 - \sigma^2$,
$c_+ = (\sigma_\text{maj}^2 - \sigma_\text{min}^2)\cos 2\beta / 2$,
$c_\times = (\sigma_\text{maj}^2 - \sigma_\text{min}^2)\sin 2\beta / 2$.

Each term is a template whose amplitudes are these numbers, per detector:

- **FWHM**, $\tfrac12\nabla^2 s$, does not depend on the orientation of the
  detector: a sky template, the alm of the model times $-\ell(\ell+1)/2$ for T
  and $-(\ell(\ell+1) - 4)/2$ for E and B (the Laplacian of a spin-2 field;
  the $-4$ matters only at low $\ell$);
- **offsets** and **ellipticity** turn with the detector (as $\psi$ and
  $2\psi$): they are not sky templates, but TODs computed for each detector,
  the derivatives of the model along $x$ and $y$ at its pointing.

The derivatives are exact, computed in harmonic space with the spin-raising
and lowering operators $\eth$, $\bar\eth$ (the deprojection templates of
the temperature, and of the polarization too). Along the direction at angle
$b$, for a field of any spin (T, or $P = Q + iU$ with covariant derivatives,
which keep the polarization right near the poles):

$$
\partial_x = -\tfrac12\left(e^{-ib}\,\eth + e^{ib}\,\bar\eth\right),\qquad
\tfrac12(\partial_{xx} - \partial_{yy}) = \tfrac14\left(e^{-2ib}\,\eth^2 + e^{2ib}\,\bar\eth^2\right),
$$

and $\partial_y$, $\partial_{xy}$ the same with $b \to b + \pi/2$, $b + \pi/4$.
$\eth T$, $\eth^2 T$, $\eth P$, $\bar\eth P$, $\eth^2 P$, $\bar\eth^2 P$ are
fields of spin 1, 2, 3, 1, 4, 0: the alm of the model times
$\sqrt{(\ell \mp s)(\ell \pm s + 1)}$ factors, synthesized at the pointing
with ducc0 `synthesis_general`. The polarization enters the TOD as
$\gamma\,\mathrm{Re}[e^{-2ia}\,\partial P]$, with $a$ the angle of the
signal model, so the HWP is taken into account. A test compares them with
finite differences of the model resampled with the beam moved along $x$ and
$y$ (`test_beam_tods_derivatives`).

The frame: $x = \cos b\, e_\theta + \sin b\, e_\phi$,
$y = -\sin b\, e_\theta + \cos b\, e_\phi$ with $b = \psi + \text{pol\_angle}$,
in the coordinates of the maps; $b$ turns in the same sense as the polarization
angle $a$ of the [signal model](#conventions), so without HWP $x$ is at the
polarization angle of the detector. A different fixed choice (e.g. $b = \psi$)
only rotates the amplitudes of each detector.

The helpers:

- {func}`~pysanepic.fwhm_template`: the FWHM template, with `sum_zero=True`
  (`sum_zero=False` with pointing offsets, see below);
- {func}`~pysanepic.beam_templates`: the TODs of offsets and ellipticity
  for all the `DetectorData` (added to their `templates`), and the
  `Template` objects for them, with the constraints below;
  {func}`~pysanepic.beam_tods` computes the TODs of one `DetectorData`, e.g.
  in a generator that builds the data one detector at a time;
- {func}`~pysanepic.beam_parameters`: from the fitted amplitudes to the
  offsets and the shapes of the beams.

```python
from pysanepic import beam_parameters, beam_templates, fwhm_template

# sky_alm: T, E, B of the sky model through the mean beam (sigma), in Galactic coordinates
data, templates = beam_templates(data, sky_alm, "G")   # offset_x, offset_y, ell_plus, ell_cross
templates["fwhm"] = fwhm_template(sky_alm)
result = make_maps(data, nside, coordinates="G", params=params, sky_model=sky_alm, templates=templates)
result.templates["offset_x"]["d0"]      # rho_x [rad]
beams = beam_parameters(result, templates, sigma=sigma)
beams["d0"]   # {"offset_x", "offset_y", "c_plus", "c_cross", "delta_sigma2",
              #  "sigma_major", "sigma_minor", "angle"} [rad, rad^2]
```

- **What is fitted**: the second moments are about the nominal pointing, so
  an offset adds $\rho\rho^T$ to them: the amplitudes are
  $c_+ + (\rho_x^2 - \rho_y^2)/2$, $c_\times + \rho_x \rho_y$ and
  $\delta\sigma^2 + |\rho|^2/2$. `beam_parameters` subtracts the fitted
  offsets to get the shape of the beam; values fitted with `sum_zero` are
  relative to the mean of the detectors, and so are $\sigma_\text{maj}$,
  $\sigma_\text{min}$ computed from them around the `sigma` of the model.
- **Common parts**: a FWHM change common to all the detectors is a change
  of the sky, and the maps absorb it (`sum_zero=True`). Without HWP so is a
  common ellipticity (in the frame of each detector, which turns with its
  polarization angle): it leaks the gradients of I into Q/U exactly as a
  polarization map would. With an HWP the polarization turns with
  $2\chi$ and the beam does not: the common ellipticity is measured, no
  `sum_zero` (`beam_templates` chooses from the `hwp_angle` of the data).
  The common parts are absorbed by the maps only down to the pixel: the sky model should be smoothed with the mean beam of the
  detectors, or the rest of the structure below the pixel biases the fit (by
  ~1% in the test).
- **Pointing offsets and calibration**: to second order an offset $\rho$
  adds $\tfrac12 \rho^T \nabla\nabla^T s$ to the TOD, whose common part
  (the mean over the detectors, and over the orientations given by the scan)
  is a smoothing of the sky, $\delta\sigma^2 = \langle|\rho|^2\rangle/2$:
  with `sum_zero=True` it is not in the model, and with calibration the
  common gain absorbs it through the structure of `sky_model` below the
  pixel. The bias grows as $\rho^2$. Give `fwhm_template(sky_alm)` with
  `sum_zero=False` (`dataclasses.replace(t, sum_zero=False)`) when the
  offsets are fitted: the FWHM amplitudes are then
  $\delta\sigma^2_i + |\rho_i|^2/2$ with respect to `sky_model`, and
  `beam_parameters` subtracts $|\rho_i|^2/2$. On G100 (litebird_sim, 4
  detectors at 94 GHz, 1 year, beams, bands, polarization angles and gain
  drifts all perturbed, 1/f noise, `sky_model` with the mean beam), the
  common gain divided by the true one:

  | pointing offsets (per axis) | FWHM `sum_zero=True` | FWHM `sum_zero=False` |
  |---|---|---|
  | none (offsets still fitted) | $1.000037$ | – |
  | 10″ | $0.999875$ | $1.000013$ |
  | 1′ | $0.99422$ | $0.999994$ |

  (gain errors $1.6\times10^{-5}$ rms). With the free FWHM amplitudes their
  mean is $1.20\times10^{-7}$ rad² for $\langle|\rho|^2\rangle/2 =
  1.22\times10^{-7}$ rad² (1′), and the dipole left in the I map by the
  wrong calibration drops from 19 µK to 0.02 µK (10″: from 0.41 to 0.04 µK);
  Q/U and the fitted offsets do not change.
- **Linearization**: the error on the fitted values is of second order. In
  a noiseless test (`tests/test_mapmaker.py::test_beam_templates`: FWHM
  2.4°, $\ell \le 47$, offsets of a few arcminutes and ellipticities of a
  few %, with and without HWP), ~1% of the largest value, ~0.1% for 10 times
  smaller errors.
- **Cost**: per detector, 2 syntheses at the pointing for T only, 9 with
  polarization (each like the sampling of `sky_model`); the 4 templates are
  kept in memory as 4 more TODs per detector (4 bytes per sample each in the
  GLS, see `GLSParameters.template_dtype`). Fit only what is needed
  (`beam_templates(..., ellipticity=False)` for the offsets: 1 or 3
  syntheses).

### Beams as alm: a fixed template

Without HWP the ellipticity common to the detectors is degenerate with Q/U
(see above): no fit can remove its leakage of I into Q/U, which needs the
beams known from elsewhere. With the beams as alm (as in litebird_sim:
`gauss_beam_to_alm` for elliptical Gaussians, or GRASP), the TOD of their
asymmetries enters the model as a **fixed template**, `Template(fixed=True)`:
amplitude 1, times the gain with calibration, not fitted and not put back in
the maps:

$$
d = g\,(P m + D + S + F + \textstyle\sum_k a_k B_k) + n,\qquad
F = \text{sky} \circledast (B - B_\text{ref}).
$$

$B_\text{ref}$ is a beam symmetric about its axis: T at $m = 0$ and, for
Q/U, the co-rotating part at $m = 2$, $(b^E_{\ell 2} - i b^B_{\ell 2})/2$,
the only one a circular beam has. Which one is set by `model_bl`:

- **`model_bl` given** (recommended): the circular beam the sky model `S` was
  smoothed with, as its window $b_\ell$ (healpy's `gauss_beam(fwhm,
  pol=True)`, or `(lmax+1,)` for T only). Then $S + F$ is the sky through
  the true beam of each detector: exact, whatever the beams.
- **without** (default, as before): each beam minus its own symmetric part,
  $B_\text{sym}$. The symmetric parts that differ between the detectors (the
  maps have one beam) are left to {func}`~pysanepic.fwhm_template`, which
  follows them to order $\ell^2$ only, $\delta\sigma^2\,\ell(\ell+1)/2$. An
  elliptical beam is not a Gaussian in its symmetric part: with
  $\sigma^2_\text{major} = \sigma^2 e$, $\sigma^2_\text{minor} = \sigma^2/e$,
  their mean $\bar\sigma^2$ and difference $\Delta$, its $m = 0$ part is
  $e^{-\ell^2 \bar\sigma^2/2} I_0(\ell^2 \Delta/4)$, with terms of order
  $\ell^4$ that grow as $(e - 1)^2$. What is left of them biases the
  absolute calibration (see [calibration and beam ellipticity](validation.md#calibration-and-beam-ellipticity)): use this only
  with beams of the same symmetric part, or nearly circular ones.

The convolution is `ducc0.totalconvolve.Interpolator`, as in litebird_sim
(with $m_\text{max}$ from the asymmetries, see the cost below), so the
conventions are those of the simulation: `blm` in the frame of the detector,
with its polarization angle in it (`psi_pol_rad` = `DetectorData.pol_angle_rad`,
which also orients the polarization of $B_\text{ref}$), $\psi$ of the pointing
as in litebird_sim, and the polarization normalized as `gauss_beam_to_alm`
($-\sqrt{2}$ and Challinor's $e^{2\sigma^2}$): a circular `gauss_beam_to_alm`
beam minus `model_bl = hp.gauss_beam(fwhm, pol=True)` of the same FWHM is 0
(`test_blm_tods_model_bl`).

```python
import healpy as hp
from pysanepic import blm_templates

# sky_alm: T, E, B of the sky, NOT convolved with a beam; blms: {detector name: blm}
# sky_model: sky_alm through the circular beam of window bl (the mean beam of the detectors)
bl = hp.gauss_beam(fwhm_model, lmax=lmax, pol=True)
data, templates = blm_templates(data, sky_alm, blms, "G", model_bl=bl)   # {"beam": Template(fixed=True)}
result = make_maps(data, nside, coordinates="G", params=params, templates=templates, sky_model=sky_model)
```

- **With {func}`~pysanepic.fwhm_template`**: with `model_bl` the FWHM
  differences are in the fixed template, so the fitted `fwhm` amplitudes are
  what the beams are not: the second moment $\langle|\rho|^2\rangle/2$ of
  pointing offsets (see {func}`~pysanepic.beam_parameters`), or errors of
  the beams given. The template can stay (`sum_zero=False` too) as a check.

- **Sky model**: the template needs the sky itself, not through a beam,
  with the lmax of the beams; an external model is enough, since the error
  is the product of the beam asymmetry and the error of the model, and I
  (bright, well known) dominates the leakage into Q/U.
- **Check**: for a slightly elliptical Gaussian it is the first order of
  `ell_plus`, `ell_cross` of {func}`~pysanepic.beam_tods`, with the major
  axis at `psi_ell - pol_angle` from $x$ (`test_blm_tods`; the LevelS formula
  is a flat-sky expansion, off by ~0.1% at $\ell \sim 80$, more at low $\ell$).
  With it, the maps of elliptical beams are those of their symmetric parts,
  and so are the gains (`test_blm_fixed_template`). With `model_bl` and
  beams of different ellipticities, the gains are exact (7e-13) and the
  maps those of the circular beam of the model; without, the gains are off
  by 3e-3 (`test_blm_fixed_template_model_bl`).
- **HWP**: not yet. The polarization turns with $2\chi$ and the beam does
  not, so the beam needs the Mueller formalism (litebird_sim's
  `MuellerConvolver`); with an HWP the leakage of a common ellipticity is
  modulated away anyway.
- **Cost**: one `Interpolator` per detector, for all its `DetectorData`
  (its cube, of size
  $\sim (2\ell_\text{max})(4\ell_\text{max})(2m_\text{max}+1)$ per
  component, is built and freed in turn) and one more TOD per detector,
  kept in float64 in the model (it is not a GLS template). Building it and
  interpolating both grow with $2m_\text{max}+1$: by default `mmax` is the
  largest $m$ where the asymmetries have a coefficient above `epsilon`
  (default 1e-6, the accuracy of the convolution) times their largest, a
  few $m$ for an elliptical Gaussian ($m = 2k \sim e^k$), 10-15 for
  ellipticities 1.05-1.2 at 30' and $\ell_\text{max}$ 1024, while
  `generate_gauss_beam_alms` gives the beam up to $m = \ell_\text{max}$.
  Pass `mmax` to choose it (as litebird_sim's
  `BeamConvolutionParameters.mmax`; `kmax` in ducc0).

### Masks

Each fit has its own mask (1 for pixels used, 0 for pixels left out):
`CalibrationParameters(mask=...)` for the gains and `Template(mask=...)` for
each template, also without calibration. A template without a mask has that
of the gains (with calibration; else all pixels), so that by default each
sample is in all the fits or in none, the case where the calibration
converges fastest; for all pixels with calibration, pass a mask of ones.
Equal masks are kept in memory once. E.g. `galaxy * sources` for the gains
and `sources` for `Template(sky=dfg_dnu, sum_zero=True, mask=sources)`: the
bandpass amplitudes are fitted on the Galaxy, where the foreground is bright,
and point sources missing from the model are left out of everything. Samples
left out of every fit are flagged during the fit, then one more solve makes
the maps with all the samples and the fitted values. Samples left out of some
fits only keep the other templates; with calibration they are calibrated with
the values of the previous iteration, so the iterations converge slower
(`make_maps` warns; see `CalibrationParameters(gain_tol=...)`). In a 1-year
test with gains on $|b| > 10°$ and bandpass templates on the whole sky, 4
iterations left gain errors up to $10^{-3}$ in the periods with a weak dipole;
with the gain mask for the templates too, the errors were those of the noise,
and the polarization maps on the Galactic plane were the same. See
the [notebook](notebooks/sky_model).
