"""High-level map-making: from detector timelines and pointings to maps.

Any pipeline (litebird_sim, files on disk, ...) only needs to provide
:class:`DetectorData` objects, as a list or as a generator (then only one
detector's pointings are in memory at a time), each with a :class:`Pointing`:
either angles, which pysanepic rotates to the output coordinates and
pixelizes at the output resolution, or HEALPix pixels already computed at that
resolution and in those coordinates. pysanepic computes polarization angles,
chunks and noise weights itself, with one set of conventions documented here.
"""

import time
import warnings
from dataclasses import dataclass

import ducc0
import healpy as hp
import numba
import numpy as np

from .gls import GLS, fast_fft_length, numba_threads, one_over_f_weights, psd_weights, resolve_nthreads, two_angle
from .gls import _TRIG_COS, _TRIG_SIN, _add_pol, _axpy, _calibrated, _fill_columns, _project_iqu


_COORDINATES = ("E", "G", "C")


@dataclass(frozen=True, kw_only=True)
class Pointing:
    """Pointing of one detector over one stretch of time, in one of two forms.

    - Angles: `theta`, `phi` (colatitude and longitude [rad]) in `coordinates`.
      :func:`make_maps` rotates them to the coordinates of the output maps (with
      `psi`) and computes the pixels at the output resolution.
    - Pixels: `pix`, HEALPix pixel indices at resolution `nside` in `ordering`
      ("RING" or "NESTED"), computed in `coordinates`. They are used as they
      are, so `nside` and `coordinates` must be those of the output maps; pass
      them as int32 to avoid a copy.

    `psi` is the orientation angle of the detector frame [rad] in
    `coordinates`; `coordinates` is "E" (ecliptic), "G" (galactic) or "C"
    (equatorial). The arrays are checked when the object is created.
    """

    psi: np.ndarray
    coordinates: str
    theta: np.ndarray | None = None
    phi: np.ndarray | None = None
    pix: np.ndarray | None = None
    nside: int | None = None
    ordering: str = "RING"

    def __post_init__(self):
        if self.coordinates not in _COORDINATES:
            raise ValueError(f"Pointing.coordinates must be one of {_COORDINATES}, not {self.coordinates!r}")
        n = np.shape(self.psi)
        if len(n) != 1:
            raise ValueError("Pointing.psi must be a 1D array")
        if self.pix is None:
            if self.theta is None or self.phi is None:
                raise ValueError("Pointing needs either theta and phi, or pix and nside")
            if self.nside is not None:
                raise ValueError("Pointing.nside is only for pix: angles are pixelized at the output nside")
            if np.shape(self.theta) != n or np.shape(self.phi) != n:
                raise ValueError("Pointing.theta, phi and psi must have the same length")
            return
        if self.theta is not None or self.phi is not None:
            raise ValueError("Pointing takes either theta and phi, or pix, not both")
        if np.shape(self.pix) != n:
            raise ValueError("Pointing.pix and psi must have the same length")
        if not np.issubdtype(np.asarray(self.pix).dtype, np.integer):
            raise ValueError("Pointing.pix must be an integer array")
        if self.ordering not in ("RING", "NESTED"):
            raise ValueError(f"Unknown ordering {self.ordering!r}: use 'RING' or 'NESTED'")
        if self.nside is None or self.nside < 1 or (self.ordering == "NESTED" and self.nside & (self.nside - 1)):
            raise ValueError(f"Pointing.pix needs a valid nside, not {self.nside!r}")
        pix = np.asarray(self.pix)
        if pix.size and (pix.min() < 0 or pix.max() >= 12 * self.nside**2):
            raise ValueError(f"Pointing.pix outside [0, 12 nside^2) for nside={self.nside}")

    def __len__(self):
        return len(self.psi)

    def angles(self, coordinates, nthreads=1):
        """(theta, phi, psi) in `coordinates`, phi in [0, 2 pi); needs angles, not pixels."""
        if self.pix is not None:
            raise ValueError("this needs a Pointing with angles (theta, phi), not pixels")
        theta, phi, psi = self.theta, self.phi, self.psi
        if self.coordinates != coordinates:
            R = hp.Rotator(coord=[self.coordinates, coordinates]).mat
            out = np.empty((3, len(psi)))
            with numba_threads(nthreads):
                _rotate(np.ascontiguousarray(theta), np.ascontiguousarray(phi), np.ascontiguousarray(psi), R, out)
            theta, phi, psi = out
        return theta, np.mod(phi, 2 * np.pi), psi

    def pixels_and_psi(self, nside, coordinates, nthreads=1):
        """RING pixels at `nside` and psi in `coordinates` (the output maps)."""
        return self._pixels_psi_loc(nside, coordinates, nthreads)[:2]

    def _pixels_psi_loc(self, nside, coordinates, nthreads=1):
        """pixels_and_psi and the (theta, phi) of the samples, (n, 2), in
        `coordinates` (None for pixels): the angles are rotated only once."""
        if self.pix is not None:
            if self.nside != nside or self.coordinates != coordinates:
                raise ValueError(
                    f"Pointing.pix is at nside={self.nside} in coordinates {self.coordinates!r}: "
                    f"the maps need nside={nside} in coordinates {coordinates!r}"
                )
            pix = np.asarray(self.pix)
            if self.ordering == "NESTED":
                pix = ducc0.healpix.Healpix_Base(nside, "NEST").nest2ring(pix, nthreads=nthreads)
            return pix, np.asarray(self.psi), None
        theta, phi, psi = self.angles(coordinates, nthreads)
        loc = np.stack([theta, phi], axis=1)
        return ducc0.healpix.Healpix_Base(nside, "RING").ang2pix(loc, nthreads=nthreads), psi, loc


@dataclass(frozen=True, kw_only=True)
class OneOverFNoise:
    """1/f noise model of a detector, with sigma = net_ukrts * sqrt(fs) * 1e-6
    the white-noise rms per sample [K] at sampling rate fs:

    - `model="toast"` (default): P(f) = sigma^2 (f^alpha + fknee^alpha) /
      (f^alpha + fmin^alpha);
    - `model="keshner"`: P(f) = sigma^2 ((f^2 + fknee^2) / (f^2 +
      fmin^2))^(alpha/2), litebird_sim's noise with engine="ducc".

    fknee_hz = 0 or alpha = 0 is white noise. With fmin_hz = 0 the mean of each chunk is not
    used. One object can be shared by all the stretches of time of a detector.
    """

    net_ukrts: float
    fknee_hz: float = 0.0
    alpha: float = 1.0
    fmin_hz: float = 0.0
    model: str = "toast"

    def __post_init__(self):
        if self.model not in ("toast", "keshner"):
            raise ValueError(f"OneOverFNoise.model must be 'toast' or 'keshner', not {self.model!r}")
        if not (self.net_ukrts > 0 and self.fknee_hz >= 0 and self.fmin_hz >= 0 and self.alpha >= 0):
            raise ValueError("OneOverFNoise: net_ukrts must be > 0, fknee_hz, fmin_hz and alpha >= 0")

    def weights(self, ns, fs):
        """Inverse PSD on the rfft grid of a chunk of ns samples at fs [Hz]."""
        sigma = self.net_ukrts * np.sqrt(fs) * 1e-6
        return one_over_f_weights(ns, fs, sigma, self.fknee_hz, self.alpha, self.fmin_hz, self.model)


@dataclass(frozen=True, kw_only=True, eq=False)
class TabulatedNoise:
    """Noise of a detector given by a tabulated one-sided PSD `psd` [K^2/Hz] at
    `freqs_hz` (increasing, > 0), interpolated linearly in log f - log P and
    held constant outside the table: e.g. a measured or estimated spectrum.
    White noise of rms sigma per sample at sampling rate fs is 2 sigma^2 / fs.
    """

    freqs_hz: np.ndarray
    psd: np.ndarray

    def __post_init__(self):
        f, p = np.asarray(self.freqs_hz, dtype=float), np.asarray(self.psd, dtype=float)
        if f.ndim != 1 or f.shape != p.shape or f.size < 2:
            raise ValueError("TabulatedNoise: freqs_hz and psd must be 1D arrays of the same length >= 2")
        if not (np.all(np.isfinite(f)) and np.all(np.isfinite(p)) and f[0] > 0 and np.all(np.diff(f) > 0) and np.all(p > 0)):
            raise ValueError("TabulatedNoise: frequencies must be > 0 and increasing, the PSD finite and > 0")
        object.__setattr__(self, "freqs_hz", f)
        object.__setattr__(self, "psd", p)

    def weights(self, ns, fs):
        """Inverse PSD on the rfft grid of a chunk of ns samples at fs [Hz]."""
        return psd_weights(ns, fs, self.freqs_hz, self.psd)


@dataclass(frozen=True, kw_only=True)
class DetectorData:
    """One detector over one contiguous stretch of time.

    Polarization conventions (same as litebird_sim): the polarization angle seen by
    the sky is psi + pol_angle_rad without HWP, and
    psi + 2 hwp_angle - pol_angle_rad with an ideal HWP, with psi from `pointing`.
    The TOD model is d = I + pol_efficiency (Q cos 2a + U sin 2a) + noise, with
    `a` that angle; with calibration, d = g (I + pol_efficiency (Q cos 2a +
    U sin 2a) + dipole) + noise (see :class:`CalibrationParameters`).

    noise: the noise model, a :class:`OneOverFNoise` or a
    :class:`TabulatedNoise` (any object with a `weights(ns, fs)` method giving
    the inverse PSD on the rfft grid of a chunk), with the TOD in K.

    name: the detector, optional: estimate_noise averages the spectrum over
    all the DetectorData with the same name (e.g. the days of a detector).

    flags: boolean array like `tod`, True for samples to exclude (glitches,
    gaps, ...), or None. Flagged samples do not enter the maps and their TOD is
    not used (it may be NaN): each one is fitted by an unknown of its own,
    which is the exact GLS solution without them.

    dipole: the dipole signal in the TOD (K, like `tod`), or None; needed with
    calibration (see :class:`CalibrationParameters`), and then subtracted from
    the calibrated TOD. Ignored without calibration: subtract it from `tod`
    beforehand if it should not be in the maps.

    templates: the TOD of the templates of :func:`make_maps` given per
    detector, {template name: array like `tod`} (e.g. a thermometer reading),
    or None. A DetectorData without a template has no amplitude for it.
    """

    tod: np.ndarray
    pointing: Pointing
    sampling_rate_hz: float
    noise: OneOverFNoise | TabulatedNoise
    pol_angle_rad: float = 0.0
    pol_efficiency: float = 1.0
    hwp_angle: np.ndarray | None = None
    flags: np.ndarray | None = None
    name: str | None = None
    dipole: np.ndarray | None = None
    templates: dict | None = None

    def __post_init__(self):
        if self.dipole is not None and len(self.dipole) != len(self.tod):
            raise ValueError("DetectorData.dipole and tod must have the same length")
        if any(np.shape(t) != (len(self.tod),) for t in (self.templates or {}).values()):
            raise ValueError("DetectorData.templates must have one value per sample, like tod")
        if self.flags is not None:
            if not (isinstance(self.flags, np.ndarray) and self.flags.dtype == bool and self.flags.shape == (len(self.tod),)):
                raise ValueError("DetectorData.flags must be a boolean array with one value per sample (True = flagged)")
        if not callable(getattr(self.noise, "weights", None)):
            raise TypeError("DetectorData.noise must be a OneOverFNoise or a TabulatedNoise")
        if not isinstance(self.pointing, Pointing):
            raise TypeError("DetectorData.pointing must be a pysanepic.Pointing")
        if len(self.tod) != len(self.pointing):
            raise ValueError(f"DetectorData.tod has {len(self.tod)} samples, the pointing {len(self.pointing)}")
        if self.hwp_angle is not None and len(self.hwp_angle) != len(self.tod):
            raise ValueError("DetectorData.hwp_angle and tod must have the same length")


@dataclass(frozen=True, kw_only=True)
class CommonMode:
    """A noise mode shared by the detectors of a DetectorGroup: detector i sees
    coupling[i] * c(t), with c(t) of spectrum `noise` (a OneOverFNoise or a
    TabulatedNoise, in the units of the TOD: K)."""

    coupling: np.ndarray
    noise: "OneOverFNoise | TabulatedNoise"

    def __post_init__(self):
        object.__setattr__(self, "coupling", np.asarray(self.coupling, dtype=float))
        if self.coupling.ndim != 1 or not np.all(np.isfinite(self.coupling)):
            raise ValueError("CommonMode.coupling must be a 1D array, one value per detector of the group")
        if not callable(getattr(self.noise, "weights", None)):
            raise TypeError("CommonMode.noise must be a OneOverFNoise or a TabulatedNoise")


@dataclass(frozen=True, kw_only=True)
class DetectorGroup:
    """Detectors observed at the same times whose noise is correlated through
    common modes: N = D + U Lambda U^T per frequency, with D the own noise of
    each detector (DetectorData.noise), U the couplings and Lambda the spectra
    of the modes. The detectors must have the same number of samples and
    sampling rate; their chunks are not split at gaps (split_gap_s). A group
    must be on one MPI process."""

    detectors: list
    modes: list

    def __post_init__(self):
        d0 = self.detectors[0]
        if any(len(d.tod) != len(d0.tod) or d.sampling_rate_hz != d0.sampling_rate_hz for d in self.detectors):
            raise ValueError("the detectors of a DetectorGroup must have the same samples and sampling rate")
        if any(len(mode.coupling) != len(self.detectors) for mode in self.modes):
            raise ValueError("each CommonMode needs one coupling per detector of the group")


@dataclass
class MapResult:
    """Output of :func:`make_maps`.

    Attributes
    ----------
    maps : (3, npix) I/Q/U in K, or (1, npix) if pol=False; RING ordering,
        hp.UNSEEN where not solved. With calibration: from the calibrated TOD,
        DetectorData.dipole subtracted.
    hit_map : number of samples per pixel.
    nside, coordinates : pixelization of the maps.
    iterations : number of PCG iterations.
    converged : whether norm(r)^2 / norm(b)^2 < tol was reached.
    residuals : norm(r)^2 / norm(b)^2 at each iteration. With calibration,
        iterations, converged and residuals are those of the last GLS solve.
    gains : with calibration, the gains of the DetectorData of this rank (in
        the order of `data`, groups expanded): one array each, one gain per
        period of gain_period_s from its first sample. None otherwise.
    templates : with templates, the amplitudes of each one, {name:
        amplitudes}: {detector name: amplitude} (on every rank) for one
        amplitude per detector, a float for a common one, and for amplitudes
        per period one array per DetectorData of this rank (in the order of
        `data`, groups expanded; empty where it has no template), as `gains`.
        None otherwise.
    template_covariance : with template_covariance, the covariance of the
        amplitudes per detector and common, marginalized over the maps and the
        amplitudes per period (gains included): {"labels": [(template name,
        detector name or None for a common amplitude), ...], "matrix": (n, n)
        array}, on every rank. With calibration, that of the last iteration.
        The amplitudes summing to zero have the covariance of the
        constrained amplitudes. None otherwise.
    calibration_iterations : with calibration, the number of calibration
        iterations done (n_iter, or fewer with gain_tol). None otherwise.
    """

    maps: np.ndarray
    hit_map: np.ndarray
    nside: int
    coordinates: str
    iterations: int
    converged: bool
    residuals: np.ndarray
    gains: list | None = None
    templates: dict | None = None
    template_covariance: dict | None = None
    calibration_iterations: int | None = None


@dataclass(kw_only=True)
class GLSParameters:
    """Settings of the GLS map-maker (everything but the data, the output
    pixelization and the parallelization).

    Attributes
    ----------
    chunk_s : N^-1 is applied independently on chunks of this duration [s];
        longer chunks capture lower frequencies but cost more.
    pad_s : each chunk is extended by this duration [s] on both sides before
        applying N^-1, to avoid the wrap-around of the FFT (SANEPIC's "inpaint").
        The padding is fitted by one offset per margin ("virtual pixels", not in
        the output maps). 0 disables it.
    pad_fill : "zeros" or "extrapolate" (SANEPIC's choice: linear extrapolation
        tapered to the chunk mean). Zeros are exactly described by the margin
        offsets, so they add no bias; "extrapolate" leaves a small unmodelled
        signal that 1/f weighting spreads into the map, but with a steep 1/f
        noise (drifting data) it reduces the edge effects more.
    pol : solve for I/Q/U (True) or I only.
    preconditioner : "block" (3x3 I/Q/U block per pixel) or "jacobi" (SANEPIC's).
    min_pol_rcond : Q/U are solved only in pixels whose polarization-angle
        coverage gives a reciprocal condition number >= this value (0: no
        check, as SANEPIC).
    tol, maxiter : PCG stops when norm(r)^2 / norm(b)^2 < tol or after maxiter iterations.
    covariance_tol : tol of the PCG solves of make_maps(template_covariance=True),
        one per amplitude. 1e-6 (default) gives sigma and correlations to
        ~1e-4 (in a test with calibration, bandpass, angle and FWHM templates:
        sigma to 4e-5, correlations to 8e-4 against 1e-10), with ~1/3 fewer
        iterations than 1e-10.
    angle_dtype : storage of twice the polarization angle: np.float32 (4 bytes
        per sample, angle error ~1e-7 rad) or np.float64 (exact).
    flag_method : how runs of flagged samples (DetectorData.flags) shorter than
        split_gap_s are handled: "marginalize" (one unknown per sample, the
        exact GLS without them; ~70 bytes per flagged sample) or "offset" (one
        unknown per run, its samples set to 0: no bias, nearly optimal, no
        memory).
    fft_dtype : precision of the FFTs applying N^-1 to the PCG search
        directions: np.float64 (default) or np.float32 (~45% faster FFTs; the
        right-hand side and the true residual stay in float64, see GLS).
    template_dtype : storage of the TOD of the templates fitted with the maps
        (the gains of the calibration and those of make_maps' `templates`):
        np.float32 (default, 4 bytes per sample and template, rounded to a
        relative ~6e-8, far below the noise) or np.float64 (exact, 8 bytes).
    split_gap_s : runs of flagged samples at least this long [s] split the
        chunk, and their samples are dropped: the noise correlations across
        the gap are neglected, but long gaps do not slow down the PCG. inf:
        never; 0: every run.
    coarse_nside : two-level preconditioner: the PCG also solves exactly for
        the I map at a coarse resolution, which takes care of the slow
        large-scale modes. 0 (default): the I monopole alone (one coarse pixel,
        the whole sky), poorly constrained with 1/f weights and the slowest
        mode of the PCG, especially with calibration, where it is coupled to
        the common gain; its setup costs one PCG iteration (in a test it cut
        the iterations from 31 to 22 for maps, from 43 to 33 with
        calibration). N >= 1 (a power of 2, <= nside): the I map at nside N,
        whose setup costs about one PCG iteration per coarse pixel (12 at
        N = 1); with common modes 1 halves the iterations (46 -> 21 on G100,
        -33% time). None: no coarse level (the preconditioner of pysanepic
        0.4, where 0 meant this). The maps are the same within the tolerance.
    """

    chunk_s: float = 3600.0
    pad_s: float = 0.0
    pad_fill: str = "zeros"
    pol: bool = True
    preconditioner: str = "block"
    min_pol_rcond: float = 1e-2
    tol: float = 1e-12
    covariance_tol: float = 1e-6
    maxiter: int = 2000
    angle_dtype: type = np.float32
    flag_method: str = "marginalize"
    split_gap_s: float = 300.0
    fft_dtype: type = np.float64
    coarse_nside: int | None = 0
    template_dtype: type = np.float32


@dataclass(kw_only=True)
class CalibrationParameters:
    """Gain calibration on the dipole, fitted together with the maps.

    The TOD is modeled as d = g (P m + D + S + sum_k a_k B_k) + n, with P m the
    I/Q/U signal (as in :class:`DetectorData`), D = DetectorData.dipole, S and
    B_k the sky model and the templates of :func:`make_maps` (0 if
    not given), and one gain g per detector and period. Linearized around the
    current values (g = 1, a = 0 at the start): d / g - D - S - sum a B =
    P m_res + T dg + sum B da + n, with the template T = D + S + sum a B +
    P m_res,prev (m_res: the maps minus the sky model of the previous
    iteration, polarization included), solved for the maps, dg and da at
    once by the PCG, then g <- g (1 + dg), a <- a + da and again, `n_iter`
    times. Neglected: dg P (m_res - m_res,prev), which vanishes as the
    iterations converge, and the noise rescaled by 1 / g (weights unchanged).

    Only D (and the sky model, put back in the maps) is subtracted: a dipole
    left in the map (e.g. the solar one) must not be in D. A gain common to
    all periods would rescale the map too: it is fixed by the part of D that
    is not a sky signal (the orbital dipole), so D must include it; the
    solar dipole and the sky give the relative gains. Samples in pixels whose
    Q/U are not solved (see GLSParameters.min_pol_rcond) do not enter the
    fits (their polarization is not modelled) but are calibrated; from the
    second iteration they are flagged like those out of every mask (one more
    GLS setup), otherwise the iterations converge much slower. Signal not
    described by one sky map for all detectors biases the gains: structure
    below the pixel size (give the sky model as alm) and foregrounds seen
    through different bandpasses (a template of the foregrounds); mask what is left.
    The gains are unknowns local to the process of their detector (no MPI
    communication). The pol_efficiency and the polarization angles are not
    calibrated.

    Attributes
    ----------
    gain_period_s : one gain per detector and period of this duration [s],
        counted from the first sample of each DetectorData; inf: one gain per
        DetectorData.
    n_iter : calibration iterations (each one a full GLS solve; the setup of
        the GLS is built once and reused, and each solve starts from the
        maps of the previous one). The first one uses T = D + S. 2 (default)
        is enough when the noise dominates the second-order terms; increase
        it for noiseless tests or large gain errors. With gain_tol, the
        maximum number of iterations.
    gain_tol : None (default): always n_iter iterations. A number: stop
        after the first iteration whose largest absolute gain correction dg (all
        detectors and periods, all MPI processes) is below it. The
        iterations converge geometrically but slower where the gains are
        less constrained (periods with a weak dipole), and much slower with
        samples left out of some fits only (see mask): gain_tol stops on the
        worst period. MapResult.calibration_iterations has the number of
        iterations done.
    mask : HEALPix map (nside and coordinates of the output maps), 1 (True)
        for pixels used and 0 (False) for pixels left out of the gain fit, as
        the usual masks (combine them with a product), or None: all pixels.
        It is also the mask of the templates that have none (Template.mask),
        so that by default every sample is in all the fits or in none.
        Samples left out of all the active fits (gains and templates) are
        flagged in the n_iter
        iterations (with GLSParameters.flag_method; "offset" takes no memory;
        together with DetectorData.flags), then one more GLS solve makes the
        maps with all the samples and the final values; samples left out of
        some fits only keep those templates at 0, and the iterations converge
        slower (they are calibrated with the values of the previous
        iteration: ~5x per iteration in a noiseless test, against ~10^4 with
        the samples flagged). Zeroing a template does not stop signal that is
        not in the model from reaching the fit through the maps and the 1/f
        weights: mask it for every fit (e.g. point sources not in the sky
        model). E.g. galaxy * sources here and sources in the templates
        (make_maps warns when the masks of gains and templates differ).
    cache_model : keep the TOD of the sky model and of the sky templates
        of every detector in memory between the iterations (8 bytes per sample
        each) instead of computing them again (with alm, an interpolation at
        every sample in each iteration).
    covariance_fixed_gains : make_maps(template_covariance=True) with the
        gains fixed at their fitted values instead of marginalized over: the
        gain amplitudes couple to the sky templates through the sky signal in
        the gain template and slow down the PCG solves (~2x their iterations).
        In a test with a gain mask and bandpass, angle and FWHM sky templates,
        the errors were 0.05-0.13% smaller, with 35% fewer iterations.
        False (default): marginalized.
    """

    gain_period_s: float = 3600.0
    n_iter: int = 2
    gain_tol: float | None = None
    mask: np.ndarray | None = None
    cache_model: bool = False
    covariance_fixed_gains: bool = False

    def __post_init__(self):
        if self.gain_tol is not None and not self.gain_tol > 0:
            raise ValueError("CalibrationParameters.gain_tol must be None or > 0")
        if isinstance(self.mask, dict):
            raise TypeError("CalibrationParameters.mask is the mask of the gains (one map); "
                            "the templates have their own, Template.mask")


@dataclass(frozen=True, kw_only=True)
class Template:
    """A template fitted with the maps by :func:`make_maps` (templates=):
    the TOD is modeled as P m + sum_k a_k T_k(t) + n, with T_k(t) the
    template and a_k its amplitudes (inside the gain with calibration: the
    amplitudes are in the units of the calibrated TOD, see
    :class:`CalibrationParameters`). Templates nearly proportional to each
    other, or to the sky signal, are not separable: see
    MapResult.template_covariance.

    Attributes
    ----------
    sky : the template as a sky signal (e.g. dF/dnu for bandpass mismatch),
        maps or alm as make_maps' sky_model, sampled for every detector. None
        (default): the TOD given per detector in DetectorData.templates, under
        the name of the template.
    period_s : None (default): one amplitude per detector (DetectorData.name,
        shared by its DetectorData, also across MPI processes). A duration
        [s]: one amplitude per DetectorData and period of this duration,
        counted from its first sample, as the gains (inf: one per
        DetectorData); unknowns local to each process, without MPI
        communication, as many as wanted.
    common : one amplitude for all the detectors (with period_s None).
    sum_zero : the amplitudes of the detectors sum to zero (with period_s
        None): only the differences between detectors are fitted. Needed for
        a sky template as maps, whose common part is the map itself.
    mask : HEALPix map (nside and coordinates of the output maps), 1 (True)
        for pixels where the template is fitted and 0 (False) for pixels left
        out of its fit, or None (default): with calibration, the mask of the
        gains (CalibrationParameters.mask), else all pixels; all pixels with
        calibration: a mask of ones. Equal masks are kept once in memory.
        Samples left out of all the fits (gains included) are flagged in the
        fit and one more GLS solve makes the maps with all the samples and
        the fitted amplitudes;
        samples left out of some fits only keep those templates at 0 (see
        CalibrationParameters.mask).
    """

    sky: np.ndarray | None = None
    period_s: float | None = None
    common: bool = False
    sum_zero: bool = False
    mask: np.ndarray | None = None

    def __post_init__(self):
        if self.period_s is not None and not self.period_s > 0:
            raise ValueError("Template.period_s must be None or > 0")
        if self.period_s is not None and (self.common or self.sum_zero):
            raise ValueError("Template: common and sum_zero need period_s None (one amplitude per detector)")
        if self.common and self.sum_zero:
            raise ValueError("Template: a common amplitude summing to zero is 0; use one of common and sum_zero")


@numba.njit(parallel=True, cache=True)
def _rotate(theta, phi, psi, R, out):
    """Rotate pointings by matrix R into out[0]=theta, out[1]=phi, out[2]=psi.

    psi is corrected by the angle between the old and new local meridians
    (same as healpy Rotator.angle_ref and litebird_sim rotate_coordinates_e2g).
    """
    for i in numba.prange(theta.size):
        st = np.sin(theta[i])
        x, y, z = st * np.cos(phi[i]), st * np.sin(phi[i]), np.cos(theta[i])
        vx = R[0, 0] * x + R[0, 1] * y + R[0, 2] * z
        vy = R[1, 0] * x + R[1, 1] * y + R[1, 2] * z
        vz = R[2, 0] * x + R[2, 1] * y + R[2, 2] * z
        out[0, i] = np.arccos(min(1.0, max(-1.0, vz)))
        out[1, i] = np.arctan2(vy, vx) % (2 * np.pi)
        # a: old north (old z-axis = R[:, 2]) and b: new north, projected on the tangent plane at v
        zx, zy, zz = R[0, 2], R[1, 2], R[2, 2]
        d = zx * vx + zy * vy + zz * vz
        ax, ay, az = zx - d * vx, zy - d * vy, zz - d * vz
        bx, by, bz = -vz * vx, -vz * vy, 1.0 - vz * vz
        cx, cy, cz = by * az - bz * ay, bz * ax - bx * az, bx * ay - by * ax
        out[2, i] = psi[i] + np.arctan2(cx * vx + cy * vy + cz * vz, bx * ax + by * ay + bz * az)


def _pixels_and_angles(d, nside, coordinates, nthreads, loc=False):
    """Pixels and polarization angles; with loc, also the (theta, phi) of the
    samples for alm models (else, or for pixels, None)."""
    pix, psi, xy = d.pointing._pixels_psi_loc(nside, coordinates, nthreads)
    angle = psi + d.pol_angle_rad if d.hwp_angle is None else psi + 2 * d.hwp_angle - d.pol_angle_rad
    return pix, angle, xy if loc else None


def _pad(tod, m, fill, nfit_max=20000):
    """Extend a chunk by m samples on each side so that N^-1, applied with FFTs,
    does not see the jump between its two ends (the SANEPIC "inpaint").

    "zeros": zero padding. "extrapolate" (SANEPIC's choice, with its bugs fixed):
    linear fit over the first/last samples, extrapolated into the margins and
    tapered with a cosine to the chunk mean, so that the two ends join
    continuously.
    """
    if fill == "zeros":
        return np.concatenate([np.zeros(m), tod, np.zeros(m)])
    if fill != "extrapolate":
        raise ValueError(f"Unknown pad_fill {fill!r}: use 'extrapolate' or 'zeros'")
    nfit = min(nfit_max, tod.size)
    x = np.arange(nfit) - (nfit - 1) / 2

    def fit(seg):
        a = seg.mean()
        return a, np.dot(seg - a, x) / np.dot(x, x)

    a_l, b_l = fit(tod[:nfit])
    a_r, b_r = fit(tod[-nfit:])
    left = a_l + b_l * (np.arange(-m, 0) - (nfit - 1) / 2)
    right = a_r + b_r * (np.arange(nfit, nfit + m) - (nfit - 1) / 2)
    mean = tod.mean()
    taper = (1 - np.cos(np.pi * np.arange(m) / m)) / 2  # 0 far from the data, -> 1 next to it
    return np.concatenate([(left - mean) * taper + mean, tod, (right - mean) * taper[::-1] + mean])


def _chunk_bounds(n, ns):
    """Split n samples into chunks of ns; the leftover goes into the last chunk."""
    nchunk = max(1, n // ns)
    starts = np.arange(nchunk) * ns
    return zip(starts, np.append(starts[1:], n))


def _flag_runs(f):
    """(start, stop) of the runs of True in a boolean array."""
    edges = np.flatnonzero(np.diff(np.concatenate([[False], f, [False]]).astype(np.int8)))
    return edges[0::2], edges[1::2]


def _split_at_gaps(a, b, flags, min_len):
    """Pieces of the chunk [a, b) between runs of flagged samples of at least
    min_len samples, which are dropped."""
    if flags is None or not np.isfinite(min_len):
        return [(a, b)]
    starts, stops = _flag_runs(flags[a:b])
    long = stops - starts >= max(min_len, 1)
    cuts = [a] + [x for s0, s1 in zip(starts[long], stops[long]) for x in (a + s0, a + s1)] + [b]
    return [(lo, hi) for lo, hi in zip(cuts[0::2], cuts[1::2]) if hi > lo]


def _flag_labels(f, method):
    """Negative labels of the flagged samples of a chunk for GLS: one unknown
    per sample ("marginalize") or per run ("offset")."""
    if method == "marginalize":
        return -1 - np.arange(int(f.sum()), dtype=np.int32)
    starts, stops = _flag_runs(f)
    return np.repeat(-1 - np.arange(starts.size, dtype=np.int32), stops - starts)


def _monopole_pixel(nside):
    """Pixel labels -> 0 for the pixels of the sky, -1 for virtual pixels: a
    coarse level with the I monopole alone."""
    return lambda labels: np.where(labels < 12 * nside * nside, 0, -1)


def _coarse_pixels(nside, coarse_nside):
    """Pixel labels (RING, nside; virtual pixels beyond) -> coarse pixel (NESTED, coarse_nside), or -1."""
    shift = 2 * (int(nside).bit_length() - int(coarse_nside).bit_length())

    def coarse(labels):
        out = np.full(labels.size, -1, np.int64)
        real = labels < 12 * nside * nside
        out[real] = hp.ring2nest(nside, labels[real]) >> shift
        return out

    return coarse


def make_maps(data, nside, coordinates="G", params=None, comm=None, nthreads=None, verbose=False, calibration=None,
              sky_model=None, templates=None, template_covariance=False):
    """GLS maps with 1/f noise from a list of :class:`DetectorData`.

    Parameters
    ----------
    data : iterable of DetectorData
        A list, or a generator to keep the pointings of only one detector in
        memory at a time (the TODs are used without copies). With MPI, the data
        local to this rank (possibly empty).
    nside : int
        HEALPix resolution of the output maps (RING ordering).
    coordinates : str
        Coordinate system of the output maps ("G", "E" or "C").
    params : GLSParameters or None
        Settings of the map-maker; None: the defaults.
    comm : mpi4py communicator or None
        Every rank of `comm` must call this function.
    nthreads : int or None
        Threads per process (FFTs, ducc0, numba). None: OMP_NUM_THREADS if set,
        else 1, as in litebird_sim; with MPI, use (cores per node) / (processes
        per node).
    calibration : CalibrationParameters or None
        Fit the gains on the dipole together with the maps (see
        CalibrationParameters); `data` is then read n_iter times (+ 1 with a
        mask, also of a template): a generator is turned into a list first, so the pointings are
        still computed one detector at a time but the DetectorData are all
        kept. The gains are in MapResult.gains.
    sky_model : array or None
        A model of the whole sky seen by all the detectors (e.g. CMB +
        foregrounds at the reference frequency), I or I/Q/U in K in the
        coordinates of the maps: HEALPix maps at `nside` (real array) are
        sampled in the pixel, alm (complex, healpy order, mmax = lmax) are
        interpolated at the pointing (ducc0 synthesis_general; needs Pointing
        angles). Its TOD is
        subtracted and its binned map put back in the maps, so they are still
        those of the whole sky: with alm, the model removes its structure
        below the pixel size, which would otherwise leak through the 1/f
        weights (into the gains with calibration); with maps the maps change
        only in pixels whose Q/U are not solved (the model removes their
        polarization, which would leak too), and the calibration template.
    templates : dict or None
        {name: Template}: templates fitted with the maps (in one solve, or in
        each calibration iteration), e.g. {"fg_deriv1": Template(sky=dF/dnu,
        sum_zero=True)} for the bandpass mismatch of the foregrounds (dF/dnu
        around the frequency of the foreground in sky_model, not part of it:
        the amplitudes are then the shifts of the bands with respect to their
        mean), or {"thermal": Template(period_s=86400.0)} with the TOD of a
        thermometer in DetectorData.templates["thermal"]. See
        :class:`Template` for the amplitudes (per detector, common, summing
        to zero, or per period), and the documentation for the common part
        of sky templates with calibration. The amplitudes are in
        MapResult.templates.
    template_covariance : bool
        Also compute the covariance of the amplitudes per detector and common
        (MapResult.template_covariance), marginalized over the maps and the
        amplitudes per period: one more PCG solve per amplitude (with
        calibration, after the last iteration). False (default).
    """
    sky = None if sky_model is None else _model(sky_model, nside, "sky_model")
    tmpls = dict(templates or {})
    if any(not isinstance(t, Template) for t in tmpls.values()):
        raise TypeError("templates must be {name: pysanepic.Template}")
    skies = {k: _model(t.sky, nside, f"templates[{k!r}].sky") for k, t in tmpls.items() if t.sky is not None}
    for k, x in skies.items():
        if not np.iscomplexobj(x) and not tmpls[k].sum_zero:
            raise ValueError(f"templates[{k!r}].sky as maps needs sum_zero=True: the common part of its amplitudes "
                             "is the map itself (pass alm to fit it)")
    if calibration is None and sky is None and not tmpls:
        return _make_maps(data, nside, coordinates, params, comm, nthreads, verbose)[0]
    data = list(data)
    dets = [d for x in data for d in (x.detectors if isinstance(x, DetectorGroup) else [x])]
    alm = [k for k, x in [("sky_model", sky), *((f"templates[{k!r}].sky", b) for k, b in skies.items())]
           if x is not None and np.iscomplexobj(x)]
    if alm and any(d.pointing.pix is not None for d in dets):
        raise ValueError(f"{', '.join(alm)} as alm are interpolated at the pointing: they need Pointing angles "
                         "(theta, phi), not pixels (or pass them as maps)")
    if calibration is not None and any(d.dipole is None for d in dets):
        raise ValueError("calibration needs DetectorData.dipole")
    # shared amplitudes: one per template and detector (with the template), or one per template
    per_det = [k for k, t in tmpls.items() if t.period_s is None and not t.common]
    if per_det and any(d.name is None for d in dets):
        raise ValueError(f"templates {per_det} have one amplitude per detector: they need DetectorData.name")
    names = {k: sorted({d.name for d in dets if k in skies or k in (d.templates or {})}) for k in per_det}
    if comm is not None:
        names = {k: sorted(set().union(*comm.allgather(v))) for k, v in names.items()}
    labels = [(k, n) for k, t in tmpls.items() if t.period_s is None for n in names.get(k, [None])]
    index = {x: i for i, x in enumerate(labels)}
    mask_gain = _mask(None if calibration is None else calibration.mask, nside, "CalibrationParameters.mask")
    # a template without a mask has that of the gains; equal masks are one array (and one test per sample)
    masks = {k: mask_gain if t.mask is None else _mask(t.mask, nside, f"templates[{k!r}].mask")
             for k, t in tmpls.items()}
    distinct = [] if mask_gain is None else [mask_gain]
    for k, mk in masks.items():
        same = None if mk is None else next((u for u in distinct if u is mk or np.array_equal(u, mk)), None)
        if mk is not None and same is None:
            distinct.append(mk)
        masks[k] = mk if same is None else same
    if calibration is not None and any(mk is not mask_gain for mk in masks.values()):
        warnings.warn("templates with a mask different from that of the gains: their samples are calibrated with "
                      "the gains of the previous iteration and the calibration converges slower (see "
                      "CalibrationParameters.mask)", stacklevel=2)
    fit = _Fit(calibration=calibration, sky=sky, templates=tmpls, skies=skies, index=index, amp=np.zeros(len(index)),
               groups=[[index[k, n] for n in names[k]] for k in per_det if tmpls[k].sum_zero],
               local={k: None for k, t in tmpls.items() if t.period_s is not None},
               mask_gain=mask_gain, masks=masks,
               cache={} if calibration is not None and calibration.cache_model else None)
    n_cal = 0
    for _ in range(1 if calibration is None else calibration.n_iter):
        res = _make_maps(data, nside, coordinates, params, comm, nthreads, verbose, fit)[0]
        n_cal += 1
        if calibration is not None:
            dg = fit.dg_max if comm is None else max(comm.allgather(fit.dg_max))
            if verbose and (comm is None or comm.rank == 0):
                print(f"calibration iteration {n_cal}: max |dg| = {dg:.3e}")
            if calibration.gain_tol is not None and dg < calibration.gain_tol:
                break
            if n_cal == 1 and calibration.n_iter > 1:
                _mask_unsolved_pol(fit, res, verbose and (comm is None or comm.rank == 0))
    cov = None
    if template_covariance and index:
        p = GLSParameters() if params is None else params
        if calibration is not None and calibration.covariance_fixed_gains and fit.no_gains is not None:
            fit.gls.set_templates(*fit.no_gains)  # collective
        cov = {"labels": labels, "matrix": fit.gls.shared_covariance(tol=p.covariance_tol, maxiter=p.maxiter, verbose=verbose)}
    if fit.mask_gain is not None or any(m is not None for m in fit.masks.values()):  # samples may have been left out
        fit.fit, fit.gls = False, None  # other flags: a new GLS
        res = _make_maps(data, nside, coordinates, params, comm, nthreads, verbose, fit)[0]
    out = {}
    for k, t in tmpls.items():
        if t.period_s is not None:
            out[k] = fit.local[k]
        elif t.common:
            out[k] = float(fit.amp[index[k, None]])
        else:
            out[k] = {n: float(fit.amp[index[k, n]]) for n in names[k]}
    res.templates = out or None
    res.template_covariance = cov
    res.calibration_iterations = None if calibration is None else n_cal
    return res


@dataclass
class _Fit:
    """What _make_maps fits besides the maps, and the current estimates."""

    calibration: "CalibrationParameters | None"  # None: no gains (and no dipole)
    sky: np.ndarray | None  # sky model (maps or alm), subtracted and put back
    templates: dict  # name -> Template
    skies: dict  # name -> map or alm of the templates given as a sky signal
    index: dict  # (template name, detector name or None) -> shared amplitude
    amp: np.ndarray  # current shared amplitudes
    groups: list  # shared amplitudes summing to zero
    local: dict  # name -> current amplitudes per period, one array per DetectorData (None at the start)
    mask_gain: np.ndarray | None
    masks: dict  # name -> mask of the template, or None
    gains: list | None = None  # current gains, one array per DetectorData
    dg_max: float = 0.0  # largest |dg| of the last iteration on this rank
    residual: np.ndarray | None = None  # maps minus sky model of the previous iteration
    x0: np.ndarray | None = None  # the GLS solution of the previous iteration: the start of the next one
    gls: "GLS | None" = None  # the GLS of the fit iterations, reused (same pixels and flags)
    cache: dict | None = None  # with cache_model: DetectorData index -> (sky model TOD, sky template TODs)
    no_gains: tuple | None = None  # the templates of the GLS but the gains (covariance_fixed_gains)
    fit: bool = True  # False: final maps with gains and amplitudes fixed

    @property
    def alm(self):
        """Some model is alm: the samples need (theta, phi)."""
        return any(x is not None and np.iscomplexobj(x) for x in (self.sky, *self.skies.values()))


def _model(x, nside, name):
    """A sky model or template as (ncomp, n): HEALPix maps (real, (npix,) or
    (ncomp, npix)) or alm (complex, (nalm,) or (ncomp, nalm), healpy order,
    mmax = lmax); ncomp is 1 (I) or 3 (I/Q/U)."""
    if isinstance(x, dict):
        raise ValueError(f"{name}: one array (sum the components)")
    x = np.atleast_2d(np.asarray(x))
    if np.iscomplexobj(x):
        lmax = (int(np.sqrt(8 * x.shape[1] + 1)) - 3) // 2
        if (lmax + 1) * (lmax + 2) // 2 != x.shape[1]:
            raise ValueError(f"{name}: alm must have (lmax+1)(lmax+2)/2 coefficients (mmax = lmax)")
    elif x.shape[1] != 12 * nside * nside:
        raise ValueError(f"{name}: maps must have 12 nside^2 = {12 * nside * nside} pixels (alm: complex)")
    if x.shape[0] not in (1, 3):
        raise ValueError(f"{name}: I or I/Q/U, not {x.shape[0]} components")
    return x


def _mask(mask, nside, name):
    """A mask of a fit as an array, checked, or None (also for a mask using
    every pixel)."""
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.shape != (12 * nside * nside,):
        raise ValueError(f"{name} must have 12 nside^2 = {12 * nside * nside} pixels")
    return None if np.all(mask != 0) else mask


def _mask_unsolved_pol(fit, res, verbose):
    """After the first calibration iteration: the samples in pixels whose Q/U
    are not solved are out of every fit (their templates are 0, see
    GLS.set_templates) but in the maps, calibrated with the gains of the
    previous iteration, which slows the iterations down as a mask of the gains
    alone. Leave those pixels out of every mask: the samples are flagged in the
    next iterations (a new GLS) and back in the final maps. `res` is full on
    every rank, so all of them take the same decision."""
    if res.maps.shape[0] == 1:
        return
    out = (res.maps[1] == hp.UNSEEN) & (res.hit_map > 0)
    if not out.any():
        return
    if verbose:
        print(f"calibration: samples in {out.sum()} pixels without Q/U flagged in the next iterations")
    keep, new = ~out, {}

    def cut(m):  # equal masks stay one array
        if m is None:
            return keep
        if id(m) not in new:
            new[id(m)] = (m != 0) & keep
        return new[id(m)]

    fit.mask_gain = cut(fit.mask_gain)
    fit.masks = {k: cut(m) for k, m in fit.masks.items()}
    fit.gls = None


def _sample(x, d, pix, angle, loc, nthreads):
    """The TOD of a sky model (from _model, or None: 0) for one DetectorData:
    maps sampled in the pixel, alm interpolated at the pointing `loc` ((n, 2)
    theta, phi in the coordinates of the maps; ducc0 synthesis_general)."""
    if x is None:
        return np.zeros(len(pix))
    if not np.iscomplexobj(x):
        return _project_maps(x, pix, angle, d.pol_efficiency, nthreads)
    lmax = (int(np.sqrt(8 * x.shape[1] + 1)) - 3) // 2
    eps = 1e-6 if x.dtype == np.complex64 else 1e-10
    tod = ducc0.sht.synthesis_general(alm=x[:1], spin=0, lmax=lmax, loc=loc, epsilon=eps, nthreads=nthreads)[0]
    if x.shape[0] == 3:
        q, u = ducc0.sht.synthesis_general(alm=x[1:], spin=2, lmax=lmax, loc=loc, epsilon=eps, nthreads=nthreads)
        with numba_threads(nthreads):  # one parallel pass, cos and sin from a table (numpy: ~10 serial passes)
            _add_pol(tod, q, u, np.asarray(angle, dtype=float), float(d.pol_efficiency), _TRIG_COS, _TRIG_SIN)
    return tod


def _make_maps(data, nside, coordinates, params, comm, nthreads, verbose, fit=None):
    """make_maps with the current estimates in `fit` (a _Fit, or None), which
    are updated; returns (MapResult, GLS). In the calibration iterations after
    the first, the GLS of the first one is reused (same pixels, flags and
    chunks): only the TOD and the templates are built."""
    reuse = fit is not None and fit.fit and fit.gls is not None
    params = GLSParameters() if params is None else params
    if params.flag_method not in ("marginalize", "offset"):
        raise ValueError(f"Unknown flag_method {params.flag_method!r}: use 'marginalize' or 'offset'")
    nc = params.coarse_nside
    if nc is not None and (nc < 0 or (nc and (nc & (nc - 1) or nc > nside))):
        raise ValueError(f"coarse_nside must be None, 0 (the monopole) or a power of 2 <= nside, not {nc}")
    nthreads = resolve_nthreads(nthreads)
    npix = 12 * nside * nside
    rank, size = (0, 1) if comm is None else (comm.rank, comm.size)
    n_virtual = 0  # padding margins on this rank; labels are unique across ranks
    pix_c, a2_c, gamma_c, tod_c, w_c, nfft_c = [], [], [], [], [], []
    weights = {}  # chunks with the same length and noise share one array
    dtype = params.angle_dtype
    common_modes = []
    templates, shared, model_c = [], [], []  # local and shared templates (chunk, values, labels), sky model TOD
    gains, offsets = [], []  # gains of each DetectorData and their first label
    local = {} if fit is None else {k: [] for k in fit.local}  # (current amplitudes, first label) per DetectorData
    cal = None if fit is None else fit.calibration
    nchunk = ndata = nlab = 0  # chunks, DetectorData and local amplitudes so far
    gain_ids = set()  # ids of the gain templates in `templates`
    t0 = time.perf_counter()  # the time of each step, with verbose

    def add(d, split=True):
        """Chunks of one DetectorData; returns their indices."""
        nonlocal n_virtual, nchunk, ndata, nlab
        first = nchunk
        ndata += 1
        sample = fit is not None and (fit.cache is None or ndata not in fit.cache)
        # the (theta, phi) only for alm models, not to keep them otherwise
        pix, angle, loc = _pixels_and_angles(d, nside, coordinates, nthreads, loc=sample and fit.alm)
        fs = d.sampling_rate_hz
        m = int(round(params.pad_s * fs))
        flags, tod_all, tmpl, ttod, tlab = d.flags, d.tod, None, {}, {}
        if fit is not None:
            # d = g (P m + D + S + sum_k a_k B_k) + n, linearized around the current g, a:
            # d / g - D - S - sum a B = P m_res + T dg + sum B da + n, T = D + S + sum a B + P m_res_prev
            if not sample:
                sky_tod, cached = fit.cache[ndata]
                ttod = {k: t.copy() for k, t in cached.items()}  # zeroed below in the masks
            else:
                sky_tod = _sample(fit.sky, d, pix, angle, loc, nthreads)
                ttod = {k: _sample(b, d, pix, angle, loc, nthreads) for k, b in fit.skies.items()}
                if fit.cache is not None:
                    fit.cache[ndata] = sky_tod, {k: t.copy() for k, t in ttod.items()}
            ttod.update({k: np.array(v, dtype=float) for k, v in (d.templates or {}).items()
                         if k in fit.templates and k not in fit.skies})

            def period(seconds):  # the period of each sample
                per = int(round(seconds * fs)) if np.isfinite(seconds) else len(d.tod)
                return np.arange(len(d.tod)) // max(1, per)

            known = np.array(sky_tod, dtype=float)
            with numba_threads(nthreads):
                for k, t in ttod.items():
                    tp = fit.templates[k]
                    if tp.period_s is None:
                        _axpy(known, float(fit.amp[fit.index[k, None if tp.common else d.name]]), t)
            if cal is not None:
                p = period(cal.gain_period_s)
                gain = np.ones(p[-1] + 1) if fit.gains is None else fit.gains[len(gains)]
                offsets.append(nlab)
                nlab += gain.size
                gains.append(gain)
                gain_period = p
                labels = offsets[-1] + p
            for k, amps in local.items():
                if k not in ttod:
                    amps.append((np.zeros(0), nlab))
                    continue
                p = period(fit.templates[k].period_s)
                amp = np.zeros(p[-1] + 1) if fit.local[k] is None else fit.local[k][ndata - 1]
                known += amp[p] * ttod[k]
                amps.append((amp, nlab))
                tlab[k] = nlab + p
                nlab += amp.size
            if cal is not None and fit.fit:
                tmpl = d.dipole + known
                if fit.residual is not None:
                    with numba_threads(nthreads):
                        _axpy(tmpl, 1.0, _project_maps(fit.residual, pix, angle, d.pol_efficiency, nthreads))
            if cal is not None:
                tod_all = np.empty(len(d.tod))
                with numba_threads(nthreads):
                    _calibrated(d.tod, gain, gain_period, d.dipole, known, tod_all)
            else:
                tod_all = tod_all - known
            # masks: samples out of every active fit are flagged; out of some, their templates are 0
            active = [(tmpl, fit.mask_gain)] if tmpl is not None else []
            active += [(t, fit.masks[k]) for k, t in ttod.items()] if fit.fit else []
            if any(mk is not None for _, mk in active):
                out = {}  # samples out of each distinct mask
                for _, mk in active:
                    if mk is not None and id(mk) not in out:
                        out[id(mk)] = mk[pix] == 0
                drop = None if any(mk is None for _, mk in active) else np.logical_and.reduce(list(out.values()))
                for t, mk in active:
                    if mk is not None and (drop is None or len(out) > 1):
                        t[out[id(mk)] if drop is None else out[id(mk)] & ~drop] = 0.0
                if drop is not None:
                    flags = drop if flags is None else flags | drop
        chunks = _chunk_bounds(len(d.tod), max(1, int(round(params.chunk_s * fs))))
        gap = params.split_gap_s * fs if split else np.inf
        for a, b in (piece for c in chunks for piece in _split_at_gaps(*c, flags, gap)):
            k = nchunk
            nchunk += 1
            tod = tod_all[a:b]  # a view: the TOD is not copied
            if fit is not None:
                # values as template_dtype and int32 labels (0 in the margins, where
                # the values are 0), one int for an amplitude in the whole chunk
                dt = params.template_dtype
                if tmpl is not None:
                    templates.append((k, _padded(tmpl, a, b, m, dt), _padded(labels, a, b, m, np.int32)))
                    gain_ids.add(id(templates[-1]))
                if fit.fit:
                    cols = []  # the templates with one amplitude in the chunk: one 2D array, one pass in the PCG
                    for name, t in ttod.items():
                        if name in tlab:
                            templates.append((k, _padded(t, a, b, m, dt), _padded(tlab[name], a, b, m, np.int32)))
                        else:
                            cols.append((t, fit.index[name, None if fit.templates[name].common else d.name]))
                    if len(cols) == 1:
                        shared.append((k, _padded(cols[0][0], a, b, m, dt), cols[0][1]))
                    elif cols:
                        values = np.zeros((b - a + 2 * m, len(cols)), dt)
                        with numba_threads(nthreads):
                            _fill_columns(values, tuple(np.asarray(t, dtype=float) for t, _ in cols), a, m)
                        shared.append((k, values, np.array([i for _, i in cols])))
                if fit.sky is not None:  # the templates are not put back
                    model_c.append(_padded(sky_tod, a, b, m, np.float64))
            flagged = None if flags is None else flags[a:b]
            if m and flagged is not None and flagged.any():  # the padding is built from the chunk: not from flagged samples
                tod = np.where(flagged, tod[~flagged].mean() if not flagged.all() else 0.0, tod)
            tod_c.append(tod if m == 0 else _pad(tod, m, params.pad_fill))
            if reuse:  # the GLS has pixels, angles and weights already
                continue
            n = b - a + 2 * m
            nfft = fast_fft_length(n)  # e.g. chunks split at gaps have any length
            key = (nfft, fs, d.noise)  # the dict keeps the noise objects alive: no reused ids
            if key not in weights:
                weights[key] = d.noise.weights(nfft, fs)
            w_c.append(weights[key])
            nfft_c.append(nfft)
            # 2a computed in float64, then stored as `dtype`; the margins of a
            # padded chunk carry no polarization (gamma = 0: intensity-only
            # virtual pixels), so only padded chunks need gamma per sample
            if params.pol:
                a2 = np.zeros(n, dtype)
                a2[m : n - m] = two_angle(angle[a:b])
                a2_c.append(a2)
                if m == 0:
                    gamma_c.append(d.pol_efficiency)
                else:
                    g = np.zeros(n, dtype)
                    g[m : n - m] = d.pol_efficiency
                    gamma_c.append(g)
            p = pix[a:b].astype(np.int32)
            if flagged is not None and flagged.any():
                p[flagged] = _flag_labels(flagged, params.flag_method)  # unknowns not in the maps (see GLS)
            if m == 0:
                pix_c.append(p)
            else:
                left = npix + 2 * (n_virtual * size + rank)
                n_virtual += 1
                pix_c.append(np.concatenate([np.full(m, left, np.int32), p, np.full(m, left + 1, np.int32)]))
        del pix, angle
        return list(range(first, nchunk))

    for d in data:  # one detector (or group) at a time: only its pointings are in memory
        if not isinstance(d, DetectorGroup):
            add(d)
            continue
        # a group: chunks at the same times, not split at gaps (that would break
        # the alignment); one common_modes entry per chunk of time
        idx = [add(x, split=False) for x in d.detectors]
        if reuse:
            continue
        fs = d.detectors[0].sampling_rate_hz
        U = np.array([mode.coupling for mode in d.modes], dtype=float).T  # (ndet, r)
        for chunks in zip(*idx):
            nfft = nfft_c[chunks[0]]
            lam = np.array([1 / np.maximum(w, w.max() * 1e-12) for w in (mode.noise.weights(nfft, fs) for mode in d.modes)])
            common_modes.append((list(chunks), U, lam))

    t1 = time.perf_counter()
    if fit is not None and fit.fit and cal is not None and cal.covariance_fixed_gains:
        fit.no_gains = ([t for t in templates if id(t) not in gain_ids], shared)  # the same arrays: no copies
    if reuse:
        gls = fit.gls
        gls.set_templates(templates, shared)
    else:
        gls = _new_gls(pix_c, w_c, a2_c, gamma_c, nfft_c, common_modes, templates, shared, params, nside, comm, nthreads, fit)
        if fit is not None and fit.fit:
            fit.gls = gls
    del pix_c, a2_c, gamma_c
    # warm start from the previous calibration iteration (if the pixels are the same)
    x0 = None if fit is None or fit.x0 is None or fit.x0.shape != (gls.ncomp, gls.nobs) else fit.x0
    t2 = time.perf_counter()
    m, info = gls.solve(tod_c, x0=x0, tol=params.tol, maxiter=params.maxiter, verbose=verbose)
    t3 = time.perf_counter()
    if fit is not None:
        fit.x0 = m
        if fit.fit:
            fit.residual = gls.to_healpix(m, nside, fill=hp.UNSEEN)
            if cal is not None:
                dg = [info["amplitudes"][o : o + g.size] for g, o in zip(gains, offsets)]
                fit.dg_max = max((float(np.abs(x).max()) for x in dg if x.size), default=0.0)
                gains = [g * (1 + x) for g, x in zip(gains, dg)]
            for k, amps in local.items():
                fit.local[k] = [a + info["amplitudes"][o : o + a.size] for a, o in amps]
            if fit.index:
                fit.amp = fit.amp + info["shared_amplitudes"]
        fit.gains = gains if cal is not None else None
        if fit.sky is not None:  # put back the sky model, as binned from its TOD: the maps of the whole sky
            m = m + gls.binned(model_c)

    hit_map = np.zeros(npix, dtype=np.int64)
    n = min(npix, gls.nobs)
    hit_map[:n] = gls.full(gls.hits)[:n]
    maps = gls.to_healpix(m, nside, fill=hp.UNSEEN)
    if verbose and rank == 0:
        print(f"TOD and templates {t1 - t0:.1f} s, GLS setup {t2 - t1:.1f} s, PCG {t3 - t2:.1f} s "
              f"({info['iterations']} iterations), maps {time.perf_counter() - t3:.1f} s", flush=True)
    return MapResult(
        maps=maps,
        hit_map=hit_map,
        nside=nside,
        coordinates=coordinates,
        iterations=info["iterations"],
        converged=info["converged"],
        residuals=info["residuals"],
        gains=None if fit is None else fit.gains,
    ), gls


def _padded(x, a, b, m, dtype):
    """x[a:b] as `dtype` with m zeros on both sides (a view if m = 0 and x is of dtype)."""
    if m == 0:
        return np.asarray(x[a:b], dtype)
    out = np.zeros(b - a + 2 * m, dtype)
    out[m : m + b - a] = x[a:b]
    return out


def _new_gls(pix_c, w_c, a2_c, gamma_c, nfft_c, common_modes, templates, shared, params, nside, comm, nthreads, fit):
    nc = params.coarse_nside
    return GLS(
        pix_c,
        None,
        w_c,
        pol_angles=(a2_c, gamma_c) if params.pol else None,
        preconditioner=params.preconditioner,
        min_pol_rcond=params.min_pol_rcond,
        comm=comm,
        nthreads=nthreads,
        angle_dtype=params.angle_dtype,
        overwrite_pix=True,  # pix_c holds copies made here
        fft_lengths=nfft_c,
        fft_dtype=params.fft_dtype,
        common_modes=common_modes,
        coarse=None if nc is None else _coarse_pixels(nside, nc) if nc else _monopole_pixel(nside),
        templates=templates,
        shared_templates=shared,
        shared_groups=fit.groups if fit is not None and fit.fit else None,
        template_dtype=params.template_dtype,
    )


def _model_tod(d, maps, nside, coordinates, nthreads):
    """P m for one DetectorData: the TOD of the maps (UNSEEN Q/U taken as 0)."""
    pix, angle, _ = _pixels_and_angles(d, nside, coordinates, nthreads)
    return _project_maps(maps, pix, angle, d.pol_efficiency, nthreads)


def _project_maps(maps, pix, angle, pol_efficiency, nthreads=1):
    """P m for pixels and polarization angles (UNSEEN taken as 0), in parallel."""
    out = np.empty(len(pix))
    with numba_threads(nthreads):
        _project_iqu(np.ascontiguousarray(maps, dtype=float), np.asarray(pix), np.asarray(angle, dtype=float),
                     float(pol_efficiency), _TRIG_COS, _TRIG_SIN, out)
    return out


def _fit_one_over_f(f, psd, fs, model):
    """Least-squares fit of log P(f) with OneOverFNoise(model): parameters log
    sigma^2, log fknee, alpha, log fmin; a few Levenberg-Marquardt steps."""
    y = np.log(psd)

    def logp(q):
        s2, fk, alpha, fm = np.exp(q[0]), np.exp(q[1]), q[2], np.exp(q[3])
        if model == "toast":
            shape = (f**alpha + fk**alpha) / (f**alpha + fm**alpha)
        else:
            shape = ((f**2 + fk**2) / (f**2 + fm**2)) ** (alpha / 2)
        return np.log(2 * s2 / fs * shape)

    white = np.median(psd[f > f[-1] / 4])
    q = np.array([np.log(white * fs / 2), np.log(max(f[np.argmin(np.abs(psd - 2 * white))], f[0])), 1.0, np.log(f[0])])
    lam = 1e-2
    for _ in range(200):
        r = y - logp(q)
        J = np.empty((f.size, 4))
        for j in range(4):
            dq = np.zeros(4)
            dq[j] = 1e-6
            J[:, j] = (logp(q + dq) - logp(q - dq)) / 2e-6
        # damped least squares; lstsq, as fmin may be unconstrained (below the lowest bin)
        damp = np.sqrt(lam * np.maximum(np.diag(J.T @ J), 1e-12))
        step = np.linalg.lstsq(np.vstack([J, np.diag(damp)]), np.concatenate([r, np.zeros(4)]), rcond=None)[0]
        if np.sum((y - logp(q + step)) ** 2) < np.sum(r**2):
            q, lam = q + step, lam / 3
            if np.abs(step).max() < 1e-8:
                break
        else:
            lam *= 10
    sigma = np.sqrt(np.exp(q[0]))
    return OneOverFNoise(net_ukrts=sigma / np.sqrt(fs) * 1e6, fknee_hz=float(np.exp(q[1])),
                         alpha=float(q[2]), fmin_hz=float(np.exp(q[3])), model=model)


def _group_cross_spectra(g, result, chunk_s, nbins, nthreads, acc):
    """Add the binned cross-periodograms of the residuals of the detectors of
    DetectorGroup `g` to acc[names] = [fs, R (nbins, n, n), modes, sum of log f]."""
    dets = g.detectors
    if any(d.name is None for d in dets):
        raise ValueError("estimate_noise: the detectors of a DetectorGroup need a name")
    key, n, fs = tuple(d.name for d in dets), len(dets), dets[0].sampling_rate_hz
    res = np.array([np.asarray(d.tod, dtype=float) - _model_tod(d, result.maps, result.nside, result.coordinates, nthreads) for d in dets])
    good = np.ones(res.shape, bool)
    for i, d in enumerate(dets):
        if d.flags is not None:
            good[i] = ~d.flags
    res[~good] = 0.0
    edges = np.geomspace(1.0 / chunk_s, fs / 2, nbins + 1)
    a_ = acc.setdefault(key, [fs, np.zeros((nbins, n, n)), np.zeros(nbins), np.zeros(nbins)])
    if a_[0] != fs:
        raise ValueError(f"group {key!r}: different sampling rates")
    for a, b in _chunk_bounds(res.shape[1], max(2, int(round(chunk_s * fs)))):
        if good[:, a:b].mean(axis=1).min() < 0.5:
            continue
        win = np.hanning(b - a)
        gw = good[:, a:b] * win
        norm = 2.0 / (fs * (gw @ gw.T))  # flagged samples: normalized by the overlap of each pair
        X = np.array([ducc0.fft.r2c(x * win, nthreads=nthreads) for x in res[:, a:b]])[:, 1:-1]
        f = np.fft.rfftfreq(b - a, 1 / fs)[1:-1]
        lo, hi = np.searchsorted(f, edges[:-1]), np.searchsorted(f, edges[1:])
        for k in np.flatnonzero(hi > lo):
            x = X[:, lo[k] : hi[k]]
            a_[1][k] += norm * (x @ x.conj().T).real
            a_[2][k] += hi[k] - lo[k]
            a_[3][k] += np.log(f[lo[k] : hi[k]]).sum()


def _fit_common_modes(R, w, r, maxiter=2000, tol=1e-10):
    """EM fit of R_b = A diag(P_b) A^T + diag(N_b) to the averaged
    cross-spectra R (nb, n, n), w (nb) modes per bin (the spectral matching
    of SMICA, as in sanePS of the IAS SANEPIC). Returns A (n, r), each column
    with rms 1, P (nb, r), N (nb, n)."""
    nb, n, _ = R.shape
    d = np.einsum("bii->bi", R)
    _, V = np.linalg.eigh(np.einsum("b,bij->ij", w, R / d.mean(1)[:, None, None]))
    A = V[:, ::-1][:, :r] * np.sqrt(n)
    A *= np.where(A.sum(0) < 0, -1.0, 1.0)
    P = np.tile(0.5 * d.mean(1)[:, None] / r, (1, r))
    N = 0.5 * d
    last = None
    for _ in range(maxiter):
        # E step: moments of the modes given the data, per bin
        C = np.linalg.inv(np.einsum("ia,bi,ic->bac", A, 1 / N, A) + np.eye(r) / P[:, :, None])
        W = np.einsum("bac,ic,bi->bai", C, A, 1 / N)
        Rxs = np.einsum("bij,baj->bia", R, W)
        Rss = np.einsum("bai,bij,bcj->bac", W, R, W) + C
        # M step: A row by row (the own noise depends on the bin), then P and N
        for i in range(n):
            A[i] = np.linalg.solve(np.einsum("b,bac->ac", w / N[:, i], Rss), np.einsum("b,ba->a", w / N[:, i], Rxs[:, i]))
        P = np.einsum("baa->ba", Rss)
        AR = np.einsum("ia,bja->bij", A, Rxs)
        N = np.maximum(np.einsum("bii->bi", R - AR - AR.transpose(0, 2, 1) + np.einsum("ia,bac,jc->bij", A, Rss, A)), 1e-6 * d)
        SR = np.linalg.solve(np.einsum("ia,ba,ja->bij", A, P, A) + N[:, :, None] * np.eye(n), R)
        crit = np.sum(w * (np.einsum("bii->b", SR) - np.linalg.slogdet(SR)[1] - n))
        if last is not None and abs(last - crit) <= tol * abs(crit):
            break
        last = crit
    s = np.sqrt(np.mean(A**2, axis=0))
    return A / s, P * s**2, N


def estimate_noise(data, result, chunk_s=86400.0, nbins=60, fit=None, comm=None, nthreads=None, n_modes=1):
    """Noise spectrum of each detector from the residual of a map.

    The residual d - P m of each DetectorData, with m the maps of `result`
    (a MapResult, e.g. a first make_maps), is cut in chunks of `chunk_s`
    seconds; the periodogram of each chunk (Hann window, flagged samples set
    to 0 and their fraction corrected) is averaged over all the chunks of the
    DetectorData with the same `name` (on all MPI processes), in `nbins`
    logarithmic bins from 1/chunk_s to fs/2.

    For a DetectorGroup the cross-spectra of its detectors are averaged in the
    same way over all the groups with the same detectors (e.g. the days of a
    wafer), and fitted with `n_modes` common modes plus the own noise of each
    detector, R(f) = A P(f) A^T + diag N(f), by expectation-maximization (as
    sanePS of the IAS SANEPIC). The modes of the group in `data` are not used.

    Returns a dict {name: noise}: a TabulatedNoise with the binned one-sided
    PSD [K^2/Hz], or with `fit="toast"` or `"keshner"` a OneOverFNoise fitted
    to it. DetectorData without a name are keyed by their position in `data`.
    For the detectors of a group the noise is their own noise, and the key
    tuple(names) gives the list of the `n_modes` CommonMode (couplings with
    rms 1), so that ``DetectorGroup(detectors=[replace(d, noise=est[d.name])
    ...], modes=est[names])`` uses the estimate.

    The residual misses the part of the noise that the map absorbs (small
    when there are many more samples than pixels; mostly at the harmonics of
    the scan), so iterating make_maps / estimate_noise once is usually enough.
    """
    nthreads = resolve_nthreads(nthreads)
    acc = {}  # name -> [fs, sum of P per bin, count per bin, sum of log f per bin]
    gacc = {}  # names of a group -> [fs, sum of the cross-spectra per bin, count, sum of log f]
    for i, d in enumerate(data):
        if isinstance(d, DetectorGroup):
            _group_cross_spectra(d, result, chunk_s, nbins, nthreads, gacc)
            continue
        key = i if d.name is None else d.name
        fs = d.sampling_rate_hz
        res = np.asarray(d.tod, dtype=float) - _model_tod(d, result.maps, result.nside, result.coordinates, nthreads)
        if d.flags is not None:
            res[d.flags] = 0.0
        edges = np.geomspace(1.0 / chunk_s, fs / 2, nbins + 1)
        a_ = acc.setdefault(key, [fs, np.zeros(nbins), np.zeros(nbins), np.zeros(nbins)])
        if a_[0] != fs:
            raise ValueError(f"detector {key!r}: DetectorData with different sampling rates")
        for a, b in _chunk_bounds(res.size, max(2, int(round(chunk_s * fs)))):
            x = res[a:b]
            good = 1.0 if d.flags is None else 1.0 - d.flags[a:b].mean()
            if good < 0.5:
                continue
            win = np.hanning(x.size)
            p = 2 * np.abs(ducc0.fft.r2c(x * win, nthreads=nthreads)) ** 2 / (fs * np.sum(win**2) * good)
            f = np.fft.rfftfreq(x.size, 1 / fs)
            k = np.searchsorted(edges, f[1:-1], side="right") - 1
            ok = (k >= 0) & (k < nbins)
            np.add.at(a_[1], k[ok], p[1:-1][ok])
            np.add.at(a_[2], k[ok], 1)
            np.add.at(a_[3], k[ok], np.log(f[1:-1][ok]))
    if comm is not None:  # the DetectorData of a detector (or the groups of a wafer) may be on several processes
        merged, gmerged = {}, {}
        for part, gpart in comm.allgather((acc, gacc)):
            for src, dst in ((part, merged), (gpart, gmerged)):
                for key, (fs, s, n, lf) in src.items():
                    m = dst.setdefault(key, [fs, np.zeros_like(s), np.zeros_like(n), np.zeros_like(lf)])
                    m[1] += s
                    m[2] += n
                    m[3] += lf
        acc, gacc = merged, gmerged
    grouped = {name for key in gacc for name in key}
    if grouped & set(acc) or len(grouped) != sum(len(key) for key in gacc):
        raise ValueError("estimate_noise: a detector is in different groups, or both alone and in a group")

    def noise(f, psd, fs):
        return TabulatedNoise(freqs_hz=f, psd=psd) if fit is None else _fit_one_over_f(f, psd, fs, fit)

    out = {}
    for key, (fs, s, n, lf) in acc.items():
        ok = n > 0
        out[key] = noise(np.exp(lf[ok] / n[ok]), s[ok] / n[ok], fs)
    for key, (fs, R, n, lf) in gacc.items():
        if (len(key) - n_modes) ** 2 < len(key) + n_modes:
            raise ValueError(f"estimate_noise: {n_modes} common modes cannot be estimated from {len(key)} detectors")
        ok = n > 0
        f = np.exp(lf[ok] / n[ok])
        A, P, N = _fit_common_modes(R[ok] / n[ok][:, None, None], n[ok], n_modes)
        for i, name in enumerate(key):
            out[name] = noise(f, N[:, i], fs)
        out[key] = [CommonMode(coupling=A[:, k], noise=noise(f, P[:, k], fs)) for k in range(n_modes)]
    return out

