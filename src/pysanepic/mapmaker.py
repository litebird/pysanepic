"""High-level map-making: from detector timelines and pointings to maps.

Any pipeline (litebird_sim, files on disk, ...) only needs to provide
:class:`DetectorData` objects, as a list or as a generator (then only one
detector's pointings are in memory at a time), each with a :class:`Pointing`:
either angles, which pysanepic rotates to the output coordinates and
pixelizes at the output resolution, or HEALPix pixels already computed at that
resolution and in those coordinates. pysanepic computes polarization angles,
chunks and noise weights itself, with one set of conventions documented here.
"""

import functools
import time
import warnings
from types import SimpleNamespace
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from enum import Enum

import ducc0
import healpy as hp
import numba
import numpy as np

from .gls import GLS, fast_fft_length, numba_threads, one_over_f_weights, psd_weights, resolve_nthreads, two_angle
from .gls import _GapFilledTOD, _Timings, _timing_table, _Q2, _TRIG_COS, _TRIG_SIN, _add_pol, _axpy, _calibrated, _fill_columns, _project_iqu, _validate_act_ncoeff
from .gapfill import conditional_gap_fill, _validate_gap_solve, _validate_gap_preconditioner


class Coordinates(str, Enum):
    """Coordinate system of the pointings and of the maps, with the healpy
    letters as values. It is a str: "E", "G" and "C" can be passed instead,
    and compare equal to the members."""

    Ecliptic = "E"
    Galactic = "G"
    Equatorial = "C"

    def __str__(self):
        return self.value


def _coordinates(x, what):
    """`x` ("E", "G", "C" or a Coordinates) as a Coordinates; ValueError otherwise."""
    try:
        return Coordinates(x)
    except ValueError:
        raise ValueError(f"{what} must be one of {[c.value for c in Coordinates]} (or a pysanepic.Coordinates), "
                         f"not {x!r}") from None


# astropy frames, as litebird_sim (ecliptic: BarycentricMeanEcliptic, equatorial: ICRS)
_ASTROPY_FRAMES = {Coordinates.Ecliptic: "barycentricmeanecliptic", Coordinates.Galactic: "galactic",
                   Coordinates.Equatorial: "icrs"}


@functools.lru_cache
def _rotation_matrix(coord_in, coord_out):
    """3x3 matrix from `coord_in` to `coord_out` ("E", "G" or "C"), with astropy
    as litebird_sim's ECL_TO_GAL_ROT_MATRIX: its columns are the axes of
    `coord_in` expressed in `coord_out`."""
    from astropy.coordinates import CartesianRepresentation, SkyCoord

    axes = SkyCoord(CartesianRepresentation(np.eye(3)), frame=_ASTROPY_FRAMES[coord_in])
    return np.ascontiguousarray(axes.transform_to(_ASTROPY_FRAMES[coord_out]).cartesian.xyz.value)


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
    `coordinates`; `coordinates` is a :class:`Coordinates` or its value, "E"
    (ecliptic), "G" (galactic) or "C" (equatorial), stored as a
    :class:`Coordinates`, with the astropy frames of litebird_sim
    (BarycentricMeanEcliptic, Galactic, ICRS). The arrays are checked when the object is created.
    """

    psi: np.ndarray
    coordinates: Coordinates
    theta: np.ndarray | None = None
    phi: np.ndarray | None = None
    pix: np.ndarray | None = None
    nside: int | None = None
    ordering: str = "RING"

    def __post_init__(self):
        object.__setattr__(self, "coordinates", _coordinates(self.coordinates, "Pointing.coordinates"))
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
        coordinates = _coordinates(coordinates, "coordinates")
        theta, phi, psi = self.theta, self.phi, self.psi
        if self.coordinates != coordinates:
            R = _rotation_matrix(self.coordinates, coordinates)
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
        coordinates = _coordinates(coordinates, "coordinates")
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
    not used (it may be NaN). By default each one is fitted by an unknown of
    its own; GLSParameters.flag_method selects other treatments.

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

    With MPI, maps, hit_map, gradient and hessian are on every rank: a copy
    per rank with full_maps "rank", or shared by the ranks of a node with
    "node" (one copy per node, read-only: copy them to change them).

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
    gap_fill_iterations : with a conditional :class:`GapFill`, the largest
        number of CG iterations of a gap fill (over the chunks or DetectorGroup
        chunks of every rank; with calibration, of the last pass), to compare
        with GapFill.maxiter. None otherwise.
    gradient : with GLSParameters.subpixel_order >= 1, (2, npix) the
        gradient of I in each pixel, in K per pixel side sqrt(4 pi / npix),
        along theta (south) and phi (east) at the center of the pixel;
        hp.UNSEEN where it is not solved (see GLS `offsets`) or I is not.
        Where it is, I is the mean of the sky over the pixel, to first order
        (to second order with the Hessian): compare with the input averaged
        over each pixel (e.g. hp.ud_grade from a finer map); hp.pixwin, the
        window averaged over the pixels, misses the shapes of the HEALPix
        pixels. None otherwise.
    hessian : with GLSParameters.subpixel_order = 2, (3, npix) the second
        derivatives of I (Ixx, Ixy, Iyy), in K per pixel side^2, along the
        axes of `gradient`; hp.UNSEEN where they are not solved (see GLS
        `subpixel_order`) or I is not. None otherwise.
    calibration : the CalibrationParameters of make_maps (gain_period_s
        gives the period of each gain), or None.
    template_definitions : the {name: Template} of make_maps (fixed ones
        included), or None. With `gains` and `templates` they give the TOD
        model fitted by make_maps, d = g (P m + D + F + sum_k a_k B_k) + n,
        whose residual :func:`estimate_noise` uses.
    """

    maps: np.ndarray
    hit_map: np.ndarray
    nside: int
    coordinates: Coordinates
    iterations: int
    converged: bool
    residuals: np.ndarray
    gains: list | None = None
    templates: dict | None = None
    template_covariance: dict | None = None
    calibration_iterations: int | None = None
    gap_fill_iterations: int | None = None
    gradient: np.ndarray | None = None
    hessian: np.ndarray | None = None
    calibration: "CalibrationParameters | None" = None
    template_definitions: dict | None = None


class _FlagTreatment:
    """What make_maps does with the flagged samples of a GLSParameters.flag_method
    (from _flag_treatment): their pixel labels for GLS (`_labels`: negative,
    a repeated label is one unknown; -1 for the treatments of GLS itself), the
    TOD fill (`_local_fill`: once, before the GLS setup; `_conditional`: after
    it, conditional on the valid noise; `_seed` of the random fills) and the
    flag arguments of GLS (`_gls_args`)."""

    _local_fill = False
    _conditional = False
    _seed = 0

    def _labels(self, flagged):
        """The labels of the flagged samples `flagged` (a mask) of a chunk."""
        return -1

    def _gls_args(self, fs):
        """The flag arguments of GLS for the sampling rates `fs` of the chunks."""
        raise NotImplementedError

    def _check(self, fit):
        """Raise ValueError if the treatment cannot be used with `fit` (a _Fit or None)."""


@dataclass(frozen=True)
class _Marginalize(_FlagTreatment):
    """flag_method "marginalize": one unknown per flagged sample."""

    def _labels(self, flagged):
        return -1 - np.arange(int(flagged.sum()), dtype=np.int32)

    def _gls_args(self, fs):
        return {"flag_method": "marginalize"}


@dataclass(frozen=True)
class _Offset(_Marginalize):
    """flag_method "offset": one unknown per run of flagged samples (repeated
    labels, for the "marginalize" method of GLS)."""

    def _labels(self, flagged):
        starts, stops = _flag_runs(flagged)
        return np.repeat(-1 - np.arange(starts.size, dtype=np.int32), stops - starts)


@dataclass(frozen=True, kw_only=True)
class ACTCuts(_FlagTreatment):
    """GLSParameters.flag_method: each run of flagged samples is fitted by its
    own polynomial coefficients (the compressed cut model of Naess et al. 2025,
    Sect. 3.3.3), and its TOD is set to 0 (no random filling). Exact GLS with
    these extra unknowns: no bias if the flags do not depend on the noise.

    ncoeff : None (default), positive integer, or callable. None keeps the
        paper's rule (from the gap duration, tuned for ACT, not for LiteBIRD),
        an integer fixes the number of coefficients per gap (1: offset, 3:
        quadratic), and a callable takes (gap lengths array in samples,
        sampling rate in Hz) and returns a positive integer scalar or an
        array of counts, one per gap. Counts are capped at the gap length
        (all of them: the same as "marginalize"); high orders use discrete
        orthonormal polynomials spanning the same space. Pure, deterministic
        callables preserve reproducibility and the calibration setup reused
        between iterations.
    """

    ncoeff: int | Callable | None = None

    def __post_init__(self):
        _validate_act_ncoeff(self.ncoeff)

    def _gls_args(self, fs):
        return {"flag_method": "act", "flag_sampling_rate_hz": fs, "act_ncoeff": self.ncoeff}


@dataclass(frozen=True, kw_only=True)
class GapFill(_FlagTreatment):
    """GLSParameters.flag_method: flagged samples have no unknowns; their TOD
    is filled once, held fixed during the map solve, and their pointing and
    template rows are excluded (Patanchon et al. 2008, Sect. 3.6). Approximate:
    it can bias the signal, especially on large scales.

    mode : "local_white" (default; SANEPIC's fill: a line fitted to the valid
        samples within 20 samples of each run, plus white noise with the rms
        of the fit residual), "conditional_mean" or "constrained" (the mean,
        or a random draw, of the noise in the gaps conditional on the valid
        noise residuals of the chunk, jointly across a DetectorGroup, with the
        GLS noise model; the padding is filled too). The conditional modes
        need make_maps' sky_model, subtracted before conditioning: their bias
        is linear in its error. Not a Gibbs sampler.
    seed : non-negative integer, default 0, of the random fills
        ("local_white", "constrained"), with independent streams per MPI rank.
        Reproducible for the same inputs, flags, order, chunking and MPI
        partition; a changed thread count can leave rounding differences.
        Calibration resets the stream each pass. A non-zero seed with
        "conditional_mean" (no random numbers) warns.
    tol, maxiter : squared relative residual threshold (default 1e-12) and
        maximum CG iterations (default 1000) of each conditional fill; failure
        raises RuntimeError. MapResult.gap_fill_iterations has the largest
        number used.
    preconditioner : "sine" (default: own-noise sine blocks for long gaps,
        diagonal scaling for short ones) or None (plain CG); the true residual
        controls convergence in either case.

    tol, maxiter and preconditioner are for the conditional modes: values
    other than the defaults with "local_white" raise ValueError.
    """

    mode: str = "local_white"
    seed: int = 0
    tol: float = 1e-12
    maxiter: int = 1000
    preconditioner: str | None = "sine"

    def __post_init__(self):
        if self.mode not in ("local_white", "conditional_mean", "constrained"):
            raise ValueError(f"GapFill.mode must be 'local_white', 'conditional_mean' or 'constrained', not {self.mode!r}")
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("GapFill.seed must be a non-negative integer")
        _validate_gap_solve(self.tol, self.maxiter)
        _validate_gap_preconditioner(self.preconditioner)
        default = {f.name: f.default for f in fields(self)}
        if self.mode == "local_white" and any(getattr(self, k) != default[k] for k in ("tol", "maxiter", "preconditioner")):
            raise ValueError("GapFill tol, maxiter and preconditioner are for mode='conditional_mean' or 'constrained'")
        if self.mode == "conditional_mean" and self.seed != default["seed"]:
            warnings.warn("GapFill.seed has no effect with mode='conditional_mean': no random numbers", stacklevel=3)

    @property
    def _local_fill(self):
        return self.mode == "local_white"

    @property
    def _conditional(self):
        return self.mode != "local_white"

    @property
    def _seed(self):
        return self.seed

    def _gls_args(self, fs):
        return {"flag_method": "fill"}

    def _check(self, fit):
        if self._conditional and (fit is None or fit.sky is None):
            raise ValueError("a conditional GapFill requires sky_model to condition noise residuals "
                             "(pass a zero model for noise-only TOD)")


def _flag_treatment(method, fit):
    """GLSParameters.flag_method as a _FlagTreatment, checked for `fit` (a _Fit or None)."""
    if isinstance(method, str) and method in ("marginalize", "offset", "act"):
        method = {"marginalize": _Marginalize, "offset": _Offset, "act": ACTCuts}[method]()
    if not isinstance(method, (ACTCuts, GapFill, _Marginalize)):
        hint = " (now GapFill(...))" if method == "sanepic" else ""
        raise ValueError(f"Unknown flag_method {method!r}{hint}: use 'marginalize', 'offset', 'act', ACTCuts(...) or GapFill(...)")  # noqa: TRY004
    method._check(fit)
    return method


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
        exact GLS without them; ~70 bytes per flagged sample), "offset" (one
        unknown per run, its samples set to 0), an :class:`ACTCuts` (local
        polynomial coefficients per run; "act" is ACTCuts()), or a
        :class:`GapFill` (no unknowns: the gaps are filled and their pointing
        rows excluded, as in SANEPIC; approximate, it can bias the signal).
        The options of each treatment are in its own class.
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
    sample_batch : with a sky model or sky templates given as alm, the number
        of DetectorData whose samples are interpolated together, one
        ducc0.sht.synthesis_general call per field (once with
        CalibrationParameters.cache_model, else at each calibration
        iteration). Each call has a fixed cost, a spherical harmonic
        transform at the lmax of the alm, comparable at lmax 1536 to
        interpolating one day at 37.5 Hz (G100: 0.04 s spin 0, 0.06 s spin 2,
        48 threads): 8 DetectorData of a day per call take ~3 times less per
        day, 32 ~3.5 times. Useful with short DetectorData (e.g. litebird_sim
        observations of a day); with DetectorData of a week or more the cost
        is amortized already. 1 (default): one call per DetectorData. Costs
        ~50 bytes per sample of the batch, and the batch is read from `data`
        before it is processed (a generator yields `sample_batch` items at
        a time). The samples are the same within the epsilon of the
        interpolation (1e-10, or 1e-6 for complex64 alm).
    subpixel_order : 0, 1 or 2, the sub-pixel terms of I (GLS `offsets`,
        `subpixel_order`). 0 (default): none, each pixel is a constant and the
        maps are means of their samples, which depend on where the samples
        fall in the pixels. 1: the gradient of I per pixel, two more unknowns:
        the error of the structure of the sky below the pixel size, in the
        maps and in the gains, drops to second order (~ the curvature of the
        sky times the pixel size^2). 2: also the Hessian, three more unknowns:
        the error drops to third order. With 1 or 2, I is the mean of the
        fitted model over each pixel, from the geometric moments of the
        HEALPix pixels (compare with the input averaged over each pixel).
        Needs a Pointing with angles. Costs 8 bytes per sample (the offsets
        in angle_dtype) and ~1.3-1.5x the PCG time with 1, ~1.15-1.3x more
        with 2 (no memory per sample); the white noise of I grows by <~1%.
        Polarization has no sub-pixel terms.
    full_maps : with MPI, where the full maps are: "rank" (default: a copy
        per rank, fastest) or "node" (one copy shared by the ranks of a
        node: less memory with many ranks per node at large nside, slower,
        and the output maps are read-only). See GLS `full_maps`.
    binned_weights : weights of the binned map of the sky model put back in
        the maps (make_maps' `sky_model`; unused without it): "noise"
        (default) weights each detector by its white-noise level 1 /
        sigma^2 (the high-frequency inverse PSD of DetectorData.noise, see
        GLS.white_weights), as the GLS weights its samples; "equal": every
        sample alike (pysanepic <= 0.6.4); or {detector name: weight > 0}
        for every DetectorData.name, e.g. one weight per pair of detectors
        in a horn, 2 / (sigma_a^2 + sigma_b^2) for both, so that their
        difference cancels the intensity in Q/U as in the pair.
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
    flag_method: str | ACTCuts | GapFill = "marginalize"
    split_gap_s: float = 300.0
    fft_dtype: type = np.float64
    coarse_nside: int | None = 0
    template_dtype: type = np.float32
    sample_batch: int = 1
    subpixel_order: int = 0
    binned_weights: str | dict = "noise"
    full_maps: str = "rank"

    def __post_init__(self):
        if self.full_maps not in ("node", "rank"):
            raise ValueError(f"GLSParameters.full_maps must be 'node' or 'rank', not {self.full_maps!r}")
        if self.subpixel_order not in (0, 1, 2):
            raise ValueError(f"GLSParameters.subpixel_order must be 0, 1 or 2, not {self.subpixel_order!r}")
        if not (isinstance(self.sample_batch, (int, np.integer)) and self.sample_batch >= 1):
            raise ValueError(f"GLSParameters.sample_batch must be an int >= 1, not {self.sample_batch!r}")
        nc = self.coarse_nside
        if nc is not None and (nc < 0 or nc & (nc - 1)):
            raise ValueError(f"coarse_nside must be None, 0 (the monopole) or a power of 2 <= nside, not {nc}")
        bw = self.binned_weights
        if isinstance(bw, dict):
            if not all(isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v) and v > 0 for v in bw.values()):
                raise ValueError("GLSParameters.binned_weights: the weights must be finite and > 0")
        elif bw not in ("noise", "equal"):
            raise ValueError(f"GLSParameters.binned_weights must be 'noise', 'equal' or {{detector name: weight}}, not {bw!r}")

    def _gls_settings(self, nside):
        """The solver settings among the arguments of GLS, for the maps at `nside`."""
        nc = self.coarse_nside
        if nc is not None and nc > nside:
            raise ValueError(f"coarse_nside must be None, 0 (the monopole) or a power of 2 <= nside, not {nc}")
        return {
            "preconditioner": self.preconditioner,
            "min_pol_rcond": self.min_pol_rcond,
            "angle_dtype": self.angle_dtype,
            "fft_dtype": self.fft_dtype,
            "template_dtype": self.template_dtype,
            "coarse": None if nc is None else _coarse_pixels(nside, nc) if nc else _monopole_pixel(nside),
            "subpixel_order": self.subpixel_order,
            "full_maps": self.full_maps,
        }


@dataclass(frozen=True, kw_only=True)
class Prior:
    """An independent Gaussian prior on fitted amplitudes, each one ~ N(mean,
    sigma^2): Template(prior=...) for the amplitudes of a template,
    CalibrationParameters(prior=...) for the gains. make_maps then gives the
    maximum of the posterior, and MapResult.template_covariance its
    covariance. With calibration the prior is on the values after each
    iteration (g, a), not on their linearized corrections. With
    Template.sum_zero the prior acts within the constraint: give means
    summing to zero.

    Attributes
    ----------
    sigma : the standard deviation, one for all the amplitudes, or {detector
        name: value} (DetectorData.name, every detector with amplitudes);
        inf: no prior. Amplitudes per period (Template.period_s, the gains)
        have each the value of their detector.
    mean : None (default): the nominal value, 0 for the amplitudes of a
        template and 1 for the gains; or one value, or {detector name: value}.
    """

    sigma: float | dict
    mean: float | dict | None = None

    def __post_init__(self):
        for attr, x in (("sigma", self.sigma), ("mean", self.mean)):
            values = np.array(list(x.values()) if isinstance(x, dict) else [] if x is None else [x], dtype=float)
            if attr == "sigma" and not np.all(values > 0):
                raise ValueError("Prior.sigma must be > 0 (inf: no prior)")
            if attr == "mean" and not np.all(np.isfinite(values)):
                raise ValueError("Prior.mean must be finite")

    def _terms(self, det, nominal, name):
        """(precision, mean) of the amplitudes of detector `det` (None: a
        common amplitude, or a DetectorData without a name)."""

        def value(x, field):
            if not isinstance(x, dict):
                return float(x)
            if det not in x:
                raise ValueError(f"{name}.{field}: no value for detector {det!r} (one value per detector needs "
                                 "DetectorData.name, and a common amplitude one value)")
            return float(x[det])

        return 1.0 / value(self.sigma, "sigma") ** 2, nominal if self.mean is None else value(self.mean, "mean")


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
        iterations (with GLSParameters.flag_method; "offset" uses one unknown per run;
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
    prior : :class:`Prior` on the gains g (mean None: 1), or None (default).
    tol : tolerance of the PCG in the first calibration iterations (as
        GLSParameters.tol, norm(r)^2 / norm(b)^2): None (default):
        GLSParameters.tol throughout; a number: that one; "adaptive":
        max(GLSParameters.tol, tol_factor * dg^2) with dg the largest
        correction ``max |dg|`` of the previous iteration (1e-6 in the first),
        tighter as the gains converge. Their maps and corrections only feed
        the next iteration, so a looser tolerance (e.g. 1e-6 for 1e-10)
        saves PCG iterations. It is used until an iteration has ``max |dg|`` <
        gain_tol, and never in the last of the n_iter: the iterations go on
        with GLSParameters.tol and stop (gain_tol) only on one of those,
        since a loose solve resolves dg only down to its tolerance
        (warm-started, it can stop at once with dg = 0). The final maps and
        gains are therefore those of a solve at GLSParameters.tol. It pays
        when the calibration needs many iterations; in a 30-day e2e test
        (noise, tol 1e-10, 3 iterations) it was slower (+6% adaptive, +24%
        with 1e-6): the less accurate early gains cost more iterations.
    tol_factor : with tol="adaptive", the factor of dg^2 (default 1e-3).
    """

    gain_period_s: float = 3600.0
    n_iter: int = 2
    gain_tol: float | None = None
    mask: np.ndarray | None = None
    cache_model: bool = False
    covariance_fixed_gains: bool = False
    prior: Prior | None = None
    tol: float | str | None = None
    tol_factor: float = 1e-3

    def __post_init__(self):
        if self.gain_tol is not None and not self.gain_tol > 0:
            raise ValueError("CalibrationParameters.gain_tol must be None or > 0")
        if isinstance(self.tol, str):
            if self.tol != "adaptive":
                raise ValueError(f"CalibrationParameters.tol must be None, a number > 0 or 'adaptive', not {self.tol!r}")
        elif self.tol is not None and not (np.isfinite(self.tol) and self.tol > 0):
            raise ValueError("CalibrationParameters.tol must be None, a number > 0 or 'adaptive'")
        if not (np.isfinite(self.tol_factor) and self.tol_factor > 0):
            raise ValueError("CalibrationParameters.tol_factor must be > 0")
        if self.prior is not None and not isinstance(self.prior, Prior):
            raise TypeError("CalibrationParameters.prior must be a pysanepic.Prior")
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
    fixed : a known signal, not fitted: the template is in the TOD model with
        amplitude 1 (times the gain with calibration) and is not put back in
        the maps, e.g. the beam asymmetries from the beams' alm
        (:func:`~pysanepic.blm_templates`). Not in MapResult.templates; no
        period_s, common, sum_zero, mask or prior.
    prior : :class:`Prior` on the amplitudes (mean None: 0), or None (default).
    """

    sky: np.ndarray | None = None
    period_s: float | None = None
    common: bool = False
    sum_zero: bool = False
    mask: np.ndarray | None = None
    fixed: bool = False
    prior: Prior | None = None

    def __post_init__(self):
        if self.fixed and (self.period_s is not None or self.common or self.sum_zero or self.mask is not None
                           or self.prior is not None):
            raise ValueError("Template: a fixed template has no amplitudes: no period_s, common, sum_zero, mask or prior")
        if self.prior is not None and not isinstance(self.prior, Prior):
            raise TypeError("Template.prior must be a pysanepic.Prior")
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


@numba.njit(parallel=True, cache=True)
def _offsets(loc, center, scale, gx, gy):
    """Offsets of the samples at loc (n, 2) (theta, phi) from the centers of
    their pixels `center` (n, 2), along e_theta and e_phi at the center,
    times `scale` (1 / pixel side): (v - v_c) . e, v the unit vectors."""
    for i in numba.prange(loc.shape[0]):
        st, ct = np.sin(loc[i, 0]), np.cos(loc[i, 0])
        sc, cc = np.sin(center[i, 0]), np.cos(center[i, 0])
        sp, cp = np.sin(loc[i, 1]), np.cos(loc[i, 1])
        spc, cpc = np.sin(center[i, 1]), np.cos(center[i, 1])
        dx, dy, dz = st * cp - sc * cpc, st * sp - sc * spc, ct - cc
        gx[i] = scale * (cc * cpc * dx + cc * spc * dy - sc * dz)
        gy[i] = scale * (-spc * dx + cpc * dy)


def _pixel_offsets(pix, loc, nside, dtype, nthreads):
    """(gx, gy) of the samples (see _offsets) in pixel sides, as `dtype`."""
    center = ducc0.healpix.Healpix_Base(nside, "RING").pix2ang(np.asarray(pix, dtype=np.int64), nthreads=nthreads)
    gx, gy = np.empty(len(pix)), np.empty(len(pix))
    with numba_threads(nthreads):
        _offsets(np.ascontiguousarray(loc, dtype=float), center, nside * np.sqrt(3.0 / np.pi), gx, gy)
    return gx.astype(dtype, copy=False), gy.astype(dtype, copy=False)


# The sub-pixel terms of I (GLSParameters.subpixel_order) in two forms, for the
# offsets (x, y) of the samples from the pixel center in pixel sides (_offsets):
# - GLS: the solution of GLS, I/Q/U then the components s multiplying
#   (x, y[, x^2 - 1/12, x y, y^2 - 1/12]) in the TOD (gls._project, and
#   _subpixel_tod here): I + s0 x + s1 y [+ s2 (x^2 - 1/12) + s3 x y + s4 (y^2 - 1/12)];
# - MapResult: maps with I the mean of this model over the pixel (geometric
#   moments of the HEALPix pixels), gradient = (s0, s1), hessian = (Ixx, Ixy, Iyy) = (2 s2, s3, 2 s4).
# _subpixel_result converts the first into the second (make_maps), _subpixel_gls back
# (the TOD model of a MapResult, estimate_noise). Inside the fit (e.g. the gain template)
# everything stays in the GLS form.
_HESSIAN = np.array([2.0, 1.0, 2.0])  # MapResult.hessian / GLS components


def _subpixel_tod(sub, pix, gx, gy):
    """The sub-pixel terms of the TOD from the GLS components `sub` (2 or 5,
    npix) and the offsets (gx, gy) of the samples, UNSEEN as 0: the same
    model as the projection of GLS."""
    g = sub[:, pix]
    g[g == hp.UNSEEN] = 0.0
    tod = g[0] * gx + g[1] * gy
    if len(sub) > 2:
        tod += g[2] * (gx * gx - _Q2) + g[3] * gx * gy + g[4] * (gy * gy - _Q2)
    return tod


@functools.lru_cache(maxsize=4)
def _pixel_moments(nside, k=8, nthreads=1):
    """Geometric moments of the pixels in the offsets of _offsets (pixel
    sides): (5, npix) mean of x, y, x^2, x y, y^2 over each pixel, from the
    centres of its k^2 children at nside * k (the second ones divided by 1 -
    1 / k^2, the moments of the children themselves). Computed in one
    quarter of each ring: a rotation by pi / 2 in phi keeps e_theta, e_phi."""
    npix = 12 * nside**2
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    ang = base.pix2ang(np.arange(npix), nthreads=nthreads)
    ang[:, 1] = np.fmod(ang[:, 1], np.pi / 2)
    reps, which = np.unique(base.ang2pix(ang, nthreads=nthreads), return_inverse=True)
    children = ducc0.healpix.Healpix_Base(nside * k, "NEST")
    out = np.empty((5, len(reps)))
    step = max(1, 2**22 // k**2)
    for a in range(0, len(reps), step):
        r = reps[a : a + step]
        loc = children.pix2ang((base.ring2nest(r)[:, None] * k**2 + np.arange(k**2)).ravel(), nthreads=nthreads)
        gx, gy = (g.reshape(len(r), k**2) for g in _pixel_offsets(np.repeat(r, k**2), loc, nside, float, nthreads))
        out[:, a : a + len(r)] = gx.mean(1), gy.mean(1), (gx * gx).mean(1), (gx * gy).mean(1), (gy * gy).mean(1)
    out[2:] /= 1 - 1 / k**2
    return out[:, which]


def _pixel_mean_shift(nside, gradient, hessian, nthreads):
    """I_mean - I_p per pixel (npix,): the geometric centring of the sub-pixel
    terms (MapResult.gradient, .hessian or None), Ix <x> + Iy <y> where the
    gradient is solved, + Ixx (<x^2> - 1/12) / 2 + Ixy <x y> + Iyy (<y^2> -
    1/12) / 2 where the Hessian is, the moments over the pixel (see
    _pixel_moments)."""
    mom = _pixel_moments(nside, nthreads=nthreads)
    shift = np.zeros(gradient.shape[1])
    solved = gradient[0] != hp.UNSEEN
    shift[solved] = (gradient[0] * mom[0] + gradient[1] * mom[1])[solved]
    if hessian is not None:
        solved = hessian[0] != hp.UNSEEN
        shift[solved] += (hessian[0] * (mom[2] - _Q2) / 2 + hessian[1] * mom[3] + hessian[2] * (mom[4] - _Q2) / 2)[solved]
    return shift


def _subpixel_result(maps, npol, order, nside, nthreads, write=True):
    """GLS form -> MapResult form, in place: (maps (npol, npix) with I the
    pixel mean, gradient or None, hessian or None), views of the GLS maps
    `maps` (npol + 0, 2 or 5 components, UNSEEN where not solved) for
    subpixel_order `order`. Only the ranks with `write` change `maps` (one
    per node when the ranks of a node share it)."""
    sub, maps = maps[npol:], maps[:npol]
    if not order:
        return maps, None, None
    gradient = sub[:2]
    hessian = sub[2:] if order > 1 else None
    if write:
        if hessian is not None:
            np.multiply(hessian, _HESSIAN[:, None], out=hessian, where=hessian != hp.UNSEEN)
        maps[0] += _pixel_mean_shift(nside, gradient, hessian, nthreads)
    return maps, gradient, hessian


def _subpixel_gls(maps, gradient, hessian, nside, nthreads):
    """MapResult form -> GLS form: (maps with I as in the TOD model of GLS
    (a copy), the GLS sub-pixel components) of MapResult.maps, .gradient (not
    None) and .hessian; the inverse of _subpixel_result."""
    maps = maps.copy()
    maps[0] -= _pixel_mean_shift(nside, gradient, hessian, nthreads)
    if hessian is None:
        return maps, gradient
    return maps, np.concatenate([gradient, np.where(hessian == hp.UNSEEN, hp.UNSEEN, hessian / _HESSIAN[:, None])])


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


@numba.njit(nogil=True, cache=True)
def _fill_flagged_runs(tod, flags, starts, stops, white, nfit, out):
    """Two-pass centered line fits, without Python allocations per glitch."""
    pos = 0
    for start, stop in zip(starts, stops):
        # The windows can contain other gaps: read only the original good TOD.
        windows = ((max(0, start - nfit), start), (stop, min(tod.size, stop + nfit)))
        n, sx, sy = 0, 0.0, 0.0
        for lo, hi in windows:
            for i in range(lo, hi):
                if not flags[i]:
                    n += 1
                    sx += i - start
                    sy += tod[i]
        mx, my = (sx / n, sy / n) if n else (0.0, 0.0)
        xx, xy = 0.0, 0.0
        for lo, hi in windows:
            for i in range(lo, hi):
                if not flags[i]:
                    x = i - start - mx
                    xx += x * x
                    xy += x * (tod[i] - my)
        slope = xy / xx if xx > 0 else 0.0
        rss = 0.0
        for lo, hi in windows:
            for i in range(lo, hi):
                if not flags[i]:
                    residual = tod[i] - my - slope * (i - start - mx)
                    rss += residual * residual
        sigma = np.sqrt(rss / (n - 2)) if n > 2 else 0.0
        for i in range(start, stop):
            out[i] = my + slope * (i - start - mx) + sigma * white[pos]
            pos += 1


def _fill_flagged(tod, flags, rng, nfit=20):
    """SANEPIC gap filling (Patanchon et al. 2008, Sect. 2.2).

    Fit a line to valid samples in the nfit-sample windows on each side of
    each run, add white noise with rms estimated from the fit residual, and
    leave good data untouched. This is approximate, not a constrained noise
    realization. At an edge use the available side; with one neighbor use a
    constant, with none use zero. Never read flagged TOD, even at other gaps.
    """
    out = np.array(tod, dtype=float, copy=True)
    starts, stops = _flag_runs(flags)
    white = rng.standard_normal(int(np.sum(stops - starts)))
    _fill_flagged_runs(tod, flags, starts, stops, white, nfit, out)
    return out


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
    coordinates : Coordinates or str
        Coordinate system of the output maps: a :class:`Coordinates` or "G",
        "E" or "C" (MapResult.coordinates is a Coordinates).
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
    coordinates = _coordinates(coordinates, "make_maps: coordinates")
    sky = None if sky_model is None else _model(sky_model, nside, "sky_model")
    tmpls = dict(templates or {})
    if any(not isinstance(t, Template) for t in tmpls.values()):
        raise TypeError("templates must be {name: pysanepic.Template}")
    skies = {k: _model(t.sky, nside, f"templates[{k!r}].sky") for k, t in tmpls.items() if t.sky is not None}
    for k, x in skies.items():
        if not np.iscomplexobj(x) and not tmpls[k].sum_zero and not tmpls[k].fixed:
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
    bw = None if params is None else params.binned_weights
    if sky is not None and isinstance(bw, dict):  # checked on every rank before the collective GLS
        missing = sorted({str(d.name) for d in dets if d.name not in bw})
        if comm is not None:
            missing = sorted(set().union(*comm.allgather(missing)))
        if missing:
            raise ValueError(f"GLSParameters.binned_weights has no weight for the detectors {missing} (DetectorData.name)")
    amplitudes = _Amplitudes.build(tmpls, calibration, dets, comm)
    fitted = {k: t for k, t in tmpls.items() if not t.fixed}
    mask_gain = _mask(None if calibration is None else calibration.mask, nside, "CalibrationParameters.mask")
    # a template without a mask has that of the gains; equal masks are one array (and one test per sample)
    masks = {k: mask_gain if t.mask is None else _mask(t.mask, nside, f"templates[{k!r}].mask")
             for k, t in fitted.items()}
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
    fit = _Fit(calibration=calibration, sky=sky, templates=tmpls, skies=skies, amplitudes=amplitudes,
               mask_gain=mask_gain, masks=masks,
               cache={} if calibration is not None and calibration.cache_model else None)
    n_cal = 0
    n_iter = 1 if calibration is None else calibration.n_iter
    loose = calibration is not None and calibration.tol is not None
    final_tol = (GLSParameters() if params is None else params).tol
    dg = None  # max |dg| of the previous iteration
    for _ in range(n_iter):
        # CalibrationParameters.tol until the corrections are small, and never in the last iteration:
        # a loose solve can stop with dg ~ 0 (a warm start within its tolerance), not a converged calibration
        fit.tol = None
        if loose and n_cal < n_iter - 1:
            tol = calibration.tol
            if tol == "adaptive":
                tol = 1e-6 if dg is None else calibration.tol_factor * dg**2
            fit.tol = tol if tol > final_tol else None
        res = _make_maps(data, nside, coordinates, params, comm, nthreads, verbose, fit)[0]
        n_cal += 1
        if calibration is not None:
            dg = amplitudes.dg_max if comm is None else max(comm.allgather(amplitudes.dg_max))
            if verbose and (comm is None or comm.rank == 0):
                tol = "" if fit.tol is None else f" (PCG tol {fit.tol:g})"
                print(f"calibration iteration {n_cal}{tol}: max |dg| = {dg:.3e}")
            if calibration.gain_tol is not None and dg < calibration.gain_tol:
                if fit.tol is None:
                    break
                loose = False  # converged at the loose tolerance: the next iterations at GLSParameters.tol
            if n_cal == 1 and calibration.n_iter > 1:
                _mask_unsolved_pol(fit, res, verbose and (comm is None or comm.rank == 0))
    cov = None
    if template_covariance and amplitudes.index:
        p = GLSParameters() if params is None else params
        if calibration is not None and calibration.covariance_fixed_gains and fit.no_gains is not None:
            fit.gls.set_templates(*fit.no_gains)  # collective
        cov = {"labels": amplitudes.labels, "matrix": fit.gls.shared_covariance(tol=p.covariance_tol, maxiter=p.maxiter, verbose=verbose)}
    if fit.mask_gain is not None or any(m is not None for m in fit.masks.values()):  # samples may have been left out
        fit.fit, fit.gls = False, None  # other flags: a new GLS
        res = _make_maps(data, nside, coordinates, params, comm, nthreads, verbose, fit)[0]
    res.templates = amplitudes.result_templates()
    res.template_covariance = cov
    res.calibration_iterations = None if calibration is None else n_cal
    res.calibration = calibration
    res.template_definitions = tmpls or None
    return res


class _Amplitudes:
    """The fitted amplitudes of make_maps:

    - the gains (calibration), one per DetectorData and gain period;
    - the shared amplitudes of the templates with period_s None, one per
      template and detector name (or one per template with common), the same
      on every MPI process; `labels` [(name, detector name or None)];
    - the local amplitudes of the templates with period_s, one per
      DetectorData and period.

    The DetectorData are those of make_maps, in the same order (groups
    expanded): `j` is the position of one among them. The current values are
    read by _TODModel; in each pass of the GLS, the labels of the local
    amplitudes (gains first) are assigned DetectorData by DetectorData
    (start_pass, add), with the priors linearized around the current values,
    and a solve updates them (update)."""

    def __init__(self, templates, calibration, labels, shared=None, prior=None, groups=(), local=None, gains=None):
        self.templates = templates  # name -> Template
        self.calibration = calibration
        self.labels = labels
        self.index = {x: i for i, x in enumerate(labels)}
        self.shared = np.zeros(len(labels)) if shared is None else shared
        self.prior = prior  # (precision, mean) of the shared amplitudes (2, n), or None
        self.groups = list(groups)  # shared amplitudes summing to zero
        # name -> amplitudes per period, one array per DetectorData (None: zeros)
        self.local = {k: None for k, t in templates.items() if t.period_s is not None} if local is None else local
        self.gains = gains  # one array per DetectorData (None: ones)
        self.dg_max = 0.0  # largest |dg| of the last update on this rank
        self._pass = None

    @classmethod
    def build(cls, templates, calibration, dets, comm):
        """The amplitudes of make_maps for `templates` and `calibration`, all 0
        (gains 1), with the shared labels of the detectors `dets` of all the
        MPI processes; the priors are checked."""
        fitted = {k: t for k, t in templates.items() if not t.fixed}
        per_det = [k for k, t in fitted.items() if t.period_s is None and not t.common]
        if per_det and any(d.name is None for d in dets):
            raise ValueError(f"templates {per_det} have one amplitude per detector: they need DetectorData.name")

        def has(k, d):
            return templates[k].sky is not None or k in (d.templates or {})

        names = {k: sorted({d.name for d in dets if has(k, d)}) for k in per_det}
        if comm is not None:
            names = {k: sorted(set().union(*comm.allgather(v))) for k, v in names.items()}
        labels = [(k, n) for k, t in fitted.items() if t.period_s is None for n in names.get(k, [None])]
        index = {x: i for i, x in enumerate(labels)}
        # the priors: (precision, mean) of the shared amplitudes; those of the local ones checked here
        prior = None
        if any(fitted[k].prior is not None for k, _ in labels):
            prior = np.array([(0.0, 0.0) if fitted[k].prior is None else fitted[k].prior._terms(n, 0.0, f"templates[{k!r}].prior")
                              for k, n in labels]).T
        for k, t in fitted.items():
            if t.period_s is not None and t.prior is not None:
                for d in dets:
                    if has(k, d):
                        t.prior._terms(d.name, 0.0, f"templates[{k!r}].prior")
        if calibration is not None and calibration.prior is not None:
            for d in dets:
                calibration.prior._terms(d.name, 1.0, "CalibrationParameters.prior")
        groups = [[index[k, n] for n in names[k]] for k in per_det if templates[k].sum_zero]
        return cls(templates, calibration, labels, prior=prior, groups=groups)

    @classmethod
    def from_result(cls, result):
        """The final amplitudes of a MapResult of make_maps."""
        templates = dict(result.template_definitions or {})
        amps = result.templates or {}
        labels, shared = [], []
        for k, t in templates.items():
            if t.fixed or t.period_s is not None or k not in amps:
                continue
            a = amps[k] if t.common else None
            for n, v in ([(None, a)] if t.common else amps[k].items()):
                labels.append((k, n))
                shared.append(v)
        local = {k: amps.get(k) for k, t in templates.items() if t.period_s is not None}
        gains = result.gains if result.calibration is not None else None
        return cls(templates, result.calibration, labels, np.array(shared, dtype=float), local=local, gains=gains)

    @property
    def count(self):
        """The number of DetectorData of the gains or amplitudes per period, or None."""
        if self.calibration is not None and self.gains is not None:
            return len(self.gains)
        return next((len(v) for v in self.local.values() if v is not None), None)

    # the current values

    def gain(self, d, j):
        """(gains, period of each sample) of DetectorData d at j, or None without calibration."""
        if self.calibration is None:
            return None
        p = _periods(len(d.tod), d.sampling_rate_hz, self.calibration.gain_period_s)
        return (np.ones(p[-1] + 1) if self.gains is None else self.gains[j]), p

    def shared_label(self, k, d):
        """The label of the shared amplitude of template k for d, or None."""
        return self.index.get((k, None if self.templates[k].common else d.name))

    def shared_value(self, k, d):
        """The shared amplitude of template k for d, or None."""
        i = self.shared_label(k, d)
        return None if i is None else float(self.shared[i])

    def local_value(self, k, d, j):
        """(amplitudes, period of each sample) of template k for d at j (empty: no amplitudes)."""
        p = _periods(len(d.tod), d.sampling_rate_hz, self.templates[k].period_s)
        amps = self.local[k]
        return (np.zeros(p[-1] + 1) if amps is None else amps[j]), p

    # a pass of the GLS

    def start_pass(self, fit):
        """A new pass: labels from 0; with `fit` False the amplitudes are fixed (no priors, no update)."""
        cal = self.calibration
        priors = fit and ((cal is not None and cal.prior is not None)
                          or any(self.templates[k].prior is not None for k in self.local))
        # gains and (current amplitudes, first label) of each DetectorData; (precision, mean) of the
        # local amplitudes in the order of their labels, with a prior on some of them
        self._pass = SimpleNamespace(fit=fit, nlab=0, gains=[], offsets=[], local={k: [] for k in self.local},
                                     priors=[] if priors else None)

    def add(self, d, gain, local):
        """The labels of the local amplitudes of DetectorData d in this pass,
        in the order of `data`: (labels of the gains or None, {name: labels});
        `gain` from gain(), `local` {name: local_value()} of the templates of d."""
        ps, cal = self._pass, self.calibration
        gain_labels, labels = None, {}
        if gain is not None:
            ps.offsets.append(ps.nlab)
            ps.nlab += gain[0].size
            ps.gains.append(gain[0])
            gain_labels = ps.offsets[-1] + gain[1]
            if ps.priors is not None:
                prec, g0 = (0.0, 1.0) if cal.prior is None else cal.prior._terms(d.name, 1.0, "CalibrationParameters.prior")
                # g (1 + dg) ~ N(g0, 1 / prec): dg ~ N((g0 - g) / g, 1 / (prec g^2))
                ps.priors.append((prec * gain[0]**2, (g0 - gain[0]) / gain[0]))
        for k, amps in ps.local.items():
            if k not in local:
                amps.append((np.zeros(0), ps.nlab))
                continue
            amp, p = local[k]
            amps.append((amp, ps.nlab))
            if ps.priors is not None:
                tp = self.templates[k].prior
                prec, a0 = (0.0, 0.0) if tp is None else tp._terms(d.name, 0.0, f"templates[{k!r}].prior")
                ps.priors.append((np.full(amp.size, prec), a0 - amp))  # on a + da
            labels[k] = ps.nlab + p
            ps.nlab += amp.size
        return gain_labels, labels

    def gls_priors(self):
        """(prior of the local amplitudes, prior of the shared ones) of this
        pass for GLS: (precision, mean of the correction), or None."""
        ps = self._pass
        local = None if not ps.priors else tuple(np.concatenate(x) for x in zip(*ps.priors))
        shared = None if not ps.fit or self.prior is None else (self.prior[0], self.prior[1] - self.shared)
        return local, shared

    def update(self, info):
        """After the solve of this pass: g (1 + dg), a + da and A + dA from
        info["amplitudes"] and info["shared_amplitudes"] (or None: fixed
        amplitudes, the gains of the pass are kept)."""
        ps = self._pass
        gains = ps.gains
        if info is not None:
            if self.calibration is not None:
                dg = [info["amplitudes"][o : o + g.size] for g, o in zip(gains, ps.offsets)]
                self.dg_max = max((float(np.abs(x).max()) for x in dg if x.size), default=0.0)
                gains = [g * (1 + x) for g, x in zip(gains, dg)]
            for k, amps in ps.local.items():
                self.local[k] = [a + info["amplitudes"][o : o + a.size] for a, o in amps]
            if self.index:
                self.shared = self.shared + info["shared_amplitudes"]
        self.gains = gains if self.calibration is not None else None

    def result_templates(self):
        """MapResult.templates: name -> one array per DetectorData (period_s),
        a number (common) or {detector name: amplitude}; None without templates."""
        out = {}
        for k, t in self.templates.items():
            if t.fixed:
                continue
            if t.period_s is not None:
                out[k] = self.local[k]
            elif t.common:
                out[k] = float(self.shared[self.index[k, None]])
            else:
                out[k] = {n: float(self.shared[i]) for (kk, n), i in self.index.items() if kk == k}
        return out or None


@dataclass
class _Fit:
    """What _make_maps fits besides the maps, and the current estimates."""

    calibration: "CalibrationParameters | None"  # None: no gains (and no dipole)
    sky: np.ndarray | None  # sky model (maps or alm), subtracted and put back
    templates: dict  # name -> Template
    skies: dict  # name -> map or alm of the templates given as a sky signal
    amplitudes: "_Amplitudes"  # the gains and the amplitudes of the templates: current values, labels, priors
    mask_gain: np.ndarray | None
    masks: dict  # name -> mask of the template, or None
    residual: np.ndarray | None = None  # maps minus sky model of the previous iteration
    x0: np.ndarray | None = None  # the GLS solution of the previous iteration: the start of the next one
    gls: "GLS | None" = None  # the GLS of the fit iterations, reused (same pixels and flags)
    cache: dict | None = None  # with cache_model: DetectorData index -> (sky model TOD, sky template TODs)
    no_gains: tuple | None = None  # the templates and priors of the GLS but the gain templates (covariance_fixed_gains)
    fit: bool = True  # False: final maps with gains and amplitudes fixed
    tol: float | None = None  # PCG tolerance of the next fit iteration (CalibrationParameters.tol), None: GLSParameters.tol

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
    return _combine(_synthesis(x, loc, nthreads), d, angle, nthreads)


def _synthesis(x, loc, nthreads):
    """(I, Q, U) of the alm x (Q, U None for T only) at loc (n, 2)."""
    lmax = (int(np.sqrt(8 * x.shape[1] + 1)) - 3) // 2
    eps = 1e-6 if x.dtype == np.complex64 else 1e-10
    i = ducc0.sht.synthesis_general(alm=x[:1], spin=0, lmax=lmax, loc=loc, epsilon=eps, nthreads=nthreads)[0]
    if x.shape[0] < 3:
        return i, None, None
    q, u = ducc0.sht.synthesis_general(alm=x[1:], spin=2, lmax=lmax, loc=loc, epsilon=eps, nthreads=nthreads)
    return i, q, u


def _combine(iqu, d, angle, nthreads):
    """I + gamma (Q cos 2a + U sin 2a) (a new array)."""
    i, q, u = iqu
    tod = np.array(i, dtype=float)
    if q is not None:
        with numba_threads(nthreads):  # one parallel pass, cos and sin from a table (numpy: ~10 serial passes)
            _add_pol(tod, q, u, np.asarray(angle, dtype=float), float(d.pol_efficiency), _TRIG_COS, _TRIG_SIN)
    return tod


def _sample_batch(fields, dets, nside, coordinates, nthreads):
    """The TODs of the sky model and templates `fields` ({key: maps, alm or
    None}) for several DetectorData, with one synthesis_general call per alm
    field for all of them; returns, per DetectorData, (pix, angle, {key: TOD}, loc)."""
    geo = [_pixels_and_angles(d, nside, coordinates, nthreads, loc=True) for d in dets]
    loc = np.concatenate([g[2] for g in geo]) if len(geo) > 1 else geo[0][2]
    bounds = np.cumsum([0] + [len(g[0]) for g in geo])
    out = [(pix, angle, {}, xy) for pix, angle, xy in geo]
    for key, x in fields.items():
        if x is None or not np.iscomplexobj(x):
            for d, (pix, angle, tods, _) in zip(dets, out):
                tods[key] = _sample(x, d, pix, angle, None, nthreads)
            continue
        i, q, u = _synthesis(x, loc, nthreads)
        for j, (d, (_, angle, tods, _)) in enumerate(zip(dets, out)):
            sl = slice(bounds[j], bounds[j + 1])
            tods[key] = _combine((i[sl], None if q is None else q[sl], None if u is None else u[sl]), d, angle, nthreads)
    return out


def _conditional_tod(tod, gls, npix, common_modes, fill, rng, nthreads):
    """Fill noise residuals with the same precision as GLS, one group at a time;
    returns the largest number of CG iterations on this rank.

    Padding has no observations and must be sampled, not conditioned upon.
    Unpadded chunks retain compact synthetic values instead of full TOD copies.
    """
    grouped = {k for ks, _, _ in common_modes for k in ks}
    groups = list(common_modes) + [([k], None, None) for k in range(len(tod)) if k not in grouped]
    iterations = 0
    for ks, coupling, mode_psd in groups:
        masks = [(gls.ip[k] < 0) | (gls.ip[k] >= npix) for k in ks]
        if not any(mask.any() for mask in masks):
            continue
        filled, info = conditional_gap_fill(
            np.stack([np.asarray(tod[k], dtype=float) for k in ks]), np.stack(masks),
            np.stack([gls.weights[k] for k in ks]), fft_length=gls._nfft[ks[0]],
            coupling=coupling, mode_psd=mode_psd, draw=fill.mode == "constrained",
            rng=rng, tol=fill.tol, maxiter=fill.maxiter, nthreads=nthreads, preconditioner=fill.preconditioner,
        )
        iterations = max(iterations, info["iterations"])
        for i, (k, mask) in enumerate(zip(ks, masks)):
            if np.any(gls.ip[k] >= npix):  # padded data already occupies a full copy
                tod[k] = filled[i].copy()
            elif mask.any():
                tod[k] = _GapFilledTOD(tod[k], mask, filled[i, mask])
        del filled
    return iterations


@dataclass
class _DetectorTOD:
    """The TOD of one DetectorData for the GLS, from _ChunkBuilder._detector_tod."""

    tod: np.ndarray  # calibrated, minus the known part of the model
    flags: np.ndarray | None  # with the samples out of every fit
    sky: np.ndarray | None  # TOD of the sky model (put back in the maps)
    gain: np.ndarray | None = None  # template of the gains (T dg)
    gain_labels: np.ndarray | None = None
    templates: dict = field(default_factory=dict)  # name -> TOD of the fitted templates
    labels: dict = field(default_factory=dict)  # name -> labels of the templates with amplitudes per period


class _ChunkBuilder:
    """The chunks of the GLS of _make_maps, built one DetectorData at a time
    (add): their TOD, templates and sky model TOD at every pass, and their
    pixels, angles, noise weights and sub-pixel offsets the first time only
    (in the calibration iterations after the first, `reuse`, the GLS keeps
    them). The DetectorData are indexed by their position `j` among those
    given to make_maps (groups expanded), which picks their gains, amplitudes
    per period, cached sky model TODs and batched samples."""

    def __init__(self, nside, coordinates, params, comm, nthreads, fit):
        self.nside, self.coordinates, self.params, self.nthreads, self.fit = nside, coordinates, params, nthreads, fit
        self.reuse = fit is not None and fit.fit and fit.gls is not None
        self.method = _flag_treatment(params.flag_method, fit)
        self.npix = 12 * nside * nside
        self.rank, self.size = (0, 1) if comm is None else (comm.rank, comm.size)
        self.flag_rng = np.random.default_rng(np.random.SeedSequence([self.method._seed, self.rank]))
        self.dtype = params.angle_dtype
        self.npol = 3 if params.pol else 1
        self.model = None if fit is None else _TODModel.from_fit(fit, nthreads)  # with the current estimates
        self.cal = None if fit is None else fit.calibration
        # every pass, one entry per chunk (TOD, sky model TOD, detector name) or per template
        self.tod, self.model_tod, self.names = [], [], []
        self.templates, self.shared = [], []  # local and shared templates (chunk, values, labels)
        self.gain_ids = set()  # ids of the gain templates in `templates`
        if fit is not None:  # labels of the local amplitudes from 0
            fit.amplitudes.start_pass(fit.fit)
        self.common_modes = []
        # first pass only, one entry per chunk
        self.pix, self.a2, self.gamma, self.w, self.nfft, self.fs = [], [], [], [], [], []
        self.gx, self.gy = [], []  # offsets from the pixel centers (subpixel_order >= 1)
        self.weights = {}  # chunks with the same length and noise share one array
        self.n_virtual = 0  # padding margins on this rank; labels are unique across ranks
        self.presampled = {}  # j -> (pix, angle, {None: sky model TOD, key: sky template TOD}, loc)
        self.timings = _Timings()  # seconds of the phases, for verbose

    def items(self, data):
        """(j, item) of `data` (a DetectorData or a DetectorGroup, j of its
        first DetectorData); sample_batch DetectorData at a time when the alm
        models are sampled in this pass (their TODs then in `presampled`)."""
        fit, params = self.fit, self.params
        batching = fit is not None and fit.alm and params.sample_batch > 1 and not (fit.cache is not None and fit.cache)
        it, j = iter(data), 0
        while True:
            items, dets = [], []
            for d in it:
                items.append(d)
                dets += d.detectors if isinstance(d, DetectorGroup) else [d]
                if not batching or len(dets) >= params.sample_batch:
                    break
            if not items:
                return
            if batching:
                fields = {None: fit.sky, **fit.skies}
                with self.timings.phase("pointing + sky sampling (batches)"):
                    for i, r in enumerate(_sample_batch(fields, dets, self.nside, self.coordinates, self.nthreads)):
                        self.presampled[j + i] = r
            for d in items:
                yield j, d
                j += len(d.detectors) if isinstance(d, DetectorGroup) else 1

    def add_item(self, j, d):
        """The chunks of a DetectorData or a DetectorGroup with its first DetectorData at j."""
        if not isinstance(d, DetectorGroup):
            self.add(d, j)
            return
        # a group: chunks at the same times, not split at gaps (that would break
        # the alignment); one common_modes entry per chunk of time
        idx = [self.add(x, j + i, split=False) for i, x in enumerate(d.detectors)]
        if self.reuse and not self.method._conditional:
            return
        fs = d.detectors[0].sampling_rate_hz
        U = np.array([mode.coupling for mode in d.modes], dtype=float).T  # (ndet, r)
        for chunks in zip(*idx):
            nfft = self.fit.gls._nfft[chunks[0]] if self.reuse else self.nfft[chunks[0]]
            lam = np.array([1 / np.maximum(w, w.max() * 1e-12) for w in (mode.noise.weights(nfft, fs) for mode in d.modes)])
            self.common_modes.append((list(chunks), U, lam))

    def add(self, d, j, split=True):
        """The chunks of DetectorData d, at position j; returns their indices."""
        fit, params = self.fit, self.params
        sample = fit is not None and (fit.cache is None or j not in fit.cache)
        batch = self.presampled.pop(j, None)  # from _sample_batch (GLSParameters.sample_batch > 1)
        if batch is not None:
            pix, angle, sampled, loc = batch
        else:
            # the (theta, phi) only for alm models or offsets, not to keep them otherwise
            sampled = None
            with self.timings.phase("pointing"):
                pix, angle, loc = _pixels_and_angles(d, self.nside, self.coordinates, self.nthreads,
                                                     loc=(sample and fit.alm) or params.subpixel_order > 0)
        offsets = None
        if params.subpixel_order:
            if loc is None:
                raise ValueError("GLSParameters.subpixel_order needs a Pointing with angles (theta, phi), not pixels")
            with self.timings.phase("pointing"):
                offsets = _pixel_offsets(pix, loc, self.nside, self.dtype, self.nthreads)
        with self.timings.phase("TOD model"):
            det = (_DetectorTOD(d.tod, d.flags, None) if fit is None
                   else self._detector_tod(d, j, pix, angle, loc, offsets, sample, sampled))
        gap = params.split_gap_s * d.sampling_rate_hz if split else np.inf
        chunks = _chunk_bounds(len(d.tod), max(1, int(round(params.chunk_s * d.sampling_rate_hz))))
        pieces = [piece for c in chunks for piece in _split_at_gaps(*c, det.flags, gap)]
        first = len(self.tod)
        with self.timings.phase("chunks"):
            self._add_tod(d, det, pieces)
            if not self.reuse:  # the GLS has pixels, angles and weights already
                self._add_layout(d, det.flags, pix, angle, offsets, pieces)
        return list(range(first, len(self.tod)))

    def _detector_tod(self, d, j, pix, angle, loc, offsets, sample, sampled):
        """d = g (P m + D + S + F + sum_k a_k B_k) + n (F: the fixed templates), linearized around the
        current g, a: d / g - D - S - F - sum a B = P m_res + T dg + sum B da + n,
        T = D + S + F + sum a B + P m_res_prev."""
        fit, model, cal, nthreads = self.fit, self.model, self.cal, self.nthreads
        if not sample:
            sky_tod, cached = fit.cache[j]
            ttod = {k: t.copy() for k, t in cached.items()}  # zeroed below in the masks
        else:
            if sampled is not None:
                sky_tod = sampled.pop(None)
                ttod = sampled
            else:
                with self.timings.phase("TOD model: of which sky sampling"):
                    sky_tod, ttod = model.sample(d, pix, angle, loc)
            if fit.cache is not None:
                fit.cache[j] = sky_tod, {k: t.copy() for k, t in ttod.items()}
        ttod = model.tods(d, ttod)
        known, amps_d = model.known(d, j, sky_tod, ttod)
        gain = model.gain(d, j)
        det = _DetectorTOD(None, d.flags, sky_tod, templates=ttod)
        det.gain_labels, det.labels = fit.amplitudes.add(d, gain, amps_d)
        if cal is not None and fit.fit:
            det.gain = d.dipole + known
            if fit.residual is not None:
                with numba_threads(nthreads):
                    _axpy(det.gain, 1.0, _project_maps(fit.residual[:self.npol], pix, angle, d.pol_efficiency, nthreads))
                    if offsets is not None:  # fit.residual is in the GLS form, as _subpixel_tod
                        _axpy(det.gain, 1.0, _subpixel_tod(fit.residual[self.npol:], pix, *offsets))
        det.tod = model.calibrated(d, known, gain)
        # masks: samples out of every active fit are flagged; out of some, their templates are 0
        active = [(det.gain, fit.mask_gain)] if det.gain is not None else []
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
                det.flags = drop if det.flags is None else det.flags | drop
        return det

    def _add_tod(self, d, det, pieces):
        """The TOD, templates and sky model TOD of the chunks `pieces` of d (every pass)."""
        fit, params = self.fit, self.params
        m = int(round(params.pad_s * d.sampling_rate_hz))
        dt = params.template_dtype
        for a, b in pieces:
            k = len(self.tod)
            tod = det.tod[a:b]  # a view: the TOD is not copied
            if fit is not None:
                # values as template_dtype and int32 labels (0 in the margins, where
                # the values are 0), one int for an amplitude in the whole chunk
                if det.gain is not None:
                    self.templates.append((k, _padded(det.gain, a, b, m, dt), _padded(det.gain_labels, a, b, m, np.int32)))
                    self.gain_ids.add(id(self.templates[-1]))
                if fit.fit:
                    cols = []  # the templates with one amplitude in the chunk: one 2D array, one pass in the PCG
                    for name, t in det.templates.items():
                        if name in det.labels:
                            self.templates.append((k, _padded(t, a, b, m, dt), _padded(det.labels[name], a, b, m, np.int32)))
                        else:
                            cols.append((t, fit.amplitudes.shared_label(name, d)))
                    if len(cols) == 1:
                        self.shared.append((k, _padded(cols[0][0], a, b, m, dt), cols[0][1]))
                    elif cols:
                        values = np.zeros((b - a + 2 * m, len(cols)), dt)
                        with numba_threads(self.nthreads):
                            _fill_columns(values, tuple(np.asarray(t, dtype=float) for t, _ in cols), a, m)
                        self.shared.append((k, values, np.array([i for _, i in cols])))
                if fit.sky is not None:  # the templates are not put back
                    self.model_tod.append(_padded(det.sky, a, b, m, np.float64))
            flagged = None if det.flags is None else det.flags[a:b]
            if flagged is not None and flagged.any():
                if self.method._local_fill:
                    filled = _fill_flagged(tod, flagged, self.flag_rng)
                    tod = filled if m else _GapFilledTOD(tod, flagged, filled[flagged])
                    del filled
                elif m:  # the padding must not use flagged TOD
                    tod = np.where(flagged, tod[~flagged].mean() if not flagged.all() else 0.0, tod)
            self.tod.append(tod if m == 0 else _pad(tod, m, params.pad_fill))
            self.names.append(d.name)  # for binned_weights by name

    def _add_layout(self, d, flags, pix, angle, offsets, pieces):
        """The pixels, angles, noise weights and offsets of the chunks `pieces` of d (first pass)."""
        params, dtype = self.params, self.dtype
        fs = d.sampling_rate_hz
        m = int(round(params.pad_s * fs))
        for a, b in pieces:
            n = b - a + 2 * m
            nfft = fast_fft_length(n)  # e.g. chunks split at gaps have any length
            key = (nfft, fs, d.noise)  # the dict keeps the noise objects alive: no reused ids
            if key not in self.weights:
                self.weights[key] = d.noise.weights(nfft, fs)
            self.w.append(self.weights[key])
            self.nfft.append(nfft)
            self.fs.append(fs)
            # 2a computed in float64, then stored as `dtype`; the margins of a
            # padded chunk carry no polarization (gamma = 0: intensity-only
            # virtual pixels), so only padded chunks need gamma per sample
            if params.pol:
                a2 = np.zeros(n, dtype)
                a2[m : n - m] = two_angle(angle[a:b])
                self.a2.append(a2)
                if m == 0:
                    self.gamma.append(d.pol_efficiency)
                else:
                    g = np.zeros(n, dtype)
                    g[m : n - m] = d.pol_efficiency
                    self.gamma.append(g)
            if offsets is not None:
                self.gx.append(_padded(offsets[0], a, b, m, dtype))
                self.gy.append(_padded(offsets[1], a, b, m, dtype))
            p = pix[a:b].astype(np.int32)
            flagged = None if flags is None else flags[a:b]
            if flagged is not None and flagged.any():
                p[flagged] = self.method._labels(flagged)
            if m == 0:
                self.pix.append(p)
            else:
                left = self.npix + 2 * (self.n_virtual * self.size + self.rank)
                self.n_virtual += 1
                self.pix.append(np.concatenate([np.full(m, left, np.int32), p, np.full(m, left + 1, np.int32)]))


def _make_maps(data, nside, coordinates, params, comm, nthreads, verbose, fit=None):
    """make_maps with the current estimates in `fit` (a _Fit, or None), which
    are updated; returns (MapResult, GLS). In the calibration iterations after
    the first, the GLS of the first one is reused (same pixels, flags and
    chunks): only the TOD and the templates are built."""
    params = GLSParameters() if params is None else params
    settings = params._gls_settings(nside)
    nthreads = resolve_nthreads(nthreads)
    npix = 12 * nside * nside
    npol = 3 if params.pol else 1
    t0 = time.perf_counter()  # the time of each step, with verbose
    chunks = _ChunkBuilder(nside, coordinates, params, comm, nthreads, fit)
    for j, d in chunks.items(data):  # one detector (or group) at a time: only its pointings are in memory (but a batch)
        chunks.add_item(j, d)
    reuse, conditional, cal = chunks.reuse, chunks.method._conditional, chunks.cal
    templates, shared, gain_ids = chunks.templates, chunks.shared, chunks.gain_ids
    tod_c, model_c, common_modes = chunks.tod, chunks.model_tod, chunks.common_modes

    t1 = time.perf_counter()
    prior, shared_prior = (None, None) if fit is None else fit.amplitudes.gls_priors()
    if fit is not None and fit.fit and cal is not None and cal.covariance_fixed_gains:
        # the same arrays: no copies
        fit.no_gains = ([t for t in templates if id(t) not in gain_ids], shared, prior, shared_prior)
    if reuse:
        gls = fit.gls
        gls.timings = _Timings()
        gls.set_templates(templates, shared, prior, shared_prior)
    else:
        gls = GLS(
            chunks.pix,
            None,
            chunks.w,
            pol_angles=(chunks.a2, chunks.gamma) if params.pol else None,
            comm=comm,
            nthreads=nthreads,
            overwrite_pix=True,  # chunks.pix holds copies made here
            fft_lengths=chunks.nfft,
            common_modes=common_modes,
            templates=templates,
            shared_templates=shared,
            prior=prior,
            shared_prior=shared_prior,
            shared_groups=fit.amplitudes.groups if fit is not None and fit.fit else None,
            offsets=(chunks.gx, chunks.gy) if params.subpixel_order else None,
            **settings,
            **chunks.method._gls_args(chunks.fs),
        )
        if fit is not None and fit.fit:
            fit.gls = gls
    chunks.pix = chunks.a2 = chunks.gamma = chunks.gx = chunks.gy = None
    gap_iterations = None
    if conditional:
        gap_iterations = _conditional_tod(tod_c, gls, npix, common_modes, chunks.method, chunks.flag_rng, nthreads)
        if comm is not None:
            gap_iterations = max(comm.allgather(gap_iterations))
    # warm start from the previous calibration iteration (if the pixels are the same)
    x0 = None if fit is None or fit.x0 is None or fit.x0.shape != (gls.ncomp, gls.nobs) else fit.x0
    t2 = time.perf_counter()
    # a looser tolerance in the first calibration iterations (CalibrationParameters.tol), not in the final maps
    tol = fit.tol if fit is not None and fit.fit and fit.tol is not None else params.tol
    m, info = gls.solve(tod_c, x0=x0, tol=tol, maxiter=params.maxiter, verbose=verbose)
    t3 = time.perf_counter()
    if fit is not None:
        fit.x0 = m
        if fit.fit:
            fit.residual = gls._freeze(gls._healpix(m, nside, hp.UNSEEN))
        fit.amplitudes.update(info if fit.fit else None)  # g (1 + dg), a + da, A + dA
        if fit.sky is not None:  # put back the sky model, as binned from its TOD: the maps of the whole sky
            bw = params.binned_weights
            if isinstance(bw, dict):  # every name checked in make_maps
                bw = np.array([bw[n] for n in chunks.names], dtype=float)
            m = gls._sum(m, gls.binned(model_c, weights=bw))

    # with full_maps "node", the ranks of a node share the full maps (read-only)
    hit_map = gls._freeze(gls._healpix(gls.full(gls.hits)[None], nside, 0, masked=False))[0]
    full = gls._healpix(m, nside, hp.UNSEEN)
    _subpixel_result(full, npol, params.subpixel_order, nside, nthreads, gls._writer)
    gls._freeze(full)  # before the views: they inherit read-only
    maps, gradient, hessian = _subpixel_result(full, npol, params.subpixel_order, nside, nthreads, write=False)
    if verbose and chunks.rank == 0:
        setup = f"GLS setup + conditional fill ({gap_iterations} CG iterations at most)" if conditional else "GLS setup"
        steps = [("TOD and templates", t1 - t0, chunks.timings), (setup, t2 - t1, gls.timings),
                 (f"PCG ({info['iterations']} iterations)", t3 - t2, gls.pcg_timings),
                 ("maps", time.perf_counter() - t3, None)]
        lines = [f"timings (phases of rank 0; full maps per {gls.full_maps_info}):", *_timing_table(steps)]
        print("\n".join(lines), flush=True)
    return MapResult(
        maps=maps,
        hit_map=hit_map,
        nside=nside,
        coordinates=coordinates,
        iterations=info["iterations"],
        converged=info["converged"],
        residuals=info["residuals"],
        gains=None if fit is None else fit.amplitudes.gains,
        gap_fill_iterations=gap_iterations,
        gradient=gradient,
        hessian=hessian,
    ), gls


def _padded(x, a, b, m, dtype):
    """x[a:b] as `dtype` with m zeros on both sides (a view if m = 0 and x is of dtype)."""
    if m == 0:
        return np.asarray(x[a:b], dtype)
    out = np.zeros(b - a + 2 * m, dtype)
    out[m : m + b - a] = x[a:b]
    return out


def _periods(n, fs, seconds):
    """The period of each of n samples, for periods of `seconds` (inf: one)
    counted from the first sample, as the gains and the amplitudes per period."""
    per = int(round(seconds * fs)) if np.isfinite(seconds) else n
    return np.arange(n) // max(1, per)


class _TODModel:
    """The TOD model of make_maps, d = g (P m + D + S + F + sum_k a_k B_k) + n:
    gains g, dipole D, sky model S, fixed templates F and fitted templates
    B_k with amplitudes a_k. Two sources of the estimates: the current ones
    of the fit iterations (from_fit, used by _make_maps) and the final ones
    of a MapResult (from_result, used by estimate_noise; S is then in the
    maps, put back as its binned map).

    The DetectorData are those given to make_maps, in the same order (groups
    expanded): `j` is the position of a DetectorData among them, which picks
    its gains and its amplitudes per period."""

    def __init__(self, templates, skies, sky, amplitudes, nthreads):
        self.templates = templates  # name -> Template
        self.skies = skies  # name -> map or alm (_model) of the templates given as a sky signal
        self.sky = sky  # sky model S (_model) or None
        self.amplitudes = amplitudes  # _Amplitudes: gains, shared and local amplitudes
        self.calibration = amplitudes.calibration
        self.nthreads = nthreads

    @classmethod
    def from_fit(cls, fit, nthreads):
        return cls(fit.templates, fit.skies, fit.sky, fit.amplitudes, nthreads)

    @classmethod
    def from_result(cls, result, nthreads):
        templates = dict(result.template_definitions or {})
        skies = {k: _model(t.sky, result.nside, f"templates[{k!r}].sky") for k, t in templates.items() if t.sky is not None}
        model = cls(templates, skies, None, _Amplitudes.from_result(result), nthreads)
        model.result, model.maps = result, result.maps
        if result.gradient is not None:  # back to the GLS form of the sub-pixel terms, once
            model.maps, model.sub = _subpixel_gls(result.maps, result.gradient, result.hessian, result.nside, nthreads)
        return model

    @property
    def alm(self):
        """Some model is alm: the samples need (theta, phi)."""
        return any(x is not None and np.iscomplexobj(x) for x in (self.sky, *self.skies.values()))

    def sample(self, d, pix, angle, loc):
        """(S, {name: B_k}) of the sky model and the sky templates for d (S None without a sky model)."""
        nthreads = self.nthreads
        sky = None if self.sky is None else _sample(self.sky, d, pix, angle, loc, nthreads)
        return sky, {k: _sample(b, d, pix, angle, loc, nthreads) for k, b in self.skies.items()}

    def tods(self, d, sampled):
        """{name: B_k} of d: the sky templates `sampled` (from sample, a dict
        that is updated) and the TOD templates of d fitted by make_maps."""
        sampled.update({k: np.array(v, dtype=float) for k, v in (d.templates or {}).items()
                        if k in self.templates and k not in self.skies})
        return sampled

    def gain(self, d, j):
        """(gains, period of each sample) of d, or None without calibration."""
        return self.amplitudes.gain(d, j)

    def known(self, d, j, sky, tods):
        """S + F + sum_k a_k B_k of d (a new array) from the B_k `tods` (from
        tods), of which the fixed ones are removed; returns it and, per template
        with amplitudes per period present in `tods`, (amplitudes, period of each
        sample)."""
        known = np.zeros(len(d.tod)) if sky is None else np.array(sky, dtype=float)
        with numba_threads(self.nthreads):
            for k in [k for k in tods if self.templates[k].fixed]:
                _axpy(known, 1.0, tods.pop(k))
            for k, t in tods.items():
                if self.templates[k].period_s is None:
                    a = self.amplitudes.shared_value(k, d)
                    if a is not None:
                        _axpy(known, a, t)
        local = {}
        for k in self.amplitudes.local:
            if k not in tods:
                continue
            amp, p = self.amplitudes.local_value(k, d, j)
            if amp.size:
                known += amp[p] * tods[k]
            local[k] = amp, p
        return known, local

    def calibrated(self, d, known, gain):
        """d / g - D - known (d - known without calibration: `gain` None, else from gain)."""
        if gain is None:
            return d.tod - known
        out = np.empty(len(d.tod))
        with numba_threads(self.nthreads):
            _calibrated(d.tod, gain[0], gain[1], d.dipole, known, out)
        return out

    def check(self, n):
        """n DetectorData were used: as many as in make_maps."""
        count = self.amplitudes.count
        if count is not None and n != count:
            raise ValueError(f"estimate_noise: {n} DetectorData, but the MapResult has gains or amplitudes per period "
                             f"for {count}: pass the data of make_maps, in the same order")

    def residual(self, d, j):
        """d / g - D - F - sum_k a_k B_k - P m for DetectorData d (a new array;
        from_result only)."""
        r, nthreads = self.result, self.nthreads
        count = self.amplitudes.count
        if count is not None and j >= count:
            self.check(j + 1)
        if self.calibration is not None and d.dipole is None:
            raise ValueError("estimate_noise: the MapResult has gains: the DetectorData need DetectorData.dipole, "
                             "as in make_maps")
        if (self.alm or r.gradient is not None) and d.pointing.pix is not None:
            raise ValueError("estimate_noise: templates as alm and the sub-pixel gradient need Pointing angles "
                             "(theta, phi), as in make_maps, not pixels")
        pix, angle, loc = _pixels_and_angles(d, r.nside, r.coordinates, nthreads,
                                             loc=self.alm or r.gradient is not None)
        tods = self.tods(d, self.sample(d, pix, angle, loc)[1])
        for k, t in self.templates.items():
            # a DetectorData may lack a TOD template; not one with amplitudes of its own
            if k not in tods and not t.fixed and (t.period_s is not None or (not t.common and d.templates is None)):
                a = self.amplitudes.local[k] if t.period_s is not None else self.amplitudes.shared_value(k, d)
                if t.period_s is not None and a is not None:
                    a = a[j]
                if a is not None and np.size(a):
                    raise ValueError(f"estimate_noise: DetectorData {d.name!r} has no template {k!r}, fitted by "
                                     "make_maps: pass the DetectorData.templates given to make_maps")
        res = self.calibrated(d, self.known(d, j, None, tods)[0], self.gain(d, j))
        res -= _project_maps(self.maps, pix, angle, d.pol_efficiency, nthreads)
        if r.gradient is not None:
            res -= _subpixel_tod(self.sub, pix, *_pixel_offsets(pix, loc, r.nside, float, nthreads))
        return res


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


def _group_cross_spectra(g, model, j, chunk_s, nbins, nthreads, acc):
    """Add the binned cross-periodograms of the residuals (_TODModel `model`;
    the first detector is DetectorData j of make_maps) of the detectors of
    DetectorGroup `g` to acc[names] = [fs, R (nbins, n, n), modes, sum of log f]."""
    dets = g.detectors
    if any(d.name is None for d in dets):
        raise ValueError("estimate_noise: the detectors of a DetectorGroup need a name")
    key, n, fs = tuple(d.name for d in dets), len(dets), dets[0].sampling_rate_hz
    res = np.array([model.residual(d, j + i) for i, d in enumerate(dets)])
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

    The residual of each DetectorData with respect to the TOD model of
    `result` (a MapResult, e.g. a first make_maps), d - P m with m its maps,
    or with calibration and templates d / g - D - F - sum_k a_k B_k - P m with
    the fitted gains and amplitudes (the noise in the units of the calibrated
    TOD; `data` must then be the data of make_maps, in the same order), is
    cut in chunks of `chunk_s`
    seconds; the periodogram of each chunk (Hann window, flagged samples set
    to 0, normalized by the window energy left) is averaged over all the chunks of the
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
    model = _TODModel.from_result(result, nthreads)
    acc = {}  # name -> [fs, sum of P per bin, count per bin, sum of log f per bin]
    gacc = {}  # names of a group -> [fs, sum of the cross-spectra per bin, count, sum of log f]
    j = 0  # DetectorData so far, groups expanded: the index of their gains and amplitudes per period
    for i, d in enumerate(data):
        if isinstance(d, DetectorGroup):
            _group_cross_spectra(d, model, j, chunk_s, nbins, nthreads, gacc)
            j += len(d.detectors)
            continue
        key = i if d.name is None else d.name
        fs = d.sampling_rate_hz
        res = model.residual(d, j)
        j += 1
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
            w2 = win**2 if d.flags is None else np.where(d.flags[a:b], 0.0, win**2)  # flagged: the window energy left
            p = 2 * np.abs(ducc0.fft.r2c(x * win, nthreads=nthreads)) ** 2 / (fs * np.sum(w2))
            f = np.fft.rfftfreq(x.size, 1 / fs)
            k = np.searchsorted(edges, f[1:-1], side="right") - 1
            ok = (k >= 0) & (k < nbins)
            np.add.at(a_[1], k[ok], p[1:-1][ok])
            np.add.at(a_[2], k[ok], 1)
            np.add.at(a_[3], k[ok], np.log(f[1:-1][ok]))
    model.check(j)
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

