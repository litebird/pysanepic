"""High-level map-making: from detector timelines and pointings to maps.

Any pipeline (litebird_sim, files on disk, ...) only needs to provide
:class:`DetectorData` objects, as a list or as a generator (then only one
detector's pointings are in memory at a time), each with a :class:`Pointing`:
either angles, which pysanepic rotates to the output coordinates and
pixelizes at the output resolution, or HEALPix pixels already computed at that
resolution and in those coordinates. pysanepic computes polarization angles,
chunks and noise weights itself, with one set of conventions documented here.
"""

from dataclasses import dataclass

import ducc0
import healpy as hp
import numba
import numpy as np

from .gls import GLS, fast_fft_length, numba_threads, one_over_f_weights, psd_weights, resolve_nthreads, two_angle


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

    def pixels_and_psi(self, nside, coordinates, nthreads=1):
        """RING pixels at `nside` and psi in `coordinates` (the output maps)."""
        if self.pix is not None:
            if self.nside != nside or self.coordinates != coordinates:
                raise ValueError(
                    f"Pointing.pix is at nside={self.nside} in coordinates {self.coordinates!r}: "
                    f"the maps need nside={nside} in coordinates {coordinates!r}"
                )
            pix = np.asarray(self.pix)
            if self.ordering == "NESTED":
                pix = ducc0.healpix.Healpix_Base(nside, "NEST").nest2ring(pix, nthreads=nthreads)
            return pix, np.asarray(self.psi)
        theta, phi, psi = self.theta, self.phi, self.psi
        if self.coordinates != coordinates:
            R = hp.Rotator(coord=[self.coordinates, coordinates]).mat
            out = np.empty((3, len(psi)))
            with numba_threads(nthreads):
                _rotate(np.ascontiguousarray(theta), np.ascontiguousarray(phi), np.ascontiguousarray(psi), R, out)
            theta, phi, psi = out
        pix = ducc0.healpix.Healpix_Base(nside, "RING").ang2pix(np.stack([theta, phi], axis=1), nthreads=nthreads)
        return pix, psi


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
    `a` that angle.

    noise: the noise model, a :class:`OneOverFNoise` or a
    :class:`TabulatedNoise` (any object with a `weights(ns, fs)` method giving
    the inverse PSD on the rfft grid of a chunk), with the TOD in K.

    name: the detector, optional: estimate_noise averages the spectrum over
    all the DetectorData with the same name (e.g. the days of a detector).

    flags: boolean array like `tod`, True for samples to exclude (glitches,
    gaps, ...), or None. Flagged samples do not enter the maps and their TOD is
    not used (it may be NaN): each one is fitted by an unknown of its own,
    which is the exact GLS solution without them.
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

    def __post_init__(self):
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
        hp.UNSEEN where not solved.
    hit_map : number of samples per pixel.
    nside, coordinates : pixelization of the maps.
    iterations : number of PCG iterations.
    converged : whether norm(r)^2 / norm(b)^2 < tol was reached.
    residuals : norm(r)^2 / norm(b)^2 at each iteration.
    """

    maps: np.ndarray
    hit_map: np.ndarray
    nside: int
    coordinates: str
    iterations: int
    converged: bool
    residuals: np.ndarray


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
    split_gap_s : runs of flagged samples at least this long [s] split the
        chunk, and their samples are dropped: the noise correlations across
        the gap are neglected, but long gaps do not slow down the PCG. inf:
        never; 0: every run.
    coarse_nside : two-level preconditioner: the PCG also solves exactly for
        the I map at this HEALPix resolution (power of 2, <= nside), which
        takes care of the slow large-scale modes. The setup costs about one
        PCG iteration per coarse pixel (12 at coarse_nside=1). With common
        modes 1 halves the iterations (46 -> 21 on G100, -33% time); without
        them it saves fewer iterations than its setup costs. 0 (default):
        off.
    """

    chunk_s: float = 3600.0
    pad_s: float = 0.0
    pad_fill: str = "zeros"
    pol: bool = True
    preconditioner: str = "block"
    min_pol_rcond: float = 1e-2
    tol: float = 1e-12
    maxiter: int = 2000
    angle_dtype: type = np.float32
    flag_method: str = "marginalize"
    split_gap_s: float = 300.0
    fft_dtype: type = np.float64
    coarse_nside: int = 0


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


def _pixels_and_angles(d, nside, coordinates, nthreads):
    pix, psi = d.pointing.pixels_and_psi(nside, coordinates, nthreads)
    if d.hwp_angle is None:
        return pix, psi + d.pol_angle_rad
    return pix, psi + 2 * d.hwp_angle - d.pol_angle_rad


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


def _coarse_pixels(nside, coarse_nside):
    """Pixel labels (RING, nside; virtual pixels beyond) -> coarse pixel (NESTED, coarse_nside), or -1."""
    shift = 2 * (int(nside).bit_length() - int(coarse_nside).bit_length())

    def coarse(labels):
        out = np.full(labels.size, -1, np.int64)
        real = labels < 12 * nside * nside
        out[real] = hp.ring2nest(nside, labels[real]) >> shift
        return out

    return coarse


def make_maps(data, nside, coordinates="G", params=None, comm=None, nthreads=None, verbose=False):
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
    """
    params = GLSParameters() if params is None else params
    if params.flag_method not in ("marginalize", "offset"):
        raise ValueError(f"Unknown flag_method {params.flag_method!r}: use 'marginalize' or 'offset'")
    nc = params.coarse_nside
    if nc < 0 or (nc and (nc & (nc - 1) or nc > nside)):
        raise ValueError(f"coarse_nside must be 0 or a power of 2 <= nside, not {nc}")
    nthreads = resolve_nthreads(nthreads)
    npix = 12 * nside * nside
    rank, size = (0, 1) if comm is None else (comm.rank, comm.size)
    n_virtual = 0  # padding margins on this rank; labels are unique across ranks
    pix_c, a2_c, gamma_c, tod_c, w_c, nfft_c = [], [], [], [], [], []
    weights = {}  # chunks with the same length and noise share one array
    dtype = params.angle_dtype
    common_modes = []

    def add(d, split=True):
        """Chunks of one DetectorData; returns their indices."""
        nonlocal n_virtual
        first = len(pix_c)
        pix, angle = _pixels_and_angles(d, nside, coordinates, nthreads)
        fs = d.sampling_rate_hz
        m = int(round(params.pad_s * fs))
        chunks = _chunk_bounds(len(d.tod), max(1, int(round(params.chunk_s * fs))))
        gap = params.split_gap_s * fs if split else np.inf
        for a, b in (piece for c in chunks for piece in _split_at_gaps(*c, d.flags, gap)):
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
            tod = d.tod[a:b]  # a view: the TOD is not copied
            flagged = None if d.flags is None else d.flags[a:b]
            if flagged is not None and flagged.any():
                p[flagged] = _flag_labels(flagged, params.flag_method)  # unknowns not in the maps (see GLS)
                if m:  # the padding is built from the chunk: not from flagged samples
                    tod = np.where(flagged, tod[~flagged].mean() if not flagged.all() else 0.0, tod)
            if m == 0:
                pix_c.append(p)
                tod_c.append(tod)
            else:
                left = npix + 2 * (n_virtual * size + rank)
                n_virtual += 1
                pix_c.append(np.concatenate([np.full(m, left, np.int32), p, np.full(m, left + 1, np.int32)]))
                tod_c.append(_pad(tod, m, params.pad_fill))
        del pix, angle
        return list(range(first, len(pix_c)))

    for d in data:  # one detector (or group) at a time: only its pointings are in memory
        if not isinstance(d, DetectorGroup):
            add(d)
            continue
        # a group: chunks at the same times, not split at gaps (that would break
        # the alignment); one common_modes entry per chunk of time
        idx = [add(x, split=False) for x in d.detectors]
        fs = d.detectors[0].sampling_rate_hz
        U = np.array([mode.coupling for mode in d.modes], dtype=float).T  # (ndet, r)
        for chunks in zip(*idx):
            nfft = nfft_c[chunks[0]]
            lam = np.array([1 / np.maximum(w, w.max() * 1e-12) for w in (mode.noise.weights(nfft, fs) for mode in d.modes)])
            common_modes.append((list(chunks), U, lam))

    gls = GLS(
        pix_c,
        None,
        w_c,
        pol_angles=(a2_c, gamma_c) if params.pol else None,
        preconditioner=params.preconditioner,
        min_pol_rcond=params.min_pol_rcond,
        comm=comm,
        nthreads=nthreads,
        angle_dtype=dtype,
        overwrite_pix=True,  # pix_c holds copies made here
        fft_lengths=nfft_c,
        fft_dtype=params.fft_dtype,
        common_modes=common_modes,
        coarse=_coarse_pixels(nside, nc) if nc else None,
    )
    del pix_c, a2_c, gamma_c
    m, info = gls.solve(tod_c, tol=params.tol, maxiter=params.maxiter, verbose=verbose)

    hit_map = np.zeros(npix, dtype=np.int64)
    n = min(npix, gls.nobs)
    hit_map[:n] = gls.full(gls.hits)[:n]
    return MapResult(
        maps=gls.to_healpix(m, nside, fill=hp.UNSEEN),
        hit_map=hit_map,
        nside=nside,
        coordinates=coordinates,
        iterations=info["iterations"],
        converged=info["converged"],
        residuals=info["residuals"],
    )


def _model_tod(d, maps, nside, coordinates, nthreads):
    """P m for one DetectorData: the TOD of the maps (UNSEEN Q/U taken as 0)."""
    pix, angle = _pixels_and_angles(d, nside, coordinates, nthreads)
    m = np.where(maps == hp.UNSEEN, 0.0, maps)
    tod = m[0, pix]
    if m.shape[0] == 3:
        tod = tod + d.pol_efficiency * (m[1, pix] * np.cos(2 * angle) + m[2, pix] * np.sin(2 * angle))
    return tod


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

