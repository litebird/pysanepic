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

from .gls import GLS, numba_threads, one_over_f_weights, psd_weights, resolve_nthreads, two_angle


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
class DetectorData:
    """One detector over one contiguous stretch of time.

    Polarization conventions (same as litebird_sim): the polarization angle seen by
    the sky is psi + pol_angle_rad without HWP, and
    psi + 2 hwp_angle - pol_angle_rad with an ideal HWP, with psi from `pointing`.
    The TOD model is d = I + pol_efficiency (Q cos 2a + U sin 2a) + noise, with
    `a` that angle.

    Noise, one of:

    - `noise_model="toast"` (default): P(f) = sigma^2 (f^alpha + fknee^alpha) /
      (f^alpha + fmin^alpha);
    - `noise_model="keshner"`: P(f) = sigma^2 ((f^2 + fknee^2) / (f^2 +
      fmin^2))^(alpha/2), litebird_sim's noise with engine="ducc";
    - `noise_psd=(freqs_hz, psd)`: a tabulated one-sided PSD in K^2/Hz, which
      replaces the model (net_ukrts, fknee_hz, alpha, fmin_hz are not used);

    with sigma = net_ukrts * sqrt(sampling_rate_hz) * 1e-6, i.e. TOD in K.

    flags: boolean array like `tod`, True for samples to exclude (glitches,
    gaps, ...), or None. Flagged samples do not enter the maps and their TOD is
    not used (it may be NaN): each one is fitted by an unknown of its own,
    which is the exact GLS solution without them.
    """

    tod: np.ndarray
    pointing: Pointing
    sampling_rate_hz: float
    net_ukrts: float
    fknee_hz: float = 0.0
    alpha: float = 1.0
    fmin_hz: float = 0.0
    pol_angle_rad: float = 0.0
    pol_efficiency: float = 1.0
    hwp_angle: np.ndarray | None = None
    noise_model: str = "toast"
    noise_psd: tuple | None = None
    flags: np.ndarray | None = None

    def __post_init__(self):
        if self.flags is not None:
            if not (isinstance(self.flags, np.ndarray) and self.flags.dtype == bool and self.flags.shape == (len(self.tod),)):
                raise ValueError("DetectorData.flags must be a boolean array with one value per sample (True = flagged)")
        if self.noise_model not in ("toast", "keshner"):
            raise ValueError(f"DetectorData.noise_model must be 'toast' or 'keshner', not {self.noise_model!r}")
        if self.noise_psd is not None:
            f, p = (np.asarray(x, dtype=float) for x in self.noise_psd)
            if f.ndim != 1 or f.shape != p.shape or f.size < 2:
                raise ValueError("DetectorData.noise_psd must be (freqs_hz, psd), two 1D arrays of the same length >= 2")
            if not (np.all(np.isfinite(f)) and np.all(np.isfinite(p)) and f[0] > 0 and np.all(np.diff(f) > 0) and np.all(p > 0)):
                raise ValueError("DetectorData.noise_psd: frequencies must be > 0 and increasing, the PSD finite and > 0")
            object.__setattr__(self, "noise_psd", (f, p))
        if not isinstance(self.pointing, Pointing):
            raise TypeError("DetectorData.pointing must be a pysanepic.Pointing")
        if len(self.tod) != len(self.pointing):
            raise ValueError(f"DetectorData.tod has {len(self.tod)} samples, the pointing {len(self.pointing)}")
        if self.hwp_angle is not None and len(self.hwp_angle) != len(self.tod):
            raise ValueError("DetectorData.hwp_angle and tod must have the same length")


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
    split_gap_s : runs of flagged samples at least this long [s] split the
        chunk, and their samples are dropped: the noise correlations across
        the gap are neglected, but long gaps do not slow down the PCG. inf:
        never; 0: every run.
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
    nthreads = resolve_nthreads(nthreads)
    npix = 12 * nside * nside
    rank, size = (0, 1) if comm is None else (comm.rank, comm.size)
    n_virtual = 0  # padding margins on this rank; labels are unique across ranks
    pix_c, a2_c, gamma_c, tod_c, w_c = [], [], [], [], []
    weights = {}  # chunks with the same length and noise share one array
    dtype = params.angle_dtype
    for d in data:  # one detector at a time: only its pointings are in memory
        pix, angle = _pixels_and_angles(d, nside, coordinates, nthreads)
        fs = d.sampling_rate_hz
        sigma = d.net_ukrts * np.sqrt(fs) * 1e-6
        m = int(round(params.pad_s * fs))
        chunks = _chunk_bounds(len(d.tod), max(1, int(round(params.chunk_s * fs))))
        gap = params.split_gap_s * fs
        for a, b in (piece for c in chunks for piece in _split_at_gaps(*c, d.flags, gap)):
            n = b - a + 2 * m
            if d.noise_psd is not None:
                key = (n, fs, d.noise_psd[0].tobytes(), d.noise_psd[1].tobytes())  # by value: ids get reused
                if key not in weights:
                    weights[key] = psd_weights(n, fs, *d.noise_psd)
            else:
                key = (n, fs, sigma, d.fknee_hz, d.alpha, d.fmin_hz, d.noise_model)
                if key not in weights:
                    weights[key] = one_over_f_weights(n, fs, sigma, d.fknee_hz, d.alpha, d.fmin_hz, d.noise_model)
            w_c.append(weights[key])
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
