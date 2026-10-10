"""High-level map-making: from detector timelines and pointings to maps.

Any pipeline (litebird_sim, files on disk, ...) only needs to provide
:class:`DetectorData` objects, as a list or as a generator (then only one
detector's pointings are in memory at a time); pysanepic computes pixels,
polarization angles, chunks and noise weights itself, with one set of
conventions documented here.
"""

from dataclasses import dataclass

import ducc0
import healpy as hp
import numba
import numpy as np

from .gls import GLS, numba_threads, one_over_f_weights, resolve_nthreads


@dataclass(kw_only=True)
class DetectorData:
    """One detector over one contiguous stretch of time.

    Pointing: either `theta`, `phi` (colatitude and longitude in `coordinates`,
    which is then required: "E" ecliptic, "G" galactic, "C" equatorial), or
    `pix`, the HEALPix pixel at resolution `nside` in `ordering` ("RING" or
    "NESTED"). With `pix`, `nside` and `coordinates` (None: the same as the
    output) must be those of the output maps. `psi` is the orientation angle of
    the detector's frame in `coordinates`.

    Polarization conventions (same as litebird_sim): The polarization angle seen by
    the sky is psi + pol_angle_rad without HWP, and
    psi + 2 hwp_angle - pol_angle_rad with an ideal HWP. The TOD model is
    d = I + pol_efficiency (Q cos 2a + U sin 2a) + noise, with `a` that angle.

    Noise: P(f) = sigma^2 (f^alpha + fknee^alpha) / (f^alpha + fmin^alpha) with
    sigma = net_ukrts * sqrt(sampling_rate_hz) * 1e-6, i.e. TOD in K.
    """

    tod: np.ndarray
    psi: np.ndarray
    theta: np.ndarray | None = None
    phi: np.ndarray | None = None
    pix: np.ndarray | None = None
    nside: int | None = None  # resolution of pix
    ordering: str = "RING"  # ordering of pix: "RING" or "NESTED"
    sampling_rate_hz: float
    net_ukrts: float
    fknee_hz: float = 0.0
    alpha: float = 1.0
    fmin_hz: float = 0.0
    pol_angle_rad: float = 0.0
    pol_efficiency: float = 1.0
    hwp_angle: np.ndarray | None = None
    coordinates: str | None = None  # healpy code: "E" ecliptic, "G" galactic, "C" equatorial


@dataclass
class MapResult:
    """maps: (3, npix) I/Q/U, or (1, npix) if pol=False; hp.UNSEEN where not solved."""

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
    tol, maxiter : PCG stops when |r|^2/|b|^2 < tol or after maxiter iterations.
    angle_dtype : storage of gamma cos 2a and gamma sin 2a: np.float32 (4 bytes
        per sample each, angle error ~1e-7 rad) or np.float64 (exact).
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
    theta, phi, psi = d.theta, d.phi, d.psi
    if d.pix is not None:
        if d.nside != nside or d.coordinates not in (None, coordinates):
            raise ValueError(
                f"DetectorData.pix is at nside={d.nside}, coordinates={d.coordinates!r}: "
                f"the maps need nside={nside}, coordinates={coordinates!r}"
            )
        if d.ordering not in ("RING", "NESTED"):
            raise ValueError(f"Unknown ordering {d.ordering!r}: use 'RING' or 'NESTED'")
    elif theta is None or phi is None:
        raise ValueError("DetectorData needs either theta and phi, or pix")
    elif d.coordinates is None:
        raise ValueError("DetectorData.coordinates is required with theta and phi")
    elif d.coordinates != coordinates:
        R = hp.Rotator(coord=[d.coordinates, coordinates]).mat
        out = np.empty((3, theta.size))
        with numba_threads(nthreads):
            _rotate(np.ascontiguousarray(theta), np.ascontiguousarray(phi), np.ascontiguousarray(psi), R, out)
        theta, phi, psi = out
    if d.hwp_angle is None:
        angle = psi + d.pol_angle_rad
    else:
        angle = psi + 2 * d.hwp_angle - d.pol_angle_rad
    if d.pix is not None:
        pix = np.asarray(d.pix)
        if pix.size and (pix.min() < 0 or pix.max() >= 12 * nside * nside):
            raise ValueError(f"DetectorData.pix outside [0, 12 nside^2) for nside={nside}")
        if d.ordering == "NESTED":
            pix = ducc0.healpix.Healpix_Base(nside, "NEST").nest2ring(pix, nthreads=nthreads)
        return pix, angle
    pix = ducc0.healpix.Healpix_Base(nside, "RING").ang2pix(np.stack([theta, phi], axis=1), nthreads=nthreads)
    return pix, angle


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
    nthreads = resolve_nthreads(nthreads)
    npix = 12 * nside * nside
    rank, size = (0, 1) if comm is None else (comm.rank, comm.size)
    n_virtual = 0  # padding margins on this rank; labels are unique across ranks
    pix_c, cos_c, sin_c, tod_c, w_c = [], [], [], [], []
    weights = {}  # chunks with the same length and noise share one array
    dtype = params.angle_dtype
    for d in data:  # one detector at a time: only its pointings are in memory
        pix, angle = _pixels_and_angles(d, nside, coordinates, nthreads)
        fs = d.sampling_rate_hz
        sigma = d.net_ukrts * np.sqrt(fs) * 1e-6
        m = int(round(params.pad_s * fs))
        for a, b in _chunk_bounds(len(d.tod), max(1, int(round(params.chunk_s * fs)))):
            n = b - a + 2 * m
            key = (n, fs, sigma, d.fknee_hz, d.alpha, d.fmin_hz)
            if key not in weights:
                weights[key] = one_over_f_weights(n, fs, sigma, d.fknee_hz, d.alpha, d.fmin_hz)
            w_c.append(weights[key])
            # gamma cos 2a, gamma sin 2a computed in float64, then stored as `dtype`;
            # the margins of a padded chunk carry no polarization (intensity-only
            # virtual pixels)
            if params.pol:
                c2, s2 = np.zeros(n, dtype), np.zeros(n, dtype)
                c2[m : n - m] = d.pol_efficiency * np.cos(2 * angle[a:b])
                s2[m : n - m] = d.pol_efficiency * np.sin(2 * angle[a:b])
                cos_c.append(c2)
                sin_c.append(s2)
            if m == 0:
                pix_c.append(pix[a:b].astype(np.int32))
                tod_c.append(d.tod[a:b])  # a view: the TOD is not copied
            else:
                left = npix + 2 * (n_virtual * size + rank)
                n_virtual += 1
                pix_c.append(np.concatenate([np.full(m, left), pix[a:b], np.full(m, left + 1)]).astype(np.int32))
                tod_c.append(_pad(d.tod[a:b], m, params.pad_fill))
        del pix, angle

    gls = GLS(
        pix_c,
        None,
        w_c,
        pol_weights=(cos_c, sin_c) if params.pol else None,
        preconditioner=params.preconditioner,
        min_pol_rcond=params.min_pol_rcond,
        comm=comm,
        nthreads=nthreads,
        angle_dtype=dtype,
    )
    del pix_c, cos_c, sin_c
    m, info = gls.solve(tod_c, tol=params.tol, maxiter=params.maxiter, verbose=verbose)

    hit_map = np.zeros(npix, dtype=np.int64)
    n = min(npix, gls.nobs)
    hit_map[:n] = gls.hits[:n]
    return MapResult(
        maps=gls.to_healpix(m, nside, fill=hp.UNSEEN),
        hit_map=hit_map,
        nside=nside,
        coordinates=coordinates,
        iterations=info["iterations"],
        converged=info["converged"],
        residuals=info["residuals"],
    )
