"""High-level map-making: from detector timelines and pointings to maps.

Any pipeline (litebird_sim, files on disk, ...) only needs to fill a list of
:class:`DetectorData`; pysanepic computes pixels, polarization angles, chunks
and noise weights itself, with one set of conventions documented here.
"""

from dataclasses import dataclass

import ducc0
import healpy as hp
import numba
import numpy as np

from .gls import GLS, one_over_f_weights


@dataclass
class DetectorData:
    """One detector over one contiguous stretch of time.

    Pointing and polarization conventions (same as litebird_sim):
    theta, phi are colatitude and longitude in `coordinates`; psi is the
    orientation angle of the detector's frame. The polarization angle seen by
    the sky is psi + pol_angle_rad without HWP, and
    psi + 2 hwp_angle - pol_angle_rad with an ideal HWP. The TOD model is
    d = I + pol_efficiency (Q cos 2a + U sin 2a) + noise, with `a` that angle.

    Noise: P(f) = sigma^2 (f^alpha + fknee^alpha) / (f^alpha + fmin^alpha) with
    sigma = net_ukrts * sqrt(sampling_rate_hz) * 1e-6, i.e. TOD in K.
    """

    tod: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    psi: np.ndarray
    sampling_rate_hz: float
    net_ukrts: float
    fknee_hz: float = 0.0
    alpha: float = 1.0
    fmin_hz: float = 0.0
    pol_angle_rad: float = 0.0
    pol_efficiency: float = 1.0
    hwp_angle: np.ndarray | None = None
    coordinates: str = "E"  # healpy code: "E" ecliptic, "G" galactic, "C" equatorial


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


def _pixels_and_angles(d, nside, coordinates):
    theta, phi, psi = d.theta, d.phi, d.psi
    if d.coordinates != coordinates:
        R = hp.Rotator(coord=[d.coordinates, coordinates]).mat
        out = np.empty((3, theta.size))
        _rotate(np.ascontiguousarray(theta), np.ascontiguousarray(phi), np.ascontiguousarray(psi), R, out)
        theta, phi, psi = out
    if d.hwp_angle is None:
        angle = psi + d.pol_angle_rad
    else:
        angle = psi + 2 * d.hwp_angle - d.pol_angle_rad
    pix = ducc0.healpix.Healpix_Base(nside, "RING").ang2pix(np.stack([theta, phi], axis=1), nthreads=0)
    return pix, angle


def _chunk_bounds(n, ns):
    """Split n samples into chunks of ns; the leftover goes into the last chunk."""
    nchunk = max(1, n // ns)
    starts = np.arange(nchunk) * ns
    return zip(starts, np.append(starts[1:], n))


def make_maps(
    data,
    nside,
    coordinates="G",
    chunk_s=3600.0,
    pol=True,
    preconditioner="block",
    min_pol_rcond=1e-2,
    tol=1e-12,
    maxiter=2000,
    comm=None,
    verbose=False,
):
    """GLS maps with 1/f noise from a list of :class:`DetectorData`.

    Parameters
    ----------
    data : list of DetectorData
        With MPI, the data local to this rank (possibly empty).
    nside : int
        HEALPix resolution of the output maps (RING ordering).
    coordinates : str
        Coordinate system of the output maps ("G", "E" or "C").
    chunk_s : float
        N^-1 is applied independently on chunks of this duration (seconds);
        longer chunks capture lower frequencies but cost more.
    pol : bool
        Solve for I/Q/U (True) or I only.
    preconditioner : str
        "block" (default, 3x3 I/Q/U block per pixel) or "jacobi" (SANEPIC's).
    min_pol_rcond : float
        Q/U are solved only in pixels whose polarization-angle coverage gives a
        reciprocal condition number >= this value (0: no check, as SANEPIC).
    tol, maxiter : float, int
        PCG stops when |r|^2/|b|^2 < tol or after maxiter iterations.
    comm : mpi4py communicator or None
        Every rank of `comm` must call this function.
    """
    # Group chunks by length: each group is one GLS block
    groups = {}
    for d in data:
        pix, angle = _pixels_and_angles(d, nside, coordinates)
        fs = d.sampling_rate_hz
        sigma = d.net_ukrts * np.sqrt(fs) * 1e-6
        for a, b in _chunk_bounds(len(d.tod), max(1, int(round(chunk_s * fs)))):
            g = groups.setdefault(b - a, {"pix": [], "psi": [], "tod": [], "w": [], "gamma": []})
            g["pix"].append(pix[a:b])
            g["psi"].append(angle[a:b])
            g["tod"].append(d.tod[a:b])
            g["gamma"].append(d.pol_efficiency)
            g["w"].append(one_over_f_weights(b - a, fs, sigma, d.fknee_hz, d.alpha, d.fmin_hz))

    blocks = {k: [np.array(g[k]) for g in groups.values()] for k in ("pix", "psi", "tod", "w", "gamma")}
    gls = GLS(
        blocks["pix"],
        blocks["psi"] if pol else None,
        blocks["w"],
        pol_efficiency=blocks["gamma"] if pol else None,
        preconditioner=preconditioner,
        min_pol_rcond=min_pol_rcond,
        comm=comm,
    )
    m, info = gls.solve(blocks["tod"], tol=tol, maxiter=maxiter, verbose=verbose)

    hit_map = np.zeros(12 * nside * nside, dtype=np.int64)
    hit_map[gls.pixels] = gls.hits
    return MapResult(
        maps=gls.to_healpix(m, nside, fill=hp.UNSEEN),
        hit_map=hit_map,
        nside=nside,
        coordinates=coordinates,
        iterations=info["iterations"],
        converged=info["converged"],
        residuals=info["residuals"],
    )
