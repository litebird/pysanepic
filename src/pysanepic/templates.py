"""Templates of the TOD for common systematics, from a model of the sky.

Polarization angles, FWHM, pointing offsets and ellipticities of the beams,
to first order: the :class:`~pysanepic.Template` objects (and, for the
beams, the TOD of each detector) to pass to :func:`~pysanepic.make_maps`,
with the constraints (sum_zero) that their degeneracies with the maps need;
:func:`beam_parameters` turns the fitted amplitudes into beam parameters.
See the documentation (Templates) for what the amplitudes are.
"""

import dataclasses

import healpy as hp
import numpy as np

from .mapmaker import Template, _model, _sample

BEAM_NAMES = ("offset_x", "offset_y", "ell_plus", "ell_cross")


def pol_angle_template(sky, *, mask=None):
    """The template of the polarization angle errors of the detectors.

    sky: the sky model, alm T, E, B (complex, healpy order, mmax = lmax) or
    HEALPix maps I, Q, U, in the coordinates of the maps. An error delta on
    the angle changes the TOD by delta 2 gamma (-Q sin 2a + U cos 2a), the
    signal of the sky [0, 2U, -2Q] (alm [0, 2B, -2E]).

    Returns a Template with sum_zero=True: the amplitudes are delta_i -
    mean(delta) [rad], the common rotation goes into the maps. With an ideal
    HWP the angle is psi + 2 hwp_angle - pol_angle: the amplitudes are
    -(delta_i - mean).
    """
    x = np.atleast_2d(np.asarray(sky))
    if x.shape[0] != 3:
        raise ValueError("pol_angle_template: the sky needs 3 components (T, E, B or I, Q, U)")
    return Template(sky=np.array([0 * x[0], 2 * x[2], -2 * x[1]]), sum_zero=True, mask=mask)


def fwhm_template(sky_alm, *, mask=None):
    """The template of the FWHM differences of the beams.

    sky_alm: the sky model through the mean beam of the detectors, alm (T or
    T, E, B; complex, healpy order, mmax = lmax) in the coordinates of the
    maps. A change delta sigma^2 of the beam variance (FWHM = sigma
    sqrt(8 ln 2)) changes the TOD by delta sigma^2 / 2 times the Laplacian of
    the sky: -l(l+1)/2 on T, -(l(l+1) - 4)/2 on E and B (spin 2).

    Returns a Template with sum_zero=True: the amplitudes are delta sigma^2_i
    - mean [rad^2] (the common part is a change of the sky); delta FWHM =
    8 ln 2 delta sigma^2 / (2 FWHM).
    """
    if not np.iscomplexobj(sky_alm):
        raise ValueError("fwhm_template: the sky model as alm (complex), not maps")
    x = _model(sky_alm, 0, "fwhm_template")
    lmax = hp.Alm.getlmax(x.shape[1])
    ell = np.arange(lmax + 1, dtype=float)
    lap = [hp.almxfl(a, -(ell * (ell + 1) - (4.0 if k else 0.0)) / 2) for k, a in enumerate(x)]
    return Template(sky=np.array(lap), sum_zero=True, mask=mask)


def _displaced(theta, phi, psi, dx, dy, frame):
    """The pointing moved by (dx, dy) [rad] along x = cos b e_theta + sin b
    e_phi, y = -sin b e_theta + cos b e_phi, b = psi + frame, with psi
    parallel-transported along the geodesic: (theta, phi, psi)."""
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    n = np.array([st * cp, st * sp, ct])
    et, ep = np.array([ct * cp, ct * sp, -st]), np.array([-sp, cp, 0 * sp])
    rho, b = np.hypot(dx, dy), psi + frame
    t = (dx * (np.cos(b) * et + np.sin(b) * ep) + dy * (-np.sin(b) * et + np.cos(b) * ep)) / rho
    w = np.cross(n, t, axis=0)
    n2, t2 = np.cos(rho) * n + np.sin(rho) * t, -np.sin(rho) * n + np.cos(rho) * t
    p = np.cos(psi) * et + np.sin(psi) * ep  # the direction of psi, transported
    p2 = np.sum(p * t, 0) * t2 + np.sum(p * w, 0) * w
    th2, ph2 = np.arccos(np.clip(n2[2], -1, 1)), np.arctan2(n2[1], n2[0]) % (2 * np.pi)
    et2 = np.array([np.cos(th2) * np.cos(ph2), np.cos(th2) * np.sin(ph2), -np.sin(th2)])
    ep2 = np.array([-np.sin(ph2), np.cos(ph2), 0 * ph2])
    return th2, ph2, np.arctan2(np.sum(p2 * ep2, 0), np.sum(p2 * et2, 0))


def beam_tods(d, sky_alm, coordinates, *, offsets=True, ellipticity=True, eps=None, nthreads=1):
    """The beam templates of one DetectorData: pointing offsets and
    ellipticity, as TODs.

    d: a DetectorData with a Pointing as angles (theta, phi). sky_alm: the sky
    model through the mean beam, alm (T or T, E, B) in `coordinates`, those
    of the maps. The model is resampled with the beam moved by +-eps [rad]
    (default 0.1 / lmax) along the axes of the detector, x = cos b e_theta +
    sin b e_phi and y = -sin b e_theta + cos b e_phi with b = psi +
    pol_angle_rad, transporting the polarization frame: first differences
    for the offsets (rho_x, rho_y [rad]), second differences for the
    ellipticity, (d_xx - d_yy) / 2 and d_xy, whose amplitudes are the changes
    c+, cx of the second moments [rad^2] (Sigma_xx = sigma^2 + c+, Sigma_yy =
    sigma^2 - c+, Sigma_xy = cx). HWP angle and pol_efficiency are those of
    `d`. 4 resamplings for the offsets, 8 for both.

    Returns {"offset_x", "offset_y", "ell_plus", "ell_cross": array like
    d.tod} (those asked for).
    """
    if not (offsets or ellipticity):
        raise ValueError("beam_tods: nothing to compute (offsets=False, ellipticity=False)")
    if not np.iscomplexobj(sky_alm):
        raise ValueError("beam_tods: the sky model as alm (complex), not maps")
    x = _model(sky_alm, 0, "beam_tods")
    if eps is None:
        eps = 0.1 / hp.Alm.getlmax(x.shape[1])
    theta, phi, psi = d.pointing.angles(coordinates, nthreads)
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1)] + ([(1, 1), (-1, -1), (1, -1), (-1, 1)] if ellipticity else [])
    s = {}
    for dx, dy in steps:
        th, ph, ps = _displaced(theta, phi, psi, dx * eps, dy * eps, d.pol_angle_rad)
        angle = ps + d.pol_angle_rad if d.hwp_angle is None else ps + 2 * d.hwp_angle - d.pol_angle_rad
        s[dx, dy] = _sample(x, d, None, angle, np.stack([th, ph], axis=1), nthreads)
    out = {}
    if offsets:
        out["offset_x"] = (s[1, 0] - s[-1, 0]) / (2 * eps)
        out["offset_y"] = (s[0, 1] - s[0, -1]) / (2 * eps)
    if ellipticity:
        out["ell_plus"] = (s[1, 0] + s[-1, 0] - s[0, 1] - s[0, -1]) / (2 * eps**2)
        out["ell_cross"] = (s[1, 1] + s[-1, -1] - s[1, -1] - s[-1, 1]) / (4 * eps**2)
    return out


def beam_templates(data, sky_alm, coordinates, *, offsets=True, ellipticity=True, eps=None, nthreads=1):
    """The beam templates of all the DetectorData (see :func:`beam_tods`).

    Returns (data, templates): the DetectorData as a list, with the TODs added
    to their `templates`, and {name: Template} for make_maps: one offset per
    detector (DetectorData.name), and one ellipticity per detector, summing
    to zero without HWP (a common ellipticity turns with the polarization
    angle and is degenerate with Q/U) and free with an HWP. All the TODs are
    computed here and kept in memory: with a generator, use beam_tods.
    """
    data = list(data)
    hwp = {d.hwp_angle is not None for d in data}
    if len(hwp) > 1:
        raise ValueError("beam_templates: DetectorData with and without HWP (the ellipticity constraints differ)")
    out = []
    for d in data:
        tods = beam_tods(d, sky_alm, coordinates, offsets=offsets, ellipticity=ellipticity, eps=eps, nthreads=nthreads)
        old = d.templates or {}
        if set(old) & set(tods):
            raise ValueError(f"beam_templates: DetectorData.templates already has {sorted(set(old) & set(tods))}")
        out.append(dataclasses.replace(d, templates={**old, **tods}))
    sum_zero = hwp == {False}
    templates = {}
    if offsets:
        templates.update(offset_x=Template(), offset_y=Template())
    if ellipticity:
        templates.update(ell_plus=Template(sum_zero=sum_zero), ell_cross=Template(sum_zero=sum_zero))
    return out, templates


def beam_parameters(result, templates, sigma=None, fwhm="fwhm"):
    """The beam parameters of each detector from the fitted amplitudes.

    result: the MapResult of make_maps; templates: the dict of Templates
    passed to make_maps, with those of :func:`beam_templates` (offset_x,
    offset_y, ell_plus, ell_cross) and of :func:`fwhm_template` (key `fwhm`),
    any of them: the missing ones are taken as 0. The fitted second moments
    are about the nominal pointing and include rho rho^T; here it is
    subtracted (also from amplitudes summing to zero: then the values are
    relative to the mean of the detectors).

    Returns {detector: {"offset_x", "offset_y" [rad], "c_plus", "c_cross",
    "delta_sigma2" [rad^2]}}; with `sigma`, the sigma of the beam of the sky
    model [rad], also "sigma_major", "sigma_minor" [rad] and "angle" [rad],
    the major axis from x (FWHM = sigma sqrt(8 ln 2)).
    """
    names = {"offset_x": "offset_x", "offset_y": "offset_y", "c_plus": "ell_plus",
             "c_cross": "ell_cross", "delta_sigma2": fwhm}
    fitted = {k: result.templates[v] for k, v in names.items() if v in templates}
    if not fitted:
        raise ValueError(f"beam_parameters: none of {sorted(names.values())} in templates")
    dets = sorted({det for amp in fitted.values() for det in amp}, key=str)
    v = {k: np.array([fitted[k].get(det, 0.0) if k in fitted else 0.0 for det in dets]) for k in names}
    rx, ry = v["offset_x"], v["offset_y"]
    for k, corr in (("c_plus", (rx**2 - ry**2) / 2), ("c_cross", rx * ry), ("delta_sigma2", (rx**2 + ry**2) / 2)):
        if k in fitted:
            v[k] = v[k] - (corr - corr.mean() if templates[names[k]].sum_zero else corr)
    out = {det: {k: float(v[k][i]) for k in names} for i, det in enumerate(dets)}
    if sigma is not None:
        mean = sigma**2 + v["delta_sigma2"]
        half = np.hypot(v["c_plus"], v["c_cross"])
        for i, det in enumerate(dets):
            out[det].update(sigma_major=float(np.sqrt(mean[i] + half[i])), sigma_minor=float(np.sqrt(mean[i] - half[i])),
                            angle=float(np.arctan2(v["c_cross"][i], v["c_plus"][i]) / 2))
    return out
