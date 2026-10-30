"""Templates of the TOD for common systematics, from a model of the sky.

Polarization angles, FWHM, pointing offsets and ellipticities of the beams,
to first order: the :class:`~pysanepic.Template` objects (and, for the
beams, the TOD of each detector) to pass to :func:`~pysanepic.make_maps`,
with the constraints (sum_zero) that their degeneracies with the maps need;
:func:`beam_parameters` turns the fitted amplitudes into beam parameters.
:func:`blm_templates`: the asymmetries of beams known as alm, as a fixed
template.
See the documentation (Templates) for what the amplitudes are.
"""

import dataclasses

import ducc0
import healpy as hp
import numpy as np

from .mapmaker import Template, _model

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


def _spin(alm, spin, lmax, loc, nthreads):
    """The complex field -sum_lm (alm[0] + i alm[1]) sY_lm at `loc` (healpy
    and ducc0 convention for spin > 0; for spin 0, sum_lm (alm[0] + i
    alm[1]) Y_lm)."""
    eps = 1e-6 if alm.dtype == np.complex64 else 1e-10
    if spin == 0:
        re, im = (ducc0.sht.synthesis_general(alm=a[None], spin=0, lmax=lmax, loc=loc, epsilon=eps, nthreads=nthreads)[0]
                  for a in alm)
        return re + 1j * im
    re, im = ducc0.sht.synthesis_general(alm=alm, spin=spin, lmax=lmax, loc=loc, epsilon=eps, nthreads=nthreads)
    return re + 1j * im


def beam_tods(d, sky_alm, coordinates, *, offsets=True, ellipticity=True, nthreads=1):
    """The beam templates of one DetectorData: pointing offsets and
    ellipticity, as TODs.

    d: a DetectorData with a Pointing as angles (theta, phi). sky_alm: the sky
    model through the mean beam, alm (T or T, E, B) in `coordinates`, those
    of the maps. The templates are the derivatives of the model along the
    axes of the detector, x = cos b e_theta + sin b e_phi and y = -sin b
    e_theta + cos b e_phi with b = psi + pol_angle_rad: the first for the
    offsets (rho_x, rho_y [rad]), (d_xx - d_yy) / 2 and d_xy for the
    ellipticity, whose amplitudes are the changes c+, cx of the second
    moments [rad^2] (Sigma_xx = sigma^2 + c+, Sigma_yy = sigma^2 - c+,
    Sigma_xy = cx). They are exact, from the spin-raising and lowering
    operators in harmonic space (ducc0 synthesis_general with spin 1 and 2 for
    T; 0 to 4 for the covariant derivatives of Q + iU); HWP angle and
    pol_efficiency are those of `d`. 2 syntheses for T, 3 + 4 more with
    polarization.

    Returns {"offset_x", "offset_y", "ell_plus", "ell_cross": array like
    d.tod} (those asked for).
    """
    if not (offsets or ellipticity):
        raise ValueError("beam_tods: nothing to compute (offsets=False, ellipticity=False)")
    if not np.iscomplexobj(sky_alm):
        raise ValueError("beam_tods: the sky model as alm (complex), not maps")
    x = _model(sky_alm, 0, "beam_tods")
    lmax = hp.Alm.getlmax(x.shape[1])
    ell = np.arange(lmax + 1, dtype=float)
    theta, phi, psi = d.pointing.angles(coordinates, nthreads)
    loc = np.stack([theta, phi], axis=1)
    a = psi + d.pol_angle_rad if d.hwp_angle is None else psi + 2 * d.hwp_angle - d.pol_angle_rad
    eb = np.exp(-1j * (psi + d.pol_angle_rad))  # e^{-ib}
    # the polarization along the derivative: Re[gamma e^{-2ia} (derivative of Q + iU)]
    pol = d.pol_efficiency * np.exp(-2j * a) if x.shape[0] == 3 else None

    def fl(f, *alm):
        return np.array([hp.almxfl(c, np.sqrt(np.maximum(f, 0))) for c in alm])

    zero = np.zeros_like(x[0])
    out = {}
    if offsets:
        # d_b = -(e^{-ib} edth + e^{ib} edthbar) / 2 for any spin
        dt = _spin(-fl(ell * (ell + 1), x[0], zero), 1, lmax, loc, nthreads)  # edth T
        gx, gy = -(eb * dt).real, -(eb * dt).imag
        if pol is not None:
            dp = _spin(fl((ell - 2) * (ell + 3), x[1], x[2]), 3, lmax, loc, nthreads)  # edth P
            dpb = _spin(-fl((ell + 2) * (ell - 1), x[1], x[2]), 1, lmax, loc, nthreads)  # edthbar P
            gx = gx + (pol * -(eb * dp + np.conj(eb) * dpb) / 2).real
            gy = gy + (pol * -(-1j * eb * dp + 1j * np.conj(eb) * dpb) / 2).real
        out["offset_x"], out["offset_y"] = gx, gy
    if ellipticity:
        # (d_xx - d_yy) / 2 = (e^{-2ib} edth^2 + e^{2ib} edthbar^2) / 4, d_xy = -i (e^{-2ib} edth^2 - c.c.) / 4
        e2 = eb**2
        tt = _spin(-fl((ell - 1) * ell * (ell + 1) * (ell + 2), x[0], zero), 2, lmax, loc, nthreads)  # edth^2 T
        hp_, hx = (e2 * tt).real / 2, (e2 * tt).imag / 2
        if pol is not None:
            f2 = (ell - 1) * ell * (ell + 1) * (ell + 2)
            pp = _spin(fl((ell - 2) * (ell + 3) * (ell - 3) * (ell + 4), x[1], x[2]), 4, lmax, loc, nthreads)  # edth^2 P
            ppb = -_spin(fl(f2, x[1], x[2]), 0, lmax, loc, nthreads)  # edthbar^2 P
            hp_ = hp_ + (pol * (e2 * pp + np.conj(e2) * ppb) / 4).real
            hx = hx + (pol * -1j * (e2 * pp - np.conj(e2) * ppb) / 4).real
        out["ell_plus"], out["ell_cross"] = hp_, hx
    return out


def beam_templates(data, sky_alm, coordinates, *, offsets=True, ellipticity=True, nthreads=1):
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
        tods = beam_tods(d, sky_alm, coordinates, offsets=offsets, ellipticity=ellipticity, nthreads=nthreads)
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


def _asymmetric(blm, lmax, mmax):
    """The beam without its part symmetric about its axis: m = 0 for T, and for
    Q/U the co-rotating part at m = 2, (E - iB) / 2 (E + iB = 0), which a
    circular beam has only. That part is the beam of the maps."""
    out = blm.copy()
    out[0, :lmax + 1] = 0.0  # m = 0 comes first
    if blm.shape[0] == 3:
        if mmax >= 2:
            i = 2 * (2 * lmax - 1) // 2 + np.arange(2, lmax + 1)  # m = 2, l >= 2
            sym = (blm[1, i] - 1j * blm[2, i]) / 2
            out[1, i] -= sym
            out[2, i] -= 1j * sym
    return out


def _reference_blm(model_bl, lmax, mmax, ncomp, pol_angle):
    """The blm of the circular beam with window `model_bl` (as healpy's
    gauss_beam: (lmax+1,) for T, or (lmax+1, >= 3) with the columns T, E, B)
    in the frame of a detector at polarization angle `pol_angle`, with the
    conventions of litebird_sim's gauss_beam_to_alm: T at m = 0,
    sqrt((2l+1)/4pi) b_l, and the co-rotating polarization at m = 2,
    (E - iB)/2 = -sqrt(2) sqrt((2l+1)/32pi) b_l e^{-2i pol_angle} (E + iB = 0)."""
    bl = np.asarray(model_bl, dtype=float)
    if bl.ndim == 1:
        bl = bl[:, None]
    if bl.shape[0] < lmax + 1:
        raise ValueError(f"blm_tods: model_bl needs l up to the lmax of sky_alm ({lmax})")
    if ncomp == 3 and (bl.shape[1] < 3 or not np.allclose(bl[: lmax + 1, 1], bl[: lmax + 1, 2])):
        raise ValueError("blm_tods: with polarization, model_bl needs the columns T, E, B, with E = B (a circular beam)")
    if ncomp == 3 and mmax < 2:
        raise ValueError("blm_tods: with polarization and model_bl, the beam needs mmax >= 2")
    ell = np.arange(lmax + 1)
    out = np.zeros((ncomp, (mmax + 1) * (lmax + 1) - mmax * (mmax + 1) // 2), complex)
    out[0, : lmax + 1] = np.sqrt((2 * ell + 1) / (4 * np.pi)) * bl[: lmax + 1, 0]  # m = 0 comes first
    if ncomp == 3:
        i = 2 * (2 * lmax - 1) // 2 + ell[2:]  # m = 2, l >= 2
        sym = -np.sqrt(2) * np.sqrt((2 * ell[2:] + 1) / (32 * np.pi)) * bl[2 : lmax + 1, 1] * np.exp(-2j * pol_angle)
        out[1, i] = sym
        out[2, i] = 1j * sym
    return out


def _blm_interpolator(sky_alm, blm, kmax, epsilon, nthreads, model_bl=None, pol_angle=0.0):
    """The ducc0 Interpolator of the sky through the asymmetries of `blm` (minus
    its symmetric part, or with model_bl minus the circular beam of the model
    at `pol_angle`), with kmax (None: the largest m of the asymmetries above
    epsilon of their largest coefficient)."""
    if not np.iscomplexobj(sky_alm):
        raise ValueError("blm_tods: the sky model as alm (complex), not maps")
    x = _model(sky_alm, 0, "blm_tods")
    lmax = hp.Alm.getlmax(x.shape[1])
    blm = np.atleast_2d(np.asarray(blm, dtype=np.complex128))
    if blm.shape[0] < x.shape[0]:
        raise ValueError("blm_tods: a sky model with polarization needs the beam of T, E, B")
    blm = blm[: x.shape[0]]
    mmax = next((m for m in range(lmax + 1) if (m + 1) * (lmax + 1) - m * (m + 1) // 2 == blm.shape[1]), None)
    if mmax is None:
        raise ValueError(f"blm_tods: the beam must have (mmax+1)(lmax+1) - mmax(mmax+1)/2 coefficients with the "
                         f"lmax of sky_alm ({lmax})")
    if kmax is not None and not 0 <= kmax <= mmax:
        raise ValueError(f"blm_tods: kmax must be in [0, mmax of the beam = {mmax}]")
    if model_bl is None:
        beam = _asymmetric(blm, lmax, mmax)
    else:
        beam = blm - _reference_blm(model_bl, lmax, mmax, x.shape[0], pol_angle)
    if kmax is None:  # the largest |b_lm| of each m (healpy order: m by m)
        start = np.array([m * (2 * lmax + 1 - m) // 2 for m in range(mmax + 1)])
        peak = np.maximum.reduceat(np.abs(beam).max(axis=0), start)
        kmax = int(np.flatnonzero(peak > epsilon * peak.max()).max(initial=0))
    beam = beam[:, : (kmax + 1) * (lmax + 1) - kmax * (kmax + 1) // 2]  # m <= kmax come first
    return ducc0.totalconvolve.Interpolator(sky=np.asarray(x, dtype=np.complex128), beam=beam, separate=False,
                                            lmax=lmax, kmax=kmax, epsilon=epsilon, nthreads=nthreads)


def _no_hwp(d):
    if d.hwp_angle is not None:
        raise NotImplementedError("blm_tods: with an HWP the beam needs the Mueller formalism (litebird_sim "
                                  "MuellerConvolver); without HWP only")


def _blm_interpol(inter, d, coordinates, nthreads):
    theta, phi, psi = d.pointing.angles(coordinates, nthreads)
    return inter.interpol(np.stack([theta, phi, psi], axis=1))[0]


def blm_tods(d, sky_alm, blm, coordinates, *, model_bl=None, kmax=None, epsilon=1e-6, nthreads=1):
    """The TOD of the asymmetries of the beam of one DetectorData, from its
    alm, for a fixed template (Template(fixed=True)): the T -> P leakage of
    the ellipticity that a fit cannot separate from Q/U.

    With `model_bl`, the window b_l of the circular beam the sky model was
    smoothed with (as healpy's gauss_beam: (lmax+1,) for T, or (lmax+1, >= 3)
    with the columns T, E, B), the sky convolved with the beam minus that
    circular beam: with the sky model, the sky through the true beam of each
    detector. Without, the sky convolved with the beam minus its own
    symmetric part (T at m = 0, the co-rotating polarization at m = 2): the
    symmetric parts that differ between the detectors are then left to
    fwhm_template, which follows them to order l^2 only. The ellipticity
    changes the symmetric part beyond that (order l^4, ~(e - 1)^2), and the
    rest biases the absolute calibration: use model_bl when the beams differ.

    d: a DetectorData with a Pointing as angles, without HWP (with an HWP
    the beam needs the Mueller formalism). sky_alm: the sky model (not
    convolved with any beam), alm (T or T, E, B, healpy order, mmax = lmax)
    in `coordinates`, those of the maps. blm: the beam as litebird_sim has
    it (e.g. gauss_beam_to_alm, or from GRASP), (1 or 3, nalm(lmax, mmax))
    with the lmax of sky_alm, in the frame of the detector: its psi as in
    litebird_sim, the polarization angle (pol_angle_rad) in the beam. The
    convolution is ducc0.totalconvolve, as in litebird_sim.

    kmax: the largest m of the beam in the convolution, whose cost grows
    as 2 kmax + 1; None (default): the largest m where the asymmetries
    have a coefficient above `epsilon` times their largest (an elliptical
    Gaussian of ellipticity 1 + e has m = 2k ~ e^k: a few m, while
    generate_gauss_beam_alms gives mmax = lmax). epsilon: the accuracy of
    the convolution, relative to the template.

    Returns the TOD, an array like d.tod.
    """
    _no_hwp(d)
    inter = _blm_interpolator(sky_alm, blm, kmax, epsilon, nthreads, model_bl, float(d.pol_angle_rad))
    return _blm_interpol(inter, d, coordinates, nthreads)


def blm_templates(data, sky_alm, blms, coordinates, *, model_bl=None, name="beam", kmax=None, epsilon=1e-6,
                  nthreads=1):
    """The beam asymmetries of all the DetectorData from the beams' alm, as a
    fixed template (see :func:`blm_tods`, also for kmax and epsilon).

    blms: {DetectorData.name: blm}, as litebird_sim's generate_gauss_beam_alms.
    model_bl: the window of the beam of the sky model (see blm_tods), the
    same for all the detectors.
    Returns (data, templates): the DetectorData as a list, with the TOD in
    their `templates` under `name`, and {name: Template(fixed=True)} for
    make_maps. One convolution (ducc0 Interpolator) per detector (and
    polarization angle, with model_bl), for all its DetectorData, freed
    before the next. All the TODs are computed here and
    kept in memory: with a generator, use blm_tods.
    """
    data = list(data)
    for d in data:
        if d.name not in blms:
            raise ValueError(f"blm_templates: no beam for detector {d.name!r}")
        if name in (d.templates or {}):
            raise ValueError(f"blm_templates: DetectorData.templates already has {name!r}")
        _no_hwp(d)
    tods = {}
    # with model_bl the reference beam has the polarization angle of the DetectorData
    key = [(d.name, None if model_bl is None else float(d.pol_angle_rad)) for d in data]
    for det, pol in dict.fromkeys(key):
        inter = _blm_interpolator(sky_alm, blms[det], kmax, epsilon, nthreads, model_bl, pol or 0.0)
        tods.update({i: _blm_interpol(inter, d, coordinates, nthreads) for i, d in enumerate(data) if key[i] == (det, pol)})
        del inter
    out = [dataclasses.replace(d, templates={**(d.templates or {}), name: tods[i]}) for i, d in enumerate(data)]
    return out, {name: Template(fixed=True)}


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
