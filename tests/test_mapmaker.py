"""make_maps on synthetic DetectorData (no litebird_sim needed).

Noiseless TOD of a random Galactic sky, observed with Ecliptic pointings, an
HWP, polarization efficiency 0.9 on one detector and chunks that do not divide
the data. The TOD is built with healpy's rotation, so pysanepic's numba
rotation kernel is checked against an independent implementation.
Run with: pytest tests/  (or: python tests/test_mapmaker.py)
"""

import dataclasses

import healpy as hp
import numpy as np
import pytest

from pysanepic import DetectorData, GLSParameters, Pointing, make_maps


def _scan(n, rng):
    """Spinning scan: (theta, phi, psi) in Ecliptic, covering most of the sky."""
    t = np.arange(n)
    lam = 2 * np.pi * t / n + rng.uniform(0, 2 * np.pi)
    spin = 2 * np.pi * t / 3000.0
    a = np.array([np.cos(lam), np.sin(lam), np.zeros(n)])
    z = np.array([np.zeros(n), np.zeros(n), np.ones(n)])
    e2 = np.cross(a.T, z.T).T
    b = np.cos(1.3) * a + np.sin(1.3) * (np.cos(spin) * z + np.sin(spin) * e2)
    theta = np.arccos(np.clip(b[2], -1, 1))
    phi = np.arctan2(b[1], b[0]) % (2 * np.pi)
    return theta, phi, (spin + lam) % (2 * np.pi) - np.pi


def test_noiseless_recovers_sky():
    nside, n, fs = 16, 120_000, 10.0
    rng = np.random.default_rng(0)
    sky = rng.normal(0, 1e-4, (3, 12 * nside**2))
    rot = hp.Rotator(coord=["E", "G"])

    data = []
    for pol_angle, gamma in [(0.0, 1.0), (np.pi / 4, 0.9)]:
        theta, phi, psi = _scan(n, rng)
        hwp = 2 * np.pi * 0.7 * np.arange(n) / fs
        # Signal in Galactic coordinates, rotated with healpy (independent of pysanepic)
        th_g, ph_g = rot(theta, phi)
        angle = psi + rot.angle_ref(theta, phi) + 2 * hwp - pol_angle
        p = hp.ang2pix(nside, th_g, ph_g)
        tod = sky[0, p] + gamma * (sky[1, p] * np.cos(2 * angle) + sky[2, p] * np.sin(2 * angle))
        data.append(
            DetectorData(
                tod=tod, pointing=Pointing(theta=theta, phi=phi, psi=psi, coordinates="E"), hwp_angle=hwp,
                sampling_rate_hz=fs, net_ukrts=40.0, pol_angle_rad=pol_angle, pol_efficiency=gamma,
            )
        )

    # white noise model (fknee = 0): no degenerate offsets, so the sky is recovered to
    # the precision of the cos/sin (~1e-8 in float32); tight tol because pixels with
    # 4 hits are ill-conditioned
    for dtype, rel in [(np.float32, 1e-7), (np.float64, 1e-9)]:
        params = GLSParameters(chunk_s=1700.0, tol=1e-24, angle_dtype=dtype)
        res = make_maps(data, nside, coordinates="G", params=params)
        assert res.converged
        seen = res.maps[1] != hp.UNSEEN
        assert seen.sum() > 0.5 * seen.size
        assert res.hit_map.sum() == 2 * n
        for i in range(3):
            assert np.abs(res.maps[i][seen] - sky[i][seen]).max() < rel * np.abs(sky).max(), (dtype, i)


def _four_detectors(n, fs, rng, sky, nside, noise=False):
    """Four detectors at different polarization angles, 1/f noise model, no HWP."""
    data = []
    for pol_angle in [0.0, np.pi / 4, np.pi / 8, 3 * np.pi / 8]:
        theta, phi, psi = _scan(n, rng)
        p = hp.ang2pix(nside, theta, phi)
        a = psi + pol_angle
        tod = sky[0, p] + sky[1, p] * np.cos(2 * a) + sky[2, p] * np.sin(2 * a)
        data.append(
            DetectorData(
                tod=tod, pointing=Pointing(theta=theta, phi=phi, psi=psi, coordinates="E"), sampling_rate_hz=fs,
                net_ukrts=40.0, fknee_hz=0.05, alpha=1.0, fmin_hz=1e-4, pol_angle_rad=pol_angle,
            )
        )
    return data


def test_padding_noiseless():
    nside, n, fs = 16, 120_000, 10.0
    rng = np.random.default_rng(0)
    sky = rng.normal(0, 1e-4, (3, 12 * nside**2))
    data = _four_detectors(n, fs, rng, sky, nside)
    # compare only pixels where Q/U are well conditioned (the default criterion)
    good = make_maps(data, nside, coordinates="E", params=GLSParameters(chunk_s=1200.0, maxiter=1)).maps[1] != hp.UNSEEN
    common = dict(chunk_s=1200.0, pad_s=300.0, tol=1e-24, min_pol_rcond=0.0)

    # zeros are exactly described by the margin offsets: exact recovery
    res = make_maps(data, nside, coordinates="E", params=GLSParameters(pad_fill="zeros", **common))
    assert res.hit_map.sum() == 4 * n  # padded samples are not hits
    err = res.maps[:, good] - sky[:, good]
    err[0] -= err[0].mean()  # I monopole unconstrained with 1/f weights
    assert np.abs(err).max() < 1e-4 * np.abs(sky).max()

    # SANEPIC's extrapolation leaves a small bias (unmodelled padding shape)
    res = make_maps(data, nside, coordinates="E", params=GLSParameters(pad_fill="extrapolate", **common))
    err = res.maps[:, good] - sky[:, good]
    err[0] -= err[0].mean()
    assert np.median(np.abs(err)) < 1e-2 * 1e-4


def test_pixel_input_and_generator():
    """Pixels instead of theta/phi, and a generator instead of a list: same maps."""
    import dataclasses

    nside, n, fs = 16, 40_000, 10.0
    rng = np.random.default_rng(2)
    sky = rng.normal(0, 1e-4, (3, 12 * nside**2))
    data = _four_detectors(n, fs, rng, sky, nside)
    common = dict(coordinates="E", params=GLSParameters(chunk_s=1000.0, maxiter=20))
    ref = make_maps(data, nside, **common)

    def with_pix(ordering, nest):
        for d in data:
            p = d.pointing
            pix = hp.ang2pix(nside, p.theta, p.phi, nest=nest).astype(np.int32)
            yield dataclasses.replace(
                d, pointing=Pointing(pix=pix, nside=nside, ordering=ordering, psi=p.psi, coordinates="E")
            )

    for ordering, nest in [("RING", False), ("NESTED", True)]:
        res = make_maps(with_pix(ordering, nest), nside, **common)
        assert np.array_equal(res.maps, ref.maps), ordering

    # pixels must be at the output nside and in the output coordinates
    pix = np.zeros(n, np.int32)
    for kw, match in [(dict(nside=8), "nside=8"), (dict(nside=nside, coordinates="G"), "coordinates")]:
        bad = Pointing(pix=pix, psi=data[0].pointing.psi, **{"coordinates": "E", **kw})
        with pytest.raises(ValueError, match=match):
            make_maps([dataclasses.replace(data[0], pointing=bad)], nside, **common)


def test_pointing_checks():
    psi, n = np.zeros(10), 10
    ok = dict(psi=psi, coordinates="E")
    for kw, match in [
        (dict(theta=psi, phi=psi, coordinates="X", psi=psi), "coordinates"),
        (dict(**ok), "either"),
        (dict(theta=psi, **ok), "either"),
        (dict(theta=psi, phi=psi[:5], **ok), "same length"),
        (dict(theta=psi, phi=psi, nside=16, **ok), "only for pix"),
        (dict(theta=psi, phi=psi, pix=np.zeros(n, int), nside=16, **ok), "not both"),
        (dict(pix=np.zeros(n), nside=16, **ok), "integer"),
        (dict(pix=np.zeros(n, int), **ok), "nside"),
        (dict(pix=np.zeros(n, int), nside=12, ordering="NESTED", **ok), "nside"),
        (dict(pix=np.zeros(n, int), nside=16, ordering="X", **ok), "ordering"),
        (dict(pix=np.full(n, 12 * 16**2), nside=16, **ok), "outside"),
        (dict(pix=np.zeros(5, int), nside=16, **ok), "same length"),
    ]:
        with pytest.raises(ValueError, match=match):
            Pointing(**kw)
    p = Pointing(pix=np.zeros(n, int), nside=16, **ok)
    with pytest.raises(ValueError, match="11 samples"):
        DetectorData(tod=np.zeros(11), pointing=p, sampling_rate_hz=1.0, net_ukrts=1.0)
    with pytest.raises(TypeError):
        DetectorData(tod=np.zeros(n), pointing=None, sampling_rate_hz=1.0, net_ukrts=1.0)


def test_contiguous_run_kernel():
    """O(n) formula for contiguous unpolarized runs == sum over all sample pairs."""
    from pysanepic.gls import _block_circulant

    rng = np.random.default_rng(1)
    ns = 64
    ip = np.zeros((1, ns), dtype=np.int64)
    ip[0, 10:30] = 1  # contiguous run in pixel 1
    ip[0, 40:44] = 2
    ip[0, 50] = 2  # pixel 2 is not contiguous
    c = rng.normal(size=(1, ns))
    c = c + c[:, (-np.arange(ns)) % ns]  # symmetric, like a circulant correlation
    fast = np.zeros((3, 6))
    _block_circulant(ip, c, np.zeros((1, 1)), np.zeros((1, 1)), False, fast, 3, np.zeros(0))
    fast = fast[:, 0]
    t = np.arange(ns)
    brute = [c[0, (t[ip[0] == q][:, None] - t[ip[0] == q][None, :]) % ns].sum() for q in range(3)]
    assert np.allclose(fast, brute)


def test_pixel_order():
    """Counting sort == stable argsort, also with empty and unhit pixels."""
    from pysanepic.gls import _pixel_order

    rng = np.random.default_rng(3)
    for npix, ns in [(1, 5), (50, 0), (50, 1000), (100_000, 20_000)]:
        ip = rng.integers(0, npix, ns).astype(np.int32)
        assert np.array_equal(_pixel_order(ip, npix), np.argsort(ip, kind="stable"))


def test_cos_sin_table():
    """Table + Taylor cos/sin == libm to 1e-15, also at the edges of [-pi, pi)
    (float32 rounding goes just outside) and after reducing unreduced angles;
    per-sample gamma."""
    from pysanepic.gls import _TRIG_COS, _TRIG_SIN, _cos_sin, _reduced

    rng = np.random.default_rng(4)
    edges = np.float32([-np.pi, np.pi, 0.0]).astype(float)
    a = np.concatenate([rng.uniform(-np.pi, np.pi, 100_000), edges, _reduced([7.0, -20.0, 1e4], float)])
    g = rng.uniform(0.5, 1.0, a.size)
    c, s = np.empty(a.size), np.empty(a.size)
    _cos_sin(a, g, c, s, _TRIG_COS, _TRIG_SIN)
    assert np.abs(c - g * np.cos(a)).max() < 1e-15 and np.abs(s - g * np.sin(a)).max() < 1e-15


def test_noise_models():
    """keshner and toast weights follow their formulas and agree for white noise;
    a tabulated PSD gives back the model it samples."""
    from pysanepic import one_over_f_weights, psd_weights

    ns, fs, sigma, fk, alpha, fmin = 4096, 10.0, 2e-4, 0.05, 1.3, 1e-3
    f = np.fft.rfftfreq(ns, 1 / fs)
    toast = one_over_f_weights(ns, fs, sigma, fk, alpha, fmin)
    keshner = one_over_f_weights(ns, fs, sigma, fk, alpha, fmin, model="keshner")
    assert np.allclose(1 / (toast * sigma**2), (f**alpha + fk**alpha) / (f**alpha + fmin**alpha))
    assert np.allclose(1 / (keshner * sigma**2), ((f**2 + fk**2) / (f**2 + fmin**2)) ** (alpha / 2))
    for model in ("toast", "keshner"):
        assert np.allclose(one_over_f_weights(ns, fs, sigma, 0.0, alpha, model=model), 1 / sigma**2)
    with pytest.raises(ValueError, match="noise model"):
        one_over_f_weights(ns, fs, sigma, fk, alpha, model="pink")

    # white noise: P = 2 sigma^2 / fs; a fine table of the keshner model
    assert np.allclose(psd_weights(ns, fs, np.array([1e-3, 5.0]), np.full(2, 2 * sigma**2 / fs)), 1 / sigma**2)
    ft = np.geomspace(f[1], f[-1], 2000)
    pt = 2 * sigma**2 / fs * ((ft**2 + fk**2) / (ft**2 + fmin**2)) ** (alpha / 2)
    assert np.allclose(psd_weights(ns, fs, ft, pt)[1:], keshner[1:], rtol=1e-4)


def test_noise_psd_in_make_maps():
    """DetectorData checks noise_model and noise_psd; a white tabulated PSD gives
    the same map as the white NET model."""
    nside, n, fs, net = 8, 40_000, 10.0, 50.0
    rng = np.random.default_rng(5)
    theta, phi, psi = _scan(n, rng)
    ptg = Pointing(theta=theta, phi=phi, psi=psi, coordinates="G")
    tod = rng.normal(0, net * np.sqrt(fs) * 1e-6, n)
    with pytest.raises(ValueError, match="noise_model"):
        DetectorData(tod=tod, pointing=ptg, sampling_rate_hz=fs, net_ukrts=net, noise_model="pink")
    with pytest.raises(ValueError, match="noise_psd"):
        DetectorData(tod=tod, pointing=ptg, sampling_rate_hz=fs, net_ukrts=net, noise_psd=([0.0, 1.0], [1.0, 1.0]))
    sigma = net * np.sqrt(fs) * 1e-6
    white = np.array([1e-3, 5.0]), np.full(2, 2 * sigma**2 / fs)
    params = GLSParameters(chunk_s=1000.0, tol=1e-14)
    maps = [
        make_maps([DetectorData(tod=tod, pointing=ptg, sampling_rate_hz=fs, net_ukrts=net, **kw)],
                  nside, coordinates="G", params=params).maps
        for kw in ({}, {"noise_psd": white})
    ]
    assert np.allclose(maps[0], maps[1], rtol=1e-8, atol=0)


def test_flags_exact_gls():
    """Flagged samples (pix < 0): the GLS solution equals the dense GLS on the
    unflagged samples only, with their exact covariance N_uu."""
    from pysanepic import GLS, one_over_f_weights

    rng = np.random.default_rng(7)
    npix, ns, nchunk, fs = 6, 128, 2, 10.0
    pix = rng.integers(0, npix, (nchunk, ns))
    psi = rng.uniform(0, np.pi, (nchunk, ns))
    tod = rng.normal(size=(nchunk, ns))
    w = one_over_f_weights(ns, fs, 1.0, 0.5, 1.0, 0.05)  # DC weight > 0: N invertible
    flags = rng.random((nchunk, ns)) < 0.1
    flags[0, 40:60] = True  # a gap
    one_each = -1 - np.arange(ns)  # a distinct negative label per sample: one unknown each
    g = GLS(np.where(flags, one_each, pix), psi, np.tile(w, (nchunk, 1)), min_pol_rcond=0.0, angle_dtype=np.float64)
    assert g.nflag == flags.sum()
    m, info = g.solve(np.where(flags, np.nan, tod), tol=1e-26, maxiter=500)
    assert info["converged"]

    lag = (np.arange(ns)[:, None] - np.arange(ns)[None, :]) % ns
    cov = np.linalg.inv(np.fft.irfft(w, n=ns)[lag])  # N of a chunk
    A, b = np.zeros((3 * npix, 3 * npix)), np.zeros(3 * npix)
    for k in range(nchunk):
        u = ~flags[k]
        P = np.zeros((ns, 3 * npix))
        P[np.arange(ns), pix[k]] = 1
        P[np.arange(ns), npix + pix[k]] = np.cos(2 * psi[k])
        P[np.arange(ns), 2 * npix + pix[k]] = np.sin(2 * psi[k])
        Ninv = np.linalg.inv(cov[np.ix_(u, u)])
        A += P[u].T @ Ninv @ P[u]
        b += P[u].T @ Ninv @ tod[k][u]
    ref = np.linalg.solve(A, b).reshape(3, npix)
    assert np.abs(m - ref).max() < 1e-11 * np.abs(ref).max()
    with pytest.raises(ValueError, match="flagged"):
        g.rhs(tod)


def test_flags_in_make_maps():
    """Glitches and a gap, flagged: noiseless maps recover the sky whatever the
    flagged TOD (NaN or huge), flagged samples are not hits; flags are checked."""
    nside, n, fs = 16, 120_000, 10.0
    rng = np.random.default_rng(3)
    sky = rng.normal(0, 1e-4, (3, 12 * nside**2))
    data = _four_detectors(n, fs, rng, sky, nside)
    params = GLSParameters(chunk_s=1200.0, tol=1e-24, min_pol_rcond=0.0, angle_dtype=np.float64)
    ref = make_maps(data, nside, coordinates="E", params=params)
    out = {}
    for bad in (np.nan, 1e3):
        flagged = []
        for d in data:
            f = np.random.default_rng(1).random(n) < 0.02
            f[5000:6000] = True
            flagged.append(f)
            # pad_s > 0 too: the padding must not use the flagged samples
        # the gap (100 s) is split with split_gap_s=60
        configs = [dict(pad_s=pad, flag_method=method, split_gap_s=split)
                   for pad in (0.0, 120.0) for method in ("marginalize", "offset") for split in (np.inf, 60.0)]
        res = [
            make_maps([dataclasses.replace(d, tod=np.where(f, bad, d.tod), flags=f) for d, f in zip(data, flagged)],
                      nside, coordinates="E", params=dataclasses.replace(params, **cfg))
            for cfg in configs
        ]
        out[bad] = res
    seen = ref.maps[1] != hp.UNSEEN
    for res in out.values():
        for r in res:
            assert np.abs(r.maps[:, seen] - sky[:, seen]).max() < 1e-8 * np.abs(sky).max()
    assert ref.hit_map.sum() - out[1e3][0].hit_map.sum() == sum(f.sum() for f in flagged)
    with pytest.raises(ValueError, match="flags"):
        dataclasses.replace(data[0], flags=np.zeros(n, int))
    with pytest.raises(ValueError, match="flag_method"):
        make_maps(data, nside, params=GLSParameters(flag_method="drop"))


if __name__ == "__main__":
    test_noiseless_recovers_sky()
    test_padding_noiseless()
    test_pixel_input_and_generator()
    test_pointing_checks()
    test_contiguous_run_kernel()
    test_pixel_order()
    print("ok")
