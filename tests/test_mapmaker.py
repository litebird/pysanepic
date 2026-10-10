"""make_maps on synthetic DetectorData (no litebird_sim needed).

Noiseless TOD of a random Galactic sky, observed with Ecliptic pointings, an
HWP, polarization efficiency 0.9 on one detector and chunks that do not divide
the data. The TOD is built with healpy's rotation, so pysanepic's numba
rotation kernel is checked against an independent implementation.
Run with: pytest tests/  (or: python tests/test_mapmaker.py)
"""

import healpy as hp
import numpy as np
import pytest

from pysanepic import DetectorData, GLSParameters, make_maps


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
                tod=tod, theta=theta, phi=phi, psi=psi, hwp_angle=hwp, coordinates="E",
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
                tod=tod, theta=theta, phi=phi, psi=psi, coordinates="E", sampling_rate_hz=fs,
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
    for ordering, nest in [("RING", False), ("NESTED", True)]:
        # coordinates=None: pixels and psi are in the coordinates of the output
        with_pix = (
            dataclasses.replace(
                d, theta=None, phi=None, coordinates=None, nside=nside, ordering=ordering,
                pix=hp.ang2pix(nside, d.theta, d.phi, nest=nest),
            )
            for d in data
        )
        res = make_maps(with_pix, nside, **common)
        assert np.array_equal(res.maps, ref.maps), ordering

    def bad(**kw):
        with pytest.raises(ValueError, match=kw.pop("match")):
            make_maps([dataclasses.replace(data[0], **kw)], nside, **common)

    bad(pix=np.zeros(n, int), nside=8, match="nside")
    bad(pix=np.full(n, 12 * nside**2), nside=nside, match="outside")
    bad(pix=np.zeros(n, int), nside=nside, coordinates="G", match="coordinates")
    bad(coordinates=None, match="required")


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
    _block_circulant(ip, c, np.zeros((1, 1)), np.zeros((1, 1)), False, fast)
    fast = fast[:, 0]
    t = np.arange(ns)
    brute = [c[0, (t[ip[0] == q][:, None] - t[ip[0] == q][None, :]) % ns].sum() for q in range(3)]
    assert np.allclose(fast, brute)


if __name__ == "__main__":
    test_noiseless_recovers_sky()
    test_padding_noiseless()
    test_pixel_input_and_generator()
    test_contiguous_run_kernel()
    print("ok")
