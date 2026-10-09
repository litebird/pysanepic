"""make_maps on synthetic DetectorData (no litebird_sim needed).

Noiseless TOD of a random Galactic sky, observed with Ecliptic pointings, an
HWP, polarization efficiency 0.9 on one detector and chunks that do not divide
the data. The TOD is built with healpy's rotation, so pysanepic's numba
rotation kernel is checked against an independent implementation.
Run with: pytest tests/  (or: python tests/test_mapmaker.py)
"""

import healpy as hp
import numpy as np

from pysanepic import DetectorData, make_maps


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
    # machine precision; tight tol because pixels with 4 hits are ill-conditioned
    res = make_maps(data, nside, coordinates="G", chunk_s=1700.0, tol=1e-24)
    assert res.converged
    seen = res.maps[1] != hp.UNSEEN
    assert seen.sum() > 0.5 * seen.size
    assert res.hit_map.sum() == 2 * n
    for i in range(3):
        assert np.abs(res.maps[i][seen] - sky[i][seen]).max() < 1e-9 * np.abs(sky).max(), i


if __name__ == "__main__":
    test_noiseless_recovers_sky()
    print("ok")
