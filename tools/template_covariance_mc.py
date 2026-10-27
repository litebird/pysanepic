"""Monte Carlo check of MapResult.template_covariance: the scatter of the
fitted template amplitudes over noise realizations against the covariance
computed by make_maps (template_covariance=True).

4 detectors at nside 16, 1/f noise (fknee 50 mHz) drawn as a stationary
process over the whole TOD, while N^-1 is applied on chunks: the covariance
is that of the GLS operator, and the scatter is the true one. Templates: a
foreground map with sum-zero amplitudes, a TOD per detector and a TOD with one
common amplitude; optionally with gain calibration (n_iter = 2).

Usage: python tools/template_covariance_mc.py [n_realizations] [--calibration]
"""

import sys
import time

import healpy as hp
import numpy as np

from pysanepic import CalibrationParameters, DetectorData, GLSParameters, OneOverFNoise, Pointing, Template, make_maps

nside, n, fs = 16, 120_000, 10.0
nreal = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 50
calibrate = "--calibration" in sys.argv
rng = np.random.default_rng(2026)
noise = OneOverFNoise(net_ukrts=40.0, fknee_hz=0.05, alpha=1.0, fmin_hz=1e-4)
npix = 12 * nside**2
sky = hp.smoothing(rng.normal(0, 1e-4, (3, npix)), fwhm=np.radians(8))
fg = hp.smoothing(rng.normal(0, 3e-4, (3, npix)), fwhm=np.radians(10))
t = np.arange(n)
per = 12_000  # gain period [samples]


def scan():
    """Spinning scan (theta, phi, psi) covering most of the sky."""
    lam = 2 * np.pi * t / n + rng.uniform(0, 2 * np.pi)
    spin = 2 * np.pi * t / 3000.0
    a = np.array([np.cos(lam), np.sin(lam), np.zeros(n)])
    e2 = np.cross(a.T, [0, 0, 1]).T
    b = np.cos(1.3) * a + np.sin(1.3) * (np.cos(spin) * np.array([[0], [0], [1]]) + np.sin(spin) * e2)
    return np.arccos(np.clip(b[2], -1, 1)), np.arctan2(b[1], b[0]) % (2 * np.pi), (spin + lam) % (2 * np.pi) - np.pi, b


dets = []
for i, pol in enumerate(np.radians([0, 90, 45, 135])):
    theta, phi, psi, vec = scan()
    pix, ang = hp.ang2pix(nside, theta, phi), psi + pol
    proj = lambda m: m[0, pix] + m[1, pix] * np.cos(2 * ang) + m[2, pix] * np.sin(2 * ang)  # noqa: E731
    dipole = 3.36e-3 * (np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])) @ vec
    dipole += np.sum(2.7e-4 * np.array([-np.sin(2 * np.pi * t / n), np.cos(2 * np.pi * t / n), 0 * t]) * vec, axis=0)
    tt = {"thermal": np.convolve(rng.normal(size=n), np.ones(50) / 50, mode="same") * 1e-1,
          "hk": rng.normal(size=n) * 1e-2}
    signal = proj(sky) + (1 + 0.02 * (i - 1.5)) * proj(fg) + 1e-3 * tt["thermal"] + 1e-3 * tt["hk"]
    gain = 1 + 0.01 * rng.normal(size=n // per)
    dets.append(dict(signal=signal, dipole=dipole, gain=gain[t // per], templates=tt, pointing=Pointing(
        theta=theta, phi=phi, psi=psi, coordinates="E"), pol=pol))


def draw():
    """Stationary noise over the whole TOD."""
    return np.fft.irfft(np.fft.rfft(rng.normal(size=n)) / np.sqrt(noise.weights(n, fs)), n=n)


def realization():
    out = []
    for i, d in enumerate(dets):
        tod = d["gain"] * (d["signal"] + d["dipole"]) + draw() if calibrate else d["signal"] + draw()
        out.append(DetectorData(tod=tod, pointing=d["pointing"], sampling_rate_hz=fs, noise=noise, pol_angle_rad=d["pol"],
                                name=f"d{i}", templates=d["templates"], dipole=d["dipole"] if calibrate else None))
    return out


kw = dict(coordinates="E", params=GLSParameters(chunk_s=1200.0, tol=1e-12),
          templates={"fg": Template(sky=fg, sum_zero=True), "thermal": Template(), "hk": Template(common=True)},
          calibration=CalibrationParameters(gain_period_s=per / fs, n_iter=2) if calibrate else None)


def amplitudes(r):
    return np.array([r.templates[k] if n is None else r.templates[k][n] for k, n in labels])


t0 = time.time()
first = make_maps(realization(), nside, template_covariance=True, **kw)
labels, cov = first.template_covariance["labels"], first.template_covariance["matrix"]
amps = [amplitudes(first)]
for _ in range(nreal - 1):
    amps.append(amplitudes(make_maps(realization(), nside, **kw)))
amps = np.array(amps)
print(f"{nreal} realizations{' with calibration' if calibrate else ''} in {time.time() - t0:.0f} s")

sig_pred, sig_mc = np.sqrt(np.diag(cov)), amps.std(axis=0, ddof=1)
err = sig_mc / np.sqrt(2 * (nreal - 1))  # statistical error of sig_mc
print(f"{'amplitude':16s} {'sigma pred':>11s} {'sigma MC':>11s} {'MC/pred':>8s}")
for (k, d), sp, sm, e in zip(labels, sig_pred, sig_mc, err):
    print(f"{k + ':' + str(d):16s} {sp:11.3e} {sm:11.3e} {sm / sp:8.3f} ± {e / sp:.3f}")
corr_pred = cov / np.outer(sig_pred, sig_pred)
corr_mc = np.corrcoef(amps.T)
print(f"correlations: max |MC - pred| = {np.abs(corr_mc - corr_pred).max():.3f} "
      f"(statistical error ~{1 / np.sqrt(nreal):.3f})")
