"""Synthetic data shared by the tests (no litebird_sim needed): scans and
detectors with sky, dipole, gains, foregrounds and noise. The test files
import it as `builders` (pytest puts tests/ on sys.path)."""

import dataclasses

import healpy as hp
import numpy as np

from pysanepic import DetectorData, OneOverFNoise, Pointing


def random_sky(rng, nside):
    """White I/Q/U maps of rms 1e-4."""
    return rng.normal(0, 1e-4, (3, 12 * nside**2))


def iqu_tod(sky, pix, angle):
    """The TOD of the maps `sky` at the pixels `pix` for the polarization angles `angle`."""
    return sky[0, pix] + sky[1, pix] * np.cos(2 * angle) + sky[2, pix] * np.sin(2 * angle)


def scan(n, rng):
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


def four_detectors(n, fs, rng, sky, nside, noise=False):
    """Four detectors at different polarization angles, 1/f noise model, no HWP."""
    data = []
    for pol_angle in [0.0, np.pi / 4, np.pi / 8, 3 * np.pi / 8]:
        theta, phi, psi = scan(n, rng)
        p = hp.ang2pix(nside, theta, phi)
        a = psi + pol_angle
        tod = iqu_tod(sky, p, a)
        data.append(
            DetectorData(
                tod=tod, pointing=Pointing(theta=theta, phi=phi, psi=psi, coordinates="E"), sampling_rate_hz=fs,
                noise=OneOverFNoise(net_ukrts=40.0, fknee_hz=0.05, alpha=1.0, fmin_hz=1e-4), pol_angle_rad=pol_angle,
            )
        )
    return data


def calibration_data(nside, n, fs, per):
    """four_detectors with solar + orbital dipole (the orbital velocity turns
    once over the data, as in the year of scan) and gains drifting every `per`
    samples: (data, true gains, sky)."""
    rng = np.random.default_rng(5)
    sky = random_sky(rng, nside)
    solar = 3.36e-3 * np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    t = np.arange(n)
    orbital = 2.7e-4 * np.array([-np.sin(2 * np.pi * t / n), np.cos(2 * np.pi * t / n), 0 * t])
    data, true = [], []
    for d in four_detectors(n, fs, rng, sky, nside):
        th, ph = d.pointing.theta, d.pointing.phi
        nvec = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        dipole = solar @ nvec + np.sum(orbital * nvec, axis=0)
        g = 1 + 0.01 * rng.normal(size=n // per)
        true.append(g)
        data.append(dataclasses.replace(d, tod=g[t // per] * (d.tod + dipole), dipole=dipole))
    return data, true, sky


def bandpass_data(nside, n, fs, per, amp):
    """calibration_data plus a foreground seen as (1 + amp[i]) F by detector i."""
    data, true, sky = calibration_data(nside, n, fs, per)
    rng = np.random.default_rng(37)
    fg = hp.smoothing(rng.normal(0, 3e-4, (3, 12 * nside**2)), fwhm=np.radians(10))
    t = np.arange(n)
    out = []
    for i, (d, g) in enumerate(zip(data, true)):
        pix, angle = hp.ang2pix(nside, d.pointing.theta, d.pointing.phi), d.pointing.psi + d.pol_angle_rad
        f = iqu_tod(fg, pix, angle)
        out.append(dataclasses.replace(d, tod=d.tod + g[t // per] * (1 + amp[i]) * f, name=f"d{i}"))
    return out, true, sky, fg


def group_data(rng, nside, sky, fs, chunk, nchunk, own, mode, coupling):
    """Detectors observed at the same times (different pointings), each with
    its own noise plus coupling[i] times a common mode."""
    def draw(noise):
        w = noise.weights(chunk, fs)
        return np.concatenate([np.fft.irfft(np.fft.rfft(rng.normal(size=chunk)) / np.sqrt(w), n=chunk) for _ in range(nchunk)])

    n = chunk * nchunk
    c = draw(mode)
    dets = []
    for i, pa in enumerate(np.linspace(0, np.pi, len(coupling), endpoint=False)):
        th, ph, psi = scan(n, rng)
        a = psi + pa
        p = hp.ang2pix(nside, th, ph)
        tod = iqu_tod(sky, p, a) + draw(own) + coupling[i] * c
        dets.append(DetectorData(tod=tod, pointing=Pointing(theta=th, phi=ph, psi=psi, coordinates="E"),
                                 sampling_rate_hz=fs, noise=own, pol_angle_rad=pa, name=f"d{i}"))
    return dets


def noise_data(rng, nside, sky, fs, chunk, nchunk, shape, sigma, names=True):
    """Detectors with circulant noise of PSD 2 sigma^2 / fs * shape(f) per chunk,
    each split in stretches of 4 chunks sharing a name."""
    data = []
    for j, pa in enumerate((0.0, np.pi / 4, np.pi / 8)):
        n = nchunk * chunk
        th, ph, psi = scan(n, rng)
        a = psi + pa
        p = hp.ang2pix(nside, th, ph)
        f = np.fft.rfftfreq(chunk, 1 / fs)
        noise = np.concatenate([np.fft.irfft(np.fft.rfft(rng.normal(size=chunk)) * sigma * np.sqrt(shape(np.maximum(f, 1e-9))), n=chunk)
                                for _ in range(nchunk)])
        tod = iqu_tod(sky, p, a) + noise
        for s0 in range(0, n, 4 * chunk):
            sl = slice(s0, s0 + 4 * chunk)
            data.append(DetectorData(tod=tod[sl], pointing=Pointing(theta=th[sl], phi=ph[sl], psi=psi[sl], coordinates="E"),
                                     sampling_rate_hz=fs, noise=OneOverFNoise(net_ukrts=50.0), pol_angle_rad=pa,
                                     name=f"det{j}" if names else None))
    return data


# estimate_noise_data: three detectors with 1/f noise, ten DetectorData each,
# with a solar dipole and gains and/or TOD templates
NSIDE, FS, CHUNK, NCHUNK = 16, 10.0, 4000, 40

SIGMA = 50.0 * np.sqrt(FS) * 1e-6  # NET 50 uK sqrt(s)

PERIOD = 2 * CHUNK  # samples per amplitude of the template per period (2 per DetectorData)



def noise_shape(f):
    return (f**1.5 + 0.1**1.5) / (f**1.5 + 1e-3**1.5)


def estimate_noise_data(calibration, template):
    """noise_data (1/f noise, three detectors, ten DetectorData each) with a
    solar dipole and gains per chunk and/or one TOD template per detector."""
    rng = np.random.default_rng(71)
    sky = random_sky(rng, NSIDE)
    noise = OneOverFNoise(net_ukrts=50.0, fknee_hz=0.1, alpha=1.5, fmin_hz=1e-3)
    solar = 3.36e-3 * np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    amp = {"det0": 2e-3, "det1": -1e-3, "det2": 5e-4}
    out = []
    for d in noise_data(rng, NSIDE, sky, FS, CHUNK, NCHUNK, noise_shape, SIGMA):
        n = len(d.tod)
        th, ph = d.pointing.theta, d.pointing.phi
        signal = d.tod.copy()  # sky + noise
        kw = {"noise": noise}
        if template:
            b = np.convolve(rng.normal(size=n), np.ones(50) / 50, mode="same")
            signal += amp[d.name] * b
            kw["templates"] = {"thermal": b}
        if calibration:
            dipole = solar @ np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
            g = 1 + 0.01 * rng.normal(size=n // CHUNK)
            signal = g[np.arange(n) // CHUNK] * (signal + dipole)
            kw["dipole"] = dipole
        out.append(dataclasses.replace(d, tod=signal, **kw))
    return out
