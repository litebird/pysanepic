"""Write a small simulation in the input format of the C++ SANEPIC.

Usage: python simulate_sanepic_inputs.py OUTDIR [--fknee 0.05] [--noiseless]

Produces: pointings_{ra,dec,psi}.bi, data_<det>.bi, iSpf_<det>,
bolometer_info.txt and truth.npz (input sky in RING, I/Q/U).
"""

import argparse
from pathlib import Path

import healpy as hp
import numpy as np

from pysanepic.sanepic_io import pointingshift

p = argparse.ArgumentParser()
p.add_argument("outdir", type=Path)
p.add_argument("--nside", type=int, default=16)
p.add_argument("--ns", type=int, default=4096, help="segment length")
p.add_argument("--nseg", type=int, default=16)
p.add_argument("--fsamp", type=float, default=19.0)
p.add_argument("--fknee", type=float, default=0.05)
p.add_argument("--alpha", type=float, default=1.0)
p.add_argument("--sigma", type=float, default=0.1, help="white noise rms per sample")
p.add_argument("--noiseless", action="store_true")
p.add_argument("--seed", type=int, default=1234)
args = p.parse_args()

rng = np.random.default_rng(args.seed)
out = args.outdir
out.mkdir(parents=True, exist_ok=True)
n = args.ns * args.nseg
t = np.arange(n)

# Scan: spin axis moving along the equator once per run, boresight at 80 deg
# from it, 128 spins in total.
lam = 2 * np.pi * t / n
spin = 2 * np.pi * t / (n / 128)
a = np.array([np.cos(lam), np.sin(lam), np.zeros(n)])
e1 = np.array([np.zeros(n), np.zeros(n), np.ones(n)])
e2 = np.cross(a.T, e1.T).T
opening = np.deg2rad(80)
b = np.cos(opening) * a + np.sin(opening) * (np.cos(spin) * e1 + np.sin(spin) * e2)
ra = np.fmod(np.arctan2(b[1], b[0]) + 2 * np.pi, 2 * np.pi)
dec = np.arcsin(np.clip(b[2], -1, 1))
psi_b = np.fmod(spin + 0.3 * lam, 2 * np.pi) - np.pi
for name, arr in (("ra", ra), ("dec", dec), ("psi", psi_b)):
    arr.astype(np.float64).tofile(out / f"pointings_{name}.bi")

# Two co-pointed detectors with polarization angles 0 and 45 deg.
dets = {"det0": 0.0, "det1": 45.0}
with open(out / "bolometer_info.txt", "w") as f:
    for i, (name, ang) in enumerate(dets.items()):
        f.write(f"{i} {i} 0.0 0.0 {ang} LFT {name}\n")

npix = 12 * args.nside**2
sky = np.array([rng.normal(0, 1, npix), rng.normal(0, 0.1, npix), rng.normal(0, 0.1, npix)])

freq = np.fft.rfftfreq(n, 1 / args.fsamp)
shape = np.ones_like(freq)
shape[1:] += (args.fknee / freq[1:]) ** args.alpha
shape[0] = shape[1]

theta, phi, psi_o = pointingshift(0.0, 0.0, np.pi / 2 - dec, ra, psi_b)
pix = hp.ang2pix(args.nside, theta, np.fmod(phi + 2 * np.pi, 2 * np.pi))
for name, ang in dets.items():
    psi = psi_o + np.deg2rad(ang)
    tod = sky[0, pix] + sky[1, pix] * np.cos(2 * psi) + sky[2, pix] * np.sin(2 * psi)
    if not args.noiseless:
        white = rng.normal(0, args.sigma, n)
        tod += np.fft.irfft(np.fft.rfft(white) * np.sqrt(shape), n=n)
    tod.tofile(out / f"data_{name}.bi")

    # Inverse power spectrum on SANEPIC's full FFT grid of length ns (nssp = ns).
    k = np.arange(args.ns)
    f = np.minimum(k, args.ns - k) * args.fsamp / args.ns
    spec = np.ones(args.ns)
    spec[1:] += (args.fknee / f[1:]) ** args.alpha
    ispf = 1.0 / (args.sigma**2 * spec)
    ispf[0] = 0.0
    ispf.tofile(out / f"iSpf_{name}")

np.savez(out / "truth.npz", sky=sky, nside=args.nside, ns=args.ns, nseg=args.nseg, dets=list(dets))
print(f"wrote {n} samples x {len(dets)} detectors to {out}")
