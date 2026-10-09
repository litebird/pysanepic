"""Read inputs/outputs in the format of the original C++ SANEPIC.

Only used to validate pysanepic against the C++ code: it reproduces exactly
the pointing, pixelization and noise-spectrum handling of `sanepic.cc`, and
returns plain arrays that `pysanepic.gls` understands.
"""

from pathlib import Path

import healpy as hp
import numpy as np


def pointingshift(x, y, thetap, phip, psip):
    """Vectorized port of `pointingshift()` in sanepic.cc.

    Moves the boresight (thetap, phip, psip) by the focal-plane offset (x, y)
    and returns the detector (theta, phi, psi), with SANEPIC's psi convention.
    """
    th_ = np.arcsin(np.sqrt(x * x + y * y))
    ph_ = np.arctan2(y, x)
    st, ct = np.sin(thetap), np.cos(thetap)
    theta = np.arccos(st * np.sin(th_) * np.cos(ph_ + psip) + ct * np.cos(th_))
    phi = (
        np.arctan2(
            -np.sin(th_) * np.sin(ph_ + psip),
            -ct * np.sin(th_) * np.cos(ph_ + psip) + st * np.cos(th_),
        )
        + phip
    )

    drp0 = np.cos(psip) * np.cos(th_)
    drp1 = np.sin(psip) * np.cos(th_)
    drp2 = -np.sin(th_) * np.cos(ph_)
    dr0 = -ct * drp0 + st * drp2
    dr1 = -drp1
    dr2 = st * drp0 + ct * drp2
    aaa = (dr0 * -np.sin(phi - phip) + dr1 * np.cos(phi - phip)) / np.sqrt(
        dr0**2 + dr1**2 + dr2**2
    )
    psi = np.arccos(np.clip(aaa, -1.0, 1.0))
    psi = np.where(dr2 < 0, -psi, psi)
    psi = np.fmod(psi + np.pi / 2 + 4 * np.pi, 2 * np.pi) - np.pi
    return theta, phi, psi


def read_bolometer_info(path, names):
    """Return (offx, offy, psi) in radians for each detector, like sanepic.cc."""
    table = {}
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if len(f) == 7:
            # num pnum lel xel angle telescope name  (angles in degrees)
            table[f[6]] = (float(f[3]), float(f[2]), float(f[4]))
    off = np.deg2rad(np.array([table[n] for n in names]))
    # sanepic.cc: offx = offsets[1] (= lel), offy = offsets[0] (= xel)
    return off[:, 1], off[:, 0], off[:, 2]


def noise_weights(spn_file, nssp, ns):
    """Inverse-noise weights on the rfft grid, scaled so that
    irfft(rfft(t) * w) is exactly SANEPIC's N^-1 t (FFTW is unnormalized and
    SANEPIC divides the spectrum by ns^2)."""
    spn = np.fromfile(spn_file, dtype=np.float64, count=nssp)
    ii = np.arange(ns // 2 + 1)
    j = np.floor(ii * float(nssp) / ns).astype(np.int64)
    w = spn[j]
    w[j == 0] = spn[1] * 1e-6
    w[0] = spn[1] * 1e-6
    return w / ns


def read_inputs(
    data_dir,
    pointing_prefix,
    bolometer_info,
    detectors,
    noise_prefix,
    nssp,
    ns,
    nseg,
    nside,
    data_ext="",
    pointing_ext="",
):
    """Read a SANEPIC run (no padding, no HWP, contiguous segments).

    Returns a dict of arrays with one row per (detector, segment) chunk:
    pix (nchunk, ns) RING pixels, psi (nchunk, ns), tod (nchunk, ns),
    weights (nchunk, ns//2+1).
    """
    n = ns * nseg
    ra = np.fromfile(f"{pointing_prefix}ra{pointing_ext}.bi", count=n)
    dec = np.fromfile(f"{pointing_prefix}dec{pointing_ext}.bi", count=n)
    psib = np.fromfile(f"{pointing_prefix}psi{pointing_ext}.bi", count=n)
    offx, offy, psidet = read_bolometer_info(bolometer_info, detectors)

    pix, psi, tod, w = [], [], [], []
    for k, det in enumerate(detectors):
        theta, phi, psi_o = pointingshift(offx[k], offy[k], np.pi / 2 - dec, ra, psib)
        p = hp.ang2pix(nside, theta, np.fmod(phi + 2 * np.pi, 2 * np.pi))
        d = np.fromfile(f"{data_dir}/data_{det}{data_ext}.bi", count=n)
        wk = noise_weights(f"{noise_prefix}{det}", nssp, ns)
        pix.append(p.reshape(nseg, ns))
        psi.append((psi_o + psidet[k]).reshape(nseg, ns))
        tod.append(d.reshape(nseg, ns))
        w.append(np.broadcast_to(wk, (nseg, wk.size)))
    return {
        "pix": np.concatenate(pix),
        "psi": np.concatenate(psi),
        "tod": np.concatenate(tod),
        "weights": np.concatenate(w),
    }


def read_map(path, nside):
    """Read a full-sky map written by SANEPIC (12*nside^2 doubles, 0 = unseen)."""
    return np.fromfile(path, dtype=np.float64, count=12 * nside * nside)
