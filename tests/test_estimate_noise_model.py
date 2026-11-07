"""estimate_noise with calibration and templates: is the residual the noise?

make_maps fits d = g (P m + D + sum_k a_k B_k) + n. A residual consistent with
it is d / g - D - sum a B - P m (the noise in calibrated units). These tests
compare estimate_noise on the DetectorData of make_maps with estimate_noise
on the DetectorData cleaned by hand with the fitted gains and amplitudes and
the maps alone (the reference: what estimate_noise should compute).
"""

import dataclasses

import numpy as np
import pytest
from test_mapmaker import _noise_data

from pysanepic import (
    CalibrationParameters,
    GLSParameters,
    OneOverFNoise,
    Template,
    estimate_noise,
    make_maps,
)

NSIDE, FS, CHUNK, NCHUNK = 16, 10.0, 4000, 40
SIGMA = 50.0 * np.sqrt(FS) * 1e-6  # NET 50 uK sqrt(s)
PERIOD = 2 * CHUNK  # samples per amplitude of the template per period (2 per DetectorData)


def _shape(f):
    return (f**1.5 + 0.1**1.5) / (f**1.5 + 1e-3**1.5)


def _data(calibration, template):
    """_noise_data (1/f noise, three detectors, ten DetectorData each) with a
    solar dipole and gains per chunk and/or one TOD template per detector."""
    rng = np.random.default_rng(71)
    sky = rng.normal(0, 1e-4, (3, 12 * NSIDE**2))
    noise = OneOverFNoise(net_ukrts=50.0, fknee_hz=0.1, alpha=1.5, fmin_hz=1e-3)
    solar = 3.36e-3 * np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    amp = {"det0": 2e-3, "det1": -1e-3, "det2": 5e-4}
    out = []
    for d in _noise_data(rng, NSIDE, sky, FS, CHUNK, NCHUNK, _shape, SIGMA):
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


def _cleaned(data, result):
    """The DetectorData with the fitted gains, dipole and templates removed by
    hand, and the MapResult with the maps alone."""
    out = []
    for i, d in enumerate(data):
        tod = np.array(d.tod, dtype=float)
        t = np.arange(len(tod))
        if result.gains is not None:
            tod = tod / result.gains[i][t // CHUNK] - d.dipole
        for name, a in (result.templates or {}).items():
            a = a[i][t // PERIOD] if isinstance(a, list) else a[d.name]
            tod = tod - a * d.templates[name]
        out.append(dataclasses.replace(d, tod=tod, dipole=None, templates=None))
    maps_only = dataclasses.replace(result, gains=None, templates=None, calibration=None, template_definitions=None)
    return out, maps_only


@pytest.mark.parametrize("calibration, template", [(True, None), (False, "detector"), (True, "detector"), (True, "period")],
                         ids=["gains+dipole", "template", "both", "both-period"])
def test_estimate_noise_uses_fitted_model(calibration, template):
    data = _data(calibration, template)
    params = GLSParameters(chunk_s=CHUNK / FS, tol=1e-12)
    cal = CalibrationParameters(gain_period_s=CHUNK / FS, n_iter=3) if calibration else None
    tmpl = {"thermal": Template(period_s=None if template == "detector" else PERIOD / FS)} if template else None
    result = make_maps(data, NSIDE, coordinates="E", params=params, calibration=cal, templates=tmpl)
    kw = {"chunk_s": CHUNK / FS, "nbins": 30}
    ref = estimate_noise(*_cleaned(data, result), **kw)
    # the reference is the noise (in calibrated units): the PSD of the simulation above
    # the knee (below, few modes per bin and the errors of the fitted amplitudes)
    for e in ref.values():
        hi = e.freqs_hz > 0.4
        ratio = e.psd[hi] / (2 * SIGMA**2 / FS * _shape(e.freqs_hz[hi]))
        assert np.max(np.abs(ratio - 1)) < 0.1
    raw = estimate_noise(data, result, **kw)
    for name, e in raw.items():
        assert np.max(np.abs(e.psd / ref[name].psd - 1)) < 1e-8
    # the gains and amplitudes per period are those of the DetectorData of make_maps
    if calibration or template == "period":
        with pytest.raises(ValueError, match="same order"):
            estimate_noise(data[:-1], result, **kw)
