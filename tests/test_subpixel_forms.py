"""The two forms of the sub-pixel terms of I: GLS (the components of the GLS
solution, the model of its projection) and MapResult (I the pixel mean,
gradient and Hessian as derivatives)."""

import healpy as hp
import numpy as np
import pytest

from pysanepic.gls import _project
from pysanepic.mapmaker import (
    _pixel_mean_shift,
    _subpixel_gls,
    _subpixel_result,
    _subpixel_tod,
)

NSIDE = 4
NPIX = 12 * NSIDE**2


def _gls_maps(rng, order):
    """Random GLS maps (I/Q/U + sub-pixel components), with pixels where only
    I, or I and the gradient, are solved."""
    m = rng.normal(size=(3 + (0, 2, 5)[order], NPIX))
    m[3:, :3] = hp.UNSEEN  # no gradient
    m[5:, 3:6] = hp.UNSEEN  # no Hessian
    return m


@pytest.mark.parametrize("order", [1, 2])
def test_round_trip_and_tod(order):
    """GLS -> MapResult -> GLS gives the GLS maps back; the TOD of the GLS
    form (_subpixel_tod) is that of the projection of GLS; the Hessian of
    MapResult is the second derivatives (2 s2, s3, 2 s4) and I the pixel
    mean."""
    rng = np.random.default_rng(order)
    m = _gls_maps(rng, order)
    maps, gradient, hessian = _subpixel_result(m.copy(), 3, order, NSIDE, 1)
    assert (hessian is None) == (order == 1)
    np.testing.assert_array_equal(gradient, m[3:5])
    if order == 2:
        np.testing.assert_array_equal(hessian[:, 6:], m[5:, 6:] * np.array([2.0, 1.0, 2.0])[:, None])
        assert np.all(hessian[:, :6] == hp.UNSEEN)
    np.testing.assert_allclose(maps[0] - m[0], _pixel_mean_shift(NSIDE, gradient, hessian, 1), rtol=0, atol=1e-15)
    np.testing.assert_array_equal(maps[1:], m[1:3])
    back, sub = _subpixel_gls(maps, gradient, hessian, NSIDE, 1)
    np.testing.assert_allclose(back, m[:3], rtol=0, atol=1e-14)
    np.testing.assert_array_equal(sub, m[3:])
    # the TOD: _subpixel_tod on the GLS form = the projection of GLS (without I/Q/U)
    n = 2000
    pix = rng.integers(0, NPIX, n)
    gx, gy = rng.uniform(-0.6, 0.6, (2, n))
    out = np.empty(n)
    gls_m = np.where(m == hp.UNSEEN, 0.0, m)
    gls_m[:3] = 0.0
    _project(pix.astype(np.int32), np.ascontiguousarray(gls_m.T), np.zeros(n), np.zeros(n), True, gx, gy, order,
             np.zeros(0), out)
    np.testing.assert_allclose(_subpixel_tod(sub, pix, gx, gy), out, rtol=0, atol=1e-13)


def test_no_subpixel_terms():
    """subpixel_order 0: the maps as they are, no gradient or Hessian."""
    m = np.random.default_rng(0).normal(size=(3, NPIX))
    maps, gradient, hessian = _subpixel_result(m.copy(), 3, 0, NSIDE, 1)
    np.testing.assert_array_equal(maps, m)
    assert gradient is None and hessian is None
