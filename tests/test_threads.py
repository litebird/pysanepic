"""Thread control: OMP_NUM_THREADS / nthreads, same result for any thread count."""

import healpy as hp
import numba
import numpy as np
import pytest

from pysanepic import DetectorData, GLSParameters, OneOverFNoise, Pointing, TabulatedNoise, make_maps, resolve_nthreads
from pysanepic.gls import numba_threads
from test_mapmaker import _scan


@pytest.mark.parametrize("private", [True, False])
@pytest.mark.parametrize("nthreads", [2, 4])
def test_chunk_workers_repeat_with_different_completion_order(monkeypatch, private, nthreads):
    """Repeated RHS/A products must not depend on which worker finishes first.

    Exercise maps, shared/local templates and disjoint high-order ACT cuts,
    including both private-map and pixel-range accumulation paths.
    """
    import threading
    import time

    import pysanepic.gls as kernels
    from pysanepic import GLS

    rng = np.random.default_rng(91)
    nc, ns, npix = 12, 128, 6
    pix = rng.integers(npix, size=(nc, ns))
    pix[:, 20:40] = -1
    tod = rng.normal(size=(nc, ns)) * np.logspace(-3, 3, nc)[:, None]
    templates = rng.normal(size=(nc, ns, 2))
    monkeypatch.setattr(kernels, "_PRIVATE_MAPS_BYTES", 4e9 if private else 0)
    g = GLS(pix, None, np.ones((nc, ns // 2 + 1)), nthreads=nthreads,
            flag_method="act", flag_sampling_rate_hz=10., act_ncoeff=19,
            templates=[(k, templates[k, :, 0], 0) for k in range(nc)],
            shared_templates=[(k, templates[k, :, 1], 0) for k in range(nc)])
    v = rng.normal(size=g._nsky + g.nflag + g.namp + g._nsh_loc)
    # Compile and allocate before controlling the completion order.
    g._apply(tod=tod)
    g._apply(v)
    original = g._chunk_tod
    delayed = 0
    barrier = threading.Barrier(nthreads)

    def chunk_tod(k, *args, **kwargs):
        if k < nthreads:
            barrier.wait(timeout=10)
            if k == delayed:
                time.sleep(.025)
        return original(k, *args, **kwargs)

    monkeypatch.setattr(g, "_chunk_tod", chunk_tod)
    for kw in ({"tod": tod}, {"v": v}):
        delayed = 0
        barrier = threading.Barrier(nthreads)
        ref = g._apply(**kw)
        delayed = 1
        barrier = threading.Barrier(nthreads)
        repeated = g._apply(**kw)
        np.testing.assert_array_equal(repeated, ref)


def test_resolve_nthreads(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert resolve_nthreads() == 1  # default: one thread per process, as litebird_sim
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert resolve_nthreads() == 3
    assert resolve_nthreads(2) == 2  # explicit value wins


def test_numba_threads_restores():
    before = numba.get_num_threads()
    with numba_threads(1):
        assert numba.get_num_threads() == 1
    assert numba.get_num_threads() == before


def test_same_result_any_thread_count():
    nside, n, fs = 16, 60_000, 10.0
    rng = np.random.default_rng(0)
    sky = rng.normal(0, 1e-4, (3, 12 * nside**2))
    data = []
    for pol_angle in [0.0, np.pi / 4, np.pi / 8]:
        theta, phi, psi = _scan(n, rng)
        p = hp.ang2pix(nside, theta, phi)
        a = psi + pol_angle
        tod = sky[0, p] + sky[1, p] * np.cos(2 * a) + sky[2, p] * np.sin(2 * a) + rng.normal(0, 1e-4, n)
        data.append(
            DetectorData(
                tod=tod, pointing=Pointing(theta=theta, phi=phi, psi=psi, coordinates="E"), sampling_rate_hz=fs,
                noise=OneOverFNoise(net_ukrts=40.0, fknee_hz=0.05, alpha=1.0, fmin_hz=1e-4), pol_angle_rad=pol_angle,
            )
        )
    # coordinates="G" exercises the numba rotation kernel and ducc0 with threads
    one = make_maps(data, nside, coordinates="G", params=GLSParameters(chunk_s=1200.0), nthreads=1)
    four = make_maps(data, nside, coordinates="G", params=GLSParameters(chunk_s=1200.0), nthreads=4)
    assert one.iterations == four.iterations
    seen = one.maps != hp.UNSEEN
    assert np.array_equal(four.maps != hp.UNSEEN, seen)
    assert np.allclose(one.maps[seen], four.maps[seen], rtol=0, atol=1e-10 * np.abs(one.maps[seen]).max())
