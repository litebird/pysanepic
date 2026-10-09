"""Thread control: OMP_NUM_THREADS / nthreads, same result for any thread count."""

import healpy as hp
import numba
import numpy as np

from pysanepic import DetectorData, make_maps, resolve_nthreads
from pysanepic.gls import numba_threads
from test_mapmaker import _scan


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
                tod=tod, theta=theta, phi=phi, psi=psi, coordinates="E", sampling_rate_hz=fs,
                net_ukrts=40.0, fknee_hz=0.05, alpha=1.0, fmin_hz=1e-4, pol_angle_rad=pol_angle,
            )
        )
    # coordinates="G" exercises the numba rotation kernel and ducc0 with threads
    one = make_maps(data, nside, coordinates="G", chunk_s=1200.0, nthreads=1)
    four = make_maps(data, nside, coordinates="G", chunk_s=1200.0, nthreads=4)
    assert one.iterations == four.iterations
    seen = one.maps != hp.UNSEEN
    assert np.array_equal(four.maps != hp.UNSEEN, seen)
    assert np.allclose(one.maps[seen], four.maps[seen], rtol=0, atol=1e-10 * np.abs(one.maps[seen]).max())
