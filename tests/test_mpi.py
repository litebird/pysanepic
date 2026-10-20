"""MPI run must give the same maps as the serial run.

Run with: mpirun -n 4 python tests/test_mpi.py  (under pytest it runs on 1 rank)
"""

import tempfile

import numpy as np
import pytest
from mpi4py import MPI

from pysanepic import GLS, one_over_f_weights
from pysanepic.sanepic_io import read_inputs
from test_vs_sanepic import DETS, NS, NSEG, NSIDE, _simulate


def test_mpi_matches_serial():
    comm = MPI.COMM_WORLD
    d = comm.bcast(tempfile.mkdtemp() if comm.rank == 0 else None)
    if comm.rank == 0:
        _simulate(d)
    comm.Barrier()

    inp = read_inputs(d, f"{d}/pointings_", f"{d}/bolometer_info.txt", DETS, f"{d}/iSpf_", NS, NS, NSEG, NSIDE)
    mine = slice(comm.rank, None, comm.size)  # round-robin chunks; a rank may get none
    g = GLS(inp["pix"][mine], inp["psi"][mine], inp["weights"][mine], comm=comm)
    m, info = g.solve(inp["tod"][mine])
    # each rank keeps the PCG state only on its own pixels
    assert g.precond.shape[0] == g.nloc and comm.allreduce(g.nloc) == g.nobs
    mask = g.full(g.mask.astype(np.uint8)).astype(bool)  # collective

    if comm.rank == 0:
        g.to_healpix(m, NSIDE)  # not collective after solve: must not hang
        gs = GLS(inp["pix"], inp["psi"], inp["weights"])
        ms, infos = gs.solve(inp["tod"])
        assert np.array_equal(mask, gs.mask)
        rel = np.abs(m - ms).max() / np.abs(ms).max()
        print(f"ranks={comm.size} iters mpi/serial={info['iterations']}/{infos['iterations']} max rel diff={rel:.1e}")
        assert rel < 1e-8


def test_make_maps_mpi_padding():
    """make_maps with padding: data split across ranks gives the serial result."""
    import healpy as hp

    from pysanepic import GLSParameters, make_maps
    from test_mapmaker import _four_detectors

    comm = MPI.COMM_WORLD
    nside, n, fs = 16, 60_000, 10.0
    rng = np.random.default_rng(0)
    sky = rng.normal(0, 1e-4, (3, 12 * nside**2))
    data = _four_detectors(n, fs, rng, sky, nside)
    kw = dict(coordinates="E", params=GLSParameters(chunk_s=1200.0, pad_s=300.0, tol=1e-20))
    res = make_maps(data[comm.rank :: comm.size], nside, comm=comm, **kw)
    if comm.rank == 0:
        ref = make_maps(data, nside, **kw)
        assert np.array_equal(res.hit_map, ref.hit_map)
        seen = ref.maps != hp.UNSEEN
        assert np.array_equal(res.maps != hp.UNSEEN, seen)
        assert np.abs(res.maps[seen] - ref.maps[seen]).max() < 1e-8 * np.abs(ref.maps[seen]).max()


def test_calibration_mpi():
    """Calibration with the detectors split across ranks gives the serial gains
    (amplitudes local to each rank; default min_pol_rcond, which drops the
    templates in pixels without Q/U, a collective step)."""
    from pysanepic import CalibrationParameters, GLSParameters, make_maps
    from test_mapmaker import _calibration_data

    comm = MPI.COMM_WORLD
    data, _, _ = _calibration_data(16, 60_000, 10.0, 12_000)
    data = data[:3]  # with 4 ranks, one has no data (and no templates)
    kw = dict(coordinates="E", params=GLSParameters(chunk_s=1200.0, tol=1e-20),
              calibration=CalibrationParameters(gain_period_s=1200.0, n_iter=2))
    gains = comm.gather(make_maps(data[comm.rank :: comm.size], 16, comm=comm, **kw).gains)
    if comm.rank == 0:
        ref = make_maps(data, 16, **kw).gains
        for r in range(comm.size):
            for g, g0 in zip(gains[r], ref[r :: comm.size]):
                assert np.abs(g - g0).max() < 1e-9


def test_bandpass_mpi():
    """Bandpass amplitudes shared by the two halves of each detector, on
    different ranks: the serial result, on every rank."""
    import dataclasses

    from pysanepic import GLSParameters, make_maps
    from test_mapmaker import _bandpass_data

    comm = MPI.COMM_WORLD
    n, per = 60_000, 12_000
    data, true, _, fg = _bandpass_data(16, n, 10.0, per, np.array([0.05, -0.02, 0.03, 0.01]))
    t = np.arange(n)
    halves = []
    for d, g in zip(data, true):
        flat = d.tod / g[t // per] - d.dipole
        for sl in (slice(0, n // 2), slice(n // 2, n)):
            p = d.pointing
            halves.append(dataclasses.replace(d, tod=flat[sl], dipole=None, pointing=dataclasses.replace(
                p, theta=p.theta[sl], phi=p.phi[sl], psi=p.psi[sl])))
    kw = dict(coordinates="E", params=GLSParameters(chunk_s=1200.0, tol=1e-20), bandpass_templates={"fg": fg})
    res = make_maps(halves[comm.rank :: comm.size], 16, comm=comm, **kw).bandpass["fg"]
    ref = make_maps(halves, 16, **kw).bandpass["fg"]
    assert set(res) == set(ref) and all(abs(res[k] - ref[k]) < 1e-9 for k in ref)


def test_more_ranks_than_pixels():
    """With MPI some ranks may own no pixels at all: same result as serial."""
    comm = MPI.COMM_WORLD
    rng = np.random.default_rng(0)
    ns, nchunk = 256, 2 * comm.size
    pix = rng.integers(0, 3, (nchunk, ns))  # 3 pixels: with 4+ ranks some own none
    psi = rng.uniform(0, np.pi, (nchunk, ns))
    w = np.tile(one_over_f_weights(ns, 10.0, 1.0, 0.0, 1.0), (nchunk, 1))
    tod = rng.normal(size=(nchunk, ns))
    mine = slice(comm.rank, None, comm.size)
    g = GLS(pix[mine], psi[mine], w[mine], comm=comm, min_pol_rcond=0.0)
    m, _ = g.solve(tod[mine], tol=1e-20)
    if comm.rank == 0:
        ms, _ = GLS(pix, psi, w, min_pol_rcond=0.0).solve(tod, tol=1e-20)
        assert np.allclose(m, ms, rtol=1e-10, atol=0)


def test_mpi_count_check():
    from pysanepic.gls import _check_mpi_count

    _check_mpi_count(12 * 4096**2 * 7)  # nside 4096: fine
    with pytest.raises(ValueError, match="nside 4096"):
        _check_mpi_count(12 * 8192**2 * 7)


def test_flags_mpi():
    """Flagged samples spread over the ranks: same maps as serial."""
    comm = MPI.COMM_WORLD
    rng = np.random.default_rng(2)
    ns, nchunk, npix = 512, 3 * comm.size, 40
    pix = rng.integers(0, npix, (nchunk, ns))
    psi = rng.uniform(0, np.pi, (nchunk, ns))
    w = np.tile(one_over_f_weights(ns, 10.0, 1.0, 0.5, 1.0, 0.01), (nchunk, 1))
    tod = rng.normal(size=(nchunk, ns))
    flags = rng.random((nchunk, ns)) < 0.05
    pix[flags] = (-1 - np.arange(ns))[None].repeat(nchunk, 0)[flags]  # one unknown per sample
    mine = slice(comm.rank, None, comm.size)
    m, _ = GLS(pix[mine], psi[mine], w[mine], comm=comm, min_pol_rcond=0.0).solve(tod[mine], tol=1e-24)
    if comm.rank == 0:
        ms, _ = GLS(pix, psi, w, min_pol_rcond=0.0).solve(tod, tol=1e-24)
        assert np.allclose(m, ms, rtol=1e-9, atol=1e-12 * np.abs(ms).max())



def test_estimate_noise_mpi():
    """The spectrum of a detector whose DetectorData are spread over the ranks
    is the same as in serial."""
    from pysanepic import estimate_noise
    from pysanepic.mapmaker import MapResult
    from test_mapmaker import _noise_data

    comm = MPI.COMM_WORLD
    nside, fs, chunk = 8, 10.0, 2000
    rng = np.random.default_rng(16)
    sky = rng.normal(0, 1e-4, (3, 12 * nside**2))
    data = _noise_data(rng, nside, sky, fs, chunk, 8, lambda f: 1 + (0.1 / f) ** 1.5, 1e-4)
    truth = MapResult(maps=sky, hit_map=None, nside=nside, coordinates="E", iterations=0, converged=True, residuals=None)
    mine = estimate_noise(data[comm.rank :: comm.size], truth, chunk_s=chunk / fs, comm=comm)
    if comm.rank == 0:
        ref = estimate_noise(data, truth, chunk_s=chunk / fs)
        assert sorted(mine) == sorted(ref)
        for k in ref:
            assert np.allclose(mine[k].psd, ref[k].psd, rtol=1e-12) and np.allclose(mine[k].freqs_hz, ref[k].freqs_hz)



def test_detector_groups_mpi():
    """DetectorGroups on different ranks (each group on one rank) = serial."""
    import healpy as hp

    from pysanepic import CommonMode, DetectorGroup, GLSParameters, OneOverFNoise, make_maps
    from test_mapmaker import _group_data

    comm = MPI.COMM_WORLD
    nside, fs, chunk = 8, 10.0, 2000
    rng = np.random.default_rng(19)
    sky = rng.normal(0, 1e-4, (3, 12 * nside**2))
    own = OneOverFNoise(net_ukrts=50.0, fknee_hz=0.05, alpha=1.0, fmin_hz=1e-3)
    mode = OneOverFNoise(net_ukrts=20.0, fknee_hz=0.5, alpha=2.0, fmin_hz=1e-3)
    groups = []
    for _ in range(3):
        coupling = rng.uniform(0.8, 1.2, 3)
        dets = _group_data(rng, nside, sky, fs, chunk, 3, own, mode, coupling)
        groups.append(DetectorGroup(detectors=dets, modes=[CommonMode(coupling=coupling, noise=mode)]))
    for nc in (0, 2):  # also with the two-level preconditioner
        kw = dict(coordinates="E", params=GLSParameters(chunk_s=chunk / fs, tol=1e-20, coarse_nside=nc))
        res = make_maps(groups[comm.rank :: comm.size], nside, comm=comm, **kw)
        if comm.rank == 0:
            ref = make_maps(groups, nside, **kw)
            seen = ref.maps != hp.UNSEEN
            assert np.abs(res.maps[seen] - ref.maps[seen]).max() < 1e-8 * np.abs(ref.maps[seen]).max()
            assert abs(res.iterations - ref.iterations) <= 1


def test_estimate_common_modes_mpi():
    """Groups of the same detectors on different ranks: same estimate as serial."""
    from pysanepic import CommonMode, DetectorGroup, GLSParameters, OneOverFNoise, estimate_noise, make_maps
    from test_mapmaker import _group_data

    comm = MPI.COMM_WORLD
    nside, fs, chunk = 8, 10.0, 4000
    rng = np.random.default_rng(23)
    sky = rng.normal(0, 1e-4, (3, 12 * nside**2))
    own = OneOverFNoise(net_ukrts=50.0, fknee_hz=0.05, alpha=1.0, fmin_hz=1e-3)
    mode = OneOverFNoise(net_ukrts=30.0, fknee_hz=0.2, alpha=1.5, fmin_hz=1e-3)
    coupling = rng.uniform(0.6, 0.9, 4)
    groups = [DetectorGroup(detectors=_group_data(rng, nside, sky, fs, chunk, 2, own, mode, coupling), modes=[CommonMode(coupling=coupling, noise=mode)])
              for _ in range(3)]
    kw = dict(chunk_s=chunk / fs, nbins=12)
    first = make_maps(groups, nside, coordinates="E", params=GLSParameters(chunk_s=chunk / fs, tol=1e-10))
    est = estimate_noise(groups[comm.rank :: comm.size], first, comm=comm, **kw)
    if comm.rank == 0:
        ref = estimate_noise(groups, first, **kw)
        names = tuple(d.name for d in groups[0].detectors)
        assert np.allclose(est[names][0].coupling, ref[names][0].coupling, rtol=1e-8)
        assert np.allclose(est[names][0].noise.psd, ref[names][0].noise.psd, rtol=1e-8)
        assert all(np.allclose(est[n].psd, ref[n].psd, rtol=1e-8) for n in names)


if __name__ == "__main__":
    test_mpi_matches_serial()
    test_make_maps_mpi_padding()
    test_more_ranks_than_pixels()
    test_mpi_count_check()
    test_flags_mpi()
    test_estimate_noise_mpi()
    test_detector_groups_mpi()
    test_estimate_common_modes_mpi()
    test_calibration_mpi()
    test_bandpass_mpi()
    if MPI.COMM_WORLD.rank == 0:
        print("ok")
