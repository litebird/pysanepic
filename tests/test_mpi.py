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


if __name__ == "__main__":
    test_mpi_matches_serial()
    test_make_maps_mpi_padding()
    test_more_ranks_than_pixels()
    test_mpi_count_check()
    test_flags_mpi()
    test_estimate_noise_mpi()
    if MPI.COMM_WORLD.rank == 0:
        print("ok")
