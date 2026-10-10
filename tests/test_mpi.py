"""MPI run must give the same maps as the serial run.

Run with: mpirun -n 4 python tests/test_mpi.py  (under pytest it runs on 1 rank)
"""

import tempfile

import numpy as np
from mpi4py import MPI

from pysanepic import GLS
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

    if comm.rank == 0:
        gs = GLS(inp["pix"], inp["psi"], inp["weights"])
        ms, infos = gs.solve(inp["tod"])
        assert np.array_equal(g.mask, gs.mask)
        rel = np.abs(m - ms).max() / np.abs(ms).max()
        print(f"ranks={comm.size} iters mpi/serial={info['iterations']}/{infos['iterations']} max rel diff={rel:.1e}")
        assert rel < 1e-8


def test_make_maps_mpi_padding():
    """make_maps with padding: data split across ranks gives the serial result."""
    import healpy as hp

    from pysanepic import make_maps
    from test_mapmaker import _four_detectors

    comm = MPI.COMM_WORLD
    nside, n, fs = 16, 60_000, 10.0
    rng = np.random.default_rng(0)
    sky = rng.normal(0, 1e-4, (3, 12 * nside**2))
    data = _four_detectors(n, fs, rng, sky, nside)
    kw = dict(coordinates="E", chunk_s=1200.0, pad_s=300.0, tol=1e-20)
    res = make_maps(data[comm.rank :: comm.size], nside, comm=comm, **kw)
    if comm.rank == 0:
        ref = make_maps(data, nside, **kw)
        assert np.array_equal(res.hit_map, ref.hit_map)
        seen = ref.maps != hp.UNSEEN
        assert np.array_equal(res.maps != hp.UNSEEN, seen)
        assert np.abs(res.maps[seen] - ref.maps[seen]).max() < 1e-8 * np.abs(ref.maps[seen]).max()


if __name__ == "__main__":
    test_mpi_matches_serial()
    test_make_maps_mpi_padding()
    if MPI.COMM_WORLD.rank == 0:
        print("ok")
