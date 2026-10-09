"""Validation of pysanepic.

test_noiseless_recovers_sky: always runs; noiseless data must give back the
input sky (up to the I monopole, which SANEPIC's DC weighting leaves free).

test_matches_cpp_sanepic: runs if the C++ code is built (make -C cpp ARCH=mac)
or SANEPIC_BIN points to a sanepic executable; maps must agree with the C++ ones.

Run with: pytest tests/   (or: python tests/test_vs_sanepic.py)
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from pysanepic import GLS
from pysanepic.sanepic_io import read_inputs, read_map

ROOT = Path(__file__).parents[1]
NS, NSEG, NSIDE, DETS = 4096, 16, 16, ["det0", "det1"]


def _simulate(outdir, *extra):
    subprocess.run(
        [sys.executable, ROOT / "tools/simulate_sanepic_inputs.py", outdir, *extra],
        check=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )


def _solve(d):
    inp = read_inputs(d, f"{d}/pointings_", f"{d}/bolometer_info.txt", DETS, f"{d}/iSpf_", NS, NS, NSEG, NSIDE)
    g = GLS(inp["pix"], inp["psi"], inp["weights"])
    m, info = g.solve(inp["tod"])
    return g.to_healpix(m, NSIDE, fill=0.0), info


def test_noiseless_recovers_sky():
    with tempfile.TemporaryDirectory() as d:
        _simulate(d, "--noiseless")
        maps, _ = _solve(d)
        sky = np.load(f"{d}/truth.npz")["sky"]
    for i in range(3):
        seen = maps[i] != 0
        err = maps[i][seen] - sky[i][seen]
        assert np.abs(err - err.mean()).max() < 5e-4, i  # CG tol limits a few ill-conditioned pixels
        if i > 0:
            assert abs(err.mean()) < 1e-4, i


def test_matches_cpp_sanepic():
    exe = os.environ.get("SANEPIC_BIN", ROOT / "cpp/sanepic")
    if not Path(exe).exists():
        print(f"{exe} not found, skipping C++ comparison")
        return
    with tempfile.TemporaryDirectory() as d:
        _simulate(d)
        subprocess.run(
            ["mpirun", "-n", "5", exe, "-F", d, "-Z", f"{d}/pointings_", "-X", f"{d}/bolometer_info.txt",
             "-d", "2", "-C", "det0", "-C", "det1", "-k", f"{d}/iSpf_", "-u", str(NS), "-l", str(NS),
             "-n", str(NSEG), "-N", str(NSIDE), "-p", "1", "-E", "0", "-h", "0", "-i", "0",
             "-e", "ref", "-O", f"{d}/map"],
            check=True, capture_output=True,
        )
        maps, _ = _solve(d)
        ref = np.array([read_map(f"{d}/map_{c}_ref_1", NSIDE) for c in "IQU"])
    assert np.array_equal(maps != 0, ref != 0)
    for i in range(3):
        scale = np.abs(ref[i]).max()
        assert np.abs(maps[i] - ref[i]).max() < 1e-5 * scale, i


if __name__ == "__main__":
    test_noiseless_recovers_sky()
    test_matches_cpp_sanepic()
    print("ok")
