"""_ChunkBuilder on single DetectorData: chunks, gaps, padding and flag labels,
and the index of the DetectorData among those of make_maps."""

import numpy as np
import pytest

from pysanepic import (
    CommonMode,
    DetectorData,
    DetectorGroup,
    GapFill,
    GLSParameters,
    OneOverFNoise,
    Pointing,
)
from pysanepic.mapmaker import _ChunkBuilder

NSIDE, FS = 4, 10.0
NPIX = 12 * NSIDE**2


def _det(n=1000, flags=None, name="a"):
    pix = np.arange(n) % NPIX
    return DetectorData(tod=np.arange(n, dtype=float), pointing=Pointing(pix=pix, psi=np.zeros(n), nside=NSIDE, coordinates="G"),
                        sampling_rate_hz=FS, noise=OneOverFNoise(net_ukrts=50.0), flags=flags, name=name)


@pytest.mark.parametrize("method", ["offset", "marginalize", GapFill()], ids=["offset", "marginalize", "gapfill"])
def test_chunks_gaps_padding_and_flags(method):
    """Chunks of 400 samples (the leftover in the last: 400 and 600); a run of 50 flagged
    samples (>= split_gap_s) splits its chunk and is dropped, a run of 5 stays
    with labels of its treatment; the padding margins have their own virtual
    pixels and gamma 0."""
    flags = np.zeros(1000, bool)
    flags[500:550] = True  # split
    flags[900:905] = True  # kept
    m = 20
    params = GLSParameters(chunk_s=40.0, split_gap_s=3.0, pad_s=m / FS, flag_method=method)
    b = _ChunkBuilder(NSIDE, "G", params, None, 1, None)
    assert b.add(_det(flags=flags), 0) == [0, 1, 2]
    pieces = [(0, 400), (400, 500), (550, 1000)]
    assert [len(p) for p in b.pix] == [hi - lo + 2 * m for lo, hi in pieces]
    for k, (lo, hi) in enumerate(pieces):
        np.testing.assert_array_equal(b.pix[k][:m], NPIX + 2 * k)  # one virtual pixel per margin
        np.testing.assert_array_equal(b.pix[k][-m:], NPIX + 2 * k + 1)
        assert np.all(b.gamma[k][:m] == 0) and np.all(b.gamma[k][-m:] == 0)
        np.testing.assert_array_equal(np.asarray(b.tod[k])[m:-m][~flags[lo:hi]], np.arange(lo, hi)[~flags[lo:hi]])
    last = b.pix[2][m:-m]
    expected = {"offset": [-1] * 5, "marginalize": [-1, -2, -3, -4, -5]}.get(method, [-1] * 5)  # GapFill: -1
    np.testing.assert_array_equal(last[350:355], expected)
    np.testing.assert_array_equal(last[:350], np.arange(550, 900) % NPIX)
    assert b.n_virtual == 3 and len(b.weights) == len(set(b.nfft))


def test_reuse_builds_only_the_tod():
    """With a GLS kept from the first pass (reuse), only the TOD of the chunks is built."""
    b = _ChunkBuilder(NSIDE, "G", GLSParameters(chunk_s=40.0), None, 1, None)
    b.reuse = True
    assert b.add(_det(), 0) == [0, 1]
    assert len(b.tod) == 2 and b.pix == [] and b.w == [] and b.a2 == []


def test_index_of_detector_data():
    """j is the position of a DetectorData among those given to make_maps,
    groups expanded: that of the first one for a group."""
    mode = CommonMode(coupling=np.ones(2), noise=OneOverFNoise(net_ukrts=10.0))
    data = [_det(name="a"), DetectorGroup(detectors=[_det(name="b"), _det(name="c")], modes=[mode]), _det(name="d")]
    b = _ChunkBuilder(NSIDE, "G", GLSParameters(), None, 1, None)
    assert [j for j, _ in b.items(data)] == [0, 1, 3]
