"""An exception on one MPI rank aborts all of them instead of hanging (node.abort_on_error)."""

import shutil
import subprocess
import sys
import textwrap

import pytest

from pysanepic.node import ABORT_ENV, abort_on_error


class FakeComm:
    def __init__(self, size):
        self.size, self.aborted = size, None

    def Get_size(self):
        return self.size

    def Get_rank(self):
        return 1

    def Abort(self, code):
        self.aborted = code


@abort_on_error
def _fails(x, comm=None):
    raise ValueError("fake failure")


class _Solver:
    def __init__(self, comm):
        self.comm = comm

    @abort_on_error
    def solve(self):
        raise ValueError("fake failure")


def test_abort_on_error(monkeypatch, capsys):
    """Several ranks: traceback and Abort(1), with the communicator argument or
    self.comm; one rank, no communicator or PYSANEPIC_MPI_ABORT=0: raised."""
    monkeypatch.delenv(ABORT_ENV, raising=False)
    for call in (lambda c: _fails(1, comm=c), lambda c: _Solver(c).solve()):
        comm = FakeComm(2)
        with pytest.raises(ValueError, match="fake failure"):  # raised too, if Abort returns
            call(comm)
        assert comm.aborted == 1
        err = capsys.readouterr().err
        assert "rank 1 of 2 failed" in err and "ValueError: fake failure" in err
        for comm in (FakeComm(1), None):
            with pytest.raises(ValueError, match="fake failure"):
                call(comm)
    monkeypatch.setenv(ABORT_ENV, "0")
    comm = FakeComm(2)
    with pytest.raises(ValueError, match="fake failure"):
        _fails(1, comm=comm)
    assert comm.aborted is None


SCRIPT = textwrap.dedent("""
    import numpy as np
    from mpi4py import MPI
    from pysanepic import GLS, one_over_f_weights

    comm = MPI.COMM_WORLD
    pix = (np.arange(400, dtype=np.int32) % 50).reshape(4, 100)
    w = [one_over_f_weights(100, 1.0, 1.0, 0.0, 1.0)] * 4
    mine = slice(comm.rank, None, comm.size)
    g = GLS(pix[mine], None, w[mine], comm=comm)
    if comm.rank == 1:  # fails inside solve; rank 0 waits in a collective
        def fail(*args, **kwargs):
            raise ValueError("fake failure")
        g._chunk_tod = fail
    g.solve(np.ones((4, 100))[mine], maxiter=5)
    print("rank", comm.rank, "not aborted")
""")


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="needs mpirun")
def test_abort_mpi(tmp_path):
    """A real failure on one of 2 ranks inside GLS.solve: the job ends with an
    error, it does not hang."""
    pytest.importorskip("mpi4py")
    script = tmp_path / "fail.py"
    script.write_text(SCRIPT)
    try:
        run = subprocess.run(["mpirun", "-n", "2", sys.executable, str(script)], capture_output=True, text=True,
                             timeout=120)
    except subprocess.TimeoutExpired:
        pytest.fail("the ranks hung instead of aborting")
    assert run.returncode != 0
    assert "rank 1 of 2 failed in GLS.solve" in run.stderr and "fake failure" in run.stderr, run.stderr
    assert "not aborted" not in run.stdout
