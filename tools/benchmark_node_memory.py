"""Memory per node of the distributed GLS at large nside, under MPI.

Each rank observes random pixels over the whole sky (as LiteBIRD: every rank
sees nearly every pixel), so the full maps dominate the memory. Reports, per
rank, the peak PSS (proportional set size: a page shared by the k ranks of a
node counts 1/k for each, so the sum over the ranks of a node is the memory
of the node) during the setup and during the PCG, and the time per iteration.

    srun python tools/benchmark_node_memory.py --src <snapshot>/src --label branch \
        --nside 2048 --output-dir results

Rank 0 saves the I/Q/U solution (float32) to compare two versions.
"""

import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np


def pss():
    """PSS and RSS of this process in bytes (Linux)."""
    out = {}
    if not Path("/proc/self/smaps_rollup").exists():  # not Linux: a smoke test only
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {"Pss": rss, "Rss": rss}
    with open("/proc/self/smaps_rollup") as f:
        for line in f:
            key, _, value = line.partition(":")
            if key in ("Pss", "Rss"):
                out[key] = int(value.split()[0]) * 1024
    return out


class Peak:
    """Peak PSS/RSS sampled every 20 ms in the current phase."""

    def __init__(self):
        self.row, self.rows, self.stop = None, {}, threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        while not self.stop.wait(0.02):
            self.sample()

    def sample(self):
        row = self.row
        if row is not None:
            m = pss()
            row["pss"], row["rss"] = max(row["pss"], m["Pss"]), max(row["rss"], m["Rss"])

    def phase(self, name):
        m = pss()
        self.row = self.rows[name] = {"pss": m["Pss"], "rss": m["Rss"]}
        self.sample()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--nside", type=int, default=2048)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--ns", type=int, default=2**20)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--nthreads", type=int, default=1)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.src)

    from mpi4py import MPI

    import pysanepic
    from pysanepic import GLS, one_over_f_weights

    comm = MPI.COMM_WORLD
    out = Path(args.output_dir)
    if comm.rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    npix = 12 * args.nside**2
    rng = np.random.default_rng(1000 + comm.rank)
    shape = (args.chunks, args.ns)
    pix = rng.integers(0, npix, shape, dtype=np.int32)
    if comm.rank == 0:
        pix[0, :2] = 0, npix - 1  # the same nobs = npix on every version
    angle = rng.uniform(-np.pi, np.pi, shape).astype(np.float32)
    tod = rng.normal(size=shape)
    w = one_over_f_weights(args.ns, 19.0, 1.0, 0.05, 1.0)
    comm.Barrier()

    peak = Peak()
    peak.phase("setup")
    t0 = time.perf_counter()
    g = GLS(pix, None, [w] * args.chunks, pol_angles=(angle, [1.0] * args.chunks), comm=comm,
            nthreads=args.nthreads, overwrite_pix=True, min_pol_rcond=0.0)
    comm.Barrier()
    t1 = time.perf_counter()
    peak.phase("pcg")
    m, info = g.solve(tod, tol=1e-30, maxiter=args.iterations)
    comm.Barrier()
    t2 = time.perf_counter()
    peak.phase("end")
    peak.stop.set()
    peak.thread.join()

    row = {"rank": comm.rank, "host": socket.gethostname(),
           **{f"{k}_{x}_gb": v[x] / 1e9 for k, v in peak.rows.items() for x in ("pss", "rss")}}
    rows = comm.gather(row)
    if comm.rank == 0:
        hosts = sorted({r["host"] for r in rows})
        per_node = {h: {k: sum(r[k] for r in rows if r["host"] == h) for k in rows[0] if k.endswith("pss_gb")}
                    for h in hosts}
        summary = dict(label=args.label, version=pysanepic.__version__, ranks=comm.size, nodes=len(hosts),
                       nside=args.nside, nobs=g.nobs, ncomp=g.ncomp, nthreads=args.nthreads,
                       samples_per_rank=args.chunks * args.ns, iterations=info["iterations"],
                       setup_s=t1 - t0, pcg_s=t2 - t1, s_per_iteration=(t2 - t1) / max(info["iterations"], 1),
                       per_node_peak_pss_gb=per_node, ranks_detail=rows)
        (out / f"{args.label}.json").write_text(json.dumps(summary, indent=1))
        np.save(out / f"{args.label}_maps.npy", m.astype(np.float32))
        print(json.dumps({k: v for k, v in summary.items() if k != "ranks_detail"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
