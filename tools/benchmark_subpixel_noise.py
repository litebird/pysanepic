"""Memory per node of estimate_noise with the sub-pixel terms (issue #86), under MPI.

The result has maps, gradient and Hessian at `nside`, shared by the ranks of
each node (as make_maps with full_maps="node"); each rank estimates the noise
of one detector observing random directions over the whole sky. Reports the
peak PSS per node during estimate_noise (see benchmark_node_memory.py) and
its time.

    srun python tools/benchmark_subpixel_noise.py --src <snapshot>/src --label branch --output-dir results
"""

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_node_memory import Peak  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--nside", type=int, default=2048)
    ap.add_argument("--ns", type=int, default=2**20, help="samples per rank")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.src)
    from mpi4py import MPI

    import pysanepic
    from pysanepic import DetectorData, OneOverFNoise, Pointing, estimate_noise
    from pysanepic.mapmaker import MapResult
    from pysanepic.node import node_of

    comm = MPI.COMM_WORLD
    out = Path(args.output_dir)
    if comm.rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    npix = 12 * args.nside**2
    node = node_of(comm)
    arrays = {k: node.output((n, npix), np.float64) for k, n in (("maps", 3), ("gradient", 2), ("hessian", 3))}
    if node.leader:
        rng = np.random.default_rng(0)
        for k, a in arrays.items():
            a[:] = rng.normal(size=a.shape) * (1.0 if k == "maps" else 0.01)
    for a in arrays.values():
        node.freeze(a)
    result = MapResult(hit_map=None, nside=args.nside, coordinates="E", iterations=0, converged=True, residuals=None,
                       **arrays)
    rng = np.random.default_rng(1000 + comm.rank)
    fs = 19.0
    pointing = Pointing(theta=np.arccos(rng.uniform(-1, 1, args.ns)), phi=rng.uniform(0, 2 * np.pi, args.ns),
                        psi=rng.uniform(-np.pi, np.pi, args.ns), coordinates="E")
    data = [DetectorData(tod=rng.normal(size=args.ns), pointing=pointing, sampling_rate_hz=fs,
                         noise=OneOverFNoise(net_ukrts=50.0), name=f"det{comm.rank}")]
    comm.Barrier()

    peak = Peak()
    peak.phase("estimate")
    t0 = time.perf_counter()
    est = estimate_noise(data, result, chunk_s=args.ns / 4 / fs, comm=comm)
    comm.Barrier()
    t1 = time.perf_counter()
    peak.phase("end")
    peak.stop.set()
    peak.thread.join()

    row = {"rank": comm.rank, "host": socket.gethostname(),
           **{f"{k}_{x}_gb": v[x] / 1e9 for k, v in peak.rows.items() for x in ("pss", "rss")}}
    rows = comm.gather(row)
    psd = comm.gather(est[f"det{comm.rank}"].psd)
    if comm.rank == 0:
        hosts = sorted({r["host"] for r in rows})
        per_node = {h: {k: sum(r[k] for r in rows if r["host"] == h) for k in rows[0] if k.endswith("pss_gb")}
                    for h in hosts}
        summary = dict(label=args.label, version=pysanepic.__version__, ranks=comm.size, nodes=len(hosts),
                       nside=args.nside, samples_per_rank=args.ns, estimate_s=t1 - t0,
                       per_node_peak_pss_gb=per_node, ranks_detail=rows)
        (out / f"{args.label}.json").write_text(json.dumps(summary, indent=1))
        np.save(out / f"{args.label}_psd.npy", np.array(psd))
        print(json.dumps({k: v for k, v in summary.items() if k != "ranks_detail"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
