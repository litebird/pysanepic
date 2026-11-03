"""Compare GLS memory and timings in isolated processes on a Linux compute node.

Example (source snapshots, no installation needed):
    python tools/benchmark_memory.py --baseline-src baseline/src \
        --optimized-src optimized/src --output-dir results --nthreads 48

Synthetic cases isolate coverage, heterogeneous FFT workspaces, and prior
shifts with flags. Uniform chunks are a control. RSS and virtual memory are
both measured: unused np.empty capacity need not consume physical memory.
"""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import threading
import time

import numpy as np


CASES = ("coverage", "uniform", "heterogeneous64", "heterogeneous32", "flags_prior")


def memory():
    """Linux process counters in bytes; ru_maxrss is a lifetime high-water mark."""
    status = {}
    with open("/proc/self/status") as stream:
        for line in stream:
            key, _, value = line.partition(":")
            if key in ("VmRSS", "VmSize", "VmPeak"):
                status[key] = int(value.split()[0]) * 1024
    return dict(rss=status["VmRSS"], virtual=status["VmSize"],
                peak_virtual=status["VmPeak"],
                peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


class Measurements:
    """Sample nested phase RSS/VmSize; also retain the kernel's exact RSS HWM."""

    def __init__(self):
        self.phases = {}
        self.active = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def sample(self):
        while not self.stop.wait(.02):
            self.update()

    def update(self):
        current = memory()
        with self.lock:
            for row in self.active:
                row["peak_rss_bytes"] = max(row["peak_rss_bytes"], current["rss"])
                row["peak_virtual_bytes"] = max(row["peak_virtual_bytes"], current["virtual"])

    @contextmanager
    def phase(self, name):
        current = memory()
        row = dict(start_rss_bytes=current["rss"], peak_rss_bytes=current["rss"],
                   peak_virtual_bytes=current["virtual"])
        with self.lock:
            self.phases[name] = row
            self.active.append(row)
        started = time.perf_counter()
        try:
            yield
        finally:
            row["seconds"] = time.perf_counter() - started
            self.update()
            with self.lock:
                self.active.remove(row)

    def close(self):
        self.stop.set()
        self.thread.join()


def warmup(nthreads, single):
    """Compile on tiny inputs outside every measured phase, for both versions."""
    from pysanepic import GLS, one_over_f_weights

    rng = np.random.default_rng(10)
    pix = rng.integers(4, size=(4, 256), dtype=np.int32)
    pix[:, 50:60] = -1 - np.arange(10, dtype=np.int32)
    angle = rng.uniform(-3., 3., pix.shape).astype(np.float32)
    values = rng.normal(size=pix.shape).astype(np.float32)
    w = one_over_f_weights(256, 19., 1., .05, 1., .005)
    g = GLS(pix, None, [w] * 4, pol_angles=(angle, [1.] * 4), nthreads=nthreads,
            fft_dtype=np.float32 if single else np.float64, overwrite_pix=True,
            shared_templates=[(k, values[k], 0) for k in range(4)],
            shared_prior=(np.ones(1), np.array([.7])))
    g.solve(rng.normal(size=pix.shape), tol=1e-12, maxiter=3)
    if g._pool is not None:
        g._pool.shutdown(wait=True)


def workspace_bytes(g):
    arrays = []
    for slot in g._slots:
        arrays.extend(slot[k] for k in ("t", "c2", "s2") if slot[k] is not None)
        arrays.extend(a for buffers in slot["fft"].values() for a in buffers)
    return sum(a.nbytes for a in arrays)


def run_case(args):
    import ducc0
    import numba
    import pysanepic
    from pysanepic import GLS, one_over_f_weights

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / f"{args.case}_{args.label}_{args.repeat}"
    warmup(args.nthreads, args.case == "heterogeneous32")
    rng = np.random.default_rng(12345)
    metrics = Measurements()
    row = dict(case=args.case, label=args.label, repeat=args.repeat,
               nthreads=args.nthreads, seed=12345, quick=args.quick,
               python=sys.version, executable=sys.executable,
               platform=platform.platform(), hostname=platform.node(),
               numpy=np.__version__, numba=numba.__version__, ducc0=ducc0.__version__,
               source=str(Path(pysanepic.__file__).resolve()))
    try:
        if args.case == "coverage":
            nside, ns, nc = (16, 1024, 8) if args.quick else (1024, 524288, 16)
            g = GLS.__new__(GLS)
            g.nobs, g.nthreads, g.pol = 12 * nside**2, args.nthreads, True
            g.ip = [rng.integers(g.nobs, size=ns, dtype=np.int32) for _ in range(nc)]
            g.angle2 = [rng.uniform(-3., 3., ns).astype(np.float32) for _ in range(nc)]
            g.gamma = [np.ones(1)] * nc
            g._c2, g._dummy, g._pool = np.empty(0, np.float32), np.zeros(1, np.float32), None
            with metrics.phase("coverage"):
                cov = g._chunk_coverage()
            assert cov[:, 0].sum() == ns * nc
            np.save(prefix.with_suffix(".npy"), cov)
            row.update(nside=nside, samples=ns * nc, coverage_bytes=cov.nbytes,
                       array_file=prefix.with_suffix(".npy").name)
        else:
            nside, nc, ns = (4, 8, 1024) if args.quick else (64, 96, 262144)
            lengths = [ns] * nc
            if args.case.startswith("heterogeneous"):
                lengths[0] = 16 * ns
            sky = rng.normal(size=(3, 12 * nside**2))
            pix, angles, tod, weights, templates = [], [], [], [], []
            cache = {}
            for k, n in enumerate(lengths):
                p = rng.integers(sky.shape[1], size=n, dtype=np.int32)
                a = rng.uniform(-3., 3., n).astype(np.float32)
                t = (sky[0, p] + np.cos(a) * sky[1, p] + np.sin(a) * sky[2, p]
                     + rng.normal(0., .2, n)).astype(np.float32)
                if args.case == "flags_prior":
                    values = rng.normal(size=n).astype(np.float32)
                    t += .7 * values
                    templates.append((k, values, 0))
                    start, count = n // 4, n // 5
                    p[start:start + count] = -1 - np.arange(count, dtype=np.int32)
                    t[start:start + count] = np.nan
                if n not in cache:
                    cache[n] = one_over_f_weights(n, 19., 1., .05, 1., .005)
                pix.append(p)
                angles.append(a)
                tod.append(t)
                weights.append(cache[n])
            options = {}
            if templates:
                options.update(shared_templates=templates, shared_prior=(np.array([.1]), np.array([.7])))
            with metrics.phase("setup"):
                g = GLS(pix, None, weights, pol_angles=(angles, [1.] * nc),
                        nthreads=args.nthreads, overwrite_pix=True,
                        fft_dtype=np.float32 if args.case == "heterogeneous32" else np.float64,
                        **options)
            if args.operator_products:
                v = rng.normal(size=g._nsky + g.nflag + g.namp + g._nsh_loc)
                for _ in range(3):
                    g._apply(v)  # allocate buffers and warm the actual chunk layout
                timings = []
                per_batch = args.operator_products // 5
                with metrics.phase("operator"):
                    for _ in range(5):
                        started = time.perf_counter()
                        for _ in range(per_batch):
                            g._apply(v)
                        timings.append((time.perf_counter() - started) / per_batch)
                row["operator_seconds_per_product"] = timings
            original_pcg = g._pcg

            def pcg(*pargs, **kwargs):
                row["pcg_start_rss_bytes"] = memory()["rss"]
                row["system_unknowns"] = pargs[0].size
                with metrics.phase("pcg"):
                    result = original_pcg(*pargs, **kwargs)
                row["worker_workspace_bytes"] = workspace_bytes(g)
                return result

            g._pcg = pcg
            with metrics.phase("solve"):
                maps, info = g.solve(tod, tol=1e-12, maxiter=80)
            assert info["converged"], info
            np.savez(prefix.with_suffix(".npz"), maps=maps, amplitudes=info["shared_amplitudes"])
            row.update(nside=nside, samples=sum(lengths), chunks=nc, lengths=lengths,
                       flagged_unknowns=g.nflag, iterations=info["iterations"],
                       residual=float(info["residuals"][-1]), converged=bool(info["converged"]),
                       array_file=prefix.with_suffix(".npz").name)
        row["phases"] = metrics.phases
        row["process_peak_rss_bytes"] = memory()["peak_rss"]
        row["process_peak_virtual_bytes"] = memory()["peak_virtual"]
        prefix.with_suffix(".json").write_text(json.dumps(row, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    finally:
        metrics.close()


def compare_arrays(first, second):
    """Compare full coverage arrays in blocks; solver maps fit in memory."""
    if first.suffix == ".npy":
        a, b = np.load(first, mmap_mode="r"), np.load(second, mmap_mode="r")
        pairs = ((a[i:i + 65536], b[i:i + 65536]) for i in range(0, a.shape[0], 65536))
    else:
        with np.load(first) as a, np.load(second) as b:
            pairs = [(a[k], b[k]) for k in a.files]
    error, scale = 0., 0.
    for a, b in pairs:
        assert a.shape == b.shape
        if a.size:
            error = max(error, float(np.max(np.abs(a - b))))
            scale = max(scale, float(np.max(np.abs(a))))
            np.testing.assert_allclose(b, a, rtol=1e-9, atol=1e-10)
    return dict(max_absolute_difference=error, max_relative_difference=error / max(scale, 1e-300))


def compare(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, comparisons = [], []
    snapshots = dict(baseline=Path(args.baseline_src).resolve(), optimized=Path(args.optimized_src).resolve())
    for case in args.cases:
        for repeat in range(args.repeats):
            order = ("baseline", "optimized") if repeat % 2 == 0 else ("optimized", "baseline")
            pair = {}
            for label in order:
                command = [sys.executable, str(Path(__file__).resolve()), "--case", case,
                           "--label", label, "--repeat", str(repeat), "--nthreads", str(args.nthreads),
                           "--output-dir", str(output), "--operator-products", str(args.operator_products)]
                if args.quick:
                    command.append("--quick")
                env = dict(os.environ, PYTHONPATH=str(snapshots[label]),
                           NUMBA_CACHE_DIR=str(output / "numba_cache" / label))
                print(f"Starting {case} {label} repeat={repeat}", flush=True)
                subprocess.run(command, env=env, check=True)
                row = json.loads((output / f"{case}_{label}_{repeat}.json").read_text())
                assert Path(row["source"]).is_relative_to(snapshots[label])
                rows.append(row)
                pair[label] = row
            difference = compare_arrays(output / pair["baseline"]["array_file"], output / pair["optimized"]["array_file"])
            comparisons.append(dict(case=case, repeat=repeat, **difference))
            print(json.dumps(comparisons[-1]), flush=True)
    summary = []
    for case in args.cases:
        entry = dict(case=case)
        for label in snapshots:
            selected = [r for r in rows if r["case"] == case and r["label"] == label]
            entry[label] = dict(
                median_peak_rss_bytes=float(np.median([r["process_peak_rss_bytes"] for r in selected])),
                median_peak_virtual_bytes=float(np.median([r["process_peak_virtual_bytes"] for r in selected])),
                median_seconds=float(np.median([sum(p["seconds"] for k, p in r["phases"].items()
                                                     if k in ("coverage", "setup", "solve")) for r in selected])),
                phases={phase: dict(median_seconds=float(np.median([r["phases"][phase]["seconds"] for r in selected])),
                                    median_peak_rss_bytes=float(np.median([r["phases"][phase]["peak_rss_bytes"] for r in selected])))
                        for phase in selected[0]["phases"]})
            if args.operator_products and case != "coverage":
                entry[label]["median_operator_seconds_per_product"] = float(np.median(
                    [np.median(r["operator_seconds_per_product"]) for r in selected]))
        entry["rss_reduction_fraction"] = 1 - entry["optimized"]["median_peak_rss_bytes"] / entry["baseline"]["median_peak_rss_bytes"]
        entry["time_ratio_optimized_over_baseline"] = entry["optimized"]["median_seconds"] / entry["baseline"]["median_seconds"]
        if args.operator_products and case != "coverage":
            entry["operator_time_ratio_optimized_over_baseline"] = (
                entry["optimized"]["median_operator_seconds_per_product"] /
                entry["baseline"]["median_operator_seconds_per_product"])
        entry["max_relative_difference"] = max(c["max_relative_difference"] for c in comparisons if c["case"] == case)
        summary.append(entry)
    report = dict(summary=summary, comparisons=comparisons, runs=rows)
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-src")
    parser.add_argument("--optimized-src")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nthreads", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cases", choices=CASES, nargs="+", default=CASES)
    parser.add_argument("--operator-products", type=int, default=0,
                        help="Also time this many repeated normal products (multiple of 5)")
    parser.add_argument("--quick", action="store_true", help="Tiny smoke-test inputs; Linux required")
    parser.add_argument("--case", choices=CASES, help="Internal worker mode")
    parser.add_argument("--label", choices=("baseline", "optimized"))
    parser.add_argument("--repeat", type=int, default=0)
    args = parser.parse_args()
    if args.nthreads < 1 or args.repeats < 1:
        parser.error("nthreads and repeats must be positive")
    if args.operator_products < 0 or args.operator_products % 5:
        parser.error("operator-products must be a nonnegative multiple of 5")
    if args.case:
        if args.label is None:
            parser.error("worker mode requires --label")
        run_case(args)
    else:
        if not args.baseline_src or not args.optimized_src:
            parser.error("comparison requires --baseline-src and --optimized-src")
        compare(args)
