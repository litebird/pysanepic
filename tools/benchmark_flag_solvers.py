"""Compare conditional-fill PCG/CG and matrix-free high-order ACT on a cluster.

Example: python tools/benchmark_flag_solvers.py --ns 262144 --nthreads 4
This is a numerical/performance benchmark, not a LiteBIRD map-bias validation.
"""

import argparse
import json
import platform
import time

import numpy as np

from pysanepic import conditional_gap_fill, one_over_f_weights
from pysanepic.gls import _cut_project, _cut_transpose


def benchmark(ns, nthreads):
    """Return reproducible configurations, timings, residuals and errors."""
    if ns < 4096:
        raise ValueError("ns must be >= 4096")
    rng = np.random.default_rng(41)
    # Correlated data force multiple iterations, compiling all vector kernels
    # and warming both FFT/DST paths before timing. No benchmark RNG is used.
    warm_flags = np.zeros(128, bool)
    warm_flags[16:64] = True
    for method in (None, "sine"):
        conditional_gap_fill(np.arange(128.), warm_flags, np.linspace(.1, 1., 65),
                             draw=False, nthreads=nthreads, preconditioner=method)
    warm_cuts = np.array([[0, 16, 0, 15]], dtype=np.int64)
    warm_tod = np.empty(16)
    _cut_project(warm_cuts, np.ones(15), warm_tod)
    _cut_transpose(warm_cuts, warm_tod, np.zeros(15))
    results = []
    for name, alpha, knee, grouped in (("dispersed", 1., .025, False),
                                       ("long", 2., .1, False),
                                       ("steep", 2., 1., False),
                                       ("common_modes", 2., .1, True)):
        ndet = 2 if grouped else 1
        data = rng.normal(size=(ndet, ns))
        flags = rng.random(data.shape) < .03
        if name != "dispersed":
            for k in range(ndet):
                first = ns // 8 + 37 * k
                flags[k, first:first + ns // 16] = True
        w = np.array([one_over_f_weights(ns, 19., 1. + .2 * k, knee, alpha, 1e-5) for k in range(ndet)])
        U = np.array([[1.], [.8]]) if grouped else None
        lam = (.5 / one_over_f_weights(ns, 19., 1., knee, alpha, 1e-5))[None] if grouped else None
        case = dict(case=name, ns=ns, fs_hz=19., alpha=alpha, fknee_hz=knee, fmin_hz=1e-5,
                    flagged_samples=flags.sum(axis=1).tolist(), seed=12, tol=1e-16, solves=[])
        solutions = {}
        for method in (None, "sine"):
            started = time.perf_counter()
            try:
                filled, info = conditional_gap_fill(np.where(flags, np.nan, data), flags, w, rng=12,
                                                    draw=True, coupling=U, mode_psd=lam,
                                                    tol=case["tol"], maxiter=2000, nthreads=nthreads,
                                                    preconditioner=method)
                assert np.array_equal(filled[~flags], data[~flags])
                solutions[method] = filled[flags]
                row = dict(preconditioner=method, **info)
            except RuntimeError as exc:
                row = dict(preconditioner=method, converged=False, error=str(exc))
            row["seconds"] = time.perf_counter() - started
            case["solves"].append(row)
        if len(solutions) == 2:
            case["relative_solution_difference"] = float(np.linalg.norm(solutions[None] - solutions["sine"])
                                                          / np.linalg.norm(solutions["sine"]))
        if "sine" not in solutions:
            raise RuntimeError(f"preconditioned fill failed: {case}")
        results.append(case)
        print(json.dumps(case), flush=True)

    act = []
    # Large enough to exercise endpoint rescaling, without storing n*k values.
    for n in (512, 2048):
        k = n - 1
        cuts = np.array([[0, n, 0, k]], dtype=np.int64)
        amp = rng.normal(size=k)
        tod, back = np.empty(n), np.zeros(k)
        started = time.perf_counter()
        _cut_project(cuts, amp, tod)
        _cut_transpose(cuts, tod, back)
        error = float(np.linalg.norm(back - amp) / np.linalg.norm(amp))
        assert error < 1e-10, error
        act.append(dict(samples=n, coefficients=k, gram_relative_error=error,
                        seconds=time.perf_counter() - started))
    return dict(platform=platform.platform(), numpy=np.__version__, nthreads=nthreads,
                fills=results, act=act)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=int, default=262144)
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--output", help="Optional JSON results path")
    args = parser.parse_args()
    result = benchmark(args.ns, args.nthreads)
    if args.output:
        with open(args.output, "w") as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)
