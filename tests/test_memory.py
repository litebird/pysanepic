"""Allocation bounds and numerical results for memory-limited GLS paths."""

import tracemalloc
import weakref

import numpy as np
import pytest

from pysanepic import GLS, fast_fft_length, one_over_f_weights
import pysanepic.gls as kernels


@pytest.mark.parametrize("workers", [0, 1, 2, 4])
def test_coverage_memory_budget(monkeypatch, workers):
    """Bound the peak, retain just one accumulator, and preserve coverage."""
    rng = np.random.default_rng(73)
    npix, ns, nc = 4096, 512, 8
    pix = rng.integers(npix, size=(nc, ns), dtype=np.int32)
    pix[-1, -1] = npix - 1
    psi = rng.uniform(-np.pi, np.pi, (nc, ns))
    g = GLS(pix, psi, np.ones((nc, ns // 2 + 1)), nthreads=4, angle_dtype=np.float64)
    map_bytes = npix * 7 * 8
    monkeypatch.setattr(kernels, "_COVERAGE_BYTES", workers * map_bytes)
    g._chunk_coverage()  # warm compiled kernels and the worker pool
    tracemalloc.start()
    try:
        cov = g._chunk_coverage()
        retained, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Allow worker angle buffers and Python bookkeeping in addition to the maps.
    assert peak < max(1, workers) * map_bytes + 4 * ns * 16 + 65536
    assert retained < map_bytes + 65536
    ip = pix.ravel()
    angle = np.concatenate(g.angle2)
    c, s = np.cos(angle), np.sin(angle)
    hits = np.bincount(ip, minlength=npix)
    expected = np.column_stack([hits, hits, *(
        np.bincount(ip, weights=w, minlength=npix) for w in (c, s, c*c, c*s, s*s))])
    np.testing.assert_allclose(cov, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("private", [True, False])
@pytest.mark.parametrize("pol", [True, False])
def test_heterogeneous_worker_buffers(monkeypatch, private, pol):
    """One long chunk must not inflate every worker's persistent buffers."""
    monkeypatch.setattr(kernels, "_PRIVATE_MAPS_BYTES", 4e9 if private else 0)
    rng = np.random.default_rng(27)
    lengths = [256, 48, 80, 32, 96, 64, 48]
    pix = [rng.integers(6, size=n, dtype=np.int32) for n in lengths]
    for p in pix:
        p[10:13] = [-1, -2, -3]
    psi = [rng.uniform(-np.pi, np.pi, n) for n in lengths] if pol else None
    nfft = [fast_fft_length(n + 11) for n in lengths]
    weights = [one_over_f_weights(m, 10., 1., .05, 1., .001) for m in nfft]
    tod = [rng.normal(size=n) for n in lengths]
    local = [(k, rng.normal(size=n), k) for k, n in enumerate(lengths)]
    shared = [(k, rng.normal(size=n), k % 2) for k, n in enumerate(lengths)]
    opts = dict(fft_lengths=nfft, fft_dtype=np.float32, angle_dtype=np.float64,
                templates=local, shared_templates=shared, template_dtype=np.float64)
    serial = GLS(pix, psi, weights, nthreads=1, **opts)
    parallel = GLS(pix, psi, weights, nthreads=4, **opts)
    v = rng.normal(size=serial._nsky + serial.nflag + serial.namp + serial._nsh_loc)
    for args in ({"tod": tod}, {"v": v, "exact": True}, {"v": v}, {"v": v, "exact": True}):
        np.testing.assert_allclose(parallel._apply(**args), serial._apply(**args), rtol=1e-12, atol=1e-12)
    slots = parallel._slots
    sample_bytes = sum(s["t"].nbytes + (s["c2"].nbytes + s["s2"].nbytes if pol else 0) for s in slots)
    global_sample_bytes = 4 * max(lengths) * (24 if pol else 8)
    fft_bytes = sum(a.nbytes for s in slots for buffers in s["fft"].values() for a in buffers)
    global_fft_bytes = 4 * (max(nfft) * 36 + 24)
    assert sample_bytes < .75 * global_sample_bytes
    assert fft_bytes < .75 * global_fft_bytes
    # Reusing the same layout must not reallocate the workspaces.
    buffers = [s["t"] for s in slots] + [a for s in slots for b in s["fft"].values() for a in b]
    parallel._apply(v)
    repeated = [s["t"] for s in slots] + [a for s in slots for b in s["fft"].values() for a in b]
    assert all(a is b for a, b in zip(buffers, repeated))


@pytest.mark.parametrize("mean", [0., .7])
@pytest.mark.parametrize("shared", [True, False])
def test_prior_shift_not_retained_during_pcg(monkeypatch, mean, shared):
    """Keep only amplitude means during PCG, including with flagged samples."""
    rng = np.random.default_rng(98)
    pix = rng.integers(3, size=(2, 64), dtype=np.int32)
    sky = rng.normal(size=3)
    values = rng.normal(size=pix.shape)
    tod = sky[pix] + mean * values
    pix[:, 10:14] = [-1, -2, -3, -4]
    tod[:, 10:14] = np.nan
    templates = [(k, values[k], 0) for k in range(2)]
    prior = (np.ones(1), np.array([mean]))
    opts = {"shared_templates": templates, "shared_prior": prior} if shared else {"templates": templates, "prior": prior}
    g = GLS(pix, None, np.ones((2, 33)), nthreads=1, template_dtype=np.float64, **opts)
    n = g._nsky + g.nflag + g.namp + g._nsh_loc
    allocated = []
    original_zeros_like, original_pcg = np.zeros_like, g._pcg

    def zeros_like(*args, **kwargs):
        out = original_zeros_like(*args, **kwargs)
        if out.shape == (n,):
            allocated.append(weakref.ref(out))
        return out

    def pcg(*args, **kwargs):
        # The initial solution may be live, but no full-size prior shift.
        assert sum(ref() is not None for ref in allocated) <= 1
        return original_pcg(*args, **kwargs)

    monkeypatch.setattr(np, "zeros_like", zeros_like)
    monkeypatch.setattr(g, "_pcg", pcg)
    result, info = g.solve(tod, tol=1e-24)
    assert info["converged"]
    np.testing.assert_allclose(result[0], sky, rtol=1e-11, atol=1e-11)
    amplitudes = info["shared_amplitudes" if shared else "amplitudes"]
    np.testing.assert_allclose(amplitudes, [mean], rtol=1e-11, atol=1e-11)
