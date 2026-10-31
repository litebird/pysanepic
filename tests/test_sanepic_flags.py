"""The paper's approximate gap treatment, independently checked as dense GLS."""

import dataclasses

import healpy as hp
import numpy as np
import pytest

from pysanepic import DetectorData, GLS, GLSParameters, OneOverFNoise, Pointing, make_maps, one_over_f_weights
from pysanepic.mapmaker import _fill_flagged
from pysanepic.gls import _GapFilledTOD, _chunks


@pytest.mark.parametrize("pol", [False, True])
@pytest.mark.parametrize("nthreads", [1, 2])
@pytest.mark.parametrize("grouped", [False, True])
def test_sanepic_flags_dense(pol, nthreads, grouped):
    """No Schur correction: zero pointing rows, retain the gap-filled RHS.

    Include a fully flagged chunk, edge gaps, templates, polarization, padded
    FFT lengths, and correlated detectors. Check the exact preconditioner too.
    """
    rng = np.random.default_rng(32)
    ns, nfft, npix, nc = 64, 80, 4, 3
    pix = rng.integers(npix, size=(nc, ns))
    psi = rng.uniform(-np.pi, np.pi, size=pix.shape) if pol else None
    flags = rng.random(pix.shape) < 0.15
    flags[0, :3] = flags[1, -4:] = flags[2] = True
    labels = np.where(flags, -1, pix)
    tod = rng.normal(size=pix.shape)  # finite, already gap-filled
    w = np.array([one_over_f_weights(nfft, 10.0, s, 0.5, 1.0, 0.05) for s in (1.0, 1.3, 0.8)])
    U = np.array([[1.0], [0.7], [0.9]])
    lam = np.full((1, nfft // 2 + 1), 0.4)
    common = [(list(range(nc)), U, lam)] if grouped else None
    values = rng.normal(size=(nc, ns))
    original = labels.copy()
    g = GLS(labels, psi, w, min_pol_rcond=0, angle_dtype=np.float64, fft_lengths=nfft,
            common_modes=common, templates=[(k, values[k], 0) for k in range(nc)],
            template_dtype=np.float64, nthreads=nthreads, flag_method="sanepic")
    assert g.nflag == 0
    assert g._locprec.size == 1  # the template only, no state for flags
    assert np.array_equal(labels, original)
    m, info = g.solve(tod, tol=1e-26)
    assert info["converged"]

    ncomp = 3 if pol else 1
    P = np.zeros((nc * ns, ncomp * npix))
    for k in range(nc):
        rows = k * ns + np.arange(ns)
        P[rows, pix[k]] = 1
        if pol:
            P[rows, npix + pix[k]] = np.cos(2 * psi[k])
            P[rows, 2 * npix + pix[k]] = np.sin(2 * psi[k])
    P[flags.ravel()] = 0
    T = np.where(flags, 0, values).reshape(-1, 1)
    design = np.column_stack([P, T])
    lag = (np.arange(ns)[:, None] - np.arange(ns)[None, :]) % nfft
    W = np.zeros((nc * ns, nc * ns))
    Wown = W.copy()
    spectral = np.array([np.diag(1 / w[:, f]) + (U * lam[0, f]) @ U.T
                         for f in range(w.shape[1])])
    inv = np.linalg.inv(spectral)
    for i in range(nc):
        for j in range(nc):
            block = np.fft.irfft(inv[:, i, j], n=nfft)[lag] if grouped else np.zeros((ns, ns))
            if i == j:
                Wown[i * ns:(i + 1) * ns, i * ns:(i + 1) * ns] = np.fft.irfft(w[i], n=nfft)[lag]
                if not grouped:
                    block = Wown[i * ns:(i + 1) * ns, i * ns:(i + 1) * ns]
            W[i * ns:(i + 1) * ns, j * ns:(j + 1) * ns] = block
    A = design.T @ W @ design
    b = design.T @ W @ tod.ravel()
    ref = np.linalg.solve(A, b)
    assert np.allclose(m, ref[:-1].reshape(ncomp, npix), rtol=1e-10, atol=1e-12)
    assert np.allclose(info["amplitudes"], ref[-1:], rtol=1e-10, atol=1e-12)
    assert np.array_equal(g.hits, np.bincount(pix[~flags], minlength=npix))
    own = P.T @ Wown @ P
    for p in range(npix):
        idx = p + np.arange(ncomp) * npix
        prec = g.precond[p] if pol else np.array([[g.precond[0, p]]])
        assert np.allclose(prec, np.linalg.inv(own[np.ix_(idx, idx)]), rtol=1e-10)

    # Holding the gap filling fixed can bias the signal: this option must
    # not accidentally become either exact marginalization or offset fitting.
    signal = P @ rng.normal(size=ncomp * npix)
    wrong = np.linalg.solve(P.T @ W @ P, P.T @ W @ (signal + flags.ravel()))
    right = np.linalg.solve(P.T @ W @ P, P.T @ W @ signal)
    if not grouped:
        assert np.linalg.norm(wrong - right) > 1e-4


def test_sanepic_fill():
    """Only valid neighbors enter the line and noise estimate; edges are safe."""
    x = np.arange(120, dtype=float)
    flags = np.zeros(x.size, bool)
    flags[:3] = flags[30:35] = flags[38:40] = flags[-4:] = True
    linear = 2.0 + 0.03 * x
    bad = np.where(flags, np.nan, linear)
    filled = _fill_flagged(bad, flags, np.random.default_rng(7))
    assert np.allclose(filled, linear, rtol=0, atol=1e-13)
    assert np.array_equal(filled[~flags], bad[~flags])
    assert np.isnan(bad[flags]).all()  # no mutation
    noisy = linear + np.random.default_rng(8).normal(size=x.size)
    a = _fill_flagged(np.where(flags, np.nan, noisy), flags, np.random.default_rng(7))
    b = _fill_flagged(np.where(flags, 1e30, noisy), flags, np.random.default_rng(7))
    c = _fill_flagged(noisy, flags, np.random.default_rng(9))
    assert np.array_equal(a, b)
    assert not np.array_equal(a[flags], c[flags])
    assert np.array_equal(_fill_flagged(np.full(3, np.nan), np.ones(3, bool), np.random.default_rng(0)), np.zeros(3))
    assert np.array_equal(_fill_flagged(np.array([np.nan, 2., np.nan]), np.array([True, False, True]),
                                      np.random.default_rng(0)), [2., 2., 2.])


@pytest.mark.parametrize("pad_s", [0, 2])
@pytest.mark.parametrize("split_gap_s", [np.inf, 2])
def test_sanepic_make_maps(pad_s, split_gap_s):
    """Public option: flags are not hits, NaNs are ignored, seed is repeatable."""
    rng = np.random.default_rng(5)
    nside, ns, fs = 1, 2000, 10.0
    pix = rng.integers(12, size=ns)
    psi = rng.uniform(-np.pi, np.pi, size=ns)
    tod = rng.normal(size=ns)
    flags = rng.random(ns) < 0.05
    flags[200:240] = True
    data = DetectorData(tod=np.where(flags, np.nan, tod), flags=flags,
                        pointing=Pointing(pix=pix, psi=psi, nside=nside, coordinates="E"),
                        sampling_rate_hz=fs, noise=OneOverFNoise(net_ukrts=40, fknee_hz=0.5, fmin_hz=0.05))
    params = GLSParameters(chunk_s=100, pad_s=pad_s, split_gap_s=split_gap_s,
                           flag_method="sanepic", flag_seed=17, tol=1e-20,
                           min_pol_rcond=0, angle_dtype=np.float64)
    a = make_maps([data], nside, coordinates="E", params=params)
    b = make_maps([dataclasses.replace(data, tod=np.where(flags, 1e30, tod))], nside,
                  coordinates="E", params=params, nthreads=2)
    assert a.converged and b.converged
    assert a.hit_map.sum() == (~flags).sum()
    assert np.array_equal(a.hit_map, np.bincount(pix[~flags], minlength=12))
    assert np.allclose(a.maps, b.maps, rtol=0, atol=1e-10)
    assert np.isfinite(a.maps).all() and np.all(a.maps != hp.UNSEEN)
    assert np.isnan(data.tod[flags]).all()
    c = make_maps([data], nside, coordinates="E", params=dataclasses.replace(params, flag_seed=18))
    assert not np.allclose(a.maps, c.maps, rtol=0, atol=1e-10)
    with pytest.raises(ValueError, match="flag_seed"):
        make_maps([data], nside, params=dataclasses.replace(params, flag_seed=-1))


def test_sanepic_low_level_requires_filled_data():
    g = GLS(np.array([0, -1, 0]), None, np.ones(2), flag_method="sanepic")
    with pytest.raises(ValueError, match="gap-filled"):
        g.solve(np.array([1., np.nan, 1.]))
    with pytest.raises(ValueError, match="flag_method"):
        GLS(np.array([0]), None, np.ones(1), flag_method="unknown")
    m, info = g.solve(np.array([1., 0., 1.]), tol=1e-20)
    assert info["converged"] and np.allclose(m, 1.)
    assert np.allclose(g.apply_A(np.array([[2.]])), [[4.]])
    assert np.allclose(g.rhs(np.array([1., 0., 1.])), [[2.]])
    assert np.allclose(g.binned(np.array([1., np.nan, 1.])), [[1.]])
    empty = GLS(np.array([-1, -1]), None, np.ones(2), flag_method="sanepic")
    m, info = empty.solve(np.zeros(2))
    assert info["converged"] and m.shape == (1, 0)


def test_sanepic_sparse_filled_tod():
    """Chunk splitting must preserve compact fills, not materialize every TOD."""
    tod = np.arange(1000, dtype=np.float32)
    flags = np.zeros(tod.size, bool)
    flags[20:22] = True
    tod[flags] = np.nan
    filled = _GapFilledTOD(tod, flags, np.array([4., 5.]))
    chunks = _chunks([filled, filled])
    assert all(c is filled for c in chunks)
    assert filled.tod is tod and filled.flags is flags
    assert filled.values.nbytes == 16
    want = np.arange(tod.size, dtype=float)
    want[flags] = [4., 5.]
    assert np.array_equal(np.asarray(filled, dtype=float), want)
    assert np.isnan(tod[flags]).all()
