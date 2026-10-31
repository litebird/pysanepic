"""Conditional gap statistics against independent dense Gaussian formulae."""

import dataclasses
import warnings

import numpy as np
import pytest

from pysanepic import (CalibrationParameters, CommonMode, DetectorData, DetectorGroup, GLSParameters, OneOverFNoise,
                       Pointing, conditional_gap_fill, make_maps, one_over_f_weights)


def _problem(ndet, ns, m):
    rng = np.random.default_rng(13)
    data = rng.normal(size=(ndet, ns))
    flags = np.zeros(data.shape, bool)
    flags[0, [0, 3, ns - 1]] = True
    if ndet > 1:
        flags[1, [2, 3]] = True
    w = np.array([one_over_f_weights(m, 10., 1. + 0.2 * k, 0.8, 1., 0.05) for k in range(ndet)])
    U = np.column_stack([np.linspace(0.7, 1.2, ndet), np.linspace(-0.5, 0.6, ndet)]) if ndet > 1 else None
    lam = np.array([np.full(m // 2 + 1, 0.9), np.linspace(1., 0., m // 2 + 1)]) if ndet > 1 else None
    spectral = np.array([np.diag(1 / w[:, f]) + ((U * lam[:, f]) @ U.T if ndet > 1 else 0.)
                         for f in range(w.shape[1])])
    q = np.linalg.inv(spectral)
    lag = (np.arange(ns)[:, None] - np.arange(ns)[None, :]) % m
    Q = np.block([[np.fft.irfft(q[:, i, j], n=m)[lag] for j in range(ndet)] for i in range(ndet)])
    N = np.linalg.inv(Q)  # independent dense covariance of the principal-block model
    f, u = flags.ravel(), ~flags.ravel()
    mu = N[np.ix_(f, u)] @ np.linalg.solve(N[np.ix_(u, u)], data.ravel()[u])
    cov = N[np.ix_(f, f)] - N[np.ix_(f, u)] @ np.linalg.solve(N[np.ix_(u, u)], N[np.ix_(u, f)])
    return data, flags, w, U, lam, mu, cov


@pytest.mark.parametrize("ndet", [1, 2])
@pytest.mark.parametrize("ns,m", [(11, 11), (12, 16)])
def test_conditional_mean_dense(ndet, ns, m):
    data, flags, w, U, lam, mu, _ = _problem(ndet, ns, m)
    rng = np.random.default_rng(4)
    state = str(rng.bit_generator.state)
    masked = np.where(flags, np.nan, data)
    filled, info = conditional_gap_fill(masked, flags, w, coupling=U, mode_psd=lam,
                                        fft_length=m, draw=False, rng=rng, tol=1e-25)
    assert info["converged"] and info["residual"] <= 1e-25
    assert np.allclose(filled[flags], mu, rtol=1e-11, atol=1e-12)
    assert np.array_equal(filled[~flags], data[~flags])
    assert np.isnan(masked[flags]).all()
    assert state == str(rng.bit_generator.state)  # conditional mean consumes no draws


@pytest.mark.parametrize("ndet", [1, 2])
@pytest.mark.parametrize("m", [12, 16])
def test_conditional_realizations_dense(ndet, m):
    data, flags, w, U, lam, mu, cov = _problem(ndet, 12, m)
    rng = np.random.default_rng(8)
    chol = np.linalg.cholesky(cov)
    samples = []
    for _ in range(3000):
        filled, _ = conditional_gap_fill(np.where(flags, np.nan, data), flags, w,
                                          coupling=U, mode_psd=lam, fft_length=m, rng=rng, tol=1e-24)
        samples.append(filled[flags])
    # Whiten with the independently derived conditional covariance. Statistical
    # bounds (~5 sigma at this ensemble size), rather than matching RNG internals.
    z = np.linalg.solve(chol, (np.asarray(samples) - mu).T).T
    assert np.max(np.abs(z.mean(axis=0))) < 0.09
    assert np.max(np.abs(np.cov(z, rowvar=False) - np.eye(mu.size))) < 0.13


def test_conditional_white_and_dc_null():
    data = np.array([3., 1., np.nan, np.nan, 2., 4.])
    flags = np.isnan(data)
    mean, _ = conditional_gap_fill(data, flags, np.full(4, 0.25), draw=False)
    assert np.allclose(mean[flags], 0., rtol=0, atol=1e-14)  # white noise carries no conditioning information
    a, _ = conditional_gap_fill(data, flags, np.full(4, 0.25), rng=4)
    b, _ = conditional_gap_fill(np.where(flags, 1e30, data), flags, np.full(4, 0.25), rng=4, nthreads=2)
    assert np.allclose(a, b, rtol=0, atol=1e-13)
    # No proper full DC covariance, but Q_ff is positive definite with anchors.
    w = np.array([0., 1., 1., 1.])
    mean, _ = conditional_gap_fill(data, flags, w, draw=False, tol=1e-24)
    assert np.allclose(mean[flags], data[~flags].mean())
    with pytest.raises(ValueError, match="singular"):
        conditional_gap_fill(np.full(6, np.nan), np.ones(6, bool), w)
    full, _ = conditional_gap_fill(np.full(6, np.nan), np.ones(6, bool), np.ones(4), rng=3)
    assert np.isfinite(full).all()  # a fully missing chunk is proper with finite DC variance


def test_conditional_failures():
    data, flags, w, _, _, _, _ = _problem(1, 12, 12)
    with pytest.raises(RuntimeError, match="did not converge"):
        conditional_gap_fill(data, flags, w, maxiter=1, tol=1e-28, rng=5)
    for tol, maxiter in [(0., 10), (np.nan, 10), (1., 10), (1e-12, 0), (1e-12, 2.5)]:
        with pytest.raises(ValueError):
            conditional_gap_fill(data, flags, w, tol=tol, maxiter=maxiter)
    with pytest.raises(ValueError, match="flags"):
        conditional_gap_fill(data, flags.astype(int), w)
    with pytest.raises(ValueError, match="weights"):
        conditional_gap_fill(data, flags, np.zeros_like(w))
    with pytest.raises(ValueError, match="finite"):
        conditional_gap_fill(np.full_like(data, np.nan), flags, w)
    with pytest.raises(ValueError, match="mode_psd"):
        conditional_gap_fill(data, flags, w, coupling=np.ones((1, 1)))
    out, info = conditional_gap_fill(data, np.zeros_like(flags), w)
    assert np.array_equal(out, data) and info["iterations"] == 0


def test_conditional_long_gap_iterations():
    """Steep 1/f with a long gap: restarting CG at each true-residual check took ~1000 iterations."""
    ns = 2**14
    w = one_over_f_weights(ns, 19., 1., 0.1, 2., 1e-4)
    flags = np.zeros(ns, bool)
    flags[500:4500] = True
    data = np.random.default_rng(2).standard_normal(ns)
    _, info = conditional_gap_fill(data, flags, w, draw=False)
    assert info["iterations"] < 200


@pytest.mark.parametrize("ndet", [1, 2])
@pytest.mark.parametrize("m", [95, 128])
@pytest.mark.parametrize("draw", [False, True])
def test_sine_preconditioner_preserves_long_gap_solution(ndet, m, draw):
    """PCG and CG solve the same seeded conditional, including common modes."""
    data, flags, w, U, lam, _, _ = _problem(ndet, 95, m)
    flags[:, 10:58] = True
    masked = np.where(flags, np.nan, data)
    kw = dict(fft_length=m, coupling=U, mode_psd=lam, draw=draw, rng=7, tol=1e-24)
    a, ia = conditional_gap_fill(masked, flags, w, **kw)
    b, ib = conditional_gap_fill(masked, flags, w, preconditioner=None, **kw)
    assert ia["residual"] <= kw["tol"] and ib["residual"] <= kw["tol"]
    assert np.allclose(a, b, rtol=1e-10, atol=1e-11)
    assert np.array_equal(a[~flags], data[~flags])
    if not draw:
        # Dense reference for the new long mask, not the helper's short mask.
        spectral = np.array([np.diag(1 / w[:, f]) + ((U * lam[:, f]) @ U.T if ndet > 1 else 0.)
                             for f in range(w.shape[1])])
        q = np.linalg.inv(spectral)
        lag = (np.arange(95)[:, None] - np.arange(95)[None, :]) % m
        Q = np.block([[np.fft.irfft(q[:, i, j], n=m)[lag] for j in range(ndet)] for i in range(ndet)])
        f, u = flags.ravel(), ~flags.ravel()
        ref = np.linalg.solve(Q[np.ix_(f, f)], -Q[np.ix_(f, u)] @ data.ravel()[u])
        assert np.allclose(a[flags], ref, rtol=1e-10, atol=1e-11)


def test_sine_preconditioner_accelerates_steep_noise():
    ns = 4096
    w = one_over_f_weights(ns, 19., 1., 1., 2., 1e-4)
    flags = np.zeros(ns, bool)
    flags[100:1100] = True
    data = np.random.default_rng(2).normal(size=ns)
    a, ia = conditional_gap_fill(data, flags, w, draw=False, tol=1e-18)
    b, ib = conditional_gap_fill(data, flags, w, draw=False, tol=1e-18, preconditioner=None)
    assert ia["iterations"] < ib["iterations"] / 3
    assert np.allclose(a, b, rtol=1e-6, atol=1e-7)


def test_sine_preconditioner_spd_white_full_and_null_dc():
    from pysanepic.gapfill import _gap_preconditioner

    ns = 95
    flags = np.zeros((1, ns), bool)
    flags[:, 5:50] = True
    flags[:, 60:63] = True
    w = one_over_f_weights(ns, 19., 1., 1., 2., 0.)[None]
    apply = _gap_preconditioner(flags, w, ns, 1)
    nf = flags.sum()
    M = np.empty((nf, nf))
    for j in range(nf):
        v = np.zeros(nf)
        v[j] = 1.
        apply(v, M[:, j])
    assert np.allclose(M, M.T, rtol=1e-13, atol=1e-13)
    assert np.linalg.eigvalsh(M).min() > 0
    data = np.random.default_rng(2).normal(size=ns)
    a, _ = conditional_gap_fill(data, flags[0], w[0], draw=False, tol=1e-22)
    b, _ = conditional_gap_fill(data, flags[0], w[0], draw=False, tol=1e-22, preconditioner=None)
    assert np.allclose(a, b, rtol=1e-9, atol=1e-10)
    full = np.ones(ns, bool)
    _, info = conditional_gap_fill(data, full, w[0] + .01, rng=5, maxiter=1)
    assert info["iterations"] == 1
    white, _ = conditional_gap_fill(data, flags[0], np.full(ns // 2 + 1, .25), rng=5)
    plain, _ = conditional_gap_fill(data, flags[0], np.full(ns // 2 + 1, .25), rng=5, preconditioner=None)
    assert np.allclose(white, plain, rtol=0, atol=1e-13)
    with pytest.raises(ValueError, match="preconditioner"):
        conditional_gap_fill(data, flags[0], w[0], preconditioner="invalid")
    with pytest.raises(ValueError, match="gap_fill_preconditioner"):
        make_maps([], 1, params=GLSParameters(gap_fill_preconditioner=None))


def test_conditional_dc_null_draw_statistics():
    """Zero DC yields a proper conditional with mean anchored by the good data."""
    data = np.array([3., 1., np.nan, np.nan, 2., 4.])
    flags = np.isnan(data)
    rng = np.random.default_rng(12)
    samples = np.array([conditional_gap_fill(data, flags, np.array([0., 1., 1., 1.]), rng=rng)[0][flags]
                        for _ in range(2500)])
    # Q=I-11^T/6, so Q_ff^-1=I+11^T/4 and mu is the observed mean.
    cov = np.eye(2) + np.ones((2, 2)) / 4.
    z = np.linalg.solve(np.linalg.cholesky(cov), (samples - data[~flags].mean()).T).T
    assert np.max(np.abs(z.mean(axis=0))) < 0.1
    assert np.max(np.abs(np.cov(z, rowvar=False) - np.eye(2))) < 0.15


@pytest.mark.parametrize("fill", ["conditional_mean", "constrained"])
@pytest.mark.parametrize("pad_s", [0., 0.5])
@pytest.mark.parametrize("grouped", [False, True])
def test_conditional_make_maps(fill, pad_s, grouped):
    rng = np.random.default_rng(1)
    ns, fs, nside = 500, 10., 1
    sky = rng.normal(size=(3, 12))
    dets = []
    for k in range(2):
        pix = rng.integers(12, size=ns)
        psi = rng.uniform(-np.pi, np.pi, size=ns)
        tod = sky[0, pix] + sky[1, pix] * np.cos(2 * psi) + sky[2, pix] * np.sin(2 * psi)
        flags = np.zeros(ns, bool)
        flags[30 + k:36 + k] = flags[110:145] = True
        dets.append(DetectorData(tod=np.where(flags, np.nan, tod), flags=flags,
                                pointing=Pointing(pix=pix, psi=psi, nside=nside, coordinates="E"),
                                sampling_rate_hz=fs, noise=OneOverFNoise(net_ukrts=40., fknee_hz=0.5, fmin_hz=0.05)))
    modes = [CommonMode(coupling=np.array([1., 0.8]), noise=OneOverFNoise(net_ukrts=60., fknee_hz=0.8, fmin_hz=0.05))]
    pack = lambda ds: [DetectorGroup(detectors=ds, modes=modes)] if grouped else ds
    params = GLSParameters(flag_method="sanepic", gap_fill=fill, pad_s=pad_s, split_gap_s=2.,
                           chunk_s=30., gap_fill_tol=1e-20, flag_seed=9 if fill == "constrained" else 0,
                           angle_dtype=np.float64, tol=1e-22)
    a = make_maps(pack(dets), nside, coordinates="E", params=params, sky_model=sky)
    bad = [dataclasses.replace(d, tod=np.where(d.flags, 1e30, d.tod)) for d in dets]
    b = make_maps(pack(bad), nside, coordinates="E", params=params, sky_model=sky, nthreads=2)
    assert a.converged and b.converged
    assert 0 < a.gap_fill_iterations < params.gap_fill_maxiter
    assert np.allclose(a.maps, b.maps, rtol=0, atol=1e-9)
    assert a.hit_map.sum() == sum((~d.flags).sum() for d in dets)
    assert all(np.isnan(d.tod[d.flags]).all() for d in dets)
    with warnings.catch_warnings(record=True) as caught:  # conditional_mean: the seed has no effect, and says so
        warnings.simplefilter("always")
        c = make_maps(pack(dets), nside, coordinates="E", params=dataclasses.replace(params, flag_seed=10), sky_model=sky)
    assert any("flag_seed has no effect" in str(w.message) for w in caught) == (fill == "conditional_mean")
    if fill == "conditional_mean":
        assert np.allclose(a.maps, sky, rtol=0, atol=1e-10)
        assert np.array_equal(a.maps, c.maps)
    else:
        assert not np.allclose(a.maps, c.maps, rtol=0, atol=1e-10)
    with pytest.raises(ValueError, match="sky_model"):
        make_maps(pack(dets), nside, params=params)
    with pytest.raises(ValueError, match="flag_method"):
        make_maps(pack(dets), nside, params=dataclasses.replace(params, flag_method="act"), sky_model=sky)


def test_conditional_calibration_reuse():
    """Joint fills still have their group spectra when calibration reuses GLS."""
    rng = np.random.default_rng(7)
    ns, fs, nside = 600, 10., 1
    sky = rng.normal(size=(3, 12))
    dets = []
    for _ in range(2):
        pix = rng.integers(12, size=ns)
        psi = rng.uniform(-np.pi, np.pi, size=ns)
        dipole = rng.normal(size=ns)
        tod = sky[0, pix] + sky[1, pix] * np.cos(2 * psi) + sky[2, pix] * np.sin(2 * psi) + dipole
        flags = np.zeros(ns, bool)
        flags[50:80] = True
        dets.append(DetectorData(tod=np.where(flags, np.nan, tod), flags=flags, dipole=dipole,
                                pointing=Pointing(pix=pix, psi=psi, nside=nside, coordinates="E"),
                                sampling_rate_hz=fs, noise=OneOverFNoise(net_ukrts=40., fknee_hz=0.5, fmin_hz=0.05)))
    group = DetectorGroup(detectors=dets, modes=[CommonMode(coupling=np.array([1., 0.8]),
                          noise=OneOverFNoise(net_ukrts=60., fknee_hz=0.8, fmin_hz=0.05))])
    params = GLSParameters(flag_method="sanepic", gap_fill="conditional_mean", chunk_s=30.,
                           angle_dtype=np.float64, tol=1e-22)
    result = make_maps([group], nside, coordinates="E", params=params, sky_model=sky,
                       calibration=CalibrationParameters(n_iter=2))
    assert result.calibration_iterations == 2 and result.converged
    assert np.allclose(result.maps, sky, rtol=0, atol=1e-10)
    assert all(np.allclose(g, 1., rtol=0, atol=1e-10) for g in result.gains)
