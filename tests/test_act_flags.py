"""Compressed ACT cut model: independent dense GLS and limiting cases."""

import dataclasses

import numpy as np
import pytest

from pysanepic import DetectorData, GLS, GLSParameters, OneOverFNoise, Pointing, make_maps, one_over_f_weights
from pysanepic.gls import _act_ncoeff


def test_act_coefficient_counts():
    lengths = np.array([1, 2, 3, 4, 6, 7, 20, 21, 120, 200, 240])
    assert np.array_equal(_act_ncoeff(lengths, 20.), [1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 8])
    assert _act_ncoeff([120], 5.)[0] == 11  # duration, not sample count, above 20
    assert _act_ncoeff([120], 20.)[0] == 7  # nearest integer, ties upward
    assert _act_ncoeff([21], 0.01)[0] == 21  # never more coefficients than samples
    assert np.array_equal(_act_ncoeff(lengths, 20., 3), np.minimum(3, lengths))
    assert np.array_equal(_act_ncoeff(lengths, 20., 10**100), lengths)
    assert np.array_equal(_act_ncoeff(lengths, 20., lambda n, fs: n), lengths)
    assert np.array_equal(_act_ncoeff(lengths, 20., lambda n, fs: 2), np.minimum(2, lengths))
    assert _act_ncoeff([100], 20., lambda n, fs: 1 + np.ceil(n / fs / 2).astype(int))[0] == 4


@pytest.mark.parametrize("pol", [False, True])
@pytest.mark.parametrize("nthreads", [1, 2])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("rule", [None, 1, 3, "duration"])
def test_act_dense_gls(pol, nthreads, grouped, rule):
    """I/Q/U, templates, common modes, exact cut diagonals and long-cut FFTs."""
    rng = np.random.default_rng(25)
    nc, ns, nfft, npix, fs = 3, 160, 192, 5, 20.
    pix = rng.integers(npix, size=(nc, ns))
    psi = rng.uniform(-np.pi, np.pi, size=pix.shape) if pol else None
    flags = np.zeros(pix.shape, bool)
    gaps = [(0, 0, 1, 1), (0, 10, 13, 2), (0, 20, 26, 3), (0, 40, 60, 4),
            (1, 1, 8, 4), (1, 25, 46, 5), (1, 60, 100, 6), (2, 0, 160, 7)]
    if rule == "duration":
        gaps = [(k, a, b, min(b - a, 1 + int(np.ceil((b - a) / fs / 2)))) for k, a, b, _ in gaps]
        rule = lambda n, fs: 1 + np.ceil(n / fs / 2).astype(int)
    elif rule is not None:
        gaps = [(k, a, b, min(rule, b - a)) for k, a, b, _ in gaps]
    ncoef = sum(gap[3] for gap in gaps)
    C = np.zeros((nc * ns, ncoef))
    first = 0
    for k, start, stop, count in gaps:
        flags[k, start:stop] = True
        basis = np.eye(count) if count == stop - start else np.polynomial.legendre.legvander(np.linspace(-1., 1., stop - start), count - 1)
        C[k * ns + start:k * ns + stop, first:first + count] = basis
        first += count
    ncomp = 3 if pol else 1
    P = np.zeros((nc * ns, ncomp * npix))
    for k in range(nc):
        rows = k * ns + np.arange(ns)
        P[rows, pix[k]] = 1.
        if pol:
            P[rows, npix + pix[k]] = np.cos(2 * psi[k])
            P[rows, 2 * npix + pix[k]] = np.sin(2 * psi[k])
    P[flags.ravel()] = 0.
    values = rng.normal(size=(nc, ns, 2))
    T = values.reshape(-1, 2).copy()
    T[flags.ravel()] = 0.
    design = np.column_stack([P, C, T])
    tod = rng.normal(size=pix.shape)
    w = np.array([one_over_f_weights(nfft, fs, s, 0.5, 1., 0.05) for s in (1., 1.3, 0.8)])
    U, lam = np.array([[1.], [0.7], [0.9]]), np.full((1, nfft // 2 + 1), 0.4)
    common = [(list(range(nc)), U, lam)] if grouped else None
    labels = np.where(flags, -1, pix)
    g = GLS(labels, psi, w, min_pol_rcond=0, angle_dtype=np.float64, fft_lengths=nfft,
            common_modes=common, templates=[(k, values[k, :, 0], 0) for k in range(nc)],
            shared_templates=[(k, values[k, :, 1], 0) for k in range(nc)],
            template_dtype=np.float64, nthreads=nthreads, flag_method="act", flag_sampling_rate_hz=fs, act_ncoeff=rule)
    assert g.nflag == ncoef < flags.sum()
    assert g._locprec.size == ncoef + 2
    assert np.array_equal(labels, np.where(flags, -1, pix))  # no caller mutation
    assert sum(cut.nbytes for cut in g._cuts) == 32 * sum(count > 1 for _, _, _, count in gaps)
    m, info = g.solve(np.where(flags, np.nan, tod), tol=1e-25)
    assert info["converged"]

    lag = (np.arange(ns)[:, None] - np.arange(ns)[None, :]) % nfft
    W, Wown = np.zeros((nc * ns, nc * ns)), np.zeros((nc * ns, nc * ns))
    spectral = np.array([np.diag(1 / w[:, f]) + (U * lam[0, f]) @ U.T for f in range(w.shape[1])])
    inv = np.linalg.inv(spectral)
    for i in range(nc):
        for j in range(nc):
            if i == j:
                Wown[i * ns:(i + 1) * ns, i * ns:(i + 1) * ns] = np.fft.irfft(w[i], n=nfft)[lag]
            if grouped:
                W[i * ns:(i + 1) * ns, j * ns:(j + 1) * ns] = np.fft.irfft(inv[:, i, j], n=nfft)[lag]
    if not grouped:
        W = Wown
    A = design.T @ W @ design
    b = design.T @ W @ np.where(flags, 0., tod).ravel()
    ref = np.linalg.solve(A, b)
    assert np.allclose(m, ref[:ncomp * npix].reshape(ncomp, npix), rtol=1e-9, atol=1e-11)
    assert np.allclose(info["amplitudes"], ref[-2:-1], rtol=1e-9, atol=1e-11)
    assert np.allclose(info["shared_amplitudes"], ref[-1:], rtol=1e-9, atol=1e-11)
    assert np.allclose(1 / g._flagprec, np.diag(C.T @ Wown @ C), rtol=1e-10, atol=1e-12)
    assert np.array_equal(g.hits, np.bincount(pix[~flags], minlength=npix))


@pytest.mark.parametrize("fs, full_gap, rule", [(20., False, None), (0.01, True, None), (20., True, lambda n, fs: n)])
def test_act_full_rank_equals_marginalize(fs, full_gap, rule):
    """Isolated flags, and cuts with as many coefficients as samples: exact GLS."""
    rng = np.random.default_rng(11)
    ns = 96
    pix = rng.integers(5, size=(2, ns))
    psi = rng.uniform(-np.pi, np.pi, size=pix.shape)
    flags = np.zeros(pix.shape, bool)
    flags[:, ::4] = True
    if full_gap:
        flags[:, 20:41] = True
    labels = np.where(flags, -1 - np.arange(ns), pix)
    tod = np.where(flags, np.nan, rng.normal(size=pix.shape))
    w = np.tile(one_over_f_weights(ns, fs, 1., 0.5, 1., 0.05), (2, 1))
    kw = dict(min_pol_rcond=0., angle_dtype=np.float64)
    act = GLS(labels, psi, w, flag_method="act", flag_sampling_rate_hz=fs, act_ncoeff=rule, **kw)
    assert act.nflag == flags.sum()
    a, ia = act.solve(tod, tol=1e-24)
    b, ib = GLS(labels, psi, w, **kw).solve(tod, tol=1e-24)
    assert ia["converged"] and ib["converged"]
    assert np.allclose(a, b, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("pad_s", [0., 2.])
@pytest.mark.parametrize("ncoeff", [None, 1, 3])
def test_act_make_maps(pad_s, ncoeff):
    """Zero filling and fitted coefficients: no seed dependence or signal bias."""
    rng = np.random.default_rng(31)
    nside, ns, fs = 1, 3000, 20.
    pix = rng.integers(12, size=ns)
    psi = rng.uniform(-np.pi, np.pi, size=ns)
    sky = rng.normal(size=(3, 12))
    tod = sky[0, pix] + sky[1, pix] * np.cos(2 * psi) + sky[2, pix] * np.sin(2 * psi)
    flags = rng.random(ns) < 0.03
    flags[:6] = flags[50:90] = flags[500:620] = flags[-7:] = True
    d = DetectorData(tod=np.where(flags, np.nan, tod), flags=flags,
                     pointing=Pointing(pix=pix, psi=psi, nside=nside, coordinates="E"),
                     sampling_rate_hz=fs, noise=OneOverFNoise(net_ukrts=40., fknee_hz=0.5, fmin_hz=0.05))
    params = GLSParameters(flag_method="act", act_ncoeff=ncoeff, chunk_s=50., split_gap_s=4., pad_s=pad_s,
                           angle_dtype=np.float64, min_pol_rcond=0, tol=1e-24)
    a = make_maps([d], nside, coordinates="E", params=params)
    with pytest.warns(UserWarning, match="flag_seed has no effect"):
        b = make_maps([dataclasses.replace(d, tod=np.where(flags, 1e30, tod))], nside, coordinates="E",
                      params=dataclasses.replace(params, flag_seed=19), nthreads=2)
    assert a.converged and b.converged
    assert np.allclose(a.maps, sky, rtol=0, atol=1e-9)
    assert np.allclose(a.maps, b.maps, rtol=0, atol=1e-10)
    assert a.hit_map.sum() == (~flags).sum()
    assert np.isnan(d.tod[flags]).all()
    if ncoeff == 1:
        offset = make_maps([d], nside, coordinates="E", params=dataclasses.replace(params, flag_method="offset", act_ncoeff=None))
        assert np.allclose(a.maps, offset.maps, rtol=0, atol=1e-10)


def test_act_sampling_rate_validation():
    args = (np.array([0, -1, 0]), None, np.ones(2))
    for fs in (None, 0., -1., np.inf, np.nan):
        with pytest.raises(ValueError, match="flag_sampling_rate_hz"):
            GLS(*args, flag_method="act", flag_sampling_rate_hz=fs)


@pytest.mark.parametrize("rule", [0, -1, 2.5, True, [2], lambda n, fs: 0,
                                  lambda n, fs: np.ones(n.shape), lambda n, fs: [1, 2],
                                  lambda n, fs: [[1]], lambda n, fs: np.nan])
def test_act_ncoeff_validation(rule):
    with pytest.raises(ValueError, match="act_ncoeff"):
        GLS(np.array([0, -1, 0]), None, np.ones(2), flag_method="act", flag_sampling_rate_hz=20., act_ncoeff=rule)


def test_options_of_other_flag_methods():
    """Options of another flag treatment raise (flag_seed only warns)."""
    for kw in ({"act_ncoeff": 3}, {"act_ncoeff": 3, "flag_method": "sanepic"}, {"gap_fill_tol": 1e-20},
               {"gap_fill_maxiter": 50, "flag_method": "sanepic"}):
        with pytest.raises(ValueError, match="act_ncoeff|gap_fill_tol"):
            make_maps([], 1, params=GLSParameters(**kw))
    for kw in ({"flag_method": "act"}, {"flag_method": "marginalize"},
               {"flag_method": "sanepic", "gap_fill": "conditional_mean"}):
        with pytest.warns(UserWarning, match="flag_seed has no effect"):
            make_maps([], 1, params=GLSParameters(flag_seed=3, **kw), sky_model=np.zeros((3, 12)))
    for kw in ({"act_ncoeff": 2}, {"flag_sampling_rate_hz": 20.}):
        for method in ("marginalize", "sanepic"):
            with pytest.raises(ValueError, match="flag_method='act'"):
                GLS(np.array([0, -1, 0]), None, np.ones(2), flag_method=method, **kw)


def test_act_invalid_ncoeff_without_flags():
    """Reject invalid static options before data is read, even without cuts."""
    with pytest.raises(ValueError, match="act_ncoeff"):
        make_maps([], 1, params=GLSParameters(flag_method="act", act_ncoeff=0))


@pytest.mark.parametrize("ns", [20, 64])
def test_act_fully_flagged_chunk_null_constant(ns):
    """A full-chunk constant has zero weight at fmin=0, not a negative diagonal."""
    rng = np.random.default_rng(2)
    pix = rng.integers(4, size=(2, ns))
    pix[1] = -1
    w = np.tile(one_over_f_weights(ns, 20., 1., 0.5, 1., 0.), (2, 1))
    g = GLS(pix, None, w, flag_method="act", flag_sampling_rate_hz=20.)
    assert g._flagprec[0] == 0.0
    assert np.isfinite(g._flagprec).all() and np.all(g._flagprec >= 0)
    tod = rng.normal(size=pix.shape)
    tod[1] = np.nan
    m, info = g.solve(tod, tol=1e-24)
    ref, ir = GLS(pix[0], None, w[0]).solve(tod[0], tol=1e-24)
    assert info["converged"] and ir["converged"]
    assert np.allclose(m - m.mean(), ref - ref.mean(), rtol=0, atol=1e-10)
