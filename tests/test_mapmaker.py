"""make_maps on synthetic DetectorData (no litebird_sim needed).

Noiseless TOD of a random Galactic sky, observed with Ecliptic pointings, an
HWP, polarization efficiency 0.9 on one detector and chunks that do not divide
the data. The TOD is built with healpy's rotation, so pysanepic's numba
rotation kernel is checked against an independent implementation.
Run with: pytest tests/  (or: python tests/test_mapmaker.py)
"""

import dataclasses
import warnings

import ducc0
import healpy as hp
import numpy as np
import pytest

from pysanepic import Coordinates, DetectorData, GLSParameters, OneOverFNoise, Pointing, TabulatedNoise, Template, make_maps, one_over_f_weights, pol_angle_template


def _scan(n, rng):
    """Spinning scan: (theta, phi, psi) in Ecliptic, covering most of the sky."""
    t = np.arange(n)
    lam = 2 * np.pi * t / n + rng.uniform(0, 2 * np.pi)
    spin = 2 * np.pi * t / 3000.0
    a = np.array([np.cos(lam), np.sin(lam), np.zeros(n)])
    z = np.array([np.zeros(n), np.zeros(n), np.ones(n)])
    e2 = np.cross(a.T, z.T).T
    b = np.cos(1.3) * a + np.sin(1.3) * (np.cos(spin) * z + np.sin(spin) * e2)
    theta = np.arccos(np.clip(b[2], -1, 1))
    phi = np.arctan2(b[1], b[0]) % (2 * np.pi)
    return theta, phi, (spin + lam) % (2 * np.pi) - np.pi


def test_noiseless_recovers_sky():
    nside, n, fs = 16, 120_000, 10.0
    rng = np.random.default_rng(0)
    sky = rng.normal(0, 1e-4, (3, 12 * nside**2))
    rot = hp.Rotator(coord=["E", "G"])

    data = []
    for pol_angle, gamma in [(0.0, 1.0), (np.pi / 4, 0.9)]:
        theta, phi, psi = _scan(n, rng)
        hwp = 2 * np.pi * 0.7 * np.arange(n) / fs
        # Signal in Galactic coordinates, rotated with healpy (independent of pysanepic)
        th_g, ph_g = rot(theta, phi)
        angle = psi + rot.angle_ref(theta, phi) + 2 * hwp - pol_angle
        p = hp.ang2pix(nside, th_g, ph_g)
        tod = sky[0, p] + gamma * (sky[1, p] * np.cos(2 * angle) + sky[2, p] * np.sin(2 * angle))
        data.append(
            DetectorData(
                tod=tod, pointing=Pointing(theta=theta, phi=phi, psi=psi, coordinates="E"), hwp_angle=hwp,
                sampling_rate_hz=fs, noise=OneOverFNoise(net_ukrts=40.0), pol_angle_rad=pol_angle, pol_efficiency=gamma,
            )
        )

    # white noise model (fknee = 0): no degenerate offsets, so the sky is recovered to
    # the precision of the cos/sin (~1e-8 in float32); tight tol because pixels with
    # 4 hits are ill-conditioned
    for dtype, rel in [(np.float32, 1e-7), (np.float64, 1e-9)]:
        params = GLSParameters(chunk_s=1700.0, tol=1e-24, angle_dtype=dtype)
        res = make_maps(data, nside, coordinates="G", params=params)
        assert res.converged
        seen = res.maps[1] != hp.UNSEEN
        assert seen.sum() > 0.5 * seen.size
        assert res.hit_map.sum() == 2 * n
        for i in range(3):
            assert np.abs(res.maps[i][seen] - sky[i][seen]).max() < rel * np.abs(sky).max(), (dtype, i)


def _four_detectors(n, fs, rng, sky, nside, noise=False):
    """Four detectors at different polarization angles, 1/f noise model, no HWP."""
    data = []
    for pol_angle in [0.0, np.pi / 4, np.pi / 8, 3 * np.pi / 8]:
        theta, phi, psi = _scan(n, rng)
        p = hp.ang2pix(nside, theta, phi)
        a = psi + pol_angle
        tod = sky[0, p] + sky[1, p] * np.cos(2 * a) + sky[2, p] * np.sin(2 * a)
        data.append(
            DetectorData(
                tod=tod, pointing=Pointing(theta=theta, phi=phi, psi=psi, coordinates="E"), sampling_rate_hz=fs,
                noise=OneOverFNoise(net_ukrts=40.0, fknee_hz=0.05, alpha=1.0, fmin_hz=1e-4), pol_angle_rad=pol_angle,
            )
        )
    return data


def test_padding_noiseless():
    nside, n, fs = 16, 120_000, 10.0
    rng = np.random.default_rng(0)
    sky = rng.normal(0, 1e-4, (3, 12 * nside**2))
    data = _four_detectors(n, fs, rng, sky, nside)
    # compare only pixels where Q/U are well conditioned (the default criterion)
    good = make_maps(data, nside, coordinates="E", params=GLSParameters(chunk_s=1200.0, maxiter=1)).maps[1] != hp.UNSEEN
    common = dict(chunk_s=1200.0, pad_s=300.0, tol=1e-24, min_pol_rcond=0.0)

    # zeros are exactly described by the margin offsets: exact recovery
    res = make_maps(data, nside, coordinates="E", params=GLSParameters(pad_fill="zeros", **common))
    assert res.hit_map.sum() == 4 * n  # padded samples are not hits
    err = res.maps[:, good] - sky[:, good]
    err[0] -= err[0].mean()  # I monopole unconstrained with 1/f weights
    assert np.abs(err).max() < 1e-4 * np.abs(sky).max()

    # SANEPIC's extrapolation leaves a small bias (unmodelled padding shape)
    res = make_maps(data, nside, coordinates="E", params=GLSParameters(pad_fill="extrapolate", **common))
    err = res.maps[:, good] - sky[:, good]
    err[0] -= err[0].mean()
    assert np.median(np.abs(err)) < 1e-2 * 1e-4


def test_pixel_input_and_generator():
    """Pixels instead of theta/phi, and a generator instead of a list: same maps."""
    import dataclasses

    nside, n, fs = 16, 40_000, 10.0
    rng = np.random.default_rng(2)
    sky = rng.normal(0, 1e-4, (3, 12 * nside**2))
    data = _four_detectors(n, fs, rng, sky, nside)
    common = dict(coordinates="E", params=GLSParameters(chunk_s=1000.0, maxiter=20))
    ref = make_maps(data, nside, **common)

    def with_pix(ordering, nest):
        for d in data:
            p = d.pointing
            pix = hp.ang2pix(nside, p.theta, p.phi, nest=nest).astype(np.int32)
            yield dataclasses.replace(
                d, pointing=Pointing(pix=pix, nside=nside, ordering=ordering, psi=p.psi, coordinates="E")
            )

    for ordering, nest in [("RING", False), ("NESTED", True)]:
        res = make_maps(with_pix(ordering, nest), nside, **common)
        assert np.array_equal(res.maps, ref.maps), ordering

    # pixels must be at the output nside and in the output coordinates
    pix = np.zeros(n, np.int32)
    for kw, match in [(dict(nside=8), "nside=8"), (dict(nside=nside, coordinates="G"), "coordinates")]:
        bad = Pointing(pix=pix, psi=data[0].pointing.psi, **{"coordinates": "E", **kw})
        with pytest.raises(ValueError, match=match):
            make_maps([dataclasses.replace(data[0], pointing=bad)], nside, **common)


def test_pointing_checks():
    psi, n = np.zeros(10), 10
    ok = dict(psi=psi, coordinates="E")
    for kw, match in [
        (dict(theta=psi, phi=psi, coordinates="X", psi=psi), "coordinates"),
        (dict(**ok), "either"),
        (dict(theta=psi, **ok), "either"),
        (dict(theta=psi, phi=psi[:5], **ok), "same length"),
        (dict(theta=psi, phi=psi, nside=16, **ok), "only for pix"),
        (dict(theta=psi, phi=psi, pix=np.zeros(n, int), nside=16, **ok), "not both"),
        (dict(pix=np.zeros(n), nside=16, **ok), "integer"),
        (dict(pix=np.zeros(n, int), **ok), "nside"),
        (dict(pix=np.zeros(n, int), nside=12, ordering="NESTED", **ok), "nside"),
        (dict(pix=np.zeros(n, int), nside=16, ordering="X", **ok), "ordering"),
        (dict(pix=np.full(n, 12 * 16**2), nside=16, **ok), "outside"),
        (dict(pix=np.zeros(5, int), nside=16, **ok), "same length"),
    ]:
        with pytest.raises(ValueError, match=match):
            Pointing(**kw)
    p = Pointing(pix=np.zeros(n, int), nside=16, **ok)
    with pytest.raises(ValueError, match="11 samples"):
        DetectorData(tod=np.zeros(11), pointing=p, sampling_rate_hz=1.0, noise=OneOverFNoise(net_ukrts=1.0))
    with pytest.raises(ValueError, match="coordinates"):
        make_maps([], 16, coordinates="gal")
    # strings and members are the same coordinates
    assert Pointing(pix=np.zeros(n, int), nside=16, psi=psi, coordinates=Coordinates.Ecliptic).coordinates is Coordinates.Ecliptic
    assert p.coordinates is Coordinates.Ecliptic and p.coordinates == "E" and f"{p.coordinates}" == "E"
    with pytest.raises(TypeError):
        DetectorData(tod=np.zeros(n), pointing=None, sampling_rate_hz=1.0, noise=OneOverFNoise(net_ukrts=1.0))


def test_contiguous_run_kernel():
    """O(n) formula for contiguous unpolarized runs == sum over all sample pairs."""
    from pysanepic.gls import _block_circulant

    rng = np.random.default_rng(1)
    ns = 64
    ip = np.zeros((1, ns), dtype=np.int64)
    ip[0, 10:30] = 1  # contiguous run in pixel 1
    ip[0, 40:44] = 2
    ip[0, 50] = 2  # pixel 2 is not contiguous
    c = rng.normal(size=(1, ns))
    c = c + c[:, (-np.arange(ns)) % ns]  # symmetric, like a circulant correlation
    fast = np.zeros((3, 6))
    _block_circulant(ip, c, np.zeros((1, 1)), np.zeros((1, 1)), False, fast, 3, np.zeros(0))
    fast = fast[:, 0]
    t = np.arange(ns)
    brute = [c[0, (t[ip[0] == q][:, None] - t[ip[0] == q][None, :]) % ns].sum() for q in range(3)]
    assert np.allclose(fast, brute)


def test_fft_lengths():
    """With an FFT longer than the chunk (zero-extended, result truncated), the
    solution equals the dense GLS with weight W = the ns x ns block of the m x m
    circulant N^-1, and fast_fft_length gives even 5-smooth-like lengths."""
    from pysanepic import GLS, fast_fft_length, one_over_f_weights

    assert fast_fft_length(6_480_000) == 6_480_000
    assert fast_fft_length(6_480_001) == 6_553_600
    assert all(fast_fft_length(n) >= n and fast_fft_length(n) % 2 == 0 for n in range(1, 300))
    rng = np.random.default_rng(11)
    npix, ns, m, nchunk = 6, 101, 128, 2
    pix = rng.integers(0, npix, (nchunk, ns))
    psi = rng.uniform(0, np.pi, (nchunk, ns))
    tod = rng.normal(size=(nchunk, ns))
    w = one_over_f_weights(m, 10.0, 1.0, 0.5, 1.0, 0.05)
    g = GLS(pix, psi, np.tile(w, (nchunk, 1)), min_pol_rcond=0.0, angle_dtype=np.float64, fft_lengths=m)
    x, info = g.solve(tod, tol=1e-26, maxiter=500)
    lag = (np.arange(ns)[:, None] - np.arange(ns)[None, :]) % m
    W = np.fft.irfft(w, n=m)[lag]  # ns x ns block of the m x m circulant
    A, b = np.zeros((3 * npix, 3 * npix)), np.zeros(3 * npix)
    for k in range(nchunk):
        P = np.zeros((ns, 3 * npix))
        P[np.arange(ns), pix[k]] = 1
        P[np.arange(ns), npix + pix[k]] = np.cos(2 * psi[k])
        P[np.arange(ns), 2 * npix + pix[k]] = np.sin(2 * psi[k])
        A += P.T @ W @ P
        b += P.T @ W @ tod[k]
    ref = np.linalg.solve(A, b).reshape(3, npix)
    assert np.abs(x - ref).max() < 1e-11 * np.abs(ref).max()
    with pytest.raises(ValueError, match="FFT length"):
        GLS(pix, psi, np.tile(w, (nchunk, 1)), fft_lengths=64)


def test_fft_float32():
    """fft_dtype=np.float32: the PCG still reaches tol 1e-12 (the true residual
    is checked in float64) and the maps match the float64 ones."""
    nside, n, fs = 16, 120_000, 10.0
    rng = np.random.default_rng(13)
    sky = rng.normal(0, 1e-4, (3, 12 * nside**2))
    data = _four_detectors(n, fs, rng, sky, nside)
    for d in data:
        d.tod[:] += rng.normal(0, 1e-5, n)  # some noise, so that the solution is not trivial
    params = GLSParameters(chunk_s=1200.0, tol=1e-12)
    r64 = make_maps(data, nside, coordinates="E", params=params)
    r32 = make_maps(data, nside, coordinates="E", params=dataclasses.replace(params, fft_dtype=np.float32))
    assert r32.converged and r32.residuals[-1] <= 1e-12
    seen = r64.maps[1] != hp.UNSEEN
    assert np.abs(r32.maps[:, seen] - r64.maps[:, seen]).max() < 1e-5 * np.abs(r64.maps[:, seen]).max()
    with pytest.raises(ValueError, match="fft_dtype"):
        make_maps(data, nside, coordinates="E", params=dataclasses.replace(params, fft_dtype=np.float16))


def test_common_modes_exact_gls():
    """Chunks with common modes (N = D + U Lambda U^T per frequency, one or two
    modes), plus an independent chunk and flagged samples: the GLS solution
    equals the dense GLS with the full covariance of the group."""
    from pysanepic import GLS, one_over_f_weights

    rng = np.random.default_rng(17)
    npix, ns, fs = 6, 96, 10.0
    f = np.fft.rfftfreq(ns, 1 / fs)
    for r in (1, 2):
        pix = rng.integers(0, npix, (4, ns))  # chunks 0-2: a group observed together; 3: alone
        psi = rng.uniform(0, np.pi, (4, ns))
        tod = rng.normal(size=(4, ns))
        w = np.array([one_over_f_weights(ns, fs, s, 0.5, 1.0, 0.05) for s in (1.0, 1.3, 0.8, 1.1)])
        U = rng.uniform(0.5, 1.5, (3, r))
        lam = np.array([3.0 / (1 + (f / (0.2 * (i + 1))) ** 2) for i in range(r)])  # red modes
        flags = np.zeros((4, ns), bool)
        flags[1, 30:36] = flags[3, 70] = True
        labels = np.where(flags, -1 - np.arange(ns), pix)
        g = GLS(labels, psi, w, min_pol_rcond=0.0, angle_dtype=np.float64, common_modes=[([0, 1, 2], U, lam)])
        x, info = g.solve(np.where(flags, np.nan, tod), tol=1e-26, maxiter=1000)
        assert info["converged"]

        lag = (np.arange(ns)[:, None] - np.arange(ns)[None, :]) % ns
        circ = lambda spec: np.fft.irfft(spec, n=ns)[lag]  # noqa: E731
        Ng = np.block([[circ((i == j) / w[i] + (U[i] * U[j]) @ lam) for j in range(3)] for i in range(3)])
        Ps, _ = _dense_gls(pix, psi, np.eye(ns), npix)
        A, b = np.zeros((3 * npix, 3 * npix)), np.zeros(3 * npix)
        for rows, N in ((list(range(3)), Ng), ([3], circ(1 / w[3]))):
            P = np.vstack([Ps[k] for k in rows])
            d = np.concatenate([tod[k] for k in rows])
            keep = ~np.concatenate([flags[k] for k in rows])
            Ninv = np.linalg.inv(N[np.ix_(keep, keep)])  # exact GLS without the flagged samples
            A += P[keep].T @ Ninv @ P[keep]
            b += P[keep].T @ Ninv @ d[keep]
        ref = np.linalg.solve(A, b).reshape(3, npix)
        assert np.abs(x - ref).max() < 1e-9 * np.abs(ref).max(), r
    with pytest.raises(ValueError, match="common mode"):  # chunks of different lengths
        GLS([pix[0, :50], pix[1, :48]], [psi[0, :50], psi[1, :48]],
            [one_over_f_weights(50, fs, 1.0, 0.5, 1.0), one_over_f_weights(48, fs, 1.0, 0.5, 1.0)],
            common_modes=[([0, 1], np.ones((2, 1)), np.ones((1, 26)))])


def test_templates_exact_gls():
    """Templates fitted with the map (amplitudes shared by chunks and by two
    templates, a common-mode group, flagged samples): map and amplitudes equal
    the dense GLS with the template columns appended to P."""
    from pysanepic import GLS, one_over_f_weights

    rng = np.random.default_rng(23)
    npix, ns, fs = 6, 96, 10.0
    f = np.fft.rfftfreq(ns, 1 / fs)
    pix = rng.integers(0, npix, (4, ns))
    psi = rng.uniform(0, np.pi, (4, ns))
    tod = rng.normal(size=(4, ns))
    w = np.array([one_over_f_weights(ns, fs, s, 0.5, 1.0, 0.05) for s in (1.0, 1.3, 0.8, 1.1)])
    U, lam = rng.uniform(0.5, 1.5, (3, 1)), 3.0 / (1 + (f / 0.2) ** 2)[None]
    flags = np.zeros((4, ns), bool)
    flags[1, 30:36] = flags[3, 70] = True
    half = np.arange(ns) >= ns // 2
    templates = [  # (chunk, values, amplitude labels)
        (0, rng.normal(size=ns), half.astype(int)),  # two amplitudes on chunk 0
        (1, rng.normal(size=ns), np.full(ns, 2)),
        (3, rng.normal(size=ns), np.where(half, -1, 2)),  # shares amplitude 2, half of the chunk
        (3, rng.normal(size=ns), np.full(ns, 3)),  # a second template on chunk 3
    ]
    g = GLS(np.where(flags, -1 - np.arange(ns), pix), psi, w, min_pol_rcond=0.0, angle_dtype=np.float64,
            common_modes=[([0, 1, 2], U, lam)], templates=templates, template_dtype=np.float64)
    assert g.namp == 4
    x, info = g.solve(np.where(flags, np.nan, tod), tol=1e-26, maxiter=1000)
    assert info["converged"]

    lag = (np.arange(ns)[:, None] - np.arange(ns)[None, :]) % ns
    circ = lambda spec: np.fft.irfft(spec, n=ns)[lag]  # noqa: E731
    Ng = np.block([[circ((i == j) / w[i] + (U[i] * U[j]) @ lam) for j in range(3)] for i in range(3)])
    Ps, _ = _dense_gls(pix, psi, np.eye(ns), npix)
    Ps = [np.hstack([P, np.zeros((ns, 4))]) for P in Ps]
    for k, values, labels in templates:
        Ps[k][labels >= 0, 3 * npix + labels[labels >= 0]] += values[labels >= 0]
    A, b = np.zeros((3 * npix + 4, 3 * npix + 4)), np.zeros(3 * npix + 4)
    for rows, N in ((list(range(3)), Ng), ([3], circ(1 / w[3]))):
        P = np.vstack([Ps[k] for k in rows])
        d = np.concatenate([tod[k] for k in rows])
        keep = ~np.concatenate([flags[k] for k in rows])
        Ninv = np.linalg.inv(N[np.ix_(keep, keep)])
        A += P[keep].T @ Ninv @ P[keep]
        b += P[keep].T @ Ninv @ d[keep]
    ref = np.linalg.solve(A, b)
    assert np.abs(x - ref[: 3 * npix].reshape(3, npix)).max() < 1e-9 * np.abs(ref).max()
    assert np.allclose(info["amplitudes"], ref[3 * npix :], rtol=1e-9)
    with pytest.raises(ValueError, match="templates"):
        GLS(pix, psi, w, templates=templates[:1]).rhs(tod)


def test_shared_templates_exact_gls():
    """Shared amplitudes (one per "detector", over chunks) with a sum-zero
    constraint, plus a local one: equal the dense GLS restricted to the
    constraint (null-space basis), also when the shared template is a sky
    signal (P B), whose common amplitude is degenerate with the map."""
    from pysanepic import GLS, one_over_f_weights

    rng = np.random.default_rng(29)
    npix, ns, fs, nchunk = 6, 96, 10.0, 4
    pix = rng.integers(0, npix, (nchunk, ns))
    psi = rng.uniform(0, np.pi, (nchunk, ns))
    tod = rng.normal(size=(nchunk, ns))
    w = np.tile(one_over_f_weights(ns, fs, 1.0, 0.5, 1.0, 0.05), (nchunk, 1))
    Ps, _ = _dense_gls(pix, psi, np.eye(ns), npix)
    B = rng.normal(size=3 * npix)  # a sky template: exactly degenerate in its common amplitude
    det = [0, 1, 0, 1]  # chunks 0, 2 of "detector" 0, chunks 1, 3 of detector 1
    shared = [(k, Ps[k] @ B, np.full(ns, det[k])) for k in range(nchunk)]
    shared += [(k, rng.normal(size=ns), np.full(ns, 2 + det[k])) for k in range(nchunk)]  # a second template
    local = [(2, rng.normal(size=ns), np.zeros(ns, int))]
    g = GLS(pix, psi, w, min_pol_rcond=0.0, angle_dtype=np.float64, templates=local, template_dtype=np.float64,
            shared_templates=shared, shared_groups=[[0, 1], [2, 3]])
    x, info = g.solve(tod, tol=1e-26, maxiter=1000)
    assert info["converged"] and info["shared_amplitudes"].shape == (4,)

    # dense: unknowns [map, local, shared]; the constraint by a null-space basis
    n = 3 * npix
    cols = [np.hstack([P, np.zeros((ns, 5))]) for P in Ps]
    cols[2][:, n] += local[0][1]
    for k, values, labels in shared:
        cols[k][np.arange(ns), n + 1 + labels] += values
    lag = (np.arange(ns)[:, None] - np.arange(ns)[None, :]) % ns
    Ninv = np.linalg.inv(np.linalg.inv(np.fft.irfft(w[0], n=ns)[lag]))
    A = sum(P.T @ Ninv @ P for P in cols)
    b = sum(P.T @ Ninv @ d for P, d in zip(cols, tod))
    C = np.zeros((2, n + 5))
    C[0, n + 1 : n + 3] = C[1, n + 3 : n + 5] = 1
    Z = np.linalg.svd(C)[2][2:].T  # basis of the null space of C
    ref = Z @ np.linalg.solve(Z.T @ A @ Z, Z.T @ b)
    assert np.abs(x - ref[:n].reshape(3, npix)).max() < 1e-9 * np.abs(ref).max()
    assert np.allclose(info["amplitudes"], ref[n : n + 1], rtol=1e-8)
    assert np.allclose(info["shared_amplitudes"], ref[n + 1 :], rtol=1e-8, atol=1e-12)
    # covariance of the shared amplitudes: the shared block of Z (Z^T A Z)^-1 Z^T
    cov = (Z @ np.linalg.solve(Z.T @ A @ Z, Z.T))[n + 1 :, n + 1 :]
    assert np.allclose(g.shared_covariance(tol=1e-26), cov, rtol=1e-8, atol=1e-10 * np.abs(cov).max())


def _calibration_data(nside, n, fs, per):
    """_four_detectors with solar + orbital dipole (the orbital velocity turns
    once over the data, as in the year of _scan) and gains drifting every `per`
    samples: (data, true gains, sky)."""
    rng = np.random.default_rng(5)
    sky = rng.normal(0, 1e-4, (3, 12 * nside**2))
    solar = 3.36e-3 * np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    t = np.arange(n)
    orbital = 2.7e-4 * np.array([-np.sin(2 * np.pi * t / n), np.cos(2 * np.pi * t / n), 0 * t])
    data, true = [], []
    for d in _four_detectors(n, fs, rng, sky, nside):
        th, ph = d.pointing.theta, d.pointing.phi
        nvec = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        dipole = solar @ nvec + np.sum(orbital * nvec, axis=0)
        g = 1 + 0.01 * rng.normal(size=n // per)
        true.append(g)
        data.append(dataclasses.replace(d, tod=g[t // per] * (d.tod + dipole), dipole=dipole))
    return data, true, sky


def test_calibration_noiseless():
    """Gains drifting every 1200 s, solar + orbital dipole: the gains and the
    maps are recovered, also with a mask and with an external sky template."""
    from pysanepic import CalibrationParameters

    nside, n, fs, per = 16, 120_000, 10.0, 12_000
    data, true, sky = _calibration_data(nside, n, fs, per)
    # every pixel solved in Q/U: elsewhere the unmodelled polarization leaks
    # through the 1/f weights, in the maps and in the gains
    params = GLSParameters(chunk_s=1200.0, tol=1e-24, min_pol_rcond=0.0, angle_dtype=np.float64)
    band = np.abs(hp.pix2vec(nside, np.arange(12 * nside**2))[2]) < 0.3
    # noiseless: 3 iterations (the default 2 stops at ~3e-8)
    for cal, model in ((CalibrationParameters(gain_period_s=per / fs, n_iter=3), None),
                       (CalibrationParameters(gain_period_s=per / fs, mask=~band, n_iter=3), None),  # 1 = used: the band is left out
                       (CalibrationParameters(gain_period_s=per / fs, n_iter=3, cache_model=True), sky)):  # sky model: a map
        r = make_maps(data, nside, coordinates="E", params=params, calibration=cal, sky_model=model)
        assert max(np.abs(g - g0).max() for g, g0 in zip(r.gains, true)) < 1e-10
        seen = r.maps[1] != hp.UNSEEN
        err = r.maps[:, seen] - sky[:, seen]
        err[0] -= err[0].mean()  # I monopole unconstrained with 1/f weights
        assert np.abs(err).max() < 1e-7 * np.abs(sky).max()
    # gain_tol: stops when the corrections are below it, before n_iter
    r = make_maps(data, nside, coordinates="E", params=params,
                  calibration=CalibrationParameters(gain_period_s=per / fs, n_iter=10, gain_tol=1e-12))
    assert r.calibration_iterations < 10
    assert max(np.abs(g - g0).max() for g, g0 in zip(r.gains, true)) < 1e-10
    assert make_maps(data[:1], nside, coordinates="E", params=params).gains is None
    with pytest.raises(ValueError, match="gain_tol"):
        CalibrationParameters(gain_tol=0.0)
    with pytest.raises(ValueError, match="dipole"):
        make_maps([dataclasses.replace(data[0], dipole=None)], nside, calibration=CalibrationParameters())
    with pytest.raises(ValueError, match="mask"):
        make_maps(data, nside, calibration=CalibrationParameters(mask=band[:10]))


@pytest.mark.parametrize("ncoeff", [None, 3])
def test_calibration_act_flags(ncoeff):
    """flag_method="act" with calibration: the cuts (also Legendre ones, in
    chunk workers) are reused between iterations; gains and maps are recovered
    as without flags, and the flagged TOD is never read."""
    from pysanepic import CalibrationParameters

    nside, n, fs, per = 16, 120_000, 10.0, 12_000
    data, true, sky = _calibration_data(nside, n, fs, per)
    rng = np.random.default_rng(8)
    flagged = []
    for d in data:
        flags = rng.random(n) < 0.01
        for start, length in zip(rng.integers(0, n - 400, 12), (2, 5, 9, 25, 40, 80, 150, 300) * 2):
            flags[start:start + length] = True
        flagged.append(dataclasses.replace(d, tod=np.where(flags, np.nan, d.tod), flags=flags))
    params = GLSParameters(chunk_s=1200.0, tol=1e-24, min_pol_rcond=0.0, angle_dtype=np.float64,
                           flag_method="act", act_ncoeff=ncoeff)
    r = make_maps(flagged, nside, coordinates="E", params=params, nthreads=2,
                  calibration=CalibrationParameters(gain_period_s=per / fs, n_iter=3))
    assert max(np.abs(g - g0).max() for g, g0 in zip(r.gains, true)) < 1e-10
    seen = r.maps[1] != hp.UNSEEN
    err = r.maps[:, seen] - sky[:, seen]
    err[0] -= err[0].mean()  # I monopole unconstrained with 1/f weights
    assert np.abs(err).max() < 1e-7 * np.abs(sky).max()
    assert r.hit_map.sum() == sum((~d.flags).sum() for d in flagged)


def test_flag_methods_noiseless_response():
    """Signal response with no noise, i.e. the bias of each flag treatment:
    marginalize, act and conditional_mean with the true sky_model are exact;
    conditional_mean is biased linearly in the sky_model error; the local
    SANEPIC fill is biased (a white sky is its worst case: ~20% rms here)."""
    nside, n, fs = 8, 60_000, 10.0
    rng = np.random.default_rng(3)
    sky = rng.normal(0, 1e-4, (3, 12 * nside**2))
    data = []
    for d in _four_detectors(n, fs, rng, sky, nside):
        flags = rng.random(n) < 0.01
        for start, length in zip(rng.integers(0, n - 400, 8), (3, 8, 20, 40, 80, 150, 300, 300)):
            flags[start:start + length] = True
        data.append(dataclasses.replace(d, tod=np.where(flags, np.nan, d.tod), flags=flags))
    base = {"chunk_s": 1200.0, "tol": 1e-24, "min_pol_rcond": 0.0, "angle_dtype": np.float64}

    def error(model=None, **kw):
        r = make_maps(data, nside, coordinates="E", params=GLSParameters(**base, **kw), sky_model=model)
        assert r.converged
        seen = r.maps[1] != hp.UNSEEN
        err = r.maps[:, seen] - sky[:, seen]
        err[0] -= err[0].mean()  # I monopole unconstrained with 1/f weights
        return err / np.abs(sky).max()

    conditional = {"flag_method": "sanepic", "gap_fill": "conditional_mean"}
    for err in (error(), error(flag_method="act"), error(sky, **conditional)):
        assert np.abs(err).max() < 1e-8
    assert np.allclose(error(0 * sky, **conditional), 2 * error(0.5 * sky, **conditional), rtol=0, atol=1e-8)
    assert np.abs(error(0 * sky, **conditional)).max() > 0.1
    assert np.sqrt(np.mean(error(flag_method="sanepic") ** 2)) > 0.01


def test_calibration_unsolved_pol():
    """Few detectors, Q/U not solved in most pixels (default min_pol_rcond):
    their samples are flagged from the second iteration, so the iterations
    converge as with every pixel solved (without, ~2x per iteration: 6e-3
    after 10), and the final maps have I there."""
    from pysanepic import CalibrationParameters

    nside = 16
    data, true, _ = _calibration_data(nside, 60_000, 10.0, 12_000)
    data = data[:3]
    r = make_maps(data, nside, coordinates="E", params=GLSParameters(chunk_s=1200.0, tol=1e-20),
                  calibration=CalibrationParameters(gain_period_s=1200.0, n_iter=4))
    assert max(np.abs(g - g0).max() for g, g0 in zip(r.gains, true)) < 1e-8
    seen_i, seen_qu = r.maps[0] != hp.UNSEEN, r.maps[1] != hp.UNSEEN
    assert seen_qu.sum() < seen_i.sum() / 2 and np.array_equal(seen_i, r.hit_map > 0)


def test_sky_model_sampling():
    """alm interpolated at the pixel centers equal the alm2map maps sampled in
    the pixel (I and Q/U with the polarization angle and efficiency); a
    rotated pointing gives the same signal."""
    from pysanepic.mapmaker import _model, _pixels_and_angles, _project_maps, _sample

    nside, lmax = 8, 16
    rng = np.random.default_rng(31)
    alm = (rng.normal(size=(3, hp.Alm.getsize(lmax))) + 1j * rng.normal(size=(3, hp.Alm.getsize(lmax))))
    alm[:, : lmax + 1] = alm[:, : lmax + 1].real  # m = 0: real
    maps = hp.alm2map(alm, nside, lmax=lmax)
    pix = np.arange(12 * nside**2)
    theta, phi = hp.pix2ang(nside, pix)
    psi = rng.uniform(0, np.pi, pix.size)
    d = DetectorData(tod=np.zeros(pix.size), pointing=Pointing(theta=theta, phi=phi, psi=psi, coordinates="G"),
                     sampling_rate_hz=1.0, noise=OneOverFNoise(net_ukrts=1.0), pol_angle_rad=0.3, pol_efficiency=0.9)
    angle = psi + 0.3
    ref = _project_maps(maps, pix, angle, 0.9)
    p, a, loc = _pixels_and_angles(d, nside, "G", 1, loc=True)
    assert np.array_equal(p, pix) and np.allclose(a, angle)
    assert np.abs(_sample(_model(alm, nside, "x"), d, pix, angle, loc, 1) - ref).max() < 1e-8 * np.abs(ref).max()
    assert np.allclose(_sample(_model(maps, nside, "x"), d, pix, angle, None, 1), ref)
    # the same pointing in Ecliptic coordinates, interpolated in Galactic
    th_e, ph_e = hp.Rotator(coord=["G", "E"])(theta, phi)
    psi_e = psi - hp.Rotator(coord=["G", "E"]).angle_ref(theta, phi)
    d_e = dataclasses.replace(d, pointing=Pointing(theta=th_e, phi=ph_e, psi=psi_e, coordinates="E"))
    _, _, loc_e = _pixels_and_angles(d_e, nside, "G", 1, loc=True)
    assert np.abs(_sample(_model(alm, nside, "x"), d_e, pix, angle, loc_e, 1) - ref).max() < 1e-6 * np.abs(ref).max()
    assert _pixels_and_angles(dataclasses.replace(d, pointing=Pointing(pix=pix, nside=nside, psi=psi, coordinates="G")),
                              nside, "G", 1, loc=True)[2] is None
    with pytest.raises(ValueError, match="alm"):
        _model(alm[:, :-1], nside, "x")
    with pytest.raises(ValueError, match="pixels"):
        _model(maps[:, :-1], nside, "x")
    with pytest.raises(ValueError, match="one array"):
        _model({"cmb": maps}, nside, "x")
    # alm need angles: a pixel pointing is refused before any detector is processed
    d_pix = dataclasses.replace(d, pointing=Pointing(pix=pix, nside=nside, psi=psi, coordinates="G"))
    with pytest.raises(ValueError, match="sky_model as alm"):
        make_maps([d_pix], nside, "G", sky_model=alm)
    with pytest.raises(ValueError, match=r"templates\['fg'\].sky as alm"):
        make_maps([dataclasses.replace(d_pix, name="d0")], nside, "G", templates={"fg": Template(sky=alm)})
    make_maps([d_pix], nside, "G", sky_model=maps)  # maps are sampled in the pixel


def _subpixel_data(nside, n, fs, hi=64, seed=33):
    """_four_detectors on a sky with structure below the pixel size: the TOD
    interpolated from alm at lmax 2 hi; returns (data, alm)."""
    rng = np.random.default_rng(seed)
    lmax = 2 * hi
    cl = 1e-8 / (1 + np.arange(lmax + 1)) ** 1.5
    alm = np.array([hp.synalm(cl, lmax=lmax) for _ in range(3)])
    data = []
    for i, pol_angle in enumerate([0.0, np.pi / 4, np.pi / 8, 3 * np.pi / 8]):
        theta, phi, psi = _scan(n, rng)
        loc = np.stack([theta, phi % (2 * np.pi)], axis=1)
        t = ducc0.sht.synthesis_general(alm=alm[:1], spin=0, lmax=lmax, loc=loc, epsilon=1e-12)[0]
        q, u = ducc0.sht.synthesis_general(alm=alm[1:], spin=2, lmax=lmax, loc=loc, epsilon=1e-12)
        a = psi + pol_angle
        data.append(DetectorData(tod=t + q * np.cos(2 * a) + u * np.sin(2 * a),
                                 pointing=Pointing(theta=theta, phi=phi, psi=psi, coordinates="E"), sampling_rate_hz=fs,
                                 noise=OneOverFNoise(net_ukrts=40.0, fknee_hz=0.05, alpha=1.0, fmin_hz=1e-4),
                                 pol_angle_rad=pol_angle, name=f"d{i}"))
    return data, alm


def test_sky_model():
    """A sky model as alm removes the structure below the pixel size: noiseless
    1/f GLS maps equal the binned maps (the white-noise GLS), which they do
    not without the model; as maps it changes nothing if Q/U are solved
    everywhere."""
    nside, n, fs = 16, 60_000, 10.0
    data, alm = _subpixel_data(nside, n, fs)
    # default min_pol_rcond: in near-singular pixels the iterative and the direct (binned) solutions differ
    params = GLSParameters(chunk_s=1200.0, tol=1e-22, angle_dtype=np.float64)
    plain = make_maps(data, nside, coordinates="E", params=params)
    seen = plain.maps[1] != hp.UNSEEN
    white = make_maps([dataclasses.replace(d, noise=OneOverFNoise(net_ukrts=40.0)) for d in data], nside,
                      coordinates="E", params=params)
    as_alm = make_maps(data, nside, coordinates="E", params=params, sky_model=alm)

    def diff(a, b):
        e = (a.maps - b.maps)[:, seen]
        e[0] -= e[0].mean()  # I monopole unconstrained with 1/f weights
        return np.abs(e).max() / np.abs(white.maps[:, seen]).max()

    assert diff(plain, white) > 1e-3  # the structure below the pixel size leaks through the 1/f weights
    assert diff(as_alm, white) < 1e-10
    # as maps (sampled in the pixel), with Q/U solved everywhere: no change
    p0 = dataclasses.replace(params, min_pol_rcond=0.0)
    ref = make_maps(data, nside, coordinates="E", params=p0)
    as_maps = make_maps(data, nside, coordinates="E", params=p0, sky_model=hp.alm2map(alm, nside))
    assert diff(as_maps, ref) < 1e-8


def _bandpass_data(nside, n, fs, per, amp):
    """_calibration_data plus a foreground seen as (1 + amp[i]) F by detector i."""
    data, true, sky = _calibration_data(nside, n, fs, per)
    rng = np.random.default_rng(37)
    fg = hp.smoothing(rng.normal(0, 3e-4, (3, 12 * nside**2)), fwhm=np.radians(10))
    t = np.arange(n)
    out = []
    for i, (d, g) in enumerate(zip(data, true)):
        pix, angle = hp.ang2pix(nside, d.pointing.theta, d.pointing.phi), d.pointing.psi + d.pol_angle_rad
        f = fg[0, pix] + fg[1, pix] * np.cos(2 * angle) + fg[2, pix] * np.sin(2 * angle)
        out.append(dataclasses.replace(d, tod=d.tod + g[t // per] * (1 + amp[i]) * f, name=f"d{i}"))
    return out, true, sky, fg


def test_bandpass_templates():
    """A foreground seen with a different amplitude by each detector, fitted
    with a sky template summing to zero: the amplitudes (minus their mean)
    are recovered, alone and with the gains, and the maps are sky + (1 +
    mean) fg; also with masks for each fit."""
    from pysanepic import CalibrationParameters

    nside, n, fs, per = 16, 120_000, 10.0, 12_000
    amp = np.array([0.05, -0.02, 0.03, 0.01])
    data, true, sky, fg = _bandpass_data(nside, n, fs, per, amp)
    params = GLSParameters(chunk_s=1200.0, tol=1e-24, min_pol_rcond=0.0, angle_dtype=np.float64)
    expect = {f"d{i}": a - amp.mean() for i, a in enumerate(amp)}
    fg_tmpl = {"fg": Template(sky=fg, sum_zero=True)}
    band = np.abs(hp.pix2vec(nside, np.arange(12 * nside**2))[2]) > 0.3  # 1 = used
    spot = np.ones(12 * nside**2, bool)
    spot[hp.query_disc(nside, hp.ang2vec(1.0, 2.0), np.radians(10))] = False
    # samples out of the gain fit only (band): calibrated with the gains of the
    # previous iteration, so the iterations converge slower (~5x each), with a warning
    masked = lambda m: {"fg": Template(sky=fg, sum_zero=True, mask=m)}  # noqa: E731
    cal = lambda mask=None, n_iter=3, **kw: CalibrationParameters(gain_period_s=per / fs, mask=mask,  # noqa: E731
                                                                  n_iter=n_iter, **kw)
    cases = [(cal(), fg_tmpl, 1e-9, None, False),
             (cal(band * spot, cache_model=True), masked(band * spot), 1e-9, None, False),  # equal masks: kept once
             (cal(band * spot), fg_tmpl, 1e-9, None, False),  # no template mask: that of the gains
             (cal(band * spot, n_iter=10), masked(spot), 1e-8, None, True),
             (cal(), fg_tmpl, 1e-9, fg, False)]  # with a sky model too
    for c, tmpl, tol, model, warn in cases:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = make_maps(data, nside, coordinates="E", params=params, templates=tmpl, calibration=c, sky_model=model)
        assert any("converges slower" in str(x.message) for x in w) == warn
        assert max(abs(r.templates["fg"][k] - v) for k, v in expect.items()) < tol
        assert max(np.abs(g - g0).max() for g, g0 in zip(r.gains, true)) < tol
        seen = r.maps[1] != hp.UNSEEN
        err = (r.maps - sky - (1 + amp.mean()) * fg)[:, seen]
        err[0] -= err[0].mean()
        assert np.abs(err).max() < 1e4 * tol * np.abs(sky).max()
    # without gains (calibrated data, dipole removed): one solve
    t = np.arange(n)
    flat = [dataclasses.replace(d, tod=d.tod / g[t // per] - d.dipole, dipole=None) for d, g in zip(data, true)]
    r = make_maps(flat, nside, coordinates="E", params=params, templates=fg_tmpl)
    assert r.gains is None and max(abs(r.templates["fg"][k] - v) for k, v in expect.items()) < 1e-9
    # a template mask without calibration: the masked samples are left out of the fit, then back in the maps
    r = make_maps(flat, nside, coordinates="E", params=params, templates=masked(spot))
    assert max(abs(r.templates["fg"][k] - v) for k, v in expect.items()) < 1e-9
    seen = r.maps[1] != hp.UNSEEN
    assert np.all(seen[~spot] | (r.hit_map[~spot] == 0))  # the masked pixels are in the maps
    err = (r.maps - sky - (1 + amp.mean()) * fg)[:, seen]
    err[0] -= err[0].mean()
    assert np.abs(err).max() < 1e-5 * np.abs(sky).max()
    with pytest.raises(ValueError, match="name"):
        make_maps([dataclasses.replace(flat[0], name=None)], nside, templates=fg_tmpl)
    with pytest.raises(ValueError, match="sum_zero"):
        make_maps(flat, nside, templates={"fg": Template(sky=fg)})  # as maps: the common part is the map
    with pytest.raises(TypeError, match="Template"):
        make_maps(flat, nside, templates={"fg": fg})
    with pytest.raises(TypeError, match="Template.mask"):
        CalibrationParameters(mask={"gain": band})
    with pytest.raises(ValueError, match=r"templates\['fg'\].mask"):
        make_maps(flat, nside, templates=masked(spot[:10]))


def test_bandpass_common():
    """Templates as alm, a foreground seen as (1 + a_i) F, F also in the sky
    model: with mean(a) = 0 the sum-zero amplitudes are a_i; with mean(a) !=
    0 free amplitudes fit the common part too (constrained by the structure
    of F below the pixel size) and they are a_i again."""
    nside, n, fs = 16, 60_000, 10.0
    data, alm = _subpixel_data(nside, n, fs)
    lmax = (int(np.sqrt(8 * alm.shape[1] + 1)) - 3) // 2
    np.random.seed(43)
    fg = np.array([hp.synalm(3e-9 / (1 + np.arange(lmax + 1)) ** 2, lmax=lmax) for _ in range(3)])
    fg_tod = []
    for d in data:
        loc = np.stack([d.pointing.theta, d.pointing.phi % (2 * np.pi)], axis=1)
        t = ducc0.sht.synthesis_general(alm=fg[:1], spin=0, lmax=lmax, loc=loc, epsilon=1e-12)[0]
        q, u = ducc0.sht.synthesis_general(alm=fg[1:], spin=2, lmax=lmax, loc=loc, epsilon=1e-12)
        a = d.pointing.psi + d.pol_angle_rad
        fg_tod.append(t + q * np.cos(2 * a) + u * np.sin(2 * a))
    # Q/U solved everywhere: elsewhere the template is 0 and the polarization of a_i F leaks
    params = GLSParameters(chunk_s=1200.0, tol=1e-22, angle_dtype=np.float64, min_pol_rcond=0.0)
    kw = dict(coordinates="E", params=params, sky_model=alm + fg)
    for amp, common in ((np.array([0.04, -0.03, 0.02, -0.03]), False), (np.array([0.05, -0.02, 0.03, 0.01]), True)):
        out = [dataclasses.replace(d, tod=d.tod + (1 + a) * f) for d, f, a in zip(data, fg_tod, amp)]
        r = make_maps(out, nside, templates={"fg": Template(sky=fg, sum_zero=not common)}, **kw)
        got = np.array(list(r.templates["fg"].values()))
        assert np.abs(got - amp).max() < 1e-6, (common, got - amp)


def test_tod_templates():
    """Templates given as TOD in DetectorData.templates, in noiseless 1/f data:
    one amplitude per detector (one detector without the template), one
    common to all, one per DetectorData and period; recovered with the maps,
    in one solve and with the gains. The covariance of the amplitudes per
    detector and common is labelled and positive definite."""
    from pysanepic import CalibrationParameters

    nside, n, fs, per, per_drift = 16, 120_000, 10.0, 12_000, 30_000
    data, true, _ = _calibration_data(nside, n, fs, per)
    rng = np.random.default_rng(41)
    t = np.arange(n)
    a_th = {"d0": 2e-3, "d1": -1e-3, "d2": 5e-4}
    a_hk = 3e-4
    a_drift = [rng.normal(0, 1e-4, n // per_drift) for _ in data]
    tmpl = {"thermal": Template(), "hk": Template(common=True), "drift": Template(period_s=per_drift / fs)}
    cal_data, flat = [], []
    for i, (d, g) in enumerate(zip(data, true)):
        tt = {"hk": rng.normal(size=n), "drift": (t % per_drift) / per_drift - 0.5}
        if i < 3:
            tt["thermal"] = np.convolve(rng.normal(size=n), np.ones(50) / 50, mode="same")
        extra = a_hk * tt["hk"] + a_drift[i][t // per_drift] * tt["drift"] + a_th.get(f"d{i}", 0) * tt.get("thermal", 0)
        dd = dataclasses.replace(d, tod=d.tod + g[t // per] * extra, name=f"d{i}", templates=tt)
        cal_data.append(dd)
        flat.append(dataclasses.replace(dd, tod=dd.tod / g[t // per] - d.dipole, dipole=None))
    params = GLSParameters(chunk_s=1200.0, tol=1e-24, covariance_tol=1e-20, min_pol_rcond=0.0, angle_dtype=np.float64)
    cal = CalibrationParameters(gain_period_s=per / fs, n_iter=3)
    for dd, c in ((flat, None), (cal_data, cal)):
        r = make_maps(dd, nside, coordinates="E", params=params, calibration=c, templates=tmpl,
                      template_covariance=c is None)
        assert r.templates["thermal"].keys() == a_th.keys()
        assert max(abs(r.templates["thermal"][k] - v) for k, v in a_th.items()) < 1e-11
        assert abs(r.templates["hk"] - a_hk) < 1e-11
        assert max(np.abs(x - x0).max() for x, x0 in zip(r.templates["drift"], a_drift)) < 1e-11
        if c is None:
            cov = r.template_covariance
        else:
            assert max(np.abs(g - g0).max() for g, g0 in zip(r.gains, true)) < 1e-9
            assert r.template_covariance is None
    assert cov["labels"] == [("thermal", "d0"), ("thermal", "d1"), ("thermal", "d2"), ("hk", None)]
    assert np.allclose(cov["matrix"], cov["matrix"].T, rtol=1e-6, atol=1e-6 * np.abs(cov["matrix"]).max())
    assert np.all(np.linalg.eigvalsh(cov["matrix"]) > 0)
    with pytest.raises(ValueError, match="period_s"):
        Template(period_s=10.0, common=True)
    with pytest.raises(ValueError, match="one value per sample"):
        dataclasses.replace(flat[0], templates={"hk": np.zeros(10)})


def test_pol_angle_template():
    """Polarization angles off by delta_i: pol_angle_template, the sky
    template [0, 2U, -2Q] (alm [0, 2B, -2E]) with sum-zero amplitudes, recovers delta_i - mean(delta) to
    first order, with and without calibration, and the maps are the sky
    rotated by the mean angle; the error is O(delta^2). With an ideal HWP
    the angle is psi + 2 chi - pol_angle: the amplitudes are -delta_i."""
    from pysanepic import CalibrationParameters

    nside, n, fs, per = 16, 120_000, 10.0, 12_000
    data, true, sky = _calibration_data(nside, n, fs, per)
    # the template as maps and as alm
    lmax = 2 * nside
    alm = np.array(hp.map2alm(sky, lmax=lmax))
    _, q, u = hp.alm2map(np.array([0 * alm[0], 2 * alm[2], -2 * alm[1]]), nside, lmax=lmax)
    _, q0, u0 = hp.alm2map(alm, nside, lmax=lmax)
    assert np.allclose(q, 2 * u0) and np.allclose(u, -2 * q0)
    tmpl = {"pol_angle": pol_angle_template(sky)}
    assert tmpl["pol_angle"].sum_zero and np.array_equal(tmpl["pol_angle"].sky, [0 * sky[0], 2 * sky[2], -2 * sky[1]])
    assert np.array_equal(pol_angle_template(alm).sky, [0 * alm[0], 2 * alm[2], -2 * alm[1]])
    params = GLSParameters(chunk_s=1200.0, tol=1e-24, min_pol_rcond=0.0, angle_dtype=np.float64)
    cal = CalibrationParameters(gain_period_s=per / fs, n_iter=4)
    t = np.arange(n)
    chi = 2 * np.pi * t / 600.0  # an HWP turning every minute

    def tod(d, delta, hwp):
        pix = hp.ang2pix(nside, d.pointing.theta, d.pointing.phi)
        a = d.pointing.psi + (2 * chi - d.pol_angle_rad - delta if hwp else d.pol_angle_rad + delta)
        return sky[0, pix] + sky[1, pix] * np.cos(2 * a) + sky[2, pix] * np.sin(2 * a)

    # 1e-4 rad: exact to ~delta^2; 1 degree: 0.4% (the second order)
    for delta, tol in ((np.array([1e-4, -2e-4, 0.5e-4, 1.5e-4]), 1e-8), (np.radians([1.0, -0.5, 0.3, -0.8]), 2e-4)):
        expect = delta - delta.mean()
        m = delta.mean()  # the maps are the sky rotated by the mean angle
        rotated = np.array([sky[0], sky[1] * np.cos(2 * m) + sky[2] * np.sin(2 * m),
                            -sky[1] * np.sin(2 * m) + sky[2] * np.cos(2 * m)])
        flat = [dataclasses.replace(d, tod=tod(d, x, False), dipole=None, name=f"d{i}")
                for i, (d, x) in enumerate(zip(data, delta))]
        calib = [dataclasses.replace(d, tod=g[t // per] * (tod(d, x, False) + d.dipole), name=f"d{i}")
                 for i, (d, g, x) in enumerate(zip(data, true, delta))]
        for dd, c in ((flat, None), (calib, cal)):
            r = make_maps(dd, nside, coordinates="E", params=params, templates=tmpl, calibration=c)
            got = np.array([r.templates["pol_angle"][f"d{i}"] for i in range(4)])
            assert np.abs(got - expect).max() < tol, (c, got - expect)
            if c is not None:
                assert max(np.abs(g - g0).max() for g, g0 in zip(r.gains, true)) < 0.2 * np.max(delta**2) + 1e-8
            seen = r.maps[1] != hp.UNSEEN
            err = (r.maps - rotated)[:, seen]
            err[0] -= err[0].mean()
            assert err.std() < 3 * np.max(delta**2) * sky.std()
    # with an ideal HWP the sign of the angle is reversed
    delta = np.array([1e-4, -2e-4, 0.5e-4, 1.5e-4])
    hwp = [dataclasses.replace(d, tod=tod(d, x, True), dipole=None, name=f"d{i}", hwp_angle=chi)
           for i, (d, x) in enumerate(zip(data, delta))]
    r = make_maps(hwp, nside, coordinates="E", params=params, templates=tmpl)
    got = np.array([r.templates["pol_angle"][f"d{i}"] for i in range(4)])
    assert np.abs(got + (delta - delta.mean())).max() < 1e-8


def _displaced(theta, phi, psi, dx, dy, frame):
    """The pointing moved by (dx, dy) [rad] along x = cos b e_theta + sin b
    e_phi, y = -sin b e_theta + cos b e_phi, b = psi + frame (as in
    docs/usage.md), with psi parallel-transported along the geodesic."""
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    n = np.array([st * cp, st * sp, ct])
    et, ep = np.array([ct * cp, ct * sp, -st]), np.array([-sp, cp, 0 * sp])
    rho = np.hypot(dx, dy)
    if rho == 0:
        return theta, phi, psi
    b = psi + frame
    t = (dx * (np.cos(b) * et + np.sin(b) * ep) + dy * (-np.sin(b) * et + np.cos(b) * ep)) / rho
    w = np.cross(n.T, t.T).T
    n2, t2 = np.cos(rho) * n + np.sin(rho) * t, -np.sin(rho) * n + np.cos(rho) * t
    p = np.cos(psi) * et + np.sin(psi) * ep
    p2 = np.sum(p * t, 0) * t2 + np.sum(p * w, 0) * w
    th2, ph2 = np.arccos(np.clip(n2[2], -1, 1)), np.arctan2(n2[1], n2[0]) % (2 * np.pi)
    et2 = np.array([np.cos(th2) * np.cos(ph2), np.cos(th2) * np.sin(ph2), -np.sin(th2)])
    ep2 = np.array([-np.sin(ph2), np.cos(ph2), 0 * ph2])
    return th2, ph2, np.arctan2(np.sum(p2 * ep2, 0), np.sum(p2 * et2, 0))


def _observe(alm, theta, phi, angle):
    """I + Q cos 2a + U sin 2a of the alm (T, E, B) at (theta, phi)."""
    lmax = hp.Alm.getlmax(alm.shape[1])
    loc = np.stack([theta, phi], axis=1)
    i = ducc0.sht.synthesis_general(alm=alm[:1], spin=0, lmax=lmax, loc=loc, epsilon=1e-12)[0]
    q, u = ducc0.sht.synthesis_general(alm=alm[1:], spin=2, lmax=lmax, loc=loc, epsilon=1e-12)
    return i + q * np.cos(2 * angle) + u * np.sin(2 * angle)


def test_beam_templates():
    """Pointing offsets, ellipticities and FWHM of the beams: beam_templates
    (TOD templates, the derivatives of the sky model along the axes of the
    detectors) and fwhm_template (the sky template of the
    Laplacian, -l(l+1)/2 on T, -(l(l+1)-4)/2 on E, B). The
    TOD is the sky through elliptical Gaussian beams (Gauss-Hermite
    quadrature, with the offset) minus the same through the circular beam of
    the model, plus the model: the fit recovers the offsets and the second
    moments about the nominal pointing (the beam's plus rho rho^T) to second
    order. Without HWP the ellipticity in the polarization frame common to
    the detectors is degenerate with Q/U (sum_zero); with an HWP it is not.
    beam_parameters subtracts rho rho^T and gives the beam axes."""
    from pysanepic import beam_parameters, beam_templates, fwhm_template

    nside, n, fs = 16, 120_000, 10.0
    lmax, sig0 = 3 * nside - 1, np.radians(1.0)
    rng = np.random.default_rng(3)
    ell = np.arange(lmax + 1)
    sky = np.array([hp.synalm(f * 1e-8 / (1 + ell), lmax=lmax) for f in (1.0, 0.1, 0.03)])
    # the circular beam of the quadrature: a Gaussian in the tangent plane
    r = np.linspace(0, 12 * sig0, 20001)
    bl = [np.trapezoid(r / sig0**2 * np.exp(-r**2 / 2 / sig0**2) * np.polynomial.legendre.Legendre.basis(k)(np.cos(r)), r)
          for k in ell]
    model = np.array([hp.almxfl(a, bl) for a in sky])
    gh, gw = np.polynomial.hermite.hermgauss(5)
    pols = np.radians([0, 90, 45, 135])
    scans = [_scan(n, rng) for _ in pols]
    chi = 2 * np.pi * np.arange(n) / 600.0
    params = GLSParameters(chunk_s=1200.0, tol=1e-24, angle_dtype=np.float64)
    draw = rng.normal(size=(len(pols), 5))

    def run(scale, hwp):
        off = scale * np.radians(3 / 60) * draw[:, :2]  # a few arcmin
        c = scale * sig0**2 * (0.05 * draw[:, 2:4] + [0.04, -0.03])  # c+, cx, with a common part
        ds2 = scale * sig0**2 * 0.03 * draw[:, 4]
        # second moments about the nominal pointing; without HWP sum to zero (see the docstring)
        c_eff = c + np.stack([(off[:, 0] ** 2 - off[:, 1] ** 2) / 2, off[:, 0] * off[:, 1]], axis=1)
        if not hwp:
            c, c_eff = c - c_eff.mean(0), c_eff - c_eff.mean(0)
        ds2 -= (ds2 + (off**2).sum(1) / 2).mean()
        ds2_eff = ds2 + (off**2).sum(1) / 2
        data = []
        for i, ((theta, phi, psi), pol) in enumerate(zip(scans, pols)):
            def angle(p, pol=pol):
                return p + 2 * chi - pol if hwp else p + pol

            tod = _observe(model, theta, phi, angle(psi))
            cov = (sig0**2 + ds2[i]) * np.eye(2) + [[c[i, 0], c[i, 1]], [c[i, 1], -c[i, 0]]]
            for sign, L, o in ((1, np.linalg.cholesky(cov), off[i]), (-1, sig0 * np.eye(2), 0 * off[i])):
                for a, wa in zip(gh, gw):
                    for b, wb in zip(gh, gw):
                        u = L @ (np.sqrt(2) * np.array([a, b])) + o
                        th, ph, ps = _displaced(theta, phi, psi, u[0], u[1], pol)
                        tod += sign * wa * wb / np.pi * _observe(sky, th, ph, angle(ps))
            data.append(DetectorData(tod=tod, pointing=Pointing(theta=theta, phi=phi, psi=psi, coordinates="E"),
                                     sampling_rate_hz=fs, noise=OneOverFNoise(net_ukrts=40.0), pol_angle_rad=pol,
                                     hwp_angle=chi if hwp else None, name=f"d{i}", templates={"other": 0 * tod}))
        data, tmpl = beam_templates(data, model, "E")
        assert [t.sum_zero for t in tmpl.values()] == [False, False, not hwp, not hwp]
        assert set(data[0].templates) == {"other", "offset_x", "offset_y", "ell_plus", "ell_cross"}
        tmpl["fwhm"] = fwhm_template(model)
        res = make_maps(data, nside, coordinates="E", params=params, sky_model=model, templates=tmpl)
        assert res.converged
        err = {}
        # relative to the largest offset, ellipticity and FWHM change
        for k, truth, norm in (("offset_x", off[:, 0], off), ("offset_y", off[:, 1], off), ("ell_plus", c_eff[:, 0], c_eff),
                               ("ell_cross", c_eff[:, 1], c_eff), ("fwhm", ds2_eff - ds2_eff.mean(), ds2_eff - ds2_eff.mean())):
            got = np.array([res.templates[k][f"d{i}"] for i in range(len(pols))])
            err[k] = np.abs(got - truth).max() / np.abs(norm).max()
        # the beams: rho rho^T subtracted; c relative to the mean without HWP
        par = beam_parameters(res, tmpl, sigma=sig0)
        cc = c if hwp else c - c.mean(0)
        for k, truth, norm in (("c_plus", cc[:, 0], cc), ("c_cross", cc[:, 1], cc), ("delta_sigma2", ds2 - ds2.mean(), ds2)):
            got = np.array([par[f"d{i}"][k] for i in range(len(pols))])
            err["par_" + k] = np.abs(got - truth).max() / np.abs(norm).max()
        if hwp:
            h = np.hypot(c[:, 0], c[:, 1])
            for k, truth in (("sigma_major", np.sqrt(sig0**2 + ds2 + h)), ("sigma_minor", np.sqrt(sig0**2 + ds2 - h))):
                got = np.array([par[f"d{i}"][k] for i in range(len(pols))])
                err["par_" + k] = np.abs(got / truth - 1).max()
            got = np.array([par[f"d{i}"]["angle"] for i in range(len(pols))])
            err["par_angle"] = np.abs(np.angle(np.exp(2j * (got - np.arctan2(c[:, 1], c[:, 0]) / 2)))).max() / 2
        return err

    # offsets of a few arcmin and ellipticities of a few % at FWHM 2.4 deg, l <= 47:
    # ~1%; 10 times smaller: ~0.1%, the second order
    for scale, tol in ((1.0, 2e-2), (0.1, 2e-3)):
        err = run(scale, hwp=False)
        assert max(err.values()) < tol, (scale, err)
    err = run(1.0, hwp=True)
    assert max(err.values()) < 2e-2, err


def test_beam_tods_derivatives():
    """beam_tods (spin-raising and lowering operators) against finite
    differences of the sky resampled with the beam moved along the axes of
    the detector, with psi transported: T, E and B separately, with and
    without HWP, pol_efficiency 0.9; equal to the truncation of the
    differences, (l eps)^2 / 12."""
    from pysanepic import beam_tods
    from pysanepic.mapmaker import _sample

    rng = np.random.default_rng(5)
    lmax, eps = 40, 2e-4
    ell = np.arange(lmax + 1)
    base = np.array([hp.synalm(1 / (1 + ell) ** 2, lmax=lmax) for _ in range(3)])
    theta, phi, psi = _scan(20000, rng)
    chi = rng.uniform(0, 2 * np.pi, len(psi))
    for k in range(3):
        alm = base * (np.arange(3) == k)[:, None]
        for hwp in (None, chi):
            d = DetectorData(tod=0 * psi, pointing=Pointing(theta=theta, phi=phi, psi=psi, coordinates="E"),
                             sampling_rate_hz=1.0, noise=OneOverFNoise(net_ukrts=1.0), pol_angle_rad=0.3,
                             pol_efficiency=0.9, hwp_angle=hwp)
            got = beam_tods(d, alm, "E")
            s = {}
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
                th, ph, ps = _displaced(theta, phi, psi, dx * eps, dy * eps, d.pol_angle_rad)
                angle = ps + d.pol_angle_rad if hwp is None else ps + 2 * hwp - d.pol_angle_rad
                s[dx, dy] = _sample(alm, d, None, angle, np.stack([th, ph], axis=1), 1)
            fd = {"offset_x": (s[1, 0] - s[-1, 0]) / (2 * eps), "offset_y": (s[0, 1] - s[0, -1]) / (2 * eps),
                  "ell_plus": (s[1, 0] + s[-1, 0] - s[0, 1] - s[0, -1]) / (2 * eps**2),
                  "ell_cross": (s[1, 1] + s[-1, -1] - s[1, -1] - s[-1, 1]) / (4 * eps**2)}
            for name in fd:
                assert np.std(got[name] - fd[name]) < 2e-5 * np.std(fd[name]), (k, hwp is None, name)


def _gauss_blm(lmax, mmax, fwhm, ellipticity=1.0, psi_ell=0.0, pol_angle=0.0):
    """The blm of an elliptical Gaussian beam, (3, nalm(lmax, mmax)), as
    litebird_sim's gauss_beam_to_alm (Planck LevelS): FWHM sqrt(major minor),
    ellipticity major / minor, major axis at psi_ell, polarization angle
    pol_angle (psi_pol_rad), no cross-polar leakage."""
    def iv(n, x):  # modified Bessel function I_n: exact trapezoid of a periodic integrand
        u = np.linspace(0, 2 * np.pi, 256, endpoint=False)
        return np.mean(np.exp(np.multiply.outer(x, np.cos(u))) * np.cos(n * u), axis=-1)

    b = np.zeros((3, (mmax + 1) * (lmax + 1) - mmax * (mmax + 1) // 2), complex)
    e2 = 1 - 1 / ellipticity**2
    rho = pol_angle - psi_ell
    for m in range(0, mmax + 1, 2):
        ell = np.arange(m, lmax + 1)
        t = ell * (ell + 1) * fwhm**2 * ellipticity / (8 * np.log(2))
        i = m * (2 * lmax + 1 - m) // 2 + ell
        g = np.exp(-0.5 * t * (1 - e2 / 2))
        b[0, i] = np.sqrt((2 * ell + 1) / (4 * np.pi)) * g * iv(m // 2, t * e2 / 4)
        if m == 0:
            v = np.where(ell >= 2, np.sqrt((2 * ell + 1) / (8 * np.pi)) * g * iv(1, t * e2 / 4), 0)
            b[1, i], b[2, i] = v * np.cos(2 * rho), v * np.sin(2 * rho)
        else:
            b1 = np.exp(-2j * rho) * iv((m - 2) // 2, t * e2 / 4)
            b2 = np.exp(2j * rho) * iv((m + 2) // 2, t * e2 / 4)
            pre = np.sqrt((2 * ell + 1) / (32 * np.pi)) * g
            b[1, i], b[2, i] = pre * (b1 + b2), 1j * pre * (b1 - b2)
        b[:, i] *= np.exp(-1j * m * psi_ell)
    b[1:] *= -np.sqrt(2) * np.exp(fwhm**2 / (4 * np.log(2)))  # Challinor normalization, exp(2 sigma^2)
    return b


def _convolve(alm, blm, d):
    """The TOD of a DetectorData seeing `alm` through the beam `blm`
    (ducc0.totalconvolve, as litebird_sim without HWP)."""
    lmax = hp.Alm.getlmax(alm.shape[1])
    mmax = next(m for m in range(lmax + 1) if (m + 1) * (lmax + 1) - m * (m + 1) // 2 == blm.shape[1])
    inter = ducc0.totalconvolve.Interpolator(sky=alm, beam=blm, separate=False, lmax=lmax, kmax=mmax, epsilon=1e-12,
                                             nthreads=1)
    p = d.pointing
    return inter.interpol(np.stack([p.theta, p.phi, p.psi], axis=1))[0]


def _symmetric_sky(sky, blm, pol_angle):
    """`sky` through the symmetric part of the beam `blm`, a circular beam:
    b_l of T from m = 0 and of Q/U from m = 2."""
    from pysanepic.templates import _asymmetric

    lmax = hp.Alm.getlmax(sky.shape[1])
    ell = np.arange(lmax + 1)
    sym = blm - _asymmetric(blm, lmax, 2)
    i2 = 2 * (2 * lmax - 1) // 2 + ell[2:]
    bt = sym[0, : lmax + 1].real / np.sqrt((2 * ell + 1) / (4 * np.pi))
    norm = -np.sqrt(2) * np.exp(-2j * pol_angle) * np.sqrt((2 * ell[2:] + 1) / (32 * np.pi))
    bp = np.concatenate([[0, 0], (sym[1, i2] / norm).real])
    return np.array([hp.almxfl(sky[0], bt), hp.almxfl(sky[1], bp), hp.almxfl(sky[2], bp)])


def test_blm_tods():
    """blm_tods: 0 for a circular beam (the beam of the maps); for a slightly
    elliptical one, equal to first order to the ellipticity templates of
    beam_tods (sky through the symmetric part of the beam), with
    c+ + i cx = (sigma_major^2 - sigma_minor^2) / 2 e^{2i (psi_ell -
    pol_angle)}: the major axis from the x of the detector, at its
    polarization angle. T and E/B, pol_angle 0.3. Not to the second order
    only: LevelS's formula is a flat-sky expansion, which differs from the
    exact derivatives on the sphere by a part decreasing with l (0.4% here,
    1% for P at l ~ 15, 0.1% at l ~ 80), independent of the ellipticity."""
    from pysanepic import beam_tods, blm_tods
    from pysanepic.templates import _asymmetric

    rng = np.random.default_rng(2)
    lmax, mmax, fwhm, pol = 128, 4, np.radians(3.0), 0.3
    ell = np.arange(lmax + 1)
    sky = np.array([hp.synalm(1 / (ell + 5.0) ** 2, lmax=lmax) for _ in range(3)])
    theta, phi, psi = _scan(20000, rng)
    d = DetectorData(tod=0 * psi, pointing=Pointing(theta=theta, phi=phi, psi=psi, coordinates="E"),
                     sampling_rate_hz=1.0, noise=OneOverFNoise(net_ukrts=1.0), pol_angle_rad=pol)
    circ = blm_tods(d, sky, _gauss_blm(lmax, mmax, fwhm, pol_angle=pol), "E")
    assert np.abs(circ).max() < 1e-12 * np.abs(_convolve(sky, _gauss_blm(lmax, mmax, fwhm, pol_angle=pol), d)).max()
    for e in (1.02, 1.002):
        b = _gauss_blm(lmax, mmax, fwhm, ellipticity=e, psi_ell=0.4, pol_angle=pol)
        t = beam_tods(d, _symmetric_sky(sky, b, pol), "E", offsets=False)
        h = fwhm**2 / (8 * np.log(2)) * (e - 1 / e) / 2
        pred = h * (np.cos(2 * (0.4 - pol)) * t["ell_plus"] + np.sin(2 * (0.4 - pol)) * t["ell_cross"])
        got = blm_tods(d, sky, b, "E", epsilon=1e-12)
        assert np.std(got - pred) < 1e-2 * np.std(got), e
        assert np.allclose(got + _convolve(sky, b - _asymmetric(b, lmax, mmax), d), _convolve(sky, b, d), atol=1e-12)


def test_blm_fixed_template():
    """Four detectors with elliptical beams of different axes, no HWP, noiseless:
    with the fixed template of blm_templates, the maps (and with calibration
    the gains) are those of the same detectors with the symmetric part of
    their beams; without it, the T -> P leakage of the ellipticity, degenerate
    with Q/U, is in the maps."""
    from pysanepic import CalibrationParameters, blm_templates
    from pysanepic.templates import _asymmetric

    rng = np.random.default_rng(7)
    nside, lmax, mmax, n, fs, fwhm = 16, 40, 4, 60_000, 10.0, np.radians(4.0)
    ell = np.arange(lmax + 1)
    sky = np.array([hp.synalm(1e-8 / (ell + 3.0) ** 2, lmax=lmax) for _ in range(3)])
    sky[0] *= 10  # T brighter than Q/U
    solar = 3.36e-3 * np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    full, sym, blms, gains = [], [], {}, []
    for i, pol in enumerate([0.0, np.pi / 4, np.pi / 8, 3 * np.pi / 8]):
        theta, phi, psi = _scan(n, rng)
        nvec = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        d = DetectorData(tod=0 * psi, pointing=Pointing(theta=theta, phi=phi, psi=psi, coordinates="E"),
                         sampling_rate_hz=fs, noise=OneOverFNoise(net_ukrts=40.0, fknee_hz=0.05, alpha=1.0, fmin_hz=1e-4),
                         pol_angle_rad=pol, name=f"d{i}", dipole=solar @ nvec)
        b = _gauss_blm(lmax, mmax, fwhm, ellipticity=1.2, psi_ell=0.3 * i + 0.2, pol_angle=pol)
        blms[d.name] = b
        full.append(dataclasses.replace(d, tod=_convolve(sky, b, d)))
        sym.append(dataclasses.replace(d, tod=_convolve(sky, b - _asymmetric(b, lmax, mmax), d)))
        gains.append(1 + 0.01 * rng.normal(size=5))
    # the sky through the symmetric part of the beams (the same for all) as sky
    # model: an exact model, without the signal below the pixel size
    model = _symmetric_sky(sky, blms["d0"], 0.0)
    params = GLSParameters(chunk_s=1200.0, tol=1e-24, angle_dtype=np.float64)
    data, templates = blm_templates(full, sky, blms, "E", epsilon=1e-12)
    assert templates == {"beam": Template(fixed=True)}
    ref = make_maps(sym, nside, "E", params, sky_model=model)
    r = make_maps(data, nside, "E", params, templates=templates, sky_model=model)
    seen = ref.maps[1] != hp.UNSEEN
    scale = np.std(ref.maps[1:, seen])
    assert r.templates is None
    assert np.abs(r.maps[1:, seen] - ref.maps[1:, seen]).max() < 1e-8 * scale
    assert np.abs(make_maps(data, nside, "E", params, sky_model=model).maps[1:, seen]
                  - ref.maps[1:, seen]).max() > 1e-2 * scale
    # calibration: the fixed template is in the model times the gain
    cal = CalibrationParameters(gain_period_s=n / 5 / fs, n_iter=3)
    t = np.arange(n) * 5 // n
    data, templates = blm_templates([dataclasses.replace(d, tod=g[t] * (d.tod + d.dipole)) for d, g in zip(full, gains)],
                                    sky, blms, "E", epsilon=1e-12)
    r = make_maps(data, nside, "E", params, calibration=cal, templates=templates, sky_model=model)
    assert max(np.abs(g - g0).max() for g, g0 in zip(r.gains, gains)) < 1e-10
    assert np.abs(r.maps[1:, seen] - ref.maps[1:, seen]).max() < 1e-7 * scale


def test_blm_mmax(monkeypatch):
    """blm_tods with a beam of mmax = lmax (as generate_gauss_beam_alms): mmax
    of the convolution from the asymmetries, a few m, the TOD as with the
    whole beam to epsilon;
    blm_templates: one Interpolator per detector for all its DetectorData."""
    from pysanepic import blm_templates, blm_tods

    rng = np.random.default_rng(3)
    lmax, fwhm = 64, np.radians(3.0)
    ell = np.arange(lmax + 1)
    sky = np.array([hp.synalm(1 / (ell + 5.0) ** 2, lmax=lmax) for _ in range(3)])
    b = _gauss_blm(lmax, lmax, fwhm, ellipticity=1.1, psi_ell=0.4, pol_angle=0.3)
    built = []
    interpolator = ducc0.totalconvolve.Interpolator

    def counted(**kw):
        built.append(kw["kmax"])
        return interpolator(**kw)

    monkeypatch.setattr(ducc0.totalconvolve, "Interpolator", counted)
    data = []
    for name in ("d0", "d0", "d1"):
        theta, phi, psi = _scan(5000, rng)
        data.append(DetectorData(tod=0 * psi, pointing=Pointing(theta=theta, phi=phi, psi=psi, coordinates="E"),
                                 sampling_rate_hz=1.0, noise=OneOverFNoise(net_ukrts=1.0), pol_angle_rad=0.3,
                                 name=name))
    exact = blm_tods(data[0], sky, b, "E", mmax=lmax, epsilon=1e-12)
    got = blm_tods(data[0], sky, b, "E")
    assert built[1] < 12
    assert np.std(got - exact) < 1e-5 * np.std(exact)
    built.clear()
    out, _ = blm_templates(data, sky, {"d0": b, "d1": b}, "E")
    assert len(built) == 2 and [d.name for d in out] == ["d0", "d0", "d1"]
    for d, o in zip(data, out):
        assert np.array_equal(o.templates["beam"], blm_tods(d, sky, b, "E"))
    with pytest.raises(ValueError, match="mmax must be"):
        blm_tods(data[0], sky, b[:, :3 * (lmax + 1) - 3], "E", mmax=3)


def test_blm_tods_model_bl():
    """blm_tods with model_bl, the window of the beam of the sky model: 0 for
    the circular beam of that window (the conventions of litebird_sim's
    gauss_beam_to_alm, as _gauss_blm, and healpy's gauss_beam agree), and for an elliptical
    beam the sky through it minus the sky through that circular beam."""
    from pysanepic import blm_tods

    rng = np.random.default_rng(3)
    lmax, mmax, fwhm, pol = 128, 4, np.radians(3.0), 0.3
    ell = np.arange(lmax + 1)
    sky = np.array([hp.synalm(1 / (ell + 5.0) ** 2, lmax=lmax) for _ in range(3)])
    theta, phi, psi = _scan(20000, rng)
    d = DetectorData(tod=0 * psi, pointing=Pointing(theta=theta, phi=phi, psi=psi, coordinates="E"),
                     sampling_rate_hz=1.0, noise=OneOverFNoise(net_ukrts=1.0), pol_angle_rad=pol)
    bl = hp.gauss_beam(fwhm, lmax=lmax, pol=True)
    circ = _gauss_blm(lmax, mmax, fwhm, pol_angle=pol)
    scale = np.abs(_convolve(sky, circ, d)).max()
    assert np.abs(blm_tods(d, sky, circ, "E", model_bl=bl, epsilon=1e-12)).max() < 1e-12 * scale
    assert np.abs(blm_tods(d, sky[:1], circ[:1], "E", model_bl=bl[:, 0], epsilon=1e-12)).max() < 1e-12 * scale
    b = _gauss_blm(lmax, mmax, fwhm * 1.01, ellipticity=1.2, psi_ell=0.4, pol_angle=pol)
    assert np.allclose(blm_tods(d, sky, b, "E", model_bl=bl, epsilon=1e-12) + _convolve(sky, circ, d), _convolve(sky, b, d), atol=1e-12)
    with pytest.raises(ValueError, match="E = B"):
        blm_tods(d, sky, b, "E", model_bl=bl[:, 0], epsilon=1e-12)


def test_blm_fixed_template_model_bl():
    """Four detectors with different ellipticities (as different FWHMs, their
    symmetric parts differ), no HWP, noiseless, with calibration on the
    dipole and the sky model through a circular beam: with model_bl the
    fixed template makes the model exact, gains and maps; with the beams
    minus their own symmetric parts, the part of the symmetric beams beyond
    the mean (order l^4 of the ellipticity, ~(e - 1)^2) biases the gains."""
    from pysanepic import CalibrationParameters, blm_templates

    rng = np.random.default_rng(11)
    nside, lmax, mmax, n, fs, fwhm = 16, 40, 4, 60_000, 10.0, np.radians(4.0)
    ell = np.arange(lmax + 1)
    sky = np.array([hp.synalm(1e-8 / (ell + 3.0) ** 2, lmax=lmax) for _ in range(3)])
    sky[0] *= 10  # T brighter than Q/U
    bl = hp.gauss_beam(fwhm, lmax=lmax, pol=True)
    model = np.array([hp.almxfl(a, bl[:, k]) for k, a in enumerate(sky)])
    solar = 3.36e-3 * np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    data, blms, gains = [], {}, []
    t = np.arange(n) * 5 // n
    for i, (pol, e) in enumerate(zip([0.0, np.pi / 4, np.pi / 8, 3 * np.pi / 8], [1.0, 1.1, 1.2, 1.3])):
        theta, phi, psi = _scan(n, rng)
        nvec = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        d = DetectorData(tod=0 * psi, pointing=Pointing(theta=theta, phi=phi, psi=psi, coordinates="E"),
                         sampling_rate_hz=fs, noise=OneOverFNoise(net_ukrts=40.0, fknee_hz=0.05, alpha=1.0, fmin_hz=1e-4),
                         pol_angle_rad=pol, name=f"d{i}", dipole=solar @ nvec)
        b = _gauss_blm(lmax, mmax, fwhm, ellipticity=e, psi_ell=0.3 * i + 0.2, pol_angle=pol)
        blms[d.name] = b
        gains.append(1 + 0.01 * rng.normal(size=5))
        data.append(dataclasses.replace(d, tod=gains[-1][t] * (_convolve(sky, b, d) + d.dipole)))
    params = GLSParameters(chunk_s=1200.0, tol=1e-24, angle_dtype=np.float64)
    cal = CalibrationParameters(gain_period_s=n / 5 / fs, n_iter=4)

    def gain_error(**kw):
        dd, templates = blm_templates(data, sky, blms, "E", epsilon=1e-12, **kw)
        r = make_maps(dd, nside, "E", params, calibration=cal, templates=templates, sky_model=model)
        return max(np.abs(g / g0 - 1).max() for g, g0 in zip(r.gains, gains)), r

    err, r = gain_error(model_bl=bl)
    assert err < 1e-9
    # the maps of the same detectors with the circular beam of the model
    circ = [dataclasses.replace(d, tod=_convolve(sky, _gauss_blm(lmax, mmax, fwhm, pol_angle=d.pol_angle_rad), d))
            for d in data]
    ref = make_maps(circ, nside, "E", params, sky_model=model)
    seen = ref.maps[1] != hp.UNSEEN
    assert np.abs(r.maps[:, seen] - ref.maps[:, seen]).max() < 1e-7 * np.std(ref.maps[1:, seen])
    assert gain_error()[0] > 1e3 * err


def test_template_helpers_errors():
    """The checks of the template helpers."""
    from pysanepic import beam_parameters, beam_templates, beam_tods, fwhm_template

    rng = np.random.default_rng(0)
    alm = np.array([hp.synalm(np.ones(9), lmax=8) for _ in range(3)])
    theta, phi, psi = _scan(100, rng)
    d = DetectorData(tod=np.zeros(100), pointing=Pointing(theta=theta, phi=phi, psi=psi, coordinates="E"),
                     sampling_rate_hz=1.0, noise=OneOverFNoise(net_ukrts=1.0), name="d0")
    assert set(beam_tods(d, alm, "E", ellipticity=False)) == {"offset_x", "offset_y"}
    assert set(beam_tods(d, alm[:1], "E", offsets=False)) == {"ell_plus", "ell_cross"}
    with pytest.raises(ValueError, match="3 components"):
        pol_angle_template(alm[:1])
    with pytest.raises(ValueError, match="alm"):
        fwhm_template(hp.alm2map(alm, 4))
    with pytest.raises(ValueError, match="alm"):
        beam_tods(d, hp.alm2map(alm, 4), "E")
    with pytest.raises(ValueError, match="nothing"):
        beam_tods(d, alm, "E", offsets=False, ellipticity=False)
    with pytest.raises(ValueError, match="angles"):
        beam_tods(dataclasses.replace(d, pointing=Pointing(pix=np.zeros(100, int), nside=4, psi=psi, coordinates="E")),
                  alm, "E")
    with pytest.raises(ValueError, match="with and without HWP"):
        beam_templates([d, dataclasses.replace(d, hwp_angle=psi)], alm, "E")
    with pytest.raises(ValueError, match="already has"):
        beam_templates([dataclasses.replace(d, templates={"offset_x": psi})], alm, "E")
    with pytest.raises(ValueError, match="none of"):
        beam_parameters(None, {"thermal": Template()})
    from pysanepic import blm_templates, blm_tods

    blm = _gauss_blm(8, 2, 0.1)
    with pytest.raises(NotImplementedError, match="HWP"):
        blm_tods(dataclasses.replace(d, hwp_angle=psi), alm, blm, "E")
    with pytest.raises(ValueError, match="alm"):
        blm_tods(d, hp.alm2map(alm, 4), blm, "E")
    with pytest.raises(ValueError, match="T, E, B"):
        blm_tods(d, alm, blm[:1], "E")
    with pytest.raises(ValueError, match="lmax of sky_alm"):
        blm_tods(d, alm, _gauss_blm(9, 2, 0.1), "E")
    assert blm_tods(d, alm[:1], blm, "E").shape == (100,)  # T only: the T beam
    with pytest.raises(ValueError, match="no beam"):
        blm_templates([d], alm, {"d1": blm}, "E")
    with pytest.raises(ValueError, match="already has"):
        blm_templates([dataclasses.replace(d, templates={"beam": psi})], alm, {"d0": blm}, "E")
    for bad in (dict(sum_zero=True), dict(common=True), dict(period_s=1.0), dict(mask=np.ones(12))):
        with pytest.raises(ValueError, match="fixed"):
            Template(fixed=True, **bad)


def test_coarse_preconditioner():
    """The two-level preconditioner (coarse_nside) gives the same maps in
    fewer iterations with a common mode, also with padding (virtual pixels)
    and flagged samples; invalid values are rejected."""
    from pysanepic import CommonMode, DetectorGroup

    nside, fs, chunk = 16, 10.0, 6000
    rng = np.random.default_rng(21)
    sky = hp.smoothing(rng.normal(0, 1e-4, (3, 12 * nside**2)), fwhm=np.radians(6))
    own = OneOverFNoise(net_ukrts=50.0, fknee_hz=0.05, alpha=1.0, fmin_hz=1e-4)
    coupling = np.full(4, np.sqrt(0.5))
    dets = _group_data(rng, nside, sky, fs, chunk, 6, own, own, coupling)
    flags = np.zeros(len(dets[0].tod), bool)
    flags[1000:1010] = True
    dets[1] = dataclasses.replace(dets[1], flags=flags)
    group = [DetectorGroup(detectors=dets, modes=[CommonMode(coupling=coupling, noise=own)])]
    params = GLSParameters(chunk_s=chunk / fs, tol=1e-14, pad_s=60.0, split_gap_s=np.inf, coarse_nside=None)
    ref = make_maps(group, nside, coordinates="E", params=params)
    for nc in (1, 2):
        r = make_maps(group, nside, coordinates="E", params=dataclasses.replace(params, coarse_nside=nc))
        assert r.converged and r.iterations < ref.iterations - 5, (nc, r.iterations, ref.iterations)
        seen = ref.maps != hp.UNSEEN
        assert np.array_equal(seen, r.maps != hp.UNSEEN)
        assert np.abs(r.maps[seen] - ref.maps[seen]).max() < 1e-5 * np.abs(ref.maps[seen]).max()
    for bad in (3, 32, -1):
        with pytest.raises(ValueError, match="coarse_nside"):
            make_maps(group, nside, coordinates="E", params=dataclasses.replace(params, coarse_nside=bad))


def test_coarse_monopole():
    """The coarse level on the I monopole alone (coarse_nside=0, the default):
    the same maps in fewer iterations than without (None), without and with
    calibration."""
    from pysanepic import CalibrationParameters

    nside, n, fs, per = 16, 120_000, 10.0, 12_000
    data, _, _ = _calibration_data(nside, n, fs, per)
    rng = np.random.default_rng(41)
    noisy = [dataclasses.replace(d, tod=d.tod + 2e-5 * np.cumsum(rng.normal(size=n)) / np.sqrt(n) * 30
                                 + 1e-5 * rng.normal(size=n)) for d in data]
    params = GLSParameters(chunk_s=1200.0, tol=1e-12)
    for cal in (None, CalibrationParameters(gain_period_s=per / fs)):
        kw = dict(coordinates="E", calibration=cal)
        if cal is None:
            kw["data"] = [dataclasses.replace(d, tod=d.tod - d.dipole) for d in noisy]
        else:
            kw["data"] = noisy
        off = make_maps(nside=nside, params=dataclasses.replace(params, coarse_nside=None), **kw)
        on = make_maps(nside=nside, params=params, **kw)
        assert on.iterations <= off.iterations - 5, (cal, on.iterations, off.iterations)  # 35 -> 25, 49 -> 39
        seen = off.maps != hp.UNSEEN
        assert np.array_equal(seen, on.maps != hp.UNSEEN)
        assert np.abs(on.maps[seen] - off.maps[seen]).max() < 1e-4 * np.abs(off.maps[seen]).max()


def test_estimate_common_modes():
    """estimate_noise on DetectorGroups (the same detectors on several groups of
    days, one with flagged samples) finds the couplings and spectra of the
    common mode and the own noise, and the maps with the estimate are as good
    as with the true model; invalid inputs are rejected."""
    from pysanepic import CommonMode, DetectorGroup, estimate_noise

    nside, fs, chunk = 16, 10.0, 36000
    rng = np.random.default_rng(22)
    sky = hp.smoothing(rng.normal(0, 1e-4, (3, 12 * nside**2)), fwhm=np.radians(6))
    own = OneOverFNoise(net_ukrts=40.0, fknee_hz=0.05, alpha=1.0, fmin_hz=1e-4)
    mode = OneOverFNoise(net_ukrts=30.0, fknee_hz=0.2, alpha=1.5, fmin_hz=1e-4)
    coupling = rng.uniform(0.6, 0.9, 6)
    groups = []
    for _ in range(2):  # e.g. two days of a wafer
        dets = _group_data(rng, nside, sky, fs, chunk, 3, own, mode, coupling)
        groups.append(DetectorGroup(detectors=dets, modes=[CommonMode(coupling=coupling, noise=mode)]))
    flags = np.zeros(len(dets[0].tod), bool)
    flags[5000:9000] = True
    groups[1].detectors[2] = dataclasses.replace(groups[1].detectors[2], flags=flags)
    params = GLSParameters(chunk_s=chunk / fs, tol=1e-10, coarse_nside=1)
    first = make_maps([d for g in groups for d in g.detectors], nside, coordinates="E", params=params)
    est = estimate_noise(groups, first, chunk_s=chunk / fs, nbins=30)
    names = tuple(d.name for d in dets)
    (cm,) = est[names]
    scale = np.sqrt(np.mean(coupling**2))
    assert np.allclose(cm.coupling * scale, coupling, rtol=0.03), cm.coupling * scale / coupling
    f = cm.noise.freqs_hz
    k = np.round(f * chunk / fs).astype(int)
    ratio_mode = mode.weights(chunk, fs)[k] / (cm.noise.weights(chunk, fs)[k] * scale**2)
    ratio_own = own.weights(chunk, fs)[k] / est[names[0]].weights(chunk, fs)[k]
    # few modes per bin at low frequencies, and the map absorbs part of the
    # noise at the scan harmonics: exact only at high frequencies
    high = f > 0.2
    assert np.all(np.abs(ratio_mode[high] - 1) < 0.1) and np.all(np.abs(ratio_own[high] - 1) < 0.1)
    assert abs(np.median(ratio_mode[f > 1e-3]) - 1) < 0.2 and abs(np.median(ratio_own[f > 1e-3]) - 1) < 0.2

    def rms_err(gs):
        r = make_maps(gs, nside, coordinates="E", params=params)
        seen = r.maps[1] != hp.UNSEEN
        return np.std((r.maps - sky)[0, seen])

    estimated = [DetectorGroup(detectors=[dataclasses.replace(d, noise=est[d.name]) for d in g.detectors], modes=est[names]) for g in groups]
    true_err, est_err = rms_err(groups), rms_err(estimated)
    ignored = np.std((first.maps - sky)[0, first.maps[1] != hp.UNSEEN])
    assert est_err < 1.02 * true_err < 0.9 * ignored, (est_err, true_err, ignored)

    with pytest.raises(ValueError, match="need a name"):
        estimate_noise([DetectorGroup(detectors=[dataclasses.replace(d, name=None) for d in dets], modes=[])], first)
    with pytest.raises(ValueError, match="cannot be estimated"):
        estimate_noise([DetectorGroup(detectors=dets[:2], modes=[])], first, nbins=10)
    with pytest.raises(ValueError, match="both alone"):
        estimate_noise([groups[0], dets[0]], first, nbins=10)


def _group_data(rng, nside, sky, fs, chunk, nchunk, own, mode, coupling):
    """Detectors observed at the same times (different pointings), each with
    its own noise plus coupling[i] times a common mode."""
    def draw(noise):
        w = noise.weights(chunk, fs)
        return np.concatenate([np.fft.irfft(np.fft.rfft(rng.normal(size=chunk)) / np.sqrt(w), n=chunk) for _ in range(nchunk)])

    n = chunk * nchunk
    c = draw(mode)
    dets = []
    for i, pa in enumerate(np.linspace(0, np.pi, len(coupling), endpoint=False)):
        th, ph, psi = _scan(n, rng)
        a = psi + pa
        p = hp.ang2pix(nside, th, ph)
        tod = sky[0, p] + sky[1, p] * np.cos(2 * a) + sky[2, p] * np.sin(2 * a) + draw(own) + coupling[i] * c
        dets.append(DetectorData(tod=tod, pointing=Pointing(theta=th, phi=ph, psi=psi, coordinates="E"),
                                 sampling_rate_hz=fs, noise=own, pol_angle_rad=pa, name=f"d{i}"))
    return dets


def test_detector_group():
    """make_maps with a DetectorGroup: noiseless data give back the sky; with a
    red common mode the maps are much better than with the mode on the
    diagonal; invalid groups are rejected."""
    from pysanepic import CommonMode, DetectorGroup, TabulatedNoise

    nside, fs, chunk = 16, 10.0, 6000
    rng = np.random.default_rng(18)
    sky = hp.smoothing(rng.normal(0, 1e-4, (3, 12 * nside**2)), fwhm=np.radians(6))
    own = OneOverFNoise(net_ukrts=50.0, fknee_hz=0.02, alpha=1.0, fmin_hz=1e-4)
    mode = OneOverFNoise(net_ukrts=20.0, fknee_hz=1.0, alpha=2.0, fmin_hz=1e-3)
    coupling = np.array([1.0, 0.9, 1.1, 1.05])
    dets = _group_data(rng, nside, sky, fs, chunk, 6, own, mode, coupling)
    group = DetectorGroup(detectors=dets, modes=[CommonMode(coupling=coupling, noise=mode)])
    params = GLSParameters(chunk_s=chunk / fs, tol=1e-10)

    # noiseless: the sky back. Q/U are set to 0 where they are not solved (too
    # few angles): that signal would leak into the other pixels through the
    # correlated noise
    p8 = dataclasses.replace(params, tol=1e-24, angle_dtype=np.float64)
    solved = make_maps([group], nside, coordinates="E", params=dataclasses.replace(p8, maxiter=0)).maps[1] != hp.UNSEEN
    sky0 = np.where(solved, sky, [[1.0], [0.0], [0.0]] * sky)
    noiseless = DetectorGroup(detectors=[dataclasses.replace(d, tod=_signal(d, sky0, nside)) for d in dets], modes=group.modes)
    r = make_maps([noiseless], nside, coordinates="E", params=p8)
    seen = r.hit_map > 0
    d = np.where(r.maps == hp.UNSEEN, 0.0, r.maps)[:, seen] - sky0[:, seen]
    d[0] -= d[0].mean()  # the I monopole is degenerate with the offsets of 1/f noise
    assert np.abs(d).max() < 1e-6 * np.abs(sky).max()

    f = np.geomspace(fs / chunk, fs / 2, 200)
    psd = lambda nz: 2 * (nz.net_ukrts * np.sqrt(fs) * 1e-6) ** 2 / fs * (f**nz.alpha + nz.fknee_hz**nz.alpha) / (f**nz.alpha + nz.fmin_hz**nz.alpha)  # noqa: E731
    diag = [dataclasses.replace(d, noise=TabulatedNoise(freqs_hz=f, psd=psd(own) + c**2 * psd(mode))) for d, c in zip(dets, coupling)]
    err = {}
    for name, data in (("diag", diag), ("group", [group])):
        r = make_maps(data, nside, coordinates="E", params=params)
        assert r.converged
        err[name] = np.std((r.maps - sky)[:, r.maps[1] != hp.UNSEEN])
    assert err["group"] < 0.6 * err["diag"]
    with pytest.raises(ValueError, match="coupling"):
        DetectorGroup(detectors=dets, modes=[CommonMode(coupling=coupling[:3], noise=mode)])
    with pytest.raises(ValueError, match="same samples"):
        DetectorGroup(detectors=[dets[0], dataclasses.replace(dets[1], sampling_rate_hz=5.0)], modes=[])


def _signal(d, sky, nside):
    """Noiseless TOD of a DetectorData (no HWP, Ecliptic pointing in the sky frame)."""
    p = hp.ang2pix(nside, d.pointing.theta, d.pointing.phi)
    a = d.pointing.psi + d.pol_angle_rad
    return sky[0, p] + sky[1, p] * np.cos(2 * a) + sky[2, p] * np.sin(2 * a)


def test_pixel_order():
    """Counting sort == stable argsort, also with empty and unhit pixels."""
    from pysanepic.gls import _pixel_order

    rng = np.random.default_rng(3)
    for npix, ns in [(1, 5), (50, 0), (50, 1000), (100_000, 20_000)]:
        ip = rng.integers(0, npix, ns).astype(np.int32)
        assert np.array_equal(_pixel_order(ip, npix), np.argsort(ip, kind="stable"))


def test_cos_sin_table():
    """Table + Taylor cos/sin == libm to 1e-15, also at the edges of [-pi, pi)
    (float32 rounding goes just outside) and after reducing unreduced angles;
    per-sample gamma."""
    from pysanepic.gls import _TRIG_COS, _TRIG_SIN, _cos_sin, _reduced

    rng = np.random.default_rng(4)
    edges = np.float32([-np.pi, np.pi, 0.0]).astype(float)
    a = np.concatenate([rng.uniform(-np.pi, np.pi, 100_000), edges, _reduced([7.0, -20.0, 1e4], float)])
    g = rng.uniform(0.5, 1.0, a.size)
    c, s = np.empty(a.size), np.empty(a.size)
    _cos_sin(a, g, c, s, _TRIG_COS, _TRIG_SIN)
    assert np.abs(c - g * np.cos(a)).max() < 1e-15 and np.abs(s - g * np.sin(a)).max() < 1e-15


def test_noise_models():
    """keshner and toast weights follow their formulas and agree for white noise;
    a tabulated PSD gives back the model it samples."""
    from pysanepic import one_over_f_weights, psd_weights

    ns, fs, sigma, fk, alpha, fmin = 4096, 10.0, 2e-4, 0.05, 1.3, 1e-3
    f = np.fft.rfftfreq(ns, 1 / fs)
    toast = one_over_f_weights(ns, fs, sigma, fk, alpha, fmin)
    keshner = one_over_f_weights(ns, fs, sigma, fk, alpha, fmin, model="keshner")
    assert np.allclose(1 / (toast * sigma**2), (f**alpha + fk**alpha) / (f**alpha + fmin**alpha))
    assert np.allclose(1 / (keshner * sigma**2), ((f**2 + fk**2) / (f**2 + fmin**2)) ** (alpha / 2))
    for model in ("toast", "keshner"):
        assert np.allclose(one_over_f_weights(ns, fs, sigma, 0.0, alpha, model=model), 1 / sigma**2)
    with pytest.raises(ValueError, match="noise model"):
        one_over_f_weights(ns, fs, sigma, fk, alpha, model="pink")

    # white noise: P = 2 sigma^2 / fs; a fine table of the keshner model
    assert np.allclose(psd_weights(ns, fs, np.array([1e-3, 5.0]), np.full(2, 2 * sigma**2 / fs)), 1 / sigma**2)
    ft = np.geomspace(f[1], f[-1], 2000)
    pt = 2 * sigma**2 / fs * ((ft**2 + fk**2) / (ft**2 + fmin**2)) ** (alpha / 2)
    assert np.allclose(psd_weights(ns, fs, ft, pt)[1:], keshner[1:], rtol=1e-4)


def test_noise_classes_in_make_maps():
    """OneOverFNoise and TabulatedNoise check their values; DetectorData needs
    one of them; a white tabulated PSD gives the same map as the NET model."""
    nside, n, fs, net = 8, 40_000, 10.0, 50.0
    rng = np.random.default_rng(5)
    theta, phi, psi = _scan(n, rng)
    ptg = Pointing(theta=theta, phi=phi, psi=psi, coordinates="G")
    tod = rng.normal(0, net * np.sqrt(fs) * 1e-6, n)
    with pytest.raises(ValueError, match="model"):
        OneOverFNoise(net_ukrts=net, model="pink")
    with pytest.raises(ValueError, match="net_ukrts"):
        OneOverFNoise(net_ukrts=0.0)
    for model in ("toast", "keshner"):  # alpha = 0 (litebird_sim's default with fknee = 0): white
        assert np.allclose(OneOverFNoise(net_ukrts=net, fknee_hz=0.05, alpha=0.0, model=model).weights(64, fs), 1 / (net * np.sqrt(fs) * 1e-6) ** 2)
    with pytest.raises(ValueError, match="frequencies"):
        TabulatedNoise(freqs_hz=[0.0, 1.0], psd=[1.0, 1.0])
    with pytest.raises(TypeError, match="noise"):
        DetectorData(tod=tod, pointing=ptg, sampling_rate_hz=fs, noise=net)
    sigma = net * np.sqrt(fs) * 1e-6
    white = TabulatedNoise(freqs_hz=[1e-3, 5.0], psd=np.full(2, 2 * sigma**2 / fs))
    params = GLSParameters(chunk_s=1000.0, tol=1e-14)
    maps = [
        make_maps([DetectorData(tod=tod, pointing=ptg, sampling_rate_hz=fs, noise=noise)],
                  nside, coordinates="G", params=params).maps
        for noise in (OneOverFNoise(net_ukrts=net), white)
    ]
    assert np.allclose(maps[0], maps[1], rtol=1e-8, atol=0)
    for model in ("toast", "keshner"):  # each model gives its own weights (toast is the default)
        noise = OneOverFNoise(net_ukrts=net, fknee_hz=0.05, alpha=1.3, fmin_hz=1e-3, **({} if model == "toast" else {"model": model}))
        assert np.allclose(noise.weights(4096, fs), one_over_f_weights(4096, fs, sigma, 0.05, 1.3, 1e-3, model))


def test_flags_exact_gls():
    """Flagged samples (pix < 0): the GLS solution equals the dense GLS on the
    unflagged samples only, with their exact covariance N_uu."""
    from pysanepic import GLS, one_over_f_weights

    rng = np.random.default_rng(7)
    npix, ns, nchunk, fs = 6, 128, 2, 10.0
    pix = rng.integers(0, npix, (nchunk, ns))
    psi = rng.uniform(0, np.pi, (nchunk, ns))
    tod = rng.normal(size=(nchunk, ns))
    w = one_over_f_weights(ns, fs, 1.0, 0.5, 1.0, 0.05)  # DC weight > 0: N invertible
    flags = rng.random((nchunk, ns)) < 0.1
    flags[0, 40:60] = True  # a gap
    one_each = -1 - np.arange(ns)  # a distinct negative label per sample: one unknown each
    g = GLS(np.where(flags, one_each, pix), psi, np.tile(w, (nchunk, 1)), min_pol_rcond=0.0, angle_dtype=np.float64)
    assert g.nflag == flags.sum()
    m, info = g.solve(np.where(flags, np.nan, tod), tol=1e-26, maxiter=500)
    assert info["converged"]

    lag = (np.arange(ns)[:, None] - np.arange(ns)[None, :]) % ns
    cov = np.linalg.inv(np.fft.irfft(w, n=ns)[lag])  # N of a chunk
    A, b = np.zeros((3 * npix, 3 * npix)), np.zeros(3 * npix)
    for k in range(nchunk):
        u = ~flags[k]
        P = np.zeros((ns, 3 * npix))
        P[np.arange(ns), pix[k]] = 1
        P[np.arange(ns), npix + pix[k]] = np.cos(2 * psi[k])
        P[np.arange(ns), 2 * npix + pix[k]] = np.sin(2 * psi[k])
        Ninv = np.linalg.inv(cov[np.ix_(u, u)])
        A += P[u].T @ Ninv @ P[u]
        b += P[u].T @ Ninv @ tod[k][u]
    ref = np.linalg.solve(A, b).reshape(3, npix)
    assert np.abs(m - ref).max() < 1e-11 * np.abs(ref).max()
    with pytest.raises(ValueError, match="flagged"):
        g.rhs(tod)


def test_flags_in_make_maps():
    """Glitches and a gap, flagged: noiseless maps recover the sky whatever the
    flagged TOD (NaN or huge), flagged samples are not hits; flags are checked."""
    nside, n, fs = 16, 120_000, 10.0
    rng = np.random.default_rng(3)
    sky = rng.normal(0, 1e-4, (3, 12 * nside**2))
    data = _four_detectors(n, fs, rng, sky, nside)
    params = GLSParameters(chunk_s=1200.0, tol=1e-24, min_pol_rcond=0.0, angle_dtype=np.float64)
    ref = make_maps(data, nside, coordinates="E", params=params)
    out = {}
    for bad in (np.nan, 1e3):
        flagged = []
        for d in data:
            f = np.random.default_rng(1).random(n) < 0.02
            f[5000:6000] = True
            flagged.append(f)
            # pad_s > 0 too: the padding must not use the flagged samples
        # the gap (100 s) is split with split_gap_s=60
        configs = [dict(pad_s=pad, flag_method=method, split_gap_s=split)
                   for pad in (0.0, 120.0) for method in ("marginalize", "offset") for split in (np.inf, 60.0)]
        res = [
            make_maps([dataclasses.replace(d, tod=np.where(f, bad, d.tod), flags=f) for d, f in zip(data, flagged)],
                      nside, coordinates="E", params=dataclasses.replace(params, **cfg))
            for cfg in configs
        ]
        out[bad] = res
    seen = ref.maps[1] != hp.UNSEEN
    for res in out.values():
        for r in res:
            assert np.abs(r.maps[:, seen] - sky[:, seen]).max() < 1e-8 * np.abs(sky).max()
    assert ref.hit_map.sum() - out[1e3][0].hit_map.sum() == sum(f.sum() for f in flagged)
    with pytest.raises(ValueError, match="flags"):
        dataclasses.replace(data[0], flags=np.zeros(n, int))
    with pytest.raises(ValueError, match="flag_method"):
        make_maps(data, nside, params=GLSParameters(flag_method="drop"))


def _dense_gls(pix, psi, cinv, npix, pol=True):
    """Dense P (per chunk) and A = sum P^T Cinv P for small problems."""
    ncomp = 3 if pol else 1
    Ps = []
    for k in range(pix.shape[0]):
        ns = pix.shape[1]
        P = np.zeros((ns, ncomp * npix))
        P[np.arange(ns), pix[k]] = 1
        if pol:
            P[np.arange(ns), npix + pix[k]] = np.cos(2 * psi[k])
            P[np.arange(ns), 2 * npix + pix[k]] = np.sin(2 * psi[k])
        Ps.append(P)
    return Ps, sum(P.T @ cinv @ P for P in Ps)


def test_intensity_only():
    """pol=False: (1, npix) maps, the intensity of an unpolarized sky recovered
    with make_maps, and GLS without angles."""
    from pysanepic import GLS, one_over_f_weights

    nside, n, fs = 16, 120_000, 10.0
    rng = np.random.default_rng(8)
    sky = rng.normal(0, 1e-4, (3, 12 * nside**2))
    sky[1:] = 0
    data = _four_detectors(n, fs, rng, sky, nside)
    r = make_maps(data, nside, coordinates="E", params=GLSParameters(chunk_s=1200.0, pol=False, tol=1e-24))
    seen = r.maps[0] != hp.UNSEEN
    assert r.maps.shape == (1, 12 * nside**2)
    assert np.abs(r.maps[0, seen] - sky[0, seen]).max() < 1e-10 * np.abs(sky[0]).max()

    pix = rng.integers(0, 5, (2, 64))
    tod = rng.normal(size=(2, 64))
    w = one_over_f_weights(64, 10.0, 1.0, 0.5, 1.0, 0.05)
    m, _ = GLS(pix, None, np.tile(w, (2, 1))).solve(tod, tol=1e-26)
    lag = (np.arange(64)[:, None] - np.arange(64)[None, :]) % 64
    Ps, A = _dense_gls(pix, None, np.fft.irfft(w, n=64)[lag], 5, pol=False)
    b = sum(P.T @ np.fft.irfft(w, n=64)[lag] @ t for P, t in zip(Ps, tod))
    assert m.shape == (1, 5) and np.allclose(m[0], np.linalg.solve(A, b), rtol=1e-10)


def test_low_level_api():
    """GLS.apply_A and GLS.rhs equal P^T N^-1 P m and P^T N^-1 d; solve from
    x0 = the solution stops at once."""
    from pysanepic import GLS, one_over_f_weights

    rng = np.random.default_rng(9)
    npix, ns, nchunk = 6, 128, 3
    pix = rng.integers(0, npix, (nchunk, ns))
    psi = rng.uniform(0, np.pi, (nchunk, ns))
    tod = rng.normal(size=(nchunk, ns))
    w = one_over_f_weights(ns, 10.0, 1.0, 0.5, 1.0, 0.05)
    g = GLS(pix, psi, np.tile(w, (nchunk, 1)), min_pol_rcond=0.0, angle_dtype=np.float64)
    lag = (np.arange(ns)[:, None] - np.arange(ns)[None, :]) % ns
    cinv = np.fft.irfft(w, n=ns)[lag]
    Ps, A = _dense_gls(pix, psi, cinv, npix)
    m = rng.normal(size=(3, npix))
    assert np.allclose(g.apply_A(m).ravel(), A @ m.ravel(), rtol=1e-11, atol=0)
    b = sum(P.T @ cinv @ t for P, t in zip(Ps, tod))
    assert np.allclose(g.rhs(tod).ravel(), b, rtol=1e-11, atol=0)
    ref = np.linalg.solve(A, b).reshape(3, npix)
    x, info = g.solve(tod, x0=ref, tol=1e-20)
    assert info["iterations"] <= 1 and np.abs(x - ref).max() < 1e-12 * np.abs(ref).max()


def test_relative_detector_weights():
    """White noise, detectors with very different NET: the GLS map is exactly
    the binned map weighted by 1/sigma^2 (relative weights are right)."""
    nside, n, fs = 8, 30_000, 10.0
    rng = np.random.default_rng(10)
    data, rows = [], []
    for net, pol_angle in [(30.0, 0.0), (300.0, np.pi / 4), (60.0, np.pi / 8), (150.0, 3 * np.pi / 8)]:
        theta, phi, psi = _scan(n, rng)
        sigma = net * np.sqrt(fs) * 1e-6
        tod = rng.normal(0, sigma, n)
        data.append(DetectorData(tod=tod, pointing=Pointing(theta=theta, phi=phi, psi=psi, coordinates="G"),
                                 sampling_rate_hz=fs, noise=OneOverFNoise(net_ukrts=net), pol_angle_rad=pol_angle))
        rows.append((hp.ang2pix(nside, theta, phi), psi + pol_angle, tod, 1 / sigma**2))
    # default min_pol_rcond: nearly degenerate Q/U pixels would amplify roundoff
    params = GLSParameters(chunk_s=1000.0, tol=1e-26, angle_dtype=np.float64)
    r = make_maps(data, nside, coordinates="G", params=params)

    def binned(weighted):
        npix = 12 * nside**2
        M, v = np.zeros((npix, 3, 3)), np.zeros((npix, 3))
        for p, a, t, wt in rows:
            wt = wt if weighted else 1.0
            u = np.stack([np.ones_like(a), np.cos(2 * a), np.sin(2 * a)], axis=1)
            np.add.at(M, p, wt * u[:, :, None] * u[:, None, :])
            np.add.at(v, p, wt * u * t[:, None])
        return M, v

    M, v = binned(True)
    seen = r.maps[1] != hp.UNSEEN
    ref = np.linalg.solve(M[seen], v[seen][..., None])[..., 0].T
    assert np.abs(r.maps[:, seen] - ref).max() < 1e-9 * np.abs(ref).max()
    M1, v1 = binned(False)  # equal weights: clearly different, so the test is sensitive
    flat = np.linalg.solve(M1[seen], v1[seen][..., None])[..., 0].T
    assert np.abs(flat - ref).max() > 0.1 * np.abs(ref).max()


def test_noise_variance_monte_carlo():
    """1/f noise drawn from the toast PSD written here: the scatter of the GLS
    maps over realizations matches diag((P^T N^-1 P)^-1), with N from the same
    PSD (so the weights of OneOverFNoise are checked against the noise)."""
    from pysanepic import GLS

    rng = np.random.default_rng(12)
    npix, ns, nchunk, fs, net, fk, alpha, fmin = 5, 256, 4, 10.0, 50.0, 0.3, 1.5, 0.02
    sigma = net * np.sqrt(fs) * 1e-6
    f = np.fft.rfftfreq(ns, 1 / fs)
    psd = sigma**2 * (f**alpha + fk**alpha) / (f**alpha + fmin**alpha)  # per-sample units
    pix = rng.integers(0, npix, (nchunk, ns))
    psi = rng.uniform(0, np.pi, (nchunk, ns))
    w = OneOverFNoise(net_ukrts=net, fknee_hz=fk, alpha=alpha, fmin_hz=fmin).weights(ns, fs)
    g = GLS(pix, psi, np.tile(w, (nchunk, 1)), min_pol_rcond=0.0, angle_dtype=np.float64)
    lag = (np.arange(ns)[:, None] - np.arange(ns)[None, :]) % ns
    N = np.fft.irfft(psd, n=ns)[lag]  # circulant covariance with eigenvalues psd
    _, A = _dense_gls(pix, psi, np.linalg.inv(N), npix)
    expected = np.diag(np.linalg.inv(A)).reshape(3, npix)
    nreal = 300
    maps = np.empty((nreal, 3, npix))
    for i in range(nreal):
        noise = np.fft.irfft(np.fft.rfft(rng.normal(size=(nchunk, ns)), axis=1) * np.sqrt(psd), n=ns, axis=1)
        maps[i], _ = g.solve(noise, tol=1e-20)
    ratio = maps.var(axis=0) / expected
    # each ratio has a relative error ~ sqrt(2 / nreal) = 8%
    assert np.all(np.abs(ratio - 1) < 0.35) and abs(ratio.mean() - 1) < 0.06


def _noise_data(rng, nside, sky, fs, chunk, nchunk, shape, sigma, names=True):
    """Detectors with circulant noise of PSD 2 sigma^2 / fs * shape(f) per chunk,
    each split in stretches of 4 chunks sharing a name."""
    data = []
    for j, pa in enumerate((0.0, np.pi / 4, np.pi / 8)):
        n = nchunk * chunk
        th, ph, psi = _scan(n, rng)
        a = psi + pa
        p = hp.ang2pix(nside, th, ph)
        f = np.fft.rfftfreq(chunk, 1 / fs)
        noise = np.concatenate([np.fft.irfft(np.fft.rfft(rng.normal(size=chunk)) * sigma * np.sqrt(shape(np.maximum(f, 1e-9))), n=chunk)
                                for _ in range(nchunk)])
        tod = sky[0, p] + sky[1, p] * np.cos(2 * a) + sky[2, p] * np.sin(2 * a) + noise
        for s0 in range(0, n, 4 * chunk):
            sl = slice(s0, s0 + 4 * chunk)
            data.append(DetectorData(tod=tod[sl], pointing=Pointing(theta=th[sl], phi=ph[sl], psi=psi[sl], coordinates="E"),
                                     sampling_rate_hz=fs, noise=OneOverFNoise(net_ukrts=50.0), pol_angle_rad=pa,
                                     name=f"det{j}" if names else None))
    return data


def test_estimate_noise():
    """With the true sky as map the residual is the noise: the binned periodogram
    matches the PSD; the fit recovers NET, fknee and alpha; flagged (NaN)
    samples are excluded; DetectorData are pooled by name, or keyed by position."""
    from pysanepic import TabulatedNoise, estimate_noise
    from pysanepic.mapmaker import MapResult

    nside, fs, chunk, sigma = 16, 10.0, 4000, 50.0 * np.sqrt(10.0) * 1e-6
    rng = np.random.default_rng(14)
    sky = rng.normal(0, 1e-4, (3, 12 * nside**2))
    shape = lambda f: (f**1.5 + 0.1**1.5) / (f**1.5 + 1e-3**1.5)  # noqa: E731
    data = _noise_data(rng, nside, sky, fs, chunk, 40, shape, sigma)
    truth = MapResult(maps=sky, hit_map=None, nside=nside, coordinates="E", iterations=0, converged=True, residuals=None)
    flags = np.zeros(len(data[0].tod), bool)
    flags[1000:1500] = True
    data[0] = dataclasses.replace(data[0], tod=np.where(flags, np.nan, data[0].tod), flags=flags)
    est = estimate_noise(data, truth, chunk_s=chunk / fs, nbins=30)
    assert sorted(est) == ["det0", "det1", "det2"] and all(isinstance(e, TabulatedNoise) for e in est.values())
    for e in est.values():
        ratio = e.psd / (2 * sigma**2 / fs * shape(e.freqs_hz))
        assert np.median(np.abs(ratio - 1)) < 0.08 and abs(np.mean(ratio[5:]) - 1) < 0.05
    for model in ("toast", "keshner"):
        fit = estimate_noise(data, truth, chunk_s=chunk / fs, nbins=30, fit=model)["det1"]
        assert fit.model == model and abs(fit.net_ukrts / 50 - 1) < 0.03
        if model == "toast":
            assert abs(fit.fknee_hz / 0.1 - 1) < 0.25 and abs(fit.alpha / 1.5 - 1) < 0.15
    anonymous = estimate_noise(_noise_data(rng, nside, sky, fs, chunk, 4, shape, sigma, names=False), truth, chunk_s=chunk / fs)
    assert sorted(anonymous) == [0, 1, 2]


def test_estimate_noise_iteration():
    """make_maps -> estimate_noise -> make_maps runs, and the second maps are
    as good as those with the true noise model."""
    from pysanepic import estimate_noise

    nside, fs, chunk, sigma = 16, 10.0, 4000, 50.0 * np.sqrt(10.0) * 1e-6
    rng = np.random.default_rng(15)
    sky = hp.smoothing(rng.normal(0, 1e-4, (3, 12 * nside**2)), fwhm=np.radians(8))  # smooth: no sub-pixel signal
    shape = lambda f: (f**1.5 + 0.1**1.5) / (f**1.5 + 1e-3**1.5)  # noqa: E731
    true = OneOverFNoise(net_ukrts=50.0, fknee_hz=0.1, alpha=1.5, fmin_hz=1e-3)
    data = [dataclasses.replace(d, noise=true) for d in _noise_data(rng, nside, sky, fs, chunk, 20, shape, sigma)]
    params = GLSParameters(chunk_s=chunk / fs, tol=1e-10)
    ref = make_maps(data, nside, coordinates="E", params=params)
    white = make_maps([dataclasses.replace(d, noise=OneOverFNoise(net_ukrts=50.0)) for d in data], nside, coordinates="E", params=params)
    est = estimate_noise(data, white, chunk_s=chunk / fs)
    r = make_maps([dataclasses.replace(d, noise=est[d.name]) for d in data], nside, coordinates="E", params=params)
    seen = ref.maps[1] != hp.UNSEEN
    err = lambda m: np.std((m - sky)[:, seen])  # noqa: E731
    assert r.converged and err(r.maps) < 1.05 * err(ref.maps) < err(white.maps)


if __name__ == "__main__":
    test_noiseless_recovers_sky()
    test_padding_noiseless()
    test_pixel_input_and_generator()
    test_pointing_checks()
    test_contiguous_run_kernel()
    test_pixel_order()
    print("ok")


def test_solve_products():
    """solve from x0=None does no product for the initial residual and recomputes
    the true residual after every recompute_every-th step, not after the first."""
    from pysanepic import GLS, one_over_f_weights

    rng = np.random.default_rng(3)
    npix, ns, nchunk = 6, 128, 3
    pix = rng.integers(0, npix, (nchunk, ns))
    psi = rng.uniform(0, np.pi, (nchunk, ns))
    w = one_over_f_weights(ns, 10.0, 1.0, 0.5, 1.0, 0.05)
    g = GLS(pix, psi, np.tile(w, (nchunk, 1)), min_pol_rcond=0.0, angle_dtype=np.float64, preconditioner="jacobi")
    calls = []
    apply = g._apply
    g._apply = lambda *a, **k: calls.append(1) or apply(*a, **k)
    _, info = g.solve(rng.normal(size=(nchunk, ns)), tol=1e-30, maxiter=12, recompute_every=5)
    assert info["iterations"] == 12
    assert len(calls) == 1 + 12 + 2  # rhs, one per iteration, true residual after steps 5 and 10


def test_template_diag_dense():
    """The preconditioner of the template amplitudes is the exact diagonal of
    T^T N^-1 T, also for labels that are interleaved and not in time order."""
    from pysanepic import GLS, one_over_f_weights

    rng = np.random.default_rng(5)
    npix, ns, nchunk = 6, 128, 2
    pix = rng.integers(0, npix, (nchunk, ns))
    w = one_over_f_weights(ns, 10.0, 1.0, 0.5, 1.0, 0.05)
    g = GLS(pix, None, np.tile(w, (nchunk, 1)), template_dtype=np.float64)
    labels = [np.repeat([3, 0, 2, 0, 1], [20, 30, 25, 33, 20]), np.where(rng.random(ns) < 0.2, -1, rng.integers(0, 4, ns))]
    values = rng.normal(size=(nchunk, ns))
    g.set_templates([(k, values[k], labels[k]) for k in range(nchunk)])
    cinv = np.fft.irfft(w, n=ns)[(np.arange(ns)[:, None] - np.arange(ns)[None, :]) % ns]
    ref = [sum(t @ cinv @ t for t in (np.where(labels[k] == a, values[k], 0.0) for k in range(nchunk))) for a in range(4)]
    assert np.allclose(g._template_diag(g._tmpl, g.namp), ref, rtol=1e-12, atol=0)


def test_template_storage():
    """Templates with one amplitude per chunk given as one int (no labels array)
    and the full labels give the same solution; float32 storage (the default)
    agrees with float64 to its rounding; the caller's arrays are not changed,
    also where samples are flagged."""
    from pysanepic import GLS, one_over_f_weights

    rng = np.random.default_rng(31)
    npix, ns, nchunk = 8, 200, 3
    pix = rng.integers(0, npix, (nchunk, ns))
    pix[1, 50:60] = -1  # flagged
    psi = rng.uniform(0, np.pi, (nchunk, ns))
    w = np.tile(one_over_f_weights(ns, 10.0, 1.0, 0.5, 1.0, 0.05), (nchunk, 1))
    values = rng.normal(size=(nchunk, ns))
    kept = values.copy()
    tod = rng.normal(size=(nchunk, ns))
    full = [(k, values[k], np.full(ns, k % 2)) for k in range(nchunk)]
    one = [(k, values[k], k % 2) for k in range(nchunk)]
    res = {}
    for name, tmpl, dt in (("full", full, np.float64), ("one", one, np.float64), ("f32", one, np.float32)):
        g = GLS(pix, psi, w, min_pol_rcond=0.0, angle_dtype=np.float64, templates=tmpl, template_dtype=dt)
        assert g.namp == 2 and g._tmpl[0][0][0].dtype == dt
        x, info = g.solve(tod, tol=1e-26, maxiter=1000)
        assert info["converged"]
        res[name] = x, info["amplitudes"]
    assert np.array_equal(values, kept)
    assert np.isscalar(g._tmpl[0][0][1])
    for name, rtol in (("one", 1e-10), ("f32", 1e-5)):
        assert np.allclose(res[name][0], res["full"][0], rtol=rtol, atol=rtol * np.abs(res["full"][0]).max())
        assert np.allclose(res[name][1], res["full"][1], rtol=rtol)
    with pytest.raises(ValueError, match="template_dtype"):
        GLS(pix, psi, w, template_dtype=np.float16)


@pytest.mark.parametrize("private", [True, False])
def test_chunk_threads(monkeypatch, private):
    """Chunks in parallel in the PCG products (one per thread, P^T in private
    maps or by pixel ranges) give the serial solution: flagged samples, local
    and shared templates, float32 FFTs."""
    from pysanepic import gls
    from pysanepic import GLS, one_over_f_weights

    if not private:
        monkeypatch.setattr(gls, "_PRIVATE_MAPS_BYTES", 0)
    rng = np.random.default_rng(41)
    npix, ns, nchunk = 40, 300, 7
    pix = rng.integers(0, npix, (nchunk, ns))
    pix[2, 100:110] = -1 - np.arange(10)  # flagged
    psi = rng.uniform(0, np.pi, (nchunk, ns))
    w = np.tile(one_over_f_weights(ns, 10.0, 1.0, 0.5, 1.0, 0.05), (nchunk, 1))
    tod = rng.normal(size=(nchunk, ns))
    local = [(k, rng.normal(size=ns), np.arange(ns) // 100 + 3 * k) for k in range(nchunk)]
    shared = [(k, rng.normal(size=ns), k % 3) for k in range(nchunk)]
    res = []
    for nthreads in (1, 4):
        g = GLS(pix, psi, w, nthreads=nthreads, min_pol_rcond=0.0, fft_dtype=np.float32, templates=local,
                shared_templates=shared, template_dtype=np.float64)
        x, info = g.solve(tod, tol=1e-24, maxiter=500)
        assert info["converged"]
        res.append((x, info["amplitudes"], info["shared_amplitudes"], g._locprec))  # template preconditioner too
    for a, b in zip(*res):
        assert np.allclose(a, b, rtol=1e-8, atol=1e-8 * np.abs(b).max())


def test_setup_flagged_exact():
    """Setup with the unknowns of flagged samples in several chunks (one per
    sample and one per run): the preconditioner of the pixels and of the
    unknowns is the diagonal of P^T N^-1 P, computed here with dense
    matrices (N^-1 the circulant of each chunk, cut to the chunk)."""
    from pysanepic import GLS, one_over_f_weights
    from pysanepic.gls import fast_fft_length

    rng = np.random.default_rng(5)
    npix, ns, nchunk = 30, 200, 4
    m = fast_fft_length(ns + 37)
    pix = rng.integers(0, npix, (nchunk, ns))
    pix[0, 10:20] = -1 - np.arange(10)  # one unknown per sample
    pix[2, 50:80] = -1  # one per run
    pix[2, 120:125] = -2
    pix[3, 0:5] = -1 - np.arange(5)
    w = one_over_f_weights(m, 10.0, 1.0, 0.5, 1.0, 0.05)
    c = np.fft.irfft(w, n=m)  # first row of the circulant N^-1
    g = GLS(pix, None, [w] * nchunk, fft_lengths=m)
    blocks, unknowns = np.zeros(npix), []
    for k in range(nchunk):
        lag = np.subtract.outer(np.arange(ns), np.arange(ns)) % m
        C = c[lag]
        good = pix[k] >= 0
        P = np.zeros((ns, npix))
        P[np.flatnonzero(good), pix[k, good]] = 1.0
        blocks += np.diag(P.T @ C @ P)
        for lab in np.unique(pix[k, ~good]):  # the unknowns of a chunk in the order of np.unique
            u = (pix[k] == lab).astype(float)
            unknowns.append(u @ C @ u)
    assert g.nflag == len(unknowns) == 10 + 2 + 5
    assert np.allclose(1 / g.precond[0], blocks, rtol=1e-10)
    assert np.allclose(1 / g._flagprec, unknowns, rtol=1e-10)


def test_template_diag_exact():
    """The preconditioner of the template amplitudes is the diagonal of
    T^T N^-1 T, with N^-1 the circulant of each chunk cut to the chunk:
    short amplitudes (FFT of twice their span), long ones (one FFT of the
    chunk), one per chunk (an int label) and with samples left out (0)."""
    from pysanepic import GLS, one_over_f_weights
    from pysanepic.gls import fast_fft_length

    rng = np.random.default_rng(9)
    npix, ns, nchunk = 30, 400, 3
    m = fast_fft_length(ns + 21)
    pix = rng.integers(0, npix, (nchunk, ns))
    w = one_over_f_weights(m, 10.0, 1.0, 0.5, 1.0, 0.05)
    c = np.fft.irfft(w, n=m)
    C = c[np.subtract.outer(np.arange(ns), np.arange(ns)) % m]
    local = []
    for k in range(nchunk):
        v = rng.normal(size=ns)
        v[150:170] = 0.0  # left out
        local.append((k, v, 5 * k + np.minimum(np.arange(ns) // 40, 4)))  # 4 short amplitudes and a long one
    shared = [(k, rng.normal(size=ns), k % 2) for k in range(nchunk)]
    g = GLS(pix, None, [w] * nchunk, fft_lengths=m, templates=local, shared_templates=shared, template_dtype=np.float64)
    want = np.zeros(5 * nchunk)
    for k, v, lab in local:
        for a in np.unique(lab):
            u = np.where(lab == a, v, 0.0)
            want[a] += u @ C @ u
    want_sh = np.zeros(2)
    for k, v, a in shared:
        want_sh[a] += v @ C @ v
    assert np.allclose(1 / g._locprec[g.nflag : g.nflag + g.namp], want, rtol=1e-10)
    assert np.allclose(1 / g._locprec[g.nflag + g.namp :], want_sh, rtol=1e-10)


def test_project_maps():
    """P m of maps (the gain template and map sky models): the numba kernel
    against numpy, with UNSEEN pixels (taken as 0) and any angle."""
    from pysanepic.mapmaker import _project_maps

    rng = np.random.default_rng(3)
    maps = rng.normal(size=(3, 48))
    maps[1:, :5] = hp.UNSEEN
    maps[0, 7] = hp.UNSEEN
    pix = rng.integers(0, 48, 5000)
    angle = rng.uniform(-20, 20, 5000)
    m = np.where(maps == hp.UNSEEN, 0.0, maps)
    want = m[0, pix] + 0.9 * (m[1, pix] * np.cos(2 * angle) + m[2, pix] * np.sin(2 * angle))
    for nthreads in (1, 4):
        assert np.allclose(_project_maps(maps, pix, angle, 0.9, nthreads), want, rtol=0, atol=1e-13)  # 2a up to 40 rad
    assert np.allclose(_project_maps(maps[:1], pix, angle, 0.9), m[0, pix], rtol=0, atol=0)


@pytest.mark.parametrize("nthreads", [1, 4])
def test_shared_template_columns(nthreads):
    """Several shared templates of a chunk as the columns of one 2D array (one
    pass in the PCG products): the same solution, amplitudes and
    preconditioner as separate templates, with flagged samples."""
    from pysanepic import GLS, one_over_f_weights

    rng = np.random.default_rng(17)
    npix, ns, nchunk = 40, 300, 6
    pix = rng.integers(0, npix, (nchunk, ns))
    pix[1, 50:60] = -1 - np.arange(10)
    psi = rng.uniform(0, np.pi, (nchunk, ns))
    w = np.tile(one_over_f_weights(ns, 10.0, 1.0, 0.5, 1.0, 0.05), (nchunk, 1))
    tod = rng.normal(size=(nchunk, ns))
    cols = [rng.normal(size=(ns, 3)) for _ in range(nchunk)]
    labels = [np.array([k % 2, 2 + k % 2, 4]) for k in range(nchunk)]
    separate = [(k, cols[k][:, j], labels[k][j]) for k in range(nchunk) for j in range(3)]
    stacked = [(k, cols[k], labels[k]) for k in range(nchunk)]
    res = []
    for shared in (separate, stacked):
        g = GLS(pix, psi, w, nthreads=nthreads, min_pol_rcond=0.0, shared_templates=shared, template_dtype=np.float64,
                shared_groups=[np.array([0, 1])])
        x, info = g.solve(tod, tol=1e-24, maxiter=500)
        assert info["converged"]
        res.append((x, info["shared_amplitudes"], g._locprec))
    for a, b in zip(*res):
        assert np.allclose(a, b, rtol=1e-9, atol=1e-9 * np.abs(b).max())


def test_covariance_fixed_gains():
    """template_covariance with calibration: the gains marginalized (default)
    or fixed at their fitted values (covariance_fixed_gains): the same
    labels, errors a little smaller with the gains fixed, the same
    amplitudes."""
    from pysanepic import CalibrationParameters

    nside, n, fs, per = 8, 60_000, 10.0, 6000
    rng = np.random.default_rng(23)
    npix = 12 * nside**2
    sky = hp.smoothing(rng.normal(0, 1e-4, (3, npix)), fwhm=np.radians(15))
    fg = hp.smoothing(rng.normal(0, 3e-4, (3, npix)), fwhm=np.radians(15))
    noise = OneOverFNoise(net_ukrts=40.0, fknee_hz=0.05, alpha=1.0, fmin_hz=1e-4)
    t = np.arange(n)
    data = []
    for i, pol in enumerate(np.radians([0, 90, 45, 135])):
        theta, phi, psi = _scan(n, rng)
        pix, ang = hp.ang2pix(nside, theta, phi), psi + pol
        proj = lambda m: m[0, pix] + m[1, pix] * np.cos(2 * ang) + m[2, pix] * np.sin(2 * ang)  # noqa: E731
        vec = np.array(hp.ang2vec(theta, phi)).T
        dipole = 3.4e-3 * np.array([0.3, -0.5, 0.8]) @ vec + 2.7e-4 * np.sin(2 * np.pi * t / n + i)
        gain = 1 + 1e-3 * rng.normal(size=n // per)
        tod = gain[t // per] * (proj(sky) + (i - 1.5) * 0.02 * proj(fg) + dipole) + 1e-6 * rng.normal(size=n)
        data.append(DetectorData(tod=tod, pointing=Pointing(theta=theta, phi=phi, psi=psi, coordinates="E"),
                                 sampling_rate_hz=fs, noise=noise, pol_angle_rad=pol, name=f"d{i}", dipole=dipole))
    out = []
    for fixed in (False, True):
        r = make_maps(data, nside, coordinates="E", params=GLSParameters(chunk_s=1500.0, tol=1e-14, covariance_tol=1e-12),
                      templates={"fg": Template(sky=fg, sum_zero=True)}, template_covariance=True,
                      calibration=CalibrationParameters(gain_period_s=per / fs, n_iter=2, covariance_fixed_gains=fixed))
        out.append(r)
    (a, b) = (np.sqrt(np.diag(r.template_covariance["matrix"])) for r in out)
    assert out[0].template_covariance["labels"] == out[1].template_covariance["labels"]
    assert np.all(b < a) and np.all(b > 0.9 * a), b / a
    assert all(out[0].templates["fg"][k] == out[1].templates["fg"][k] for k in out[0].templates["fg"])


def test_sample_alm():
    """A sky model as alm sampled at the pointing: I + gamma (Q cos 2a + U sin 2a)
    with Q, U from ducc0, combined by the numba kernel, against numpy, for any
    angle and 1 or 4 threads."""
    import ducc0

    from pysanepic.mapmaker import _sample

    rng = np.random.default_rng(29)
    lmax, n = 32, 20_000
    nalm = (lmax + 1) * (lmax + 2) // 2
    alm = rng.normal(size=(3, nalm)) + 1j * rng.normal(size=(3, nalm))
    alm[:, : lmax + 1] = alm[:, : lmax + 1].real
    loc = np.stack([np.arccos(rng.uniform(-1, 1, n)), rng.uniform(0, 2 * np.pi, n)], axis=1)
    angle = rng.uniform(-20, 20, n)
    d = DetectorData(tod=np.zeros(n), pointing=Pointing(theta=loc[:, 0], phi=loc[:, 1], psi=np.zeros(n), coordinates="G"),
                     sampling_rate_hz=10.0, noise=OneOverFNoise(net_ukrts=40.0), pol_efficiency=0.9)
    kw = dict(lmax=lmax, loc=loc, epsilon=1e-10)
    i = ducc0.sht.synthesis_general(alm=alm[:1], spin=0, **kw)[0]
    q, u = ducc0.sht.synthesis_general(alm=alm[1:], spin=2, **kw)
    want = i + 0.9 * (q * np.cos(2 * angle) + u * np.sin(2 * angle))
    for nthreads in (1, 4):
        got = _sample(alm, d, np.zeros(n, int), angle, loc, nthreads)
        assert np.allclose(got, want, rtol=0, atol=1e-13 * np.abs(want).max())  # 2a up to 40 rad


@pytest.mark.parametrize("calibrate", [False, True])
def test_sample_batch(monkeypatch, calibrate):
    """GLSParameters.sample_batch: the alm sky model and sky template of
    several DetectorData interpolated in one synthesis_general call give the
    same maps and amplitudes as one call per DetectorData (with calibration
    and cache_model too, where the samples are kept)."""
    from pysanepic import CalibrationParameters, mapmaker

    calls = []
    synthesis = mapmaker._synthesis
    monkeypatch.setattr(mapmaker, "_synthesis", lambda *a: calls.append(len(a[1])) or synthesis(*a))
    nside, n, fs, per = 16, 60_000, 10.0, 6000
    if calibrate:
        data, _, _ = _calibration_data(nside, n, fs, per)
    else:
        data, _ = _subpixel_data(nside, n, fs)
    data = [dataclasses.replace(d, name=f"d{i}") for i, d in enumerate(data)]
    lmax = 3 * nside - 1
    rng = np.random.default_rng(31)
    sky = hp.map2alm(hp.smoothing(rng.normal(0, 1e-4, (3, 12 * nside**2)), fwhm=np.radians(5)), lmax=lmax)
    fg = 0.1 * hp.map2alm(hp.smoothing(rng.normal(0, 1e-4, (3, 12 * nside**2)), fwhm=np.radians(5)), lmax=lmax)
    cal = CalibrationParameters(gain_period_s=per / fs, n_iter=2, cache_model=True) if calibrate else None
    out = []
    ncalls = []
    for batch in (1, 3):  # 4 DetectorData: batches of 3 and 1
        calls.clear()
        params = GLSParameters(chunk_s=1200.0, tol=1e-20, angle_dtype=np.float64, sample_batch=batch)
        out.append(make_maps(data, nside, coordinates="E", params=params, sky_model=sky, calibration=cal,
                             templates={"fg": Template(sky=fg, sum_zero=True)}))
        ncalls.append(len(calls))
    assert ncalls == [8, 4], ncalls  # 2 fields x (4 DetectorData, or batches of 3 and 1), once with cache_model
    a, b = out
    seen = a.maps != hp.UNSEEN
    assert np.allclose(a.maps[seen], b.maps[seen], rtol=0, atol=1e-9 * np.abs(a.maps[seen]).max())
    assert all(abs(a.templates["fg"][k] - b.templates["fg"][k]) < 1e-8 for k in a.templates["fg"])
    if calibrate:
        assert max(np.abs(g - h).max() for g, h in zip(a.gains, b.gains)) < 1e-10
    with pytest.raises(ValueError, match="sample_batch"):
        make_maps(data, nside, coordinates="E", params=GLSParameters(sample_batch=0), sky_model=sky)


def test_rotation_matrix_astropy():
    """The astropy matrices (as litebird_sim) agree with healpy's: E <-> G to
    rounding, C (ICRS) to ~0.03 arcsec."""
    from pysanepic.mapmaker import _rotation_matrix

    for a, b, tol in [("E", "G", 1e-12), ("G", "E", 1e-12), ("E", "C", 1e-6), ("G", "C", 1e-6), ("C", "G", 1e-6)]:
        R = _rotation_matrix(a, b)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.abs(R - hp.Rotator(coord=[a, b]).mat).max() < tol


def test_coarse_null_monopole():
    """fmin_hz = 0 without padding: the DC weight is 0 and the I monopole has
    no weight; the coarse level on it is left out (no singular E) and the
    solve still satisfies A x = b, also in make_maps."""
    from pysanepic import GLS

    rng = np.random.default_rng(42)
    for ns in (64, 1000):
        pix = rng.integers(0, 5, (2, ns))
        w = np.tile(one_over_f_weights(ns, 10.0, 1.0, 0.5, 1.0), (2, 1))
        tod = rng.normal(size=(2, ns))
        g = GLS(pix, None, w, coarse=lambda labels: np.zeros(len(labels), int))
        m, info = g.solve(tod, tol=1e-24)
        assert info["converged"] and np.isfinite(m).all()
        assert np.allclose(g.apply_A(m), g.rhs(tod), rtol=0, atol=1e-10 * np.abs(g.rhs(tod)).max())
    nside, n, fs = 8, 24_000, 10.0
    sky = rng.normal(0, 1e-4, (3, 12 * nside**2))
    data = [dataclasses.replace(d, noise=OneOverFNoise(net_ukrts=40.0, fknee_hz=0.05, alpha=1.0))
            for d in _four_detectors(n, fs, rng, sky, nside)]
    r = make_maps(data, nside, "E", GLSParameters(chunk_s=1200.0, tol=1e-20))
    assert r.converged


def test_estimate_noise_contiguous_flags():
    """White noise with the central quarter of every chunk flagged: the PSD is
    normalized by the energy of the window on the samples left (the fraction
    of samples would give ~53% of it)."""
    from pysanepic import estimate_noise
    from pysanepic.mapmaker import MapResult

    nside, fs, chunk, sigma = 16, 10.0, 4000, 50.0 * np.sqrt(10.0) * 1e-6
    rng = np.random.default_rng(43)
    sky = rng.normal(0, 1e-4, (3, 12 * nside**2))
    data = _noise_data(rng, nside, sky, fs, chunk, 40, np.ones_like, sigma)
    truth = MapResult(maps=sky, hit_map=None, nside=nside, coordinates="E", iterations=0, converged=True, residuals=None)
    flags = (np.arange(len(data[0].tod)) % chunk >= 3 * chunk // 8) & (np.arange(len(data[0].tod)) % chunk < 5 * chunk // 8)
    data = [dataclasses.replace(d, tod=np.where(flags, np.nan, d.tod), flags=flags) for d in data]
    for e in estimate_noise(data, truth, chunk_s=chunk / fs, nbins=30).values():
        assert abs(np.mean(e.psd) / (2 * sigma**2 / fs) - 1) < 0.05


def test_pcg_zero_rhs_and_exact_start():
    """A zero TOD gives the zero map in 0 iterations; starting from an exact
    solution (r = b - A x = 0) stops before the first iteration (no 0/0)."""
    from pysanepic import GLS

    rng = np.random.default_rng(44)
    pix = rng.integers(0, 5, (2, 64))
    w = np.tile(one_over_f_weights(64, 10.0, 1.0, 0.5, 1.0, 0.05), (2, 1))
    for x0 in (None, rng.normal(size=(1, 5))):
        m, info = GLS(pix, None, w).solve(np.zeros((2, 64)), x0=x0)
        assert info["converged"] and info["iterations"] == 0 and not m.any()
    g = GLS(pix, None, w)
    x = rng.normal(size=g._nsky)
    b = g._apply(x.copy(), exact=True)
    y, it, history = g._pcg(b, x.copy(), True, 1e-20, 100, 10, False)
    assert it == 0 and history == [0.0] and np.array_equal(y, x)


def test_shared_covariance_not_converged():
    """shared_covariance warns when its solves stop at maxiter."""
    from pysanepic import GLS

    rng = np.random.default_rng(45)
    npix, ns, nchunk = 6, 96, 4
    pix = rng.integers(0, npix, (nchunk, ns))
    psi = rng.uniform(0, np.pi, (nchunk, ns))
    w = np.tile(one_over_f_weights(ns, 10.0, 1.0, 0.5, 1.0, 0.05), (nchunk, 1))
    shared = [(k, rng.normal(size=ns), np.full(ns, k % 2)) for k in range(nchunk)]
    g = GLS(pix, psi, w, min_pol_rcond=0.0, angle_dtype=np.float64, shared_templates=shared, template_dtype=np.float64)
    with pytest.warns(UserWarning, match="did not converge"):
        g.shared_covariance(tol=1e-26, maxiter=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g.shared_covariance(tol=1e-20, maxiter=1000)
