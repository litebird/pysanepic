"""make_maps on synthetic DetectorData (no litebird_sim needed).

Noiseless TOD of a random Galactic sky, observed with Ecliptic pointings, an
HWP, polarization efficiency 0.9 on one detector and chunks that do not divide
the data. The TOD is built with healpy's rotation, so pysanepic's numba
rotation kernel is checked against an independent implementation.
Run with: pytest tests/  (or: python tests/test_mapmaker.py)
"""

import dataclasses

import healpy as hp
import numpy as np
import pytest

from pysanepic import DetectorData, GLSParameters, OneOverFNoise, Pointing, TabulatedNoise, make_maps, one_over_f_weights


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
    params = GLSParameters(chunk_s=chunk / fs, tol=1e-14, pad_s=60.0, split_gap_s=np.inf)
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
