"""Conditional Gaussian noise in gaps, using Fourier precision operators."""

import ducc0
import numpy as np

from .gls import _axpy, _xpby, numba_threads, resolve_nthreads


def _validate_gap_solve(tol, maxiter):
    if not np.isscalar(tol) or not np.isfinite(tol) or not 0 < tol < 1:
        raise ValueError("gap fill tol must be finite and between 0 and 1")
    if isinstance(maxiter, (bool, np.bool_)) or not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError("gap fill maxiter must be a positive integer")


def _validate_gap_preconditioner(preconditioner):
    if preconditioner not in (None, "sine"):
        raise ValueError("gap fill preconditioner must be 'sine' or None")


def _gap_preconditioner(missing, weights, m, nthreads):
    """SPD own-noise approximation, with independent blocks for long runs.

    Short runs use the own-noise diagonal. Long runs invert S diag(w(f_j)) S,
    f_j=j/(2*(length+1)), where S is the orthonormal DST-I. This captures
    long-gap low frequencies without dense matrices or a DC inverse. Blocks
    omit correlations between gaps and detectors; the PCG operator retains
    all of them. A long, fully missing FFT row uses its exact own-noise inverse.
    The sine-symbol construction follows the tau-preconditioning approach,
    https://arxiv.org/abs/1404.5332 (no spectral-equivalence claim here for
    arbitrary supplied spectra or detector common modes).
    """
    diagonal = (2 * weights.sum(axis=1) - weights[:, 0]
                - (weights[:, -1] if m % 2 == 0 else 0.0)) / m
    inverse = np.repeat(1.0 / diagonal, missing.sum(axis=1))
    blocks, cache = [], {}
    first = 0
    for row, mask in enumerate(missing):
        edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
        lengths = edges[1::2] - edges[::2]
        positions = first + np.cumsum(lengths) - lengths
        long = lengths > 32
        for length, position in zip(lengths[long], positions[long]):
            length, position = int(length), int(position)
            full = length == m
            key = row, length
            if key not in cache:
                spectrum = (weights[row] if full else np.interp(
                    np.arange(1, length + 1) * (m / (2.0 * (length + 1))),
                    np.arange(weights.shape[1]), weights[row]))
                cache[key] = 1.0 / spectrum
            blocks.append((slice(position, position + length), cache[key], full))
        first += int(lengths.sum())

    def apply(r, z):
        np.multiply(r, inverse, out=z)
        for sl, spectrum, full in blocks:
            if full:
                f = ducc0.fft.r2c(r[sl], nthreads=nthreads)
                f *= spectrum
                ducc0.fft.c2r(f, lastsize=m, forward=False, inorm=2, out=z[sl], nthreads=nthreads)
            else:
                ducc0.fft.dst(r[sl], type=1, inorm=1, out=z[sl], nthreads=nthreads)
                z[sl] *= spectrum
                ducc0.fft.dst(z[sl], type=1, inorm=1, out=z[sl], nthreads=nthreads)

    return apply


def conditional_gap_fill(residual, flags, weights, *, draw=True, rng=None, tol=1e-12,
                         maxiter=1000, fft_length=None, coupling=None, mode_psd=None, nthreads=None,
                         preconditioner="sine"):
    """Fill missing noise conditional on valid residuals and a fixed noise model.

    Parameters
    ----------
    residual : (ns,) or (ndet, ns) array
        Noise residual after subtracting the sky and other known signals.
        Valid samples must be finite; flagged values are never used.
    flags : boolean array like residual
        True for missing samples, including any unobserved padding.
    weights : (ns//2+1,) or (ndet, ns//2+1) array
        Inverse noise eigenvalues, as in GLS (not one-sided PSD units).
        With fft_length=m, on its rfft grid: Q is the ns x ns principal
        block of the m x m circulant precision. Positive except possibly DC.
    draw : bool
        True: constrained realization; False: conditional mean, without RNG use.
    rng : numpy Generator, seed, or None
        Source of random numbers; a Generator is advanced only with draw=True.
    tol, maxiter : float, int
        CG requires squared relative residual <= tol; nonconvergence raises
        RuntimeError, never returning a silently inaccurate realization.
    fft_length : int or None
        FFT length >= ns; None: ns, including odd lengths.
    coupling, mode_psd : arrays or None
        Optional common-mode noise of the simultaneous rows: N = diag(1/w)
        + U diag(lambda) U^T per frequency. U: (ndet, r), lambda: (r, m//2+1).
        All rows and their flags are conditioned jointly.
    nthreads : int or None
        FFT and vector-operation threads, resolved as for GLS.
    preconditioner : "sine" or None
        Default: independent own-noise DST-I blocks for gaps longer than 32
        samples, diagonal scaling for shorter gaps. None: unpreconditioned CG.
        No change to the conditional distribution or convergence threshold.

    Returns
    -------
    filled : numpy.ndarray
        Filled residual, preserving input shape and valid values. Inputs are
        never mutated.
    info : dict
        iterations, converged and residual (squared relative).

    Notes
    -----
    The conditional precision Q_ff must be positive definite. Zero DC is
    allowed when every detector with that null mode has an anchored sample
    (an FFT extension also anchors it in the principal-block model).

    Solves Q_ff x = -Q_fu residual_u + g_f, with Cov(g)=Q for draw=True.
    No dense time-domain covariance is formed. For common modes, put
    B=U sqrt(lambda), K=(I+B^T W B)^-1. For independent white streams z,y,
    g=W^1/2 z - W B K (B^T W^1/2 z + y), whose covariance is
    W-W B K B^T W. Drawing on the *full FFT grid*, then restricting to ns,
    is essential when fft_length > ns.
    """
    _validate_gap_solve(tol, maxiter)
    _validate_gap_preconditioner(preconditioner)
    residual = np.asarray(residual, dtype=float)
    flags = np.asarray(flags)
    if residual.ndim not in (1, 2) or residual.shape[-1] < 1:
        raise ValueError("residual must be a nonempty 1D or 2D timeline")
    if flags.dtype != bool or flags.shape != residual.shape:
        raise ValueError("flags must be boolean with the same shape as residual")
    if not np.all(np.isfinite(residual[~flags])):
        raise ValueError("valid residual samples must be finite")
    one = residual.ndim == 1
    data, missing = np.atleast_2d(residual), np.atleast_2d(flags)
    ndet, ns = data.shape
    m = ns if fft_length is None else fft_length
    if isinstance(m, (bool, np.bool_)) or not isinstance(m, (int, np.integer)) or m < ns:
        raise ValueError("fft_length must be an integer >= the timeline length")
    nthreads = resolve_nthreads(nthreads)
    if nthreads < 1:
        raise ValueError("nthreads must be positive")
    w = np.asarray(weights, dtype=float)
    if w.shape not in ((m // 2 + 1,), (ndet, m // 2 + 1)):
        raise ValueError("weights must have one rfft row per detector, or one shared row")
    w = np.broadcast_to(w, (ndet, m // 2 + 1))
    if not np.isfinite(w).all() or np.any(w[:, 0] < 0) or np.any(w[:, 1:] <= 0) or not np.any(w > 0):
        raise ValueError("weights must be finite and positive, except for an optional zero DC weight")
    if m == ns and np.any((w[:, 0] == 0) & missing.all(axis=1)):
        raise ValueError("conditional precision is singular: fully flagged detector with zero DC weight")
    U = root_lam = K = None
    if coupling is not None or mode_psd is not None:
        U, lam = np.asarray(coupling, dtype=float), np.asarray(mode_psd, dtype=float)
        if U.ndim != 2 or U.shape[0] != ndet or U.shape[1] < 1 or not np.isfinite(U).all():
            raise ValueError("coupling must be finite with shape (ndet, n_modes)")
        if lam.shape != (U.shape[1], m // 2 + 1) or not np.isfinite(lam).all() or np.any(lam < 0):
            raise ValueError("mode_psd must be finite and non-negative with shape (n_modes, nfreq)")
        root_lam = np.sqrt(lam)
        S = np.einsum("di,df,dj->fij", U, w, U)
        matrix = np.eye(U.shape[1]) + S * root_lam.T[:, :, None] * root_lam.T[:, None, :]
        K = np.linalg.inv(matrix)
    out = data.copy()
    idx = np.flatnonzero(missing)
    if idx.size == 0:
        return (out[0] if one else out), dict(iterations=0, converged=True, residual=0.0)
    # Reused work arrays. Scatter/gather indices retain detector-major order,
    # including when the FFT grid is longer than the observed timeline.
    rows, times = np.unravel_index(idx, missing.shape)
    fft_idx = rows * m + times
    timeline = np.zeros((ndet, m))
    spectrum = np.empty((ndet, m // 2 + 1), complex)
    filtered = np.empty_like(timeline)

    def filter_timeline(random=False):
        ducc0.fft.r2c(timeline, axes=(-1,), out=spectrum, nthreads=nthreads)
        spectrum[:] *= np.sqrt(w) if random else w
        if U is not None:
            modes = root_lam * (U.T @ spectrum)
            if random:
                white_modes = generator.standard_normal((U.shape[1], m))
                modes += ducc0.fft.r2c(white_modes, axes=(-1,), nthreads=nthreads)
            modes = np.einsum("fij,jf->if", K, modes)
            spectrum[:] -= w * (U @ (root_lam * modes))
        ducc0.fft.c2r(spectrum, axes=(-1,), lastsize=m, forward=False, inorm=2, out=filtered, nthreads=nthreads)

    np.copyto(timeline[:, :ns], data, where=~missing)
    filter_timeline()
    b = -filtered.ravel()[fft_idx]
    if draw:
        generator = np.random.default_rng(rng)
        timeline[:] = generator.standard_normal(timeline.shape)
        filter_timeline(random=True)
        b += filtered.ravel()[fft_idx]

    def apply(v, result):
        timeline[:] = 0.0
        timeline.ravel()[fft_idx] = v
        filter_timeline()
        np.take(filtered.ravel(), fft_idx, out=result)

    precondition = None if preconditioner is None else _gap_preconditioner(missing, w, m, nthreads)
    x, r, z, q = np.zeros_like(b), b.copy(), np.empty_like(b), np.empty_like(b)
    if precondition is None:
        z[:] = r
    else:
        precondition(r, z)
    p = z.copy()
    rz = float(np.dot(r, z))
    bb = float(np.dot(b, b))
    error, iteration = 0.0 if bb == 0 else 1.0, 0
    with numba_threads(nthreads):
        while error > tol and iteration < maxiter:
            apply(p, q)
            curvature = float(np.dot(p, q))
            if not np.isfinite(curvature) or curvature <= 0:
                raise RuntimeError("conditional gap CG encountered non-positive curvature")
            if not np.isfinite(rz) or rz <= 0:
                raise RuntimeError("conditional gap PCG encountered non-positive preconditioned residual")
            alpha = rz / curvature
            _axpy(x, alpha, p)
            _axpy(r, -alpha, q)
            iteration += 1
            new_rr = float(np.dot(r, r))
            # Recheck the true residual periodically and before accepting a draw.
            # Replace the recursive residual but keep the search direction:
            # restarting would discard the Krylov space (5-10x more iterations
            # with steep 1/f noise and long gaps).
            if iteration % 10 == 0 or new_rr / bb <= tol:
                apply(x, q)
                r[:] = b - q
                new_rr = float(np.dot(r, r))
                if new_rr / bb <= tol:
                    error = new_rr / bb
                    break
            if precondition is None:
                z[:] = r
            else:
                precondition(r, z)
            new_rz = float(np.dot(r, z))
            _xpby(p, z, new_rz / rz)
            rz = new_rz
            error = new_rr / bb
    if not np.isfinite(error) or error > tol:
        raise RuntimeError(f"conditional gap CG did not converge in {maxiter} iterations (squared relative residual {error:.3g})")
    out.ravel()[idx] = x
    return (out[0] if one else out), dict(iterations=iteration, converged=True, residual=error)
