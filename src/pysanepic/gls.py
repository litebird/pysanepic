"""GLS map-making with correlated noise (the SANEPIC algorithm).

Solves (P^T N^-1 P) m = P^T N^-1 d with preconditioned conjugate gradients.
N^-1 is applied in Fourier space, independently on each data chunk
(one detector x one stretch of time), as a circulant filter.

Data layout: data come in *blocks*; a block is a 2D array (nchunk, ns) of
chunks with the same length ns. Pass one array for a single block or a list
of arrays for chunks of different lengths. Noise weights of a block have shape
(nchunk, ns//2+1) and are defined so that the filter is irfft(rfft(t) * w).
"""

import os
from contextlib import contextmanager

import numba
import numpy as np
import scipy.fft


def resolve_nthreads(nthreads=None):
    """Number of threads per process: `nthreads` if given, else OMP_NUM_THREADS,
    else 1 (as in litebird_sim, to avoid oversubscription with several MPI
    processes per node)."""
    if nthreads is not None:
        return int(nthreads)
    return int(os.environ.get("OMP_NUM_THREADS", 1))


@contextmanager
def numba_threads(nthreads):
    """Run numba parallel kernels with `nthreads` threads (capped by numba's pool)."""
    old = numba.get_num_threads()
    numba.set_num_threads(max(1, min(nthreads, numba.config.NUMBA_NUM_THREADS)))
    try:
        yield
    finally:
        numba.set_num_threads(old)


def one_over_f_weights(ns, fsamp_hz, sigma, fknee_hz, alpha, fmin_hz=0.0):
    """Inverse noise PSD on the rfft grid of a chunk of `ns` samples.

    Model: P(f) = sigma^2 (f^alpha + fknee^alpha) / (f^alpha + fmin^alpha), the
    litebird_sim "toast" model; sigma is the white-noise rms per sample, so that
    for white noise the filter is exactly 1/sigma^2. With fmin = 0 the DC weight
    is 0 and the mean of each chunk is not used.
    """
    fa = np.fft.rfftfreq(ns, 1.0 / fsamp_hz) ** alpha
    num, den = fa + fmin_hz**alpha, fa + fknee_hz**alpha
    ratio = np.divide(num, den, out=np.ones_like(fa), where=den > 0)  # fknee = 0: white
    return ratio / sigma**2


def _blocks(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _filter(t, w, workers):
    ns = t.shape[-1]
    return scipy.fft.irfft(scipy.fft.rfft(t, axis=-1, workers=workers) * w, n=ns, axis=-1, workers=workers)


@numba.njit(parallel=True, cache=True)
def _project(ip, m, c2, s2, pol, out):
    """out[i] = I[p] + c2[i] Q[p] + s2[i] U[p], p = ip[i] (flattened samples)."""
    for i in numba.prange(ip.size):
        p = ip[i]
        v = m[0, p]
        if pol:
            v += m[1, p] * c2[i] + m[2, p] * s2[i]
        out[i] = v


@numba.njit(parallel=True, cache=True)
def _backproject(ip, t, c2, s2, pol, out, nthreads):
    """out += P^T t (flattened samples). Each thread accumulates into a private
    copy of the map, then the copies are summed pixel by pixel."""
    # ponytail: nthreads private maps (~75 MB each at nside 512, I/Q/U); switch to
    # pixel-sorted samples if memory per process becomes the limit
    ncomp, nobs = out.shape
    buf = np.zeros((nthreads, ncomp, nobs))
    step = (ip.size + nthreads - 1) // nthreads
    for th in numba.prange(nthreads):
        for i in range(th * step, min(ip.size, (th + 1) * step)):
            p = ip[i]
            buf[th, 0, p] += t[i]
            if pol:
                buf[th, 1, p] += t[i] * c2[i]
                buf[th, 2, p] += t[i] * s2[i]
    for j in numba.prange(nobs):
        for th in range(nthreads):
            for c in range(ncomp):
                out[c, j] += buf[th, c, j]


@numba.njit(cache=True)
def _block_circulant(ip, c, c2, s2, nobs, pol):
    """Exact per-pixel blocks of P^T C P, C circulant per chunk.

    For each pixel, sums c[(t - t') mod ns] w_t w_t'^T over all sample pairs
    that fall in it, with w = (1, c2, s2). Returns (nobs, 6) with the entries
    II, IQ, IU, QQ, QU, UU (only II if pol is False).
    Cost is sum over pixels of hits^2 per chunk.
    """
    # ponytail: O(hits^2) per pixel per chunk; switch to a lag cutoff or the
    # SANEPIC N_[0] shortcut if deep pixels with long chunks become a bottleneck
    out = np.zeros((nobs, 6))
    nchunk, ns = ip.shape
    for k in range(nchunk):
        order = np.argsort(ip[k], kind="mergesort")
        start = 0
        while start < ns:
            p = ip[k, order[start]]
            stop = start
            while stop < ns and ip[k, order[stop]] == p:
                stop += 1
            n = stop - start
            # Contiguous run of samples with no polarization sensitivity (e.g. the
            # padding of a chunk): sum over lags in O(n) instead of O(n^2)
            unpol = True
            if pol:
                for a in range(start, stop):
                    if c2[k, order[a]] != 0.0 or s2[k, order[a]] != 0.0:
                        unpol = False
                        break
            if unpol and order[stop - 1] - order[start] == n - 1:
                acc = 0.0
                for lag in range(-(n - 1), n):
                    acc += (n - abs(lag)) * c[k, lag % ns]
                out[p, 0] += acc
                start = stop
                continue
            for a in range(start, stop):
                ta = order[a]
                for b in range(start, stop):
                    tb = order[b]
                    cab = c[k, (ta - tb) % ns]
                    out[p, 0] += cab
                    if pol:
                        out[p, 1] += cab * c2[k, tb]
                        out[p, 2] += cab * s2[k, tb]
                        out[p, 3] += cab * c2[k, ta] * c2[k, tb]
                        out[p, 4] += cab * c2[k, ta] * s2[k, tb]
                        out[p, 5] += cab * s2[k, ta] * s2[k, tb]
            start = stop
    return out


class GLS:
    """Map-maker for a fixed set of pointings and noise weights.

    Parameters
    ----------
    pix : int array (nchunk, ns), or list of such blocks
        HEALPix pixel index of each sample (any scheme; only used as labels).
    psi : float array(s) like `pix`, or None
        Polarization angle of each sample. None for intensity only.
    weights : float array (nchunk, ns//2+1), or list (one per block)
        Inverse noise power spectrum of each chunk on the rfft grid.
    pol_efficiency : float array (nchunk,) or (nchunk, ns), or list (one per block), or None
        Polarization efficiency of each chunk or sample; None means 1. Samples with
        efficiency < 0.5 do not count as polarized hits (as in SANEPIC).
    min_pol_hits : int
        Q/U are not solved in pixels with fewer polarized hits (degenerate).
    min_pol_rcond : float
        Q/U are not solved in pixels whose angle-coverage matrix
        sum_t w_t w_t^T, w = (1, gamma cos 2a, gamma sin 2a), has a reciprocal
        condition number below this value (Q and U cannot be separated there).
        0 disables the check, as in the original SANEPIC.
    preconditioner : "block" or "jacobi"
        "block" (default): inverse of the exact 3x3 I/Q/U block of P^T N^-1 P in
        each pixel. "jacobi": the original SANEPIC choice, exact I diagonal and
        Q/U = 2 / diag_I (assumes uniform angle coverage).
    comm : mpi4py communicator or None
        With MPI, each rank passes only its own chunks (possibly none); maps
        are replicated on all ranks and P^T is summed with Allreduce.
    nthreads : int or None
        Threads per process for the FFTs; None: OMP_NUM_THREADS, else 1.
    """

    def __init__(self, pix, psi, weights, pol_efficiency=None, min_pol_hits=4, min_pol_rcond=1e-2, preconditioner="block", comm=None, nthreads=None):
        pix, self.weights = _blocks(pix), _blocks(weights)
        self.comm = comm
        self.nthreads = resolve_nthreads(nthreads)
        local = np.unique(np.concatenate([p.ravel() for p in pix] or [np.empty(0, np.int64)]))
        self.pixels = local if comm is None else np.unique(np.concatenate(comm.allgather(local)))
        self.ip = [np.searchsorted(self.pixels, p) for p in pix]
        self._dummy = np.zeros(1)
        self.nobs = self.pixels.size
        self.pol = psi is not None
        self.ncomp = 3 if self.pol else 1
        if self.pol:
            psi = _blocks(psi)
            eff = [1.0] * len(psi) if pol_efficiency is None else _blocks(pol_efficiency)
            g = [np.asarray(e, dtype=float) for e in eff]
            g = [e.reshape(-1, 1) if e.ndim <= 1 else e for e in g]
            self.cos2 = [gb * np.cos(2 * s) for gb, s in zip(g, psi)]
            self.sin2 = [gb * np.sin(2 * s) for gb, s in zip(g, psi)]

        self.hits = self._sum(sum((np.bincount(ip.ravel(), minlength=self.nobs) for ip in self.ip), np.zeros(self.nobs, np.int64)))
        self.mask = np.ones((self.ncomp, self.nobs), dtype=bool)
        if self.pol:
            polarized = [(c2 * c2 + s2 * s2) > 0.25 for c2, s2 in zip(self.cos2, self.sin2)]
            pol_hits = self._sum(
                sum((np.bincount(ip.ravel(), w.ravel(), minlength=self.nobs) for ip, w in zip(self.ip, polarized)), np.zeros(self.nobs))
            )
            self.mask[1:] = pol_hits >= min_pol_hits
            if min_pol_rcond > 0:
                self.mask[1:] &= self._pol_rcond() >= min_pol_rcond

        dummy = np.zeros((1, 1))
        blocks = np.zeros((self.nobs, 6))
        for k, (ip, w) in enumerate(zip(self.ip, self.weights)):
            c = scipy.fft.irfft(w, n=ip.shape[-1], axis=-1)
            c2, s2 = (self.cos2[k], self.sin2[k]) if self.pol else (dummy, dummy)
            blocks += _block_circulant(ip, c, c2, s2, self.nobs, self.pol)
        blocks = self._sum(blocks)
        if np.any(blocks[:, 0] <= 0):
            raise ValueError("Preconditioner has non-positive elements: check noise weights")
        self._set_preconditioner(blocks, preconditioner)

    def _pol_rcond(self):
        """Reciprocal condition number of the angle-coverage matrix of each pixel."""
        cov = np.zeros((6, self.nobs))
        for k, ip in enumerate(self.ip):
            ip, c, s = ip.ravel(), self.cos2[k].ravel(), self.sin2[k].ravel()
            for j, w in enumerate((c, s, c * c, c * s, s * s)):
                cov[j + 1] += np.bincount(ip, w, minlength=self.nobs)
        cov = self._sum(cov)
        n, c, s, cc, cs, ss = self.hits.astype(float), *cov[1:]
        M = np.stack([n, c, s, c, cc, cs, s, cs, ss], axis=1).reshape(-1, 3, 3)
        rcond = np.zeros(self.nobs)
        seen = self.mask[1]  # pixels with enough polarized hits
        with np.errstate(divide="ignore"):
            rcond[seen] = 1.0 / np.linalg.cond(M[seen])
        return rcond

    def _set_preconditioner(self, blocks, kind):
        diag_i = blocks[:, 0]
        if kind == "jacobi" or not self.pol:
            self.precond = np.empty((self.ncomp, self.nobs))
            self.precond[0] = 1.0 / diag_i
            if self.pol:
                self.precond[1:] = 2.0 / diag_i
            self.precond *= self.mask
            return
        if kind != "block":
            raise ValueError(f"Unknown preconditioner {kind!r}: use 'block' or 'jacobi'")
        ii, iq, iu, qq, qu, uu = blocks.T
        M = np.stack([ii, iq, iu, iq, qq, qu, iu, qu, uu], axis=1).reshape(-1, 3, 3)
        # Invert well-conditioned blocks; elsewhere fall back to the diagonal
        good = self.mask[1] & (np.linalg.det(M) > 1e-12 * ii * qq * uu)
        self.precond = np.zeros((self.nobs, 3, 3))
        self.precond[good] = np.linalg.inv(M[good])
        bad = ~good
        self.precond[bad, 0, 0] = 1.0 / ii[bad]
        pol_bad = bad & self.mask[1]
        self.precond[pol_bad, 1, 1] = 1.0 / qq[pol_bad]
        self.precond[pol_bad, 2, 2] = 1.0 / uu[pol_bad]

    def _apply_precond(self, r):
        if self.precond.ndim == 2:
            return self.precond * r
        return np.einsum("pij,jp->ip", self.precond, r)

    def _sum(self, x):
        """Sum a map-sized array over ranks (identity without MPI)."""
        if self.comm is None:
            return x
        out = np.empty_like(x)
        self.comm.Allreduce(np.ascontiguousarray(x), out)
        return out

    def _pol_arrays(self, k):
        if self.pol:
            return True, self.cos2[k].ravel(), self.sin2[k].ravel()
        return False, self._dummy, self._dummy

    def project(self, m):
        """P m: map (ncomp, nobs) -> list of timeline blocks."""
        out = []
        with numba_threads(self.nthreads):
            for k, ip in enumerate(self.ip):
                t = np.empty(ip.shape)
                pol, c2, s2 = self._pol_arrays(k)
                _project(ip.ravel(), m, c2, s2, pol, t.ravel())
                out.append(t)
        return out

    def backproject(self, tod):
        """P^T t: timeline block(s) -> map (ncomp, nobs)."""
        out = np.zeros((self.ncomp, self.nobs))
        with numba_threads(self.nthreads):
            for k, (ip, t) in enumerate(zip(self.ip, _blocks(tod))):
                pol, c2, s2 = self._pol_arrays(k)
                _backproject(ip.ravel(), np.ascontiguousarray(t).ravel(), c2, s2, pol, out, numba.get_num_threads())
        return self._sum(out) * self.mask

    def _filter_all(self, tod):
        return [_filter(t, w, self.nthreads) for t, w in zip(_blocks(tod), self.weights)]

    def apply_A(self, m):
        return self.backproject(self._filter_all(self.project(m)))

    def rhs(self, tod):
        return self.backproject(self._filter_all(tod))

    def solve(self, tod, x0=None, tol=1e-15, maxiter=2000, recompute_every=10, verbose=False):
        """Run PCG on `tod` (same block layout as `pix`).

        Stops when |r|^2 / |b|^2 < tol (same criterion as SANEPIC).
        Returns (map (ncomp, nobs), info dict).
        """
        b = self.rhs(tod)
        bb = np.vdot(b, b)
        x = np.zeros_like(b) if x0 is None else x0 * self.mask
        r = b - self.apply_A(x)
        z = self._apply_precond(r)
        d = z.copy()
        delta = np.vdot(r, z)
        history = [np.vdot(r, r) / bb]

        it = 0
        while it < maxiter and (it == 0 or history[-1] > tol):
            q = self.apply_A(d)
            alpha = delta / np.vdot(d, q)
            x += alpha * d
            if it % recompute_every == 0:
                r = b - self.apply_A(x)  # true residual, avoids drift
            else:
                r -= alpha * q
            z = self._apply_precond(r)
            delta_new = np.vdot(r, z)
            d = z + (delta_new / delta) * d
            delta = delta_new
            history.append(np.vdot(r, r) / bb)
            it += 1
            if verbose and (self.comm is None or self.comm.rank == 0):
                print(f"iter {it:4d}  |r|^2/|b|^2 = {history[-1]:.3e}")

        return x, {"iterations": it, "converged": history[-1] <= tol, "residuals": np.array(history)}

    def to_healpix(self, m, nside, fill=np.nan):
        """Expand (ncomp, nobs) to full RING maps (ncomp, 12 nside^2)."""
        full = np.full((m.shape[0], 12 * nside * nside), fill)
        sky = self.pixels < full.shape[1]  # labels beyond the map are virtual pixels
        full[:, self.pixels[sky]] = np.where(self.mask, m, fill)[:, sky]
        return full
