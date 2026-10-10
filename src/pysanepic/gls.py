"""GLS map-making with correlated noise (the SANEPIC algorithm).

Solves (P^T N^-1 P) m = P^T N^-1 d with preconditioned conjugate gradients.
N^-1 is applied in Fourier space, independently on each data chunk
(one detector x one stretch of time), as a circulant filter.

Data layout: data come as *chunks*. Pass a 2D array (nchunk, ns) for chunks of
the same length, or a list of 1D chunks and/or 2D blocks. Noise weights have
one row of length ns//2+1 per chunk, defined so that the filter is
irfft(rfft(t) * w). Chunks are used as views (no copies) and processed one at a
time, so the memory per sample is the TOD plus 12 bytes (int32 pixel, float32
gamma cos 2a and gamma sin 2a).
"""

import os
from contextlib import contextmanager

import ducc0
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


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _chunks(x):
    """1D chunks (views) from a 2D block (one chunk per row), a 1D chunk, or a list of them."""
    out = []
    for b in _as_list(x):
        b = np.asarray(b)
        out.extend(b if b.ndim == 2 else [b])
    return out


def _efficiency_chunks(eff, psi):
    """Polarization efficiency of each chunk (scalar or per sample), matching _chunks(psi)."""
    out = []
    for e, p in zip(eff, _as_list(psi)):
        e, p = np.asarray(e, dtype=float), np.asarray(p)
        if p.ndim == 1:
            out.append(e)
            continue
        e = e.reshape(-1, 1) if e.ndim < 2 else e
        out.extend(e[i if e.shape[0] > 1 else 0] for i in range(p.shape[0]))
    return out


def _filter(t, w, nthreads):
    """irfft(rfft(t) * w); ducc0 parallelizes a single 1D FFT, scipy does not."""
    f = ducc0.fft.r2c(t, nthreads=nthreads)
    f *= w
    return ducc0.fft.c2r(f, lastsize=t.size, forward=False, inorm=2, nthreads=nthreads)


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
def _backproject(ip, t, c2, s2, pol, buf):
    """buf[th] += P^T t over the samples of thread th; the caller sums buf over threads."""
    # ponytail: one private I/Q/U map per thread (~75 MB each at nside 512); switch
    # to pixel-sorted samples if memory per process becomes the limit
    nthreads = buf.shape[0]
    step = (ip.size + nthreads - 1) // nthreads
    for th in numba.prange(nthreads):
        for i in range(th * step, min(ip.size, (th + 1) * step)):
            p = ip[i]
            buf[th, 0, p] += t[i]
            if pol:
                buf[th, 1, p] += t[i] * c2[i]
                buf[th, 2, p] += t[i] * s2[i]


@numba.njit(cache=True)
def _coverage(ip, c2, s2, pol, out):
    """Per pixel: hits, polarized hits (gamma > 0.5, as SANEPIC) and the sums of
    c2, s2, c2^2, c2 s2, s2^2 (angle-coverage matrix)."""
    # ponytail: serial, one pass over the data at setup; parallelize if it shows up
    for i in range(ip.size):
        p = ip[i]
        out[p, 0] += 1.0
        if pol:
            c, s = c2[i], s2[i]
            if c * c + s * s > 0.25:
                out[p, 1] += 1.0
            out[p, 2] += c
            out[p, 3] += s
            out[p, 4] += c * c
            out[p, 5] += c * s
            out[p, 6] += s * s


@numba.njit(parallel=True, cache=True)
def _block_circulant(ip, c, c2, s2, pol, out):
    """Exact per-pixel blocks of P^T C P, C circulant per chunk.

    For each pixel, sums c[(t - t') mod ns] w_t w_t'^T over all sample pairs
    that fall in it, with w = (1, c2, s2), and adds them to out (nobs, 6) with the
    entries II, IQ, IU, QQ, QU, UU (only II if pol is False).
    Cost is sum over pixels of hits^2 per chunk; the pixels of a chunk are
    processed in parallel (each one writes its own row of out).
    """
    # ponytail: O(hits^2) per pixel per chunk; switch to a lag cutoff or the
    # SANEPIC N_[0] shortcut if deep pixels with long chunks become a bottleneck
    nchunk, ns = ip.shape
    for k in range(nchunk):
        order = np.argsort(ip[k], kind="mergesort")
        sorted_ip = ip[k][order]
        starts = np.flatnonzero(np.diff(sorted_ip)) + 1
        bounds = np.empty(starts.size + 2, np.int64)
        bounds[0] = 0
        bounds[1:-1] = starts
        bounds[-1] = ns
        for r in numba.prange(bounds.size - 1):
            start, stop = bounds[r], bounds[r + 1]
            p = sorted_ip[start]
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
                continue
            b0 = b1 = b2 = b3 = b4 = b5 = 0.0
            for a in range(start, stop):
                ta = order[a]
                for b in range(start, stop):
                    tb = order[b]
                    cab = c[k, (ta - tb) % ns]
                    b0 += cab
                    if pol:
                        b1 += cab * c2[k, tb]
                        b2 += cab * s2[k, tb]
                        b3 += cab * c2[k, ta] * c2[k, tb]
                        b4 += cab * c2[k, ta] * s2[k, tb]
                        b5 += cab * s2[k, ta] * s2[k, tb]
            out[p, 0] += b0
            out[p, 1] += b1
            out[p, 2] += b2
            out[p, 3] += b3
            out[p, 4] += b4
            out[p, 5] += b5


class GLS:
    """Map-maker for a fixed set of pointings and noise weights.

    Parameters
    ----------
    pix : int array (nchunk, ns), or list of 1D chunks / 2D blocks
        Pixel index of each sample, 0 <= pix < 2^31 (e.g. HEALPix). Maps have
        max(pix) + 1 pixels; pixels without hits are not solved. Used without
        copies if int32.
    psi : float array(s) like `pix`, or None
        Polarization angle of each sample. None for intensity only.
    weights : float array (nchunk, ns//2+1), or list like `pix`
        Inverse noise power spectrum of each chunk on the rfft grid; chunks may
        share the same array.
    pol_efficiency : scalar, or per block (nchunk,) or (nchunk, ns), or per chunk, or None
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
        Threads per process (FFTs, numba); None: OMP_NUM_THREADS, else 1.
    """

    def __init__(self, pix, psi, weights, pol_efficiency=None, min_pol_hits=4, min_pol_rcond=1e-2, preconditioner="block", comm=None, nthreads=None):
        self.comm = comm
        self.nthreads = resolve_nthreads(nthreads)
        self.ip = []
        for p in _chunks(pix):
            if p.size and (p.min() < 0 or p.max() >= 2**31):
                raise ValueError("pixel indices must be in [0, 2^31)")
            self.ip.append(np.ascontiguousarray(p, dtype=np.int32))
        self.weights = _chunks(weights)
        nobs = 1 + max((int(p.max()) for p in self.ip if p.size), default=-1)
        self.nobs = nobs if comm is None else max(comm.allgather(nobs))
        self.pol = psi is not None
        self.ncomp = 3 if self.pol else 1
        self._dummy = np.zeros(1, np.float32)
        if self.pol:
            eff = [1.0] * len(_as_list(psi)) if pol_efficiency is None else _as_list(pol_efficiency)
            gamma = _efficiency_chunks(eff, psi)
            psi = _chunks(psi)
            self.cos2 = [(g * np.cos(2 * a)).astype(np.float32) for g, a in zip(gamma, psi)]
            self.sin2 = [(g * np.sin(2 * a)).astype(np.float32) for g, a in zip(gamma, psi)]

        cov = np.zeros((self.nobs, 7))
        for k, ip in enumerate(self.ip):
            _coverage(ip, *self._pol_arrays(k)[1:], self.pol, cov)
        self._cov = self._sum(cov)
        self.hits = self._cov[:, 0].astype(np.int64)
        self.mask = np.repeat((self.hits > 0)[None], self.ncomp, axis=0)
        if self.pol:
            self.mask[1:] &= self._cov[:, 1] >= min_pol_hits
            if min_pol_rcond > 0:
                self.mask[1:] &= self._pol_rcond() >= min_pol_rcond

        blocks = np.zeros((self.nobs, 6))
        with numba_threads(self.nthreads):
            for k, (ip, w) in enumerate(zip(self.ip, self.weights)):
                c = scipy.fft.irfft(w, n=ip.size)
                _, c2, s2 = self._pol_arrays(k)
                _block_circulant(ip[None], c[None], c2[None], s2[None], self.pol, blocks)
        blocks = self._sum(blocks)
        if np.any(blocks[self.mask[0], 0] <= 0):
            raise ValueError("Preconditioner has non-positive elements: check noise weights")
        self._set_preconditioner(blocks, preconditioner)
        # private maps of the threads for P^T
        self._buf = np.zeros((min(self.nthreads, numba.config.NUMBA_NUM_THREADS), self.ncomp, self.nobs))

    def _pol_rcond(self):
        """Reciprocal condition number of the angle-coverage matrix of each pixel."""
        n, c, s, cc, cs, ss = self._cov[:, 0], *self._cov[:, 2:].T
        M = np.stack([n, c, s, c, cc, cs, s, cs, ss], axis=1).reshape(-1, 3, 3)
        rcond = np.zeros(self.nobs)
        seen = self.mask[1]  # pixels with enough polarized hits
        with np.errstate(divide="ignore"):
            rcond[seen] = 1.0 / np.linalg.cond(M[seen])
        return rcond

    def _set_preconditioner(self, blocks, kind):
        diag_i = blocks[:, 0]
        if kind == "jacobi" or not self.pol:
            inv = np.divide(1.0, diag_i, out=np.zeros_like(diag_i), where=self.mask[0])
            self.precond = np.array([inv] + [2.0 * inv] * (self.ncomp - 1)) * self.mask
            return
        if kind != "block":
            raise ValueError(f"Unknown preconditioner {kind!r}: use 'block' or 'jacobi'")
        ii, iq, iu, qq, qu, uu = blocks.T
        M = np.stack([ii, iq, iu, iq, qq, qu, iu, qu, uu], axis=1).reshape(-1, 3, 3)
        # Invert well-conditioned blocks; elsewhere fall back to the diagonal
        good = self.mask[1] & (np.linalg.det(M) > 1e-12 * ii * qq * uu)
        self.precond = np.zeros((self.nobs, 3, 3))
        self.precond[good] = np.linalg.inv(M[good])
        bad = self.mask[0] & ~good
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
            return True, self.cos2[k], self.sin2[k]
        return False, self._dummy, self._dummy

    def _apply(self, m=None, tod=None):
        """P^T N^-1 P m (if m is given) or P^T N^-1 tod, one chunk at a time."""
        buf = self._buf
        buf[:] = 0.0
        with numba_threads(self.nthreads):
            nt = numba.get_num_threads()
            for k, ip in enumerate(self.ip):
                pol, c2, s2 = self._pol_arrays(k)
                if m is None:
                    t = np.asarray(tod[k], dtype=float)
                else:
                    t = np.empty(ip.size)
                    _project(ip, m, c2, s2, pol, t)
                _backproject(ip, _filter(t, self.weights[k], self.nthreads), c2, s2, pol, buf[:nt])
        return self._sum(buf.sum(axis=0)) * self.mask

    def apply_A(self, m):
        """P^T N^-1 P m for a map (ncomp, nobs)."""
        return self._apply(m=m)

    def rhs(self, tod):
        """P^T N^-1 d for timelines laid out like `pix`."""
        return self._apply(tod=_chunks(tod))

    def solve(self, tod, x0=None, tol=1e-15, maxiter=2000, recompute_every=10, verbose=False):
        """Run PCG on `tod` (same chunk layout as `pix`).

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
        """Expand (ncomp, nobs) to full RING maps (ncomp, 12 nside^2); labels
        beyond the map (virtual pixels) are dropped."""
        full = np.full((m.shape[0], 12 * nside * nside), fill)
        n = min(full.shape[1], self.nobs)
        full[:, :n] = np.where(self.mask[:, :n], m[:, :n], fill)
        return full
