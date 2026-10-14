"""GLS map-making with correlated noise (the SANEPIC algorithm).

Solves (P^T N^-1 P) m = P^T N^-1 d with preconditioned conjugate gradients.
N^-1 is applied in Fourier space, independently on each data chunk
(one detector x one stretch of time), as a circulant filter.

Data layout: data come as *chunks*. Pass a 2D array (nchunk, ns) for chunks of
the same length, or a list of 1D chunks and/or 2D blocks. Noise weights have
one row of length ns//2+1 per chunk, defined so that the filter is
irfft(rfft(t) * w). Chunks are used as views (no copies) and processed one at a
time, so the memory per sample is the TOD plus 8 bytes (int32 pixel and
float32 2a; 12 bytes with angle_dtype=np.float64). gamma cos 2a and
gamma sin 2a are recomputed for each chunk when it is used.
"""

import os
from contextlib import contextmanager

import ducc0
import numba
import numpy as np


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


def one_over_f_weights(ns, fsamp_hz, sigma, fknee_hz, alpha, fmin_hz=0.0, model="toast"):
    """Inverse noise PSD on the rfft grid of a chunk of `ns` samples.

    Models (the two of litebird_sim's `add_noise`), with sigma the white-noise
    rms per sample, so that for white noise the filter is exactly 1/sigma^2:

    - "toast": P(f) = sigma^2 (f^alpha + fknee^alpha) / (f^alpha + fmin^alpha);
    - "keshner": P(f) = sigma^2 ((f^2 + fknee^2) / (f^2 + fmin^2))^(alpha/2),
      the one of ducc0's OofaNoise (litebird_sim's engine="ducc").

    With fmin = 0 the DC weight is 0 and the mean of each chunk is not used.
    """
    f = np.fft.rfftfreq(ns, 1.0 / fsamp_hz)
    if model == "toast":
        num, den = f**alpha + fmin_hz**alpha, f**alpha + fknee_hz**alpha
    elif model == "keshner":
        num, den = (f**2 + fmin_hz**2) ** (alpha / 2), (f**2 + fknee_hz**2) ** (alpha / 2)
    else:
        raise ValueError(f"Unknown noise model {model!r}: use 'toast' or 'keshner'")
    ratio = np.divide(num, den, out=np.ones_like(f), where=den > 0)  # fknee = 0: white
    return ratio / sigma**2


def psd_weights(ns, fsamp_hz, freqs_hz, psd):
    """Inverse noise PSD on the rfft grid of a chunk of `ns` samples from a
    tabulated one-sided PSD `psd` [K^2/Hz] at `freqs_hz` (increasing, > 0),
    interpolated linearly in log f - log P and held constant outside the table
    (also at f = 0). White noise of rms sigma per sample has P = 2 sigma^2 / fs,
    and gives the same weights 1/sigma^2 as one_over_f_weights.
    """
    f = np.fft.rfftfreq(ns, 1.0 / fsamp_hz)
    logf = np.log(np.maximum(f, freqs_hz[0]))
    p = np.exp(np.interp(logf, np.log(freqs_hz), np.log(psd)))
    return 2.0 / (fsamp_hz * p)


def two_angle(a):
    """2a reduced to [-pi, pi), where float32 has its best absolute precision."""
    return _reduce_angle(2 * np.asarray(a, dtype=float))


def _reduce_angle(x):
    """x reduced to [-pi, pi), as _cos_sin requires."""
    return np.remainder(np.asarray(x, dtype=float) + np.pi, 2 * np.pi) - np.pi


def _reduced(a, dtype):
    """a as dtype, reduced to [-pi, pi) only if needed (make_maps passes it
    reduced already; float32 rounding of +-pi is within 1e-7)."""
    a = np.asarray(a)
    if a.size and (a.min() < -np.pi - 1e-6 or a.max() > np.pi + 1e-6):
        a = _reduce_angle(a)
    return np.ascontiguousarray(a, dtype=dtype)


def _check_mpi_count(n):
    """MPI counts and displacements are int32: raise early, with a clear message,
    instead of failing inside a collective."""
    # ponytail: fine up to nside 4096; beyond, count whole pixels with a derived
    # datatype of k doubles
    if n >= 2**31:
        raise ValueError(
            f"{n} values per MPI exchange exceed the int32 counts of MPI: the "
            "distributed PCG supports maps up to nside 4096"
        )


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
    """irfft(rfft(t) * w); ducc0 parallelizes a single 1D FFT, scipy does not.
    Batching several chunks per call was 2.4x faster on an idle node but gave
    nothing on a full one (memory bandwidth bound), so chunks go one at a time."""
    f = ducc0.fft.r2c(t, nthreads=nthreads)
    f *= w
    return ducc0.fft.c2r(f, lastsize=t.size, forward=False, inorm=2, nthreads=nthreads)


@numba.njit(parallel=True, cache=True)
def _project(ip, m, c2, s2, pol, out):
    """out[i] = I[p] + c2[i] Q[p] + s2[i] U[p], p = ip[i]; m is (npix, ncomp)."""
    for i in numba.prange(ip.size):
        p = ip[i]
        v = m[p, 0]
        if pol:
            v += m[p, 1] * c2[i] + m[p, 2] * s2[i]
        out[i] = v


@numba.njit(parallel=True, cache=True)
def _backproject(ip, t, c2, s2, pol, edges, buf):
    """buf += P^T t, buf (npix, ncomp). Thread th owns the pixels
    [edges[th], edges[th + 1]): it reads all the pixel indices of the chunk but
    adds only the samples of its pixels, so the threads share one map without
    races and without private copies."""
    for th in numba.prange(edges.size - 1):
        lo, hi = edges[th], edges[th + 1]
        for i in range(ip.size):
            p = ip[i]
            if lo <= p < hi:
                buf[p, 0] += t[i]
                if pol:
                    buf[p, 1] += t[i] * c2[i]
                    buf[p, 2] += t[i] * s2[i]


# cos and sin on a grid of [-pi, pi), for _cos_sin
_TRIG_N = 4096
_TRIG_COS = np.cos(-np.pi + 2 * np.pi * np.arange(_TRIG_N) / _TRIG_N)
_TRIG_SIN = np.sin(-np.pi + 2 * np.pi * np.arange(_TRIG_N) / _TRIG_N)


@numba.njit(parallel=True, cache=True)
def _cos_sin(a2, gamma, c2, s2, tc, ts):
    """c2 = gamma cos(a2), s2 = gamma sin(a2), in float64 then stored; gamma has
    one value or one per sample; a2 in [-pi, pi) (float32 rounding may put it
    just outside). cos and sin come from the grid point below a2 (tables tc, ts)
    and the Taylor series of the remainder d < 2pi/4096: exact to ~1e-16 and
    10x faster than libm on G100 (numba without SVML). No branch: with a libm
    fallback for unreduced angles LLVM computed cos and sin for every sample."""
    one = gamma.size == 1
    n = tc.size
    step = 2 * np.pi / n
    for i in numba.prange(a2.size):
        g = gamma[0] if one else gamma[i]
        x = np.float64(a2[i]) + np.pi
        k = min(max(int(x / step), 0), n - 1)
        d = x - k * step
        d2 = d * d
        cd = 1.0 - d2 * (0.5 - d2 / 24.0)
        sd = d * (1.0 - d2 * (1.0 / 6.0 - d2 / 120.0))
        c2[i] = g * (tc[k] * cd - ts[k] * sd)
        s2[i] = g * (ts[k] * cd + tc[k] * sd)


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


@numba.njit(cache=True)
def _pixel_order(ip, npix):
    """Indices of the samples sorted by pixel, in time order within each pixel:
    a stable counting sort, O(ns + npix), instead of an O(ns log ns) argsort."""
    start = np.zeros(npix + 1, np.int64)
    for i in range(ip.size):
        start[ip[i] + 1] += 1
    for p in range(npix):
        start[p + 1] += start[p]
    order = np.empty(ip.size, np.int64)
    for i in range(ip.size):
        p = ip[i]
        order[start[p]] = i
        start[p] += 1
    return order


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
        order = _pixel_order(ip[k], out.shape[0])
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
        With MPI, each rank passes only its own chunks (possibly none). Each
        rank owns a contiguous range of pixels [lo, hi), 1/size of the map, and
        keeps the PCG vectors, the preconditioner and the mask only there: P^T
        is summed over ranks with a reduce-scatter that delivers to each rank
        its pixels, and the search direction is rebuilt on every rank for P
        with an allgather (together, the same traffic as one Allreduce). Only
        the map used by P and the map accumulating P^T are full size.
    nthreads : int or None
        Threads per process (FFTs, numba); None: OMP_NUM_THREADS, else 1.
    pol_angles : (angle2, gamma) or None
        Instead of psi and pol_efficiency: `angle2`, twice the polarization
        angle of each sample laid out like `pix`, reduced to [-pi, pi) (see
        :func:`two_angle`; used without copies if of `angle_dtype`), and
        `gamma`, the polarization efficiency of each chunk (scalar, or one
        value per sample, e.g. 0 in intensity-only samples).
    angle_dtype : np.float32 (default) or np.float64
        Storage of 2a (and of a per-sample efficiency): float32 gives an angle
        error of ~1e-7 rad.
    """

    def __init__(self, pix, psi, weights, pol_efficiency=None, min_pol_hits=4, min_pol_rcond=1e-2, preconditioner="block", comm=None, nthreads=None, angle_dtype=np.float32, pol_angles=None):
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
        self.pol = psi is not None or pol_angles is not None
        self.ncomp = 3 if self.pol else 1
        self._dummy = np.zeros(1, angle_dtype)
        if pol_angles is not None:
            angle2, gamma = pol_angles
            self.angle2 = [_reduced(a, angle_dtype) for a in _chunks(angle2)]
            gamma = _as_list(gamma)
        elif self.pol:
            eff = [1.0] * len(_as_list(psi)) if pol_efficiency is None else _as_list(pol_efficiency)
            gamma = _efficiency_chunks(eff, psi)
            self.angle2 = [two_angle(a).astype(angle_dtype) for a in _chunks(psi)]
        if self.pol:
            # one value per chunk in float64, or one per sample in angle_dtype
            self.gamma = [np.atleast_1d(np.asarray(g, dtype=float if np.ndim(g) == 0 else angle_dtype)) for g in gamma]
            nmax = max((ip.size for ip in self.ip), default=0)
            self._c2, self._s2 = np.empty(nmax, angle_dtype), np.empty(nmax, angle_dtype)

        # pixels [lo, hi) of this rank
        size, rank = (1, 0) if comm is None else (comm.size, comm.rank)
        if size > 1:
            _check_mpi_count(self.nobs * 7)  # the largest array exchanged: the coverage
        bounds = np.arange(size + 1, dtype=np.int64) * self.nobs // size
        self._counts = np.diff(bounds)
        self.lo, self.hi = int(bounds[rank]), int(bounds[rank + 1])
        self.nloc = self.hi - self.lo

        cov = np.zeros((self.nobs, 7))
        for k, ip in enumerate(self.ip):
            _coverage(ip, *self._pol_arrays(k)[1:], self.pol, cov)
        # P^T: each thread owns a range of pixels holding ~1/nthreads of this
        # rank's samples
        nt = min(self.nthreads, numba.config.NUMBA_NUM_THREADS)
        cum = np.cumsum(cov[:, 0])
        target = (cum[-1] if cum.size else 0.0) * np.arange(nt + 1) / nt
        self._edges = np.searchsorted(cum, target).astype(np.int64)
        self._edges[[0, -1]] = 0, self.nobs
        cov = self._scatter_sum(cov, by_column=True)
        self.hits = cov[:, 0].astype(np.int64)  # this rank's pixels, like mask and precond
        self.mask = np.repeat((self.hits > 0)[None], self.ncomp, axis=0)
        if self.pol:
            self.mask[1:] &= cov[:, 1] >= min_pol_hits
            if min_pol_rcond > 0:
                self.mask[1:] &= self._pol_rcond(cov) >= min_pol_rcond
        del cov

        blocks = np.zeros((self.nobs, 6))
        last_w = c = None  # consecutive chunks often share their weights array
        with numba_threads(self.nthreads):
            for k, (ip, w) in enumerate(zip(self.ip, self.weights)):
                if w is not last_w or c.size != ip.size:
                    # first row of the circulant N^-1 of the chunk
                    c = ducc0.fft.c2r(np.asarray(w, dtype=complex), lastsize=ip.size, forward=False, inorm=2, nthreads=self.nthreads)
                    last_w = w
                _, c2, s2 = self._pol_arrays(k)
                _block_circulant(ip[None], c[None], c2[None], s2[None], self.pol, blocks)
        blocks = self._scatter_sum(blocks, by_column=True)
        bad = int(np.any(blocks[self.mask[0], 0] <= 0))
        if (bad if comm is None else comm.allreduce(bad)) > 0:
            raise ValueError("Preconditioner has non-positive elements: check noise weights")
        self._set_preconditioner(blocks, preconditioner)
        del blocks
        self._full = self._buf = None  # full-size work maps, allocated by _apply

    def _pol_rcond(self, cov):
        """Reciprocal condition number of the angle-coverage matrix of each pixel."""
        n, c, s, cc, cs, ss = cov[:, 0], *cov[:, 2:].T
        M = np.stack([n, c, s, c, cc, cs, s, cs, ss], axis=1).reshape(-1, 3, 3)
        rcond = np.zeros(self.nloc)
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
        self.precond = np.zeros((self.nloc, 3, 3))
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

    def _scatter_sum(self, full, by_column=False):
        """Sum a full (nobs, k) array over ranks; return this rank's pixels (nloc, k).
        A reduce-scatter: the first half of an Allreduce. by_column (setup) caps the
        MPI temporaries at one column, not the whole array (2.8 GB for the coverage
        at nside 2048), at the price of strided copies."""
        if self.comm is None:
            return full
        out = np.empty((self.nloc, full.shape[1]), full.dtype)
        if not by_column:
            self.comm.Reduce_scatter(np.ascontiguousarray(full), out, recvcounts=self._counts * full.shape[1])
            return out
        col = np.empty(self.nloc, full.dtype)
        for j in range(full.shape[1]):
            self.comm.Reduce_scatter(np.ascontiguousarray(full[:, j]), col, recvcounts=self._counts)
            out[:, j] = col
        return out

    def _gather(self, loc, out=None):
        """Full (nobs, k) array from the (nloc, k) pieces of all ranks (allgather)."""
        # counts fit in int32: checked by _check_mpi_count at setup
        out = np.empty((self.nobs, loc.shape[1]), loc.dtype) if out is None else out
        if self.comm is None:
            out[:] = loc
        else:
            self.comm.Allgatherv(np.ascontiguousarray(loc), [out, self._counts * loc.shape[1]])
        return out

    def full(self, x):
        """Gather a per-pixel array of this rank, (k, nloc) or (nloc,), into the
        full (k, nobs) or (nobs,) array on every rank."""
        x = np.asarray(x)
        if x.ndim == 1:
            return self._gather(x[:, None])[:, 0]
        return np.ascontiguousarray(self._gather(x.T).T)

    def _dot(self, a, b):
        d = np.vdot(a, b)
        return d if self.comm is None else self.comm.allreduce(d)

    def _pol_arrays(self, k):
        """(pol, gamma cos 2a, gamma sin 2a) of chunk k, computed into buffers
        that the next call overwrites."""
        if not self.pol:
            return False, self._dummy, self._dummy
        n = self.ip[k].size
        c2, s2 = self._c2[:n], self._s2[:n]
        with numba_threads(self.nthreads):
            _cos_sin(self.angle2[k], self.gamma[k], c2, s2, _TRIG_COS, _TRIG_SIN)
        return True, c2, s2

    def _apply(self, m=None, tod=None):
        """P^T N^-1 P m (m: this rank's pixels, (ncomp, nloc)) or P^T N^-1 tod,
        one chunk at a time; returns this rank's pixels (ncomp, nloc)."""
        if self._buf is None:
            # the search direction for P, and the map for P^T
            self._full = np.empty((self.nobs, self.ncomp))
            self._buf = np.empty((self.nobs, self.ncomp))
        if m is not None:
            self._gather(m.T, out=self._full)  # the whole map, for P
        buf = self._buf
        buf[:] = 0.0
        with numba_threads(self.nthreads):
            for k, ip in enumerate(self.ip):
                pol, c2, s2 = self._pol_arrays(k)
                if m is None:
                    t = np.asarray(tod[k], dtype=float)
                else:
                    t = np.empty(ip.size)
                    _project(ip, self._full, c2, s2, pol, t)
                _backproject(ip, _filter(t, self.weights[k], self.nthreads), c2, s2, pol, self._edges, buf)
        return self._scatter_sum(buf).T * self.mask

    def apply_A(self, m):
        """P^T N^-1 P m for this rank's pixels of a map, (ncomp, nloc) (without
        MPI, the whole map); returns the same pixels. Collective with MPI."""
        return self._apply(m=m)

    def rhs(self, tod):
        """P^T N^-1 d on this rank's pixels (ncomp, nloc), for timelines laid out
        like `pix`. Collective with MPI."""
        return self._apply(tod=_chunks(tod))

    def solve(self, tod, x0=None, tol=1e-15, maxiter=2000, recompute_every=10, verbose=False):
        """Run PCG on `tod` (same chunk layout as `pix`).

        Stops when norm(r)^2 / norm(b)^2 < tol (same criterion as SANEPIC).
        x0: optional starting map (ncomp, nobs). Returns (map (ncomp, nobs) on
        every rank, info dict).
        """
        b = self.rhs(tod)
        bb = self._dot(b, b)
        x = np.zeros_like(b) if x0 is None else x0[:, self.lo : self.hi] * self.mask
        r = b - self.apply_A(x)
        z = self._apply_precond(r)
        d = z.copy()
        delta = self._dot(r, z)
        history = [self._dot(r, r) / bb]

        it = 0
        while it < maxiter and (it == 0 or history[-1] > tol):
            q = self.apply_A(d)
            alpha = delta / self._dot(d, q)
            x += alpha * d
            if it % recompute_every == 0:
                r = b - self.apply_A(x)  # true residual, avoids drift
            else:
                r -= alpha * q
            z = self._apply_precond(r)
            delta_new = self._dot(r, z)
            d = z + (delta_new / delta) * d
            delta = delta_new
            history.append(self._dot(r, r) / bb)
            it += 1
            if verbose and (self.comm is None or self.comm.rank == 0):
                print(f"iter {it:4d}  |r|^2/|b|^2 = {history[-1]:.3e}")

        info = {"iterations": it, "converged": history[-1] <= tol, "residuals": np.array(history)}
        del b, r, z, d  # q does not exist if maxiter == 0
        self._full = self._buf = None  # free the work maps before gathering the result
        # full mask now, while all ranks are here, so that to_healpix is not collective
        self._full_mask = self.full(self.mask.astype(np.uint8)).astype(bool)
        return self._gather(x.T).T, info

    def to_healpix(self, m, nside, fill=np.nan):
        """Expand a full map (ncomp, nobs) to RING maps (ncomp, 12 nside^2); labels
        beyond the map (virtual pixels) are dropped. Collective with MPI only if
        called before solve()."""
        if getattr(self, "_full_mask", None) is None:
            self._full_mask = self.full(self.mask.astype(np.uint8)).astype(bool)
        unseen = ~self._full_mask
        full = np.full((m.shape[0], 12 * nside * nside), fill)
        n = min(full.shape[1], self.nobs)
        full[:, :n] = m[:, :n]
        full[:, :n][unseen[:, :n]] = fill
        return full
