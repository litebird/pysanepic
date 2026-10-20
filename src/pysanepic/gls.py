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


def fast_fft_length(n):
    """The smallest even length >= n with only small prime factors: an FFT of
    6480001 samples (prime factor 6619) is 7 times slower than one of 6553600."""
    m = ducc0.fft.good_size(n, True)
    while m % 2:
        m = ducc0.fft.good_size(m + 1, True)
    return m


@numba.njit(parallel=True, cache=True)
def _project(ip, m, c2, s2, pol, loc, out):
    """out[i] = I[p] + c2[i] Q[p] + s2[i] U[p], p = ip[i]; m is (nobs, ncomp).
    Labels p >= nobs are flagged samples: out[i] = loc[p - nobs]."""
    nobs = m.shape[0]
    for i in numba.prange(ip.size):
        p = ip[i]
        if p >= nobs:
            out[i] = loc[p - nobs]
            continue
        v = m[p, 0]
        if pol:
            v += m[p, 1] * c2[i] + m[p, 2] * s2[i]
        out[i] = v


@numba.njit(parallel=True, cache=True)
def _backproject(ip, t, c2, s2, pol, edges, buf, locbuf):
    """buf += P^T t, buf (nobs, ncomp). Thread th owns the pixels
    [edges[th], edges[th + 1]): it reads all the pixel indices of the chunk but
    adds only the samples of its pixels, so the threads share one map without
    races and without private copies. Flagged samples (labels p >= nobs) go to
    locbuf[p - nobs], written by thread 0."""
    nobs = buf.shape[0]
    for th in numba.prange(edges.size - 1):
        lo, hi = edges[th], edges[th + 1]
        for i in range(ip.size):
            p = ip[i]
            if p >= nobs:
                if th == 0:
                    locbuf[p - nobs] += t[i]
            elif lo <= p < hi:
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
        if p >= out.shape[0]:  # flagged sample
            continue
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
def _block_circulant(ip, c, c2, s2, pol, out, nlabels, locout):
    """Exact per-pixel blocks of P^T C P, C circulant per chunk.

    For each pixel, sums c[(t - t') mod ns] w_t w_t'^T over all sample pairs
    that fall in it, with w = (1, c2, s2), and adds them to out (nobs, 6) with the
    entries II, IQ, IU, QQ, QU, UU (only II if pol is False). Labels are
    < nlabels; those >= nobs (unknowns of flagged samples, intensity only) add
    their II block to locout[label - nobs].
    Cost is sum over pixels of hits^2 per chunk; the pixels of a chunk are
    processed in parallel (each one writes its own row of out).
    """
    # ponytail: O(hits^2) per pixel per chunk; switch to a lag cutoff or the
    # SANEPIC N_[0] shortcut if deep pixels with long chunks become a bottleneck
    nchunk, ns = ip.shape
    nc = c.shape[1]  # FFT length >= ns: c is the first row of an nc x nc circulant
    for k in range(nchunk):
        order = _pixel_order(ip[k], nlabels)
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
            local = p >= out.shape[0]  # an unknown of flagged samples: no Q/U
            # Contiguous run of samples with no polarization sensitivity (e.g. the
            # padding of a chunk): sum over lags in O(n) instead of O(n^2)
            unpol = True
            if pol and not local:
                for a in range(start, stop):
                    if c2[k, order[a]] != 0.0 or s2[k, order[a]] != 0.0:
                        unpol = False
                        break
            if unpol and order[stop - 1] - order[start] == n - 1:
                acc = 0.0
                for lag in range(-(n - 1), n):
                    acc += (n - abs(lag)) * c[k, lag % nc]
                if local:
                    locout[p - out.shape[0]] += acc
                else:
                    out[p, 0] += acc
                continue
            if local:
                acc = 0.0
                for a in range(start, stop):
                    for b in range(start, stop):
                        acc += c[k, (order[a] - order[b]) % nc]
                locout[p - out.shape[0]] += acc
                continue
            b0 = b1 = b2 = b3 = b4 = b5 = 0.0
            for a in range(start, stop):
                ta = order[a]
                for b in range(start, stop):
                    tb = order[b]
                    cab = c[k, (ta - tb) % nc]
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
        Pixel index of each sample, pix < 2^31 (e.g. HEALPix). Maps have
        max(pix) + 1 pixels; pixels without hits are not solved. Used without
        copies if int32 (chunks with flagged samples are copied, see
        overwrite_pix). pix < 0 marks
        flagged samples: they do not enter the maps and their TOD is not used
        (taken as 0). Samples of a chunk with the same negative value share an
        intensity unknown, local to this rank and discarded: one value per
        sample is the exact GLS without them (marginalization), one per
        contiguous run fits an offset to it.
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
    fft_lengths : int or list of int (one per chunk), or None
        Length m >= ns of the FFT used to apply N^-1 to each chunk (the chunk is
        zero-extended and the result truncated: the weight is the ns x ns block
        of an m x m circulant, still symmetric positive definite, so the maps
        stay unbiased); `weights` are then on the rfft grid of m. None: the
        chunk length. See :func:`fast_fft_length`.
    fft_dtype : np.float64 (default) or np.float32
        Precision of the FFTs applying N^-1 to the search directions; float32
        is ~45% faster on G100 (relative error ~3e-7). The right-hand side and
        the true residual, recomputed every `recompute_every` iterations and
        at convergence, always use float64, so the solution is still
        controlled in double precision.
    common_modes : list of (chunks, coupling, mode_psd), or None
        Noise correlated between the chunks `chunks` (indices of chunks with
        the same length and FFT length, observed at the same times, e.g. the
        detectors of a wafer over one day): N = D + U Lambda U^T in Fourier
        space, with D the diagonal of the chunks' own noise (1 / weights),
        U = `coupling` (len(chunks), r) and Lambda = diag(`mode_psd`) (r,
        m//2+1), the spectra of r common modes in the units of 1 / weights.
        N^-1 is applied with the Woodbury identity in two passes over the
        chunks (one chunk in memory at a time), in float64; the
        preconditioner uses D only (see `coarse`).
    coarse : callable or None
        Two-level preconditioner: `coarse(labels)` gives the index of the
        coarse pixel of each pixel label of this rank (-1: none, e.g. virtual
        pixels). The preconditioner becomes M + Z E^-1 Z^T, with Z the
        indicators of the coarse pixels in I and E = Z^T P^T N^-1 P Z,
        computed at setup with one product by P^T N^-1 P per coarse pixel
        (the cost of ~1 PCG iteration each): it takes care of the slow
        large-scale modes of I, e.g. those of common modes. make_maps builds
        it from GLSParameters.coarse_nside.
    templates : list of (k, values, labels), or None
        Templates fitted together with the map: the TOD of chunk k is modeled
        as P m + values * a[labels], with `values` and `labels` (int, -1: no
        template) laid out like the chunk and a the template amplitudes, local
        to this rank like the unknowns of flagged samples (labels 0, 1, ... of
        this rank; several templates may share an amplitude, e.g. one gain per
        detector and period). Not applied to flagged samples, nor to samples in
        pixels whose Q/U are not solved (their polarized signal is not
        modelled and would bias the amplitudes). solve() returns
        the amplitudes in info["amplitudes"]; their preconditioner is the exact
        diagonal of T^T N^-1 T (own noise only, as for the pixels).
    overwrite_pix : bool
        Relabel the flagged samples in the int32 `pix` arrays in place instead
        of copying the chunks that have them (make_maps does, as it owns them).
    """

    def __init__(self, pix, psi, weights, pol_efficiency=None, min_pol_hits=4, min_pol_rcond=1e-2, preconditioner="block", comm=None, nthreads=None, angle_dtype=np.float32, pol_angles=None, overwrite_pix=False, fft_lengths=None, fft_dtype=np.float64, common_modes=None, coarse=None, templates=None):
        self.comm = comm
        self.nthreads = resolve_nthreads(nthreads)
        self.ip = []
        for p in _chunks(pix):
            if p.size and p.max() >= 2**31:
                raise ValueError("pixel indices must be < 2^31")
            self.ip.append(np.ascontiguousarray(p, dtype=np.int32))
        self.weights = _chunks(weights)
        if fft_dtype not in (np.float32, np.float64):
            raise ValueError("fft_dtype must be np.float32 or np.float64")
        self._single = fft_dtype == np.float32
        self._w32 = {}  # weights in float32, by id of the (shared) float64 arrays
        self._fftbuf = {}  # FFT work buffers, reused over chunks and iterations
        fl = [p.size for p in self.ip] if fft_lengths is None else np.broadcast_to(fft_lengths, (len(self.ip),)).tolist()
        self._nfft = [int(m) for m in fl]
        for p, w, m in zip(self.ip, self.weights, self._nfft):
            if m < p.size or len(w) != m // 2 + 1:
                raise ValueError(f"a chunk of {p.size} samples needs an FFT length >= {p.size} and weights of length m//2+1 (FFT length m = {m}, weights {len(w)})")
        self._groups = []  # (chunks, coupling, M = (I + Lambda S)^-1 Lambda per frequency)
        self._grouped = np.zeros(len(self.ip), bool)
        for chunks, coupling, mode_psd in common_modes or []:
            chunks = [int(k) for k in chunks]
            U = np.asarray(coupling, dtype=float).reshape(len(chunks), -1)
            lam = np.atleast_2d(np.asarray(mode_psd, dtype=float))
            m = self._nfft[chunks[0]]
            if any(self._nfft[k] != m or self.ip[k].size != self.ip[chunks[0]].size for k in chunks):
                raise ValueError("the chunks of a common mode must have the same length and FFT length")
            if lam.shape != (U.shape[1], m // 2 + 1) or np.any(lam < 0) or np.any(self._grouped[chunks]):
                raise ValueError("common_modes: mode_psd must be (r, m//2+1) and >= 0, and a chunk can be in one group only")
            S = np.zeros((m // 2 + 1, U.shape[1], U.shape[1]))  # U^T D^-1 U per frequency
            for j, k in enumerate(chunks):
                S += np.asarray(self.weights[k])[:, None, None] * np.outer(U[j], U[j])
            L = lam.T[:, :, None] * np.eye(U.shape[1])  # Lambda per frequency
            self._groups.append((chunks, U, np.linalg.solve(np.eye(U.shape[1]) + L @ S, L)))
            self._grouped[chunks] = True
        nobs = 1 + max((int(p.max()) for p in self.ip if p.size), default=-1)
        self.nobs = nobs if comm is None else max(comm.allgather(nobs))
        # flagged samples: labels nobs, nobs + 1, ... of unknowns local to this rank
        self.nflag = 0
        for k, p in enumerate(self.ip):
            flagged = np.flatnonzero(p < 0)
            if flagged.size:
                labels, unknown = np.unique(p[flagged], return_inverse=True)
                if not (overwrite_pix and p.flags.writeable):
                    p = p.copy()
                p[flagged] = self.nobs + self.nflag + unknown
                self.ip[k] = p
                self.nflag += labels.size
        if self.nobs + self.nflag >= 2**31:
            raise ValueError("too many pixels and flagged samples on this rank for int32 labels")
        # templates: amplitudes after the unknowns of flagged samples
        self._tmpl = [[] for _ in self.ip]
        self.namp = 0
        for k, values, labels in templates or []:
            labels = np.asarray(labels, dtype=np.int64)
            used = (labels >= 0) & (self.ip[k] < self.nobs)
            # ponytail: values kept in float64 (one TOD per template); float32 if memory matters
            self._tmpl[k].append((np.where(used, values, 0.0), np.where(used, labels, 0)))
            self.namp = max(self.namp, int(labels.max(initial=-1)) + 1)
        self._locprec = np.zeros(self.nflag)  # diagonal of P^T N^-1 P of each unknown, then inverted
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
        self._nsky = 0  # set below; PCG vectors: [this rank's pixels (ncomp, nloc), flagged samples]
        if size > 1:
            _check_mpi_count(self.nobs * 7)  # the largest array exchanged: the coverage
        bounds = np.arange(size + 1, dtype=np.int64) * self.nobs // size
        self._counts = np.diff(bounds)
        self.lo, self.hi = int(bounds[rank]), int(bounds[rank + 1])
        self.nloc = self.hi - self.lo
        self._nsky = self.ncomp * self.nloc

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
        if self.pol and (self.namp if comm is None else comm.allreduce(self.namp)):  # collective: all ranks
            solved = self.full(self.mask[1].astype(np.uint8)).astype(bool)
            for k, tmpl in enumerate(self._tmpl):
                bad = ~solved[np.minimum(self.ip[k], self.nobs - 1)] & (self.ip[k] < self.nobs)
                for values, _ in tmpl:
                    values[bad] = 0.0

        blocks = np.zeros((self.nobs, 6))
        last_w = c = None  # consecutive chunks often share their weights array
        with numba_threads(self.nthreads):
            for k, (ip, w) in enumerate(zip(self.ip, self.weights)):
                if w is not last_w or c.size != self._nfft[k]:
                    # first row of the circulant N^-1 of the chunk
                    c = ducc0.fft.c2r(np.asarray(w, dtype=complex), lastsize=self._nfft[k], forward=False, inorm=2, nthreads=self.nthreads)
                    last_w = w
                _, c2, s2 = self._pol_arrays(k)
                _block_circulant(ip[None], c[None], c2[None], s2[None], self.pol, blocks, self.nobs + self.nflag, self._locprec)
        self._locprec = np.concatenate([1.0 / self._locprec, self._template_precond()])
        blocks = self._scatter_sum(blocks, by_column=True)
        bad = int(np.any(blocks[self.mask[0], 0] <= 0))
        if (bad if comm is None else comm.allreduce(bad)) > 0:
            raise ValueError("Preconditioner has non-positive elements: check noise weights")
        self._set_preconditioner(blocks, preconditioner)
        del blocks
        self._full = self._buf = self._locbuf = None  # work maps, allocated by _apply
        self._einv = None
        if coarse is not None:
            self._set_coarse(coarse)

    def _template_precond(self):
        """Inverse diagonal of T^T N^-1 T (own noise of each chunk; 0 for unused
        amplitudes)."""
        # ponytail: one FFT per amplitude and chunk at setup (24 per daily chunk with
        # hourly gains); c[0] * sum(values^2) if it shows up
        diag = np.zeros(self.namp)
        for k, tmpl in enumerate(self._tmpl):
            for values, labels in tmpl:
                for a in np.unique(labels[values != 0]):
                    t = np.where(labels == a, values, 0.0)
                    diag[a] += np.dot(t, self._filter(t, k, False))
        return np.divide(1.0, diag, out=np.zeros_like(diag), where=diag > 0)

    def _template_back(self, k, f):
        """Add T^T f of chunk k (f = N^-1 t) to the amplitudes in _locbuf."""
        for values, labels in self._tmpl[k]:
            self._locbuf[self.nflag :] += np.bincount(labels, values * f, minlength=self.namp)

    def _set_coarse(self, coarse):
        """E^-1 = (Z^T A Z)^-1 of the two-level preconditioner (see `coarse`)."""
        parent = np.asarray(coarse(np.arange(self.lo, self.hi)), dtype=np.int64)
        parent[~self.mask[0]] = -1
        nz = int(parent.max(initial=-1)) + 1
        if self.comm is not None:
            nz = max(self.comm.allgather(nz))
        self._cidx = np.flatnonzero(parent >= 0)  # I entries of the PCG vectors
        self._cpar = parent[self._cidx]
        count = self._csum(np.ones(self._cidx.size), nz)
        used = np.flatnonzero(count > 0)
        E = np.zeros((nz, nz))
        x = np.zeros(self._nsky + self.nflag + self.namp)
        for q in used:  # the same coarse pixels on every rank: _apply is collective
            x[self._cidx] = self._cpar == q
            E[:, q] = np.bincount(self._cpar, self._apply(x)[self._cidx], minlength=nz)
        if self.comm is not None:
            E = self.comm.allreduce(E)
        E = E[np.ix_(used, used)]
        self._einv = np.zeros((nz, nz))
        self._einv[np.ix_(used, used)] = np.linalg.inv(0.5 * (E + E.T))
        self._nz = nz

    def _csum(self, v, nz):
        """Z^T v: sums of v (on the I entries of this rank) per coarse pixel, over ranks."""
        c = np.bincount(self._cpar, v, minlength=nz)
        return c if self.comm is None else self.comm.allreduce(c)

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

    def _filter(self, t, k, single):
        """N^-1 t for chunk k: irfft(rfft(t) * w) on an FFT of length m >= t.size
        (t zero-extended, the result truncated), in float32 if `single`. The
        work buffers are allocated once (for the longest FFT) and reused: the
        result is overwritten by the next call. ducc0 parallelizes a single 1D
        FFT; batching chunks gave nothing on a full node (memory bound)."""
        m, n, w = self._nfft[k], t.size, self.weights[k]
        real, cplx = (np.float32, np.complex64) if single else (np.float64, np.complex128)
        if real not in self._fftbuf:
            mmax = max(self._nfft)
            self._fftbuf[real] = (np.empty(mmax, real), np.empty(mmax // 2 + 1, cplx), np.empty(mmax, real))
        tin, f, out = (b[:size] for b, size in zip(self._fftbuf[real], (m, m // 2 + 1, m)))
        if m == n and not single and t.flags.c_contiguous:
            tin = t
        else:
            tin[:n] = t
            tin[n:] = 0.0
        ducc0.fft.r2c(tin, out=f, nthreads=self.nthreads)
        if single:
            if id(w) not in self._w32:
                self._w32[id(w)] = np.asarray(w, dtype=np.float32)
            w = self._w32[id(w)]
        f *= w
        ducc0.fft.c2r(f, lastsize=m, forward=False, inorm=2, out=out, nthreads=self.nthreads)
        return out[:n]

    def _apply(self, v=None, tod=None, exact=False):
        """P^T N^-1 P v or P^T N^-1 tod, one chunk at a time. v and the result
        are flat PCG vectors: this rank's pixels (ncomp, nloc) raveled, then the
        unknowns of its flagged samples. FFTs in float32 only for P^T N^-1 P v
        with fft_dtype=np.float32 and not `exact`."""
        single = self._single and v is not None and not exact
        if self._buf is None:
            # the search direction for P, and the map for P^T
            self._full = np.empty((self.nobs, self.ncomp))
            self._buf = np.empty((self.nobs, self.ncomp))
            self._locbuf = np.empty(self.nflag + self.namp)
        loc = self._locbuf[:0]
        if v is not None:
            self._gather(v[: self._nsky].reshape(self.ncomp, self.nloc).T, out=self._full)  # the whole map, for P
            loc = v[self._nsky :]
        buf = self._buf
        buf[:] = 0.0
        self._locbuf[:] = 0.0
        with numba_threads(self.nthreads):
            for k, ip in enumerate(self.ip):
                if self._grouped[k]:
                    continue
                pol, c2, s2 = self._pol_arrays(k)
                t = self._chunk_tod(k, v, tod, loc, c2, s2, pol)
                f = self._filter(t, k, single)
                _backproject(ip, f, c2, s2, pol, self._edges, buf, self._locbuf)
                self._template_back(k, f)
            for chunks, U, M in self._groups:
                self._apply_group(chunks, U, M, v, tod, loc, buf)
        return np.concatenate([(self._scatter_sum(buf).T * self.mask).ravel(), self._locbuf])

    def _chunk_tod(self, k, v, tod, loc, c2, s2, pol):
        """P v on chunk k (v given), or the TOD of chunk k (flagged samples set to 0)."""
        ip = self.ip[k]
        if v is None:
            t = np.asarray(tod[k], dtype=float)
            if self.nflag:  # the TOD of flagged samples is not used (it may be NaN)
                t = np.where(ip >= self.nobs, 0.0, t)
            return t
        t = np.empty(ip.size)
        _project(ip, self._full, c2, s2, pol, loc, t)
        for values, labels in self._tmpl[k]:
            t += values * loc[self.nflag + labels]
        return t

    def _apply_group(self, chunks, U, M, v, tod, loc, buf):
        """P^T N^-1 P on chunks with common modes: N^-1 = D^-1 - D^-1 U M U^T D^-1
        (Woodbury), M = (I + Lambda U^T D^-1 U)^-1 Lambda. First pass:
        z = sum_k U_k D_k^-1 F t_k; second pass: F^-1 D_k^-1 (F t_k - U_k M z)
        for each chunk, recomputing its FFT, so that one chunk at a time is in
        memory. Always in float64."""
        m = self._nfft[chunks[0]]

        def spectrum(k):
            pol, c2, s2 = self._pol_arrays(k)
            t = self._chunk_tod(k, v, tod, loc, c2, s2, pol)
            n = t.size
            if m > n:
                t = np.concatenate([t, np.zeros(m - n)])
            return ducc0.fft.r2c(t, nthreads=self.nthreads), n, (pol, c2, s2)

        z = np.zeros((U.shape[1], m // 2 + 1), complex)
        for j, k in enumerate(chunks):
            f, _, _ = spectrum(k)
            f *= self.weights[k]
            z += U[j][:, None] * f
        y = np.einsum("fab,bf->af", M, z)
        for j, k in enumerate(chunks):
            f, n, (pol, c2, s2) = spectrum(k)
            f -= U[j] @ y
            f *= self.weights[k]
            t = ducc0.fft.c2r(f, lastsize=m, forward=False, inorm=2, nthreads=self.nthreads)[:n]
            _backproject(self.ip[k], t, c2, s2, pol, self._edges, buf, self._locbuf)
            self._template_back(k, t)

    def _precond_flat(self, r):
        sky = self._apply_precond(r[: self._nsky].reshape(self.ncomp, self.nloc))
        z = np.concatenate([sky.ravel(), r[self._nsky :] * self._locprec])
        if self._einv is not None:
            z[self._cidx] += (self._einv @ self._csum(r[self._cidx], self._nz))[self._cpar]
        return z

    def _no_flags(self):
        if self.nflag or self.namp:
            raise ValueError("apply_A and rhs work on the pixels only: with flagged samples or templates use solve()")

    def apply_A(self, m):
        """P^T N^-1 P m for this rank's pixels of a map, (ncomp, nloc) (without
        MPI, the whole map); returns the same pixels. Collective with MPI. Not
        available with flagged samples."""
        self._no_flags()
        return self._apply(np.ravel(m)).reshape(self.ncomp, self.nloc)

    def rhs(self, tod):
        """P^T N^-1 d on this rank's pixels (ncomp, nloc), for timelines laid out
        like `pix`. Collective with MPI. Not available with flagged samples."""
        self._no_flags()
        return self._apply(tod=_chunks(tod)).reshape(self.ncomp, self.nloc)

    def solve(self, tod, x0=None, tol=1e-15, maxiter=2000, recompute_every=10, verbose=False):
        """Run PCG on `tod` (same chunk layout as `pix`).

        Stops when norm(r)^2 / norm(b)^2 < tol (same criterion as SANEPIC).
        x0: optional starting map (ncomp, nobs). Returns (map (ncomp, nobs) on
        every rank, info dict) with "iterations", "converged", "residuals"
        (norm(r)^2 / norm(b)^2 at each iteration) and "amplitudes" (the
        template amplitudes of this rank, by label; empty without templates).
        """
        b = self._apply(tod=_chunks(tod))
        bb = self._dot(b, b)
        x = np.zeros_like(b)
        if x0 is not None:
            x[: self._nsky] = (x0[:, self.lo : self.hi] * self.mask).ravel()
        r = b - self._apply(x, exact=True)
        z = self._precond_flat(r)
        d = z.copy()
        delta = self._dot(r, z)
        history = [self._dot(r, r) / bb]
        true_residual = True  # r was computed as b - A x in float64

        it = 0
        while it < maxiter and (it == 0 or history[-1] > tol):
            q = self._apply(d)
            alpha = delta / self._dot(d, q)
            x += alpha * d
            true_residual = it % recompute_every == 0
            if true_residual:
                r = b - self._apply(x, exact=True)  # true residual, avoids drift
            else:
                r -= alpha * q
            z = self._precond_flat(r)
            delta_new = self._dot(r, z)
            d = z + (delta_new / delta) * d
            delta = delta_new
            history.append(self._dot(r, r) / bb)
            it += 1
            if self._single and history[-1] <= tol and not true_residual:
                # converged in single precision: check the true residual, and
                # restart from it if it is not below tol
                r = b - self._apply(x, exact=True)
                history[-1] = self._dot(r, r) / bb
                z = self._precond_flat(r)
                d = z.copy()
                delta = self._dot(r, z)
            if verbose and (self.comm is None or self.comm.rank == 0):
                print(f"iter {it:4d}  |r|^2/|b|^2 = {history[-1]:.3e}")

        info = {"iterations": it, "converged": history[-1] <= tol, "residuals": np.array(history), "amplitudes": x[self._nsky + self.nflag :].copy()}
        del b, r, z, d  # q does not exist if maxiter == 0
        self._full = self._buf = self._locbuf = None  # free the work maps before gathering the result
        self._fftbuf = {}
        # full mask now, while all ranks are here, so that to_healpix is not collective
        self._full_mask = self.full(self.mask.astype(np.uint8)).astype(bool)
        return self._gather(x[: self._nsky].reshape(self.ncomp, self.nloc).T).T, info

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
