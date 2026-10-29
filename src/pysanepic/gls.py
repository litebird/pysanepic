"""GLS map-making with correlated noise (the SANEPIC algorithm).

Solves (P^T N^-1 P) m = P^T N^-1 d with preconditioned conjugate gradients.
N^-1 is applied in Fourier space, independently on each data chunk
(one detector x one stretch of time), as a circulant filter.

Data layout: data come as *chunks*. Pass a 2D array (nchunk, ns) for chunks of
the same length, or a list of 1D chunks and/or 2D blocks. Noise weights have
one row of length ns//2+1 per chunk, defined so that the filter is
irfft(rfft(t) * w). Chunks are used as views (no copies) and processed one at a
time (with several threads, one per thread), so the memory per sample is the
TOD plus 8 bytes (int32 pixel and float32 2a; 12 bytes with
angle_dtype=np.float64). gamma cos 2a and gamma sin 2a are recomputed for each
chunk when it is used.
"""

import os
import queue
import types
from concurrent.futures import ThreadPoolExecutor
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


@numba.njit(parallel=True, cache=True)
def _add_template(t, values, labels, amp):
    """t += values * amp[labels] (values are 0 where the template is not applied)."""
    for i in numba.prange(t.size):
        t[i] += values[i] * amp[labels[i]]


@numba.njit(nogil=True, cache=True)
def _template_dot(values, labels, f, out):
    """out[labels] += values * f: T^T f for one template of a chunk."""
    # ponytail: serial (a scatter to few amplitudes); ~ms per day of data
    for i in range(f.size):
        out[labels[i]] += values[i] * f[i]


@numba.njit(parallel=True, cache=True)
def _add_scaled(t, values, a):
    """t += values * a: a template with one amplitude in the whole chunk."""
    for i in numba.prange(t.size):
        t[i] += values[i] * a


@numba.njit(nogil=True, cache=True)
def _dot(values, f):
    s = 0.0
    for i in range(f.size):
        s += values[i] * f[i]
    return s


def _template_add(t, values, labels, amp, serial=False):
    """t += values * amp[labels], labels an array or one int (see GLS templates);
    `serial`: the single-threaded kernels of the chunk workers."""
    if np.ndim(labels) == 0:
        (_add_scaled_1 if serial else _add_scaled)(t, values, amp[labels])
    else:
        (_add_template_1 if serial else _add_template)(t, values, labels, amp)


def _template_transpose(values, labels, f, out):
    """out[labels] += values * f, labels an array or one int."""
    if np.ndim(labels) == 0:
        out[labels] += _dot(values, f)
    else:
        _template_dot(values, labels, f, out)


# cos and sin on a grid of [-pi, pi), for _cos_sin
_TRIG_N = 4096
_TRIG_COS = np.cos(-np.pi + 2 * np.pi * np.arange(_TRIG_N) / _TRIG_N)
_TRIG_SIN = np.sin(-np.pi + 2 * np.pi * np.arange(_TRIG_N) / _TRIG_N)
_UNSEEN = -1.6375e30  # healpy.UNSEEN


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


@numba.njit(parallel=True, cache=True)
def _project_iqu(m, pix, angle, gamma, tc, ts, out):
    """out = I + gamma (Q cos 2a + U sin 2a) of the maps m (ncomp, npix) in
    the pixels pix, a = angle (any value); hp.UNSEEN taken as 0. cos and sin
    as in _cos_sin."""
    n = tc.size
    step = 2 * np.pi / n
    pol = m.shape[0] == 3
    for i in numba.prange(pix.size):
        p = pix[i]
        v = m[0, p]
        t = 0.0 if v == _UNSEEN else v
        if pol:
            x = 2.0 * angle[i] + np.pi
            x -= 2 * np.pi * np.floor(x / (2 * np.pi))
            k = min(max(int(x / step), 0), n - 1)
            d = x - k * step
            d2 = d * d
            cd = 1.0 - d2 * (0.5 - d2 / 24.0)
            sd = d * (1.0 - d2 * (1.0 / 6.0 - d2 / 120.0))
            q, u = m[1, p], m[2, p]
            q = 0.0 if q == _UNSEEN else q
            u = 0.0 if u == _UNSEEN else u
            t += gamma * (q * (tc[k] * cd - ts[k] * sd) + u * (ts[k] * cd + tc[k] * sd))
        out[i] = t


@numba.njit(parallel=True, cache=True)
def _calibrated(tod, gain, period, dipole, known, out):
    """out = tod / gain[period] - dipole - known."""
    for i in numba.prange(tod.size):
        out[i] = tod[i] / gain[period[i]] - dipole[i] - known[i]


def _serial(kernel):
    """A copy of the parallel numba kernel `kernel` compiled single-threaded
    and without the GIL (prange runs as range), for the chunk workers of
    GLS._apply: each runs on its own chunk, and numba's default threading
    layer cannot run parallel kernels from several threads at once."""
    f = kernel.py_func
    g = types.FunctionType(f.__code__, f.__globals__, f.__name__ + "_1", f.__defaults__, f.__closure__)
    g.__qualname__ = f.__qualname__ + "_1"  # its own entry in numba's cache
    return numba.njit(nogil=True, cache=True)(g)


_project_1, _add_template_1, _add_scaled_1, _cos_sin_1, _backproject_1 = map(
    _serial, (_project, _add_template, _add_scaled, _cos_sin, _backproject))

# ponytail: fixed budget for the private P^T maps of the chunk workers (nside 512
# x 48 workers: 3.6 GB); above it, P^T by pixel ranges; make it a parameter if needed
_PRIVATE_MAPS_BYTES = 4e9


@numba.njit(parallel=True, cache=True)
def _reduce_maps(maps, buf):
    """buf += sum of the private maps of the chunk workers, which are zeroed."""
    for p in numba.prange(buf.shape[0]):
        for w in range(maps.shape[0]):
            for c in range(buf.shape[1]):
                buf[p, c] += maps[w, p, c]
                maps[w, p, c] = 0.0


# PCG vectors: the pixels of this rank, then the unknowns of the flagged samples
# (~20% of the samples with a Galactic mask in the calibration) and the
# amplitudes; their updates in parallel. The dot products are summed over a
# fixed number of blocks: the same result with any number of threads.
_DOT_BLOCKS = 256


@numba.njit(parallel=True, cache=True)
def _block_dots(a, b, nb):
    out = np.zeros(nb)
    n = a.size
    for j in numba.prange(nb):
        s = 0.0
        for i in range(j * n // nb, (j + 1) * n // nb):
            s += a[i] * b[i]
        out[j] = s
    return out


@numba.njit(parallel=True, cache=True)
def _axpy(y, a, x):
    """y += a x."""
    for i in numba.prange(y.size):
        y[i] += a * x[i]


@numba.njit(parallel=True, cache=True)
def _xpby(y, x, b):
    """y = x + b y."""
    for i in numba.prange(y.size):
        y[i] = x[i] + b * y[i]


@numba.njit(parallel=True, cache=True)
def _mul(out, a, b):
    """out = a b."""
    for i in numba.prange(out.size):
        out[i] = a[i] * b[i]


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
    processed in parallel (each one writes its own row of out). The unknowns
    of a chunk have contiguous labels: they are sorted with the pixels after a
    shift, so that the sort costs O(ns + nobs + unknowns of the chunk), not of
    all the unknowns of the rank (`nlabels`; 200 million in a year of data
    with a Galactic mask).
    """
    # ponytail: O(hits^2) per pixel per chunk; switch to a lag cutoff or the
    # SANEPIC N_[0] shortcut if deep pixels with long chunks become a bottleneck
    nchunk, ns = ip.shape
    nc = c.shape[1]  # FFT length >= ns: c is the first row of an nc x nc circulant
    nobs = out.shape[0]
    for k in range(nchunk):
        lo, hi = nlabels, nobs - 1  # labels of the unknowns of this chunk: [lo, hi]
        for i in range(ns):
            if ip[k, i] >= nobs:
                lo = min(lo, ip[k, i])
                hi = max(hi, ip[k, i])
        off = lo - nobs if hi >= nobs else 0  # unknown j of the chunk is sorted as nobs + j - lo
        lab = np.empty(ns, np.int64)
        for i in range(ns):
            lab[i] = ip[k, i] - off if ip[k, i] >= nobs else ip[k, i]
        order = _pixel_order(lab, hi - off + 1 if hi >= nobs else nobs)
        sorted_ip = lab[order]
        starts = np.flatnonzero(np.diff(sorted_ip)) + 1
        bounds = np.empty(starts.size + 2, np.int64)
        bounds[0] = 0
        bounds[1:-1] = starts
        bounds[-1] = ns
        for r in numba.prange(bounds.size - 1):
            start, stop = bounds[r], bounds[r + 1]
            p = sorted_ip[start]
            n = stop - start
            local = p >= nobs  # an unknown of flagged samples: no Q/U
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
                    locout[p - nobs + off] += acc
                else:
                    out[p, 0] += acc
                continue
            if local:
                acc = 0.0
                for a in range(start, stop):
                    for b in range(start, stop):
                        acc += c[k, (order[a] - order[b]) % nc]
                locout[p - nobs + off] += acc
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
        template) laid out like the chunk, or `labels` one int (one amplitude
        in the whole chunk: no array), and a the template amplitudes, local
        to this rank like the unknowns of flagged samples (labels 0, 1, ... of
        this rank; several templates may share an amplitude, e.g. one gain per
        detector and period). Not applied to flagged samples, nor to samples in
        pixels whose Q/U are not solved (their polarized signal is not
        modelled and would bias the amplitudes). solve() returns
        the amplitudes in info["amplitudes"]; their preconditioner is the exact
        diagonal of T^T N^-1 T (own noise only, as for the pixels).
    shared_templates : list of (k, values, labels), or None
        Like `templates`, but the amplitudes are shared by all the ranks
        (labels 0, 1, ... with the same meaning on every rank, e.g. one per
        detector when its chunks are on several ranks); kept on rank 0 and
        broadcast at each product (one small allreduce). solve() returns them
        in info["shared_amplitudes"] on every rank.
    template_dtype : np.float32 (default) or np.float64
        Storage of the template values: `values` of this dtype are used
        without copies (unless some of their samples are not applied, which
        are set to 0 in a copy), labels as int32 (or one int per chunk). With
        float32, 4 bytes per sample and template, the templates are rounded
        to a relative ~6e-8 (the same rounded templates in T and T^T: the
        system stays symmetric, the amplitudes are those of the rounded
        templates).
    shared_groups : list of index arrays, or None
        Constraints on the shared amplitudes: those of each group sum to zero
        (imposed by projecting the PCG vectors), e.g. the amplitudes of a sky
        template per detector, whose common part is degenerate with the map.
    overwrite_pix : bool
        Relabel the flagged samples in the int32 `pix` arrays in place instead
        of copying the chunks that have them (make_maps does, as it owns them).
    """

    def __init__(self, pix, psi, weights, pol_efficiency=None, min_pol_hits=4, min_pol_rcond=1e-2, preconditioner="block", comm=None, nthreads=None, angle_dtype=np.float32, pol_angles=None, overwrite_pix=False, fft_lengths=None, fft_dtype=np.float64, common_modes=None, coarse=None, templates=None, shared_templates=None, shared_groups=None, template_dtype=np.float32):
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
        if template_dtype not in (np.float32, np.float64):
            raise ValueError("template_dtype must be np.float32 or np.float64")
        self.template_dtype = template_dtype
        self._w32 = {}  # weights in float32, by id of the (shared) float64 arrays
        self._fftbuf = {}  # FFT work buffers, reused over chunks and iterations
        self._slots = []  # work buffers of the chunk workers (see _apply_chunks)
        self._maps = None  # their private P^T maps
        self._pool = None
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
        self._root = comm is None or comm.rank == 0
        self._shgroups = [np.asarray(g, dtype=np.int64) for g in shared_groups or []]
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
        self._cov = cov  # this rank's pixels, kept for binned

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
        self._flagprec = 1.0 / self._locprec
        self._solved = None  # the full Q/U mask, for the templates
        self.set_templates(templates, shared_templates)
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

    def set_templates(self, templates=None, shared_templates=None):
        """Set (or replace) `templates` and `shared_templates` (see the class),
        keeping the rest of the setup: pixels, preconditioner, coarse level.
        Collective with MPI. The constraints `shared_groups` are kept."""
        # local amplitudes after the unknowns of flagged samples, then (on rank 0
        # only) the shared ones
        self._tmpl, self.namp = self._parse_templates(templates)
        self._shared, nsh = self._parse_templates(shared_templates)
        comm = self.comm
        self.nshared = nsh if comm is None else max(comm.allgather(nsh))
        self._nsh_loc = self.nshared if self._root else 0  # the shared amplitudes in this rank's PCG vectors
        self._shv = np.zeros(self.nshared)  # the shared amplitudes for P, set by _apply
        if self.pol and (self.namp if comm is None else comm.allreduce(self.namp)) + self.nshared:  # collective
            if self._solved is None:
                self._solved = self.full(self.mask[1].astype(np.uint8)).astype(bool)
            for k, ip in enumerate(self.ip):
                bad = ~self._solved[np.minimum(ip, self.nobs - 1)] & (ip < self.nobs)
                if bad.any():  # copies: the arrays may be the caller's
                    for tmpl in (self._tmpl[k], self._shared[k]):
                        tmpl[:] = [(np.where(bad, 0, v).astype(v.dtype, copy=False), lab) for v, lab in tmpl]
        diag = self._template_diag(self._shared, self.nshared)
        diag = diag if comm is None else comm.allreduce(diag)
        self._locprec = np.concatenate([self._flagprec, self._inverse(self._template_diag(self._tmpl, self.namp)),
                                        self._inverse(diag)[: self._nsh_loc]])

    def _parse_templates(self, templates):
        """Per chunk, the (values, labels) of its templates (0 where not applied;
        values of template_dtype, labels int32 or one int), and the number of
        amplitudes. Copies only where needed."""
        out, n = [[] for _ in self.ip], 0
        for k, values, labels in templates or []:
            bad = self.ip[k] >= self.nobs
            if np.ndim(labels) == 0:
                labels = int(labels)
                if labels < 0:
                    continue
                n = max(n, labels + 1)
            else:
                labels = np.asarray(labels)
                bad = bad | (labels < 0)
                n = max(n, int(labels.max(initial=-1)) + 1)
            values = np.asarray(values, dtype=self.template_dtype)
            if bad.any():
                values = np.where(bad, 0, values).astype(self.template_dtype, copy=False)
                if np.ndim(labels):
                    labels = np.where(bad, 0, labels)
            if np.ndim(labels):
                labels = labels.astype(np.int32, copy=False)
            out[k].append((values, labels))
        return out, n

    def _template_diag(self, tmpls, n):
        """Diagonal of T^T N^-1 T (own noise of each chunk) for this rank's
        chunks: for the samples v of an amplitude, spanning L samples,
        v^T C v = c[0] R[0] + 2 sum_l c[l] R[l], with c the first row of the
        circulant N^-1 of the chunk and R the autocorrelation of v, computed
        with an FFT of length 2L: exact, and ~2 FFTs of the chunk in all
        instead of one per amplitude. An amplitude spanning more than a
        quarter of the FFT length m of the chunk (e.g. one per detector, the
        whole chunk) takes instead one FFT of length m: v^T C v = sum_f w_f
        |V_f|^2 / m, also exact (C is the circulant of length m cut to the
        chunk), ~4 times cheaper. The chunks in parallel, one per thread
        (see _apply_chunks)."""
        diag = np.zeros(n)
        kernels = {}  # first row of N^-1, by weights array and FFT length
        ks = [k for k, tmpl in enumerate(tmpls) if tmpl]
        for k in ks:
            key = (id(self.weights[k]), self._nfft[k])
            if key not in kernels:
                m = self._nfft[k]
                wm = np.asarray(self.weights[k], dtype=float) * (2.0 / m)  # rfft bins: twice, but 0 and m / 2
                wm[0] /= 2
                if m % 2 == 0:
                    wm[-1] /= 2
                kernels[key] = ducc0.fft.c2r(np.asarray(self.weights[k], dtype=complex), lastsize=m,
                                             forward=False, inorm=2, nthreads=self.nthreads), wm
        nw = min(self.nthreads, len(ks))
        nthreads = self.nthreads if nw <= 1 else max(1, self.nthreads // nw)

        def chunk(k):
            """(amplitude, v^T C v) of each amplitude of chunk k."""
            (c, wm), out, m_chunk = kernels[id(self.weights[k]), self._nfft[k]], [], self._nfft[k]
            for values, labels in tmpls[k]:
                if np.ndim(labels) == 0:  # one amplitude in the whole chunk: no sort
                    nz = np.flatnonzero(values)
                    if nz.size:
                        segments = [(int(labels), nz[0], nz[-1] + 1)]
                        labels = None
                    else:
                        continue
                else:
                    segments = None
                if segments is None:
                    # one pass: the samples of each amplitude, in time order (stable
                    # sort, O(n) when the labels already increase in time)
                    idx = np.flatnonzero(values != 0)
                    if idx.size == 0:  # e.g. a chunk in pixels whose Q/U are not solved
                        continue
                    idx = idx[np.argsort(labels[idx], kind="stable")]
                    lab = labels[idx]
                    cuts = np.flatnonzero(np.diff(lab)) + 1
                    segments = [(lab[lo], idx[lo], idx[hi - 1] + 1) for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, idx.size])]
                for a, i0, i1 in segments:
                    v = values[i0:i1].astype(float) if labels is None else np.where(labels[i0:i1] == a, values[i0:i1].astype(float), 0.0)
                    if 4 * v.size > m_chunk:  # v zero-extended to the FFT of the chunk (cyclic shifts: same sum)
                        f = ducc0.fft.r2c(np.concatenate([v, np.zeros(m_chunk - v.size)]), nthreads=nthreads)
                        out.append((a, np.dot(wm, f.real**2 + f.imag**2)))
                        continue
                    m = fast_fft_length(2 * v.size)
                    f = ducc0.fft.r2c(np.concatenate([v, np.zeros(m - v.size)]), nthreads=nthreads)
                    r = ducc0.fft.c2r(f * f.conj(), lastsize=m, forward=False, inorm=2, nthreads=nthreads)[: v.size]
                    out.append((a, c[0] * r[0] + 2 * np.dot(c[1 : v.size], r[1:])))
            return out

        for out in (map(chunk, ks) if nw <= 1 else self._workers(nw).map(chunk, ks)):
            for a, x in out:  # in the order of the chunks: the same sums as serial
                diag[a] += x
        return diag

    def _workers(self, nw):
        """The thread pool of the chunk workers, with at least `nw` threads."""
        if self._pool is None or self._pool._max_workers < nw:
            self._pool = ThreadPoolExecutor(nw, thread_name_prefix="pysanepic")
        return self._pool

    @staticmethod
    def _inverse(diag):
        return np.divide(1.0, diag, out=np.zeros_like(diag), where=diag > 0)

    def _project_shared(self, a):
        """Remove the mean of each group of shared amplitudes (in place)."""
        for g in self._shgroups:
            a[g] -= a[g].mean()
        return a

    def _template_back(self, k, f, loc=None, sh=None, serial=False):
        """Add T^T f of chunk k (f = N^-1 t) to the amplitudes in _locbuf and
        _shbuf (or in `loc` and `sh`, those of a chunk worker)."""
        loc = self._locbuf[self.nflag :] if loc is None else loc
        sh = self._shbuf if sh is None else sh
        for values, labels in self._tmpl[k]:
            _template_transpose(values, labels, f, loc)
        for values, labels in self._shared[k]:
            _template_transpose(values, labels, f, sh)

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
        x = np.zeros(self._nsky + self.nflag + self.namp + self._nsh_loc)
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
        d = float(_block_dots(a, b, _DOT_BLOCKS).sum())
        return d if self.comm is None else self.comm.allreduce(d)

    def _pol_arrays(self, k, slot=None):
        """(pol, gamma cos 2a, gamma sin 2a) of chunk k, computed into buffers
        that the next call overwrites (those of a chunk worker's `slot`, if given)."""
        if not self.pol:
            return False, self._dummy, self._dummy
        n = self.ip[k].size
        if slot is not None:
            c2, s2 = slot["c2"][:n], slot["s2"][:n]
            _cos_sin_1(self.angle2[k], self.gamma[k], c2, s2, _TRIG_COS, _TRIG_SIN)
            return True, c2, s2
        c2, s2 = self._c2[:n], self._s2[:n]
        with numba_threads(self.nthreads):
            _cos_sin(self.angle2[k], self.gamma[k], c2, s2, _TRIG_COS, _TRIG_SIN)
        return True, c2, s2

    def _filter(self, t, k, single, slot=None):
        """N^-1 t for chunk k: irfft(rfft(t) * w) on an FFT of length m >= t.size
        (t zero-extended, the result truncated), in float32 if `single`. The
        work buffers are allocated once (for the longest FFT) and reused: the
        result is overwritten by the next call. With a chunk worker's `slot`,
        its buffers and its share of the threads (see _apply_chunks)."""
        m, n, w = self._nfft[k], t.size, self.weights[k]
        real, cplx = (np.float32, np.complex64) if single else (np.float64, np.complex128)
        bufs, nthreads = (self._fftbuf, self.nthreads) if slot is None else (slot["fft"], slot["nthreads"])
        if real not in bufs:
            mmax = max(self._nfft)
            bufs[real] = (np.empty(mmax, real), np.empty(mmax // 2 + 1, cplx), np.empty(mmax, real))
        tin, f, out = (b[:size] for b, size in zip(bufs[real], (m, m // 2 + 1, m)))
        if m == n and not single and t.flags.c_contiguous:
            tin = t
        else:
            tin[:n] = t
            tin[n:] = 0.0
        ducc0.fft.r2c(tin, out=f, nthreads=nthreads)
        if single:
            if id(w) not in self._w32:
                self._w32[id(w)] = np.asarray(w, dtype=np.float32)
            w = self._w32[id(w)]
        f *= w
        ducc0.fft.c2r(f, lastsize=m, forward=False, inorm=2, out=out, nthreads=nthreads)
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
            if self.nshared:
                sh = self._project_shared(v[len(v) - self._nsh_loc :].copy()) if self._root else None
                self._shv = sh if self.comm is None else self.comm.bcast(sh)
        buf = self._buf
        buf[:] = 0.0
        self._locbuf[:] = 0.0
        self._shbuf = np.zeros(self.nshared)
        with numba_threads(self.nthreads):
            free = [k for k in range(len(self.ip)) if not self._grouped[k]]
            if min(self.nthreads, len(free)) > 1:
                self._apply_chunks(free, v, tod, loc, single, buf)
            else:
                for k in free:
                    pol, c2, s2 = self._pol_arrays(k)
                    t = self._chunk_tod(k, v, tod, loc, c2, s2, pol)
                    f = self._filter(t, k, single)
                    _backproject(self.ip[k], f, c2, s2, pol, self._edges, buf, self._locbuf)
                    self._template_back(k, f)
            for chunks, U, M in self._groups:
                self._apply_group(chunks, U, M, v, tod, loc, buf)
        sh = self._shbuf if self.comm is None or not self.nshared else self.comm.allreduce(self._shbuf)
        return np.concatenate([(self._scatter_sum(buf).T * self.mask).ravel(), self._locbuf,
                               self._project_shared(sh)[: self._nsh_loc]])

    def _apply_chunks(self, ks, v, tod, loc, single, buf):
        """P^T N^-1 P v (or P^T N^-1 tod) on the chunks `ks`, in parallel: one
        chunk per worker thread at a time, each worker computes N^-1 P v of
        its chunk and T^T of it in its own buffers (`slot`), with the serial
        numba kernels and single-threaded FFTs: the chunks are long and their
        FFTs memory bound, so one FFT on many threads scales poorly (2-4x on
        48 cores of G100) and one FFT per thread well (15x in float64, 25x in
        float32). P^T goes into a private map per worker if all of them fit in
        _PRIVATE_MAPS_BYTES (nside 512: 75 MB each), summed at the end;
        otherwise, in batches of one chunk per worker, all the threads add
        P^T of each chunk to the map by pixel ranges, as without workers.
        The unknowns of flagged samples belong to one chunk each: the workers
        add to them in place, without private copies (they can be ~20% of
        the samples, e.g. a Galactic mask during the calibration).
        Each worker keeps ~40 bytes per sample of the longest chunk."""
        nw = min(self.nthreads, len(ks))
        pool = self._workers(nw)
        while len(self._slots) < nw:
            nmax = max(ip.size for ip in self.ip)
            c2, s2 = (np.empty(nmax, self._c2.dtype), np.empty(nmax, self._c2.dtype)) if self.pol else (None, None)
            self._slots.append(dict(c2=c2, s2=s2, t=np.empty(nmax), fft={}))
        slots = self._slots[:nw]
        private = nw * buf.nbytes <= _PRIVATE_MAPS_BYTES
        if private and (self._maps is None or self._maps.shape[0] < nw):
            self._maps = np.zeros((nw, *buf.shape))  # zeroed again by _reduce_maps
        flag = self._locbuf[: self.nflag]  # disjoint labels for each chunk
        for j, slot in enumerate(slots):
            slot["nthreads"] = max(1, self.nthreads // nw)
            slot["loc"], slot["sh"] = np.zeros(self.namp), np.zeros(self.nshared)
            if private:
                slot["map"] = self._maps[j]
        free = queue.SimpleQueue()
        for slot in slots:
            free.put(slot)
        whole = np.array([0, self.nobs], np.int64)  # one thread: all the pixels

        def work(k):
            slot = free.get() if private else k[1]
            k = k if private else k[0]
            pol, c2, s2 = self._pol_arrays(k, slot)
            t = self._chunk_tod(k, v, tod, loc, c2, s2, pol, slot)
            f = self._filter(t, k, single, slot)
            self._template_back(k, f, slot["loc"], slot["sh"], serial=True)
            if not private:
                return pol, c2, s2, f
            _backproject_1(self.ip[k], f, c2, s2, pol, whole, slot["map"], flag)
            free.put(slot)

        if private:
            list(pool.map(work, ks))
            _reduce_maps(self._maps[:nw], buf)
        else:
            for i in range(0, len(ks), nw):
                batch = ks[i : i + nw]
                for k, (pol, c2, s2, f) in zip(batch, pool.map(work, zip(batch, slots))):
                    _backproject(self.ip[k], f, c2, s2, pol, self._edges, buf, self._locbuf)
        for slot in slots:
            self._locbuf[self.nflag :] += slot["loc"]
            self._shbuf += slot["sh"]

    def _chunk_tod(self, k, v, tod, loc, c2, s2, pol, slot=None):
        """P v on chunk k (v given), or the TOD of chunk k (flagged samples set
        to 0); with a chunk worker's `slot`, in its buffer and single-threaded."""
        ip = self.ip[k]
        if v is None:
            t = np.asarray(tod[k], dtype=float)
            if self.nflag:  # the TOD of flagged samples is not used (it may be NaN)
                t = np.where(ip >= self.nobs, 0.0, t)
            return t
        t = np.empty(ip.size) if slot is None else slot["t"][: ip.size]
        (_project if slot is None else _project_1)(ip, self._full, c2, s2, pol, loc, t)
        serial = slot is not None
        for values, labels in self._tmpl[k]:
            _template_add(t, values, labels, loc[self.nflag : self.nflag + self.namp], serial)
        for values, labels in self._shared[k]:
            _template_add(t, values, labels, self._shv, serial)
        return t

    def _apply_group(self, chunks, U, M, v, tod, loc, buf):
        """P^T N^-1 P on chunks with common modes: N^-1 = D^-1 - D^-1 U M U^T D^-1
        (Woodbury), M = (I + Lambda U^T D^-1 U)^-1 Lambda. First pass:
        z = sum_k U_k D_k^-1 F t_k; second pass: F^-1 D_k^-1 (F t_k - U_k M z)
        for each chunk, with the spectra F t_k of the first pass kept (8 bytes
        per sample of the group, while it is processed). Always in float64."""
        # ponytail: all the spectra of the group in memory; recompute them in the
        # second pass (1.5x the FFTs) if a group ever does not fit
        m = self._nfft[chunks[0]]
        z = np.zeros((U.shape[1], m // 2 + 1), complex)
        spectra = []
        for j, k in enumerate(chunks):
            pol, c2, s2 = self._pol_arrays(k)
            t = self._chunk_tod(k, v, tod, loc, c2, s2, pol)
            if m > t.size:
                t = np.concatenate([t, np.zeros(m - t.size)])
            f = ducc0.fft.r2c(t, nthreads=self.nthreads)
            spectra.append(f)
            z += U[j][:, None] * (f * self.weights[k])
        y = np.einsum("fab,bf->af", M, z)
        for j, k in enumerate(chunks):
            f, n = spectra[j], self.ip[k].size
            spectra[j] = None
            pol, c2, s2 = self._pol_arrays(k)
            f -= U[j] @ y
            f *= self.weights[k]
            t = ducc0.fft.c2r(f, lastsize=m, forward=False, inorm=2, nthreads=self.nthreads)[:n]
            _backproject(self.ip[k], t, c2, s2, pol, self._edges, buf, self._locbuf)
            self._template_back(k, t)

    def _precond_flat(self, r):
        sky = self._apply_precond(r[: self._nsky].reshape(self.ncomp, self.nloc))
        z = np.empty_like(r)
        z[: self._nsky] = sky.ravel()
        _mul(z[self._nsky :], r[self._nsky :], self._locprec)
        if self._nsh_loc:
            self._project_shared(z[len(z) - self._nsh_loc :])
        if self._einv is not None:
            z[self._cidx] += (self._einv @ self._csum(r[self._cidx], self._nz))[self._cpar]
        return z

    def _no_flags(self):
        if self.nflag or self.namp or self.nshared:
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
        template amplitudes of this rank, by label; empty without templates)
        and "shared_amplitudes" (on every rank).
        """
        b = self._apply(tod=_chunks(tod))
        x = np.zeros_like(b)
        if x0 is not None:
            x[: self._nsky] = (x0[:, self.lo : self.hi] * self.mask).ravel()
        x, it, history = self._pcg(b, x, x0 is not None, tol, maxiter, recompute_every, verbose)
        a0 = self._nsky + self.nflag
        shared = x[len(x) - self._nsh_loc :].copy() if self._root else None
        info = {"iterations": it, "converged": history[-1] <= tol, "residuals": np.array(history),
                "amplitudes": x[a0 : a0 + self.namp].copy(),
                "shared_amplitudes": shared if self.comm is None else self.comm.bcast(shared)}
        self._free()
        # full mask now, while all ranks are here, so that to_healpix is not collective
        self._full_mask = self.full(self.mask.astype(np.uint8)).astype(bool)
        return self._gather(x[: self._nsky].reshape(self.ncomp, self.nloc).T).T, info

    def shared_covariance(self, tol=1e-10, maxiter=2000, recompute_every=10, verbose=False):
        """Covariance of the shared template amplitudes, marginalized over the
        maps, the flagged samples and the local amplitudes: the shared block of
        A^-1, with A the operator of solve() (with `shared_groups`, of its
        pseudo-inverse on the constraints: the covariance of the projected
        amplitudes). One PCG solve per shared amplitude, A x = e_j, with the
        current templates. (nshared, nshared) on every rank; collective."""
        n = self._nsky + self.nflag + self.namp + self._nsh_loc
        cov = np.zeros((self.nshared, self.nshared))
        for j in range(self.nshared):
            b = np.zeros(n)
            if self._root:
                b[n - self.nshared + j] = 1.0
                self._project_shared(b[n - self.nshared :])
            x, _, _ = self._pcg(b, np.zeros(n), False, tol, maxiter, recompute_every, verbose)
            if self._root:
                cov[:, j] = x[n - self.nshared :]
        self._free()
        return cov if self.comm is None else self.comm.bcast(cov)

    def _free(self):
        """Free the work maps and FFT buffers (allocated again when needed)."""
        self._full = self._buf = self._locbuf = None
        self._fftbuf = {}
        self._slots = []
        self._maps = None

    def _pcg(self, b, x, start, tol, maxiter, recompute_every, verbose):
        """PCG on A x = b from x (`start`: x is not 0); returns (x, iterations,
        norm(r)^2 / norm(b)^2 at each iteration)."""
        with numba_threads(self.nthreads):
            return self._pcg_loop(b, x, start, tol, maxiter, recompute_every, verbose)

    def _pcg_loop(self, b, x, start, tol, maxiter, recompute_every, verbose):
        bb = self._dot(b, b)
        r = b - self._apply(x, exact=True) if start else b.copy()  # A 0 = 0: no product
        z = self._precond_flat(r)
        d = z.copy()
        delta = self._dot(r, z)
        history = [self._dot(r, r) / bb]
        true_residual = True  # r was computed as b - A x in float64

        it = 0
        while it < maxiter and (it == 0 or history[-1] > tol):
            q = self._apply(d)
            alpha = delta / self._dot(d, q)
            _axpy(x, alpha, d)
            true_residual = (it + 1) % recompute_every == 0  # after every recompute_every-th step
            if true_residual:
                r = b - self._apply(x, exact=True)  # true residual, avoids drift
            else:
                _axpy(r, -alpha, q)
            z = self._precond_flat(r)
            delta_new = self._dot(r, z)
            _xpby(d, z, delta_new / delta)
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
        return x, it, history

    def binned(self, tod):
        """Binned map (P^T P)^-1 P^T tod (white noise), (ncomp, nobs) on every
        rank, for timelines laid out like `pix`; I only where Q/U are not
        solved, 0 where not observed; flagged samples are not used. Collective
        with MPI."""
        cov = self._cov  # the coverage of the setup: the same samples
        rhs = np.zeros((self.nobs, self.ncomp))
        flag = np.zeros(self.nflag)  # not used
        with numba_threads(self.nthreads):
            for k, ip in enumerate(self.ip):
                pol, c2, s2 = self._pol_arrays(k)
                _backproject(ip, np.asarray(tod[k], dtype=float), c2, s2, pol, self._edges, rhs, flag)
        rhs = self._scatter_sum(rhs)
        out = np.zeros((self.ncomp, self.nloc))
        i_only = self.mask[0] & ~self.mask[-1] if self.pol else self.mask[0]
        out[0, i_only] = rhs[i_only, 0] / cov[i_only, 0]
        if self.pol:
            n, c, s, cc, cs, ss = cov[:, 0], *cov[:, 2:].T
            M = np.stack([n, c, s, c, cc, cs, s, cs, ss], axis=1).reshape(-1, 3, 3)
            good = self.mask[1]
            out[:, good] = np.linalg.solve(M[good], rhs[good][..., None])[..., 0].T
        return self.full(out)

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
