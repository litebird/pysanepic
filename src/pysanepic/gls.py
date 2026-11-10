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

import heapq
import os
import time
import types
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import ducc0
import numba
import numpy as np

from .node import node_of


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


class _GapFilledTOD:
    """A TOD view plus only the synthetic gap values, materialized per chunk.

    Keep no full filled copy across the solve: with sparse flags that could
    cost more memory than exact marginalization. The original TOD is untouched.
    """

    def __init__(self, tod, flags, values):
        self.tod, self.flags, self.values = tod, flags, values

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("gap-filled TOD requires a temporary copy")
        out = np.array(self.tod, dtype=float if dtype is None else dtype, copy=True)
        out[self.flags] = self.values
        return out


def _chunks(x):
    """1D chunks (views) from a 2D block (one chunk per row), a 1D chunk, or a list of them."""
    out = []
    for b in _as_list(x):
        if isinstance(b, _GapFilledTOD):
            out.append(b)
            continue
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


def _validate_act_ncoeff(rule):
    if rule is not None and not callable(rule):
        if isinstance(rule, (bool, np.bool_)) or not isinstance(rule, (int, np.integer)) or rule < 1:
            raise ValueError("act_ncoeff must be None, a positive integer, or a callable (lengths, fs) -> counts")


def _act_ncoeff(lengths, fs, rule=None):
    """Coefficient counts of Naess et al. (2025), Sect. 3.3.3.

    Durations are length / fs; nearest-integer ties round upward. Never use
    more parameters than samples (a full-rank gap is represented as identity).
    rule=None retains the ACT thresholds, an integer fixes the count, and
    a callable takes (lengths array, sampling rate) and returns integer counts.
    """
    _validate_act_ncoeff(rule)
    lengths = np.asarray(lengths, dtype=np.int64)
    if rule is None:
        # Cap in floating point before casting, also at tiny sampling rates.
        with np.errstate(over="ignore"):
            counts = np.floor(np.minimum(5.5 + lengths / fs / 4.0, lengths)).astype(np.int64)
        counts = np.select([lengths <= 1, lengths <= 3, lengths <= 6, lengths <= 20], [1, 2, 3, 4], default=counts)
    elif callable(rule):
        counts = np.asarray(rule(lengths, float(fs)))
        if counts.dtype.kind not in "iu" or np.any(counts < 1):
            raise ValueError("act_ncoeff callable must return positive integer counts")
        if counts.ndim and counts.shape != lengths.shape:
            raise ValueError("act_ncoeff callable must return a scalar or one count per gap")
    else:
        counts = min(int(rule), int(lengths.max(initial=1)))
    return np.minimum(counts, lengths).astype(np.int64)


@numba.njit(cache=True)
def _orthogonal_cut(n, ncoef):
    """Change coordinates before the sampled Legendre basis becomes ill-conditioned."""
    return 4 < ncoef < n and ncoef * ncoef > 4 * n


@numba.njit(cache=True)
def _cut_orthogonal_column(n, degree, prev, cur, out):
    """Orthonormal Gram (Hahn alpha=beta=0) polynomial on 0,...,n-1.

    Low degrees use the orthonormal three-term recurrence in degree. At high
    degrees that recurrence amplifies roundoff near the endpoints. Instead
    solve the symmetric difference equation from an endpoint to the middle,
    reflect with parity, and normalize. Its eigenvalue is degree*(degree+1)
    and edge coefficients are a_i=(i+1)*(n-1-i). Rescaling prevents overflow.
    See https://dlmf.nist.gov/18.22.ii. No design matrix is stored.
    """
    if degree == 0:
        out[:] = 1.0 / np.sqrt(n)
    elif degree * degree <= 4 * n:
        b = degree / (n - 1) * np.sqrt((n * n - degree * degree) / (4 * degree * degree - 1))
        j = degree - 1
        a = j / (n - 1) * np.sqrt((n * n - j * j) / (4 * j * j - 1)) if j else 0.0
        for i in range(n):
            out[i] = ((2.0 * i / (n - 1) - 1.0) * cur[i] - a * prev[i]) / b
    else:
        out[0] = 1.0
        for i in range((n - 1) // 2):
            a, b = (i + 1) * (n - 1 - i), i * (n - i)
            out[i + 1] = ((a + b - degree * (degree + 1)) * out[i]
                          - (b * out[i - 1] if i else 0.0)) / a
            if abs(out[i + 1]) > 1e100:
                out[:i + 2] *= 1e-100
        parity = -1.0 if degree % 2 else 1.0
        for i in range(n // 2):
            out[n - 1 - i] = parity * out[i]
        if n % 2 and degree % 2:
            out[n // 2] = 0.0
        norm2 = 0.0
        for i in range(n):
            norm2 += out[i] * out[i]
        norm = np.sqrt(norm2)
        out[:] /= norm


@numba.njit(nogil=True, cache=True)
def _cut_project(cuts, amp, tod):
    """Replace each cut by its polynomial series, without a stored matrix."""
    for start, stop, first, ncoef in cuts:
        n = stop - start
        if _orthogonal_cut(n, ncoef):
            prev, cur, column = np.zeros(n), np.zeros(n), np.empty(n)
            tod[start:stop] = 0.0
            for j in range(ncoef):
                _cut_orthogonal_column(n, j, prev, cur, column)
                for i in range(n):
                    tod[start + i] += amp[first + j] * column[i]
                prev, cur, column = cur, column, prev
            continue
        for i in range(n):
            if ncoef == n:  # same full space as marginalization, stable basis
                tod[start + i] = amp[first + i]
                continue
            x = 2.0 * i / (n - 1) - 1.0
            prev, cur = 1.0, x
            value = amp[first]
            for j in range(1, ncoef):
                value += cur * amp[first + j]
                prev, cur = cur, ((2 * j + 1) * x * cur - j * prev) / (j + 1)
            tod[start + i] = value


@numba.njit(nogil=True, cache=True)
def _cut_transpose(cuts, tod, out):
    """Add C^T tod to the local cut coefficients (disjoint across chunks)."""
    for start, stop, first, ncoef in cuts:
        n = stop - start
        if _orthogonal_cut(n, ncoef):
            prev, cur, column = np.zeros(n), np.zeros(n), np.empty(n)
            for j in range(ncoef):
                _cut_orthogonal_column(n, j, prev, cur, column)
                for i in range(n):
                    out[first + j] += column[i] * tod[start + i]
                prev, cur, column = cur, column, prev
            continue
        for i in range(n):
            value = tod[start + i]
            if ncoef == n:
                out[first + i] += value
                continue
            x = 2.0 * i / (n - 1) - 1.0
            prev, cur = 1.0, x
            out[first] += value
            for j in range(1, ncoef):
                out[first + j] += cur * value
                prev, cur = cur, ((2 * j + 1) * x * cur - j * prev) / (j + 1)


@numba.njit(nogil=True, cache=True)
def _cut_diagonal_small(cuts, c, out):
    """Exact diagonal C^T N^-1 C for short cuts, with small direct sums."""
    for start, stop, first, ncoef in cuts:
        n = stop - start
        if ncoef == n:
            out[first:first + ncoef] = c[0]
            continue
        if n > 32:
            continue
        basis = np.empty((n, ncoef))
        if _orthogonal_cut(n, ncoef):
            prev, cur = np.zeros(n), np.zeros(n)
            for j in range(ncoef):
                _cut_orthogonal_column(n, j, prev, cur, basis[:, j])
                prev, cur = cur, basis[:, j]
        else:
            for i in range(n):
                x = 2.0 * i / (n - 1) - 1.0
                basis[i, 0] = 1.0
                basis[i, 1] = x
                for j in range(1, ncoef - 1):
                    basis[i, j + 1] = ((2 * j + 1) * x * basis[i, j] - j * basis[i, j - 1]) / (j + 1)
        for j in range(ncoef):
            value = 0.0
            for a in range(n):
                for b in range(n):
                    value += basis[a, j] * c[(a - b) % c.size] * basis[b, j]
            out[first + j] = value


def _cut_diagonal(cuts, c, out, nthreads, dc_weight):
    """Own-noise cut diagonal; long cuts use FFT autocorrelations per column.

    At most three polynomial vectors of the gap length are kept, rather than
    a samples x coefficients matrix. c is the circulant filter kernel.
    """
    _cut_diagonal_small(cuts, c, out)
    for start, stop, first, ncoef in cuts:
        n = stop - start
        orthogonal = _orthogonal_cut(n, ncoef)
        if n == c.size and ncoef < n:
            # A constant over a whole unextended chunk is its DC mode.
            # Use the exact weight, avoiding cancellation to +/- roundoff
            # when fmin=0 and this direction is a null mode.
            out[first] = dc_weight if orthogonal else n * dc_weight
        if n <= 32 or ncoef == n:
            continue
        x = None if orthogonal else np.linspace(-1.0, 1.0, n)
        prev, cur = np.zeros(n), np.ones(n)
        column = np.empty(n) if orthogonal else None
        m = fast_fft_length(2 * n)
        for j in range(ncoef):
            if orthogonal:
                _cut_orthogonal_column(n, j, prev, cur, column)
                prev, cur, column = cur, column, prev
            if j != 0 or n != c.size:
                f = ducc0.fft.r2c(np.concatenate([cur, np.zeros(m - n)]), nthreads=nthreads)
                r = ducc0.fft.c2r(f * f.conj(), lastsize=m, forward=False, inorm=2, nthreads=nthreads)[:n]
                out[first + j] = c[0] * r[0] + 2 * np.dot(c[1:n], r[1:])
            if not orthogonal:
                prev, cur = cur, ((2 * j + 1) * x * cur - j * prev) / (j + 1)


# <x^2> = <y^2> over a uniformly covered square pixel of side 1: the quadratic
# sub-pixel weights are centred on it (a well conditioned basis; make_maps
# recentres I on the geometric moments of each HEALPix pixel)
_Q2 = 1.0 / 12.0


@numba.njit(parallel=True, cache=True)
def _project(ip, m, c2, s2, pol, gx, gy, grad, loc, out):
    """out[i] = I[p] + c2[i] Q[p] + s2[i] U[p] (+ the sub-pixel terms of
    order `grad`, see _weights, with the columns of m after I/Q/U), p = ip[i];
    m is (nobs, ncomp). Labels p >= nobs are local unknowns; p < 0 has no
    pixel response (an ACT polynomial, if any, is applied separately)."""
    nobs = m.shape[0]
    g = 3 if pol else 1
    for i in numba.prange(ip.size):
        p = ip[i]
        if p < 0:
            out[i] = 0.0
            continue
        if p >= nobs:
            out[i] = loc[p - nobs]
            continue
        v = m[p, 0]
        if pol:
            v += m[p, 1] * c2[i] + m[p, 2] * s2[i]
        if grad:
            x, y = gx[i], gy[i]
            v += m[p, g] * x + m[p, g + 1] * y
            if grad > 1:
                v += m[p, g + 2] * (x * x - _Q2) + m[p, g + 3] * x * y + m[p, g + 4] * (y * y - _Q2)
        out[i] = v


@numba.njit(parallel=True, cache=True)
def _backproject(ip, t, c2, s2, pol, gx, gy, grad, edges, buf, locbuf):
    """buf += P^T t, buf (nobs, ncomp). Thread th owns the pixels
    [edges[th], edges[th + 1]): it reads all the pixel indices of the chunk but
    adds only the samples of its pixels, so the threads share one map without
    races and without private copies. Flagged samples (labels p >= nobs) go to
    locbuf[p - nobs], written by thread 0. The body is written out (also in
    _backproject_sel): a per-sample helper inlined by numba made the serial
    copy (_backproject_1, the P^T of the chunk workers) 5x slower."""
    nobs = buf.shape[0]
    g = 3 if pol else 1
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
                if grad:
                    x, y = gx[i], gy[i]
                    buf[p, g] += t[i] * x
                    buf[p, g + 1] += t[i] * y
                    if grad > 1:
                        buf[p, g + 2] += t[i] * (x * x - _Q2)
                        buf[p, g + 3] += t[i] * x * y
                        buf[p, g + 4] += t[i] * (y * y - _Q2)


@numba.njit(parallel=True, cache=True)
def _backproject_sel(ip, t, c2, s2, pol, gx, gy, grad, idx, edges, buf, locbuf):
    """_backproject on the samples idx only (see _slab_order); flagged samples
    go to locbuf only if it is not empty."""
    nobs = buf.shape[0]
    g = 3 if pol else 1
    for th in numba.prange(edges.size - 1):
        lo, hi = edges[th], edges[th + 1]
        for i in idx:
            p = ip[i]
            if p >= nobs:
                if th == 0 and locbuf.size:
                    locbuf[p - nobs] += t[i]
            elif lo <= p < hi:
                buf[p, 0] += t[i]
                if pol:
                    buf[p, 1] += t[i] * c2[i]
                    buf[p, 2] += t[i] * s2[i]
                if grad:
                    x, y = gx[i], gy[i]
                    buf[p, g] += t[i] * x
                    buf[p, g + 1] += t[i] * y
                    if grad > 1:
                        buf[p, g + 2] += t[i] * (x * x - _Q2)
                        buf[p, g + 3] += t[i] * x * y
                        buf[p, g + 4] += t[i] * (y * y - _Q2)


def _slab_order(ip, nobs, nslab):
    """The samples of a chunk grouped by slab of pixels: slab s holds the
    pixels p with p nslab // nobs = s, i.e. [ceil(s nobs / nslab), ceil((s + 1)
    nobs / nslab)); then the flagged samples (p >= nobs). Returns (order,
    starts): the samples of slab s are order[starts[s]:starts[s + 1]], in time
    order; the samples with p < 0 are left out. A counting sort in parallel
    (it runs before every P^T of a chunk with a map per node): each thread
    counts the slabs of a block of consecutive samples, then writes them
    after those of the previous blocks, so the order stays the time order."""
    nb = max(1, min(numba.get_num_threads(), ip.size // 65536))  # blocks of samples, one per thread
    return _slab_order_blocks(ip, nobs, nslab, nb)


@numba.njit(parallel=True, cache=True)
def _slab_order_blocks(ip, nobs, nslab, nb):
    """_slab_order with nb blocks of samples."""
    ns = ip.size
    counts = np.zeros((nb, nslab + 1), np.int64)
    for b in numba.prange(nb):
        for i in range(b * ns // nb, (b + 1) * ns // nb):
            p = ip[i]
            if p >= 0:
                counts[b, nslab if p >= nobs else np.int64(p) * nslab // nobs] += 1
    starts = np.zeros(nslab + 2, np.int64)
    pos = np.empty((nb, nslab + 1), np.int64)  # where block b writes its first sample of slab s
    for s in range(nslab + 1):
        n = starts[s]
        for b in range(nb):
            pos[b, s] = n
            n += counts[b, s]
        starts[s + 1] = n
    order = np.empty(starts[-1], np.int64)
    for b in numba.prange(nb):
        mine = pos[b].copy()
        for i in range(b * ns // nb, (b + 1) * ns // nb):
            p = ip[i]
            if p >= 0:
                s = nslab if p >= nobs else np.int64(p) * nslab // nobs
                order[mine[s]] = i
                mine[s] += 1
    return order, starts


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


@numba.njit(parallel=True, cache=True)
def _add_columns(t, values, a):
    """t += values @ a: several templates (columns) with one amplitude each in
    the chunk, in one pass."""
    for i in numba.prange(t.size):
        s = 0.0
        for j in range(a.size):
            s += values[i, j] * a[j]
        t[i] += s


@numba.njit(nogil=True, cache=True)
def _dot_columns(values, f):
    """values^T f, (ncol,): the T^T of several templates in one pass."""
    out = np.zeros(values.shape[1])
    for i in range(f.size):
        fi = f[i]
        for j in range(out.size):
            out[j] += values[i, j] * fi
    return out


@numba.njit(parallel=True, cache=True)
def _fill_columns(out, cols, a, m):
    """out[m + i, j] = cols[j][a + i] for i < out.shape[0] - 2 m: the
    templates of a chunk as the columns of one array (margins untouched)."""
    for i in numba.prange(out.shape[0] - 2 * m):
        for j in range(len(cols)):
            out[m + i, j] = cols[j][a + i]


def _template_add(t, values, labels, amp, serial=False):
    """t += values * amp[labels], labels an array or one int (see GLS templates),
    or values (n, ncol) and labels (ncol,); `serial`: the single-threaded
    kernels of the chunk workers."""
    if np.ndim(values) == 2:
        (_add_columns_1 if serial else _add_columns)(t, values, amp[labels])
    elif np.ndim(labels) == 0:
        (_add_scaled_1 if serial else _add_scaled)(t, values, amp[labels])
    else:
        (_add_template_1 if serial else _add_template)(t, values, labels, amp)


def _template_transpose(values, labels, f, out):
    """out[labels] += values * f, labels an array or one int, or values
    (n, ncol) and labels (ncol,)."""
    if np.ndim(values) == 2:
        np.add.at(out, labels, _dot_columns(values, f))
    elif np.ndim(labels) == 0:
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
def _add_pol(tod, q, u, angle, gamma, tc, ts):
    """tod += gamma (q cos 2a + u sin 2a), a = angle (any value): the
    polarized part of a sky model sampled at the pointing, in one pass. cos and
    sin as in _cos_sin."""
    n = tc.size
    step = 2 * np.pi / n
    for i in numba.prange(tod.size):
        x = 2.0 * angle[i] + np.pi
        x -= 2 * np.pi * np.floor(x / (2 * np.pi))
        k = min(max(int(x / step), 0), n - 1)
        d = x - k * step
        d2 = d * d
        cd = 1.0 - d2 * (0.5 - d2 / 24.0)
        sd = d * (1.0 - d2 * (1.0 / 6.0 - d2 / 120.0))
        tod[i] += gamma * (q[i] * (tc[k] * cd - ts[k] * sd) + u[i] * (ts[k] * cd + tc[k] * sd))


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


_project_1, _add_template_1, _add_scaled_1, _add_columns_1, _cos_sin_1, _backproject_1 = map(
    _serial, (_project, _add_template, _add_scaled, _add_columns, _cos_sin, _backproject))

# ponytail: fixed budget for the private P^T maps of the chunk workers (nside 512
# x 48 workers: 3.6 GB); above it, P^T by pixel ranges; make it a parameter if needed
_PRIVATE_MAPS_BYTES = 4e9
_ROUND_BYTES = 2**29  # a map per node, one thread: the chunks of a set of rounds of P^T (see GLS._round_groups)

# Coverage accumulators are larger than I/Q/U maps (7 float64 columns).
# Keep at least one, even when a single full-sky accumulator exceeds the budget.
_COVERAGE_BYTES = 512 * 1024**2


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


@numba.njit(nogil=True, cache=True)
def _weights(i, c2, s2, pol, gx, gy, grad, w):
    """w[:n] = the pointing weights (1, c2, s2, gx, gy, gx^2 - 1/12, gx gy,
    gy^2 - 1/12) of sample i (c2, s2 only with `pol`, gx, gy with `grad` >= 1,
    the quadratic ones with `grad` = 2); returns n."""
    w[0] = 1.0
    n = 1
    if pol:
        w[1] = c2[i]
        w[2] = s2[i]
        n = 3
    if grad:
        x, y = gx[i], gy[i]
        w[n] = x
        w[n + 1] = y
        n += 2
        if grad > 1:
            w[n] = x * x - _Q2
            w[n + 1] = x * y
            w[n + 2] = y * y - _Q2
            n += 3
    return n


@numba.njit(nogil=True, cache=True)
def _coverage(ip, c2, s2, pol, gx, gy, grad, out, scale=1.0):
    """Per pixel: hits, polarized hits (gamma > 0.5, as SANEPIC) and the sums of
    w_a w_b, a <= b, (a, b) != (0, 0), row by row, with w the pointing
    weights (see _weights): with pol only, c2, s2, c2^2, c2 s2, s2^2 (the
    angle-coverage matrix), each sample counted `scale` times (the weight of
    the chunk in a weighted binned map). Serial: the chunks are summed in
    parallel into private arrays (GLS._chunk_coverage). The body is written
    out (also in _coverage_sel): a per-sample helper inlined by numba made
    it 1.7x slower."""
    w = np.empty(8)
    for i in range(ip.size):
        p = ip[i]
        if p < 0 or p >= out.shape[0]:  # flagged sample
            continue
        out[p, 0] += scale
        n = _weights(i, c2, s2, pol, gx, gy, grad, w)
        if pol and w[1] * w[1] + w[2] * w[2] > 0.25:
            out[p, 1] += scale
        col = 2
        for a in range(n):
            for b in range(max(a, 1), n):
                out[p, col] += scale * w[a] * w[b]
                col += 1


@numba.njit(parallel=True, cache=True)
def _coverage_sel(ip, c2, s2, pol, gx, gy, grad, idx, edges, out, scale):
    """_coverage of the samples idx (see _slab_order), thread th on the pixels
    [edges[th], edges[th + 1])."""
    for th in numba.prange(edges.size - 1):
        lo, hi = edges[th], edges[th + 1]
        w = np.empty(8)
        for i in idx:
            p = ip[i]
            if lo <= p < hi:
                out[p, 0] += scale
                n = _weights(i, c2, s2, pol, gx, gy, grad, w)
                if pol and w[1] * w[1] + w[2] * w[2] > 0.25:
                    out[p, 1] += scale
                col = 2
                for a in range(n):
                    for b in range(max(a, 1), n):
                        out[p, col] += scale * w[a] * w[b]
                        col += 1


def _pixel_order(ip, npix):
    """Indices of the samples sorted by pixel, in time order within each pixel:
    a stable counting sort, O(ns + npix), in parallel (see
    _pixel_order_blocks). Negative labels go in a last bucket."""
    nb = max(1, min(numba.get_num_threads(), ip.size // 65536))
    return _pixel_order_blocks(ip, npix, nb)


@numba.njit(parallel=True, cache=True)
def _pixel_order_blocks(ip, npix, nb):
    """_pixel_order with nb threads, in two stable passes: the samples by
    range of labels (nb ranges; each thread counts, then writes, a block of
    consecutive samples, as _slab_order), then each range by label (one
    thread per range, with the counts of its labels only: O(npix) memory in
    all, as serial)."""
    ns, nl = ip.size, npix + 1  # labels 0 ... npix (npix: the negative ones)
    counts = np.zeros((nb, nb), np.int64)  # block, range
    for b in numba.prange(nb):
        for i in range(b * ns // nb, (b + 1) * ns // nb):
            q = ip[i] if ip[i] >= 0 else npix
            counts[b, np.int64(q) * nb // nl] += 1
    rstart = np.zeros(nb + 1, np.int64)
    pos = np.empty((nb, nb), np.int64)
    for r in range(nb):
        n = rstart[r]
        for b in range(nb):
            pos[b, r] = n
            n += counts[b, r]
        rstart[r + 1] = n
    tmp = np.empty(ns, np.int64)
    for b in numba.prange(nb):
        mine = pos[b].copy()
        for i in range(b * ns // nb, (b + 1) * ns // nb):
            q = ip[i] if ip[i] >= 0 else npix
            r = np.int64(q) * nb // nl
            tmp[mine[r]] = i
            mine[r] += 1
    order = np.empty(ns, np.int64)
    for r in numba.prange(nb):
        lo, hi = (r * nl + nb - 1) // nb, ((r + 1) * nl + nb - 1) // nb  # the labels of range r
        start = np.zeros(hi - lo + 1, np.int64)
        for a in range(rstart[r], rstart[r + 1]):
            i = tmp[a]
            q = ip[i] if ip[i] >= 0 else npix
            start[q - lo + 1] += 1
        start[0] = rstart[r]
        for j in range(hi - lo):
            start[j + 1] += start[j]
        for a in range(rstart[r], rstart[r + 1]):
            i = tmp[a]
            q = ip[i] if ip[i] >= 0 else npix
            order[start[q - lo]] = i
            start[q - lo] += 1
    return order


@numba.njit(parallel=True, cache=True)
def _sym_rcond(M, out):
    """out[i] = 1 / cond(M[i]) in the 2-norm (min |eigenvalue| / max, 0 if
    M[i] = 0) of the small symmetric matrices M (n, k, k), as
    np.linalg.cond for symmetric matrices, in parallel: cyclic Jacobi
    rotations (accurate to the rounding of the largest eigenvalue)."""
    k = M.shape[1]
    for i in numba.prange(M.shape[0]):
        A = M[i].copy()
        for _ in range(100):
            off = 0.0
            for p in range(k):
                for q in range(p + 1, k):
                    off += A[p, q] * A[p, q]
            scale = 0.0
            for p in range(k):
                scale += A[p, p] * A[p, p]
            if off <= 1e-34 * scale or off == 0.0:
                break
            for p in range(k):
                for q in range(p + 1, k):
                    if A[p, q] == 0.0:
                        continue
                    theta = (A[q, q] - A[p, p]) / (2.0 * A[p, q])
                    t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + np.sqrt(theta * theta + 1.0))
                    c = 1.0 / np.sqrt(t * t + 1.0)
                    sn = t * c
                    for r in range(k):  # A <- J^T A J
                        arp, arq = A[r, p], A[r, q]
                        A[r, p], A[r, q] = c * arp - sn * arq, sn * arp + c * arq
                    for r in range(k):
                        apr, aqr = A[p, r], A[q, r]
                        A[p, r], A[q, r] = c * apr - sn * aqr, sn * apr + c * aqr
        lmin, lmax = np.inf, 0.0
        for p in range(k):
            a = abs(A[p, p])
            lmin, lmax = min(lmin, a), max(lmax, a)
        out[i] = lmin / lmax if lmax > 0 else 0.0


@numba.njit(parallel=True, cache=True)
def _det_inv(M, det, inv):
    """det[i] and inv[i] (det 0 and inv 0 where singular) of the small
    matrices M (n, k, k), as np.linalg.det and np.linalg.inv, in parallel:
    Gauss-Jordan elimination with partial pivoting."""
    k = M.shape[1]
    for i in numba.prange(M.shape[0]):
        A = M[i].copy()
        B = np.eye(k)
        d = 1.0
        for j in range(k):
            piv = j
            for r in range(j + 1, k):
                if abs(A[r, j]) > abs(A[piv, j]):
                    piv = r
            if A[piv, j] == 0.0:
                d = 0.0
                break
            if piv != j:
                d = -d
                for c in range(k):
                    A[j, c], A[piv, c] = A[piv, c], A[j, c]
                    B[j, c], B[piv, c] = B[piv, c], B[j, c]
            a = A[j, j]
            d *= a
            for c in range(k):
                A[j, c] /= a
                B[j, c] /= a
            for r in range(k):
                if r != j and A[r, j] != 0.0:
                    f = A[r, j]
                    for c in range(k):
                        A[r, c] -= f * A[j, c]
                        B[r, c] -= f * B[j, c]
        det[i] = d
        if d == 0.0:
            inv[i] = 0.0
        else:
            inv[i] = B


def _symmetric(tri, n):
    """(m, n, n) symmetric matrices from their upper triangles, row by row (m, n (n + 1) / 2)."""
    iu = np.triu_indices(n)
    M = np.empty((tri.shape[0], n, n))
    M[:, iu[0], iu[1]] = tri
    M[:, iu[1], iu[0]] = tri
    return M


def _block_circulant(ip, c, c2, s2, pol, gx, gy, grad, out, nlabels, locout):
    """Exact per-pixel blocks of P^T C P, C circulant per chunk.

    For each pixel, sums c[(t - t') mod ns] w_t w_t'^T over all sample pairs
    that fall in it, with w the pointing weights (1, c2, s2, gx, gy, ...) (see
    _weights), and adds them to out (nobs, ncomp (ncomp + 1) / 2), the upper
    triangle row by row: II, IQ, IU, QQ, QU, UU with pol only. Labels are
    < nlabels; those >= nobs (unknowns of flagged samples, intensity only) add
    their II block to locout[label - nobs].
    Cost is sum over pixels of hits^2 per chunk; the pixels of a chunk are
    processed in parallel (each one writes its own row of out). The unknowns
    of a chunk have contiguous labels: they are sorted with the pixels after a
    shift, so that the sort costs O(ns + nobs + unknowns of the chunk), not of
    all the unknowns of the rank (`nlabels`; 200 million in a year of data
    with a Galactic mask).
    """
    for k in range(ip.shape[0]):
        _block_add(*_block_sort(ip[k], out.shape[0], nlabels), c[k], c2[k], s2[k], pol, gx[k], gy[k], grad, out, locout,
                   0, out.shape[0], True)


def _block_sort(ip, nobs, nlabels):
    """The samples of a chunk sorted by label for _block_add: (order, sorted
    labels, bounds of the runs of equal labels, shift of the unknowns)."""
    nb = max(1, min(numba.get_num_threads(), ip.size // 65536))
    return _block_sort_blocks(ip, nobs, nlabels, nb)


@numba.njit(parallel=True, cache=True)
def _block_sort_blocks(ip, nobs, nlabels, nb):
    """_block_sort with nb threads."""
    ns = ip.size
    los, his = np.full(nb, nlabels, np.int64), np.full(nb, nobs - 1, np.int64)
    for b in numba.prange(nb):  # labels of the unknowns of this chunk: [lo, hi]
        for i in range(b * ns // nb, (b + 1) * ns // nb):
            if ip[i] >= nobs:
                los[b] = min(los[b], ip[i])
                his[b] = max(his[b], ip[i])
    lo, hi = los.min(), his.max()
    off = lo - nobs if hi >= nobs else 0  # unknown j of the chunk is sorted as nobs + j - lo
    lab = np.empty(ns, np.int64)
    for i in numba.prange(ns):
        lab[i] = ip[i] - off if ip[i] >= nobs else ip[i]
    order = _pixel_order_blocks(lab, hi - off + 1 if hi >= nobs else nobs, nb)
    sorted_ip = np.empty(ns, np.int64)
    for i in numba.prange(ns):
        sorted_ip[i] = lab[order[i]]
    starts = np.flatnonzero(np.diff(sorted_ip)) + 1
    bounds = np.empty(starts.size + 2, np.int64)
    bounds[0] = 0
    bounds[1:-1] = starts
    bounds[-1] = ns
    return order, sorted_ip, bounds, off


@numba.njit(parallel=True, cache=True)
def _block_add(order, sorted_ip, bounds, off, c, c2, s2, pol, gx, gy, grad, out, locout, plo, phi, local_too):
    """The blocks of one chunk (sorted by _block_sort; c, c2, s2, gx, gy of the
    chunk) for the pixels [plo, phi) only, and for its unknowns if
    `local_too` (see _block_circulant)."""
    # ponytail: O(hits^2) per pixel per chunk; switch to a lag cutoff or the
    # SANEPIC N_[0] shortcut if deep pixels with long chunks become a bottleneck
    nc = c.size  # FFT length >= ns: c is the first row of an nc x nc circulant
    nobs = out.shape[0]
    for r in numba.prange(bounds.size - 1):
        start, stop = bounds[r], bounds[r + 1]
        p = sorted_ip[start]
        if p < 0:  # excluded samples have neither map nor local parameters
            continue
        local = p >= nobs  # an unknown of flagged samples: no Q/U
        if (local and not local_too) or (not local and (p < plo or p >= phi)):
            continue
        n = stop - start
        # Contiguous run of samples with no polarization nor offsets (e.g. the
        # padding of a chunk): sum over lags in O(n) instead of O(n^2); II
        # only (with grad = 2 the quadratic weights are -1/12 there, but such
        # pixels have no gradient, hence no Hessian: their blocks are unused)
        unpol = True
        if (pol or grad) and not local:
            for a in range(start, stop):
                t = order[a]
                if (pol and (c2[t] != 0.0 or s2[t] != 0.0)) or (grad and (gx[t] != 0.0 or gy[t] != 0.0)):
                    unpol = False
                    break
        if unpol and order[stop - 1] - order[start] == n - 1:
            acc = 0.0
            for lag in range(-(n - 1), n):
                acc += (n - abs(lag)) * c[lag % nc]
            if local:
                locout[p - nobs + off] += acc
            else:
                out[p, 0] += acc
            continue
        if local:
            acc = 0.0
            for a in range(start, stop):
                for b in range(start, stop):
                    acc += c[(order[a] - order[b]) % nc]
            locout[p - nobs + off] += acc
            continue
        # B_jl = sum_a w_j(a) S_l(a), S_l(a) = sum_b c[a - b] w_l(b):
        # O(hits^2 ncomp) instead of O(hits^2 ncomp^2)
        W = np.empty((n, 8))
        nw = 1
        for a in range(n):
            nw = _weights(order[start + a], c2, s2, pol, gx, gy, grad, W[a])
        S = np.empty(8)
        blk = np.zeros(out.shape[1])
        for a in range(n):
            ta = order[start + a]
            S[:] = 0.0
            for b in range(n):
                cab = c[(ta - order[start + b]) % nc]
                for l in range(nw):
                    S[l] += cab * W[b, l]
            col = 0
            for j in range(nw):
                for l in range(j, nw):
                    blk[col] += W[a, j] * S[l]
                    col += 1
        for col in range(blk.size):
            out[p, col] += blk[col]


class _Timings(dict):
    """Seconds of the phases of a step, by name (of this rank, waits at the
    collectives included), for verbose logs: `mark(name)` adds the time
    since the previous mark, `phase(name)` the time of a block (also inside
    marked ones: names "of which ..." are parts of others)."""

    def __init__(self):
        super().__init__()
        self._last = time.perf_counter()

    def mark(self, name):
        t = time.perf_counter()
        self[name] = self.get(name, 0.0) + t - self._last
        self._last = t

    @contextmanager
    def phase(self, name):
        t = time.perf_counter()
        try:
            yield
        finally:
            self[name] = self.get(name, 0.0) + time.perf_counter() - t

    def __str__(self):
        return ", ".join(f"{k} {v:.1f} s" for k, v in self.items())


def _timing_table(steps):
    """The lines of a verbose table of timings: steps [(name, seconds,
    _Timings or None)], each step then its phases indented, and last the
    parts "of which ..." (inside one or more of those phases), seconds
    aligned."""
    rows = []
    for name, seconds, phases in steps:
        rows.append((1, name, seconds))
        items = list((phases or {}).items())
        rows += [(2, k, v) for k, v in items if "of which" not in k] + [(2, k, v) for k, v in items if "of which" in k]
    width = max(2 * level + len(name) for level, name, _ in rows)
    return [f"{'  ' * level}{name:<{width - 2 * level}} {seconds:8.1f} s" for level, name, seconds in rows]


class _NodeSum:
    """Sums over the ranks of per-pixel contributions of their chunks (P^T, the
    coverage, the blocks of the preconditioner) in one full map per node,
    shared by its n ranks (see node.py), instead of one per rank.

    The ranks add each chunk in n rounds: in round t, rank r adds the samples
    of the chunk in the slab (r + t) mod n of the pixels, so two ranks never
    write the same pixels at the same time (the threads of a rank split the
    slab, as in _backproject); a barrier of the node closes each round. Every
    rank does the rounds of the rank of its node with the most chunks
    (`nunits`, the number of chunks this rank adds), its own chunks first.
    Then the node leaders sum the nodes and each rank takes its pixels
    (`local`). The order of the sums is fixed: the result does not depend on
    the timing of the ranks. Collective; the creation zeroes the map."""

    def __init__(self, gls, ncols, nunits):
        from mpi4py import MPI

        node = gls._node
        self.gls, self.node, self.n, self.r = gls, node, node.node.size, node.node.rank
        self.rounds, self.done = node.node.allreduce(nunits, op=MPI.MAX), 0
        nobs = gls.nobs
        self.slabs = (np.arange(self.n + 1, dtype=np.int64) * nobs + self.n - 1) // self.n  # see _slab_order
        node.barrier()  # nobody reads the previous sums any more
        self.map = node.work("sum", nobs * ncols).reshape(nobs, ncols)
        self.map[self.slabs[self.r] : self.slabs[self.r + 1]] = 0.0
        node.barrier()
        nt = max(1, min(gls.nthreads, numba.config.NUMBA_NUM_THREADS))
        self.edges = [a + (b - a) * np.arange(nt + 1, dtype=np.int64) // nt for a, b in zip(self.slabs[:-1], self.slabs[1:])]

    def add(self, fn):
        """The rounds of one chunk: fn(s, first) adds its samples in slab s (and
        its local unknowns if `first`)."""
        tm = self.gls._tm
        for t in range(self.n):
            s = (self.r + t) % self.n
            fn(s, t == 0)
            with tm.phase("of which node rounds: barriers"):
                self.node.barrier()
        self.done += 1

    def backproject(self, items, locbuf):
        """P^T t of a group of chunks, items (ip, t, c2, s2, pol, gx, gy, grad),
        in one set of rounds (one barrier per round for the whole group: the
        ranks wait for each other once per group, not once per chunk); their
        flagged samples into locbuf (of this rank)."""
        with self.gls._tm.phase("of which slab order"):
            orders = [_slab_order(it[0], self.gls.nobs, self.n) for it in items]

        def fn(s, first):
            for (ip, t, c2, s2, pol, gx, gy, grad), (order, starts) in zip(items, orders):
                _backproject_sel(ip, t, c2, s2, pol, gx, gy, grad, order[starts[s] : starts[s + 1]], self.edges[s],
                                 self.map, locbuf[:0])
                if first and starts[self.n] < order.size:  # the flagged samples
                    _backproject_sel(ip, t, c2, s2, pol, gx, gy, grad, order[starts[self.n] :], self.edges[s],
                                     self.map, locbuf)

        self.add(fn)

    def coverage(self, ip, c2, s2, pol, gx, gy, grad, scale):
        """The coverage of a chunk (see _coverage)."""
        order, starts = _slab_order(ip, self.gls.nobs, self.n)
        self.add(lambda s, first: _coverage_sel(ip, c2, s2, pol, gx, gy, grad, order[starts[s] : starts[s + 1]],
                                                self.edges[s], self.map, scale))

    def blocks(self, ip, c, c2, s2, pol, gx, gy, grad, nlabels, locout):
        """The blocks of a chunk (see _block_circulant); its unknowns into locout."""
        srt = _block_sort(ip, self.gls.nobs, nlabels)
        self.add(lambda s, first: _block_add(*srt, c, c2, s2, pol, gx, gy, grad, self.map, locout,
                                             self.slabs[s], self.slabs[s + 1], first))

    def finish(self):
        """The rounds of the other ranks of the node, once this rank has added
        all its chunks: before any other collective, which would deadlock with
        the barriers of the rounds."""
        for _ in range((self.rounds - self.done) * self.n):
            self.node.barrier()
        self.done = self.rounds

    def local(self, by_column=False):
        """This rank's pixels (nloc, ncols) of the sums over all the ranks."""
        self.finish()
        g = self.gls
        self.node.reduce(self.map, g._layout, by_column)
        a = g.lo - g._layout.lo
        return self.map[a : a + g.nloc].copy()


class GLS:
    """Map-maker for a fixed set of pointings and noise weights.

    Parameters
    ----------
    pix : int array (nchunk, ns), or list of 1D chunks / 2D blocks
        Pixel index of each sample, pix < 2^31 (e.g. HEALPix). Maps have
        max(pix) + 1 pixels; pixels without hits are not solved. Used without
        copies if int32 (chunks with flagged samples are copied, see
        overwrite_pix). pix < 0 marks
        flagged samples: by default their TOD is not used (taken as 0).
        Samples of a chunk with the same negative value share an
        intensity unknown, local to this rank and discarded: one value per
        sample is the exact GLS without them (marginalization), one per
        contiguous run fits an offset to it. With flag_method="fill",
        instead they have no unknowns and their gap-filled TOD is used.
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
        "block" (default): inverse of the exact I/Q/U (and gradient) block of
        P^T N^-1 P in each pixel. "jacobi": the original SANEPIC choice, exact
        I diagonal and Q/U = 2 / diag_I (assumes uniform angle coverage);
        exact diagonal for the gradient.
    comm : mpi4py communicator or None
        With MPI, each rank passes only its own chunks (possibly none). Each
        rank owns a contiguous range of pixels [lo, hi), 1/size of the map, and
        keeps the PCG vectors, the preconditioner and the mask only there: P^T
        is summed over ranks with a reduce-scatter that delivers to each rank
        its pixels, and the search direction is rebuilt on every rank for P
        with an allgather (together, the same traffic as one Allreduce). Only
        the map used by P and the map accumulating P^T (and, at setup, the
        coverage and the blocks of the preconditioner) are full size, one
        copy per rank or per node (full_maps). The pixels are split node by
        node and only the node leaders exchange maps between nodes.
    full_maps : where the full maps are with MPI: "rank" (default: one copy
        per rank, each worker thread adding P^T to a private one when they
        fit: fastest), "node" (one copy shared by the ranks of a node, in MPI-3
        shared memory, see node.py and _NodeSum: the ranks add P^T in rounds
        over slabs of pixels, synchronized by the node; the full maps handed
        out by solve, full and binned are shared too, and read-only). "node"
        saves memory when the full maps of the ranks of a node are a large
        part of its memory (at setup ~8 nobs (ncov + ncomp (ncomp + 1) / 2)
        bytes per rank: 5 GB at nside 2048 with I/Q/U, e.g. 48 ranks per
        node); with small maps it is slower and saves nothing. The choice
        is in `full_maps_info`. Without MPI, or with one rank per node,
        "rank". With MPI the map read by P is shared by the ranks of a node
        in every mode.
    nthreads : int or None
        Threads per process (FFTs, numba); None: OMP_NUM_THREADS, else 1.
    pol_angles : (angle2, gamma) or None
        Instead of psi and pol_efficiency: `angle2`, twice the polarization
        angle of each sample laid out like `pix`, reduced to [-pi, pi) (see
        :func:`two_angle`; used without copies if of `angle_dtype`), and
        `gamma`, the polarization efficiency of each chunk (scalar, or one
        value per sample, e.g. 0 in intensity-only samples).
    angle_dtype : np.float32 (default) or np.float64
        Storage of 2a (and of a per-sample efficiency, and of `offsets`):
        float32 gives an angle error of ~1e-7 rad.
    offsets : (gx, gy) or None
        Sub-pixel gradient pointing: the offsets of each sample from the
        center of its pixel, two arrays laid out like `pix` (any units, e.g.
        pixel sides, along two orthogonal directions of the tangent plane;
        0 in virtual pixels). The I map gets a gradient per pixel,
        m_p + Ix_p gx + Iy_p gy: the maps have ncomp = 3 (or 1) + 2
        components, I, Q, U, Ix, Iy. I no longer depends, to first order,
        on where the samples fall in the pixel: the error of the structure
        below the pixel size drops to second order in the offsets. It is
        not the value at the center either: the center and the mean over
        the pixel differ at second order (the pixel window). The
        gradient is solved only in pixels with at least `min_grad_hits` hits
        whose matrix sum_t (1, gx, gy)^T (1, gx, gy) has a reciprocal
        condition number >= `min_grad_rcond` (elsewhere Ix = Iy = 0: the
        mean of the samples, as without offsets).
    subpixel_order : 0, 1, 2 or None
        The order of the sub-pixel terms of I: 0 none (`offsets` unused), 1
        the gradient above, 2 also the second order: m_p + Ix gx + Iy gy + Hxx
        (gx^2 - 1/12) + Hxy gx gy + Hyy (gy^2 - 1/12), three more components
        (Hxx = Ixx / 2, Hxy = Ixy, Hyy = Iyy / 2, last in the maps). The
        quadratic terms average to zero over a uniformly covered square pixel
        of side 1 (offsets in pixel sides), so I is the mean over the pixel
        whatever the distribution of the samples in it: the error of the
        structure below the pixel size drops to third order and the maps
        have the pixel window of uniformly covered square pixels (make_maps
        reports the mean over each HEALPix pixel instead). Solved in pixels
        with a gradient, at least `min_hess_hits` hits and a reciprocal
        condition number of the white-noise block of (1, gx, gy, the three
        quadratic weights), scaled to a unit diagonal, >= `min_hess_rcond`
        (elsewhere 0). None (default): 1 with `offsets`, 0 without.
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
        in info["shared_amplitudes"] on every rank. Several templates with
        one amplitude each in the chunk can be one entry, `values` (n, ncol)
        and `labels` (ncol,): applied in one pass over the chunk (P and P^T)
        instead of one per template.
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
    prior, shared_prior : (precision, mean) or None
        Independent Gaussian priors on the amplitudes of `templates` (labels
        0, 1, ... of this rank) and of `shared_templates` (the same on every
        rank): a ~ N(mean, 1 / precision), two arrays with one value per label
        (precision 0: no prior). They add diag(precision) to the operator and
        precision * mean to P^T N^-1 d, so solve() gives the maximum of the
        posterior and shared_covariance() its covariance. solve() works on
        the deviations from the means (the TOD minus the templates at their
        means): with tight priors, precision * mean would otherwise dominate
        norm(b) and stop the PCG before the maps converge. Labels beyond
        those of the templates are amplitudes without data (equal to the
        mean). With `shared_groups`, the prior acts within the constraints.
    overwrite_pix : bool
        Relabel the flagged samples in the int32 `pix` arrays in place instead
        of copying the chunks that have them (make_maps does, as it owns them).
    flag_method : "marginalize" (default), "act" or "fill"
        With "act", each contiguous run of negative pixels is represented by
        local Legendre coefficients following Naess et al. (2025), Sect.
        3.3.3, instead of one unknown per sample. Flagged TOD is set to zero,
        with no random filling; original flagged values are never used.
        With "fill", negative pixels have no parameters: their rows of P
        and of the templates are zero. solve()/rhs() retain their TOD, which
        must already be finite gap-filled data. The normal matrix is the sky
        block of the extended system, without the marginalization correction
        (Patanchon et al. 2008, Sect. 3.6). make_maps fills these gaps itself
        (GLSParameters.flag_method=GapFill(...)).
    flag_sampling_rate_hz : scalar or one value per chunk
        Required with flag_method="act" to convert gap lengths to seconds.
    act_ncoeff : None, positive int, or callable
        Override ACT's coefficient count: None keeps the paper's rule, an
        integer fixes the count per gap (1: offsets), a callable takes the
        array of gap lengths in samples and the chunk sampling rate in Hz,
        and returns a positive integer scalar or array of matching shape.
        Counts are capped at each gap length; degree is count minus one.
        High orders use discrete orthonormal polynomials spanning the same
        space, instead of ill-conditioned sampled Legendre columns. Full
        counts use identity. No dense basis is stored in either case.
    """

    def __init__(self, pix, psi, weights, pol_efficiency=None, min_pol_hits=4, min_pol_rcond=1e-2, preconditioner="block", comm=None, nthreads=None, angle_dtype=np.float32, pol_angles=None, overwrite_pix=False, fft_lengths=None, fft_dtype=np.float64, common_modes=None, coarse=None, templates=None, shared_templates=None, shared_groups=None, template_dtype=np.float32, flag_method="marginalize", flag_sampling_rate_hz=None, act_ncoeff=None, prior=None, shared_prior=None, offsets=None, min_grad_hits=8, min_grad_rcond=1e-2, subpixel_order=None, min_hess_hits=20, min_hess_rcond=1e-2, full_maps="rank"):
        self.comm = comm
        if full_maps not in ("node", "rank"):
            raise ValueError(f"GLS.full_maps must be 'node' or 'rank', not {full_maps!r}")
        self.timings = tm = _Timings()  # seconds of the phases of the setup (see _Timings)
        self.pcg_timings = _Timings()  # of the last solve
        self._tm = tm  # where the phases go now (_NodeSum: the setup or the solve)
        if flag_method not in ("marginalize", "act", "fill"):
            raise ValueError("GLS.flag_method must be 'marginalize', 'act' or 'fill' (offsets use repeated negative pixel labels)")
        self._filled_flags = flag_method == "fill"
        self.nthreads = resolve_nthreads(nthreads)
        self.ip = []
        for p in _chunks(pix):
            if p.size and p.max() >= 2**31:
                raise ValueError("pixel indices must be < 2^31")
            self.ip.append(np.ascontiguousarray(p, dtype=np.int32))
        self.weights = _chunks(weights)
        if flag_method != "act" and (act_ncoeff is not None or flag_sampling_rate_hz is not None):
            raise ValueError("act_ncoeff and flag_sampling_rate_hz require flag_method='act'")
        if flag_method == "act":
            _validate_act_ncoeff(act_ncoeff)
            if flag_sampling_rate_hz is None:
                raise ValueError("flag_sampling_rate_hz is required for ACT cuts")
            rates = np.broadcast_to(np.asarray(flag_sampling_rate_hz, dtype=float), (len(self.ip),))
            if np.any(~np.isfinite(rates)) or np.any(rates <= 0):
                raise ValueError("flag_sampling_rate_hz must be finite and positive")
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
        nobs = 1 + max((int(p.max(initial=-1)) for p in self.ip), default=-1)
        self.nobs = nobs if comm is None else max(comm.allgather(nobs))
        # flagged samples: labels nobs, nobs + 1, ... of unknowns local to this rank
        self.nflag = 0
        self._cuts = [np.empty((0, 4), np.int64) for _ in self.ip]
        for k, p in enumerate(self.ip):
            if self._filled_flags:
                continue  # negative labels remain excluded, without PCG state
            flagged = np.flatnonzero(p < 0)
            if flagged.size:
                if not (overwrite_pix and p.flags.writeable):
                    p = p.copy()
                if flag_method == "act":
                    edges = np.flatnonzero(np.diff(np.r_[False, p < 0, False].astype(np.int8)))
                    starts, stops = edges[::2], edges[1::2]
                    lengths = stops - starts
                    counts = _act_ncoeff(lengths, rates[k], act_ncoeff)
                    first = self.nflag + np.cumsum(counts) - counts
                    poly = counts > 1
                    self._cuts[k] = np.column_stack([starts[poly], stops[poly], first[poly], counts[poly]])
                    # Chunk workers write C^T f in place: coefficient ranges must not overlap
                    own = slice(self.nflag, self.nflag + int(counts.sum()))
                    if np.any(first < own.start) or np.any(first + counts > own.stop):
                        raise RuntimeError("ACT cut coefficients outside the range of their chunk")
                    self.nflag = own.stop
                    if self.nobs + self.nflag >= 2**31:
                        raise ValueError("too many pixels and flagged samples on this rank for int32 labels")
                    # Constant cuts use the existing offset kernels, with no
                    # run metadata. Higher orders have zero sky rows and C.
                    p[flagged] = np.repeat(np.where(poly, -1, self.nobs + first), lengths)
                else:
                    labels, unknown = np.unique(p[flagged], return_inverse=True)
                    p[flagged] = self.nobs + self.nflag + unknown
                    self.nflag += labels.size
                self.ip[k] = p
        if self.nobs + self.nflag >= 2**31:
            raise ValueError("too many pixels and flagged samples on this rank for int32 labels")
        self._root = comm is None or comm.rank == 0
        self._shgroups = [np.asarray(g, dtype=np.int64) for g in shared_groups or []]
        self._locprec = np.zeros(self.nflag)  # diagonal of P^T N^-1 P of each unknown, then inverted
        self.pol = psi is not None or pol_angles is not None
        if subpixel_order is None:
            subpixel_order = 0 if offsets is None else 1
        if subpixel_order not in (0, 1, 2):
            raise ValueError(f"GLS.subpixel_order must be 0, 1 or 2, not {subpixel_order!r}")
        if subpixel_order and offsets is None:
            raise ValueError("GLS: subpixel_order >= 1 needs offsets")
        self.grad = subpixel_order  # order of the sub-pixel terms
        self.npol = 3 if self.pol else 1  # I/Q/U or I, then the gradient Ix, Iy and the Hessian with offsets
        self.ncomp = self.npol + (0, 2, 5)[self.grad]
        self._ncov = 1 + self.ncomp * (self.ncomp + 1) // 2  # columns of the coverage (see _coverage)
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
        if self.grad:
            self.gx, self.gy = ([np.asarray(a, dtype=angle_dtype) for a in _chunks(x)] for x in offsets)

        # pixels [lo, hi) of this rank, node by node: the ranks of a node share
        # the full map of P (see node.py)
        self._node = None if comm is None else node_of(comm)
        self._pcomm = None if comm is None else self._node.pixels
        size, rank = (1, 0) if comm is None else (comm.size, self._pcomm.rank)
        self._nsky = 0  # set below; PCG vectors: [this rank's pixels (ncomp, nloc), flagged samples]
        if size > 1:
            _check_mpi_count(self.nobs * self._ncov)  # the largest array exchanged: the coverage
        bounds = np.arange(size + 1, dtype=np.int64) * self.nobs // size
        self._counts = np.diff(bounds)
        self._layout = None if comm is None else self._node.layout(bounds)
        # several ranks on a node: their sums over pixels in one map per node (full_maps)
        self._node_maps, self.full_maps_info = self._choose_full_maps(full_maps)  # the same on every rank
        self.lo, self.hi = int(bounds[rank]), int(bounds[rank + 1])
        self.nloc = self.hi - self.lo
        self._nsky = self.ncomp * self.nloc

        # P^T: each thread owns a range of pixels holding ~1/nthreads of this
        # rank's samples (with a map per node: equal ranges of each slab, see _NodeSum)
        nt = min(self.nthreads, numba.config.NUMBA_NUM_THREADS)
        tm.mark("layout")
        if self._node_maps:
            self._edges = np.arange(nt + 1, dtype=np.int64) * self.nobs // nt
            cov = self._coverage_local()
        else:
            cov = self._chunk_coverage()
            cum = np.cumsum(cov[:, 0])
            target = (cum[-1] if cum.size else 0.0) * np.arange(nt + 1) / nt
            self._edges = np.searchsorted(cum, target).astype(np.int64)
            self._edges[[0, -1]] = 0, self.nobs
            cov = self._scatter_sum(cov, by_column=True)
        tm.mark("coverage")
        self.hits = cov[:, 0].astype(np.int64)  # this rank's pixels, like mask and precond
        self.mask = np.repeat((self.hits > 0)[None], self.ncomp, axis=0)
        if self.pol:
            self.mask[1:3] &= cov[:, 1] >= min_pol_hits
            if min_pol_rcond > 0:
                self.mask[1:3] &= self._rcond(cov, [0, 1, 2], self.mask[1]) >= min_pol_rcond
        if self.grad:
            g = self.npol
            self.mask[g:] &= self.hits >= max(min_grad_hits, 3)
            if min_grad_rcond > 0:
                self.mask[g:] &= self._rcond(cov, [0, g, g + 1], self.mask[g]) >= min_grad_rcond
            if self.grad > 1:
                self.mask[g + 2 :] &= self.hits >= max(min_hess_hits, 6)
                if min_hess_rcond > 0:
                    self.mask[g + 2 :] &= self._rcond(cov, [0, *range(g, g + 5)], self.mask[g + 2], scaled=True) >= min_hess_rcond
        self._cov = cov  # this rank's pixels, kept for binned
        tm.mark("mask")

        nb = self.ncomp * (self.ncomp + 1) // 2
        blocks = _NodeSum(self, nb, len(self.ip)) if self._node_maps else np.zeros((self.nobs, nb))
        last_w = c = None  # consecutive chunks often share their weights array
        with numba_threads(self.nthreads):
            for k, (ip, w) in enumerate(zip(self.ip, self.weights)):
                if w is not last_w or c.size != self._nfft[k]:
                    # first row of the circulant N^-1 of the chunk
                    c = ducc0.fft.c2r(np.asarray(w, dtype=complex), lastsize=self._nfft[k], forward=False, inorm=2, nthreads=self.nthreads)
                    last_w = w
                _, c2, s2 = self._pol_arrays(k)
                gx, gy, _ = self._grad_arrays(k)
                if self._node_maps:
                    blocks.blocks(ip, c, c2, s2, self.pol, gx, gy, self.grad, self.nobs + self.nflag, self._locprec)
                else:
                    _block_circulant(ip[None], c[None], c2[None], s2[None], self.pol, gx[None], gy[None], self.grad,
                                     blocks, self.nobs + self.nflag, self._locprec)
                if self._cuts[k].size:
                    _cut_diagonal(self._cuts[k], c, self._locprec, self.nthreads, w[0])
        blocks = self._pixel_sums(blocks, by_column=True)
        self._flagprec = self._inverse(self._locprec)
        self._solved = None  # the full Q/U mask, for the templates
        tm.mark("blocks")
        self.set_templates(templates, shared_templates, prior, shared_prior)  # its phases in tm
        bad = int(np.any(blocks[self.mask[0], 0] <= 0))
        if (bad if comm is None else comm.allreduce(bad)) > 0:
            raise ValueError("Preconditioner has non-positive elements: check noise weights")
        self._set_preconditioner(blocks, preconditioner)
        self._full = self._buf = self._locbuf = None  # work maps, allocated by _apply
        self._einv = None
        tm.mark("preconditioner")
        if coarse is not None:
            self._set_coarse(coarse, blocks[:, 0])
            tm.mark("coarse")
        del blocks

    def _choose_full_maps(self, full_maps):
        """(maps per node, description) for GLS.full_maps (collective with MPI)."""
        if self.comm is None:
            return False, "rank (no MPI)"
        several = max(self.comm.allgather(self._node.node.size)) > 1  # else nothing to share
        node = several and full_maps == "node"
        return node, f"{'node' if node else 'rank'} ({full_maps}{'' if several else ', one rank per node'})"

    def set_templates(self, templates=None, shared_templates=None, prior=None, shared_prior=None):
        """Set (or replace) `templates`, `shared_templates` and their priors
        (see the class), keeping the rest of the setup: pixels,
        preconditioner, coarse level. Collective with MPI. The constraints
        `shared_groups` are kept."""
        # local amplitudes after the unknowns of flagged samples, then (on rank 0
        # only) the shared ones
        prior, shared_prior = self._check_prior(prior, "prior"), self._check_prior(shared_prior, "shared_prior")
        self._tmpl, namp = self._parse_templates(templates)
        self.namp = max(namp, len(prior[0]))
        self._shared, nsh = self._parse_templates(shared_templates)
        comm = self.comm
        nsh = max(nsh, len(shared_prior[0]))
        self.nshared = nsh if comm is None else max(comm.allgather(nsh))
        self._prior = self._prior_terms(prior, self.namp)
        self._shprior = self._prior_terms(shared_prior, self.nshared)
        self._nsh_loc = self.nshared if self._root else 0  # the shared amplitudes in this rank's PCG vectors
        self._shv = np.zeros(self.nshared)  # the shared amplitudes for P, set by _apply
        if self.pol and (self.namp if comm is None else comm.allreduce(self.namp)) + self.nshared:  # collective
            if self._solved is None:
                self._solved = self.full(self.mask[1].astype(np.uint8)).view(bool)
            for k, ip in enumerate(self.ip):
                valid = (ip >= 0) & (ip < self.nobs)
                bad = np.zeros(ip.size, bool)
                bad[valid] = ~self._solved[ip[valid]]
                if bad.any():  # copies: the arrays may be the caller's (none if 0 already)
                    for tmpl in (self._tmpl[k], self._shared[k]):
                        tmpl[:] = [(np.where(bad.reshape(-1, *[1] * (v.ndim - 1)), 0, v).astype(v.dtype, copy=False)
                                    if np.any(v[bad] != 0) else v, lab) for v, lab in tmpl]
        self.timings.mark("templates: Q/U mask")
        diag = self._template_diag(self._shared, self.nshared)
        self.timings.mark("templates: shared diagonal")
        diag = (diag if comm is None else comm.allreduce(diag)) + self._shprior[0]
        local = self._template_diag(self._tmpl, self.namp) + self._prior[0]
        self.timings.mark("templates: local diagonal")
        self._locprec = np.concatenate([self._flagprec, self._inverse(local), self._inverse(diag)[: self._nsh_loc]])

    @staticmethod
    def _check_prior(prior, name):
        """(precision, mean) as two float arrays, checked; empty ones for None."""
        if prior is None:
            return np.zeros(0), np.zeros(0)
        prec, mean = (np.asarray(x, dtype=float) for x in prior)
        if prec.ndim != 1 or prec.shape != mean.shape:
            raise ValueError(f"{name}: (precision, mean), two 1D arrays of the same length")
        if not (np.all(np.isfinite(prec)) and np.all(prec >= 0) and np.all(np.isfinite(mean))):
            raise ValueError(f"{name}: the precisions must be finite and >= 0, the means finite")
        return prec, mean

    @staticmethod
    def _prior_terms(prior, n):
        """(precision, mean) of n amplitudes (2, n), zero without a prior."""
        prec, mean = prior
        out = np.zeros((2, n))
        out[0, : prec.size] = prec
        out[1, : prec.size] = mean
        return out

    def _parse_templates(self, templates):
        """Per chunk, the (values, labels) of its templates (0 where not applied;
        values of template_dtype, labels int32 or one int), and the number of
        amplitudes. Copies only where needed."""
        templates = list(templates or [])

        def parse(entry):
            """(k, values, labels, number of amplitudes) of one entry, or None."""
            k, values, labels = entry
            bad = (self.ip[k] < 0) | (self.ip[k] >= self.nobs)
            if np.ndim(values) == 2:  # columns with one amplitude each
                labels = np.asarray(labels, dtype=np.int64)
                if labels.shape != (np.shape(values)[1],) or np.any(labels < 0):
                    raise ValueError("2D template values need one label >= 0 per column")
                values = np.ascontiguousarray(values, dtype=self.template_dtype)
                if bad.any() and np.any(values[bad] != 0):
                    values = np.where(bad[:, None], 0, values).astype(self.template_dtype, copy=False)
                return k, values, labels, int(labels.max(initial=-1)) + 1
            if np.ndim(labels) == 0:
                labels = int(labels)
                if labels < 0:
                    return None
                n = labels + 1
            else:
                labels = np.asarray(labels)
                bad = bad | (labels < 0)
                n = int(labels.max(initial=-1)) + 1
            values = np.asarray(values, dtype=self.template_dtype)
            if bad.any() and np.any(values[bad] != 0):  # e.g. a mask of the calibration: 0 already, no copy
                values = np.where(bad, 0, values).astype(self.template_dtype, copy=False)
                if np.ndim(labels):
                    labels = np.where(bad, 0, labels)
            if np.ndim(labels):
                labels = labels.astype(np.int32, copy=False)
            return k, values, labels, n

        nw = min(self.nthreads, len(templates))  # numpy on whole chunks: the threads run in parallel
        out, n = [[] for _ in self.ip], 0
        for r in (map(parse, templates) if nw <= 1 else self._workers(nw).map(parse, templates)):
            if r is not None:
                out[r[0]].append((r[1], r[2]))
                n = max(n, r[3])
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
            entries = [(v[:, j], int(lab[j])) for v, lab in tmpls[k] if np.ndim(v) == 2 for j in range(v.shape[1])]
            for values, labels in [e for e in tmpls[k] if np.ndim(e[0]) == 1] + entries:
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

    def _chunk_coverage(self, scale=None):
        """The coverage (nobs, _ncov) of this rank's chunks (see _coverage), each
        chunk k counted scale[k] times (None: once): the chunks in parallel,
        worker j on chunks j, j + nw, ... into its own array, summed in place
        in a fixed order. At most 8 workers, limited by _COVERAGE_BYTES (8
        _ncov bytes per pixel each, at least one array)."""
        max_workers = max(1, _COVERAGE_BYTES // max(8 * self._ncov * self.nobs, 1))
        nw = min(self.nthreads, len(self.ip), 8, max_workers)
        # Separate allocations: returning covs[0] must not retain the others.
        covs = [np.zeros((self.nobs, self._ncov)) for _ in range(max(nw, 1))]
        dtype = self._c2.dtype if self.pol else np.float32

        def work(j):
            nmax = max(ip.size for ip in self.ip[j::nw])
            c2, s2 = (np.empty(nmax, dtype), np.empty(nmax, dtype)) if self.pol else (self._dummy, self._dummy)
            for k in range(j, len(self.ip), max(nw, 1)):
                n = self.ip[k].size
                if self.pol:
                    _cos_sin_1(self.angle2[k], self.gamma[k], c2[:n], s2[:n], _TRIG_COS, _TRIG_SIN)
                _coverage(self.ip[k], c2[:n] if self.pol else c2, s2[:n] if self.pol else s2, self.pol,
                          *self._grad_arrays(k), covs[j], 1.0 if scale is None else float(scale[k]))

        list(map(work, range(nw)) if nw <= 1 else self._workers(nw).map(work, range(nw)))
        for cov in covs[1:]:
            covs[0] += cov
        return covs[0]

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
        if self._cuts[k].size:
            # Written in place, also by concurrent chunk workers: like the
            # offsets in _backproject, each chunk owns a disjoint range of
            # _locbuf[:nflag] (checked in __init__). Template amplitudes are
            # shared by chunks, so workers sum them in private `loc` buffers.
            _cut_transpose(self._cuts[k], f, self._locbuf)
        for values, labels in self._tmpl[k]:
            _template_transpose(values, labels, f, loc)
        for values, labels in self._shared[k]:
            _template_transpose(values, labels, f, sh)

    def _set_coarse(self, coarse, diag_i):
        """E^-1 = (Z^T A Z)^-1 of the two-level preconditioner (see `coarse`);
        diag_i: the I diagonal of A on this rank's pixels."""
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
        # Pseudo-inverse in units of the summed diagonal of A: coarse modes with no
        # weight (the monopole with fmin = 0 and no padding) are left to the PCG
        s = 1.0 / np.sqrt(self._csum(diag_i[self._cidx], nz)[used])
        vals, vecs = np.linalg.eigh(0.5 * (E + E.T) * np.outer(s, s))
        keep = vals > 1e-10
        vecs = vecs[:, keep] * s[:, None]
        self._einv = np.zeros((nz, nz))
        self._einv[np.ix_(used, used)] = (vecs / vals[keep]) @ vecs.T
        self._nz = nz

    def _csum(self, v, nz):
        """Z^T v: sums of v (on the I entries of this rank) per coarse pixel, over ranks."""
        c = np.bincount(self._cpar, v, minlength=nz)
        return c if self.comm is None else self.comm.allreduce(c)

    def _white(self, cov, rows=slice(None)):
        """The white-noise blocks sum_t w_t w_t^T (see _weights) of the pixels
        `rows` from their coverage, (n, ncomp, ncomp)."""
        return _symmetric(np.column_stack([cov[rows, 0], cov[rows, 2:]]), self.ncomp)

    def _rcond(self, cov, comps, seen, scaled=False):
        """Reciprocal condition number of the white-noise block of the
        components `comps` in the pixels `seen` (0 elsewhere): for I/Q/U the
        angle-coverage matrix, for I and the gradient the spread of the
        offsets; `scaled`: of the block scaled to a unit diagonal."""
        rcond = np.zeros(self.nloc)
        M = self._white(cov, seen)[:, comps][:, :, comps]
        if scaled:
            d = np.sqrt(np.einsum("nii->ni", M))
            with np.errstate(divide="ignore", invalid="ignore"):
                M = np.nan_to_num(M / d[:, :, None] / d[:, None, :])
        out = np.empty(len(M))
        with numba_threads(self.nthreads):
            _sym_rcond(np.ascontiguousarray(M), out)
        rcond[seen] = out
        return rcond

    def _set_preconditioner(self, blocks, kind):
        if kind not in ("block", "jacobi"):
            raise ValueError(f"Unknown preconditioner {kind!r}: use 'block' or 'jacobi'")
        diag = blocks[:, [j * self.ncomp - j * (j - 1) // 2 for j in range(self.ncomp)]].T  # (ncomp, nloc)
        if kind == "jacobi" or self.ncomp == 1:
            inv = np.divide(1.0, diag, out=np.zeros_like(diag), where=self.mask)
            inv[1 : self.npol] = 2.0 * inv[0]  # SANEPIC's Q/U
            self.precond = inv * self.mask
            return
        # Invert the block of the solved components; where it is ill-conditioned
        # fall back to its diagonal
        M = _symmetric(blocks, self.ncomp)
        off = ~self.mask.T  # (nloc, ncomp)
        M[off[:, :, None] | off[:, None, :]] = 0.0
        d = np.where(off, 1.0, np.abs(diag.T))
        M[:, np.arange(self.ncomp), np.arange(self.ncomp)] = d
        det, inv = np.empty(len(M)), np.empty_like(M)
        with numba_threads(self.nthreads):
            _det_inv(M, det, inv)
        good = self.mask[0] & (det > 1e-12 * d.prod(axis=1))
        self.precond = np.where(good[:, None, None], inv, 0.0)
        bad = self.mask[0] & ~good
        self.precond[bad] = np.eye(self.ncomp) / d[bad][:, None, :]
        self.precond *= self.mask.T[:, :, None] & self.mask.T[:, None, :]

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
            self._pcomm.Reduce_scatter(np.ascontiguousarray(full), out, recvcounts=self._counts * full.shape[1])
            return out
        col = np.empty(self.nloc, full.dtype)
        for j in range(full.shape[1]):
            self._pcomm.Reduce_scatter(np.ascontiguousarray(full[:, j]), col, recvcounts=self._counts)
            out[:, j] = col
        return out

    def _pixel_sums(self, acc, by_column=False):
        """This rank's pixels (nloc, k) of sums over the ranks: of a full map
        of this rank (see _scatter_sum) or of a _NodeSum."""
        return acc.local(by_column) if isinstance(acc, _NodeSum) else self._scatter_sum(acc, by_column)

    def _coverage_local(self, scale=None):
        """The coverage of this rank's pixels (nloc, _ncov), summed over the
        ranks (see _chunk_coverage)."""
        if not self._node_maps:
            return self._scatter_sum(self._chunk_coverage(scale), by_column=True)
        acc = _NodeSum(self, self._ncov, len(self.ip))
        with numba_threads(self.nthreads):
            for k, ip in enumerate(self.ip):
                _, c2, s2 = self._pol_arrays(k)
                acc.coverage(ip, c2, s2, self.pol, *self._grad_arrays(k), 1.0 if scale is None else float(scale[k]))
        return acc.local(by_column=True)

    def _pt(self, acc, k, t, c2, s2, pol, locbuf):
        """acc += P^T t for chunk k, its flagged samples into locbuf; acc a full
        map of this rank or a _NodeSum."""
        self._pt_group(acc, [(k, t, c2, s2, pol)], locbuf)

    def _pt_group(self, acc, items, locbuf):
        """_pt of the chunks items (k, t, c2, s2, pol); with a _NodeSum, in one
        set of rounds."""
        if isinstance(acc, _NodeSum):
            acc.backproject([(self.ip[k], t, c2, s2, pol, *self._grad_arrays(k)) for k, t, c2, s2, pol in items], locbuf)
        else:
            for k, t, c2, s2, pol in items:
                _backproject(self.ip[k], t, c2, s2, pol, *self._grad_arrays(k), self._edges, acc, locbuf)

    def _round_groups(self, free):
        """The chunks `free` in the groups whose P^T _apply adds in one set of
        rounds of a _NodeSum: with worker threads, the batches of
        _apply_chunks (one chunk per worker, in their buffers); else
        consecutive chunks whose filtered TOD and angles (copies, ~16 bytes
        per sample) fit in _ROUND_BYTES. One chunk per group without a map
        per node."""
        if not self._node_maps:
            return [[k] for k in free]
        nw = min(self.nthreads, len(free))
        if nw > 1:
            return [free[i : i + nw] for i in range(0, len(free), nw)]
        groups, size = [], np.inf
        for k in free:
            n = 16 * self.ip[k].size
            if size + n > _ROUND_BYTES:
                groups.append([])
                size = 0
            groups[-1].append(k)
            size += n
        return groups

    def _gather(self, loc, out=None):
        """Full (nobs, k) array from the (nloc, k) pieces of all ranks (allgather)."""
        # counts fit in int32: checked by _check_mpi_count at setup
        out = np.empty((self.nobs, loc.shape[1]), loc.dtype) if out is None else out
        if self.comm is None:
            out[:] = loc
        else:
            self._pcomm.Allgatherv(np.ascontiguousarray(loc), [out, self._counts * loc.shape[1]])
        return out

    def full(self, x):
        """Gather a per-pixel array of this rank, (k, nloc) or (nloc,), into the
        full (k, nobs) or (nobs,) array on every rank. Collective with MPI. With
        full_maps "node", a read-only array shared by the ranks of the node
        (one copy per node)."""
        x = np.asarray(x)
        if self._node_maps:
            one = x.ndim == 1
            x = x[None] if one else x
            out = self._node.output((x.shape[0], self.nobs), x.dtype)
            out[:, self.lo : self.hi] = x
            self._node.gather_rows(out, self._layout)
            self._node.freeze(out)
            return out[0] if one else out
        if x.ndim == 1:
            return self._gather(x[:, None])[:, 0]
        return np.ascontiguousarray(self._gather(x.T).T)

    def _full_mask_of(self):
        """The full mask (ncomp, nobs), gathered once (collective the first time)."""
        if getattr(self, "_full_mask", None) is None:
            self._full_mask = self.full(self.mask.astype(np.uint8)).view(bool)
        return self._full_mask

    def _freeze(self, out):
        """Hand out an array of _healpix (or written like it): read-only and
        visible to the ranks of the node if shared, as it is otherwise."""
        return self._node.freeze(out) if self._node_maps else out

    @property
    def _writer(self):
        """Whether this rank writes the maps shared by its node (all ranks without them)."""
        return not self._node_maps or self._node.leader

    def _healpix(self, m, nside, fill, masked=True):
        """to_healpix into a new array, with several ranks on a node shared by
        them (collective; each one fills a part): writable, to be finished by
        the _writer ranks and handed out with _freeze. Without `masked`, the
        unseen pixels are not filled."""
        npix = 12 * nside * nside
        if not self._node_maps:
            out, (a, b) = np.empty((m.shape[0], npix), np.asarray(fill).dtype), (0, npix)
        else:
            out = self._node.output((m.shape[0], npix), np.asarray(fill).dtype)
            a, b = self._node.part(npix)
        seen = self._full_mask_of() if masked else None
        out[:, a:b] = fill
        c = min(b, self.nobs)  # labels beyond the map (virtual pixels) dropped
        if a < c:
            out[:, a:c] = m[:, a:c]
            if masked:
                out[:, a:c][~seen[:, a:c]] = fill
        if self._node_maps:
            self._node.barrier()
        return out

    def _sum(self, a, b):
        """a + b for full maps (ncomp, nobs), shared by the ranks of a node as a and b."""
        if not self._node_maps:
            return a + b
        out = self._node.output(a.shape, np.result_type(a, b))
        lo, hi = self._node.part(a.shape[1])
        np.add(a[:, lo:hi], b[:, lo:hi], out=out[:, lo:hi])
        return self._node.freeze(out)

    def _dot(self, a, b):
        with self._tm.phase("dot products (allreduce)"):
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

    def _grad_arrays(self, k):
        """(gx, gy, grad) of chunk k (the offsets, see `offsets`), in the
        order of the kernels' arguments."""
        if not self.grad:
            return self._dummy, self._dummy, 0
        return self.gx[k], self.gy[k], self.grad

    def _filter(self, t, k, single, slot=None):
        """N^-1 t for chunk k: irfft(rfft(t) * w) on an FFT of length m >= t.size
        (t zero-extended, the result truncated), in float32 if `single`. The
        work buffers are allocated once (for the longest FFT) and reused: the
        result is overwritten by the next call. With a chunk worker's `slot`,
        its buffers and its share of the threads (see _apply_chunks)."""
        m, n, w = self._nfft[k], t.size, self.weights[k]
        real, cplx = (np.float32, np.complex64) if single else (np.float64, np.complex128)
        bufs, nthreads = (self._fftbuf, self.nthreads) if slot is None else (slot["fft"], slot["nthreads"])
        if real not in bufs or bufs[real][0].size < m:
            mmax = max(self._nfft) if slot is None else slot["nfft"]
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

    def _apply(self, v=None, tod=None, exact=False, prior=True):
        """P^T N^-1 P v (+ the precisions of the priors times v, unless not
        `prior`) or P^T N^-1 tod (without the priors), one chunk at a time. v and the result
        are flat PCG vectors: this rank's pixels (ncomp, nloc) raveled, then the
        unknowns of its flagged samples. FFTs in float32 only for P^T N^-1 P v
        with fft_dtype=np.float32 and not `exact`."""
        single = self._single and v is not None and not exact
        if self._locbuf is None:
            # the map for P^T (one per node: below; the search direction for P: _set_full)
            self._buf = None if self._node_maps else np.empty((self.nobs, self.ncomp))
            self._locbuf = np.empty(self.nflag + self.namp)
        loc = self._locbuf[:0]
        tm = self._tm
        if v is not None:
            with tm.phase("map for P (allgather)"):
                self._set_full(v[: self._nsky].reshape(self.ncomp, self.nloc).T)  # the whole map, for P
            loc = v[self._nsky :]
            if self.nshared:
                sh = self._project_shared(v[len(v) - self._nsh_loc :].copy()) if self._root else None
                self._shv = sh if self.comm is None else self.comm.bcast(sh)
        free = [k for k in range(len(self.ip)) if not self._grouped[k]]
        groups = self._round_groups(free)
        if self._node_maps:  # one set of rounds per group, and per chunk with common modes
            buf = _NodeSum(self, self.ncomp, len(groups) + sum(len(c) for c, _, _ in self._groups))
        else:
            buf = self._buf
            buf[:] = 0.0
        self._locbuf[:] = 0.0
        self._shbuf = np.zeros(self.nshared)
        with numba_threads(self.nthreads), tm.phase("chunks (P, N^-1, templates, P^T)"):
            if min(self.nthreads, len(free)) > 1:
                self._apply_chunks(free, v, tod, loc, single, buf)
            else:
                for group in groups:
                    items = []
                    for k in group:
                        pol, c2, s2 = self._pol_arrays(k)
                        t = self._chunk_tod(k, v, tod, loc, c2, s2, pol)
                        f = self._filter(t, k, single)
                        self._template_back(k, f)
                        if len(group) > 1:  # the buffers are reused by the next chunk
                            f, c2, s2 = (None if x is None else x.copy() for x in (f, c2, s2))
                        items.append((k, f, c2, s2, pol))
                    self._pt_group(buf, items, self._locbuf)
            for chunks, U, M in self._groups:
                self._apply_group(chunks, U, M, v, tod, loc, buf)
        with tm.phase("sums over the ranks (reduce-scatter)"):
            if isinstance(buf, _NodeSum):
                buf.finish()
            sh = self._shbuf if self.comm is None or not self.nshared else self.comm.allreduce(self._shbuf)
            sums = self._pixel_sums(buf)
        if v is not None and prior:  # precision * v
            self._locbuf[self.nflag :] += self._prior[0] * loc[self.nflag : self.nflag + self.namp]
            sh = sh + self._shprior[0] * self._shv
        return np.concatenate([(sums.T * self.mask).ravel(), self._locbuf, self._project_shared(sh)[: self._nsh_loc]])

    def _set_full(self, loc):
        """The search direction (nobs, ncomp) for P in self._full, from the pixels
        `loc` (nloc, ncomp) of every rank: with MPI one copy per node, shared by
        its ranks, fetched again at each product (an other GLS of the same
        communicator may have enlarged it)."""
        if self._node is None:
            if self._full is None:
                self._full = np.empty((self.nobs, self.ncomp))
            self._full[:] = loc
            return
        node = self._node
        node.barrier()  # the ranks of the node no longer read the previous direction
        self._full = node.work("full", self.nobs * self.ncomp).reshape(self.nobs, self.ncomp)
        self._full[self.lo : self.hi] = loc
        node.allgather(self._full, self._layout)

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
        Each worker keeps ~40 bytes per sample of its longest assigned chunk
        with float64 FFTs (both precisions are kept with float32 FFTs)."""
        nw = min(self.nthreads, len(ks))
        pool = self._workers(nw)
        private = not isinstance(buf, _NodeSum) and nw * buf.nbytes <= _PRIVATE_MAPS_BYTES
        if private:
            # Preserve deterministic largest-first assignment and input order.
            batches = [[] for _ in range(nw)]
            loads = [(0, j) for j in range(nw)]
            for k in sorted(ks, key=lambda k: -self._nfft[k]):
                load, j = heapq.heappop(loads)
                batches[j].append(k)
                heapq.heappush(loads, (load + self._nfft[k], j))
            batches = [sorted(batch) for batch in batches]
        else:
            batches = [ks[j::nw] for j in range(nw)]
        for j, batch in enumerate(batches):
            nmax = max(self.ip[k].size for k in batch)
            if len(self._slots) <= j:
                self._slots.append(dict(fft={}))
            slot = self._slots[j]
            if "t" not in slot or slot["t"].size < nmax:
                c2, s2 = (np.empty(nmax, self._c2.dtype), np.empty(nmax, self._c2.dtype)) if self.pol else (None, None)
                slot.update(c2=c2, s2=s2, t=np.empty(nmax))
            slot["nfft"] = max(self._nfft[k] for k in batch)
        slots = self._slots[:nw]
        if private and (self._maps is None or self._maps.shape[0] < nw):
            self._maps = np.zeros((nw, *buf.shape))  # zeroed again by _reduce_maps
        flag = self._locbuf[: self.nflag]  # disjoint labels for each chunk
        for j, slot in enumerate(slots):
            slot["nthreads"] = max(1, self.nthreads // nw)
            slot["loc"], slot["sh"] = np.zeros(self.namp), np.zeros(self.nshared)
            if private:
                slot["map"] = self._maps[j]
        whole = np.array([0, self.nobs], np.int64)  # one thread: all the pixels

        def work(task):
            k, slot = task
            pol, c2, s2 = self._pol_arrays(k, slot)
            t = self._chunk_tod(k, v, tod, loc, c2, s2, pol, slot)
            f = self._filter(t, k, single, slot)
            self._template_back(k, f, slot["loc"], slot["sh"], serial=True)
            if not private:
                return pol, c2, s2, f
            _backproject_1(self.ip[k], f, c2, s2, pol, *self._grad_arrays(k), whole, slot["map"], flag)

        if private:
            # A dynamic slot queue changes the grouping of floating-point
            # sums with task completion order, including template amplitudes.
            # Give each worker a fixed set of chunks and preserve their input
            # order. Largest-first greedy assignment balances heterogeneous
            # FFT lengths without allocating a private map for every chunk.
            def work_batch(j):
                for k in batches[j]:
                    work((k, slots[j]))

            list(pool.map(work_batch, range(nw)))
            _reduce_maps(self._maps[:nw], buf)
        else:
            for i in range(0, len(ks), nw):  # the groups of _round_groups: one set of rounds each
                batch = ks[i : i + nw]
                done = list(pool.map(work, zip(batch, slots)))
                self._pt_group(buf, [(k, f, c2, s2, pol) for k, (pol, c2, s2, f) in zip(batch, done)], self._locbuf)
        for slot in slots:
            self._locbuf[self.nflag :] += slot["loc"]
            self._shbuf += slot["sh"]

    def _chunk_tod(self, k, v, tod, loc, c2, s2, pol, slot=None):
        """P v, or chunk TOD (marginalized flags set to 0, gap fills kept);
        with a chunk worker's `slot`, in its buffer and single-threaded."""
        ip = self.ip[k]
        if v is None:
            t = np.asarray(tod[k], dtype=float)
            if self._filled_flags and not np.all(np.isfinite(t[ip < 0])):
                raise ValueError("flag_method='fill' needs finite gap-filled TOD at flagged samples; make_maps fills it (GapFill)")
            if self.nflag:  # the TOD of flagged samples is not used (it may be NaN)
                t = np.where((ip < 0) | (ip >= self.nobs), 0.0, t)
            return t
        t = np.empty(ip.size) if slot is None else slot["t"][: ip.size]
        (_project if slot is None else _project_1)(ip, self._full, c2, s2, pol, *self._grad_arrays(k), loc, t)
        if self._cuts[k].size:
            _cut_project(self._cuts[k], loc, t)
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
            self._pt(buf, k, t, c2, s2, pol, self._locbuf)
            self._template_back(k, t)

    def _precond_flat(self, r):
        with self._tm.phase("preconditioner"):
            return self._precond_flat_(r)

    def _precond_flat_(self, r):
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
        available with marginalized flagged samples."""
        self._no_flags()
        return self._apply(np.ravel(m)).reshape(self.ncomp, self.nloc)

    def rhs(self, tod):
        """P^T N^-1 d on this rank's pixels (ncomp, nloc), for timelines laid out
        like `pix`. Collective with MPI. Not available with marginalized flags."""
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
        self.pcg_timings = self._tm = _Timings()
        b = self._apply(tod=_chunks(tod))
        # the deviations from the means of the priors, x = xp + x': A x' = b + precision * mean - A xp
        a0 = self._nsky + self.nflag
        shared_mean = self._project_shared(self._shprior[1].copy()) if self._nsh_loc else np.empty(0)
        mean = np.concatenate([self._prior[1], shared_mean])
        shifted = int(np.any(mean))
        if (shifted if self.comm is None else self.comm.allreduce(shifted)) > 0:
            xp = np.zeros_like(b)
            xp[a0:] = mean
            b -= self._apply(xp, exact=True, prior=False)  # the TOD minus the templates at the means
            if self._nsh_loc:  # the part of the means out of the constraints
                b[a0 + self.namp :] += self._project_shared(self._shprior[0] * (self._shprior[1] - shared_mean))
            del xp  # only the amplitude means are needed during the PCG
        x = np.zeros_like(b)
        if x0 is not None:
            x[: self._nsky] = (x0[:, self.lo : self.hi] * self.mask).ravel()
        x, it, history = self._pcg(b, x, x0 is not None, tol, maxiter, recompute_every, verbose)
        x[a0:] += mean
        a0 = self._nsky + self.nflag
        shared = x[len(x) - self._nsh_loc :].copy() if self._root else None
        info = {"iterations": it, "converged": history[-1] <= tol, "residuals": np.array(history),
                "amplitudes": x[a0 : a0 + self.namp].copy(),
                "shared_amplitudes": shared if self.comm is None else self.comm.bcast(shared)}
        self._free()
        # full mask now, while all ranks are here, so that to_healpix is not collective
        self._full_mask = None
        self._full_mask_of()
        return self.full(x[: self._nsky].reshape(self.ncomp, self.nloc)), info

    def shared_covariance(self, tol=1e-10, maxiter=2000, recompute_every=10, verbose=False):
        """Covariance of the shared template amplitudes, marginalized over the
        maps, the flagged samples and the local amplitudes: the shared block of
        A^-1, with A the operator of solve() (with `shared_groups`, of its
        pseudo-inverse on the constraints: the covariance of the projected
        amplitudes). One PCG solve per shared amplitude, A x = e_j, with the
        current templates. (nshared, nshared) on every rank; collective. Warns
        if some of the solves do not converge within `maxiter`."""
        n = self._nsky + self.nflag + self.namp + self._nsh_loc
        self.pcg_timings = self._tm = _Timings()
        cov = np.zeros((self.nshared, self.nshared))
        failed = []
        for j in range(self.nshared):
            b = np.zeros(n)
            if self._root:
                b[n - self.nshared + j] = 1.0
                self._project_shared(b[n - self.nshared :])
            x, _, history = self._pcg(b, np.zeros(n), False, tol, maxiter, recompute_every, verbose)
            if history[-1] > tol:
                failed.append(j)
            if self._root:
                cov[:, j] = x[n - self.nshared :]
        self._free()
        if failed and (self.comm is None or self.comm.rank == 0):
            warnings.warn(f"shared_covariance: the solves of columns {failed} did not converge in {maxiter} "
                          "iterations (raise maxiter or tol)", stacklevel=2)
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
        if bb == 0:  # b = 0: the solution is x = 0
            x[:] = 0.0
            return x, 0, [0.0]
        r = b - self._apply(x, exact=True) if start else b.copy()  # A 0 = 0: no product
        z = self._precond_flat(r)
        d = z.copy()
        delta = self._dot(r, z)
        history = [self._dot(r, r) / bb]
        true_residual = True  # r was computed as b - A x in float64

        it = 0
        t = time.perf_counter()
        while it < maxiter and history[-1] > tol:
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
                now = time.perf_counter()
                print(f"iter {it:4d}  |r|^2/|b|^2 = {history[-1]:.3e}  ({now - t:.1f} s)", flush=True)
                t = now
        return x, it, history

    def white_weights(self):
        """The white-noise weight of each chunk of this rank, 1 / sigma^2 for
        white noise of rms sigma per sample: the median of its inverse PSD
        (`weights`) over the upper half of the frequencies, where 1/f noise
        is negligible (relative error ~ (fknee / (fs / 4))^alpha). The
        common modes are not included."""
        return np.array([float(np.median(np.asarray(w, dtype=float)[len(w) // 2 :])) for w in self.weights])

    def binned(self, tod, weights="equal"):
        """Binned map (P^T W P)^-1 P^T W tod, (ncomp, nobs) on every rank, for
        timelines laid out like `pix`, with W one weight per chunk: "equal"
        (default: W = 1, all the samples alike), "noise" (the white-noise
        weights of the chunks, 1 / sigma^2, see white_weights: the optimal
        binned map with white noise of different levels) or an array of one
        weight > 0 per chunk of this rank (e.g. the same weight for the
        detectors of a pair). I only where Q/U are not solved (no gradient
        where it is not), 0 where not observed; flagged samples are not
        used. Collective with MPI."""
        tod = _chunks(tod)
        if isinstance(weights, str):
            if weights not in ("equal", "noise"):
                raise ValueError(f"binned: weights must be 'equal', 'noise' or one weight per chunk, not {weights!r}")
            scale = None if weights == "equal" else self.white_weights()
        else:
            scale = np.asarray(weights, dtype=float)
            if scale.shape != (len(self.ip),) or not np.all(np.isfinite(scale) & (scale > 0)):
                raise ValueError(f"binned: weights must be one finite weight > 0 per chunk ({len(self.ip)} on this rank)")
        # the same samples as the setup: its coverage if not weighted
        cov = self._cov if scale is None else self._coverage_local(scale)
        rhs = _NodeSum(self, self.ncomp, len(self.ip)) if self._node_maps else np.zeros((self.nobs, self.ncomp))
        flag = np.zeros(self.nflag)  # not used
        with numba_threads(self.nthreads):
            for k, ip in enumerate(self.ip):
                pol, c2, s2 = self._pol_arrays(k)
                t = np.asarray(tod[k], dtype=float)
                if scale is not None:
                    t = t * scale[k]
                self._pt(rhs, k, t, c2, s2, pol, flag)
        rhs = self._pixel_sums(rhs)
        out = np.zeros((self.ncomp, self.nloc))
        i_only = self.mask[0] & ~self.mask[1:].any(axis=0)
        out[0, i_only] = rhs[i_only, 0] / cov[i_only, 0]
        more = self.mask[0] & ~i_only
        if more.any():  # the solved components only: the others as identity rows
            M = self._white(cov, more)
            off = ~self.mask[:, more].T
            M[off[:, :, None] | off[:, None, :]] = 0.0
            M[off[:, :, None] & np.eye(self.ncomp, dtype=bool)] = 1.0
            out[:, more] = np.linalg.solve(M, np.where(off, 0.0, rhs[more])[..., None])[..., 0].T
        return self.full(out)

    def to_healpix(self, m, nside, fill=np.nan):
        """Expand a full map (ncomp, nobs) to RING maps (ncomp, 12 nside^2) of this
        rank (not shared by the node, unlike the maps of make_maps); labels
        beyond the map (virtual pixels) are dropped. Collective with MPI only if
        called before solve()."""
        unseen = ~self._full_mask_of()
        full = np.full((m.shape[0], 12 * nside * nside), fill)
        n = min(full.shape[1], self.nobs)
        full[:, :n] = m[:, :n]
        full[:, :n][unseen[:, :n]] = fill
        return full
