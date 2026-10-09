"""GLS map-making with correlated noise (the SANEPIC algorithm).

Solves (P^T N^-1 P) m = P^T N^-1 d with preconditioned conjugate gradients.
N^-1 is applied in Fourier space, independently on each data chunk
(one detector x one stretch of time), as a circulant filter.

Data layout: data come in *blocks*; a block is a 2D array (nchunk, ns) of
chunks with the same length ns. Pass one array for a single block or a list
of arrays for chunks of different lengths. Noise weights of a block have shape
(nchunk, ns//2+1) and are defined so that the filter is irfft(rfft(t) * w).
"""

import numba
import numpy as np
import scipy.fft


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


def _filter(t, w):
    ns = t.shape[-1]
    return scipy.fft.irfft(scipy.fft.rfft(t, axis=-1, workers=-1) * w, n=ns, axis=-1, workers=-1)


@numba.njit(cache=True)
def _diag_circulant(ip, c, nobs):
    """Exact diagonal of P^T C P for the I component, C circulant per chunk.

    For each pixel, sums c[(t - t') mod ns] over all sample pairs that fall in
    it. Cost is sum over pixels of hits^2 per chunk.
    """
    # ponytail: O(hits^2) per pixel per chunk; switch to a lag cutoff or the
    # SANEPIC N_[0] shortcut if deep pixels with long chunks become a bottleneck
    out = np.zeros(nobs)
    nchunk, ns = ip.shape
    for k in range(nchunk):
        order = np.argsort(ip[k], kind="mergesort")
        start = 0
        while start < ns:
            p = ip[k, order[start]]
            stop = start
            while stop < ns and ip[k, order[stop]] == p:
                stop += 1
            if p >= 0:
                acc = 0.0
                for a in range(start, stop):
                    ta = order[a]
                    for b in range(start, stop):
                        acc += c[k, (ta - order[b]) % ns]
                out[p] += acc
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
    pol_efficiency : float array (nchunk,), or list (one per block), or None
        Polarization efficiency of each chunk; None means 1.
    min_pol_hits : int
        Q/U are not solved in pixels with fewer hits (degenerate).
    comm : mpi4py communicator or None
        With MPI, each rank passes only its own chunks (possibly none); maps
        are replicated on all ranks and P^T is summed with Allreduce.
    """

    def __init__(self, pix, psi, weights, pol_efficiency=None, min_pol_hits=4, comm=None):
        pix, self.weights = _blocks(pix), _blocks(weights)
        self.comm = comm
        local = np.unique(np.concatenate([p.ravel() for p in pix] or [np.empty(0, np.int64)]))
        self.pixels = local if comm is None else np.unique(np.concatenate(comm.allgather(local)))
        self.ip = [np.searchsorted(self.pixels, p) for p in pix]
        self.nobs = self.pixels.size
        self.pol = psi is not None
        self.ncomp = 3 if self.pol else 1
        if self.pol:
            psi = _blocks(psi)
            eff = [1.0] * len(psi) if pol_efficiency is None else _blocks(pol_efficiency)
            g = [np.asarray(e, dtype=float).reshape(-1, 1) for e in eff]
            self.cos2 = [gb * np.cos(2 * s) for gb, s in zip(g, psi)]
            self.sin2 = [gb * np.sin(2 * s) for gb, s in zip(g, psi)]

        self.hits = self._sum(sum((np.bincount(ip.ravel(), minlength=self.nobs) for ip in self.ip), np.zeros(self.nobs, np.int64)))
        self.mask = np.ones((self.ncomp, self.nobs), dtype=bool)
        if self.pol:
            self.mask[1:] = self.hits >= min_pol_hits

        # Jacobi preconditioner. Like SANEPIC, Q/U use diag_I / 2, which
        # assumes uniform angle coverage and unit polarization efficiency.
        diag = np.zeros(self.nobs)
        for ip, w in zip(self.ip, self.weights):
            diag += _diag_circulant(ip, scipy.fft.irfft(w, n=ip.shape[-1], axis=-1), self.nobs)
        diag = self._sum(diag)
        if np.any(diag <= 0):
            raise ValueError("Preconditioner has non-positive elements: check noise weights")
        self.precond = np.empty((self.ncomp, self.nobs))
        self.precond[0] = 1.0 / diag
        if self.pol:
            self.precond[1:] = 2.0 / diag
        self.precond *= self.mask

    def _sum(self, x):
        """Sum a map-sized array over ranks (identity without MPI)."""
        if self.comm is None:
            return x
        out = np.empty_like(x)
        self.comm.Allreduce(np.ascontiguousarray(x), out)
        return out

    def project(self, m):
        """P m: map (ncomp, nobs) -> list of timeline blocks."""
        out = []
        for k, ip in enumerate(self.ip):
            t = m[0][ip]
            if self.pol:
                t = t + m[1][ip] * self.cos2[k] + m[2][ip] * self.sin2[k]
            out.append(t)
        return out

    def backproject(self, tod):
        """P^T t: timeline block(s) -> map (ncomp, nobs)."""
        out = np.zeros((self.ncomp, self.nobs))
        for k, (ip, t) in enumerate(zip(self.ip, _blocks(tod))):
            ip = ip.ravel()
            out[0] += np.bincount(ip, t.ravel(), minlength=self.nobs)
            if self.pol:
                out[1] += np.bincount(ip, (t * self.cos2[k]).ravel(), minlength=self.nobs)
                out[2] += np.bincount(ip, (t * self.sin2[k]).ravel(), minlength=self.nobs)
        return self._sum(out) * self.mask

    def _filter_all(self, tod):
        return [_filter(t, w) for t, w in zip(_blocks(tod), self.weights)]

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
        z = self.precond * r
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
            z = self.precond * r
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
        full[:, self.pixels] = np.where(self.mask, m, fill)
        return full
