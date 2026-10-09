"""GLS map-making with correlated noise (the SANEPIC algorithm).

Solves (P^T N^-1 P) m = P^T N^-1 d with preconditioned conjugate gradients.
N^-1 is applied in Fourier space, independently on each data chunk
(one detector x one segment), as a circulant filter.

Data layout: every per-sample array has shape (nchunk, ns); noise weights
have shape (nchunk, ns//2+1) and are defined so that the filter is
irfft(rfft(t) * w).
"""

import numba
import numpy as np
import scipy.fft


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
    pix : int array (nchunk, ns)
        HEALPix pixel index of each sample (any scheme; only used as labels).
    psi : float array (nchunk, ns) or None
        Polarization angle of each sample. None for intensity only.
    weights : float array (nchunk, ns//2+1)
        Inverse noise power spectrum of each chunk on the rfft grid.
    min_pol_hits : int
        Q/U are not solved in pixels with fewer hits (degenerate).
    comm : mpi4py communicator or None
        With MPI, each rank passes only its own chunks; maps are replicated on
        all ranks and P^T is summed with Allreduce.
    """

    def __init__(self, pix, psi, weights, min_pol_hits=4, comm=None):
        self.comm = comm
        local = np.unique(pix)
        self.pixels = local if comm is None else np.unique(np.concatenate(comm.allgather(local)))
        self.ip = np.searchsorted(self.pixels, pix)
        self.nobs = self.pixels.size
        self.weights = weights
        self.pol = psi is not None
        self.ncomp = 3 if self.pol else 1
        if self.pol:
            self.cos2 = np.cos(2 * psi)
            self.sin2 = np.sin(2 * psi)

        self.hits = self._sum(np.bincount(self.ip.ravel(), minlength=self.nobs))
        self.mask = np.ones((self.ncomp, self.nobs), dtype=bool)
        if self.pol:
            self.mask[1:] = self.hits >= min_pol_hits

        # Jacobi preconditioner. Like SANEPIC, Q/U use diag_I / 2, which
        # assumes uniform angle coverage.
        c = scipy.fft.irfft(weights, n=pix.shape[-1], axis=-1)
        diag = self._sum(_diag_circulant(self.ip, c, self.nobs))
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
        """P m: map (ncomp, nobs) -> timelines (nchunk, ns)."""
        t = m[0][self.ip]
        if self.pol:
            t = t + m[1][self.ip] * self.cos2 + m[2][self.ip] * self.sin2
        return t

    def backproject(self, t):
        """P^T t: timelines -> map (ncomp, nobs)."""
        ip = self.ip.ravel()
        out = [np.bincount(ip, t.ravel(), minlength=self.nobs)]
        if self.pol:
            out.append(np.bincount(ip, (t * self.cos2).ravel(), minlength=self.nobs))
            out.append(np.bincount(ip, (t * self.sin2).ravel(), minlength=self.nobs))
        return self._sum(np.array(out)) * self.mask

    def apply_A(self, m):
        return self.backproject(_filter(self.project(m), self.weights))

    def rhs(self, tod):
        return self.backproject(_filter(tod, self.weights))

    def solve(self, tod, x0=None, tol=1e-15, maxiter=2000, recompute_every=10, verbose=False):
        """Run PCG. Stops when |r|^2 / |b|^2 < tol (same criterion as SANEPIC).

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

        return x, {"iterations": it, "residuals": np.array(history)}

    def to_healpix(self, m, nside, fill=np.nan):
        """Expand (ncomp, nobs) to full RING maps (ncomp, 12 nside^2)."""
        full = np.full((m.shape[0], 12 * nside * nside), fill)
        full[:, self.pixels] = np.where(self.mask, m, fill)
        return full
