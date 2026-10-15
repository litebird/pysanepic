# pysanepic

GLS map-making with correlated (1/f) noise for CMB experiments, in Python.

pysanepic is a Python port of [SANEPIC](https://github.com/patanch/SANEPIC)
([Patanchon et al. 2008, ApJ 681, 708](https://arxiv.org/abs/0711.3462)). It
solves the generalized least-squares problem

$$
(P^T N^{-1} P)\, m = P^T N^{-1} d
$$

with preconditioned conjugate gradients, applying $N^{-1}$ in Fourier space on
chunks of data, and returns I, Q, U HEALPix maps.

- Designed for experiments **without HWP**, where 1/f noise affects both
  intensity and polarization. An ideal HWP is supported too.
- Pure Python with numba and ducc0; runs with MPI (mpi4py) and threads.
- Supports polarization efficiency and any coordinate system.
- Independent of any simulation pipeline: data are passed as plain arrays.
  [litebird_sim](https://github.com/litebird/litebird_sim) provides a
  ready-made interface.
- Validated against the original C++ code, kept in the `cpp/` folder of the
  repository.

## Installation

```bash
pip install pysanepic
pip install "pysanepic[mpi]"                # with MPI
pip install git+https://github.com/litebird/pysanepic   # development version
```

Requirements: Python ≥ 3.10, numpy, healpy, numba, ducc0 (and mpi4py
for MPI).

```{toctree}
:maxdepth: 2
:caption: User guide

usage
parameters
algorithm
performance
low_level
validation
```

```{toctree}
:maxdepth: 1
:caption: Notebooks

notebooks/lbs_example
notebooks/preconditioner
notebooks/padding
notebooks/memory
notebooks/flags
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```
