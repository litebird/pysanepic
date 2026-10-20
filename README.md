# pysanepic

GLS map-making with correlated (1/f) noise for CMB experiments, in Python.

pysanepic is a Python port of [SANEPIC](https://github.com/patanch/SANEPIC)
([Patanchon et al. 2008, ApJ 681, 708](https://arxiv.org/abs/0711.3462)). It solves the generalized least-squares
problem (PᵀN⁻¹P) m = PᵀN⁻¹d with preconditioned conjugate gradients, applying
N⁻¹ in Fourier space on chunks of data, and returns I, Q, U HEALPix maps.

- Designed for experiments **without HWP**, where 1/f noise affects both
  intensity and polarization. An ideal HWP is supported too.
- Pure Python with numba and ducc0; runs with MPI (mpi4py) and threads.
- Supports polarization efficiency and any coordinate system.
- Noise correlated between detectors (common modes of a wafer), with a
  two-level preconditioner for the large scales (`GLSParameters.coarse_nside`).
- Noise spectra, couplings and common modes estimated from the data
  (`estimate_noise`).
- Gain calibration on the dipole, fitted with the maps
  (`CalibrationParameters`), with a mask for the Galactic plane.
- Independent of any simulation pipeline: data are passed as plain arrays.
- Validated against the original C++ code, which is kept in `cpp/`.

**Documentation: https://pysanepic.readthedocs.io** (sources in [`docs/`](https://github.com/litebird/pysanepic/tree/main/docs)).

## Installation

```bash
pip install pysanepic
pip install "pysanepic[mpi]"                # with MPI
pip install git+https://github.com/litebird/pysanepic   # development version
```

## Usage

With [litebird_sim](https://github.com/litebird/litebird_sim) (on `master`, not
yet in a release):

```python
from pysanepic import GLSParameters

result = sim.make_sanepic_gls_map(nside=512, params=GLSParameters(chunk_s=86400.0, tol=1e-8))
result.maps       # (3, npix) I, Q, U; healpy.UNSEEN where not solved
```

With any other pipeline, one `DetectorData` per detector and stretch of time:

```python
from pysanepic import DetectorData, OneOverFNoise, Pointing, make_maps

# angles in any coordinate system, or Pointing(pix=..., nside=..., psi=..., coordinates=...)
pointing = Pointing(theta=theta, phi=phi, psi=psi, coordinates="E")
noise = OneOverFNoise(net_ukrts=40.0, fknee_hz=0.02, alpha=1.0, fmin_hz=1e-5)  # or TabulatedNoise(freqs_hz=..., psd=...)
data = [DetectorData(tod=tod, pointing=pointing, pol_angle_rad=0.0, sampling_rate_hz=19.0, noise=noise)]
result = make_maps(data, nside=64, coordinates="G")
```

See the documentation for the conventions, the parameters and how to choose
them, MPI, memory and benchmarks, and the [notebooks](https://github.com/litebird/pysanepic/tree/main/notebooks).

## Tests

```bash
pip install -e ".[test]"
make -C cpp ARCH=linux        # or ARCH=mac: builds the C++ reference
pytest                        # includes the comparison with C++ SANEPIC
mpirun -n 4 python tests/test_mpi.py
```

## Documentation

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

## License

Distributed under the [GPL3 License](https://github.com/litebird/pysanepic/blob/main/LICENSE).
