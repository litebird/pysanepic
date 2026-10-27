# Validation and tests

```bash
pip install -e ".[test]"
make -C cpp ARCH=linux        # or ARCH=mac: builds the C++ reference
pytest                        # includes the comparison with C++ SANEPIC
mpirun -n 4 python tests/test_mpi.py
```

- `test_vs_sanepic.py`: with SANEPIC's preconditioner, the maps agree with the
  C++ SANEPIC to ~1e-6; the block preconditioner converges to the same map in a
  fraction of the iterations; noiseless data give back the input sky.
- `test_mapmaker.py`: `make_maps` recovers the input sky to machine precision
  with HWP, polarization efficiency and coordinate rotation.
- `test_mpi.py`: MPI and serial runs agree.
- `test_threads.py`: results do not depend on the number of threads.

The tests run on GitHub Actions for every pull request. The comparison with
litebird_sim's binner and destriper is in the
[example notebook](notebooks/lbs_example).

## Covariance of the template amplitudes

`MapResult.template_covariance` is the exact inverse of the GLS operator for
the amplitudes (checked against a dense inverse in
`test_shared_templates_exact_gls`, and between MPI and serial runs). Whether
it describes the scatter of the fitted amplitudes is checked by a Monte Carlo
over noise realizations, `tools/template_covariance_mc.py`: 4 detectors at
nside 16, 12 000 s each, 1/f noise ($f_\text{knee}$ = 50 mHz, $\alpha = 1$)
drawn as a stationary process over the whole TOD while $N^{-1}$ is applied on
chunks of 1200 s; a foreground template (sum-zero amplitudes), a TOD per
detector and a TOD with one common amplitude. Standard deviation of the
amplitudes over the realizations, divided by the predicted one:

| | realizations | foreground (4) | thermal (4) | common (1) | statistical error |
|---|---|---|---|---|---|
| without calibration | 200 | 1.10, 1.10, 0.93, 0.99 | 1.00, 1.07, 1.09, 1.04 | 0.98 | 0.05 |
| with calibration (`n_iter=2`) | 100 | 1.18, 1.07, 0.94, 1.04 | 0.97, 1.07, 1.14, 1.09 | 1.03 | 0.08 |

The correlations agree within 0.13 (0.24 with calibration), for a
statistical error of 0.07 (0.10). The predicted errors are right to ~10%,
slightly optimistic on average (by 3%, 6% with calibration): the covariance
is that of the noise model, here without the correlations across chunks,
and with calibration of the linearized problem of the last iteration.

