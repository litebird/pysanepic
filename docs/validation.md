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

## Conditional-fill solvers and high-order ACT cuts

`test_gapfill.py` checks conditional means and sampled covariances against
independent dense Gaussian formulae. Long-gap tests compare preconditioned
and ordinary CG with the same random RHS, including common modes, odd and
extended FFT grids, zero DC, and unchanged observed samples. The sine
preconditioner is checked for symmetry and positive definiteness.

`test_act_flags.py` compares the high-order cut operator and its own-noise
diagonal against a dense reference built from the symmetric Hahn difference
operator. Orthogonality and the transpose are checked up to 127 coefficients
on 128 samples, with separate tests for the zero-DC constant. The dense GLS
comparisons include chunk workers, templates and detector common modes.

For larger timing and stability checks, run on the cluster:

```bash
python tools/benchmark_flag_solvers.py --ns 262144 --nthreads 4 --output benchmark.json
```

It compares ordinary CG and the sine preconditioner on dispersed flags,
long gaps, steep 1/f noise and detector common modes, using identical seeded
conditional draws. It also checks matrix-free $C^T C$ at near-full ACT order
on 512 and 2048 samples. JSON output records spectral parameters, flag
counts, draw seeds, tolerance, iterations, true residuals and timings.
These are solver checks, not a
validation of LiteBIRD map bias or low-multipole BB residuals.

G100, 2026-10-31: 262144 samples per detector, 19 Hz, four threads,
`fmin_hz=1e-5`, squared relative residual threshold `1e-16`, NumPy 2.5.4
and ducc0 0.42.0. The valid residuals are synthetic unit Gaussian data;
both solvers condition on the same input and use the same seeded draw.
Long-gap cases add a run of 16384 missing samples to 3% dispersed flags:

| case | alpha | fknee [Hz] | CG iterations | PCG iterations | CG / PCG time [s] |
|---|---|---|---|---|---|
| dispersed flags | 1 | 0.025 | 5 | 5 | 0.136 / 0.079 |
| long gap | 2 | 0.1 | 289 | 7 | 2.798 / 0.134 |
| steep noise, long gap | 2 | 1 | not converged at 2000 | 11 | 18.435 / 0.158 |
| two detectors, common mode | 2 | 0.1 | 634 | 12 | 12.715 / 0.385 |

Times are single-run wall times including setup; first-use FFT planning can
affect the first case. Where both solvers converge, the relative differences
of the filled values are 1.9e-16, 1.5e-8 and 6.2e-8 respectively; both meet
the same true-residual criterion. High-order ACT checks give relative errors
in $C^T C a-a$ of 4.0e-15 (511 coefficients on 512 samples) and 5.4e-15
(2047 on 2048), without a stored basis matrix.

The full serial/thread suite passes 185 tests with one default thread,
including ACT calibration with 32 coefficients, and the flag/fill MPI tests
pass on two processes. With four default threads, two existing tests
(`test_pixel_input_and_generator`, `test_covariance_fixed_gains`) fail on
bitwise-equality assertions, also on the unmodified base commit `d4f86c4`.
These failures were reproduced separately and were not removed or relaxed.

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

## Calibration and beam ellipticity

End-to-end test with litebird_sim (M1_L, 37.5 Hz, the systematics of the
example e2e runs: bandpass, elliptical Gaussian beams, polarization angles,
1/f noise, gain drifts, pointing offsets), the beam asymmetries as a fixed
template ({func}`~pysanepic.blm_templates`), all the other templates fitted
with the maps and the gains, `fwhm_template` with `sum_zero=False`. Beams
from litebird_sim's `gauss_beam_to_alm`, the sky model through the circular
beam of the mean second moment of the detectors. With the beams minus their
own symmetric parts (no `model_bl`), the absolute calibration (fit / true
gain, averaged) is biased, by an amount that does not depend on the length
of the run and grows with the ellipticities $e$ of the beams:

| detectors (32 days unless noted) | ellipticities | $\langle (e-1)^2 \rangle$ | fit / true − 1 | common `fwhm` [rad²] |
|---|---|---|---|---|
| 4 | 1.008–1.018 | 1.8e-4 | −4e-5, −1e-5 | +2e-9 |
| 8 | up to 1.09 | 2.2e-3 | −2.3e-4 | −5e-9 |
| 16 | up to 1.19 | 1.2e-2 | −9.8e-4 | −3.5e-8 |
| 16, 1 year | up to 1.19 | 1.2e-2 | −9.8e-4 | −3.5e-8 |
| 16, 1 year, `fwhm` sum-zero | up to 1.19 | 1.2e-2 | +7.0e-4 | (0) |
| 16, circular horns | 1.00–1.02 | 1e-4 | −4e-5 | +1.5e-9 |
| 4 | 1.15–1.17 | 2.5e-2 | −2.0e-3 | −7.2e-8 |
| 4, **`model_bl`** | 1.15–1.17 | 2.5e-2 | **−4e-6** | +3.8e-9 |

fit / true − 1 ≈ −0.08 ⟨(e − 1)²⟩. The symmetric part of an elliptical
beam is not the Gaussian of its FWHM beyond order $\ell^2$; the rest, which
`fwhm_template` cannot follow, goes into the absolute gain, traded against
the common `fwhm` amplitude ($\delta g \approx \ell^2\,\delta\sigma^2/2$ at
the scales of the CMB, $\ell \sim 240$). The relative gains are not affected
(1.7e-5 rms in all the runs) and neither is Q/U, but the I maps carry the
error of the absolute gain times the solar dipole (3.2 µK for 16 detectors
over 1 year). With `model_bl` the fixed template is the true beam minus the
circular beam of the sky model: the calibration is right and the fitted
`fwhm` amplitudes are those of the pointing offsets alone,
$\langle|\rho|^2\rangle/2 = 3.4\times10^{-9}$ rad². Noiseless version:
`test_blm_fixed_template_model_bl` (4 detectors, $e$ = 1–1.3: gains exact
to 7e-13 with `model_bl`, off by 3e-3 without).

