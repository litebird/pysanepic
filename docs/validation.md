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
