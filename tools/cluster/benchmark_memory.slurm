#!/bin/bash
#SBATCH --account=INF26_lspe_1
#SBATCH --partition=g100_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=pysanepic_memory
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set +eu
shopt -s expand_aliases
source /g100/home/userexternal/lpagano0/.bashrc
load_simteam
set -eu
export OMP_NUM_THREADS=48
export NUMBA_NUM_THREADS=48
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$PWD/mpl_cache"
mkdir -p results
sha256sum --check SHA256SUMS
if [[ "${1:-full}" == timing ]]; then
    srun --ntasks=1 --cpus-per-task=48 --cpu-bind=cores python tools/benchmark_memory.py \
        --baseline-src baseline/src --optimized-src optimized/src \
        --output-dir results --nthreads 48 --repeats 3 \
        --cases uniform heterogeneous64 --operator-products 80
    exit
fi
srun --ntasks=1 --cpus-per-task=4 --cpu-bind=cores python tools/benchmark_memory.py \
    --baseline-src baseline/src --optimized-src optimized/src \
    --output-dir results/smoke --nthreads 4 --repeats 1 --quick
srun --ntasks=1 --cpus-per-task=48 --cpu-bind=cores python tools/benchmark_memory.py \
    --baseline-src baseline/src --optimized-src optimized/src \
    --output-dir results --nthreads 48 --repeats 2
export PYTHONPATH="$PWD/optimized/src"
export NUMBA_CACHE_DIR="$PWD/results/numba_cache/optimized"
srun --ntasks=1 --cpus-per-task=48 --cpu-bind=cores python -m pytest \
    tests/test_memory.py tests/test_threads.py -q --tb=short --junitxml=results/memory_tests.xml
OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 srun --ntasks=4 --cpus-per-task=12 --cpu-bind=cores \
    python -m pytest tests/test_mpi.py -k 'priors_mpi or more_ranks_than_pixels' \
    -q --tb=short
