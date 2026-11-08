#!/bin/bash
# Memory per node at nside 2048 of several versions (default: main, one map
# per rank, vs the branch, one per node), on 2 nodes x 24 ranks and 1 node x
# 48 ranks, plus the MPI tests of the branch on 2 real nodes. Layout of the
# submit directory: <version>/src for each version, branch/tests,
# branch/tools/simulate_sanepic_inputs.py, tools/benchmark_node_memory.py,
# SHA256SUMS. VERSIONS="main phase1 branch" sbatch ... to add versions.
#SBATCH --account=INF26_lspe_1
#SBATCH --partition=g100_usr_prod
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --job-name=pysanepic_node_memory
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set +eu
shopt -s expand_aliases
source /g100/home/userexternal/lpagano0/.bashrc
load_simteam
set -eu
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export NUMBA_CACHE_DIR="$PWD/numba_cache"
sha256sum --check SHA256SUMS

VERSIONS="${VERSIONS:-main branch}"
for v in $VERSIONS; do
    echo "== $v: 2 nodes x 24 ranks"
    srun --nodes=2 --ntasks=48 --ntasks-per-node=24 --cpu-bind=cores \
        python tools/benchmark_node_memory.py --src "$v/src" --label "${v}_2x24" --output-dir results
    echo "== $v: 1 node x 48 ranks"
    srun --nodes=1 --ntasks=48 --ntasks-per-node=48 --cpu-bind=cores \
        python tools/benchmark_node_memory.py --src "$v/src" --label "${v}_1x48" --output-dir results
done

VERSIONS="$VERSIONS" python - <<'EOF'
import os
import numpy as np
for v in os.environ["VERSIONS"].split()[1:]:
    for layout in ("2x24", "1x48"):
        a, b = (np.load(f"results/{x}_{layout}_maps.npy") for x in ("main", v))
        print(layout, f"max |{v} - main| / max |main| =", np.abs(a - b).max() / np.abs(a).max())
EOF

echo "== MPI tests of the branch, 2 ranks on each of 2 nodes"
# the tests share files through tempfile: TMPDIR on the shared file system
# (srun sets it to a directory local to the node)
mkdir -p tmp
cd branch/tests
PYTHONPATH="$SLURM_SUBMIT_DIR/branch/src" srun --nodes=2 --ntasks=4 --ntasks-per-node=2 --cpu-bind=cores \
    env TMPDIR="$SLURM_SUBMIT_DIR/tmp" python test_mpi.py
