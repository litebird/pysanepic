#!/bin/bash
# Memory per node of estimate_noise with the sub-pixel terms at nside 2048
# (issue #86): main vs branch, 1 node x 48 ranks, REPEATS times interleaved.
# Layout of the submit directory: main/src, branch/src (snapshots),
# tools/benchmark_node_memory.py, tools/benchmark_subpixel_noise.py, SHA256SUMS.
#SBATCH --account=INF26_lspe_1
#SBATCH --partition=g100_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --job-name=pysanepic_subpixel_noise
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set +eu  # /etc/bashrc and the conda activation scripts fail under -e and -u
shopt -s expand_aliases
source /g100/home/userexternal/lpagano0/.bashrc
load_lbs_e2e
set -eu
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_CACHE_DIR="$PWD/numba_cache"
sha256sum --check SHA256SUMS
echo "main $(cat main/src/COMMIT), branch $(cat branch/src/COMMIT) on ${SLURM_JOB_NODELIST}"

srun --ntasks=48 --cpu-bind=cores python tools/benchmark_subpixel_noise.py --src branch/src --label warmup --output-dir results
for rep in $(seq 1 "${REPEATS:-2}"); do
    for v in main branch; do
        echo "== ${v}_${rep}"
        srun --ntasks=48 --cpu-bind=cores python tools/benchmark_subpixel_noise.py --src "$v/src" --label "${v}_${rep}" \
            --output-dir results
    done
done

python - <<'PY'
import glob, json
import numpy as np
for f in sorted(glob.glob("results/*.json")):
    r = json.load(open(f))
    if r["label"] != "warmup":
        node = max(v["estimate_pss_gb"] for v in r["per_node_peak_pss_gb"].values())
        print(f"{r['label']:9s} estimate_noise {r['estimate_s']:6.1f} s, peak PSS per node {node:6.1f} GB")
a, b = np.load("results/main_1_psd.npy"), np.load("results/branch_1_psd.npy")
print("psd max |branch - main| / max |main|", np.abs(a - b).max() / np.abs(a).max())
PY
