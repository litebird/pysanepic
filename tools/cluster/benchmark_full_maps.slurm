#!/bin/bash
# full_maps="rank" vs "node" at nside 2048 (issue #84), one version, in the same
# job: 1 node x 48 ranks x 1 thread and 1 node x 8 ranks x 6 threads, the same
# samples in all (48 x 4 x 2^20), each pair REPEATS times, interleaved (the time
# per iteration varies a lot between jobs). Layout of the submit directory:
# src/ (a snapshot of pysanepic), tools/benchmark_node_memory.py, SHA256SUMS.
#SBATCH --account=INF26_lspe_1
#SBATCH --partition=g100_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --job-name=pysanepic_full_maps
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set +eu  # /etc/bashrc and the conda activation scripts fail under -e and -u
shopt -s expand_aliases
source /g100/home/userexternal/lpagano0/.bashrc
load_lbs_e2e
set -eu
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_CACHE_DIR="$PWD/numba_cache"
sha256sum --check SHA256SUMS
echo "pysanepic $(cat src/COMMIT) on ${SLURM_JOB_NODELIST}"

# srun warns that --cpus-per-task 6 exceeds that of the job (1): the node is
# exclusive and the steps run anyway
run() {  # label ranks threads chunks full_maps
    echo "== $1: $2 ranks x $3 threads, full_maps=$5"
    OMP_NUM_THREADS=$3 NUMBA_NUM_THREADS=$3 srun --nodes=1 --ntasks=$2 --cpus-per-task=$3 --cpu-bind=cores \
        python tools/benchmark_node_memory.py --src src --label "$1" --nside 2048 --chunks $4 --ns 1048576 \
        --nthreads $3 --iterations 10 --full-maps $5 --output-dir results
}

run warmup 8 6 24 node  # fills the numba cache
for rep in $(seq 1 "${REPEATS:-2}"); do
    for fm in rank node; do
        run "${fm}_48x1_${rep}" 48 1 4 "$fm"
        run "${fm}_8x6_${rep}" 8 6 24 "$fm"
    done
done

python - <<'EOF'
import glob, json
import numpy as np
rows = [json.load(open(f)) for f in glob.glob("results/*.json") if "warmup" not in f]
for r in sorted(rows, key=lambda r: r["label"]):
    pcg = r.get("pcg_timings_rank0", {})
    peak = max(v["pcg_pss_gb"] for v in r["per_node_peak_pss_gb"].values())
    print(f"{r['label']:12s} setup {r['setup_s']:6.1f} s  {r['s_per_iteration']:6.2f} s/it  PCG PSS {peak:6.1f} GB  "
          + ", ".join(f"{k} {v:.1f}" for k, v in pcg.items()))
ref = {}
for f in sorted(glob.glob("results/*_maps.npy")):
    if "warmup" in f:
        continue
    key = f.split("/")[-1].split("_")[1]  # layout
    ref.setdefault(key, []).append(np.load(f))
for layout, ms in ref.items():
    print(layout, "max rel diff between runs", max(np.abs(m - ms[0]).max() / np.abs(ms[0]).max() for m in ms))
EOF
