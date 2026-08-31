#!/bin/bash

set -euo pipefail

LABEL="${1:-baseline}"
if [[ "$LABEL" != "baseline" && "$LABEL" != "final" && ! "$LABEL" =~ ^iter-[0-9]+$ ]]; then
    echo "usage: $0 [baseline|iter-N|final]" >&2
    exit 2
fi

WORKSPACE="/share/home/liuyao/workspace/bag"
PYTHON_BIN="/share/home/liuyao/miniconda3/envs/kernelagent/bin/python"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
TRAJ_DIR="$WORKSPACE/trajectory/${LABEL}-${TIMESTAMP}"
mkdir -p "$TRAJ_DIR"

cd "$WORKSPACE"
export PYTHONNOUSERSITE=1
export BAG_TRITON_IMPL_ROOT="$WORKSPACE/solution"
export TRITON_CACHE_DIR="/tmp/$USER/bag-triton-opt-cache"

srun \
    --partition=3090 \
    --nodelist=g06 \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=12G \
    --time=00:15:00 \
    env \
        PYTHONNOUSERSITE="$PYTHONNOUSERSITE" \
        BAG_TRITON_IMPL_ROOT="$BAG_TRITON_IMPL_ROOT" \
        TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
        "$PYTHON_BIN" -s scripts/benchmark_triton_bag.py \
            --layout column-major row-major \
            --precision ieee tf32 \
            --size 512 1024 2048 4096 8192 \
            --warmup-ms 20 \
            --rep-ms 80 \
            --schedule seven-stream \
            --output "$TRAJ_DIR/results.csv" \
    2>&1 | tee _bench_output.txt

cp -r solution/* "$TRAJ_DIR/"
cp _bench_output.txt "$TRAJ_DIR/bench-output.txt"
echo "TRAJECTORY=$TRAJ_DIR"
