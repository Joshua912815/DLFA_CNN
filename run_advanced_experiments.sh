#!/usr/bin/env bash
set -u

cd /root/autodl-tmp/project1_instruction

PYTHON=/root/miniconda3/bin/python
COMMON_ARGS=(
  --epochs 20
  --batch-size 128
  --num-workers 8
  --lr 3e-4
  --optimizer adamw
  --weight-decay 1e-4
  --scheduler cosine
  --min-lr 1e-5
  --grad-clip 1.0
  --early-stop-patience 4
  --val-split time
  --val-ratio 0.2
)

run_step() {
  local name="$1"
  shift
  echo "===== ${name} START $(date '+%Y-%m-%d %H:%M:%S') ====="
  "$@"
  local status=$?
  echo "===== ${name} END status=${status} $(date '+%Y-%m-%d %H:%M:%S') ====="
  return "${status}"
}

run_step "advanced_resnet_stable" \
  "${PYTHON}" main.py \
  --model advanced \
  "${COMMON_ARGS[@]}" \
  --label-smoothing 0.03 \
  --run-name advanced_resnet_stable

run_step "chart_resnet18_gn_datasetnorm" \
  "${PYTHON}" main.py \
  --model chart_resnet18_gn \
  "${COMMON_ARGS[@]}" \
  --label-smoothing 0.03 \
  --normalization dataset \
  --run-name advanced_chart_resnet18_gn_datasetnorm

run_step "chart_resnet18_gn_lightaug" \
  "${PYTHON}" main.py \
  --model chart_resnet18_gn \
  "${COMMON_ARGS[@]}" \
  --label-smoothing 0.03 \
  --augmentation light \
  --run-name advanced_chart_resnet18_gn_lightaug

run_step "chart_resnet18_gn_no_smoothing" \
  "${PYTHON}" main.py \
  --model chart_resnet18_gn \
  "${COMMON_ARGS[@]}" \
  --run-name advanced_chart_resnet18_gn_no_smoothing

run_step "evaluate_all_checkpoints_on_test" \
  "${PYTHON}" evaluate_checkpoints.py \
  --runs-dir runs \
  --data-dir project1/image \
  --batch-size 256 \
  --num-workers 8

run_step "summarize_results" \
  "${PYTHON}" summarize_results.py \
  --runs-dir runs \
  --format csv \
  --output runs/summary.csv

"${PYTHON}" summarize_results.py --runs-dir runs --format markdown --output runs/summary.md
