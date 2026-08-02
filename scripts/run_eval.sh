#!/bin/bash
# Evaluate CDIS per-point 3D instance predictions (class-agnostic AP / AP50 / AP25).
#
# Usage: bash scripts/run_eval.sh [PRED_DIR] [GT_DIR]
set -e

PYTHONPATH=./:$PYTHONPATH
export PYTHONPATH

DEFAULT_PRED_DIR="output/scannet200/VX0.03_IT1_Q5_ITH0.75/save_inst_seg_pcd/cropformer_sup_depth"
DEFAULT_GT_DIR="data/scannetv2/instance_gt/validation"

PRED_DIR=${1:-$DEFAULT_PRED_DIR}
GT_DIR=${2:-$DEFAULT_GT_DIR}

if [ ! -d "$GT_DIR" ]; then
  echo "[ERROR] Ground-truth directory not found: $GT_DIR"
  echo "        Pass it explicitly: bash scripts/run_eval.sh <PRED_DIR> <GT_DIR>"
  echo "        See docs/DATA.md for the expected instance-GT layout."
  exit 1
fi

echo "[INFO] Evaluating predictions in: $PRED_DIR"
echo "[INFO] Against ground truth in:   $GT_DIR"

python evaluation/run_class_agnostic_eval.py --pred_dir "$PRED_DIR" --gt_dir "$GT_DIR"
