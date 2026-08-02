#!/bin/bash

# Function to log and exit on errors
function error_exit {
  echo "[ERROR] $1"
  exit 1
}

# Function to log info
function log_info {
  echo "[INFO] $1"
}

PYTHONPATH=./:$PYTHONPATH
export PYTHONPATH

# === Default values ===
DEFAULT_CONFIG_TYPE="scannet200"
DEFAULT_GPU_NUM=0
DEFAULT_PART=0
DEFAULT_SUBPROCESS_NUM=1

# === Use input arguments if provided, otherwise fall back to default ===
CONFIG_TYPE=${1:-$DEFAULT_CONFIG_TYPE}
GPU_NUM=${2:-$DEFAULT_GPU_NUM}
PART=${3:-$DEFAULT_PART}
SUBPROCESS_NUM=${4:-$DEFAULT_SUBPROCESS_NUM}

# === Begin ===
log_info "Running CDIS with:"
log_info "  CONFIG_TYPE:      $CONFIG_TYPE"
log_info "  GPU_NUM:          $GPU_NUM"
log_info "  PART:             $PART"
log_info "  SUBPROCESS_NUM:   $SUBPROCESS_NUM"

# Run the main script
CUDA_VISIBLE_DEVICES=$GPU_NUM python CDIS/run.py \
  --config_path "./configs/${CONFIG_TYPE}.yaml" \
  --part $PART \
  --subprocess_num $SUBPROCESS_NUM

# Completion message
log_info "CDIS processing completed for dataset type: ${CONFIG_TYPE}"

# Optional: Post-processing reminder for scannet200
if [ "$CONFIG_TYPE" == "scannet200" ]; then
  log_info "Evaluate with: bash scripts/run_eval.sh <PRED_DIR> <GT_DIR>"
fi
