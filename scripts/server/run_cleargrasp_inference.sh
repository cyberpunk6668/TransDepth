#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <trained-checkpoint.pt> <output-directory>" >&2
  exit 2
fi

storage="/ssd/polyu/TransDepth"
export CUDA_VISIBLE_DEVICES="0"
export TORCH_HOME="$storage/cache/torch"
export HF_HOME="$storage/cache/huggingface"

cd /home/hengxianli/TransDepth
exec "$storage/envs/far/bin/python" -m transdepth.cli.predict \
  --config configs/inference/predict.yaml \
  --paths configs/inference.paths.local.yaml \
  --checkpoint "$1" \
  --source cleargrasp \
  --output "$2"
