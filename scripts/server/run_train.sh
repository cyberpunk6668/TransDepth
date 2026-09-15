#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <t0|h1|h2> [extra transdepth.cli.train arguments...]" >&2
  exit 2
fi

mode="$1"
shift
case "$mode" in
  t0|h1|h2) ;;
  *) echo "mode must be t0, h1, or h2" >&2; exit 2 ;;
esac

root="/home/hengxianli/TransDepth"
storage="/ssd/polyu/TransDepth"
python="$storage/envs/far/bin/python"
export CUDA_VISIBLE_DEVICES="0,3,5"
export PIP_CACHE_DIR="$storage/cache/pip"
export TORCH_HOME="$storage/cache/torch"
export HF_HOME="$storage/cache/huggingface"
export OMP_NUM_THREADS=1

cd "$root"
exec "$python" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=3 \
  -m transdepth.cli.train \
  --config "configs/experiments/$mode.yaml" \
  --paths configs/paths.local.yaml \
  "$@"
