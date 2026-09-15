#!/usr/bin/env bash
set -euo pipefail

repo="/home/hengxianli/TransDepth"
storage="/ssd/polyu/TransDepth"
environment="$storage/envs/far"
export CONDA_PKGS_DIRS="$storage/cache/conda"
export PIP_CACHE_DIR="$storage/cache/pip"
export TORCH_HOME="$storage/cache/torch"
export HF_HOME="$storage/cache/huggingface"

mkdir -p \
  "$storage"/{vendor,models,envs,runs,exports,logs} \
  "$storage/cache"/{conda,pip,torch,huggingface} \
  "$storage/data/derived/rftrans"/{manifests,cache,qa} \
  "$storage/data/raw/cleargrasp"

if [[ ! -x "$environment/bin/python" ]]; then
  conda create --prefix "$environment" python=3.11 pip -y
fi

"$environment/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.7.1 torchvision==0.22.1
"$environment/bin/python" -m pip install \
  --index-url https://pypi.org/simple \
  -r "$repo/requirements/base.txt" -r "$repo/requirements/dev.txt"
"$environment/bin/python" -m pip install --no-deps -e "$repo"
"$environment/bin/python" -m pip check

CUDA_VISIBLE_DEVICES=0 "$environment/bin/python" - <<'PY'
import torch

assert torch.cuda.is_available()
value = torch.ones(2, device="cuda").sum()
assert value.item() == 2
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
