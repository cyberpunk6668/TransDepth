# 03｜DINOv3 H/16+：官方下载、服务器配置与 FAR 特征接口

核查日期：2026-09-15。对应当前最小计划 M3、M5–M7、M12、M13。

## 1. 交付边界与明确选择

本章是实施规格与命令模板，**没有在远端服务器下载权重、通过访问许可、执行 GPU 前向或完成显存验收**。文中的官方工具命令在满足路径、网络、账户权限后可以执行；标有“待实现”的 `transdepth` 模块和测试需要后续编写。不要把参考代码存在视为模型已经配置成功。

本项目主线固定：**Meta 原生 DINOv3 实现＋官方 H/16+ LVD-1689M `.pth` 权重**。理由是 FAR 必须可靠访问选中 block 的 packed QKV、官方 RoPE 及实际 attention 输出路径。HF Transformers 路径保留为独立兼容方案，两个后端不在一组配对实验中混用。

| 项目 | 本项目取值 | 如何证明 |
|---|---|---|
| 论文型号 | DINOv3 ViT-H/16+，官方表格也写 H+/16 | 官方模型配置与 README |
| 原生 Hub 入口 | `dinov3_vith16plus` | `dinov3/hub/backbones.py` |
| 官方来源 | `facebookresearch/dinov3` | Meta GitHub 仓库 |
| 固定代码提交 | `6876159a11b4df116f30f667f8c9888617df0751` | 本次 GitHub API 核实；提交时间 2026-07-15 |
| 原生 checkpoint 文件名 | `dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth` | Hub 构造的文件名与下载批准后的实际文件 |
| 预训练来源 | LVD-1689M | 下载项、文件名与配置共同核验 |
| HF 模型 ID | `facebook/dinov3-vith16plus-pretrain-lvd1689m` | Facebook 官方模型仓库 |
| 本次 HF revision | `c807c9eeea853df70aec4069e6f56b28ddc82acc` | 本次 HF API 核实 |
| 模型规模 | 官方约 840M；实际参数量运行后记录 | 不能用 LoRA 参数量代替总模型量 |

代码 commit 是代码版本；HF revision 是 HF 仓库版本；文件名中的 `7c1da9a5` 是官方 checkpoint 短 hash 标识。**这三者均不是你下载文件的完整 SHA256**。完整 SHA256 必须在下载完成后计算，服务器迁移后重新计算。上述来源见 [固定版本 Hub 定义](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/hub/backbones.py)、[固定版本 README](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/README.md)、[HF 模型仓库](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m)。

## 2. 模型结构必须逐项验收

### 2.1 H/16+ 的真实配置

| 结构项 | 预期值 | 常见误用 |
|---|---:|---|
| patch 边长 | 16 | 使用 DINOv2 的 /14 |
| block 数 | 32 | 用 L 模型的 24 层 |
| embedding 宽度 | 1280 | 用 B 的 768 或 L 的 1024 |
| attention heads | 20 | 把 16 heads 当成 H |
| 每头通道 | 64 | 由 1280/20 得到 |
| storage/register tokens | 4 | 忘掉它们或当空间 token |
| CLS tokens | 1 | patch 网格应排除全部 5 前缀 |
| FFN | SwiGLU，`ffn_ratio=6.0` | 用普通 GELU MLP 替代 |
| SwiGLU 隐层 | 5120 | 7680 是传给门控模块的中间配置值，官方内部乘 2/3 |
| LayerNorm epsilon | `1e-5` | `layernormbf16` 是 eps 配置名称，并非自动把全网转成 BF16 |
| packed QKV | 有 bias，Q/K/V 连续分段 | 误判为三个独立 `q_proj/k_proj/v_proj` |
| Key bias | `mask_k_bias=True` | 直接调用 `F.linear(x, weight, raw_bias)` 会丢掉 mask |
| LayerScale | 存在，构造初始值 `1e-5` | 加载权重后不能重新初始化它 |
| Q/K norm | 本版本此路径没有额外 Q/K normalization | 不额外插入 L2 normalize/LayerNorm |
| stochastic depth | Hub 配置 `drop_path_rate=0` | 不自行引入 DropPath |

“+”不是随便加到模型名字上的后缀。使用通用 `vit_huge2`、timm 中近似名字或 DINOv2 权重，均不能视为本计划的官方 H/16+。SwiGLU 和归一化细节由 [ViT 实现](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/models/vision_transformer.py) 与 [FFN 实现](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/layers/ffn_layers.py) 核实。

### 2.2 RoPE、token 与 attention 的顺序

```text
block 输入 H
  → norm1(H) = X
  → packed QKV 线性投影（保留 Key bias mask）
  → 仅指定头的 Q/K 添加 LoRA 增量
  → reshape 为 [B, heads, N, head_dim]
  → 官方 apply_rope(Q, K, rope)
  → SDPA / oracle 的真实 AV 输出
  → 合并 heads → output projection → LayerScale → 第一次残差
  → norm2 → SwiGLU → LayerScale → 第二次残差
```

RoPE 的 base=100、`normalize_coords="separate"`、配置 dtype=FP32、`rescale_coords=2`。官方只在 `.training=True` 时随机缩放坐标；FAR 主线必须让骨干和 RoPE 一直处于 `eval()`，以保持同 RGB 的原始教师确定性。官方 RoPE 只旋转空间 patch；前 5 个 token 保留前缀处理。不要手写一个与官方不同的 2D 正弦位置编码替代它。[RoPE 源码](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/layers/rope_position_encoding.py)

这个版本的普通 self-attention 使用 PyTorch SDPA；不返回完整 attention。源码里的 `attn_drop` 字段不代表 FAR 应加入 DropKey；本项目不引入随机 Key 丢弃，也不屏蔽 padding attention keys。padding、CLS、storage 等仍留在完整前向中，但不进入可靠关系集合。`masks` 参数是 patch masking 接口，**不是透明 GT mask 的输入通道**。[Attention 源码](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/layers/attention.py)

## 3. 路线 A：官方下载原生 `.pth`，用于本项目主线

### 3.1 取得模型访问资格

1. 用户本人打开 [Meta DINOv3 下载申请页](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)，按真实信息提交申请并处理许可。
2. 批准后按官方 README 的流程收到模型链接；从中选择 **ViT-H+/16、LVD-1689M、backbone** 对应条目。
3. 使用收到的完整链接下载；若链接带签名参数，保留整个字符串。不要自行猜一个无签名 URL 再把 403 当成“模型不存在”。
4. 不下载 7B 的 DPT depth head 来代替本项目自己的 H＋DPT；官方 backbone 本身不输出本项目的米制深度。

这一步需要用户的真实身份和许可操作，代码不能代替。文档编写不依赖先完成这一步。官方原生下载与 HF gated 访问是不同入口；获得其中一个入口的访问资格，不等于另一个已经授权。[官方原生下载说明](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/README.md#pretrained-backbones-via-pytorch-hub)

### 3.2 Linux 服务器命令模板

以下为**可执行的官方工具模板**；先把 `/srv/transdepth` 改成自己拥有写权限的数据盘目录。源码与权重分开放，权重目录不进 Git。这是可独立运行的下载验收路径；若已经执行01章的submodule流程，将TD_DINO_REPO指向项目的third_party/dinov3并跳过clone，生产训练只保留这一份固定源码来源。建议保存为 Bash 脚本执行；`set -euo pipefail` 使下载或存在性检查失败时停止，避免继续重命名错误文件。

```bash
set -euo pipefail
export TD_STORAGE=/srv/transdepth
export TD_DINO_REPO="$TD_STORAGE/vendor/dinov3"
export TD_DINO_WEIGHTS="$TD_STORAGE/weights/dinov3"
export TD_DINO_FILE=dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
mkdir -p "$TD_STORAGE/vendor" "$TD_DINO_WEIGHTS"

# 仅在目标目录尚未存在时执行 clone；已存在时先检查 remote 和工作区状态。
git clone https://github.com/facebookresearch/dinov3.git "$TD_DINO_REPO"
git -C "$TD_DINO_REPO" checkout --detach 6876159a11b4df116f30f667f8c9888617df0751
git -C "$TD_DINO_REPO" rev-parse HEAD
git -C "$TD_DINO_REPO" status --porcelain

# 交互粘贴用户获准的完整下载链接；不把它写入仓库或 shell 历史。
read -r -s -p 'Authorized DINOv3 H/16+ URL: ' TD_DINO_URL
printf '\n'
printf '%s\n' "$TD_DINO_URL" | wget --continue --no-verbose --input-file=- \
  --output-document="$TD_DINO_WEIGHTS/$TD_DINO_FILE.part"
unset TD_DINO_URL
```

确认 wget 成功退出、文件非空且没有返回 HTML/XML 错误页后，执行以下步骤；若目标正式文件已存在，先核验已有文件，不覆盖它。

```bash
test -s "$TD_DINO_WEIGHTS/$TD_DINO_FILE.part"
test ! -e "$TD_DINO_WEIGHTS/$TD_DINO_FILE"
mv "$TD_DINO_WEIGHTS/$TD_DINO_FILE.part" "$TD_DINO_WEIGHTS/$TD_DINO_FILE"
cd "$TD_DINO_WEIGHTS"
sha256sum "$TD_DINO_FILE" > "$TD_DINO_FILE.sha256"
sha256sum --check "$TD_DINO_FILE.sha256"
stat --printf='%n %s bytes\n' "$TD_DINO_FILE"
```

`test -s` 仅证明非空，不能代替下一节的真实 tensor 加载、strict state_dict 和结构验收。将实际 SHA256、字节数、批准下载日期写入元数据；短 hash 前缀匹配可以辅助检查，但完整 SHA256 必须保留。

### 3.3 Windows 下载后迁移到 Linux

如果服务器无法直连 Meta，可在获准、可联网的机器下载，再传到服务器。下载物必须来自用户获准的官方入口。不要在 GitHub release 或项目 Git LFS 中公开转存。

Windows 的原生 PowerShell hash 模板：

```powershell
$tdWeight = 'D:\models\dinov3\dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth'
Get-Item -LiteralPath $tdWeight | Select-Object Name,Length
Get-FileHash -LiteralPath $tdWeight -Algorithm SHA256
scp $tdWeight user@server:/srv/transdepth/weights/dinov3/
```

Linux 端用 `sha256sum` 得到完整值，与 Windows 结果逐字相同才进入加载。离线源码可用固定提交的 Git bundle 或完整源码目录迁移；环境包必须按目标 Linux/Python/CUDA 组合准备，不能拷贝 Windows 虚拟环境当 Linux 环境。

## 4. 路线 B：HF 官方仓库，供独立 Transformers 兼容方案

### 4.1 明确文件格式边界

本次查询 [HF 官方模型 API](https://huggingface.co/api/models/facebook/dinov3-vith16plus-pretrain-lvd1689m) 返回 `gated="manual"`，revision 为 `c807c9eeea853df70aec4069e6f56b28ddc82acc`。该 revision 的文件包括：`model.safetensors`、`config.json`、`preprocessor_config.json`、README、LICENSE。**没有可直接交给 Meta `torch.hub.load(..., weights=...)` 的原生 `.pth`**。

`safetensors` 与 `.pth` 不是仅改扩展名就能互换。HF 的模块命名、QKV 表达、层结构接口与 Meta 原生实现需要专门映射；`strict=False` 忽略缺失键会产生混合随机模型，禁止用它“解决”格式错误。若未来选择转换，另写转换器、双后端逐层等价检查和转换 hash；本章不把未验证转换作为主线依赖。

### 4.2 授权与下载模板

先登录 [HF 模型页](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m)，由用户提交访问申请，等待该账户获批；再创建该账户的读取 token。`hf auth login` 是登录，不会自动授予 gated 模型访问权。[HF gated 模型规则](https://huggingface.co/docs/hub/models-gated)

以下为**可执行的 HF 官方工具模板**，应放在独立的下载/兼容环境中。项目主线不要求额外安装 Transformers。

```bash
python -m pip install huggingface_hub
hf auth login
hf auth whoami
export TD_HF_DINO=/srv/transdepth/weights/hf_dinov3_vith16plus
hf download facebook/dinov3-vith16plus-pretrain-lvd1689m \
  --revision c807c9eeea853df70aec4069e6f56b28ddc82acc \
  --local-dir "$TD_HF_DINO"
cd "$TD_HF_DINO"
sha256sum model.safetensors config.json preprocessor_config.json > SHA256SUMS
sha256sum --check SHA256SUMS
```

下载权限失败就处理账户和授权状态；不是换镜像或换模型 ID 的理由。下载目录保留 config 和 processor 文件。使用 `snapshot_download` 时同样传完整 `revision`，不要只复制一个缓存 symlink 后删除其底层 blob。[HF 下载与版本固定](https://huggingface.co/docs/huggingface_hub/guides/download)

原生主线不使用本段的 Transformers 模型。独立 HF smoke 检查可使用如下参考方式；Transformers 自 `4.56.0` 开始支持 DINOv3，但兼容方案的完整依赖须另锁文件并验收，不让 `pip install -U` 改动已验证主线环境。

```python
# 独立 HF 后端参考代码；前置条件：安装并锁定兼容 Transformers 版本。
import os
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    os.environ["TD_HF_DINO"], local_files_only=True
).eval().to("cuda")
# rgb 是主项目已处理的 [B,3,384,512] float 张量，不能重复归一化。
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    outputs = model(pixel_values=rgb.to("cuda"), output_hidden_states=True)
```

HF `hidden_states` 含 embedding 状态时，block 7 输出对应 `hidden_states[8]`（0-based 索引8）；具体接口必须在锁定后端核验。不要直接把原生 `n=[7,15,23,31]` 当作 HF 的所有数组索引。若希望主项目以 HF 后端启动，FAR attention 适配器也必须重写并通过同样身份验收，不能只替换加载函数。[DINOv3 官方 HF 支持说明](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/README.md#pretrained-backbones-via-hugging-face-transformers)

## 5. Linux 环境与可复现配置

### 5.1 环境起点

采用 Python 3.11、PyTorch 2.7.1、torchvision 0.22.1、CUDA 12.6 wheel，作为本项目待服务器验收的固定起点。PyTorch 官方提供这一组合；DINOv3 的训练 README 要求 PyTorch ≥2.7.1，并说明训练/评估预期 Linux。官方 `requirements.txt` 与 `conda.yaml` 没有给所有包固定版本，因此不能把它们称为本项目可复现实验锁文件。[PyTorch 历史版本安装表](https://pytorch.org/get-started/previous-versions/)、[DINOv3 环境文件](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/conda.yaml)

```bash
# 可执行的环境模板；完整工程依赖按 01 文档维护。
python3.11 -m venv /srv/transdepth/venvs/far
source /srv/transdepth/venvs/far/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install numpy pillow
python -m pip check
nvidia-smi
python -m torch.utils.collect_env
```

这些安装命令不证明服务器驱动支持所选 wheel。把 `nvidia-smi`、`torch.version.cuda`、GPU 型号、实际张量 CUDA 运算与 NCCL 小通信测试一起记录。系统 `nvcc` 版本、wheel 的 CUDA runtime 与 NVIDIA 驱动版本是三个不同对象；主线先避免增加需要源码编译的 FlashAttention/xFormers，使用官方 SDPA。

首轮先采用 FP32 参数＋autocast 的保守路径，RoPE 保持官方 FP32。不要立即对整个模型调用 `.half()`，它会转换浮点 buffer，可能降低 RoPE `periods` 的存储精度。需要降低冻结权重驻留量时，另立“冻结线性权重 BF16、RoPE 保持 FP32”的配置，做数值等价后再配对使用。

### 5.2 项目配置契约〔待实现〕

```yaml
# 规格：configs/backbone/dinov3_h16plus.yaml
backend: meta_native
hub_entry: dinov3_vith16plus
repo_dir: ${TD_DINO_REPO}
repo_commit: 6876159a11b4df116f30f667f8c9888617df0751
checkpoint_path: ${TD_DINO_WEIGHTS}/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
checkpoint_sha256: null # 下载验收后填写64位实际值；正式训练拒绝null
checkpoint_format: meta_pth_state_dict
pretrain_dataset: LVD1689M
input_hw: [384, 512]
feature_blocks_0based: [7, 15, 23, 31]
feature_norm: frozen_backbone_norm
feature_width: 1280
prefix_tokens: 5
patch_size: 16
backbone_eval_mode: true
backbone_weights_trainable: false
amp_dtype: bf16 # 由服务器支持与数值验收锁定；失败时统一切FP16
rope_dtype: fp32
rgb_mean: [0.485, 0.456, 0.406]
rgb_std: [0.229, 0.224, 0.225]
```

`${...}` 是设计中的环境变量解析约定，不是 PyYAML 自带功能。配置加载器显式展开并验证必填字段。正式运行保存展开后的 YAML；只保留本地路径的配置文件不进公开仓库。模型 ID、源码提交、权重 hash、预处理版本作为 checkpoint 身份的一部分保存。

## 6. 原生加载与结构验证：独立参考脚本

### 6.1 为什么加载路径分成两个例子

官方常用入口是：

```python
model = torch.hub.load(
    os.environ["TD_DINO_REPO"], "dinov3_vith16plus",
    source="local", weights=checkpoint_path,
)
```

这段是官方常用接口的说明，不是下方最小环境的验收入口。该commit的`hubconf.py`还顶层导入segmentors等任务模块，其import链需要`torchmetrics`等额外依赖；仅安装torch/numpy/Pillow后直接调用整个Hub可能失败。此外，原生`weights`路径会进入Hub文件缓存。为使审计直接验证指定文件，主线采用锁定源码的官方`dinov3.hub.backbones.dinov3_vith16plus`构造函数，再用`torch.load`读取已验hash的checkpoint。这条直接factory链已核对仅需要torch/numpy，图片读取另需Pillow；没有改模型结构或参数名。

### 6.2 完整 smoke 脚本模板

以下为**可单独保存执行的参考代码，尚未在本服务器运行**。建议后续归入 `scripts/verify_dinov3.py`；它不依赖尚未实现的 `transdepth` 模块。环境变量必须指向实际存在文件，完整 SHA256 从 §3 下载结果设置。CPU 内存需要同时容纳初始化模型与 checkpoint；先单进程验收，避免 8 个 rank 同时加载造成 CPU 峰值。

```bash
# 在§3得到的环境变量仍有效时执行；SHA256记录应随权重一起迁移。
export TD_DINO_CHECKPOINT="$TD_DINO_WEIGHTS/$TD_DINO_FILE"
export TD_DINO_SHA256="$(cut -d' ' -f1 "$TD_DINO_CHECKPOINT.sha256")"
# 可选：已转换成项目画布的固定训练/Dev RGB图。
# export TD_SMOKE_RGB=/srv/transdepth/qa/smoke_rgb_384x512.png
```

```python
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
repo = Path(os.environ["TD_DINO_REPO"]).resolve(strict=True)
weight = Path(os.environ["TD_DINO_CHECKPOINT"]).resolve(strict=True)
expected_sha = os.environ["TD_DINO_SHA256"].lower()
assert len(expected_sha) == 64 and all(c in "0123456789abcdef" for c in expected_sha)
actual_commit = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
).strip()
assert actual_commit == COMMIT, (actual_commit, COMMIT)
dirty = subprocess.check_output(
    ["git", "-C", str(repo), "status", "--porcelain"], text=True
).strip()
assert not dirty, "vendor 源码有改动；先独立记录补丁并重做身份检查"
digest = hashlib.sha256()
with weight.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
assert digest.hexdigest() == expected_sha, "checkpoint SHA256 mismatch"

# 直接导入官方factory，避开整个hubconf的无关任务依赖。
sys.path.insert(0, str(repo))
import dinov3
assert Path(dinov3.__file__).resolve().is_relative_to(repo), "loaded wrong dinov3 package"
from dinov3.hub.backbones import dinov3_vith16plus

# pretrained=False仅构造结构；随后严格加载指定原生文件。
model = dinov3_vith16plus(pretrained=False)
state = torch.load(weight, map_location="cpu", weights_only=True)
assert isinstance(state, dict)
assert "blocks.0.attn.qkv.weight" in state, "不是预期的原生 backbone state_dict"
model.load_state_dict(state, strict=True)
del state
assert model.embed_dim == 1280
assert model.n_blocks == len(model.blocks) == 32
assert model.num_heads == 20 and model.patch_size == 16
assert model.n_storage_tokens == 4
assert tuple(model.storage_tokens.shape) == (1, 4, 1280)
assert model.rope_embed.normalize_coords == "separate"
assert model.rope_embed.base == 100
assert model.rope_embed.dtype == torch.float32
for block in model.blocks:
    assert block.attn.num_heads == 20
    assert tuple(block.attn.qkv.weight.shape) == (3840, 1280)
    assert block.norm1.eps == block.norm2.eps == 1e-5
    assert block.mlp.w1.out_features == block.mlp.w2.out_features == 5120
    assert block.mlp.w3.in_features == 5120
    mask = block.attn.qkv.bias_mask
    assert torch.equal(mask[:1280], torch.ones_like(mask[:1280]))
    assert torch.equal(mask[1280:2560], torch.zeros_like(mask[1280:2560]))
    assert torch.equal(mask[2560:], torch.ones_like(mask[2560:]))

model.requires_grad_(False).eval()
assert torch.cuda.is_available(), "在目标CUDA服务器上执行GPU验收"
model = model.to("cuda")
assert model.rope_embed.periods.dtype == torch.float32

# 可传入一张已按本项目 resize+padding 得到的384x512 RGB图；这里不隐式resize。
image_path = os.environ.get("TD_SMOKE_RGB")
if image_path:
    image = Image.open(image_path).convert("RGB")
    assert image.size == (512, 384), "请先用共同预处理生成画布"
    x = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()[None] / 255
else:
    # 随机输入只用于结构/有限性测试，不能证明数据预处理正确。
    x = torch.rand(1, 3, 384, 512, generator=torch.Generator().manual_seed(17))
mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
x = ((x - mean) / std).to("cuda")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    tokens, grid = model.prepare_tokens_with_masks(x, masks=None)
    assert tuple(tokens.shape) == (1, 773, 1280) and tuple(grid) == (24, 32)
    features = model.get_intermediate_layers(
        x, n=[7, 15, 23, 31], reshape=True,
        return_class_token=True, return_extra_tokens=True, norm=True,
    )
assert len(features) == 4
for patch, cls, storage in features:
    assert tuple(patch.shape) == (1, 1280, 24, 32)
    assert tuple(cls.shape) == (1, 1280)
    assert tuple(storage.shape) == (1, 4, 1280)
    assert all(torch.isfinite(t).all() for t in (patch, cls, storage))
torch.cuda.synchronize()
print(json.dumps({
    "repo_commit": actual_commit,
    "checkpoint_sha256": expected_sha,
    "checkpoint_bytes": weight.stat().st_size,
    "parameter_count": sum(p.numel() for p in model.parameters()),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "feature_shapes": [list(triple[0].shape) for triple in features],
    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    "test_precision": "fp32",
    "status": "backbone_smoke_only",
}, indent=2))
```

运行参考脚本时不要把输出状态改写为 `training_passed`。它只覆盖原生权重、模型结构和特征形状；没有覆盖 LoRA、FAR、解码器、实际数据集或性能。

本次文档审阅额外做了**仅导入检查**：在本机Python3.11、torch2.11.0+cu128环境，从锁定commit所需的19个源码文件成功导入官方factory与`DinoVisionTransformer`；CUDA未初始化，未实例化模型。这支持直接factory依赖链的核查，不代表上面的权重smoke已经执行，也不能代替Linux/torch2.7.1目标环境验收。

## 7. 特征接口和几何约定〔待实现〕

### 7.1 固定张量契约

`src/transdepth/models/backbone/dinov3.py` 提供单一包装器，输入为共同预处理后的 RGB 张量。预测器不接收 GT、透明 mask、深度、K 或文件名。路径和样本 ID 只留在外层数据管理/日志，不能进入模型特征。

| 张量/接口 | 形状与约束 |
|---|---|
| `rgb` | `[B,3,384,512]`，RGB 顺序，浮点，已按 ImageNet mean/std 标准化 |
| 内部全部 tokens | `[B,773,1280]`，顺序 CLS、4 storage、768 patches |
| `feature_blocks_0based` | `[7,15,23,31]`，指相应 block 执行后的输出 |
| 原始每层 patch tokens | `[B,768,1280]` |
| 每层 patch map | `[B,1280,24,32]` |
| `get_intermediate_layers(..., n=4)` | **最后4层 [28,29,30,31]**，不是本计划四层 |
| `reshape=True` | 自动排除前缀、还原网格，不能再次删掉5个 patch |
| `return_class_token/return_extra_tokens=False` | 返回4个 patch tensors 的 tuple，供默认 decoder 使用 |
| 两个 return 开关均 True | 每层返回 `(patch, cls, storage)`；只用于身份检查 |
| `norm=True` | 由官方冻结 backbone norm 处理每个读出层输出 |

直接调用 `model(rgb)` 通常得到 CLS head 路径，不能拿它当密集 patch 特征。也不能用最后一层复制四次来满足“4尺度”形状。上述签名以 [锁定版本中间层接口](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/models/vision_transformer.py) 为准。

### 7.2 resize、padding 与重组

1. RGB 从原图等比例缩放进 384×512，padding 位置和数值全项目统一；建议 RGB 用固定 bilinear＋antialias，并让 padding 的归一化后值为 0。
2. depth/mask/valid 同步最近邻；padding 不进深度、关系、分割监督或评测。相机 K 的同步更新只服务标签/几何审计，不是模型输入。
3. normalization 仅执行一次。OpenCV 默认 BGR 必须转 RGB；16-bit 深度不能走 RGB 的 `/255`。
4. 原生 4 个读出层的空间分辨率都为 24×32。`src/transdepth/models/reassembly.py` 的可训练投影将 1280 通道变为 128，再形成 `/4,/8,/16,/32` 特征：96×128、48×64、24×32、12×16。
5. flatten/reshape 用行优先 `patch_id = row * 32 + col`；对应完整 attention token ID 是 `5 + patch_id`。单独保存转换函数，教师和可视化共同调用，禁止重复散写 `+5`。
6. 用单个亮色网格标记、非正方形图和 padding 边检查坐标；384×512 可发现把 H/W 颠倒却在正方形输入上通过的 bug。

RGB mean/std 延续 LVD 预训练惯例；384×512 和保比例画布来自本项目研究协议，不能称为 DINOv3 官方唯一预处理。[官方预处理示例](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/README.md#image-transforms)

## 8. 单头 Q/K LoRA 与 post-RoPE attention 的实施边界〔待实现〕

### 8.1 LoRA 包装 packed QKV，不替换整套 ViT

`src/transdepth/models/backbone/selected_qk_lora.py` 建议实现 `SelectedQKLoRA`，保留原 `qkv` 模块作为冻结 `base`。前向先调用 `base(X)`，这样自动继承有效 bias mask；额外计算 Q 和 K 的低秩增量，拼回 packed 输出后交还官方 attention。包装器公开 `in_features`，因为官方 `compute_attention` 会读取它。

以 head `h`、C=1280、d=64 为例，PyTorch `Linear.weight` 的行是输出维：

| 位置 | packed 输出区间/权重行区间 |
|---|---|
| Q 的指定头 | `[h*64:(h+1)*64]` |
| K 的指定头 | `[1280+h*64:1280+(h+1)*64]` |
| V 全段 | `[2560:3840]`，始终冻结、增量为零 |

用 PyTorch 的存储约定，`Aq/Ak: [r,1280]`，`Bq/Bk: [64,r]`，计算 `eta * linear(linear(X,A),B)`；r=4、eta=1。参数数 `2*r*(1280+64)=10752`。这和论文行向量记号的 U/B 是转置关系。保存配置中的缩放方式；若用 `alpha/r`，必须取能实现当前 eta 的明确 alpha，不能隐式改实验强度。

初始化 A 为固定随机流的非零小值、B 为零。这样注入瞬间学生与原模型等价。B 为零时第一步 A 梯度可能为零是正确链式结果；检查 B 第一步非零，再检查 B 更新后 A 梯度。所有原始 bias、V、norm、LayerScale 和非选中头都不加入优化器。

### 8.2 教师必须从原始参数重算

`src/transdepth/models/backbone/attention_adapter.py` 的职责是：

1. 从被选 block 的 `norm1` 输出取得 X；此层之前无其他适配器，所以上游保持冻结确定性。
2. `base(X)` 得到原 QKV；`SelectedQKLoRA(X)` 得到学生 QKV。不要从已加 LoRA 的结果截取 Q/K 后称为 A0。
3. 用官方 `attn.apply_rope` 处理全 N 个 Q/K，之后再截取选中 head 和被采样 queries。
4. 完整 Q/K 应为 `[B,20,773,64]`；额外的所选头 query/key logits 为 `[B,1,Q,773]`，`Q<=256` 为 T/B 各最多128的上限，实际每图可能更少。
5. 教师与学生关系 logits 均使用 post-RoPE Q/K 的 FP32 matmul、除以 `sqrt(64)=8`、FP32 log_softmax。计算关系时关闭 autocast；只有 `.float()` 而外部仍处于 autocast 下，不足以保证 matmul 在 FP32 执行。
6. 教师进入 `no_grad()` 并 detach，学生保留梯度。教师保留全部773 keys，包含所有前缀和不可靠补集。只在可靠 keys 内改变概率，不把 softmax 分母截成可靠 keys 后冒充完整 A0。

**重要顺序陷阱：**不要先把 Q 切成128行，再把完整768 patch RoPE 传给官方 `apply_rope`。该方法根据 token 数与 sin/cos 长度推断前缀，这会破坏位置对应甚至触发断言。先做全 token RoPE，再抽 query；也可以写经过等价验收的带位置索引旋转函数，但首轮没有必要。

也不要额外乘两次 head scale：显式 logits 只乘 `1/sqrt(d)` 一次，官方 SDPA 自己应用默认缩放。FP32 关系重算与混合精度 SDPA 的数值差异需要测量；“同公式”不保证每个 kernel 位级一致。

### 8.3 Oracle 真正替换 AV 的位置

oracle 用 P 替换指定 head 的 attention 概率，计算 `P @ V0`；其余 heads 使用原输出。拼回所有 heads 之后继续官方 output projection、LayerScale、残差、FFN、suffix 与读出。不能只记录一张替换后的 attention 热图而深度仍走原 forward。

推荐让官方 `SelfAttention.forward` 仍运行，在适配层安装一个有明确定义的 `compute_attention` 包装器或专门兼容模块。覆盖 `forward_list` 的调用路径也须验收；不要只 hook 一个在真实 block 路径没有被调用的方法。本版本 `SelfAttentionBlock.forward` 会转入 list 路径，eval 分支内部调用其 attention 模块；必须用实际端到端检查证明钩子生效。[Block 实现](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/layers/block.py)

避免通过 Python 全局可变 `latest_attention` 保存训练图：checkpoint 重算、嵌套前向和多卡会造成旧缓存污染。返回显式 `AttentionTrace`，只包含当前调用的必要 selected-head 张量；oracle 与训练模式互斥。DDP 每进程独立拥有 context。

## 9. 冻结与 autograd：最容易让 FAR 失效的一段〔待实现〕

### 9.1 三个机制的含义

| 设置 | 做什么 | 不做什么 |
|---|---|---|
| `parameter.requires_grad_(False)` | 不为该参数积累梯度 | 不切断输入张量梯度 |
| `module.eval()` | 关闭训练模式随机行为，稳定 RoPE | 不等于关闭梯度 |
| `torch.no_grad()` | 该上下文不建反向图 | 不能套在适配层后的 suffix 上 |

T0/H0 冻结全骨干时，可用 `no_grad()` 生成特征，再训练 decoder。H1/H2 只让适配层**之前**的 prefix 使用 `no_grad()`；从适配 block 起必须保留计算图，suffix 权重虽冻结，其输入 Jacobian 仍把深度梯度传回 LoRA。多个中间层读出也要正确接入；适配层之前的分支是常量，之后的读出分支需要反向。

### 9.2 保留梯度的 forward 参考设计

下例是**包装器待实现时的核心结构参考**，不是可直接运行的项目入口。它使用的官方 block 调用方式、token 准备和 RoPE 接口均以本章锁定 commit 为准。`adapter_block` 必须先由 oracle 配置确定，不能默认成“已选中31”。

```python
def forward_with_frozen_prefix(backbone, rgb, adapter_block):
    wanted = {7, 15, 23, 31}
    maps = []
    backbone.eval()  # 不妨碍新增LoRA参数计算梯度
    with torch.no_grad():
        tokens, (gh, gw) = backbone.prepare_tokens_with_masks(rgb, masks=None)
        rope = backbone.rope_embed(H=gh, W=gw)
    for index, block in enumerate(backbone.blocks):
        if index < adapter_block:
            with torch.no_grad():
                tokens = block(tokens, rope)
        else:
            tokens = block(tokens, rope)  # 冻结suffix参数，但保留输入梯度
        if index in wanted:
            normalized = backbone.norm(tokens)
            patch = normalized[:, 5:]
            patch = patch.transpose(1, 2).reshape(rgb.shape[0], 1280, gh, gw)
            maps.append(patch)
    return tuple(maps)
```

这里不要给整个函数加 `@torch.no_grad()` 或 `@torch.inference_mode()`。T0 阶段优先 `no_grad()`；inference tensor 随后参与可训练卷积保存反向中间量时可能不适合直接复用。用 inference_mode 专门服务最终纯推理或独立 smoke，不混入训练特征缓存。

`model.train()` 会递归把 backbone/RoPE 变回训练模式，外层模型应覆盖 `train(mode)`：先调用 `super().train(mode)`，再强制 `backbone.eval()`；decoder 保持对应模式，LoRA 线性层没有 dropout，不靠 `.training` 才获得梯度。

### 9.3 checkpoint 与缓存

需要降低 suffix 激活时，采用 `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`；先通过无 checkpoint 的身份和梯度验收，再比较打开后的结果。checkpoint 重算保持官方 eval 模式，且不把教师构造作为带副作用的缓存更新藏在重算函数里。

最小版不跨 optimizer step 缓存教师。若以后缓存，key 至少包括 RGB 内容/增强指纹、样本 ID、骨干和权重 hash、层/头、预处理版本与教师参数。对同一原始图改变 crop/resize 之后，旧 attention 和 GT patch 统计不能沿用。

## 10. 数值、梯度与合并验收门槛〔待实现〕

各项输出到 `artifacts/implementation_checks/backbone_integration.json`，其中保留输入 ID/hash、dtype、kernel backend、max absolute error、归一化误差、阈值、passed 和失败原因。以下是**验收规格，不是现有测试结果**。

| 验收 | 必须比较什么 | 放行条件 |
|---|---|---|
| 身份 | 官方构造、完整hash、严格state_dict、结构表 | 所有身份字段一致，missing/unexpected keys均0 |
| 四层特征 | 官方 `get_intermediate_layers` 与冻结prefix包装器 | 同样4层、归一化、patch顺序与张量值 |
| 零 LoRA | 适配前与B=0注入后 | 同一RGB、同精度下4层及最终深度一致 |
| post-RoPE | 显式选中头 `softmax(QK^T/sqrt(d)) @ V` 与官方该头SDPA输出 | FP32数学kernel与部署kernel分开验收 |
| 身份 oracle | P=A0 后走完整替换输出路径 | 指定头、block输出、后续特征、深度均等价 |
| LoRA隔离 | 有非零增量后同层Q/K/V | 只选定头Q/K改变，V和其他头投影不变 |
| 深度梯度 | `lambda_rel=0`，只深度loss backward | B梯度非零；后续A可更新；不靠关系项掩盖断链 |
| 教师固定 | 更新LoRA前后对同X的原始A0 | 数值相同且`requires_grad=False` |
| 冻结 | optimizer更新前后逐参数hash/精确对比 | 原骨干不变，仅允许参数变化 |
| 合并 | 独立权重副本的合并前后 | Q/K、selected head、四层及最终深度等价 |
| RGB-only | 禁用预测器对标签路径的访问 | 相同RGB仍得到相同预测；评估器另读标签 |

数值起点：FP32 小输入在 `SDPBackend.MATH` 下尝试 `atol=1e-5, rtol=1e-4`；这只是排错起点，不宣称所有完整模型张量必然满足它。正式门槛先用固定 Dev 样本，测官方同路径重复运行和 FP32/部署精度差异，**在 oracle Select/Confirm 前**登记不同张量层级与精度的阈值。半精度对比同时报告 p99 与最大误差，不能只给平均值掩盖局部错位；不能每遇失败就放宽门槛。

建议先用非正方形小输入 64×80 验证运算链，再用真实 384×512 固定训练/Dev 图验证尺度。失败时先检查 pre/post RoPE、QKV切片、bias_mask、H/W、prefix、归一化、head scale、输出投影和 LayerScale。测试全集不得用来定容差。

合并 `Linear` 的 PyTorch 权重时，增量是 `eta * (B @ A)`，按上述 Q/K 行区间散射；不修改原教师权重。导出 metadata 包含 `lora_merged=true` 与原始/导出完整 hash，第二次 merge 必须报错；不能把已合并权重又当作未合并模型加一次 LoRA。

## 11. 8×3090 的资源计划：预算与实测分开

### 11.1 可直接计算的量

以官方约840M参数估算，FP32冻结权重约3.36GB十进制（约3.13GiB），BF16约1.68GB（约1.56GiB）。这些仅是权重张量算术，不含激活、临时转换、decoder、CUDA上下文、allocator保留、梯度、优化器状态和加载峰值，不能当训练显存承诺。

384×512时N=773。单图单层20 heads的完整FP32 attention矩阵约 `20*773*773*4 = 47,802,320` bytes，约45.6MiB；全部32层显式保存将额外约1.42GiB，仅此一项还未计算softmax/logits/反向副本。对一个头、256个queries、773keys，单个FP32矩阵约0.75MiB。此算术说明优先只重算指定头的采样 queries；不证明完整训练显存已经安全。

DDP在每张GPU复制一个完整模型；8×24GB不等于单进程可访问192GB。首轮microbatch=1，8卡×累积2＝有效batch16。备选4卡×累积4仍为16，但需要固定相同全局来源样本表，不能仅保证数字乘积相同。[当前最小计划的 M12](../FAR_62CAD_ClearGrasp含真实训练_局部修订与逐模块最小验证计划.md)

### 11.2 服务器验收次序

1. 单进程下载/strict加载/FP32 smoke。
2. 单卡混合精度、只有decoder的T0训练；固定样本、固定增强。
3. 单卡H1：选中block LoRA＋suffix反向；仅深度loss检查LoRA更新。
4. 单卡H2：增加FP32关系项，记录有关系/空关系两类样本的峰值。
5. 每种路径先20个完整optimizer更新预热，再测100个完整更新，CUDA同步后计时。记录 allocated/reserved、每更新耗时、数据等待、NaN/Inf/OOM。
6. 保留约10%单卡显存余量作为启动门槛；不足先清除全层attention、重复教师、全网grad或误留历史图，再对suffix checkpoint。
7. 单卡通过后再做2卡DDP smoke与8卡全局7/7/2来源表验收；保存首个更新各rank实际sample IDs。
8. 不把FP16 GradScaler跳过的一次更新计成有效训练更新；同时记录attempted与successful optimizer steps，配对预算规则见训练文档。

## 12. 文件职责、实施顺序与完成证据〔待实现〕

| 拟建文件 | 具体职责 | 上游依赖 | 最小完成证据 |
|---|---|---|---|
| `configs/backbone/dinov3_h16plus.yaml` | 型号/路径/hash/四层/RoPE设置 | 官方权重下载 | 展开配置写入run目录 |
| `scripts/verify_dinov3.py` | 官方factory原生加载、身份、形状、有限性 | 服务器环境、checkpoint | 真实JSON报告 |
| `src/transdepth/models/backbone/dinov3.py` | 冻结prefix和保梯度suffix，返回4层 | strict原生加载 | 与官方中间层等价 |
| `src/transdepth/models/backbone/selected_qk_lora.py` | 单头Q/K低秩增量与导出合并 | 锁定层/头 | 10752参数、零增量、冻结/梯度检查 |
| `src/transdepth/models/backbone/attention_adapter.py` | 原/学生post-RoPE抽样logits、oracleAV | 原生attention接口 | SDPA及身份oracle等价 |
| `src/transdepth/models/reassembly.py` | 4份1280特征→128通道四尺度 | backbone接口 | 坐标/shape/反向检查 |
| `tests/integration/test_dinov3_identity.py` | 有真实权重才运行的集成测试 | 上述实现和许可权重 | 缺权重时明确skip，不能伪装通过 |
| `artifacts/implementation_checks/` | 数值、梯度、冻结、合并报告 | 实际执行 | 版本、输入、误差、门槛可追溯 |
| `artifacts/resource_profile/` | 单卡/DDP训练资源报告 | H1/H2真实短跑 | 20预热＋100更新记录 |

实施先后为：**拿到正确权重→锁代码和环境→strict加载与四层→H＋DPT小样本拟合→LoRA零初始化与梯度→post-RoPE/身份oracle→FAR训练→合并与纯RGB导出**。只下载权重或跑通CLS特征，都不能替代这些环节。

## 13. 故障定位速查

| 现象 | 首先检查 | 正确处理 |
|---|---|---|
| 下载401/403 | 账户是否获批、签名URL是否完整/过期 | 重新取得该入口的有效权限/链接 |
| 文件很小、`torch.load`失败 | 是否下载到错误页、下载是否完整 | 核验内容、重新续传/下载，再hash |
| 大量missing/unexpected keys | `.pth`与HF safetensors是否混用 | 回到正确后端；保持strict=True |
| shape是768或1024通道 | 是否错用了B/L | 检查入口、代码commit、checkpoint |
| 空间reshape差5个tokens | 是否误把前缀放入网格或重复剔除 | 统一token布局接口 |
| 零LoRA不等价 | RoPE训练随机性、bias_mask、额外QK norm | backbone.eval，继承base投影，保留官方路径 |
| Teacher随训练漂移 | 是否用了学生QKV或已合并base | 保存未修改的原始QKV，teacher detach |
| KL能降而深度不能更新LoRA | suffix被no_grad/detach | 仅prefix无梯度，重做lambda_rel=0验收 |
| OOM但LoRA只有1万个参数 | 全网梯度/激活、全层attention、重复H教师 | 检查实际计算图与缓存；DDP不合并显存 |
| 合并结果错误 | `B@A`方向、Q/K行切片、重复merge | 独立副本合并并逐层对比 |
| Dev可跑、无标签推理失败 | 预测器依赖mask/深度/教师 | 收紧RGB-only接口并做标签访问隔离 |

完成本章的实施，应当产生“这一个官方H/16+、这一份权重、这一种预处理、这一条FAR插入路径”的可核验证据；研究是否有效仍由后续同数据H1/H2对照决定。
