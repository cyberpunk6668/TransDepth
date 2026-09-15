# 01｜项目结构、GitHub 初始化与远端环境

日期：2026-09-15。状态：**实现规格与命令模板，尚未创建远程仓库或部署训练环境。**

本文中的标准 `git`、`python -m venv`、`pip` 命令可在满足前置条件后使用；`python -m transdepth.cli.*` 是后续待实现入口。不要把目录设计当成当前已经存在的代码。

## 1. 本地与服务器的职责

- 本地 Windows 工作区：编辑、审阅、CPU 单元检查和 Git 提交。
- Linux GPU 服务器：数据盘点、权重下载验收、CUDA 集成测试、训练和推理。
- GitHub：保存源代码、配置模板、协议和小型测试；数据、基础权重、训练检查点留在服务器存储。
- 实验运行目录：保存完整配置、版本、日志、指标、预测索引和 checkpoint。一个运行目录只对应一个不可混淆的训练分支。

当前检查结果：本工作区最初只有 `tech_documents/`，不是 Git 仓库。下述操作未替你执行；GitHub 用户名、仓库名、服务器地址与数据路径仍待填写。

## 2. 目标目录：按数据流划分，不把所有逻辑塞进 train.py

```text
TransDepth/
├── README.md
├── pyproject.toml
├── requirements.txt                 # 应用依赖入口；不放 CUDA wheel 下载源
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   └── locks/                      # 验收后锁环境，按 Python/CUDA/平台命名
├── .gitignore
├── .gitattributes
├── .github/workflows/cpu-checks.yml
├── configs/
│   ├── base.yaml                   # 全项目共同默认值
│   ├── paths.example.yaml          # 可提交的路径模板
│   ├── paths.local.yaml            # 本机真实路径，忽略提交
│   ├── inference.paths.local.yaml  # 预测专用：只含源码/权重/输出路径
│   ├── data/{rftrans62,cleargrasp_syn,cleargrasp_real}.yaml
│   ├── data/{releases_v1,splits_v1,preprocessing_v1}.yaml
│   ├── protocol/mix_real_v1.yaml    # split、配比、选择和确认权限
│   ├── model/{h_a,h_b,h_c,b_a}.yaml
│   ├── backbone/dinov3_h16plus.yaml
│   ├── oracle/h_a.yaml
│   ├── export/rgb_only.yaml
│   ├── evaluation/main_384x512.yaml
│   └── experiments/{t0,mix_h0,mix_h1,mix_h2,cg_h1,cg_h2,syn_h1,syn_h2}.yaml
├── src/transdepth/
│   ├── __init__.py
│   ├── config.py                   # 合并、验证、锁配置
│   ├── contracts.py                # 数据类型和张量约定
│   ├── cli/
│   │   ├── inspect_data.py         # 只盘点，不决定 split
│   │   ├── build_manifest.py
│   │   ├── prepare_data.py
│   │   ├── split_data.py
│   │   ├── audit_data.py
│   │   ├── verify_backbone.py
│   │   ├── profile.py
│   │   ├── train.py
│   │   ├── oracle.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── export.py
│   ├── data/
│   │   ├── adapters/{base,rftrans62,cleargrasp_syn,cleargrasp_real}.py
│   │   ├── schema.py
│   │   ├── paths.py
│   │   ├── manifest.py
│   │   ├── depth_io.py
│   │   ├── depth.py
│   │   ├── masks.py
│   │   ├── geometry.py
│   │   ├── transforms.py
│   │   ├── grouping.py
│   │   ├── split.py
│   │   ├── audit.py
│   │   ├── cache.py
│   │   ├── collate.py
│   │   ├── dataset.py
│   │   ├── rgb_only.py
│   │   ├── sampling.py
│   │   └── patches.py
│   ├── models/
│   │   ├── backbone/{dinov3,attention_adapter,selected_qk_lora}.py
│   │   ├── reassembly.py
│   │   ├── blocks.py
│   │   ├── decoders/{dpt,unet,dual_unet_dpt}.py
│   │   ├── heads.py
│   │   └── predictor.py
│   ├── far/{teacher,relation_loss,diagnostics}.py
│   ├── oracle/{candidates,runner,statistics}.py
│   ├── losses/{depth,segmentation,reduction}.py
│   ├── engine/{trainer,optim,checkpoint,distributed,seed,profile,pairing}.py
│   ├── evaluation/{metrics,regions,aggregate,bootstrap,report,runner,locks}.py
│   └── inference/{preprocess,predictor,merge,serialization}.py
├── tests/
│   ├── unit/                      # 小张量、人工深度和人工关系矩阵
│   ├── integration/               # 官方骨干/真实编码，需本地授权资产
│   └── fixtures/                  # 自造微型样本，无真实 Test
├── scripts/
│   ├── server/bootstrap.sh
│   ├── server/run_train.sh
│   ├── server/run_slurm.sh
│   └── checks/                    # 模型身份与审计脚本
├── third_party/dinov3/             # Git submodule，固定 commit
├── artifacts/                     # 外部盘路径或被忽略的运行目录
└── tech_documents/
    └── implementation/            # 本套技术文档
```

花括号是简写，实施时每项建为独立 `.py` / `.yaml` 文件。Python 包目录加入 `__init__.py`。子文档可能进一步拆文件，必须在交付实现时同步此树与导入路径。

### 2.1 模块边界

1. `data` 只负责读入、几何、标签和索引，不 import 模型。
2. `models` 的正常预测 `forward(rgb)` 不接收 GT、mask、内参、来源 ID 或文件名。
3. `far` 消费训练器提供的监督和 attention trace；不进入生产预测包。
4. `engine` 组合模型、优化器、sampler、loss 和 checkpoint；CLI 仅解析参数并调用它。
5. `evaluation` 读已经落盘的预测，再读独立评估标签。
6. `inference` 只读 RGB 与部署权重，不 import 含 GT 解析器的数据集。

## 3. Git 初始化与第一次推送

### 3.1 本地初始化（PowerShell 模板）

在本轮文档审阅后、开始实现时使用。邮箱和姓名由你填写，不改整台机器的全局 Git 身份。

```powershell
Set-Location -LiteralPath 'F:\桌面\TransDepth'
git init -b main
git config user.name "YOUR_NAME"
git config user.email "YOUR_GITHUB_EMAIL"
git status --short
```

首次提交前先创建下面的忽略规则。只 stage 已审阅文件，随后检查暂存差异。

```gitignore
# Python / IDE / scratch
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
.ruff_cache/
*.egg-info/
build/
dist/
tmp/
.idea/

# Local configuration and credentials
.env
.env.*
!.env.example
configs/paths.local.yaml
configs/inference.paths.local.yaml
*.pem
*.key

# Restricted/large assets and experiment output
/data/
/datasets/
/weights/
/checkpoints/
/artifacts/
/outputs/
/runs/
/wandb/
*.pth
*.pt
*.ckpt
*.safetensors
*.zip
*.tar
*.tar.gz
*.zip.[0-9][0-9][0-9]
```

不要使用 `*.json`、`*.yaml`、`*.csv` 的全局忽略，会误伤协议和审计模板。清单可能含真实路径，实际 manifest 默认放 `artifacts/`；可提交去路径、去敏感元数据的协议摘要。

`.gitattributes` 建议内容：

```gitattributes
* text=auto
*.py text eol=lf
*.sh text eol=lf
*.yaml text eol=lf
*.yml text eol=lf
*.toml text eol=lf
*.md text eol=lf
*.pdf binary
```

GitHub 上先建空仓库，建议研究期间 Private；仓库名称以实际输入为准。不要同时在远程生成另一套 README/初始提交。然后：

```powershell
git add README.md tech_documents .gitignore .gitattributes
git diff --cached --stat
git diff --cached
git commit -m "docs: define FAR implementation and experiment protocol"
git remote add origin git@github.com:YOUR_ACCOUNT/TransDepth.git
git push -u origin main
```

若已有 origin，先 `git remote -v` 核对，不盲目重设。如果采用 HTTPS，使用凭证管理器或交互认证，不把 token 写进 remote URL。GitHub 官方的[本地仓库导入流程](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)支持这一方式。

### 3.2 服务器访问 GitHub

服务器只需拉代码时使用该仓库的只读 deploy key；要推代码才配写权限。生成新的专用 key 前先检查是否已经有可用 key。例：

```bash
ssh-keygen -t ed25519 -f ~/.ssh/transdepth_github -C "transdepth-server"
```

将 `.pub` 公钥添加到 GitHub 仓库的 Deploy keys；私钥留在服务器。写入 SSH 配置的示例：

```sshconfig
Host github-transdepth
    HostName github.com
    User git
    IdentityFile ~/.ssh/transdepth_github
    IdentitiesOnly yes
```

```bash
ssh -T git@github-transdepth
git clone --recurse-submodules git@github-transdepth:YOUR_ACCOUNT/TransDepth.git
cd TransDepth
git rev-parse HEAD
```

首次连接核对 GitHub 官方 host key 指纹；不要关闭 host-key 验证。具体 key 管理参见[GitHub SSH 文档](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)。服务器不通过 Git 同步原始数据。

### 3.3 固定 DINO 第三方代码

实施阶段执行一次 submodule 添加，后续只更新固定版本：

```bash
git submodule add https://github.com/facebookresearch/dinov3.git third_party/dinov3
git -C third_party/dinov3 checkout 6876159a11b4df116f30f667f8c9888617df0751
git add .gitmodules third_party/dinov3
git commit -m "build: pin official DINOv3 source"
```

这是本次核验到的官方 commit；权重仍需单独下载。自己的 adapter 写在 `src/transdepth/models/backbone/`，保持 submodule 工作区干净。更换 DINO 版本时，新建兼容性变更并重跑身份、RoPE、LoRA 和 merge 验收。

## 4. 依赖方案：先固定 PyTorch，再安装应用依赖

建议首个验收环境：Linux x86_64、Python 3.11、PyTorch 2.7.1、torchvision 0.22.1、CUDA 12.6 wheel。这是工程起点，不是已在你的服务器测过的环境。官方[历史版本页](https://pytorch.org/get-started/previous-versions/)给出该配套安装命令；DINO 官方依赖文件并没有为你的 FAR 项目提供现成锁文件。

先获取服务器信息：

```bash
uname -a
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python3.11 --version
df -h
```

`nvidia-smi` 顶部 CUDA 字段不是项目虚拟环境的 `torch.version.cuda`。wheel 自带用户态 CUDA 运行组件，驱动仍要兼容。先做实际 CUDA 张量和 SDPA 检查；失败时使用管理员提供的兼容环境或按 PyTorch 官方组合重新制定版本，不复制别人的 `nvcc` 路径。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install -r requirements/dev.txt
python -m pip install --no-deps -e .
python -m pip check
```

`requirements.txt` 内容：

```text
-r requirements/base.txt
```

`requirements/base.txt` 起点（下列区间是拟定约束，需在服务器解析、验收后锁成精确版本）：

```text
torch==2.7.1
torchvision==0.22.1
numpy>=1.26,<3
Pillow>=10,<12
PyYAML>=6,<7
omegaconf>=2.3,<3
opencv-python-headless>=4.10,<5
OpenEXR>=3.3,<4
scipy>=1.12,<2
tqdm>=4.66,<5
matplotlib>=3.8,<4
tensorboard>=2.16,<3
```

理由：OpenEXR 读原始浮点 EXR；OpenCV 读保留位深的 PNG 和几何图像操作；OmegaConf 用于本项目配置；SciPy 用于统计/边界辅助。03章采用锁定源码中的官方backbone构造函数直接加载，避开整个torch.hub顶层任务模块的额外依赖；若另用完整Hub入口，应按其实际import链补齐官方依赖，不能把两种入口的最小环境混用。不用RFTrans历史全套训练环境或其折射流/法线网络。HF备用下载才额外安装并锁定`huggingface_hub`；原生`.pth`主线不要求Transformers、PEFT、xformers或flash-attn。

`requirements/dev.txt`：

```text
pytest>=8,<9
ruff>=0.12,<0.13
```

不要把 `pip freeze` 导出的本机绝对 editable 路径传播到服务器。首次环境验收后导出完整依赖快照，清除仅有本地路径的本项目 editable 行，将应用版本用 Git SHA 单独记录；保留 CUDA wheel 来源、Python 小版本与平台信息。将可复现环境文件保存到 `requirements/locks/`，另存 `pip inspect` 和包下载清单到运行证据目录。

```bash
python -m pip freeze --exclude-editable > requirements/locks/linux-py311-cu126.txt
python -m pip inspect > artifacts/environment/pip-inspect.json
python -c 'import torch; print(torch.__version__, torch.version.cuda); x=torch.ones(2,device="cuda"); print(x.sum().item())'
```

执行上述导出前创建 `requirements/locks`、`artifacts/environment`。锁文件用同一 CUDA 索引恢复 torch，再装剩余锁定依赖；要支持完全离线则在同平台下载 wheelhouse 并保留 SHA256，不把 Windows wheel 复制给 Linux。

### 4.1 pyproject.toml 模板

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "transdepth-far"
version = "0.1.0"
description = "RGB-only metric depth with frozen DINOv3 and FAR"
requires-python = ">=3.11,<3.12"
dynamic = ["dependencies"]

[tool.setuptools.dynamic]
dependencies = {file = ["requirements/base.txt"]}

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "gpu: requires a CUDA device",
  "weights: requires locally approved official weights",
  "data: requires configured local dataset paths"
]

[tool.ruff]
line-length = 100
target-version = "py311"
```

避免再维护一份依赖列表导致 `pip install -e .` 与 requirements 不一致。项目名称 `transdepth-far` 与 import 包名 `transdepth` 可以不同。

## 5. 路径配置和合并规则

`configs/paths.example.yaml` 的键是接口，值是示例，不说明目录存在：

```yaml
roots:
  rftrans_62cad: /data/raw/rftrans
  cleargrasp_train: /data/raw/cleargrasp/train
  cleargrasp_eval: /data/raw/cleargrasp/test-val
storage:
  manifests: /data/derived/transdepth/manifests
  cache: /data/derived/transdepth/cache
  runs: /data/experiments/transdepth
  pretrained: /data/models/dinov3
backbone:
  repo_dir: third_party/dinov3
  weights: /data/models/dinov3/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
```

三个roots键与02章实际原包接入一致：ClearGrasp的test-val根可能同时含合成与真实来源，由adapter映射按官方split分流为逻辑R/S/Q，不要求你重排磁盘目录。物理根目录名不决定assigned_role。

配置优先级固定为：`base → protocol → model → experiment → paths.local → CLI override`。路径层只允许改路径白名单，不能改 split、loss 或模型。未知键报错；list 是整体替换，不隐式 append。训练启动前解析所有路径为绝对路径、展开完整配置、校验未填写项并保存 `resolved_config.yaml` 与 SHA256。

`head` 和 `beta` 在 oracle 锁定前必须是 `null`，学生 FAR 启动时要求 `oracle_lock.json`，不能用任意默认头悄悄开跑。任何 placeholder，如 `YOUR_*`、`/path/to`、空 hash，都不能通过正式运行校验。

预测端使用独立的`configs/inference.paths.local.yaml`，仅允许backbone源码、导出权重与输出位置；不含`roots`、manifest或标签路径。其schema拒绝训练数据根字段。评估器独立使用完整数据路径配置。这样06章的无标签隔离环境能采用同一预测入口。

## 6. 服务器运行管理

单机八卡模板（CLI 待实现；通过数据、模型、资源验收后使用）：

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  -m transdepth.cli.train \
  --config configs/experiments/mix_h2.yaml \
  --paths configs/paths.local.yaml \
  --run-dir /data/experiments/transdepth/mix-real_h-a2_s17
```

本项目把一次 `optimizer.step()` 定义为一次 update，日志用 `update`；梯度累积中的 forward 次数用 `microstep`。训练器捕获 SIGTERM 时在当前完整 update 边界原子保存中断 checkpoint；不能假装半个累积 batch 已经提交。

有 Slurm 时使用集群现有 partition/account，不虚构 GPU 资源名。示例 job 主体：

```bash
#!/usr/bin/env bash
#SBATCH --job-name=far-mix-h2
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source .venv/bin/activate
export OMP_NUM_THREADS=1
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  -m transdepth.cli.train --config configs/experiments/mix_h2.yaml \
  --paths configs/paths.local.yaml --run-dir "$RUN_DIR"
```

提交前创建 `logs/`，由命令环境设置唯一 `RUN_DIR`，实际 CPU/内存/时限按集群规定填写。没有 Slurm 时在用户自己的持久终端会话运行，不把训练绑到临时 SSH 连接。先用单卡 profile 验证；DDP 是每卡一份模型，不会把显存合并为 192GB。

## 7. CPU CI 与 GPU 验收分工

GitHub Actions 只跑无需授权资产的小测试：配置/schema、单位换算、人工分组、patch、教师数学、指标聚合、LoRA 小矩阵。示例工作流使用 `actions/checkout@v4`、`actions/setup-python@v5`，Python 3.11，先从 CPU 索引安装固定 torch/vision，再装依赖、editable 和 pytest；实施时核查 action 版本并固定完整 commit。

通过条件：`ruff check src tests`、`pytest -m 'not gpu and not weights and not data'`。真实测试集不作为 CI fixture；不要将 gated DINO 权重上传 Actions artifact。服务器另跑标记过的集成测试，给出数据/权重哈希、退出码、数值误差和资源报告。

## 8. 完成环境模块所需的实际证据

| 项目 | 证据 | 失败时处理 |
|---|---|---|
| GitHub 拉取 | server clone SHA 与预期一致 | 修复 repo/key/submodule，不手工拷未知源码 |
| 依赖 | pip check、Python/torch/vision/CUDA 版本 | 调整兼容环境，重出 lock |
| EXR/PNG | 一个浮点EXR与16bitPNG读写测试 | 修对应IO，不通过8bit转换绕过 |
| 官方骨干 | 03章结构及checkpoint严格加载报告 | 停止训练，核对模型身份 |
| CUDA | 单卡 forward/backward 与 SDPA | 检查驱动/wheel/硬件 |
| DDP | 16样本全局顺序、损失/梯度一致 | 修sampler和累积权重 |
| 可恢复 | 中断后与连续运行的状态对照 | 修checkpoint，不能只恢复模型权重 |

本文交付的是上述流程的完整设计。实机验证通过前，不将环境标记为“部署完成”。
