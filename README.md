# TransDepth / FAR

基于冻结 DINOv3 H+/16、单层单头 Q/K LoRA 和 FAR 关系监督的 RGB-only 米制透明物体首表面深度最小验证工程。

本次实机协议服从最新需求：**只用 RFTrans-62CAD 训练，ClearGrasp 仅作 RGB 推理**。它与旧文档的三源 `7R/7S/2Q` 协议不同，单独命名为 `rftrans_only_minimal_v1`，避免结果混表。

## 当前状态

- 工程、数据适配、DPT-A、FAR 教师、真实 `P@V` Oracle、单头 Q/K LoRA、三卡训练、RGB-only 推理、独立评价和配对 bootstrap 均已实现。
- RFTrans 6000 帧完整 manifest 与训练侧数据审计已通过。
- ClearGrasp 官方 Val+Test 已安全下载；共 1226 张推理 RGB，任何标签都不会进入训练。
- 官方 H+/16 源码结构与随机权重 GPU smoke 已通过。
- **官方预训练 checkpoint 仍因个人许可 HTTP 403 阻断，所以尚未运行 T0/H1/H2，当前不能声称方法有效。**

完整实测值和边界见 [`VALIDATION_STATUS.md`](VALIDATION_STATUS.md)。

## 服务器布局

```text
/home/hengxianli/TransDepth/              # Git 源码，仅小文件
/ssd/polyu/RFTrans_datas/                 # 只读 RFTrans 原始数据
/ssd/polyu/TransDepth/
├── envs/far/                             # Python 3.11 环境
├── vendor/dinov3/                        # 固定官方源码 commit
├── models/dinov3/meta_native/            # 官方 .pth（待用户授权）
├── data/raw/cleargrasp/test-val/          # ClearGrasp 推理集
├── data/derived/rftrans/{manifests,qa}/
├── runs/                                 # T0/H1/H2 checkpoint 与日志
└── cache/                                # Conda/pip/torch/HF 缓存
```

`configs/paths.local.yaml` 与 `configs/inference.paths.local.yaml` 已填写本机路径并被 Git 忽略；权重、数据和训练 checkpoint 不进入 GitHub。

## 已可复现的检查

环境初始化：

```bash
cd /home/hengxianli/TransDepth
bash scripts/server/bootstrap.sh
```

数据清单与训练侧审计：

```bash
/ssd/polyu/TransDepth/envs/far/bin/python -m transdepth.cli.build_manifest --hash-files
/ssd/polyu/TransDepth/envs/far/bin/python -m transdepth.cli.audit_data --per-role 30
```

代码测试：

```bash
/ssd/polyu/TransDepth/envs/far/bin/ruff check src tests scripts
/ssd/polyu/TransDepth/envs/far/bin/python -m pytest -q
```

## 官方 DINOv3 权重：需要用户本人完成

1. 在个人浏览器打开 [Meta DINOv3 下载申请页](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)，接受许可并取得 **ViT-H+/16、LVD-1689M、backbone、原生 `.pth`** 的完整签名 URL。
2. 不要把 URL、token 或凭据发到聊天或写进 Git；在服务器终端直接运行下方脚本，并在隐藏提示中粘贴 URL：

```bash
/ssd/polyu/TransDepth/envs/far/bin/python scripts/download_dinov3.py
```

脚本会下载到 `/ssd/polyu/TransDepth/models/dinov3/meta_native/`、计算 SHA256，并自动更新被忽略的 `configs/paths.local.yaml`。HF `model.safetensors` 不能改后缀冒充 Meta 原生 `.pth`。

随后执行正式身份验收：

```bash
CUDA_VISIBLE_DEVICES=0 /ssd/polyu/TransDepth/envs/far/bin/python -m transdepth.cli.verify_backbone
```

## 权重到位后的最小实验顺序

### 1. 公共 T0

```bash
bash scripts/server/run_train.sh t0
```

### 2. Oracle Search 与独立 Confirm

```bash
CUDA_VISIBLE_DEVICES=0 /ssd/polyu/TransDepth/envs/far/bin/python -m transdepth.cli.oracle --phase search --t0-checkpoint /ssd/polyu/TransDepth/runs/rftrans_h_a_t0_seed17/checkpoints/best.pt --output /ssd/polyu/TransDepth/runs/oracle_minimal
CUDA_VISIBLE_DEVICES=0 /ssd/polyu/TransDepth/envs/far/bin/python -m transdepth.cli.oracle --phase confirm --t0-checkpoint /ssd/polyu/TransDepth/runs/rftrans_h_a_t0_seed17/checkpoints/best.pt --output /ssd/polyu/TransDepth/runs/oracle_minimal
```

只有 `oracle_lock.json` 的 `status` 为 `passed` 才能启动 H2；失败时当前 FAR 主张停止。

### 3. 同一 T0 严格分叉 H1/H2

```bash
bash scripts/server/run_train.sh h1 --fork-from /ssd/polyu/TransDepth/runs/rftrans_h_a_t0_seed17/checkpoints/best.pt --oracle-lock /ssd/polyu/TransDepth/runs/oracle_minimal/oracle_lock.json
bash scripts/server/run_train.sh h2 --fork-from /ssd/polyu/TransDepth/runs/rftrans_h_a_t0_seed17/checkpoints/best.pt --oracle-lock /ssd/polyu/TransDepth/runs/oracle_minimal/oracle_lock.json
```

三卡固定为物理 GPU `0,3,5`，每卡 microbatch 1、累积 4、全局 batch 12。H1/H2 使用相同 seed、样本计划、T0、LoRA 初始化和更新预算；唯一研究差异是 FAR 关系项。

### 4. ClearGrasp 仅 RGB 推理

```bash
bash scripts/server/run_cleargrasp_inference.sh <H2-best.pt> /ssd/polyu/TransDepth/exports/cleargrasp_h2
```

预测为 `[384,512]` float32 米制 `.npy`；预测 manifest 不含 GT 路径。

## 文档

从 [实现文档总览](tech_documents/implementation/00_总览与模块实施路线.md) 阅读原始完整研究设计。理论依据见 [核心理论独立版](tech_documents/FAR_核心理论与逐步数学推导_独立版.pdf)。旧三源计划保留为研究历史，不代表本次 RFTrans-only 运行协议。
