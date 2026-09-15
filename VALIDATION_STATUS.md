# 最小验证实机状态

日期：2026-09-15。协议：`rftrans_only_minimal_v1`。

## 已实际完成

| 项目 | 实机结果 |
|---|---|
| GPU | 物理卡 0/3/5 均为 RTX 3090 24 GB，检查时空闲 |
| 环境 | Python 3.11.16，PyTorch 2.7.1+cu126，CUDA SDPA 与 BF16 通过 |
| 存储 | 环境、缓存、DINO 源码、权重目标、run 均在 `/ssd/polyu/TransDepth` |
| RFTrans | `train=5000`、`valid=1000`；RGB/depth/mask/recorder 逐 ID 完整配对 |
| Manifest | 6000 条，SHA256 `c89864bb11105c2f7f0f2cf5b720e3583a6e691b3ae0301060723c7769e6be56` |
| 划分 | R_train=4000、R_dev=500、R_select=250、R_confirm=250、R_hold=1000 |
| 深度编码 | PNG `raw*3/65536`；参考 EXR 为 `Y/HALF`；零值无效 |
| 编码抽查 | 训练侧四角色各 30 帧；PNG/EXR 最大差 0.7324 mm，审计通过 |
| Mask | 黑/红为背景、绿为透明、其余 ignore；120 帧抽查 unknown=0 |
| 可靠 patch | 四角色每图平均透明可靠 patch 约 36.9–52.8 |
| ClearGrasp | 官方 Val+Test 1,809,184,768 bytes，MD5 `88ffcbcfa6165969b380f572d4e655ca` |
| ClearGrasp inventory | 286 张真实 RGB + 940 张合成 RGB，共 1226 张；仅供推理 |
| DINO 源码 | 官方 commit `6876159a11b4df116f30f667f8c9888617df0751` |
| H+/16 随机结构 smoke | 840,644,352 参数；四层 `[1,1280,24,32]`；峰值分配约 3.66 GB |
| LoRA smoke | 10,752 参数；四层零初始化最大误差均为 0；up 梯度非零 |
| 代码检查 | Ruff 通过；28 个单元/真实资产测试通过；编辑器无错误 |

真实产物位于：

- `/ssd/polyu/TransDepth/data/derived/rftrans/manifests/`
- `/ssd/polyu/TransDepth/data/derived/rftrans/qa/encoding_v1/`
- `/ssd/polyu/TransDepth/data/derived/rftrans/qa/backbone/random_structure.json`
- `/ssd/polyu/TransDepth/data/raw/cleargrasp/test-val/`

## 当前阻断

官方 DINOv3 ViT-H+/16 LVD-1689M checkpoint 需要用户本人接受 DINOv3 许可并取得完整签名下载 URL。服务器未发现 Hugging Face 凭据；官方无签名 URL 实测返回 HTTP 403。随机权重 smoke 只能证明结构接入，不能用于训练或效果结论。

因此以下尚未实际执行：官方权重 strict load、T0、Oracle Search/Confirm、H1/H2 配对训练、R_hold 独立评价及 ClearGrasp RGB 推理。

## 当前有效性结论

**尚不能判断论文方法有效。** 数据、数学、模型结构和梯度路径已经通过最小工程验证；FAR 的研究有效性必须等待官方预训练权重，并由同一 T0 分叉的 H1/H2 在 RFTrans 留出集上的配对结果决定。即使 RFTrans 留出通过，也只能说明合成域证据；ClearGrasp 在本协议中只有定性推理，不能支持真实域定量有效性主张。
