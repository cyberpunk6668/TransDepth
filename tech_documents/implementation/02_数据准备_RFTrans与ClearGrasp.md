# 02｜数据准备：RFTrans-62CAD、ClearGrasp 合成与真实数据

状态：**实施规格，数据处理代码与项目 CLI 待实现；本次没有下载整套数据、读取你的训练数据或完成数据验收。**  
核实日期：2026-09-15。依据为本项目《FAR：62CAD＋ClearGrasp（含部分真实训练）局部修订与逐模块最小验证计划》的 M1、M2 与三来源划分协议。

本章的目标是让数据模块可以独立实现、独立检查，再接到训练模块。预期输出是：一份来源清单、若干不可变 JSONL 清单、统一米制深度、透明/背景/未知三态标签、384×512 数据张量，以及证明这些处理成立的审计材料。

文中 `src/transdepth/...`、`configs/...`、`python -m transdepth.cli.*` 均是**拟建项目路径和接口**；只有未来落地相应代码后，项目命令才能运行。官方仓库中的命名只作为解析证据，不能直接假定你的本地目录与其相同。

## 1. 输入、输出和本章的范围

### 1.1 三个来源必须分别登记

| 来源代号 | `source` 固定值 | 本方案使用 | 不进入最小主线的内容 |
|---|---|---|---|
| R | `rftrans_62cad` | 62CAD 的透明场景 RGB、理想几何深度、语义 mask、身份/相机记录 | RFNet、折射流监督、法线预测、active-depth 输入、全局深度优化 |
| S | `cleargrasp_synthetic` | ClearGrasp 原始合成 RGB、已发布 rectified depth、透明 mask、JSON 元数据 | RFTrans 用 ClearGrasp 模型重新渲染的数据不能冒充 S |
| Q | `cleargrasp_real` | 同一采集对的 transparent RGB、opaque-depth GT、可信 mask | opaque RGB 不作为主线输入或教师，transparent-depth 不作为 GT |

RFTrans 论文登记 5 类、62 个 CAD，并报告该来源的 5,000 张训练和 1,000 张合成测试图；另外用 ClearGrasp 的 9 个 CAD 重渲染了 5,000 张图。以上是**论文数量**，不是已经对你下载包核对过的样本计数。[RFTrans 论文 §III-D、§IV-A](https://arxiv.org/html/2311.12398v2)

### 1.2 交付给下游的两个数据接口

```text
训练/诊断接口：TrainingDataset → Sample(rgb、depth_m、mask、各valid、metadata)
部署接口：RgbOnlyDataset → rgb、sample_id、纯几何预处理记录
```

训练时数据加载器可以读取 GT；模型的 `forward(rgb)` 始终仅接收 RGB。相机内参只供坐标验收与数据几何变换使用。FAR 教师由训练程序显式构建，不能藏在 backbone 的 `masks` 参数、路径编码、输入通道或图像 padding 中。

终评时先由 RGB 清单生成预测并封存预测清单，再由独立评估入口读取深度和 mask。推理清单中不出现 GT 路径。

### 1.3 实现顺序和前置门槛

1. **P0：盘点。** 找到真实目录结构、文件配对、尺寸/通道/位深。
2. **P1：身份与职责。** 恢复组、划分 Train/Dev/Select/Confirm/Test，并检查跨源重叠。
3. **P2：标签解码。** 根据发布版本确定单位、坐标、无效值、mask 色表。
4. **P3：共同预处理。** 同步 letterbox，保存可逆几何记录，划出有效域。
5. **P4：可靠 patch。** 计算 24×32 网格的透明/背景可靠集合。
6. **P5：验收。** 抽查图、数值校验、清单 hash 和访问权限记录齐全后，才允许小样本训练。

P1 的独立划分不能靠 P5 的模型表现倒推。P2 的未知编码不能靠“训练看起来会收敛”代替验收。

## 2. 官方发布与源码：已经核实到什么程度

### 2.1 锁定本次参考版本

| 参考仓库 | 本次读取的 revision | 用途 |
|---|---|---|
| `LJY-XCX/Unity-RefractiveFlowRender` | `e08f0d313d71e98271c04a13080dbb9266e139d1` | 图像生成、PNG/EXR、mask 和相机路径 |
| `LJY-XCX/RFTrans` | `517eb83f801122b22e09bd439859c948c9eb6d50` | 已发布网络的数据读取语义 |
| `Shreeyak/cleargrasp` | `0688647e49380139e440bacbe8562132bb2afbd7` | GT 文件名、EXR 通道、合成预处理和真实采集 |
| HF `robotflow/rftrans` | `f5f465d2b63405f3a406ef9ca7d02428e04db057` | 远端发布文件列表 |

上表是在线源码/仓库版本证据。实施时还要保存本地压缩包 SHA-256、解压文件清单和每来源 `release_id`；它们不能由 Git commit 代替。

### 2.2 RFTrans 发布目录：说明文字与实际树不同

2026-09-15 查询到的 HF 文件树如下；数字后缀是压缩包分卷，不能把每卷当成独立 zip 解压：

```text
robotflow/rftrans/
├── train/
│   ├── train.zip.001
│   ├── ...
│   └── train.zip.011
├── valid/
│   ├── valid.zip.001
│   ├── valid.zip.002
│   └── valid.zip.003
├── cleargrasp/
│   ├── train.zip
│   └── intermediate.zip
└── Resources.zip
```

数据卡写 `train` / `val`，实际树是 `train` / `valid`；`cleargrasp` 是另一个重渲染来源，`Resources.zip` 是生成资产。最小主线只登记顶层 62CAD 数据，其他包默认排除；仍须核对解压后的内容与来源。[数据卡](https://huggingface.co/datasets/robotflow/rftrans)、[固定版本文件树](https://huggingface.co/datasets/robotflow/rftrans/tree/f5f465d2b63405f3a406ef9ca7d02428e04db057)

`official_split="valid"` 原样保留；`assigned_role="R_hold"` 是本项目角色。未经作者说明或样本身份核对，不在论文中写“已证明 HF valid 等于论文 test”。目录名只能说明包的名字。

### 2.3 RFTrans 生成器提供的候选配对模式

在上述生成器版本中，一帧输出使用 `RGB/rgb_<i>.png`、`depth/depth_<i>.png`、`depth/depth_<i>.exr`、`mask/mask_<i>.png`；另有 `normal`、`recorder/record_<i>.txt`、IR 和 calibration。RGB、理想深度和 mask 的生成调用为 512×512。mask 生成分支将透明对象标成绿色、不透明对象标成红色。[固定版本生成器](https://github.com/LJY-XCX/Unity-RefractiveFlowRender/blob/e08f0d313d71e98271c04a13080dbb9266e139d1/HDRPRefraction_Git/Assets/New/HDRPImageGenerator.cs)

这张表是 **adapter 首个候选布局**，必须通过真实解压目录验证：

| 项目字段 | 候选相对文件 | 是否训练必需 |
|---|---|---|
| `rgb_path` | `RGB/rgb_17.png` | 是 |
| `depth_gt_path` | `depth/depth_17.png` 或验收后的同帧 EXR | 是，二者明确选择一种规范源 |
| `depth_reference_path` | 同帧另一种格式 | 推荐，用于编码核查 |
| `mask_path` | `mask/mask_17.png` | 是，FAR 与区域均衡监督需要 |
| `metadata_path` | `recorder/record_17.txt` | 若存在必须解析，不能丢弃身份信息 |
| `active_depth_path` | `active_depth/active_depth_17.png` | 仅来源盘点；主线不读取 |

配对键为 `(release_id, source, official_split, 子目录命名空间, local_frame_id)`，禁止将每种模态分别排序后直接 `zip`。例如 RGB 缺 17、深度缺 18，数量仍相等，排序配对会静默错位。

### 2.4 ClearGrasp 发布入口与两类标签

官方 README 提供 [合成训练包](https://storage.googleapis.com/cleargrasp/cleargrasp-dataset-train.tar)、[验证与测试包](https://storage.googleapis.com/cleargrasp/cleargrasp-dataset-test-val.tar)；README 中的包大小是历史说明，规划磁盘时以实际下载元数据和展开尺寸为准。官方代码仓库包含少量样例，不能把样例目录当成完整数据集。[官方 README](https://github.com/Shreeyak/cleargrasp/blob/0688647e49380139e440bacbe8562132bb2afbd7/README.md)

| 类型 | RGB 文件模式 | 深度 GT 文件模式 | mask 候选模式 |
|---|---|---|---|
| 合成原始发布 | `<id>-rgb.jpg` | `<id>-depth-rectified.exr` | `<id>-segmentation-mask.png` 或发布版 `<id>-mask.png` |
| 真实 | `<id>-transparent-rgb-img.jpg` | `<id>-opaque-depth-img.exr` | `<id>-mask.png` |

官方评估器明确将 `-transparent-depth-img.exr` 作为输入传感器深度，将 `-opaque-depth-img.exr` 作为真实 GT；合成 GT 使用 `-depth-rectified.exr`。[官方评估器](https://github.com/Shreeyak/cleargrasp/blob/0688647e49380139e440bacbe8562132bb2afbd7/eval_depth_completion/eval_depth_completion.py)

样例合成目录存在 `rgb-imgs`、`depth-imgs-rectified`、`segmentation-masks`、`json-files`。完整训练包也可能同时有 `source-files` 和 `resized-files`。**只选择一套匹配分辨率的数据**，在 manifest 登记其变体；不把一个场景的原图与缩小图当成两张独立训练样本。`*-cameraNormals.png` 等可视化图片不匹配 RGB 输入白名单。[官方样例树](https://github.com/Shreeyak/cleargrasp/tree/0688647e49380139e440bacbe8562132bb2afbd7/data/sample_dataset)

## 3. 用户提供路径后如何接入

### 3.1 需要填写的最小信息

下面的信息可以分批补齐。当前缺少路径不会阻止代码设计；会阻止真实数据清单和数据验收。

| 配置项 | 需要用户提供/现场读取的内容 | 缺失时的处理 |
|---|---|---|
| `rftrans_root` | 62CAD 解压根目录，或分卷所在目录 | 不猜测磁盘位置 |
| `cg_train_root` | ClearGrasp 原始合成训练包位置 | 与 RFTrans-CG 重渲染分开 |
| `cg_eval_root` | ClearGrasp 原验证/测试包位置 | 盘点 real-known、real-novel、synthetic 的实际名字 |
| `rftrans_release` | 下载链接/revision、是否改过文件、是否自渲染 | 未知记 `unverified_local_release` |
| `cg_release` | 下载来源、是否已经 resize/rectify/补洞 | 不能按后缀自动放行 |
| `group_metadata` | 采集序列、场景关联、双相机对应、配对/对象表 | 先恢复身份，再决定能否独立分组 |
| `camera_metadata` | 每来源相机参数/生成配置 | 不从 RGB 尺寸杜撰焦距 |
| `derived_root` | 可写的派生缓存目录与磁盘容量 | 不在原始目录覆盖生成 |
| `artifacts_root` | QA 图、报告、清单保存目录 | 与代码和大数据分离 |

### 3.2 本地路径配置，不提交实际私人路径

建议提交 `configs/paths.example.yaml`，将以下实际填写后的文件保存在被 Git 忽略的 `configs/paths.local.yaml`。下面是**格式示例，不是你已有的服务器路径**：

```yaml
roots:
  rftrans_62cad: /data/raw/rftrans_62cad
  cleargrasp_train: /data/raw/cleargrasp_train
  cleargrasp_eval: /data/raw/cleargrasp_test_val
storage:
  manifests: /data/derived/transdepth/manifests
  cache: /data/derived/transdepth/cache
  runs: /data/experiments/transdepth
  pretrained: /data/models/dinov3
identity_tables:
  group_overrides: null
  cad_aliases: null
```

来源版本与验收状态放在`configs/data/releases_v1.yaml`，不能通过只允许改路径的`paths.local.yaml`覆盖。示例：

```yaml
releases:
  rftrans_62cad:
    revision: f5f465d2b63405f3a406ef9ca7d02428e04db057
    archive_sha256_file: /data/metadata/rftrans_archives.sha256
    local_content_matches_release: false  # 验收前保持 false
  cleargrasp:
    source: official_google_storage
    archive_sha256_file: /data/metadata/cleargrasp_archives.sha256
    local_content_matches_release: false

```

路径解析器支持 POSIX 绝对路径和 Windows 路径。JSONL 中使用 `root_key` 加相对 POSIX 路径，避免远端服务器依赖 `F:\...`。解析后确认路径仍位于已登记根目录内；存在符号链接时同时登记解析后目标。清单文件不使用 Python pickle，不能把配置当作可执行代码。

### 3.3 原始文件、派生文件和报告的建议布局

```text
/data/raw/                         # 原始包展开，之后只读
  rftrans_62cad/
  cleargrasp_train/
  cleargrasp_test_val/
/data/derived/transdepth/
  <preprocessing_version>/
    <source>/<sample_id_hash>.npz # 可选缓存，不是唯一真值来源
/data/experiments/transdepth/
  data-audit/<audit_id>/
    inventory.json
    inventory_files.jsonl
    inventory_samples.jsonl
    pairing_errors.jsonl
    encoding_evidence.json
    source_qa/
    patch_qa/
    split_audit.json
    cad_overlap.csv
    excluded_samples.jsonl
    manifest_checksums.sha256
```

缓存记录原文件 SHA-256、解析器版本、编码规格、几何参数、mask 规则。任何一项变化都使旧缓存失效。主线采用 float32 米制深度缓存、uint8 类别和 bool valid；不把深度重新量化到 8 位图片。压缩 NPZ 是启动方案，先测 I/O 再决定是否换分片格式，避免一次加载全数据进内存。

## 4. 模块 D0：盘点、配对与 JSONL manifest

### 4.1 先盘点，后解码，最后纳入训练

`inventory` 只建立文件元数据和身份候选，不随机划分，不修改图像。实施过程：

1. 遍历显式来源根目录，登记相对路径、字节数、扩展名和 archive/release 来源。
2. 按适配器的文件名白名单识别 RGB/depth/mask/metadata；未知文件保留在盘点表中。
3. 解析局部 frame ID 与父目录命名空间，建立字典式模态配对。
4. 缺 RGB、缺 GT、缺 mask、同模态重复、不同尺寸、编号冲突分别报告。
5. 逐文件计算 SHA-256；大包先做压缩完整性检查，不能仅检查总字节数。
6. 所有通过配对的样本先赋 `assigned_role="unassigned"`、`qa_status="pending"`。
7. 单独登记被排除的 RFTrans-CG、无效文件、历史缓存，不从扫描结果静默消失。

对已下载分卷，使用支持相应分卷格式的归档工具从 `.001` 检查与解压，保持所有分卷同目录，先通过压缩完整性测试。不要用文本拼接命令处理二进制文件。新增解压目录后检查内部路径安全、实际条目数与磁盘余量。这些是数据落地步骤，本次尚未执行。

### 4.2 manifest 的稳定字段

JSONL 一行一个样本。人类可读字段允许 `"unknown"`，数值/矩阵未知写 `null`；不要让字符串 `"unknown"` 出现在数值数组里。

| 字段 | 类型 | 约束/用途 |
|---|---|---|
| `schema_version` | string | 初始 `far_data_v1` |
| `sample_id` | string | 全库唯一、稳定；来源+release+命名空间+frame，不依赖清单排序 |
| `source`, `release_id` | string | 三来源枚举及实际版本 |
| `official_split`, `assigned_role` | string | 原始职责与本研究职责分开 |
| `root_key` | string | 在 `paths.local.yaml` 中解析 |
| `rgb_path`, `depth_gt_path`, `mask_path` | string/null | 相对路径，训练样本三者按角色要求校验 |
| `depth_reference_path`, `metadata_path`, `valid_mask_path` | string/null | 对照格式、采集记录、额外有效域；不存在写 null |
| `raw_hw`, `rgb_hw`, `depth_hw`, `mask_hw` | `[int,int]` | 统一 H,W 顺序；原始不对齐时禁止直接堆叠 |
| `rgb_sha256`, `depth_sha256`, `mask_sha256` | string | 原始字节 hash |
| `rgb_pixel_sha256` | string | RGB 解码后的像素 hash，用于无损重封装重复检查 |
| `depth_encoding` | string | 如 `rftrans_ideal_png_u16_3_over_65536`；未知不放行 |
| `unit_to_m` | float/null | PNG 可为 `3/65536`，米制 EXR 为 1；不能从后缀猜 |
| `depth_coordinate` | string | `optical_z` / `ray_distance` / `unknown` |
| `canonical_depth_coordinate` | string | 放行后固定 `optical_z` |
| `depth_channel`, `raw_channel_types` | string/object | EXR 显式 R 等通道；逐通道记录 HALF/FLOAT/UINT |
| `invalid_rule_id`, `mask_rule_id` | string | 对应冻结规则配置，不能存不可追踪 lambda |
| `camera_id`, `camera_model` | string | 后者例如 pinhole/带畸变/unknown |
| `K_raw`, `distortion`, `extrinsics_ref` | array/null/string | 验收/配准用；不传给预测器 |
| `scene_id`, `capture_id`, `pair_group_id` | string | 由证据恢复；同编号跨目录不自动等价 |
| `object_ids`, `cad_ids` | string list | 未知可用 `["unknown"]`；空列表表示已确认没有目标对象 |
| `leakage_group_id`, `group_confidence` | string | 最高共同采集/场景组及证据等级 |
| `identity_evidence_refs` | string list | 元数据/人工匹配表/归档说明引用 |
| `qa_status`, `qa_reason_codes` | string/list | `pending` / `passed` / `quarantined` |
| `preprocessing_version` | string | 冻结的几何与类别处理版本 |
| `canonical_path` | string/null | 可选派生缓存位置 |

简化示例只演示结构；其中路径和 ID 不是已经发现的真实样本：

```json
{"schema_version":"far_data_v1","sample_id":"R-release1-train-rgb17","source":"rftrans_62cad","release_id":"release1","official_split":"train","assigned_role":"unassigned","root_key":"rftrans_62cad","rgb_path":"train/RGB/rgb_17.png","depth_gt_path":"train/depth/depth_17.png","mask_path":"train/mask/mask_17.png","depth_encoding":"unknown","unit_to_m":null,"depth_coordinate":"unknown","scene_id":"unknown","capture_id":"unknown","pair_group_id":"unknown","leakage_group_id":"unknown","cad_ids":["unknown"],"qa_status":"pending","preprocessing_version":"pending"}
```

### 4.3 错误不能伪装为一个合法样本

- 缺 mask 不返回全背景；缺 GT 不返回全零后继续平均。
- 图像损坏、模态不齐、未验收坐标写入隔离清单；正式训练构造 dataset 时遇到这些记录直接失败。
- DataLoader worker 不能 `except: return dataset[random_index]`；这种回退会改变曝光、配对和域比例。
- 同名 ID 不唯一时输出所有冲突路径，不擅自保留“第一个”。
- 允许单独 RGB 推理没有 GT，但它使用独立 schema 和 Dataset，不通过训练样本的缺失值分支实现。

## 5. 模块 D1：深度编码、光轴坐标与首表面

### 5.1 RFTrans 理想 PNG 与 active PNG 不同

官方 `generate_active_depth.py` 的可视化分支对理想 PNG 使用 `raw × 3 / 2^16`；同一脚本把生成的 active depth 乘 1000 保存为 uint16 毫米。两者解码常数不同。最小主线只使用验收后的理想深度。[固定版本脚本](https://github.com/LJY-XCX/Unity-RefractiveFlowRender/blob/e08f0d313d71e98271c04a13080dbb9266e139d1/generate_active_depth.py)

候选解析器应当执行：

```python
# 待实现伪代码：只允许经验证的 release 使用此规格。
raw = read_png_unchanged(path)       # 不能先 convert("L") 或转 uint8
assert raw.ndim == 2
assert storage_bit_depth(path) == 16
depth_m = raw.astype(np.float32) * (3.0 / 65536.0)
valid = apply_verified_invalid_rule(raw, depth_m, spec)
```

这里分母明确为 **65536**，不是 65535；不因常见 UNORM 习惯而静默改公式。1 个原始码值对应约 0.0458 mm；最大 uint16 码值会解为略小于 3 m。这不证明最大码值就是可信的 3 m 表面，far-plane/饱和是否无效须独立确定。

`uint16` 的数组容器本身也不足以证明原文件位深：有些读库把 16 位 PNG 返回 int32。验收同时记录 PNG 头信息与原始数值范围，允许经验证的 int32 容器，禁止 8 位丢精度路径。

### 5.2 不要误读 shadergraph：本版输出链支持光轴 z

本次逐边检查了官方 shadergraph。实际连到输出的链是 `Scene Depth → Remap → BaseColor`；`m_DepthSamplingMode=2`。`Distance(Position, Camera)` 虽然存在，但它的结果**没有连接到 Remap 或输出**。Unity 对应实现的枚举 2 是 Eye，并调用 `LinearEyeDepth`。因此本版源码支持“理想深度为线性 eye/光轴深度”的解释，不支持因为图里有 Distance 节点就再做 ray→z 转换。[RFTrans shadergraph](https://github.com/LJY-XCX/Unity-RefractiveFlowRender/blob/e08f0d313d71e98271c04a13080dbb9266e139d1/HDRPRefraction_Git/Assets/New/HDRPCamera/Resources/HDRPCameraDepth.shadergraph)、[Unity 2022.2 SceneDepthNode](https://github.com/Unity-Technologies/Graphics/blob/2022.2/staging/Packages/com.unity.shadergraph/Editor/Data/Nodes/Input/Scene/SceneDepthNode.cs)

生成器对 PNG 调用 `GetDepth(..., 0, 3)`，对 EXR 调用 `GetDepthEXR(..., 0, 1)`；相机脚本分别使用 R16 与 RFloat 渲染纹理。[相机脚本](https://github.com/LJY-XCX/Unity-RefractiveFlowRender/blob/e08f0d313d71e98271c04a13080dbb9266e139d1/HDRPRefraction_Git/Assets/New/HDRPCamera/Scripts/HDRPCameraAttr.cs)

**边界：上述结论针对锁定源码，不等于已证明用户磁盘上的每个包都由该版本生成。** 下列证据齐全才把本地 release 的 `depth_coordinate` 从 `unknown` 改成 `optical_z`：

1. 同帧 PNG 与 EXR 对齐，尺寸、方向、结构一致。
2. 对非零、非饱和、远离轮廓的共同有效像素，比对 `PNG×3/65536` 和 EXR；保存逐帧误差与径向误差。
3. 误差容差由 PNG 量化与 EXR 实际 HALF/FLOAT 精度预算给出；约 1 mm 可作为初次筛查提示，不能当作任意放行容差。
4. 用源相机/生成几何检查前表面，尤其杯口、薄壁、透明对象遮挡背景的位置。
5. 平面/几何证据检查中心和角落；若猜错 ray/z，误差通常随离主点距离系统变化。

PNG 与 EXR 一致只能证明两种存储一致；它们可能共享同一个渲染语义错误，不能单独证明首表面。已知垂直光轴的平面应具有近常数 z；一般斜桌面不能直接做“像素值恒定”检查。

### 5.3 仅在证据说明原始是 ray distance 时转换

对已知针孔内参、已经处理畸变的原始图像：

\[
\mathbf d=K^{-1}[u,v,1]^\top,\qquad
Z=r\frac{d_z}{\|\mathbf d\|_2}.
\]

标准内参使 `d_z=1`，得到计划中的 `Z=r/||K^{-1}[u,v,1]^T||`。必须先统一坐标定义与像素中心约定。对未校正畸变的传感器数据先由对应相机模型求射线，不能硬套 K。

这一步在**原始分辨率、有效内容区域**执行一次，然后才 resize。已是光轴 z、已发布 rectified、或经过该转换的缓存不能再做第二次。深度取米后不做逐图 min/max、均值标准化或 test-GT 尺度拟合。

### 5.4 ClearGrasp 合成的 `rectified` 是发布标签协议

官方历史预处理代码将相机中心距离乘横纵 cos 矩阵生成 `depth-rectified.exr`；该代码用 FOV/图像尺寸构造逐像素角度。它的离散实现并不自动等同于上节精确针孔射线公式。因此主线**直接使用已发布 rectified 文件**，不根据自己的推导重新校正整套 CG 后仍称同一发布标签。若仅有未校正 EXR，应先找到匹配发布版本的处理依据；更精确的重建属于新标签版本，需另立审计与对照。[ClearGrasp 预处理源码](https://github.com/Shreeyak/cleargrasp/blob/0688647e49380139e440bacbe8562132bb2afbd7/z-ignore-scripts-helper/data_processing_script.py)

### 5.5 EXR 通道与位深的读取合同

ClearGrasp 官方 `exr_loader(ndim=1)` 显式读取 `R` 通道并请求 float32 输出；官方 saver 可将二维深度复制到 RGB，并按 HALF 写入。因此“读取后 float32”不能证明源文件是 32 位 FLOAT。[ClearGrasp EXR 工具](https://github.com/Shreeyak/cleargrasp/blob/0688647e49380139e440bacbe8562132bb2afbd7/api/utils.py)

新实现必须：

1. 先读取 EXR header：`dataWindow`、`displayWindow`、通道名、每通道 pixel type、压缩类型。
2. 按来源规格选通道。CG 预期 R；RFTrans EXR 以实际 header 与同帧对照确定。没有 R 时不能自动换“第一个通道”。
3. 若同时有 RGB，检查是否为重复深度；不把三通道深度平均，不进行 BGR/RGB 猜测。
4. 返回连续内存 float32 `[H,W]`，保留源 channel type 字段；保留 NaN/Inf 统计，之后明确生成 invalid。
5. 对非零 dataWindow 原点或窗口大小不匹配的文件先报错，显式恢复到原图坐标后才放行。
6. 启动环境做一个实际 EXR 样例读取检查。OpenCV 的 EXR 支持依赖构建，不把某台机器“能读”视为所有服务器都能读；主读取后端选择官方 OpenEXR Python 库并锁环境。

米制、光轴、通道正确是三个不同结论；后缀 `.exr` 对它们都不是充分证据。

### 5.6 真实 opaque-depth 和配准

ClearGrasp 采集工具保存透明/喷涂替代两次采集。源码的 opaque GT 会聚合后续多帧；底层 RealSense 路径将 depth 对齐 color，并乘传感器 `depth_scale`。这提供了“RGB 坐标中的米制 GT”来源证据，但不能消除替换对象的位置误差，也不能证明本地二次加工包没有错位。[采集程序](https://github.com/Shreeyak/cleargrasp/blob/0688647e49380139e440bacbe8562132bb2afbd7/dataset_capture_gui/capture_image.py)、[RealSense C++ 对齐](https://github.com/Shreeyak/cleargrasp/blob/0688647e49380139e440bacbe8562132bb2afbd7/live_demo/realsense/realsense.cpp)、[相机单位转换](https://github.com/Shreeyak/cleargrasp/blob/0688647e49380139e440bacbe8562132bb2afbd7/live_demo/realsense/camera.py)

真实训练标签验收按顺序进行：

- 用完整相对命名空间和 frame ID 配对 transparent RGB、opaque depth、mask；opaque RGB 仅供本阶段离线 QA。
- 分相机检查尺寸、镜像、上下翻转、裁切、主点与已有校正；不同相机不能借用一份 K。
- 看稳定背景结构的 RGB/深度边界对齐，区分相机配准问题与对象替换问题。
- 看透明轮廓与喷涂替代物轮廓偏移，保存偏移示例与每组异常，不用高反光纹理当几何边界真值。
- GT 全零/NaN 区不能补成透明传感器深度；已存在的官方补洞/滤波需登记来源。
- 首轮不添加自动光流或单应性配准来“修复”对象替换：深度值是几何量，不可任意二维扭曲后仍称原 GT。
- 若证据要求扩大不可信边缘，在 Train/Dev 上确定固定规则；H1/H2 一致使用，冻结后用于 Test。不能针对测试误差逐图擦除边界。

## 6. 模块 D2：透明类别、ignore 与有效域

### 6.1 RFTrans 的绿色标签必须按语义解析

RFTrans 官方 loader 读取 `mask[...,1]/255`。这不是“RGB 任意通道非零就是透明”；红色不透明对象按任意非零规则会被误标透明。[固定版本 RFTrans loader](https://github.com/LJY-XCX/RFTrans/blob/517eb83f801122b22e09bd439859c948c9eb6d50/pytorch_networks/rgb2normal/dataloader.py)

设计采用三态 `labels`：`0=已知背景/非透明`、`1=可信透明`、`255=unknown/ignore`。建议首个 RFTrans 规则：

1. 保存 RGB 调色板与 G 通道直方图；先确认透明确为 G=255、已知非透明为 G=0。
2. 纯绿色标 1；有证据的红色/背景色标 0。
3. 抗锯齿产生的中间值、未知颜色、无法解释的 alpha 标 255；不把所有 G>0 像素自动扩张成透明。
4. 数据若原本为灰度语义图，进入另外的已验收规则，不能仍假定有绿色通道。
5. 缺 mask 的图片隔离，或另立仅深度监督数据变体；主线不默认为全背景。

阈值与色表是 release 级配置，不能每图自适应。初始使用精确可信颜色并将其余 ignore；若覆盖损失过大，先查原色表，再在 Train/Dev 上预注册阈值，而不是随意放宽直到 FAR 有足够样本。

### 6.2 ClearGrasp mask 不能与 outlines 混淆

ClearGrasp 官方 masks loader 使用灰度阈值 `mask>=100` 得到前景；生成的 synthetic segmentation mask 常见为 0/255。正式适配器必须先检查实际值域，记录是否沿用发布阈值、是否存在软边界；历史 loader 的阈值不是全部未知版本的许可证。[官方 masks loader](https://github.com/Shreeyak/cleargrasp/blob/0688647e49380139e440bacbe8562132bb2afbd7/pytorch_networks/masks/dataloader.py)

- `segmentation-mask` / `mask` 是透明对象区域；`outlineSegmentation` 的多类标签不是区域 mask。
- `variantMasks.exr` 是实例/变体标识；若需要重建语义图，结合对应 JSON 的 ID 表，而非 `value>0` 无条件合并。
- 空洞、杯口和柄内空白保留；不得用轮廓填充函数将整个外轮廓内部都视为实体。
- 类别未知与深度无效分别存储。未知的分类不等于已知背景。

### 6.3 四个有效域与主线补齐选择

定义所有张量均为同一坐标系的 bool `[H,W]`：

\[
V_{depth}=V_{content}\cap V_{finite}\cap V_{source-rule},\quad
V_{known}=\{labels\in\{0,1\}\},
\]

\[
V_{rel}=V_{depth}\cap V_{known},\qquad
V_{seg}=V_{content}\cap V_{known}.
\]

本设计补齐选择：主线的区域均衡深度项采用 `valid_depth & mask_known`；`valid_depth` 本身仍保存全部可信深度，方便诊断。**类别 unknown 的深度暂不进入主线 loss**，统计因此损失的有效深度覆盖率。原修订允许选择保留这些深度；若后续启用，应命名独立变体，明确第三类像素的权重与归一化，不能偷偷当作背景或加入第三组均值。

没有深度的可信类别像素可以参与 C 的分割项；有深度但类别未知的像素不能参与 FAR 或透明/背景区域均值。padding 不能进入任意监督、可靠 patch 或指标。

### 6.4 无效深度规则的配置设计

```yaml
# 待核验的配置格式；null 表示禁止放行，不是自动取默认范围。
depth_rules:
  rftrans_62cad:
    encoding: rftrans_ideal_png_u16_3_over_65536
    raw_zero_invalid: null
    raw_max_invalid: null
    range_policy: pending   # 验收后明确选择 no_extra_clip 或 fixed_interval
    accepted_depth_m: null
    coordinate: unknown
    evidence_file: null
  cleargrasp_synthetic:
    encoding: exr_rectified_m
    channel: R
    finite_required: true
    positive_required: true
    range_policy: pending   # 验收后明确选择 no_extra_clip 或 fixed_interval
    accepted_depth_m: null
  cleargrasp_real:
    encoding: exr_opaque_m
    channel: R
    finite_required: true
    positive_required: true
    range_policy: pending   # 验收后明确选择 no_extra_clip 或 fixed_interval
    accepted_depth_m: null
```

`range_policy=pending`在验收前阻止正式运行。验收后明确写`no_extra_clip`时允许`accepted_depth_m=null`，表示不加额外距离裁切；选择`fixed_interval`则必须给出[min,max]。对有限正数以外是否还需裁切，必须由发布约定/Train/Dev决定并记录。没有依据就不凭空设置 0.1–10 m；未知的零值/饱和值含义则是放行阻断项。

## 7. 模块 D3：384×512 letterbox 与内参

### 7.1 一个共同变换作用于全部模态

目标 H=384、W=512，patch=16，因此图像 token 网格固定 24×32。对原图 H0,W0：

\[
s=\min(384/H_0,512/W_0),\quad
H_1=\operatorname{round}(sH_0),\quad W_1=\operatorname{round}(sW_0).
\]

把整数量化规则锁定为同一实现，H1/W1 最少为 1、最大不超目标。实际尺度 `sy=H1/H0`、`sx=W1/W0` 分别保存，不能用未取整的 s 代替。

```text
pad_top  = floor((384-H1)/2)       pad_bottom = 384-H1-pad_top
pad_left = floor((512-W1)/2)       pad_right  = 512-W1-pad_left
```

首轮实现固定以下插值规则：RGB 使用一个锁定后端的 bilinear + antialias；depth、labels、valid 使用共同中心坐标映射的最近邻，禁止对类别或深度跨边界双线性混合。RGB 先缩放，再按骨干预处理规范归一化；padding 可在归一化后填 0，与填模型通道均值等价，明确保存该约定。深度 padding=0，labels padding=255，valid padding=false。

建议共同后端使用 `torch.nn.functional.interpolate`：RGB 为 `mode="bilinear", align_corners=False, antialias=True`，离散标签和深度为 `mode="nearest-exact"`。后者与普通 `nearest` 的采样行为不同，项目应锁定一种，不能 RGB/PIL、depth/OpenCV、mask/PyTorch 各取不同像素中心规则。bool/整数标签可暂转 float32 采样后恢复原类型，不插值生成类别。[PyTorch 2.7 interpolate 文档](https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.interpolate.html)

**不要将 512×512 的 RFTrans 拉伸成 384×512。** 它应变成 384×384，左右各补 64 像素；内容比例 75%，padding 比例 25%。原图若是 288×512 则保持内容尺寸，上下各补 48，padding 同为 25%。这些是形状算例，不代表全部数据都拥有这些原始尺寸。

### 7.2 像素中心和 K 变换必须成对定义

本设计使用整数坐标表示像素中心，resize 对应半像素中心映射：

\[
u'=s_x(u+0.5)-0.5+p_l,\quad
v'=s_y(v+0.5)-0.5+p_t.
\]

于是

\[
A=\begin{bmatrix}
s_x&0&p_l+(s_x-1)/2\\
0&s_y&p_t+(s_y-1)/2\\
0&0&1
\end{bmatrix},\qquad K'=AK.
\]

具体为 `fx'=sx*fx`、`fy'=sy*fy`、`cx'=sx*(cx+0.5)-0.5+pad_left`、`cy'=sy*(cy+0.5)-0.5+pad_top`。如果实际 resize 后端采用另一种像素中心定义，应同时调整该公式并通过投影验证；不能将 half-pixel 图像采样与角点式 `cx'=s*cx+pad` 混用。

统一保存：`original_hw`、`resized_hw`、`output_hw`、四边 padding、`scale_xy`、`pixel_center_convention`、A、`K_raw`、`K_processed`。验证时用若干 3D 点比较 `A·project(K,X)` 与 `project(K',X)`；这种测试检查坐标变换，不需要训练模型。

图像 resize 不改变光轴深度数值的物理单位。不能将深度值乘 s。DINO 输出 patch 是对整张 padded 图像的网格，可靠 patch 仅由有效内容支持。

### 7.3 训练增强的最小方案

- M3 小样本拟合关闭全部随机增强。
- 正式训练起点采用冻结的轻量 RGB 颜色增强，不改 GT 数值和 mask。
- 第一轮不做随机 crop/旋转，减少几何检查变量。之后若加水平翻转，RGB、GT、valid、labels 共同变换；K/几何记录也要对应镜像。
- 每个样本曝光使用可复现的 `augmentation_seed`；H1/H2 对应曝光使用同一随机流。
- Dev/Select/Confirm/Test 只使用确定性共同预处理。
- 源图若带 EXIF 方向，方向变换必须同步应用于标签；不能只让 RGB 库自动旋转。

### 7.4 评价与导出的坐标

主评价在固定 384×512 的有效内容坐标内，去除 padding，使用按相同最近邻采样的 GT。边界带的 3 像素宽度指该评价坐标。可另输出恢复原分辨率的深度用于展示，保存插值方式；该结果不与主协议混表。不要将 GT 变回原分辨率多次插值，也不要在 Test 上改 letterbox 改善数字。

## 8. 模块 D4：可靠 patch 的精确实现

### 8.1 输入和输出

输入：`depth_m[384,512]`、`labels[384,512]`、`valid_depth[384,512]`、`content_mask[384,512]`。输出使用 `[24,32]` 或 row-major `[768]`，flatten 约定 `index=row*32+col`。全 token 下标由 backbone adapter 加 prefix 偏移；数据模块只产生 patch 下标，不把 CLS/register 标为图像 patch。

每个 16×16 patch P：

\[
v=|V_{rel}\cap P|/256,\qquad
t=|V_{rel}\cap\{labels=1\}\cap P|/|V_{rel}\cap P|,
\]

\[
z=\operatorname{median}\{Z_p:p\in V_{rel}\cap P\},\quad
iqr=Q_{.75}(\log(Z/1m))-Q_{.25}(\log(Z/1m)).
\]

先检查 `content_mask[patch].all()`：**只要 patch 中含一个 padding 像素，整个 patch 就归 U**。对完全位于内容区域内的 patch，`v>=0.9`、`t>=0.9`、`iqr<=0.1` 判为透明可靠 query/key T；将 `t>=0.9` 换为 `1-t>=0.9` 得背景 B，其余 U。Krel=T∪B。由于面积 256，`v>=0.9` 至少要求 231 个有效像素。空分母直接 U，不能让 NaN 被 `nan_to_num` 变成背景。

### 8.2 为什么要分清几个分母

- `v` 分母为完整 256 像素，缺深度和 unknown 会降低可靠度。含 padding 的 patch 另有整块排除门槛；不能因有效像素占比仍大于 90% 而放行。
- `t` 分母只有深度和类别均可信的像素；unknown 不进入透明或背景。
- 深度中位数与 log-IQR 使用相同 Vrel；不能 median 用整 patch、纯度用有效域。
- 中位数和 quantile 使用锁定定义，例如 `torch.quantile(..., interpolation="linear")`；偶数样本中位数不能有的路径取下中位数、有的取两数均值。
- 可靠集合应在此次增强/resize 之后计算；不能复用前一幅几何变换的 patch 标签。

待实现伪代码：

```python
def reliable_patches(depth_m, labels, valid_depth, content_mask, cfg):
    valid_rel = valid_depth & content_mask & (labels != 255)
    result = empty_patch_targets(hw=(24, 32), state="U")
    for row, col, patch in iter_nonoverlap_patches(size=16):
        if not content_mask[patch].all():
            continue                       # partial padding 也整块排除
        v = valid_rel[patch]
        n = int(v.sum())
        if n == 0:
            continue
        z = depth_m[patch][v].float()
        assert torch.isfinite(z).all() and (z > 0).all()
        valid_fraction = n / 256
        transparent_fraction = (labels[patch][v] == 1).float().mean()
        q25, q50, q75 = torch.quantile(z.log(), q_tensor([.25, .50, .75]))
        depth_median = torch.quantile(z, .5)
        if valid_fraction >= .9 and (q75-q25) <= .1:
            if transparent_fraction >= .9:
                result.state[row, col] = "T"
            elif 1-transparent_fraction >= .9:
                result.state[row, col] = "B"
        result.save_stats(row, col, depth_median, valid_fraction,
                          transparent_fraction, q75-q25)
    return result
```

公式中的 `log(Z/1m)` 在数值上是以米存储的正数取 log，不能对 mm 直接求配套绝对量或对 0 加任意 epsilon 掩盖无效域。

### 8.3 每来源必须输出的 patch 统计

| 字段 | 解释 |
|---|---|
| `num_T`, `num_B`, `num_Krel` | 单图可靠透明、背景与全部 key 数 |
| `transparent_coverage` | 可靠透明 patch 覆盖的可信透明像素 / 全部可信透明像素 |
| `unknown_fraction`, `padding_fraction` | unknown 和 padding 占比，分别记录 |
| `valid_depth_not_used_fraction` | 深度有效但主线区域监督未使用的像素比例 |
| `no_relation_sample_rate` | T和B均无可用query，关系loss为0的样本比例 |
| `no_transparent_intervention_rate` | 无可靠透明query或无可靠key的样本比例；仍有B时可以存在保持A0的背景KL |
| `num_queries_used` | 后续每组最多 128 query 抽样后实际使用数，与候选数分开 |

没有可靠关系的样本仍可有深度/分割损失。FAR loss仅在单图内对非空T/B类别归一；整图无关系时为0，仍占全局16图均值的一份。无T但有B仍计算背景保持KL；不能生成伪 patch 让统计变好。若 Q 几乎没有可靠关系，结果中明确报告，并另立“Q 仅深度/分割，FAR 仅合成”的命名变体；主协议不要静默变更。

## 9. 模块 D5：分组划分与泄漏审计

### 9.1 原始来源与研究角色映射

| 原来源 | 本研究角色 | 目标比例/权限 |
|---|---|---|
| RFTrans 62CAD 官方 train | `R_train`, `R_dev` | 按组约 90/10 |
| RFTrans 62CAD 官方留出包 | `R_hold` | 不参与本轮模型选择；先核对发布职责 |
| ClearGrasp 原始合成 train | `S_train`, `S_dev` | 按组约 90/10 |
| ClearGrasp 独立合成 validation | `S_select`, `S_confirm` | 按组约 50/50，oracle 筛选/一次确认 |
| ClearGrasp 原 real-known | `Q_train`, `Q_dev`, `Q_test_known` | 按最高共同组约 60/20/20 |
| ClearGrasp 原 real-novel | `Q_test_novel` | 整组封存，主终评域 |
| ClearGrasp 原 Syn-novel | `S_test_novel` | 整组封存，辅助终评域 |

比例是组级目标，不是精确图像数量；输出实际组数与图数，不强行拆大组凑满 60%。发布目录 `real-val` 等如何映射到论文 known/novel 必须核对说明与对象身份，不能字面自动分配。

### 9.2 如何构建 `leakage_group_id`

建议使用 union-find/连通分量完成绑定：

1. 每个样本先为独立节点，但这不等于已有独立性证据。
2. 明确同一场景、采集序列、近邻视角、同次采集不同相机、透明/喷涂对建立边。
3. 原图/resize 版本、重复导出、同几何换材质或背景的受控变体建立边。
4. 每条边保存 `reason`、`evidence_ref`、`confidence`、建立者/规则版本。
5. 连通分量整体作为泄漏组，组 ID 由成员稳定标识生成；图像顺序改变不能改变分组。
6. 真正独立的同 CAD 不必全部合并为场景组，但 `cad_id` 必须进入形状重叠审计。

RFTrans recorder 中的 prefab、相机、位置等字段可作为线索，但必须检查记录时机与字段含义。不能把文件编号每 10 张划一个 scene，也不能仅因相邻编号就断言同次采集。元数据中的初始落体位置与最终渲染状态可能不同，使用时应验证。

真实数据不能靠同编号跨 d415/d435 自动判断同一场景，也不能默认不同相机是不同组。缺明确配对表时，以背景结构、物体布局和可确认采集信息建立人工审计表。

### 9.3 划分算法的可复现规格

- 使用独立的 `split_seed=17` 作为本设计起点，与训练种子概念分开；正式清单冻结后训练种子不重新划分数据。
- 在组不可拆前提下，按可确认对象、距离区间、透明面积和相机分布近似分层。
- 距离/面积分层只用准许划分池内的标签/元数据；阈值冻结前不通过 Test 表现挑分组。
- 小数据集可用确定性贪心分配，再输出偏离目标的统计；目标是可追踪且不拆组，不是精确最优比例。
- 独立组不足时预先定义分组折与确认规则；不能在多个划分里训练后挑最有利的一份。
- 若组身份无法恢复，记录 `group_confidence=unknown`。可以继续探索性“图像留出”方案，但独立场景泛化验收未通过，不能在文档中称独立真实场景测试。

### 9.4 跨来源重复与 CAD 重叠

进行四层检查，输出候选和人工判定，不仅给一个“无泄漏”布尔值：

1. 原文件 SHA-256：完全同文件。
2. 解码 RGB 像素 hash：无损重封装/元数据差异。
3. 低分辨率图像近重复检索：发现 JPEG 重编码、resize、少量裁切等候选；阈值只产生候选，不自动证明场景相同。
4. 对象/CAD 别名表：名称、模型来源、网格指纹/可确认几何对应，检查训练/Dev/Select/Confirm 与 novel 留出。

`cad_aliases.csv` 建议字段：`source, release_id, raw_cad_id, canonical_cad_id, evidence, status`。名称不同不等于形状不同；同名也不自动相同。原 novel 只有在联合训练和选择数据均未出现相同几何身份时，才称相对联合数据未见形状；否则报告已见/未见/未知身份子集，或称原 novel 组的留出场景。

泄漏候选在划分前处理。若封存集与训练来源发现确证近重复，优先从训练候选移除关联组并重建尚未使用的清单，保留记录；若已经训练或开封，则报告污染并重新立协议，不抹掉历史结果。

### 9.5 角色访问矩阵

| 入口 | 可读角色 | 允许读取 GT |
|---|---|---|
| `audit_data` 开发标签 QA | R/S/Q Train/Dev 的抽样 | 是 |
| `train` | 当前数据协议的 Train；按规则周期评估 Dev | 是 |
| `oracle_select` | S_select | 是，特权诊断 |
| `oracle_confirm` | 锁定配置的 S_confirm | 是，仅登记的一次确认 |
| `predict` | RGB-only 清单，可包括封存集 RGB | 否 |
| `evaluate` | 已有预测所对应的封存标签清单 | 是，预测封存后 |

测试集路径/字节 hash/身份去重属于数据治理，不能等同模型调参。但实现 `audit_data` 时仍默认只画 Train/Dev 标签；Test 不输出供选模型使用的深度分布、失败样例或误差榜单。正式 Test 质量规则只能由已冻结处理规则执行，不能逐图根据模型表现更改。

输出到`storage.manifests/<protocol_version>/*.jsonl`；以下CLI用项目内`artifacts/manifests`作路径示例，服务器可映射到所配置的外部盘。对每份清单计算 SHA-256；形成 `split_audit.json`，内容至少包括：每域角色图数/组数、全部组交集、重复对、CAD 重叠、未知身份比例、排除表、划分配置、代码版本、处理版本。只存在清单文件但没有通过审计，不能称划分已完成。

## 10. 具体 Python 文件职责与函数接口

以下均为**待实现代码设计**，模块可独立开发，适配器不能夹带训练逻辑：

| 文件 | 职责与主要接口 |
|---|---|
| `src/transdepth/data/schema.py` | `SampleRecord`、`Sample`、`RGBSample`、`GeometryRecord`、`PatchTargets`；字段类型与角色枚举 |
| `src/transdepth/data/paths.py` | `resolve_asset(root_key, relative_path, paths) -> Path`；路径解析与根目录边界检查 |
| `src/transdepth/data/manifest.py` | `read_manifest(path, allowed_roles) -> list[SampleRecord]`；JSON schema、唯一 ID、hash、角色校验 |
| `src/transdepth/data/adapters/base.py` | 统一 `scan(root, release) -> Iterable[SampleRecord]` 和 `load_raw(record)` 合同 |
| `src/transdepth/data/adapters/rftrans62.py` | 按 ID 字典配对，排除 CG 重渲染；PNG/EXR/recorder 候选布局 |
| `src/transdepth/data/adapters/cleargrasp_syn.py` | 原始合成配对、rectified深度与语义mask，禁止法线图作为RGB |
| `src/transdepth/data/adapters/cleargrasp_real.py` | 真实数据配对opaque GT与transparent RGB，核对采集与相机身份 |
| `src/transdepth/data/depth_io.py` | `read_rgb_u8`、`read_png_preserve_depth`、`read_exr_channel`，返回源格式元数据 |
| `src/transdepth/data/depth.py` | `decode_depth(raw, spec, camera) -> (depth_m, valid_depth)`；单位、ray→z、invalid 分开 |
| `src/transdepth/data/masks.py` | `decode_semantics(raw, rule) -> labels_u8`；色表、ignore、未知统计 |
| `src/transdepth/data/geometry.py` | `letterbox_bundle(...)`；尺寸、插值、K、A、content_mask、反变换记录 |
| `src/transdepth/data/patches.py` | `build_patch_targets(...) -> PatchTargets`；M2/M3 纯度与 IQR |
| `src/transdepth/data/grouping.py` | `build_leakage_groups(records, edges)`；证据边与连通组 |
| `src/transdepth/data/split.py` | `assign_roles(records, split_spec)`；组级确定性划分与权责校验 |
| `src/transdepth/data/audit.py` | 文件/像素 hash、近重复候选、CAD 交集、配对/编码/patch 报告 |
| `src/transdepth/data/cache.py` | 按 hash+处理版本建立/失效派生缓存，原子写入 |
| `src/transdepth/data/dataset.py` | `TrainingDataset`；读取允许角色的监督样本 |
| `src/transdepth/data/rgb_only.py` | 独立`RgbOnlyDataset`；只读RGB清单，与06一致 |
| `src/transdepth/data/transforms.py` | 确定性预处理组合和独立随机流增强，不在dataset内隐藏几何操作 |
| `src/transdepth/data/sampling.py` | 05定义的每update全局16样本表，精确实现来源配比 |
| `src/transdepth/data/collate.py` | 堆叠固定 384×512 张量，metadata 保留 list，拒绝混角色 |
| `src/transdepth/cli/inspect_data.py` | 参数校验→扫描→文件元数据清单，尚无训练角色 |
| `src/transdepth/cli/build_manifest.py` | 文件盘点→显式 adapter 配对→候选 `SampleRecord`，不得自行决定 split |
| `src/transdepth/cli/split_data.py` | 加载身份边/别名表→划分→交集审计→保存候选清单 |
| `src/transdepth/cli/prepare_data.py` | 经过审计的来源规格→统一处理/可选缓存 |
| `src/transdepth/cli/audit_data.py` | 受角色约束的叠图、数值验证和门槛报告 |

### 10.1 核心类型

```python
# 待实现的接口草案，字段可放入 dataclass；numpy 是 CPU 数据层。
@dataclass
class Sample:
    rgb: Tensor                 # float32 [3,384,512]，已按 DINO 规范归一化
    depth_m: Tensor             # float32 [1,384,512]
    mask: Tensor                # uint8 [1,384,512]，0/1/255；即文中 labels
    valid_depth: Tensor         # bool [1,384,512]，全部可信深度
    mask_known: Tensor          # bool [1,384,512]，可信类别且内容内
    content_mask: Tensor        # bool [1,384,512]
    patch_targets: PatchTargets # 在数据层/训练目标层统一生成，禁止输入模型
    metadata: dict              # sample_id/source/role/geometry，无训练隐式条件

@dataclass
class RGBSample:
    rgb: Tensor
    sample_id: str
    geometry: GeometryRecord    # 仅由 RGB 尺寸算出的预处理信息，无 GT/K 输入

def read_exr_channel(path: Path, channel: str) -> tuple[np.ndarray, dict]: ...
def decode_depth(raw: np.ndarray, spec: DepthSpec,
                 camera: CameraSpec | None) -> tuple[np.ndarray, np.ndarray]: ...
def letterbox_bundle(rgb_u8, depth_m, labels, valid_depth,
                     K_raw, target_hw=(384,512)) -> CanonicalBundle: ...
def build_patch_targets(depth_m, labels, valid_depth,
                        content_mask, spec: PatchSpec) -> PatchTargets: ...
```

`TrainBatch`由collate将Sample的张量增加batch维，metadata和PatchTargets按样本保存；固定shape的RGB/depth/mask/valid才直接stack。类型在data/schema.py定义，contracts.py统一导出公共类型。训练 `collate` 可把 `mask_known` 推导为 mask!=255，但所有调用点采用同一函数，避免 teacher 与 loss 的有效域漂移。本文解析器内部将三态数组称作 `labels`；对外 `Sample` 统一字段名为 `mask`，类型与语义不变。每 batch 检查 shape、source、role；数据层不把大字符串 metadata 复制到 GPU。

### 10.2 一次训练样本读取的精确流程

```python
# 待实现伪代码
record = manifest[index]
require_role(record.assigned_role, allowed_train_roles)
require_passed_release_and_sample_audit(record)
raw = adapter.load_raw(record)
rgb = read_as_rgb_u8(raw.rgb)
depth_m, valid_depth = decode_depth(raw.depth, source_spec.depth, raw.camera)
labels = decode_semantics(raw.mask, source_spec.mask)
assert_aligned_coordinates(rgb, depth_m, labels, record)
bundle = letterbox_bundle(rgb, depth_m, labels, valid_depth, raw.camera.K)
bundle = apply_paired_augmentation(bundle, exposure_seed)  # 小样本拟合时 identity
patch_targets = build_patch_targets(bundle.depth_m, bundle.labels,
                                    bundle.valid_depth, bundle.content_mask, patch_spec)
return build_training_sample(bundle, patch_targets, record)
```

来源采样表由训练 sampler 负责：每个全局完整更新 R/S/Q=7/7/2，而不是每个 GPU 都凑 7/7/2。数据集返回稳定 `sample_id`，训练记录 `(update, rank, accumulation_slot, sample_id, augmentation_seed)`，支持曝光审计和 H1/H2 复用相同输入。

## 11. 未来执行命令、产物与验收

### 11.1 项目 CLI（全部待实现）

下列命令是未来接口合同，本次不要将它们理解为已经存在或已经运行成功：

```bash
# 待实现：只盘点指定来源，生成候选配对。
python -m transdepth.cli.inspect_data \
  --paths configs/paths.local.yaml \
  --sources rftrans_62cad cleargrasp_synthetic cleargrasp_real \
  --output artifacts/data-audit/inventory_v1

# 待实现：用adapter配对键构造候选JSONL，缺失/冲突写独立报告。
python -m transdepth.cli.build_manifest \
  --inventory artifacts/data-audit/inventory_v1/inventory_files.jsonl \
  --paths configs/paths.local.yaml \
  --output artifacts/data-audit/inventory_v1/inventory_samples.jsonl

# 待实现：按组候选划分，未知身份会在报告中显式失败/降级。
python -m transdepth.cli.split_data \
  --inventory artifacts/data-audit/inventory_v1/inventory_samples.jsonl \
  --config configs/data/splits_v1.yaml \
  --output artifacts/manifests/FAR_R62_CGSyn_CGReal_train_v1

# 待实现：解析规则与几何规格必须已填写证据，禁止自动猜单位。
python -m transdepth.cli.prepare_data \
  --paths configs/paths.local.yaml \
  --manifest artifacts/manifests/FAR_R62_CGSyn_CGReal_train_v1/train.jsonl \
  --config configs/data/preprocessing_v1.yaml \
  --output artifacts/data-audit/prepare_v1

# 待实现：只抽 Train/Dev，最少每来源 30 组；不足则全量并报实际组数。
python -m transdepth.cli.audit_data \
  --paths configs/paths.local.yaml \
  --manifest-dir artifacts/manifests/FAR_R62_CGSyn_CGReal_train_v1 \
  --roles R_train R_dev S_train S_dev Q_train Q_dev \
  --groups-per-source 30 \
  --config configs/data/preprocessing_v1.yaml \
  --output artifacts/data-audit/acceptance_v1
```

每个 CLI 支持 `--help`、明确 exit code、配置快照、日志和 `--dry-run`。失败时不产生 `PASSED` 标记；输出先写临时文件，再原子更名，避免中断后半份 JSONL 被下游当完整清单。

### 11.2 最小 QA 图与数值表

每个来源选最少 30 个**组**，组不足则全量并声明不足；QA 抽样固定种子，覆盖相机、近远距离、透明面积大小、细柄/杯口/孔洞、遮挡、贴边与 padding。真实标签图仅来自 Train/Dev。

每组输出以下多面板图：

1. transparent RGB 原图；
2. 原始数值深度的固定米制色条可视化；
3. mask 色表/三态透明叠图；
4. valid_depth、unknown、padding 各自颜色；
5. 384×512 处理结果与内容边界；
6. T/B/U patch 网格；
7. 有同帧 EXR 时的 PNG-EXR 差图；
8. 真实 Train/Dev 可加 opaque RGB 与轮廓误差图。

每来源数值表列：样本数、独立组数、原始尺寸分布、位深/EXR 通道类型、GT 1%/50%/99% 分位数、零/NaN/Inf/饱和值比例、透明与 unknown 面积、padding 比例、T/B/可靠 key 数、空关系比例、深度有效但未监督比例。不要将三域混成单一均值掩盖某来源错误。

### 11.3 必须通过的有意义检查

| 检查 | 最小构造/真实证据 | 通过标准 | 失败处理 |
|---|---|---|---|
| 文件配对 | 故意缺不同编号但保持数量相同 | 明确发现缺失，不能错位配对 | 修 adapter/隔离文件 |
| PNG 位深 | 0、1、32768、65535 的 16 位样例 | 解码不经 uint8；常数按冻结 spec | 修读库/编码规则 |
| EXR 通道 | R/G/B 不同、HALF/FLOAT 两类小样例 | 按名字选通道，返回 float32 且记录源精度 | 修 EXR 后端 |
| ray/z | 已知 K 与平面几何 | 只对 ray 数据转换；光轴输入保持值 | 修坐标与幂等标记 |
| RGB-GT 对齐 | 非对称图形和真实叠图 | 不上下颠倒/镜像/半图错配 | 修方向/配准 |
| letterbox/K | 不同长宽比＋3D 投影点 | 空间变换与 K 投影一致，padding 全无效 | 修中心约定/舍入 |
| mask | 红、绿、黑、混合边缘、孔洞 | 红不透明、绿透明、未知 ignore、洞保留 | 修色表，禁止填洞 |
| patch | 230/231 个有效像素；纯度边界；空域；高 IQR；仅一个 padding 像素 | 与公式一致，空集合不 NaN、不变背景，partial padding 整块 U | 修分母/quantile/内容门槛 |
| group split | 同场景跨相机/原图缩图/透明喷涂对 | 所有绑定分量只有一个角色 | 重建分组与划分 |
| 角色隔离 | 把 Q_test 塞进 train manifest | train 构造立即拒绝 | 修角色校验 |
| RGB-only | 标签根目录不可读的 RgbOnlyDataset | RGB 张量与允许标签时一致，预测入口不碰 GT | 分离 Dataset/缓存 |

这些是数据正确性测试，不能用来声称方法有效。真实首表面、颜色语义、场景身份仍需要实际材料，单元测试不能制造证据。

### 11.4 D0–D5 的最终放行条件

- [ ] 三来源都能唯一配对，实际样本数、组数与异常记录可追溯。
- [ ] R 的 62CAD 与 RFTrans-CG 重渲染已区分，CG 原始来源没有混淆。
- [ ] 本地 release 的深度编码、单位、坐标、首表面与 invalid 规则各有证据。
- [ ] CG synthetic 采用发布 rectified；Q 采用配对 opaque GT。
- [ ] mask 调色板、unknown、孔洞和真实替换误差已检查。
- [ ] 384×512 所有模态同步，padding 不监督，K 变换验证通过。
- [ ] T/B/U 与公式一致，教师覆盖率分 R/S/Q 输出。
- [ ] Train/Dev/Select/Confirm/Test 按组划分，所有已知泄漏边不跨角色。
- [ ] novel 的已见/未见/未知 CAD 状态已明确，不能靠原标签命名推断。
- [ ] 清单、来源规格、预处理版本和 QA 报告的 SHA-256 已冻结。
- [ ] 推理只读 RGB 清单，GT 访问已与评估拆开。

任一项缺实证，标记为 `pending` 或 `failed`，不能用另一个来源通过来代替。可以先做未受影响来源的解析开发；混合训练 M1/M2 验收仍未完成。

## 12. 当前仍待现场确认的信息

1. 用户实际三来源存储路径、是否完整、是否已有二次预处理。
2. RFTrans 本地包与源码版本是否对应、同帧理想 PNG/EXR 是否齐全、far-plane/无效值语义。
3. 可用的 RFTrans recorder、ClearGrasp JSON、CAD 对照、真实跨相机/近邻采集分组证据。
4. ClearGrasp 完整包中 known/novel/validation 的实际层级和原图/resize 变体；样例树不能代替整包验收。
5. 三来源真实可用的独立组数、真实 Train/Dev/Test 分配后的图数、联合 CAD 重叠程度。
6. 实际相机内参和畸变/配准元数据、opaque 替代物误差分布。
7. 实际 EXR 文件通道及 HALF/FLOAT 编码、服务器读取后端是否可用。

这些信息用于将本章设计实例化。文档中没有把它们伪装为已完成检查；用户给出路径后，第一步应运行盘点与编码/身份验收，再接 H＋DPT 的小样本拟合。
