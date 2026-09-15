# 06｜验证、纯 RGB 推理、模型导出与验收

版本：2026-09-15。状态：**待实现技术规格；不代表已部署、已推理或已取得测试结果。**

依据：修订计划 M13/M14、§4–§7；理论 PDF §6.4、§13.4–§13.7。当前目标是有限真实监督下的 RGB-only 留出泛化，最终标签以新划分的真实 Test 为准。本文所有 Python 文件、配置和命令是后续实现接口。

## 1. 验证与推理分成三条路径

| 路径 | 可以读取什么 | 不能做什么 | 输出 |
|---|---|---|---|
| 学生 Dev 验证 | predictor 只读 RGB；evaluator 读指定 Dev 标签 | 不读取 Test；不按 GT 修预测尺度 | 按域逐图/组指标，选择 checkpoint |
| Oracle Select/Confirm | 明确标记的特权 runner 读本阶段 RGB/GT/mask | 不输出部署模型；不复用测试标签诊断 | 固定 T0 的配对干预收益 |
| 独立 Test | predictor 只读 RGB 与导出权重；预测落盘后 evaluator 读 Test GT/mask | 不重选架构/位置/lambda/配比；不做测试尺度对齐 | 冻结预测、封存终评表 |

工程上用不同入口、不同 dataset 类型和不同进程，避免一个 `mode="test"` 分支里仍携带整个训练 batch。

```text
Raw transparent RGB
        │
        ▼
RgbOnlyDataset ──► preprocess_rgb ──► RGB predictor ──► immutable depth predictions
                                                              │
                                                              ▼
Sealed labels ─────────────────────────────────────────► offline evaluator
                                                              │
                                                              ▼
                                      per-image → per-group → domain summary
```

## 2. 待实现文件和接口职责

| 文件 | 职责 |
|---|---|
| `src/transdepth/data/rgb_only.py` | 只解析 RGB manifest，不支持 depth/mask 字段 |
| `src/transdepth/inference/predictor.py` | 纯 RGB 前向、batch 推理、输出米制深度 |
| `src/transdepth/inference/serialization.py` | 预测及 metadata 原子写入、完整性核验 |
| `src/transdepth/inference/merge.py` | 复制权重、合并 Q/K、生成部署 manifest |
| `src/transdepth/evaluation/regions.py` | GT 有效域及透明/背景/边界定义 |
| `src/transdepth/evaluation/metrics.py` | 逐图 MAE/RMSE/AbsRel/delta |
| `src/transdepth/evaluation/aggregate.py` | 图→组→域宏平均和补充像素池化 |
| `src/transdepth/evaluation/bootstrap.py` | 同组配对差值、bootstrap、种子结果表 |
| `src/transdepth/evaluation/runner.py` | 预测完整后读取指定标签及评价 |
| `src/transdepth/evaluation/locks.py` | Test 模型/协议锁与角色权限检查 |
| `src/transdepth/cli/predict.py`、`evaluate.py`、`export.py` | YAML 参数、输入输出检查与主流程 |
| `tests/integration/test_rgb_only_isolation.py` | 标签物理不可读情况下的推理验收 |
| `tests/integration/test_merge_equivalence.py` | 合并前后同 RGB 等价性 |
| `tests/unit/test_evaluation_aggregation.py` | 指标、空区域、组权重与非有限预测策略 |

建议类型边界：

```python
class RgbItem:
    sample_id: str
    rgb_path: str
    rgb_sha256: str

class PredictorInput:
    rgb: Tensor                   # float32 [B,3,384,512]，官方锁定归一化

class Prediction:
    depth_m: Tensor               # float32 [B,1,384,512]，米制正深度
    seg_logits: Tensor | None     # 模型篇公共结构；部署可不返回分割输出
    trace: AttentionTrace | None  # 部署为None，不保留训练关系诊断

class PredictionRecord:
    sample_id: str
    prediction_path: str
    rgb_sha256: str
    transform_meta: dict          # 来自RGB尺寸，不来自GT
    model_export_sha256: str
    status: str
```

`PredictorInput` 不能含 GT depth、GT mask、opaque RGB、相机内参、来源 one-hot、CAD/对象身份或带 GT 派生信息的特征。sample_id 仅在模型外用于结果关联，不喂入网络。原始文件名不能作为模型输入编码。

## 3. RGB 预处理、画布与落盘格式

### 3.1 RGB-only 预处理

模型统一输入 384×512。输入读取为 RGB 而不是 OpenCV 默认 BGR；明确颜色通道、数值范围、官方归一化及插值。完整参数由 DINOv3 配置篇锁定，不在 predictor 中再抄一套不同常量。

保持长宽比缩放再 padding。`TransformMeta` 至少包含：

```text
original_hw, canvas_hw, resized_hw
scale_h, scale_w, pad_top, pad_bottom, pad_left, pad_right
rgb_interpolation, pixel_coordinate_convention
preprocessing_version, normalization_id
```

因缩放到整数像素后的 `scale_h`、`scale_w` 可能有微小不同，应记录实际比例，而不是事后假定同一浮点 scale。padding 位置仅由 RGB 尺寸决定；不能用透明 mask 裁对象或用 GT valid 区域选 crop。

### 3.2 主评估坐标

主表在 384×512 模型画布的有效内容坐标评价：使用 RGB 生成的 resize/pad 元数据，GT depth、mask、valid 同步最近邻变换；去掉 padding 后比较。不先将预测放大回原图再混进主表。

用户查看时可额外输出原始 RGB 尺寸的深度图：先去 padding，再按已锁定插值还原到原尺寸。必须另标 `original_resolution_view`，明确用于查看或另行定义的指标，不覆盖主评估用画布结果。

若重现官方历史 144×256 评价，建立独立 `metrics_protocol_id`、相同方法全集的独立结果表；不能把不同分辨率/区域/新 known 子集下的数值直接排列为同协议排名。

### 3.3 预测主文件

建议最小主格式为 NumPy `.npy`：单图 `[384,512]`、float32、单位米、无 pickle。它保留浮点数并避免未经验证的 uint16 截断/单位假设。可附透明度、彩色深度可视化 PNG，但彩图不是评价输入。

每图原子写入：先临时文件→flush/关闭→rename→计算 SHA256。文件名采用安全 sample_id 或其稳定 hash；不含能修改目录的路径片段。完成后才向 prediction manifest 追加成功记录。

```json
{
  "schema_version": "prediction_v1",
  "sample_id": "opaque_safe_id",
  "depth_path": "depth/opaque_safe_id.npy",
  "depth_sha256": "filled_after_write",
  "dtype": "float32",
  "unit": "m",
  "coordinate": "camera_z_first_surface",
  "canvas_hw": [384, 512],
  "rgb_sha256": "filled_from_rgb_manifest",
  "export_hash": "filled_from_export_manifest",
  "transform_meta": {"...": "from_rgb_only"},
  "status": "ok"
}
```

此 JSON 是字段规格，省略号不应进入正式输出。主预测 manifest 只保存关联 ID 和 RGB 来源，不复制标签路径。

完整性检查必须核对目标清单所有 sample_id **恰好一次**、文件 hash/shape/dtype、是否缺图/重复、是否有限且正、模型/预处理一致。在 `predictions.complete.json` 中记录列表 hash 和失败项；尚未完成不能启动正常评价。

## 4. GT 有效域与评价区域

### 4.1 validity 只由标签和固定协议决定

设：

- `V_content`：非 padding 内容像素，由 RGB 的变换元数据决定。
- `V_depth_gt`：发布约定、编码、米制转换、首命中含义确认后的合法 GT；有限、正、在预先锁定范围内，排除发布者无效值/far-plane 饱和等。
- `V_mask_known`：可信透明或背景标签；ignore 不做自动背景。

则 `V_eval = V_content & V_depth_gt & V_mask_known`，透明 `Omega_T=V_eval & (M=1)`，背景 `Omega_B=V_eval & (M=0)`。深度合法但 mask unknown 的像素另报覆盖，可另列全深度辅助指标，不能混入透明/背景主表。

不允许 `V_eval &= isfinite(prediction)`、按预测范围删除像素、按预测 mask 限制透明区。否则差预测会自动从评估中消失。

真实 GT 使用对应 opaque-depth，真实 transparent-depth 是原始传感器观察，不替代目标前表面。RFTrans 与 ClearGrasp 的发布无效值和转换均复用数据 QA 的已锁规则，evaluator 不猜单位。

### 4.2 边界带的具体定义

源计划起点为透明边界内外各 3 像素，最小实现建议明确采用**8 邻域、Chebyshev 半径 3、7×7 方形结构元素**，在缩放后画布坐标计算。这是将“3 像素边界”转为可复现实作的工程约定，需在 Dev 前锁定。

可实现为：

```text
known = V_content & V_mask_known
near_T = binary_dilation(M == transparent, square_radius=3)
near_B = binary_dilation(M == background,  square_radius=3)
safe_known = binary_erosion(known, square_radius=3, outside_value=False)
edge_band = near_T & near_B & safe_known
Omega_edge = edge_band & V_depth_gt
```

`near_T & near_B` 选到同时接近两类的一圈内外带。`safe_known` 避免未知标签/padding 的边缘被误识别成透明物体边界；深度无效区域最终再裁除。纯背景图不产生虚构边界。边界只作评价子区域，透明/背景指标仍包含各自合法边界像素，三个指标不是互斥分区。

若选择 Euclidean 半径或其他官方边界实现，另设协议版本并对所有方法一致使用。禁止在看 Test 上边界误差后改变膨胀半径。

### 4.3 空区域

某图 `Omega_T` 为空，透明指标写 `null` 并记录 `empty_region`，不写 0；该图仍可有背景指标。某组没有任何有效透明图，组透明指标同样 `null`；其组 ID 和缺失原因仍保留。

有效图/组集合由 GT 定义，各方法共享同一集合。缺预测、NaN 或负预测不能借“空区域”剔除。

## 5. 指标公式与三层聚合

### 5.1 每图每区域先算完整指标

令区域内 n 个合法 GT 像素的残差 `r=prediction-GT`：

\[
\operatorname{MAE}_I=\frac1n\sum|r|,\qquad
\operatorname{RMSE}_I=\sqrt{\frac1n\sum r^2},\qquad
\operatorname{AbsRel}_I=\frac1n\sum\frac{|r|}{Z^*}.
\]

\[
\delta_{t,I}=\frac1n\sum\mathbf1\!\left[
\max(\widehat Z/Z^*,Z^*/\widehat Z)<t\right],\quad
t\in\{1.05,1.10,1.25\}.
\]

MAE/RMSE 主展示为 mm：米制结果乘 1000；AbsRel 无单位；delta 同时在机器结果中保留 `[0,1]`，展示 `%` 时乘 100。表头直接写 `delta_1.05`、`delta_1.10`、`delta_1.25`，不能写未定义的 delta1/2/3 后套用 1.25 的平方/立方。

建议离线指标用 float64 累加，以降低大区域求和误差。阈值为严格 `<`，测试必须覆盖恰等于阈值的情况。

### 5.2 主结果：图→泄漏组→域

对指标 m、区域 r，在独立组 g 中取有合法区域的图像均值，再对合法组等权：

\[
m_g=\frac1{|I_g^r|}\sum_{I\in I_g^r}m_I,\quad
m_{domain}=\frac1{|G_{eff}^r|}\sum_{g\in G_{eff}^r}m_g.
\]

组来自 `leakage_group_id` 的最高可确认采集单元，不用图像文件名前缀随意造组。组身份无法恢复时必须显式把结果称“图像留出”并报告聚合条件，不能写独立场景泛化。

RMSE 也先每图开方、再组均值、再域均值。因此主表字段建议命名为 `group_macro_of_image_rmse_mm`。两个等尺寸图的 MSE 若为 0 和 4，逐图 RMSE 均值为 1，像素池化 RMSE 为根号 2；不能在汇总时混用。

### 5.3 补充像素池化

可以补 `pixel_pooled_mae_mm`、`pixel_pooled_rmse_mm`，但明确其不同权重：分别累计 `sum_abs/n`、`sqrt(sum_sq/n)`。不要对已开方的逐图 RMSE 再平方重建主表，也不要拿各 rank 的 pooled RMSE 平均。

同时报告每域有效图数、独立组数、透明/背景/边界有效像素数、空区域图/组数、缺失预测数、失败图率，便于读者理解指标覆盖。

### 5.4 分布式评价不重复、不丢尾部

标准训练 DistributedSampler 可能为整除而填充重复索引；评价不能直接沿用。建议 `sample_indices = all_indices[rank::world_size]`，允许每 rank 图数不同，推理阶段不使用需要等步同步的 DDP forward；各 rank 使用独立 eval 模型或已分发权重的普通模型。

每 rank 写 `per_image.rank<n>.csv/jsonl`。rank 0 在所有分片完成后合并，按 sample_id 验证唯一与完整，再按 `leakage_group_id` 聚合；同一组可能横跨 rank，必须合并后再平均。不要对 8 个 rank 的均值再平均。

样本较少时汇总逐图记录最清楚；规模很大时可按 `(group_id, region, metric)` 汇总图指标和图数，再 merge 同组。像素池化另用 float64 和整型统计量求和，不能替代组宏平均。

## 6. 非有限预测、越界与失败规则

原计划要求 NaN/Inf 报失败并按预定规则处理，未给唯一数值罚分。最小主实现建议采用**严格失败策略**，开 Test 前锁定：

1. 若任何目标图在 `V_eval` 上缺失预测，或存在 NaN/Inf/非正预测，该图标 `prediction_failed`；GT 有效集合保持不变。
2. 对应主区域/域结果标 `failed`，不输出一个仅由成功像素算出的正常主表数值。可保存成功部分诊断值，但字段名必须含 `diagnostic_success_subset`，不能用于优胜声明。
3. 报失败图/组率、非有限/非正像素数和 sample_id；对 delta，失败像素如补充罚分统计则明确按 0 命中计。
4. 任一模型有失败图时，主 H1/H2 对比状态为失败或证据不足，不能删除该图后继续宣布通过。
5. 有限但很大的预测保留原数值参与误差；不按 GT 合法范围裁掉预测，以免改善结果。

若未来确需完整有限罚分表，可在 **Dev 阶段**预注册固定深度罚值/失败图罚分并对全部模型一致执行，单列 `penalized` 指标；不能测试后为失败模型临时设计有利规则。

部署可以返回错误状态并让调用端处理；这与论文评价保留失败样本是两件事。不要用 nearest fill、GT mask 或逐图尺度恢复让坏输出悄悄变好。

## 7. FAR 权重导出与合并等价性

### 7.1 导出前的输入清单

导出程序读取：锁定 checkpoint、冻结 DINOv3 本地权重引用与 hash、模型结构配置、LoRA 位置/r/eta、预处理版本、环境版本。拒绝使用未经清单核验的“同名权重”。训练 optimizer 和数据标签不属于部署依赖。

最小部署形式为 Python/PyTorch `state_dict` 包，加严格的导出配置和本地骨干引用。ONNX、TensorRT 或 `torch.export` 不是当前最小实现必须项；若后续需要应另验 RoPE、动态尺寸、算子支持和数值误差，不能在本轮承诺通用导出兼容性。

### 7.2 合并到独立副本

数学使用 `[in,out]` 权重，PyTorch `Linear.weight` 为 `[out,in]`。对单头 h、C=1280、d=64，packed QKV 输出行：

```text
Q rows: [h*d : (h+1)*d]
K rows: [C+h*d : C+(h+1)*d]
V rows: [2*C : 3*C]，全部不改
```

在独立 FP32 权重副本上做：

\[
W_Q^{deploy}[rows]+= (\eta U_QB_Q)^\top,\quad
W_K^{deploy}[rows]+= (\eta U_KB_K)^\top.
\]

执行步骤：

1. 校验未合并 checkpoint 的 `merged=false`；构造不共享原 Tensor storage 的骨干副本。
2. 按官方 QKV shape 和已锁 head 位置确认切片；校验 delta shape。
3. 在 FP32 中形成 delta 并写入对应 Q/K 行；V、bias mask、其余 blocks/heads 均不修改。
4. 移除 LoRA 计算支路及训练 hook，导出 manifest 写 `merged=true`、原骨干 hash、adapter hash、merge 实现版本。
5. 原训练实例及其原 Q/K 保持不变；它们仍是教师的来源。禁止直接覆盖训练 checkpoint 后再恢复教师训练。
6. 再次对已合并包调用 merge 时明确报错，防止重复加 delta。

C 的语义特征支参与深度融合，必须保留；仅在确认不影响深度后省略不需要输出的 SegTail/分割图输出。不能把整个语义分支称为“训练时辅助、推理时免费删除”。

### 7.3 合并验收分三层

同一固定 Dev RGB、同一预处理、eval 模式，比较未合并学生与合并副本：

| 层级 | 检查内容 | 输出 |
|---|---|---|
| 参数／投影 | 选中 Q/K、未选 Q/K/V；FP32 小张量 | max abs、relative error、未选部分 checksum |
| Attention | post-RoPE Q/K 与选中 head 输出 | 逐层误差、同后端信息 |
| 全模型 | 最终米制深度，A/B/C 分别验收 | max/mean abs error_m，Dev 区域指标差 |

先建立 FP32 数值容差，再为已验收的 BF16/FP16 部署精度单独定容差；误差口径、阈值和输入集在测试前锁定。可用 `abs_diff <= atol + rtol * abs(reference)` 判定。`atol/rtol` 不能临时放宽到错误通过，且相对误差须处理接近零的参考值。

源计划没有给适用于所有融合后端的唯一容差，本设计不填虚构“实测误差”。门禁要求 `merge_tolerance_policy.json` 已由 FP32 和部署精度 Dev 身份测试形成，实际阈值非 null；否则导出状态只能是未验收。

### 7.4 部署包两种交付形态

1. **服务器本地包**：导出的可训练模块／合并权重 + 冻结骨干 hash 引用 + 本地路径配置，适合本项目服务器；加载失败时不能自动联网换另一版。
2. **自包含包**：按权重许可允许范围，将完整已合并 state_dict 放入导出目录并记录来源及许可。Git 仓库通常只放代码/配置/hash，数 GB 权重另存受控模型目录；不把访问 token 写进包。

选用哪一种由部署篇确定；两种都需要运行时本地文件完整性检查。训练 checkpoint 可含完整恢复状态，部署包不带 optimizer/RNG/训练标签。后续公开发布权重之前按实际官方许可办理，不在文档生成阶段默认为已获公开再分发授权。

首个服务器版本采用第1种，具体包合同为：原始DINO权重引用/hash＋完整reassembly/decoder/tail状态＋已经合并后的选中block完整`attn.qkv.weight`替换张量。加载时先严格读原始骨干，再按明确白名单替换这一张量并严格加载读出；检查其未选Q/K行及全部V行仍与原权重一致，不安装LoRA支路。它不是“只存decoder却忘了LoRA”的包，也不在每次forward再做低秩计算。若存完整自包含state_dict，则不再叠加该替换包。两种形式都记录唯一`export_manifest`和全部组成文件hash。

## 8. 纯 RGB 物理隔离验收

只看函数签名不足以证明没有隐藏标签访问。需要以下三个独立证据：

### 8.1 结构证据

- `RgbOnlyDataset` 只接受 `sample_id,rgb_path,rgb_sha256`，遇到标签字段严格拒绝。
- predictor 不 import 训练 dataset、teacher、loss、oracle；model forward 只需要 RGB。
- runtime 不接收 GT mask、相机内参、domain ID 或 opaque RGB；模型导出 manifest 不含标签路径。
- 对现有代码做静态检查和调用图审查，但不把 grep 没找到关键词当完整证据。

predict 使用独立 `configs/inference.paths.local.yaml`，只允许已登记的骨干源码/权重和输出路径字段；拒绝 `roots` 下的 RFTrans/ClearGrasp 等数据根、任何 GT/mask/opaque 路径、训练/标签 manifest。不要把全量 `configs/paths.local.yaml` 交给 predictor 后仅承诺“不会读”。自包含导出包可完全不需要 paths 参数；export/evaluate 仍可使用完整路径配置，各自权限分开。

### 8.2 运行时物理不可读

在远端 Linux 建立一个只包含已锁代码/环境、导出模型与少量 **Dev** RGB 的临时隔离运行目录或容器，标签目录不挂载、不设置数据根路径。Dev 样本必须包含不同来源、长宽比及 padding 情形。

通过容器 mount 白名单、独立账号文件权限或等价隔离确保该进程打不开训练/Dev/Test GT 与 mask。使用副本完成测试，不移动、删除或修改原始标签文件。另记录可访问路径清单、启动命令、导出 hash 和读取日志（可用系统文件访问审计工具）。

先用同一导出包在普通 Dev RGB 路径生成预测，再在标签不可读环境运行同一批 RGB；比较输出和数值容差。通过条件是全部成功且预测等价，文件访问只涉及代码/依赖、模型、RGB、输出目录。

### 8.3 隐藏依赖扰动

在隔离副本中随机化 sample_id、重新命名 RGB 文件，保持 RGB bytes 与预处理相同；输出深度应不变。另向窄推理配置故意加入标签字段，预期启动即拒绝，不能读取后继续运行。标签由 evaluator 后读，预测器没有可用接口读取它。

所有隔离验收用 Dev 完成；Test 运行沿用相同镜像/环境/导出包和权限结构，不能借隔离测试提前查看 Test GT。

## 9. 锁定后终评流程

### 9.1 最终测试域与表述

| 域 | 角色 | 应如何命名 |
|---|---|---|
| 新 Q-test-known | 原 real-known 按组留出的测试子集 | 新 known 留出，不称原完整 real-known benchmark |
| 原 real-novel | 主终评域，整组保留 | 身份审计通过才称相对联合训练未见形状；否则称原 novel 组留出场景 |
| 原 Syn-novel | 合成辅助终评域 | 单独列域，不与真实域混合 |
| RFTrans 官方留出 | 本轮不参与选择，后续审计 | 不能代替主真实终评或提前调参 |

训练已经访问 Q-train，所以不能把结果写为零样本真实迁移。跨 R/S/Q 的 CAD 身份表决定未知形状结论；原数据集的 novel 命名本身不构成联合未见证据。

### 9.2 `test_protocol.lock.json` 的必填字段

```text
protocol_id, schema_version, created_at
git_commit, environment_lock_hash, implementation_check_hash
data_split_lock_hash, test_manifest_hashes, group_identity_audit_hash
selected_runs: [run_id, seed, checkpoint_kind, checkpoint_sha256, export_sha256]
primary_comparison: Mix-H2_vs_Mix-H1
primary_domain: cleargrasp_real_novel_heldout
primary_metric: transparent_group_macro_image_mae_mm
background_edge_tolerances_by_domain
practical_relative_gain_threshold: 0.02
checkpoint_selection_rule, last_update_budget
preprocessing_version, metrics_protocol_id, region_rule_version
gt_depth_validity_version, invalid_prediction_policy
bootstrap_unit, bootstrap_replicates, bootstrap_seed, interval_type
report_domains, report_metrics, report_seeds, qualitative_sample_selection_rule
test_scale_alignment: none
oracle_confirm_family_lock_hash, rgb_only_export_check_hash
```

JSON 的实际值只能来自已完成的前置验收，不能用 `TBD` 开 Test。主相对门槛 0.02 是计划预定起点；若按 Dev 决定改动，应在开封前一次锁定，保留修订记录。

Test CLI 在加载任何标签前校验锁文件与所有 hashes。模型变化、数据重新分组、区域规则变化或环境关键版本变化都要失效原锁；修复是否需要新留出应依据是否看过 Test 结果及是否改变研究选择来记录。

### 9.3 实际执行顺序

1. 对所有预注册模型/种子完成 Dev-best/last 选择，并明确主表的 checkpoint 类别。
2. 完成合并、纯 RGB 隔离、单卡推理 profile；导出 hash 冻结。
3. 创建 Test RGB manifest 与单独评价标签清单；predictor 仅能打开前者。
4. 运行全部预注册预测，结果主文件落盘，写文件 hashes；不得先逐图查看 GT 再调整下一张预测。
5. 验证预测完整性，生成 `predictions.complete.json`；失败按已锁策略记录。
6. 独立 evaluator 进程读取冻结预测和封存标签；生成逐图、逐组、逐域结果及配对统计。
7. 一次性生成主表、全部种子、best/last 辅表、失败与最大退化示例；不删除不利域或种子。
8. Test 之后不再据结果更改架构/lambda/配比/训练预算/尺度后宣称同一独立终评；下一轮需要新留出或明确探索性报告。

## 10. 配对统计与研究结论

每种子 s、域 d、有效独立组 g 先计算：

\[
\Delta_{s,d,g}=e_{H1,s,d,g}-e_{H2,s,d,g},\quad
G_{s,d}=\operatorname{mean}_g\Delta_{s,d,g}.
\]

透明正值表示 FAR 有益；背景/边界用 `e_H2 - e_H1` 表示退化并与锁定容忍量比较。相对增益为 `(e_H1-e_H2)/e_H1`，仅基线误差>0 时定义；同时报告绝对 mm 差。

### 10.1 组区间与种子波动分开

- 对每个种子按独立组配对 bootstrap，默认 2,000 次与 oracle 保持易审计一致；输出单侧或双侧区间由 Test lock 明确。建议主学生结果给双侧 95% 区间，确认方向另列单侧下界，不能混表不注明。
- 三种子分别列 H1/H2/G；另报种子均值、标准差、最小/最大增益。三个优化种子不当成数千个新独立场景。
- 若需要跨种子平均方法的组区间，先对固定三个种子的 `Delta[s,g]` 在每个组取均值，再重抽组；说明该区间条件于这三个已运行种子，不覆盖全部优化随机性。
- 平均三个模型的指标不是 ensemble；只有先平均预测再评估才是 ensemble，必须另列额外推理成本及预注册流程。

### 10.2 主表建议字段

```text
protocol_id, domain, backbone, decoder, method_id, seed, checkpoint_kind
successful_t0_updates, successful_t1_updates, train_samples_R/S/Q
transparent_mae_mm, transparent_rmse_mm, transparent_absrel
transparent_delta_1.05, transparent_delta_1.10, transparent_delta_1.25
background_mae_mm, edge_mae_mm
effective_groups_by_region, valid_images_by_region, failed_images
frozen_params, trainable_decoder_params, lora_params
train_gpu_hours, inference_latency_median_ms, inference_latency_p95_ms
inference_peak_allocated_gib, export_sha256, metric_protocol_hash
```

分割 IoU、Teacher KL 可作辅助表，但不能取代深度主表或用于 Test 选赢家。所有训练数据域损失分别列原始／加权值；源域好不能代替目标真实域好。

### 10.3 可以声称与尚不能声称

| 结果 | 合理表述 |
|---|---|
| Oracle 合成确认通过 | 固定读出上的该合成关系干预有益 |
| 学生 KL 降 | 单 RGB 学生能拟合部分关系目标 |
| H2>H1，单 seed Dev | 进入复核的信号 |
| 三种子真实留出一致获益，区间/门槛/退化约束通过 | 该数据和训练协议下 FAR 有增量证据 |
| 只 Mix 比 CG 好 | 固定预算的数据曝光替换有效；方法增量另验 |
| 加真实训练后优于纯合成 | 有限真实监督有效；不是零样本能力 |
| 仅 A 上有收益 | FAR 证据受该读出条件限制 |
| CI 跨零或种子不一致 | 证据不足，保留原始结果与失败分析 |

## 11. 可视化与推理资源验收

### 11.1 定性结果选择

在 Test 前锁定预选规则：固定随机样本、指定属性分层样本，以及结果产生后按算法自动选出的最大收益/最大退化样本。三者明确标识，不能只展示漂亮的杯子。

每组图显示 RGB、GT、H1、H2、误差图、可信透明/边界覆盖；同域统一颜色范围或清楚标出逐图范围。误差图和彩色深度只是展示，不送回模型。标签覆盖率低、配准疑点及未知 CAD 身份要随图说明。

用户使用的单 RGB demo 仅显示 RGB 与预测；需要 GT 的论文 panel 由 evaluator/report 生成，不进入部署 UI。

### 11.2 单卡 3090 profile

源计划要求 batch=1、384×512、一张 3090。模型预热后至少测 200 张图，CUDA 计时前后同步，统计中位/P95、峰值 allocated/reserved、吞吐、失败率；记录是否包括图像读取、decode/resize、CPU→GPU、文件写出。

建议分别给两项：

1. `model_only_latency`：已准备张量的设备前向；用于比较解码器开销。
2. `end_to_end_latency`：磁盘 RGB 到落盘深度；用于用户部署预期。

GPU-hours 用实际卡数与墙钟累计，8 卡独立样本推理属于吞吐扩展，不表示单图时延自动除以 8。C 保留的语义分支算在参数/延迟里。

profile JSON 记录硬件名称、显存、GPU 拓扑、CPU/RAM、存储、driver、torch/CUDA、SDPA backend、dtype、模型/hash、batch/输入、warm-up、样本数及 timing_scope。当前未运行服务器，所有资源结果应为 `pending`，不能将理论权重内存当实测。

## 12. 验收门禁与输出

### 12.1 完整 gate 表

| Gate | 需要证据 | 阻止的后续动作 |
|---|---|---|
| G-data | 三来源单位/坐标/首命中/有效域、配对/泄漏组与划分锁 | 任何正式训练 |
| G-backbone | 官方模型身份、权重 hash、四层 shape、预处理、RoPE/SDPA 身份 | Oracle 与 LoRA |
| G-loss | M8 手算、空区域、单项梯度、16图/DDP归一 | 配对正式训练 |
| G-resource-train | 20 warm-up +100 完整更新实际 profile | 长预算训练 |
| G-T0 | 可用读出、固定父 checkpoint、Dev 质量记录 | Oracle 价值判断 |
| G-oracle | Select lock、一次 Confirm／族控制、组统计 | 已确认 FAR 学生主流程 |
| G-pair | 同 T0/初始化/样本/增强/参数/优化器/预算/选择规则 | H1/H2 因果比较 |
| G-resume | 连续与中断恢复的等价性 | 依赖恢复的正式结果 |
| G-merge | FP32/部署精度参数→attention→深度等价；重复合并保护 | 部署模型验收 |
| G-rgb-only | 标签不可读的隔离运行、文件名扰动等价 | 纯 RGB 部署主张 |
| G-metrics | 逐图/组/域/池化/分布式一致性、失败策略 | 正式报告 |
| G-test-lock | 模型/指标/区域/种子/门槛/数据/哈希全部锁定 | 打开 Test 标签 |
| G-predictions | 预注册清单恰一次、文件/hash/shape完整 | 正常 Test 评价 |
| G-resource-infer | 单卡至少200图延迟/峰值/失败率 | 已验收的资源声明 |

门禁文件有 `pending/pass/fail`、时间、实现 commit、输入 hashes、结果报告路径、失败原因。`pass` 必须由实际证据产生，不因创建了测试脚本或报告模板就自动填入。

### 12.2 必要评价测试

| 测试 | 关键预期 |
|---|---|
| 预测=GT | MAE/RMSE/AbsRel=0，全部 delta=1 |
| 预测=2GT | AbsRel=1，三个 delta=0，米/mm换算一致 |
| 阈值等号 | ratio==1.05 不计入 delta_1.05 |
| 图面积不等 | 主图/组宏均值与像素池化可以不同，字段不能混 |
| 组图数不等 | 两组等权，不给更多视角组更高主表权重 |
| 空透明、全背景、ignore边缘 | null 及边界规则正确，无虚构 0 分 |
| NaN/Inf/负预测 | GT valid 不变，按失败规则阻止正常优胜声明 |
| 单进程与多 rank | 同逐图集合、同组均值，无重复尾部 |
| mask/GT不可读 | predictor 成功且等价，evaluator 单独拥有标签权限 |
| 文件名和样本ID改变 | 相同 RGB 的预测不变 |
| 合并与重复合并 | 第一次误差在锁定范围，第二次拒绝 |
| 标签位置错配 | evaluator 依据 sample_id/RGB/hash/transform 显式报错 |

### 12.3 结果目录建议

以下 `exports/`、`predictions/`、`evaluation/` 都是 01 篇配置的产物根目录下的逻辑子目录，不要求直接写到仓库根。运行时解析为完整的 `export_dir`、`prediction_dir`、`evaluation_dir`，在 `resolved_config.yaml` 中保存；训练运行使用同一约定下的 `run_dir`。

```text
exports/<export_id>/
  export_manifest.json
  model_config.yaml
  preprocessing.json
  weights/...
  checks/merge_equivalence.json
  checks/rgb_only_isolation.json
  checks/allowed_runtime_paths.json
  resource/inference_profile.json
predictions/<test_run_id>/
  rgb_manifest.jsonl
  prediction_manifest.jsonl
  depth/*.npy
  predictions.complete.json
  failures.jsonl
evaluation/<evaluation_id>/
  test_protocol.lock.json
  per_image.csv
  per_group.csv
  domain_summary.json
  paired_by_seed.csv
  bootstrap_by_seed.json
  best_and_last.csv
  source_ablation.csv
  decoder_ablation.csv
  failure_report.json
  qualitative/...
  report.md
```

Test 标签清单及原始数据由环境配置指向服务器受控目录；不放进公开导出包。结果文件中的记录应足够复核公式与覆盖，不只保存一张最终表截图。

## 13. 未来 CLI 设计

以下为项目实现后的 Linux 目标命令，当前文档本身不提供可运行脚本。公共运行参数与配置结构由工程篇统一。

```bash
# 导出前用固定 Dev RGB 完成数值检查；不打开Test
python -m transdepth.cli.export \
  --checkpoint <locked_best_or_last> \
  --config configs/export/rgb_only.yaml \
  --paths configs/paths.local.yaml --output <export_dir>

# Dev同样先存预测，再评价；用于checkpoint规则验证
python -m transdepth.cli.predict \
  --export <export_dir> --paths configs/inference.paths.local.yaml \
  --rgb-manifest <q_dev_rgb_manifest> \
  --output <dev_prediction_dir>
python -m transdepth.cli.evaluate \
  --predictions <dev_prediction_dir> --paths configs/paths.local.yaml \
  --labels <q_dev_label_manifest> \
  --config configs/evaluation/main_384x512.yaml \
  --role dev --output <dev_evaluation_dir>

# Test预测阶段只收到RGB；Test标签仅由独立evaluate进程读取
python -m transdepth.cli.predict \
  --export <locked_export_dir> --paths configs/inference.paths.local.yaml \
  --rgb-manifest <sealed_test_rgb_manifest> \
  --test-lock <test_protocol.lock.json> \
  --output <test_prediction_dir>
python -m transdepth.cli.evaluate \
  --predictions <test_prediction_dir> --paths configs/paths.local.yaml \
  --labels <sealed_test_label_manifest> \
  --test-lock <test_protocol.lock.json> \
  --config configs/evaluation/main_384x512.yaml \
  --role test --output <test_evaluation_dir>
```

evaluate 的 `--role=test` 必须要求 Test lock 与 prediction completion manifest，且 `--scale-alignment` 不提供可开启选项；解析到旧配置中的非 `none` 值时直接报错。predict 可接受多个模型输出目录批处理，但不能从 Test 指标动态选模型。

## 14. 常见失败与处置

| 失败 | 首查 | 处置 |
|---|---|---|
| 合并后形状对但输出大变 | [out,in]转置、Q/K行、eta、重复合并、bias mask | 停止导出，修后重跑Dev等价 |
| 断开标签目录后predict失败 | 偷用训练dataset、GT有效域crop、teacher import | 修边界，不以补回标签解决 |
| 原图/画布误差坐标错位 | pad和resize元数据、depth最近邻、GT帧匹配 | 修预处理版本，重跑所有配对 |
| 多卡指标变好 | sampler补齐重复、丢尾部、rank均值平均 | 汇总唯一逐图记录重算 |
| empty_region数随模型改变 | valid依赖预测 | 修GT-only有效域 |
| NaN图被删除 | 评估器先筛finite预测 | 恢复完整清单，按失败策略报告 |
| H2仅best优于H1 | 选择步数、过拟合、选模规则 | 同时报last和完整Dev曲线 |
| 新known数值与历史不一致 | 子集/分辨率/区域/训练权限 | 分表描述，不能直接排名 |
| novel CAD曾在62CAD中出现 | 联合CAD身份审计 | 改为留出场景，另报已见/未见CAD子集 |
| Test开封后想调lambda/尺度 | 评价锁已失效 | 新留出或明确探索，保留旧结果 |
| 无GPU实测却资源表有数字 | 理论预算被当实测 | 改为pending，等实际profile |

**完成标准**：代码、权重身份、数据协议、可复现训练状态、纯 RGB 部署接口、预测落盘与离线评价均有独立验收记录；主结论来自锁定的 H1/H2 同容量配对，而不是 oracle 热图、训练 loss 或单一漂亮样例。
