# FAR：62CAD＋ClearGrasp（含部分真实训练）局部修订与逐模块最小验证计划

修订日期：2026-09-15。

依据：[原三架构整合研究计划](<F:/桌面/push_CVPR/ViT_Try/paper_thought1/paperthought1/FAR_ClearGrasp_RGBOnly_DINOv3_H16plus_8x3090_三架构整合研究计划.md>)。

**保留原论文的 FAR 主线、DINOv3 H/16+ 主骨干、三种解码架构、单层少量 Q/K LoRA 和 RGB-only 推理。修改训练数据协议：RFTrans 的 62CAD 合成数据＋ClearGrasp 原始合成数据＋一部分 ClearGrasp 真实数据；最终在独立 ClearGrasp 留出数据上推理和评测。**

用户已明确选择“加入部分真实数据训练，并重新划分独立真实测试集”。因此，本版替换原文中“所有 ClearGrasp 真实数据只作终评”的规定。原文件保留，本文件作为修订执行版，按下表更新相关章节。本文没有进行新数据下载验收、GPU 训练或性能测量；所有样本量、步数和阈值均为启动预算或预注册建议。

## 0. 具体改哪里，保留什么

| 原文位置 | 本次处理 | 保留的研究问题 |
|---|---|---|
| §0、§3 数据来源与职责 | 改成三来源训练，重新划分真实 Train/Dev/Test | 同一任务、同一米制前表面深度 |
| §1、§2 模型和几何 | 保留；补充 62CAD 深度编码与坐标验收 | H/16+、384×512、单 RGB 输入 |
| §4–§9 FAR 理论 | 核心公式不改；补充分来源教师覆盖率和真实标签噪声检查 | 有益干预能否通过 Q/K 学到 |
| §10–§11 三架构 | 保留，补充最低成本验证顺序 | A：DPT；B：U-Net；C：双分支 U-Net＋DPT |
| §12–§13 资源与训练 | 保留 8×3090 上限；新增三来源采样和曝光次数记录 | 公共预热→严格分叉，等预算比较 |
| §14 实验矩阵 | 保留 B0–B2/H0–H2；增加数据来源与 FAR 对照 | 分清数据、真实监督、普通 LoRA、FAR 的贡献 |
| §15–§16 终评与决策 | 改为“含真实训练的独立留出评测” | 不借测试 GT 调参或校准尺度 |
| §17 跨材质扩展 | 继续作为后续扩展 | 不把配对 RGB 教师加入当前主线 |
| §18–§19 贡献和验收 | 贡献主张保留；新增数据协议验收 | 方法有效性须靠实际实验 |

论文仍依次回答：

1. 保持当前 V，只改变少量 attention 关系，能否改善前表面深度？
2. 单 RGB 学生能否通过小规模 Q/K LoRA 学到有用关系？
3. 同数据、同容量、同训练预算下，FAR 是否优于普通 LoRA？
4. 这种增益能否在独立真实图像与不同解码器上成立？

使用 RFTrans 数据不要求引入 RFNet、折射流预测、法线网络或 RGB-D 全局优化；这些会改变原论文的方法范围。

## 1. 修订后的数据协议

### 1.1 “RFTrans 的 62CAD”指什么

RFTrans 论文的 62CAD 数据包含 5 类、62 个 CAD 模型；论文报告 5,000 张合成训练图和 1,000 张合成测试图。作者还另用 ClearGrasp 的 9 个 CAD 生成过 5,000 张训练图。两者应分别登记，后者也不等于 ClearGrasp 原始发布图像。[RFTrans 原文 §III-D、§IV-A](https://arxiv.org/html/2311.12398v2)

作者发布包的顶层 62CAD 数据与其 ClearGrasp 重渲染部分需分开识别。发布页说明写 train/val，实际目录需核对 valid 等名称；目录名不能代替论文中的 split 职责，也不能仅据名称断定 valid 与论文 test 的严格对应。[作者数据发布页](https://huggingface.co/datasets/robotflow/rftrans)

本版的 62CAD 指上述 62 个 CAD 模型的数据，不是 62 张图片或 62 个类别；不默认纳入 RFTrans 的 ClearGrasp 重渲染部分。实际完整对数以下载后的 manifest 为准。

### 1.2 训练和评测职责

记 R 为 RFTrans-62CAD，S 为 ClearGrasp 原始合成数据，Q 为 ClearGrasp 真实数据。

| 集合 | 来源与划分建议 | 用途 |
|---|---|---|
| \(D_R^{tr}\)、\(D_R^{dev}\) | 62CAD 官方训练来源，按可确认场景组约 90%/10% | 训练；源域诊断 |
| \(D_R^{hold}\) | 62CAD 官方留出包单独登记，先核对职责 | 本轮不参与选择，可保留供后续审计 |
| \(D_S^{tr}\)、\(D_S^{dev}\) | ClearGrasp 原始合成训练来源，按组约 90%/10% | 训练；合成开发 |
| \(D_{sel}\)、\(D_{conf}\) | 沿用原计划：ClearGrasp 独立合成验证来源，按组约 50%/50% | oracle 选择；锁定配置的一次确认 |
| \(D_Q^{tr}\)、\(D_Q^{dev}\)、\(D_Q^{test,k}\) | 从原 real-known 按可确认场景组约 60%/20%/20% 重划分 | 真实训练；目标域开发；独立真实 known 留出 |
| \(D_Q^{test,n}\) | 原 real-novel 整组保留，检查与其他源的身份关系 | 主终评域；不训练、不选模型 |
| \(D_S^{test}\) | 原 Syn-novel 保留 | 合成终评辅助域 |

比例按独立组执行，不能按图像强行凑整。实际每组图数可能不同，不事先声称真实训练图数恰好为某个数字。

模型、学习率、关系系数和三架构选择以 \(D_Q^{dev}\) 为目标域依据，同时约束背景/边界退化；\(D_S^{dev}\) 监控合成退化，\(D_R^{dev}\) 只作源域诊断，不将三域误差混成一个有利均值。

**划分时必须完成：**

1. 同场景、近邻视角、同次采集、同场景不同相机，以及透明/喷涂配对绑定为同一泄漏组。按最高可确认的共同采集单元切分。
2. 可行时按对象、距离、透明面积分层，同时保证组不可拆。独立组不足时用预先规定的分组折；无法恢复组身份时明确报告“图像留出”，不称独立场景泛化。
3. 跨 R/S/Q 检查 hash、近重复、CAD 身份和对象身份。相同 CAD 不一定是图像泄漏，但会改变“未知形状”的结论。
4. 原 real-novel、Syn-novel 只表示原数据集分组。只有核实其几何身份在联合训练和选择数据中均未出现，才称“相对联合数据未见的形状”。否则另报已见/未见 CAD 子集，或称“原 novel 组的留出场景评测”。
5. Select/Confirm 仍来自合成 CG，因此 oracle 的正结论首先只适用于合成域。真实效果通过真实 Dev 学生对照和真实 Test 判断；不称已经验证真实 oracle。
6. 若后续研究真实 oracle，先独立安排未用于训练/调参的真实 Select/Confirm，另立协议；不能复用真实 Dev 冒充独立确认，更不能用真实 Test 诊断后继续调参。

### 1.3 标签只在允许的阶段读取

| 数据来源 | RGB 输入 | 深度监督 | 透明监督 |
|---|---|---|---|
| R：62CAD | 透明场景 RGB | 理想几何 depth，经编码和坐标验收 | 按作者标签通道解析为二值/ignore |
| S：ClearGrasp 合成 | 原始透明 RGB | 已发布 rectified depth | 已发布透明 mask，核对语义 |
| Q：ClearGrasp 真实训练 | transparent RGB | 对应 opaque-depth GT | 对应可信 mask |
| Q：真实推理/测试 | transparent RGB | 模型不读取；评估器预测后读取 | 模型不读取 GT；仅评估器使用 |

ClearGrasp 官方评估代码区分合成 depth-rectified.exr 与真实 opaque-depth-img.exr；真实 transparent-depth-img.exr 是传感器观测，不能替代 GT。[ClearGrasp 官方评估实现](https://github.com/Shreeyak/cleargrasp/blob/master/eval_depth_completion/eval_depth_completion.py)

真实训练组的 opaque RGB 在主线中不作输入或教师。C 的分割输出来自同一透明 RGB。标签不通过 DINOv3 patch masks、融合 mask、内参或文件名传入预测器。

### 1.4 最小数据清单

~~~text
sample_id, source, official_split, assigned_role
rgb_path, depth_gt_path, mask_path, valid_mask_path
depth_encoding, unit_to_m, depth_coordinate, invalid_rule
object_ids, cad_ids, scene_id, capture_id, camera_id, leakage_group_id
pair_group_id, rgb_sha256, preprocessing_version
~~~

未知身份字段写 unknown，不从文件编号编造 scene_id。预处理、异常剔除和深度有效范围由训练/开发数据与发布约定确定，然后冻结用于测试。

## 2. 总模型、总损失和最小实验路线

### 2.1 模型不变

$$
\widehat Z=f_{\phi,\psi}(I)
=\operatorname{PositiveDepth}\left[
D_\psi\left(\mathcal R(E_{\theta_0,\phi}(I))\right)\right].
\tag{F1}
$$

\(\theta_0\) 是冻结 DINOv3；\(\phi\) 只是一层选中头的 Q/K LoRA；\(\psi\) 包含重组和读出，C 还含共享收缩路径、两扩张分支与融合。A/B/C 均输出相同米制前表面深度。

### 2.2 三来源采样是新增的训练组织项

$$
\mathcal L_{\mathrm{mix}}
=\sum_{d\in\{R,S,Q\}}\pi_d\,\mathbb E_{x\sim D_d^{tr}}
\left[\mathcal L_{\mathrm{depth}}(x)
+\lambda_{\mathrm{rel}}\mathcal L_{\mathrm{rel}}(x)
+\mathbb1_{\mathrm{Seg}}\lambda_{\mathrm{seg}}\mathcal L_{\mathrm{seg}}(x)\right].
\tag{F2}
$$

起点 \((\pi_R,\pi_S,\pi_Q)=(7/16,7/16,2/16)\)：一个完整有效 batch=16 的更新含 7 张 R、7 张 S、2 张真实 Q，在 8 卡×microbatch 1×累积 2 的全局样本表中实现。比例是待验证起点。

按此配比采样后，对 16 个逐样本损失等权平均即可，**不要再乘一次 \(\pi_d\)**。若用每域独立均值实现 F2，才显式乘权重。空监督区域在每样本内部重新归一化。

每域每图平均曝光为

$$
\mathrm{Exposure}_d=\frac{16T\pi_d}{|D_d^{tr}|}.
\tag{F3}
$$

T 是完整 optimizer 更新次数。真实训练集若有 100 张，2,000 更新就约有 40 次/图曝光；这是算术示例，不是实测图数。记录真实组记忆化、训练/开发差距、best 和 last checkpoint，不能照搬原文基于 50k 图的 epoch 换算。

### 2.3 从最便宜的验证开始

~~~mermaid
flowchart TD
  A[数据身份、编码与划分] --> B[H加DPT小样本拟合与资源验收]
  B --> C[混合数据公共预热]
  C --> D[固定读出上的FAR oracle]
  D --> E[相同QK LoRA：无FAR与有FAR]
  E --> F[分离62CAD、真实监督与FAR收益]
  F --> G[B/H六组与三个配对种子]
  G --> H[U-Net和双分支加DPT]
  H --> I[锁定模型与纯RGB导出]
  I --> J[ClearGrasp独立真实终评]
~~~

M1–M14 分别给出核心公式和最小验证。原文详细推导继续有效；每个阶段区分“实现正确”和“研究上有效”。

## 3. 逐模块公式与最小验证

### M1. 三来源深度统一与有效域〔原 §2–§3〕

**核心公式**

$$
Z^*_p=s_dD^{raw}_p,\qquad
Z^*_p=\frac{r_p}{\|K_{cam}^{-1}(u,v,1)^\top\|_2}
\quad\text{仅当输入是射线距离 }r_p.
\tag{M1}
$$

\(s_d\) 由实际编码确定；已是光轴 z-depth 就不再做第二个转换。

RFTrans 官方脚本对理想深度 PNG 使用 \(Z^*=D_{\rm png}\cdot3/2^{16}\) 的解码示例，对 active-depth PNG 则以毫米保存。不能混用；本任务只用理想深度监督。该比例仅适用于核实匹配的发布版本，不能套用到所有 PNG/EXR。[官方生成脚本](https://github.com/LJY-XCX/Unity-RefractiveFlowRender/blob/main/generate_active_depth.py)

**按步实现**

1. 每来源至少抽 30 组，真实图来自 Train/Dev；输出 RGB/GT/mask/valid 叠图和深度 1%/50%/99% 分位数。
2. 每格式独立转换，PNG 保留 bit depth。核对零值、最大值、far-plane 饱和值是否为无效标签，不将所有正数一概当真值。
3. 用同帧 EXR、相机或生成几何核对光轴/射线约定，特别检查图像角落和薄壁首命中。缺少证据时记录验收未完成，不能凭“看起来像米”放行。
4. 保持长宽比缩放、padding 到 384×512；depth/mask/valid 同步最近邻，RGB 固定插值。padding 不进监督，分来源记录 padding 比例。
5. 核验几何时同步更新内参；预测接口仍只有 RGB。

**通过标准：** 编码、单位、首命中语义和有效域有证据；抽查无错位；跨 split 无已知泄漏。尚不证明混合训练有效。

**失败处理：** 修解析器或隔离错误样本；不靠逐图归一化、猜常数或测试 GT 校准掩盖问题。

### M2. 透明 mask 与可靠 patch〔原 §3、§5.1〕

**核心公式**

$$
v_i=\frac{|V\cap\mathcal P_i|}{|\mathcal P_i|},\quad
t_i=\frac{|V\cap M_T\cap\mathcal P_i|}{|V\cap\mathcal P_i|},\quad
z_i^*=\operatorname{median}_{p\in V\cap\mathcal P_i}Z_p^*.
\tag{M2}
$$

这里 \(\mathcal P_i\) 是第 i 个 16×16 patch，\(M_T\) 为可信透明像素集合，明确 \(V=V_{\rm rel}=V_{\rm depth}\cap V_{\rm mask-known}\)，即深度有效且类别标签已知的内容像素。ignore 不进关系的透明/背景分组或分割项。仅深度有效但类别未知的像素若保留深度监督，应单独记录有效域，不能默认为背景。

$$
i\in T\iff v_i\ge0.9,\quad t_i\ge0.9,\quad
\operatorname{IQR}_{p\in V\cap\mathcal P_i}(\log(Z_p^*/1\mathrm m))\le0.1.
\tag{M3}
$$

背景 B 使用 \(1-t_i\ge0.9\)，其余条件相同；\(K_{\rm rel}=T\cup B\)。混合、无效、padding patch 及前缀 token 不进可靠集合。分母为零时直接归入无效集合。

RFTrans 官方 loader 从 mask 绿色通道取透明标签，不能把 RGB 任意非零值都视为透明。[官方数据读取实现](https://github.com/LJY-XCX/RFTrans/blob/main/pytorch_networks/rgb2normal/dataloader.py)

**按步实现**

1. 核对每源色表，不确定边缘定义 ignore；缺失 mask 不当全背景。
2. T/B/U 叠到 RGB，抽查杯口、细柄、孔洞与遮挡。边界被排除的只是关系项，仍可信的像素继续参与深度项。
3. 分 R/S/Q 统计 T/B 数量、可靠 key 数、无关系样本率和透明覆盖率。
4. 真实数据检查喷涂替换/配准误差；先沿用共同规则，确有证据才预定更严格有效域，对 H1/H2 一致应用。

**通过标准：** 可信 patch 覆盖正确透明表面且数量足够；孔洞未被错误填实。

**失败处理：** 修标签或跳过关系项。若真实域几乎无可信关系，可另立“真实只提供深度/分割，FAR 仅在合成域训练”的变体，不能声称真实教师已经生效。

### M3. H/16+ 特征与四尺度重组〔原 §1、§10.1〕

**核心公式**

$$
F_s=\operatorname{Grid}(\operatorname{PatchTokens}(E^{(\ell_s)}(I))),\quad
P_s=\mathcal R_s(F_s),\qquad \ell_s\in[7,15,23,31].
\tag{M4}
$$

原特征均为 \(1280\times24\times32\)。投影至 c=128 后重组为 96×128、48×64、24×32、12×16，即 /4、/8、/16、/32。5 个前缀 token 不变成图像网格。

**按步实现**

1. 锁定原文官方 H/16+ 配置、权重 hash 和预处理，检查 32 blocks、20 heads、head dim=64。
2. 单图打印四层及重组形状，检查空间坐标。
3. 每来源各 8 张固定训练图，配 A 头训练 300–1,000 更新，关闭随机增强。
4. 与仅从训练集估计的常数深度比较；检查误差下降与距离差异。

**通过标准：** 模型身份、形状、单位正确，小样本可拟合。训练 MAE 较初始化下降约 50% 可作排错参考，不是通用定律或泛化证据。

**失败处理：** 查数据、loss、输出映射和学习率；弱读出不适合用来否定 oracle。

### M4. A：DPT 深度读出〔原 §10.2〕

**核心公式**

$$
Y_4=RB_4(P_4),\quad
Y_s=RB_s(RB'_s(P_s)+\operatorname{Up}(Y_{s+1})),\ s=3,2,1,\quad
u_A=\operatorname{Tail}(Y_1).
\tag{M5}
$$

RB 为原文的卷积残差块；GroupNorm/GELU、Tail 与正值映射按原计划统一。

**按步实现**

1. 完成 M3 的小样本拟合和单卡资源验收。
2. 冻结骨干，只训练重组和 DPT，在三来源混合数据上做公共 T0。
3. 固定间隔评估真实/合成 Dev，保存曲线与公共 checkpoint。
4. 读出有可用深度预测后固定它，进入 oracle。

**通过标准：** A 成为可用公共读出；不是证明 DPT 最佳。

**失败处理：** 先解决欠拟合、真实过拟合和单位问题，暂不并行开启三架构大扫。

### M5. 保留原偏好的 GT 关系教师〔原 §4–§5〕

**核心公式**

$$
A^0_{ij}=\operatorname{softmax}_{j=1..N}
\left(\frac{\widetilde q_i^0(\widetilde k_j^0)^\top}{\sqrt d}\right),\qquad
c_{ij}=\min\left(\frac{\log^2(z_i^*/z_j^*)}{2\sigma_z^2},c_{\max}\right).
\tag{M6}
$$

波浪号表示官方 RoPE 后的 Q/K；\(S_i^0\) 是该 head 原始完整 key logits，\(\Delta\) 为可靠 keys 上概率单纯形。教师保留 CLS/register 等补集概率，但这些 token 不具有像素深度标签。

$$
m_i=\sum_{j\in K_{\rm rel}}A^0_{ij},\qquad
q_i=\arg\min_{q\in\Delta}
\left[D_{\rm KL}(q\|A^0_{i,K_{\rm rel}}/m_i)+\sum_jq_jc_{ij}\right]
=\operatorname{softmax}_{K_{\rm rel}}(S_i^0-c_i).
\tag{M7}
$$

$$
G_{ij}=\begin{cases}m_iq_{ij},&j\in K_{\rm rel}\\A^0_{ij},&j\notin K_{\rm rel},\end{cases}
\quad
P_i=\begin{cases}(1-\beta)A_i^0+\beta G_i,&i\in T\\A_i^0,&i\notin T.\end{cases}
\tag{M8}
$$

起点 \(\sigma_z=0.1,c_{\max}=4,\beta\in\{0,0.25,0.5\}\)。作用是重分配可靠 keys 内的概率，保持总质量和补集；同深度不必然意味着有益关系。

**按步实现**

1. 小矩阵检查 P 非负、行和为 1、可靠总质量为 m、补集逐元素不变。
2. 检查 beta=0、代价全零、空可靠集、m 极小时回退；教师 FP32 且 detach。
3. 接真实 backbone 的 post-RoPE logits，分来源统计 m、KL(P||A0) 和回退率。
4. 同 RGB 的教师不随 LoRA 更新改变；单适配层上游冻结且确定性，可由同一 X 重算原 A0。

**通过标准：** 数学约束正确、有实际干预量、覆盖可解释；教师是否有益由 M6 判断。

**失败处理：** 先查 mask、纯度、尺度与数值，不立即叠加物理教师或全网关系监督。

### M6. Oracle：关系改变能否改善深度〔原 §6〕

**核心公式**

$$
O_h^{oracle}=P_hV_h^0,\quad
\Delta_g(c)=e_g(f_0)-e_g(f_c^{oracle}),\quad
e_g(f)=\frac1{|g_T|}\sum_{I\in g_T}
\frac1{|\Omega_T(I)|}\sum_{p\in\Omega_T(I)}|f(I)_p-Z_p^*|.
\tag{M9}
$$

c 是 block/head/beta 候选，正 Delta 表示改善。固定下游及读出，干预后完整重算。无有效透明图像的组不参与透明均值。

**按步实现**

1. 先做 P=A0 身份干预，核对 post-RoPE 显式 AV 与官方 SDPA，不能只改热图。
2. 从新混合数据 T0 重新选头，旧 CG-only 选择不自动继承。
3. H blocks [15,23,31]，每层预抽 8 heads，正 beta 两档，共 48 候选。Select 约 100–200 图初筛，独立组数量优先。
4. 前 3 候选在完整 Select 比较透明/背景/边界，锁定 1 个。
5. 独立 Confirm 仅检验锁定方案，按组做 2,000 次配对 bootstrap。

**通过标准：** 透明 MAE 平均改善、单侧 95% 收益下界为正，背景/边界不过预定容忍量。组数少则标证据不足。

容忍量沿用原起点：每域分别由相应 Dev 固定 \(\epsilon_B,\epsilon_{edge}=\max(1\mathrm{mm},2\%\times\text{该区域基线MAE})\)，不看 Confirm 后调整。

**失败处理：** 排除实现问题后，本范围无支持就停止当前教师主张。重搜索需新确认数据或预定折。Oracle 使用 GT，是特权诊断，不是部署模型或学生严格上界。

### M7. 单层单头 Q/K LoRA〔原 §7、§12.3〕

**核心公式**

$$
Q^s_{sel}=XW_Q^0+b_Q^0+\eta XU_QB_Q,\qquad
K^s_{sel}=XW_K^0+b_K^0+\eta XU_KB_K,
\tag{M10}
$$
$$
U\in\mathbb R^{C\times r},\ B\in\mathbb R^{r\times kd},\quad
P_\phi=2r(C+kd)=10{,}752
\quad(C=1280,d=64,r=4,k=1).
\tag{M11}
$$

增量在 RoPE 前；V、原 Q/K、其余骨干冻结。PyTorch packed QKV 按对应输出行插入，不能误改 V 或有效 bias mask。

**按步实现**

1. 两份同 T0 解码器，U 同随机初始化、B=0。
2. 零 LoRA 等于基线；初始 B 梯度可非零、U 梯度为零正常，不能双因子全零。
3. lambda_rel=0，仅深度做一次更新，确认梯度能到 LoRA；适配层和 suffix 保留 autograd。
4. 原骨干 checksum 不变，仅允许参数变化。
5. 固定 32–64 张混合训练图跑 300–1,000 更新，同时看独立 CG Dev 的深度、KL、背景误差。

**通过标准：** 深度可训练 LoRA，关系可拟合；KL 下降不等于深度提升。

**失败处理：** 查 no_grad、旧 suffix 缓存、QKV 切片、双零初始化。算子正确但容量不足时单独考虑 r=8，不同时改层/头/教师。

### M8. 深度、关系与分割损失〔原 §8、§11〕

**核心公式**

$$
\widehat Z_p=1\mathrm m[\operatorname{softplus}(u_p)+10^{-6}],\quad
e_p=\log(\widehat Z_p/Z_p^*),\quad
\mathcal L_{\rm depth}=\operatorname{AvgBal}_{\Omega_T,\Omega_B}\rho_{0.1}(e_p),
\tag{M12}
$$
$$
\rho_\delta(e)=\begin{cases}e^2/2,&|e|\le\delta\\\delta(|e|-\delta/2),&|e|>\delta.\end{cases}
$$
$$
\mathcal L_{\rm rel}=\frac1k\sum_h\operatorname{AvgBal}_{i\in T,B}
\sum_{j=1}^N\operatorname{sg}(P_{h,ij})
\log\frac{\operatorname{sg}(P_{h,ij})}{A^s_{h,ij}},\quad
\frac{\partial\mathrm{KL}(P_i\|A_i^s)}{\partial S^s_{ij}}=A^s_{ij}-P_{ij}.
\tag{M13}
$$
$$
\mathcal L_{\rm seg}=\operatorname{AvgBal}_{m=1,m=0}\mathrm{BCEWithLogits}(a_p,m_p),\quad
\mathcal L=\mathcal L_{\rm depth}+\lambda_{\rm rel}\mathcal L_{\rm rel}
+\mathbb1_{\rm Seg}\lambda_{\rm seg}\mathcal L_{\rm seg}.
\tag{M14}
$$

AvgBal：每样本先对非空透明/背景组各取均值，再等权平均；样本间再平均。关系行先 sum 全部 keys，只抽 query，不裁剪 keys 或多除 N。

**按步实现**

1. 人造预测=GT 时深度损失为零，预测=2GT 时为正；不逐图尺度对齐。
2. 手算小分布 KL，核对方向、reduction、detach 与 logits 梯度。
3. 检查空透明/背景、无关系、缺分割监督；无关系图仍算合法深度项。
4. 分 R/S/Q 记录原始/加权 loss、LoRA 梯度范数；每类最多 128 query。
5. lambda_rel 候选为 {0,0.01,0.1,1}，lambda_seg 为 {0,0.01,0.1}；先短程 Dev 筛选，再固定做配对复核。

**通过标准：** 数值和梯度正确；有关系项的真实 Dev 深度优于普通 LoRA，才有任务意义。

**失败处理：** 优先修 reduction 和权重，此阶段不堆叠法线、折射流、Dice 或边缘损失。

### M9. B：U-Net 读出〔原 §10.3〕

**核心公式**

$$
E_1=RB(P_1),\quad E_s=CC_s([P_s,\operatorname{Down}(E_{s-1})]),\ s=2,3,4,
$$
$$
U_4=RB(E_4),\quad U_s=CC'_s([E_s,\operatorname{Up}(U_{s+1})]),\ s=3,2,1,\quad
u_B=\operatorname{Tail}(U_1).
\tag{M15}
$$

CC 是拼接后卷积，2c 通道回到 c；B 最后不再接 DPT。

**按步实现**

1. 沿用相同 H 特征、reassembly、c=128、Tail、数据和损失，通过形状、梯度及小样本拟合。
2. 完成 H-B 公共预热。
3. 第一轮复用 A 锁定的 FAR 位置/beta，做 H-B1/H-B2 配对；参数、初始化、数据完全相同。
4. 比较透明/边界/背景 MAE、参数、显存和推理延迟。

**通过标准：** B2 优于 B1 才支持 FAR 在 B 成立；B 优于 A 是解码器系统比较，须结合成本解释。

**失败处理：** 保留 A。B 重选头属于额外实验，需新声明选择/确认预算。

### M10. C：几何/语义双分支与 DPT〔原 §10.4〕

**核心公式**

$$
G_4=RB_g(E_4),\quad S_4=RB_m(E_4),\quad
G_s=CC_s^g([E_s,\operatorname{Up}(G_{s+1})]),\quad
S_s=CC_s^m([E_s,\operatorname{Up}(S_{s+1})]),\ s=3,2,1,
$$
$$
\widehat M=\sigma(\operatorname{SegTail}(S_1)),\quad
\widetilde P_s=P_s+\operatorname{Conv}_{1\times1,s}([G_s,S_s]),\quad
u_C=\operatorname{Tail}\left(\operatorname{DPTFuse}(\{\widetilde P_s\})_1\right).
\tag{M16}
$$

共享一套 U-Net 收缩路径 E，两条扩张分支，最终深度来自 DPT。融合后已有四尺度，不重复 token reassembly；几何支不新增受监督的中间深度。

**按步实现**

1. 小输入检查各尺度，只开深度损失，确认 G/S 两支有梯度。
2. 固定 24–48 张训练图拟合，随后比较结构相同的 C-noSeg 与完整 C。
3. C1/C2 使用相同分割权重，第一轮复用 A 锁定的 FAR 位置。
4. C 有值得保留的精度收益才开展 M11，否则停止扩大分支复杂度。

**通过标准：** 深度改善且成本可接受；mask IoU 改善不能代替深度改善。

**失败处理：** 检查融合、任务梯度冲突和过拟合，必要时降低分割权重。C 不成立不影响 A/B 上已建立的 FAR 证据。

### M11. C 的分割辅助与结构归因〔原 §11.2–§11.3〕

**核心公式**

写 \(\widetilde P=P+W_gG+W_sS\)，直接融合路径的深度梯度为

$$
\left.\nabla_G\mathcal L_{\rm depth}\right|_{\rm direct}=W_g^\top g,\quad
\left.\nabla_S\mathcal L_{\rm depth}\right|_{\rm direct}=W_s^\top g,\quad
g=\nabla_{\widetilde P}\mathcal L_{\rm depth}.
\tag{M17}
$$

总梯度还含到更细尺度的链式路径。这说明可训练性，不证明任务互利。

**按步实现**

1. C-noSeg/完整 C：是否需要分割标签？
2. A+Seg、B+Seg/完整 C：是否一般辅助监督即可解释收益？
3. C-noSemFuse：保留分割训练，去掉语义特征进入深度融合的路径。
4. C-noDPT：融合 /4 特征直接接相同 Tail。
5. 若 C 参数明显更多，缩窄 C 或调整对照宽度，使读出含 reassembly 参数约在 ±10% 内，并报告真实延迟。

**通过标准：** 每项结构主张有对应控制，所有 C1/C2 的分割监督相同。

**失败处理：** A+Seg 追平 C 时，将收益归为辅助监督，不能独占归给双分支融合。

### M12. 训练冻结路径与 8×3090 资源〔原 §12–§13〕

**核心公式**

$$
\nabla_\phi\mathcal L_{\rm depth}
=J_{H_{\ell^*},\phi}^{\top}
J_{F_{\rm suffix},H_{\ell^*}}^{\top}
J_{\widehat Z,F_{\rm suffix}}^{\top}
\nabla_{\widehat Z}\mathcal L_{\rm depth},\qquad
B_{\rm eff}=n_{\rm GPU}b_{\rm micro}a_{\rm accum}.
\tag{M18}
$$

该式示意一条经过 suffix 的链式路径；多个中间读出路径的贡献还需相加。冻结 suffix 权重不等于切断输入梯度。

**按步实现**

1. 单卡加载 H，冻结 prefix 用 no_grad；适配 block、suffix、读出保留梯度。backbone/RoPE 保持 eval。
2. microbatch=1，关系 FP32，其余混合精度；仅额外重算所选头抽样 query，但保留全部 keys。
3. 预热 20 更新，再测 100 完整更新，记录 allocated/reserved、时延、加载占比、NaN/OOM。
4. 需要时 checkpoint suffix；确认重算不污染教师缓存。
5. 4 卡累积 4 或 8 卡累积 2，均为有效 batch 16；固定来源样本表和独立增强随机流。有利时两组 4 卡并行跑 H1/H2。

**通过标准：** 允许参数更新正确，单卡约有 10% 显存余量，训练稳定。DDP 不把 8×24GB 变成单卡 192GB。

**失败处理：** 去意外全网 attention/重复教师，再 checkpoint、减少关系 query；必须改输入或宽度时对相关配对统一修改。

### M13. 合并与纯 RGB 推理〔原 §7.4、§12.5〕

**核心公式**

$$
W^{deploy}_{Q/K}=W^0_{Q/K}+\operatorname{Scatter}(\eta U_{Q/K}B_{Q/K}),\qquad
\widehat Z=f^{deploy}(I).
\tag{M19}
$$

**按步实现**

1. 在独立权重副本合并 Q/K，不覆盖训练时用于教师的原 Q/K。
2. 同批 Dev RGB 比较合并前后的 Q/K、选中头输出和最终深度。
3. 容差由 FP32 小输入与部署精度身份误差预先确定，不为通过错误实现反复放宽。
4. RGB-only 推理清单禁用标签路径访问，仍应输出相同深度。
5. 先保存预测，评估器再读 GT/mask。C 保留参与融合的语义特征支，可省略不用的分割图输出。

**通过标准：** 合并等价，输入只有 RGB，推理不依赖教师。

**失败处理：** 查转置、head 切片、重复合并、预处理和隐藏标签依赖；不在测试时用 GT mask 修正输出。

### M14. 评价与有效性判据〔原 §14–§16〕

**核心公式**

$$
\mathrm{MAE}=\frac1{|\Omega|}\sum_p|\widehat Z_p-Z_p^*|,\quad
\mathrm{RMSE}=\sqrt{\frac1{|\Omega|}\sum_p(\widehat Z_p-Z_p^*)^2},\quad
\mathrm{AbsRel}=\frac1{|\Omega|}\sum_p\frac{|\widehat Z_p-Z_p^*|}{Z_p^*}.
\tag{M20}
$$

$$
\delta_t=\frac1{|\Omega|}\sum_p\mathbb1\left[
\max\left(\frac{\widehat Z_p}{Z_p^*},\frac{Z_p^*}{\widehat Z_p}\right)<t\right],
\qquad G_{\rm FAR}=e_{\rm H1}-e_{\rm H2}.
\tag{M21}
$$

MAE/RMSE 展示为 mm 时乘 1000，AbsRel 无单位。沿用原文显式命名的 \(\delta_{1.05},\delta_{1.10},\delta_{1.25}\)。

**按步实现**

1. 统一 384×512 有效内容坐标去 padding 评估；历史 144×256 协议另表。
2. 分透明、背景、边界带（起点为内外各 3 像素），valid 不依赖预测。
3. 先逐图、再场景组宏平均；补充像素池化时明确命名，不能混用两种 RMSE。
4. 每种子列 H1/H2 差值；测试组配对区间和优化种子波动分别报告。
5. 保存预选样例、典型收益和最大退化；NaN/Inf 报失败率，不能删坏像素。

**通过标准：** 同协议 H2 优于 H1。原文约 2% 相对透明 MAE 改善可保留为预定实用门槛，背景/边界同时满足约束；单种子达标只是进入复核的理由。

**失败处理：** 区间跨零或种子不一致则记证据不足；测试后不挑子集、调尺度或更换赢家。

## 4. 新数据下最小而能说明问题的实验矩阵

### 4.1 首先只在 H＋DPT 上做六个学生配置

每个数据协议各自完成 T0 后分叉 H1/H2。H1 与 H2 使用同样 oracle 选定的 LoRA 位置，区别仅关系项。

| 数据协议 | 每有效 batch=16 的 R/S/Q 数 | 普通 LoRA | 同容量 LoRA＋FAR | 主要回答 |
|---|---:|---|---|---|
| CG-real：仅 ClearGrasp，含相同真实训练组 | 0/14/2 | CG-H1 | CG-H2 | 没有 62CAD 的对照 |
| Mix-real：用户指定主协议 | 7/7/2 | Mix-H1 | Mix-H2 | 联合数据下 FAR 是否成立 |
| Mix-syn：纯合成参照 | 8/8/0 | Syn-H1 | Syn-H2 | 加入真实监督的变化 |

最低预算顺序：先主协议 Mix-H1/H2，再 CG 两组，最后纯合成两组。“六个”指学生配置，不含预热、oracle、超参短跑和额外种子。

主协议 H0 继续作为等量训练的冻结头基线。此表中的 Mix-H1/H2 与原六组主表复用，不重复计数。

### 4.2 核心差分

在同一 Dev 或锁定的同一 Test 上，记 e 为透明 MAE：

$$
\begin{aligned}
G_{\rm FAR}^{mix}&=e_{\rm Mix-H1}-e_{\rm Mix-H2},\\
G_{\rm 62CAD}^{LoRA}&=e_{\rm CG-H1}-e_{\rm Mix-H1},\\
G_{\rm real}^{LoRA}&=e_{\rm Syn-H1}-e_{\rm Mix-H1},\\
I_{\rm data\times FAR}
&=(e_{\rm Mix-H1}-e_{\rm Mix-H2})-(e_{\rm CG-H1}-e_{\rm CG-H2}).
\end{aligned}
\tag{E1}
$$

第一项才是 FAR 主证据；第二项是固定总预算下用 62CAD 替换部分 CG 合成曝光的效果；第三项是同预算下真实样本替换部分合成曝光的效果。后两项不能解释为“其他曝光完全不变时单纯追加数据”的效果。

若单独声称数据量/多样性贡献，再补 CG 曝光不变的预算敏感性实验，报告增加更新数和 GPU-hours。

**配对条件：**

- 每协议内部 H1/H2 的 T0、初始化、层/头/秩、读出、数据顺序、增强、batch、更新数、优化器、分割监督和 checkpoint 规则相同。
- CG-real 与 Mix-real 使用同一真实划分和真实曝光表。
- CG-real 的 T0/T1 均不使用 62CAD，不能复用见过 62CAD 的主线预热权重。
- Mix-syn 的 T0/T1 均不使用真实训练；若超参仍按真实 Dev 选择，应称“使用真实验证的纯合成训练”，不是零真实数据访问。
- 各数据协议可按相同候选预算独立选头，比较完整流程；此时数据差分包含选择配置变化。若要隔离固定位置下的数据作用，补复用同一层/头/beta 的敏感性比较。
- 三协议总更新预算须预先统一锁定；同一规则选择 best checkpoint 不保证每组 best 出现步数相同，因此同时保留等步数 last 结果。

**跨协议 Confirm 规则：** 最小执行默认只确认主协议一次，数据控制组固定迁移其层/头/beta，不声称控制组各自通过独立 oracle 确认。若改做三协议独立选头，必须在任何 Confirm 开封前登记整个确认族，采用 Holm 等多重比较校正，或在独立组数量足够时预拆确认折；不能主协议确认后再临时追加两轮并仍称“只确认一次”。固定位置迁移得到的是该锁定位置下的数据效果，不代表每个数据协议已获得最佳位置。

### 4.3 原六组和三架构主表保留

主训练协议统一为 Mix-real：

| 骨干 | 冻结骨干＋DPT | 普通 Q/K LoRA＋DPT | 相同 LoRA＋FAR＋DPT |
|---|---|---|---|
| B/16 | B0 | B1 | B2 |
| H/16+ | H0 | H1 | H2 |

H 再扩展 H-B0/B1/B2、H-C0/C1/C2；H-A0/A1/A2 就是 H0/H1/H2。

原 12 个主配置不变；新增 CG-real 两个和 Mix-syn 两个学生配置后，共 **16 个独立配置**。全部做 3 种子为 48 条训练结果，仍不含机制/C 消融、预热、搜索和外部基线。最小验证按阶段做，不把完整论文预算写成 6 次训练。

优先复核 Mix-H1/H2 三个配对种子，再扩展其他配置；不省掉主配对而堆跨架构展示。

### 4.4 FAR 机制的最少强对照

在 H-A 主协议上保留原 §14.4 的关键控制：

1. P=A0、lambda>0：是否只是保持原关系的正则？
2. \(q\propto e^{-c}\)，其余 m/补集/beta 相同：条件分布内原 attention 偏好是否必要？
3. 2D 空间代价：是否主要是局部平滑？
4. 仅在可靠 keys 内置换最终 P 的概率值：保持质量与熵，检验正确对应。
5. 同容量随机/固定末层头：任务收益选头是否值得？

前两项优先小预算筛查；最终若声称完整机制，补足对应控制和相同调优预算。打乱深度代价不保证最终熵不变，不能替代第 4 项。

## 5. 按步执行排程

### 第 1 步：冻结协议和来源

- 输出三来源 manifest、跨 split/CAD 重叠表、真实 Train/Dev/Test。
- 封存真实 Test 预测评估入口，各阶段只加载相应清单。
- 通过 M1/M2；真实独立组或标签条件不足时先修协议。

### 第 2 步：单卡最小基线

- 每来源 8–16 张固定训练图，H＋DPT，300–1,000 更新。
- 完成模型、形状、loss、梯度与显存检查。
- 只判断“实现能学”，不据训练图声称泛化。

### 第 3 步：主协议公共预热 T0

- 初轮约 2,000 完整更新，R/S/Q=7/7/2。
- 每 250–500 更新看真实 Dev，依据欠拟合/过拟合确定公共预算。
- 如果欠拟合且 Dev 持续改善，可在原文最多 20k 范围内重新预定预算；20k 不作为必跑默认值。
- discovery seed=17 锁定超参，保留 oracle 固定 T0。

### 第 4 步：执行 oracle

- 身份干预通过后，在 CG Select 按有限候选筛选。
- Confirm 一次确认，锁定一个 block/head/beta。
- 此阶段只证明特权干预价值，未证明 RGB 学生能做到。

### 第 5 步：H0/H1/H2 严格分叉

- 0 组继续头；1 组头＋LoRA；2 组同参数＋关系。
- decoder 优化器状态统一重置，学习率日程重启。
- 先约 2,000 更新检查真实 Dev，再共同锁定正式 T1；原文 10k 是需评估的上限起点，受真实过拟合约束。
- lambda_rel=0 是必要强对照。三组都有相同真实数据访问。

### 第 6 步：判断 FAR 是否值得继续

- 比较 Mix-H1/H2 的真实 Dev 透明/背景/边界和 KL。
- 达到预定实用门槛且背景可控，进入 seeds [17,29,43] 复核。
- 每种子独立训练 T0，再内部共享分叉；位置和超参跨种子复用，不重新消耗 Confirm。
- KL 下降而深度不改善，记录为“可拟合关系但未建立任务收益”，不扩大架构掩盖问题。

### 第 7 步：数据收益控制

- CG-real H1/H2 保持相同真实划分和曝光。
- Mix-syn H1/H2 明确真实 Dev 访问权限。
- 报 E1 差分，分清 62CAD、真实监督和 FAR。

### 第 8 步：B/H 六组与 U-Net/C

- B 按原相对层位协议独立选头，不复制 H 的绝对 head 编号。
- H 的 B/C 解码器首轮复用 A 的 FAR 位置，各自 1/2 配对。
- C 有实质深度优势后做 M11；C 不必获胜才认可 FAR。

### 第 9 步：机制、基线与导出

- 补 §4.4 机制控制和原计划必要的强 RGB-only 基线。
- 外部基线有相同三来源训练/监督权限；RFTrans/ClearGrasp 原方法的 RGB-D 数字另列模态和训练协议。
- 通过 M13 合并、RGB-only 接口与单卡资源验收。

### 第 10 步：锁定后终评

- 主假设：在封存 \(D_Q^{test,n}\) 上 H2 比 H1 透明 MAE 改善且背景/边界满足约束；novel 表述服从身份审计。
- 同时报新 \(D_Q^{test,k}\)、原 Syn-novel，逐域报告。
- 新 known 测试只是原 real-known 的子集，不能标为原完整 benchmark。
- Test 开封后不再改架构、lambda、数据配比或追加真实训练来选赢家；下一轮需要新留出或公开探索性质。

## 6. 配置、记录和论文表述

### 6.1 本轮新增/替换的配置

以下是规格，不是已可运行系统；实施时填写已验收路径和 hash。

~~~yaml
protocol: FAR_R62_CGSyn_CGReal_train_v1
backbone: dinov3_vith16plus
input_hw: [384, 512]
feature_blocks_0based: [7, 15, 23, 31]
decoder: A_DPT  # 后续 B_UNet / C_DualUNet_DPT
decoder_width: 128
train_sources:
  rftrans_62cad: verified_train_manifest_required
  cleargrasp_synthetic: verified_train_manifest_required
  cleargrasp_real: verified_real_train_manifest_required
source_count_per_global_update: [7, 7, 2]
real_known_group_split_target: [0.60, 0.20, 0.20]
real_novel_role: sealed_test
target_model_selection: cleargrasp_real_dev
oracle_select: cleargrasp_synthetic_select
oracle_confirm: cleargrasp_synthetic_confirm
rftrans_cg_rerender_included: false
cad_overlap_status: audit_required
real_depth_gt: opaque_depth
rftrans_depth_encoding: verify_actual_release
rgb_only_predictor: true
global_batch: 16
microbatch_per_gpu: 1
gpu_count_max: 8
grad_accum_at_8gpu: 2
far_blocks_count: 1
far_heads_count: 1
far_rank: 4
far_scale: 1.0
far_block_0based: lock_after_oracle
far_head_0based: lock_after_oracle
beta: lock_after_oracle
sigma_log_depth: 0.1
cost_max: 4.0
query_per_group_max: 128
loss_key_reduction: sum
loss_group_reduction: mean_of_nonempty_groups_per_image
seeds: [17, 29, 43]
test_scale_alignment: none
split_manifest_sha256: fill_after_audit
~~~

另沿用原文优化起点：decoder LR=1e-4，LoRA LR=1e-4，AdamW；decoder weight decay=0.01，LoRA/bias/norm 为 0；500 更新 warm-up＋cosine，clip=1。短跑也必须预定日程；相关对照一致调整。BF16/FP16 由实际验收决定，不预填实测显存/吞吐。

记录各来源总曝光、每真实图曝光分布与重复组次数；教师分 R/S/Q 记录覆盖、质量、KL 和跳过比例。参数拆分冻结骨干、读出、LoRA，不能只报 10,752 个参数。

### 6.2 最少结果文件

| 文件/表 | 内容 |
|---|---|
| data_manifest、split_audit | 来源、单位、泄漏组、职责、hash |
| data_QA、label_QA | 叠图、异常和转换证据 |
| implementation_checks | 身份、RoPE、梯度、冻结、合并 |
| teacher_stats_by_source | 分来源覆盖率和教师强度 |
| oracle_select/confirm | 候选、固定基线、分组收益、区间 |
| paired_dev_metrics | 每种子 H1/H2、loss、分域指标和曝光 |
| source_ablation | 三数据协议同预算差分 |
| decoder_ablation | 三架构、C 控制、参数和成本 |
| sealed_test_metrics | 新 known 留出、原 novel 组、Syn-novel |
| export_check、resource_profile | RGB-only 等价、显存、时延、失败率 |

原文小张量核验可作为已有代数证据；不能将“33 项通过”延伸为本版数据、真实划分、官方模型集成或性能已通过。

### 6.3 可直接替换原计划的数据摘要

> 本研究采用 DINOv3 ViT-H/16+ 作为视觉骨干，在冻结原参数的基础上，通过单层少量 Q/K LoRA 学习经深度任务收益验证的注意力关系。训练使用 RFTrans-62CAD 合成数据、ClearGrasp 原始合成数据和按组划出的 ClearGrasp 真实训练子集，各方法共享相同数据职责、监督权限与训练预算。模型部署仅输入单张透明场景 RGB，输出米制前表面深度，并在独立 ClearGrasp 真实留出数据上评测。实验保留 DPT、U-Net、双分支 U-Net＋DPT 三类读出，通过普通 LoRA、数据来源和辅助分割对照检验 FAR 的增量有效性。

结果尚未产生时，写“研究/检验/评估”，不写“显著提升/最优”。加入真实训练后，方向是**有限真实监督下的 RGB-only 留出泛化与关系适配增益**，不能继续称零样本真实迁移。

### 6.4 原 §17 扩展模块保留，暂不进入最小主线

跨材质教师仍为

$$
\overline A^o=\frac1J\sum_{j=1}^JA^{o,j},\quad
s_i=\operatorname{clip}(1-\mathrm{JS}_i/\log J,0,1),\quad
P_i^{opaque}=(1-\beta t_i s_i)A_i^0+\beta t_i s_i\overline A_i^o.
\tag{X1}
$$

J≥2；J=1 只能约定 s=1。t 是可靠透明 query 指示。

最小顺序：仅所属训练组配对→核验同几何同相机→固定读出 oracle→同容量学生→与 GT 教师分表比较。喷涂配对存在替换误差，不自动满足严格几何一致；真实 Test 配对仍封存。

背景鲁棒性扩展保留

$$
\mathrm{Drift}=\operatorname{mean}_{p\in\Omega_T}
\sqrt{\frac1V\sum_v(\widehat Z_p^{(v)}-\overline Z_p)^2}.
\tag{X2}
$$

最小顺序：严格同几何变背景的小批受控图→同时报每视图 MAE 和 Drift→单独改变几何检验敏感性。缺严格配对时不开展；低 Drift 的常数预测不算成功。若另有无物体背景深度，可按原 X5 增加背景偏向指标；原透明传感器深度不能自动代替它。

## 7. 最终决策表

| 实际结果 | 结论 | 下一步 |
|---|---|---|
| 标签/算子验收失败 | 实现或协议尚未成立 | 修 M1–M8 |
| 公共基线弱 | 无法可靠判断关系价值 | 先训练可用读出 |
| oracle 无支持 | 当前范围教师价值未建立 | 保留普通 LoRA |
| oracle 有益、KL 不降 | 低秩参数化或训练未学会 | 查梯度与容量 |
| KL 降、H2 不优于 H1 | 关系拟合不等于深度改善 | 不称 FAR 有任务增益 |
| Mix 优于 CG、H2≈H1 | 数据/训练协议有益，FAR 未建立 | 分开数据与方法贡献 |
| 含真实训练优于纯合成 | 真实监督有益 | 不称零样本迁移能力 |
| H2 在独立真实组优于 H1 | 该协议下有 FAR 增量证据 | 完成机制和跨架构复核 |
| 仅 A 成立 | 证据受解码器条件限制 | 主推 A，保留负结果 |
| C 被 A+Seg 追平 | 一般辅助监督可解释收益 | 收窄双分支主张 |

**第一批工作集中在三件事：三来源标签与真实划分验收；H＋DPT 公共基线；同数据 Mix-H1/Mix-H2 严格验证。之后按门槛展开数据控制、三架构和机制实验。**
