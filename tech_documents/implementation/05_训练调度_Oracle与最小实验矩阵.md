# 05｜训练调度、Oracle 与最小实验矩阵

版本：2026-09-15。状态：**待实现技术规格；不代表已有训练代码、模型结果或 GPU 验收。**

本篇依据修订计划 M6–M12、§4–§6，以及理论 PDF §5、§7、§12–§13。数据职责、含真实训练的配比和初轮预算以修订计划为准；理论 PDF 中旧的“真实数据全部封存”及旧 epoch 换算不再适用。文中 `src/transdepth/`、`configs/`、Python API 和 CLI 均为计划创建的工程接口。

## 1. 本模块要交付什么

训练模块不是一个包含所有逻辑的 `train.py`。需要分开实现采样、损失、优化更新、状态恢复、oracle 选择与配对检查，才能回答“FAR 是否超过同容量普通 LoRA”。

| 待实现文件 | 职责 | 主要输入／输出 |
|---|---|---|
| `src/transdepth/losses/reduction.py` | 单图非空类别均值 | values、类别有效域 → scalar、计数 |
| `src/transdepth/losses/depth.py` | 米制正输出与 log-Huber | prediction、GT、valid、mask → 每图 loss |
| `src/transdepth/far/relation_loss.py` | 全 key、可抽 query 的 KL | 教师 P、学生 logits、T/B → 每图 loss |
| `src/transdepth/losses/segmentation.py` | 类别平衡 BCE | segmentation logits、可信 mask → 每图 loss |
| `src/transdepth/data/sampling.py` | 全局 16 样本表及来源队列 | manifests、seed、update → rank 样本 |
| `src/transdepth/engine/trainer.py` | microstep、累积、同步、更新 | 配置、模型、样本表 → checkpoint、日志 |
| `src/transdepth/engine/optim.py` | 参数组、AdamW、学习率日程 | trainable 参数清单 → optimizer/scheduler |
| `src/transdepth/engine/checkpoint.py` | T0 分叉、T1 恢复、原子保存 | 状态 → 可完整恢复的快照 |
| `src/transdepth/engine/pairing.py` | H1/H2 公共因素一致性检查 | 两份 run manifest → 审计报告 |
| `src/transdepth/oracle/candidates.py` | 预先生成候选与 shortlist | 锁定随机种子 → 候选 JSON |
| `src/transdepth/oracle/runner.py` | 固定 T0 的身份和 PV 干预 | T0、Select/Confirm → 逐图误差 |
| `src/transdepth/oracle/statistics.py` | 配对组 bootstrap 与约束 | 逐组差值 → 下界与判定 |
| `src/transdepth/cli/train.py`、`oracle.py` | 参数解析及 gate 检查 | YAML／锁文件 → 调用上述模块 |

公共数据对象使用数据篇的 `Sample`、`TrainBatch`、manifest 字段；前向对象使用模型篇的 `Prediction(depth_m, seg_logits, trace)`。`AttentionTrace` 保存 selected heads 的 `q_student/k_student/q_base/k_base`，形状 `[B,k,N,d]`。训练循环不能自己重新解释 RFTrans PNG 或 ClearGrasp GT 文件名。

## 2. 先固定训练目标，再写 reduction

### 2.1 单图有效域

定义 `V_depth` 为数据模块确认的有限、正、范围合法、非 padding 深度域；`V_mask_known` 为已知透明／背景类别域。

- 主实现深度平衡域：`Omega_T = V_depth & (mask == transparent)`，`Omega_B = V_depth & (mask == background)`。
- mask 为 ignore 的像素不自动变成背景。`V_depth & ~V_mask_known` 仍保留诊断计数；最小主实现不引入第三类平衡损失。
- 关系可靠域：`V_rel = V_depth & V_mask_known`，按 M2 生成 T、B、可靠 keys；不是直接拿像素透明 mask 当 attention mask。
- 分割有效域：非 padding 的可信类别标签；无需假设每个可信 mask 像素都有有效深度。数据 QA 判为配准不可信的位置应置 ignore。
- 两类深度域都为空的图像，在训练 manifest 生成时排除并记录原因。运行中若发现与 manifest 断言不一致，整个作业报数据失败，不能仅某 rank 随机补一张图。

这是对原方案未唯一规定的空深度样本策略的**工程补充**。空关系图可继续训练，与空深度图不可混淆。若以后保留类别未知像素的深度项，要预注册另一个变体并让所有配对一致使用。

### 2.2 深度损失必须保留米制尺度

输出映射和损失严格采用 M12：

\[
\widehat Z=1\mathrm m\,[\operatorname{softplus}(u)+10^{-6}],\quad
e=\log\widehat Z-\log Z^*,\quad
\rho_{0.1}(e)=
\begin{cases}e^2/2,&|e|\le0.1\\0.1(|e|-0.05),&|e|>0.1.\end{cases}
\]

每张图先在透明、背景的**非空**集合内分别平均，再对存在的集合等权平均。只有一类时除以 1；两类都有时除以 2。不能在整个 batch 把全部透明像素池化后再算 loss。

实现时先索引合法 GT 再取 log，避免 `log(0)` 后乘 0 仍产生 NaN。正输出映射、log、Huber 和标量累加使用 FP32。`torch.nn.functional.huber_loss(..., delta=0.1, reduction="none")` 的定义匹配该式；不要直接换成不同缩放的 SmoothL1。PyTorch 的 Huber 与 SmoothL1 相差 delta 因子，使用时应检查固定版本文档。[PyTorch HuberLoss](https://docs.pytorch.org/docs/2.7/generated/torch.nn.HuberLoss.html)

必要数值案例：预测=GT 时为 0；预测=2GT 时约为 `0.064314718`；预测=GT/2 时相同。禁止每图 median scaling、减 log 残差均值、scale/shift 或 test-time affine 对齐。

### 2.3 FAR 关系损失

对选中 head 和每个抽中 query：

\[
\ell_{h,i}=\sum_{j=1}^{N}P_{h,ij}
\left(\log P_{h,ij}-\log A^s_{h,ij}\right),\quad
\mathcal L^{I}_{rel}=\frac1k\sum_h\operatorname{AvgBal}_{i\in T,B}\ell_{h,i}.
\]

处理顺序必须是：

1. 从同一增强 RGB 的冻结 prefix 得到 X；使用原 Q/K 与官方 RoPE 构造原关系，生成 FP32 且 detach 的 P。
2. 学生前向先在 RoPE 前加 LoRA；做完整官方 RoPE 后才选 query 行。
3. 每类均匀不放回抽最多 128 个 query；T/B 各自抽样，同一图的所有 selected heads 共享 query 索引。
4. 学生 selected-query logits 保留全部 `N=773` keys，包括前缀和不可靠 patch；沿 key 轴 `log_softmax`。
5. KL 使用逐元素结果后 `sum(dim=-1)`；随后按非空 T/B 取均值、按 head 均值。不能直接使用默认 `mean` 或假定任意形状的 `batchmean` 等于本目标。
6. P 的零概率按 `0 log 0 = 0` 处理，例如使用经过数值验证的逐元素 KL；不要先在零值上生成 `-inf` 再相乘。
7. 整图没有可靠 query 时 `L_rel=0`，该图仍保留有效深度项，并继续在 16 图平均的分母中占一份。

背景 query 的目标为 A0，不能漏掉其保持项；没有可靠透明 query、但有可靠背景 query 的图像仍可有合法背景 KL。教师回退率和“整图无关系”率分别记录。[PyTorch kl_div 的 input 与 reduction 定义](https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.kl_div.html)

教师只由当前训练样本标签构造，不能用 `student_attention.detach()` 冒充 A0。也不能将教师 KL 的 `A_student - P` 梯度当作深度项的梯度；深度需要真实通过 decoder、suffix、AV、softmax、RoPE 回传。

### 2.4 分割与总损失

C 的分割使用 `binary_cross_entropy_with_logits(..., reduction="none")`，可信透明／背景各自均值后对非空类别平均；不用 Dice、边界损失、法线或中间深度监督。A+Seg/B+Seg 控制采用相同规则。

\[
L_I=L_{depth,I}+\lambda_{rel}L_{rel,I}
+\mathbf1_{Seg}\lambda_{seg}L_{seg,I},\qquad
L_{update}=\frac1{16}\sum_{I=1}^{16}L_I.
\]

缺分割有效像素的图像令该项为可反传的零；仍占 1/16。若 C-noSeg 完全不使用 SegTail，建 optimizer/DDP 前冻结或省略 SegTail 参数；语义特征分支参与深度融合的参数继续训练。不能留下永远 unused 的可训练参数再用假梯度掩盖结构问题。

### 2.5 损失验收表

| 检查 | 输入 | 要验证的性质 |
|---|---|---|
| 区域平衡 | 透明 1 像素、背景 100 像素，各有固定误差 | 两类各占 1/2，不按像素数定权 |
| 空类 | 只有透明／只有背景 | 剩余类权重为 1 |
| 空关系 | GT 深度有效、无可靠 patch | 深度梯度仍存在，关系为有限的 0 |
| KL 手算 | 小概率分布及极小概率 | KL 方向、key 求和、A-P 梯度正确 |
| padding | 两张内容相同但 padding 不同的标签 | padding 不贡献 depth/rel/seg 监督 |
| 单项梯度 | depth-only、relation-only、seg-only | 梯度分别到允许参数；教师无梯度 |
| 原参数冻结 | 一次完整 optimizer update 前后 | 原骨干 checksum 不变，LoRA/读出允许变化 |
| DDP 等价 | 相同 16 个合成小样本，单进程与 2/4/8 rank | 全局均值梯度与更新在预定容差内一致 |

这些测试应针对可手算的独立构造，不仅复制实现中的 reduction 语句作为“参考答案”。

## 3. 三来源的全局采样表

### 3.1 每个 optimizer update 精确为 7R、7S、2Q

R=RFTrans-62CAD、S=ClearGrasp 原始合成、Q=ClearGrasp 真实 Train。主协议在一次完整更新中固定 16 个槽位；不能让每个 GPU 自己按概率随机采样后仅“期望上”为 7/7/2。

设计 `GlobalUpdatePlan`：

```text
protocol_id, seed, phase, update_index
slots[16]: {slot_index, source, sample_id, draw_index, augment_seed, query_seed}
source_queue_cycle, source_queue_cursor, plan_sha256
```

生成流程：

1. 每域维护从其训练 manifest 确定的随机排列队列；队列用尽再按独立种子重排，记录 cycle。Q 小集合循环时必须记录重复曝光。
2. 每 update 从 R/S/Q 队列依次取 7/7/2 项；洗牌来源槽位，防止真实图固定由某张卡处理。
3. 全部 rank 使用同一张表及 hash；第 `micro_index * world_size + rank` 槽交给对应 rank（microbatch=1）。
4. 8 卡执行 2 微步；4 卡执行 4 微步；1 卡排错可执行 16 微步。表内样本、增强键和成功更新数相同。
5. `world_size * microbatch * accum == 16` 必须启动时校验。主实验不接受不能整除的卡数配置。
6. 每图等权后已经实现配比；不能再给 R/S/Q 损失乘 7/16、7/16、2/16。

主文中的 `sum_d pi_d * mean_d(loss)` 只适用于**先算三个域独立均值**的另一种实现。项目选用 16 图均值，避免两种路径混用。

### 3.2 曝光与跨协议共享

成功更新 T 次后的每域平均曝光为 `Exposure_d = T * count_d / n_train_d`。同时保存每个 sample_id 和 leakage_group_id 的次数分布、最大值、分位数；不能只给一个 epoch 数。

- Mix-real：7/7/2；CG-real：0/14/2；Mix-syn：8/8/0。
- CG-real 与 Mix-real 的 Q 队列和增强键独立于 S/R 队列，能够逐 update 保持同一真实样本曝光表。
- 各协议的 T0 和 T1 都遵守自己的数据权限。CG-real 不载入用 R 预热过的 T0；Mix-syn 不载入用 Q 预热过的 T0。
- Mix-syn 允许使用 Q-dev 选择超参数时，结果名称必须注明“使用真实验证的纯合成训练”。
- H1/H2 使用相同 plan 文件或可重建的等价计划 hash。切勿只比较 seed 数字相同。

### 3.3 随机流独立

建议将随机键派生为 `hash(seed, phase, update, slot, sample_id, purpose)`；purpose 至少区分 `source_order`、`augment`、`lora_init`、`relation_query`、`bootstrap`。这是一项工程约定。

FAR 多一次 query 抽样不能改变后续 RGB 增强；验证、日志抽样不能消耗训练随机流。DataLoader 使用无状态的样本增强键，重启时从已提交 update 的下一个槽位恢复；预取队列里尚未提交的样本全部作废。仅保存 `torch.manual_seed` 无法解决 worker 预取导致的恢复偏移。

## 4. 正确的 DDP 和累积

### 4.1 全局梯度的归一化

令 rank r 在两个微步的每图损失为 `L[r,0]`、`L[r,1]`。每微步反传 `L/2`；标准 DDP 对 8 个 rank 梯度求平均：

\[
\frac1{8}\sum_{r=0}^7\sum_{a=0}^1\nabla(L_{r,a}/2)
=\nabla\left(\frac1{16}\sum_{r,a}L_{r,a}\right).
\]

因此不再除 world_size，不再除 16，不对某域另除该域图数。4 卡累积 4 同理。不同区域像素数、不同 query 数已经在每图内归一，不应跨 GPU 再按有效像素数重加权。

PyTorch 标准 DDP 的梯度会在 rank 间平均，且 `no_sync` 必须同时包住 forward/backward。项目首轮不注册改变归一化的自定义通信 hook。[PyTorch 2.7 DDP](https://docs.pytorch.org/docs/2.7/generated/torch.nn.parallel.DistributedDataParallel.html)

### 4.2 一个完整更新的伪代码

以下是**结构规格**，`compute_losses`、`finite_check_all_ranks` 等待实现，不是可直接复制运行的代码。

```python
optimizer.zero_grad(set_to_none=True)
for micro_index in range(accum):
    batch = loader.for_slot(update, micro_index, rank)
    sync_context = ddp.no_sync() if micro_index < accum - 1 else nullcontext()
    with sync_context:                         # forward 也在上下文中
        with torch.autocast("cuda", dtype=amp_dtype):
            output = ddp(batch.rgb, return_aux=True)  # 04章的Prediction；标签不进模型
        with torch.autocast("cuda", enabled=False):
            loss_i = compute_losses(output, batch).total.float()
            backward_loss = loss_i / accum
        scaler.scale(backward_loss).backward()  # BF16 模式 scaler 可 disabled

scaler.unscale_(optimizer)                     # 仅完整累积结束后一次
finite_check_all_ranks(model, loss_log)         # 任一 rank 失败则全体中止本次提交
clip_grad_norm_(trainable_parameters, 1.0)
scaler.step(optimizer)
scaler.update()
scheduler.step()                              # 仅成功 optimizer update 后
commit_update_and_sampler_cursor()
```

训练 forward 返回 04 章的 `Prediction`，其 trace 从 RGB-only backbone 路径提取所选头 Q/K；标签只在模型外构造 teacher/loss，不进入 decoder 或 Q/K 前向作为条件。H1 不执行教师构造和关系 loss，H2 执行；训练需要的 seg_logits 也通过同一辅助输出接口获得。部署包装为 `forward(rgb) -> depth_m`。

FP16 的 scale 在整个累积周期保持不变；不能中间 unscale、clip 或 update scaler。BF16/FP16 的最终选择必须经服务器验收。[PyTorch 2.7 AMP 累积示例](https://docs.pytorch.org/docs/2.7/notes/amp_examples.html)

### 4.3 空关系、分割缺失与同步

- 无关系图不跳过 forward/backward；通过深度项保持 LoRA 与读出的真实梯度路径。`L_rel` 可返回连接输出的零标量和 `valid_query_count=0`。
- 若该架构启用了 SegTail，但当前图没有可信分割像素，计算 SegTail，并以 `seg_logits.sum() * 0` 保持合法零梯度；全训练协议永远没有分割监督时应冻结 SegTail。
- 所有 rank 每轮执行相同次数的微步和 collective。不能某张卡见到空区域就 `continue`，导致其他卡等待。
- `find_unused_parameters=False` 的前提是所有声明可训练参数在每轮都接入 loss。先通过空关系／缺分割的混合 rank 用例；初期排错可开启 unused 检查，但不能把它当冻结路径正确的证明。
- rank-local 原始日志通过 detached 数值汇总，不将用于反向的 loss tensor 先 `all_reduce` 再反传。

### 4.4 非有限数和失败更新

首轮建议 BF16 数值验收通过后使用 BF16；关系与损失始终 FP32。任一 rank 出现非有限 loss、梯度或输入，写失败上下文并协调终止，不默默丢该 batch。

FP16 若发生 overflow，需要全 rank 一致记录 `attempted_update`、`successful_update`、scaler 状态及跳步。为保证 H1/H2 的成功更新与样本计划配对，正式实验不允许跳过失败表后继续；应从完整 checkpoint 恢复同一下一张全局表，并在记录过的数值修复配置下重新开始该运行或整对运行。失败前向的计算次数单独计入资源，不算成功训练曝光。不能仅给 H2 降学习率并继续称为严格配对。

### 4.5 冻结与 activation checkpoint

| 路径 | 参数可训练 | autograd | 模式 |
|---|---|---|---|
| 选中 block 前 prefix | 否 | 可 `no_grad` | backbone/RoPE eval |
| LoRA 所在 block | 仅 Q/K LoRA | 必须保留 | backbone eval 不等于 no_grad |
| 后续 suffix | 否 | 必须保留输入梯度 | eval |
| reassembly/decoder/C branches | 是 | 必须保留 | train；用 GroupNorm |
| 原 A0 与教师 P | 否 | `no_grad`、detach | 原权重、同增强视图 |

必要时按 suffix block 使用 `checkpoint(..., use_reentrant=False)`，函数应无写缓存、抽样或记录重复计数等副作用。query 索引在外部生成；重算不能覆盖原教师缓存。打开 checkpoint 前后比较相同固定输入的 loss 与可训练参数梯度。[PyTorch 2.7 checkpoint](https://docs.pytorch.org/docs/2.7/checkpoint.html)

0 组全骨干可 `no_grad`；1/2 组不能复用冻结基线缓存的 suffix 特征。任何特征缓存必须绑定 RGB hash、增强、预处理、骨干 hash 和层位。

## 5. 优化器、预算与 checkpoint

### 5.1 哪些是原计划起点，哪些是本设计补充

| 配置 | 起点／候选 | 来源与处理 |
|---|---|---|
| optimizer | AdamW | 原计划 |
| decoder LR | 1e-4；候选 5e-5/1e-4/2e-4 | 原计划 §13.2 |
| LoRA LR | 1e-4；候选 1e-4/3e-4 | 原计划 §13.2 |
| weight decay | decoder 常规权重 .01；LoRA/bias/norm 0 | 原计划；按参数所属模块分类，不只按字符串猜测 |
| AdamW betas/eps | `(0.9,0.999)` / `1e-8` | 工程默认建议；修订计划未声称最佳 |
| warm-up/cosine | 500 成功更新 warm-up，之后 cosine | 原计划；cosine 终值建议 0，须写入 resolved config |
| clip | 全部 trainable 参数合并 norm=1 | 原计划；完整累积、unscale 后 |
| T0 初轮 | 2,000 完整更新 | 修订计划；欠拟合且 Dev 改善时重新锁定，原上限起点 20k |
| T1 初轮 | 2,000 完整更新 | 修订计划；统一锁定正式预算，原 10k 为需评估的起点 |
| Dev 间隔 | 初轮 250–500 更新 | 修订计划；正式采用固定间隔并纳入 hash |
| lambda_rel | 0/.01/.1/1 | 修订计划；0=普通 LoRA |
| lambda_seg | 0/.01/.1 | 修订计划；C1/C2 必须相同 |
| 小样本拟合 | 300–1,000 更新，关闭随机增强 | 修订计划；短跑日程须单列，不自动沿用 500 warm-up |
| seeds | 17、29、43 | 修订计划；17 discovery，最终三种子配对 |

例如 300-step overfit 若仍 500-step warm-up，将全程停留预热，无法检查正常学习阶段。可将该**排错配置** warm-up 设为 20 更新，明确其不进入主实验；正式 2,000-step 短跑继续使用锁定的主日程。

每次创建 optimizer 时输出参数分组表：名称、shape、参数数、LR、WD、requires_grad。H 的单头 r=4 LoRA 应为 10,752 参数，B 为 6,656；还必须另报 reassembly/decoder 参数和冻结骨干参数。

### 5.2 T0 到 T1 是分叉，不是普通 resume

每个 `(data_protocol, backbone, decoder, seed)` 独立训练 T0。其读出质量经 Dev 检查后，锁定一个 `T0_ref`：建议正式流程使用同预算终点 `last.pt`，若用 Dev-best 则必须在 oracle 前明确规则及具体 hash。

从 `T0_ref` 建立 0/1/2：

1. 三组读出权重逐元素相同；原骨干权重 hash 相同。
2. 按修订计划，**三组 decoder optimizer 状态统一重置，学习率日程统一重启**。
3. 1/2 LoRA 使用同一初始化文件：U 小随机、B=0。0 组不训练 LoRA。
4. 建 T1 公共数据表及增强键，训练相同 T1 完整更新。H0 最终结果不是未经 T1 继续训练的 T0。
5. T0→T1 的 sampler 策略固定为新 phase 队列；三组一致，T0 及 T1 曝光分别记录。这是本设计的明确实现选择。
6. `fork_manifest.json` 记录共同父 checkpoint、LoRA 初始 hash、optimizer reset=true、scheduler restart=true、计划 hash。

seed 29、43 各自训练自己的 T0，内部再分叉；使用 seed 17 已锁定的层/head/beta/损失与预算，不再开启 Confirm。

### 5.3 T1 中断恢复必须完整恢复

同一运行 resume 恰好相反：恢复模型、optimizer moments、scheduler、scaler、成功 update、源队列状态、全部 RNG 与下一张样本计划，不再重置优化器。使用互斥 CLI 参数防止误操作：`--fork-from` 与 `--resume` 不能同时出现。

checkpoint 至少包含：

```text
schema_version, run_id, protocol_id, phase, seed
model_state (trainable weights), frozen_backbone_ref/hash, model_config/hash
optimizer_state, optimizer_param_group_order, scheduler_state, scaler_state
successful_updates, attempted_updates, next_global_update_index
python_rng, numpy_rng, torch_cpu_rng, torch_cuda_rng_by_rank
sampler_state, sample_plan_hash, augmentation_version, rng_derivation_version
best_metric_state, checkpoint_selection_rule, metrics_protocol_hash
data_manifest_hashes, split_lock_hash, oracle_lock_hash, config_resolved_hash
git_commit, dirty_diff_hash, requirements_lock_hash, environment_fingerprint
```

只在完整成功更新边界保存；先写临时文件、完成 flush/关闭，再同目录原子 rename，最后写 checkpoint hash。模型／optimizer 可由 rank 0 写，rank-specific RNG 先汇集或分别落 sidecar 并在总 manifest 中登记。恢复时检查文件齐全和所有 rank 状态版本一致。

验收：同一固定小任务连续训练 N 更新，与 K 更新保存后恢复到 N 的运行比较样本序列、LR、损失、权重、optimizer、scaler 状态。只有模型输出相近不能证明完整恢复。相同软件／硬件／world size 内给出容差；4 卡与 8 卡数学等价但不承诺浮点逐 bit 相同。

### 5.4 Dev-best 和等步数 last

建议锁定如下选择规则：在 Q-dev 上透明组宏 MAE 最小，且相对该运行所属 T0 的背景／边界退化均不超过预先固定容忍量；完全同分时取较早 step。S-dev 单独报告和应用预注册退化约束；R-dev 只作源域诊断，不与 Q-dev 混成平均。

容忍量采用计划起点 `max(1 mm, 0.02 * corresponding_region_baseline_MAE_mm)`，每域、每区域分别从相应 Dev 固定；不能随每次候选 checkpoint 重算。若没有任何合格 checkpoint，best 状态为 `no_eligible_checkpoint`，保留 last 及失败原因，不能自动改约束。

best 与 last 都保存、都报告，主结果到底用哪一个在终评锁中固定。同一 best 规则不代表 H1/H2 的 best 步数相同，因此 last 必须是同一成功更新 T1 的终点。Dev checkpoint 评估只使用 RGB 推理入口，标签由 evaluator 读取。

## 6. M6 Oracle 的实现与独立确认

### 6.1 进入条件

必须先有：M1/M2 数据验收、M3/M4 小样本拟合、可用 T0、官方 SDPA 身份检查、稳定资源配置。固定 T0 的原参数、读出及预处理，所有候选只改变被选 attention 的聚合。

身份测试 `P=A0`：对照官方 SDPA 与显式 `A0 @ V0` 的 selected head 输出、当前 block 输出、最终深度。其次真实干预用 `P @ V0`，拼回所有 heads 后通过原投影、残差和全部 suffix；不允许仅改保存的热图。

### 6.2 候选空间与预算

| 项目 | H/16+ | B/16 |
|---|---|---|
| blocks，0-based | 15、23、31 | 5、8、11 |
| 每层候选 heads | 从 20 头中预抽 8 个不同 heads | 从 12 头中预抽 8 个不同 heads |
| beta | 0 为身份；正值 .25、.5 | 相同 |
| 正干预候选数 | 3 × 8 × 2 = 48 | 48 |
| 初筛 | CG Select 约 100–200 图，按完整组抽选 | 相同预算原则 |
| 复筛 | 前 3 候选在完整 Select | 相同 |
| 确认 | 锁定 1 个候选后，只运行该候选与固定基线 | 独立预注册确认安排 |

候选 JSON 在看任何 Select 结果前生成并保存 seed/hash；不根据已有热图挑头。可按候选分配 8 张 GPU，每卡独立推理，没有 DDP 反向需求。所有候选仍需完整重算变动的 suffix；缓存只允许固定 prefix/原始基线，缓存 key 绑定图像与模型版本。

### 6.3 选择排序与约束

对区域 r、组 g，先逐图计算 MAE，再组内等图平均；透明组差值 `Delta_g = e_g(f0) - e_g(foracle)`。组内无有效透明图时该组透明指标为缺失，保留原因和其他区域指标。

Select 阶段先过滤背景和边界平均退化超过容忍量的候选，再按透明平均收益降序。若同分，建议按较小 beta、较深 block、较小 head 编号确定稳定 tie-break；这是工程补充，须写入锁配置，不能看结果临时决定。

oracle 使用 CG 合成 Select/Confirm，故约束阈值来自相应 **S-dev** 固定基线；真实学生的容忍量来自 **Q-dev**。不能用 Q-test、Confirm 本身或其他域的 MAE 制定阈值。

`oracle_choice.lock.json` 必须包含：T0 hash、候选空间 hash、Select manifest/hash、完整筛选表、选中 block/head/beta、可靠性阈值、sigma/cmax、各约束、指标协议、Confirm 集及统计方法。锁后才能读 Confirm 标签。

Confirm 后写 `confirm_decision.json`；仅确认通过时生成训练入口统一读取的 `oracle_lock.json`，其中引用选择锁与确认结果的 hash。三者分别是“开封前选择”“确认结果”“可供训练消费的汇总锁”，不得相互覆盖或把选择锁当成确认通过。

### 6.4 Confirm 的 2,000 次配对组 bootstrap

对 G 个有效独立组的差值 `d[g]`：固定 bootstrap seed，每次均匀有放回抽 G 个组索引，取这些组的均值；重复 2,000 次。下界为重抽均值的 5% 分位数，即近似单侧 95% 下界。配对意味着基线和干预始终使用同一批抽中组。

通过要求同时成立：

1. 透明组宏平均收益为正；单侧 95% 下界为正。
2. 背景和边界退化不超过 Select 前已锁定的相应阈值。
3. 无实现错误；非有限数／失败组按预先规则处理；独立组数量满足已登记最低证据条件。

源计划未给“最少独立组数”的统一数值。建议配置 `min_confirm_groups` 并在 split 审计后、开封前确定；可以以 20 个独立组作为启动讨论值，但不能把这个数冒充功效保证。若实际只有少量相关组，即使下界为正也要报告证据有限；最低条件未定义时不能自动生成 `confirm_pass=true`。

输出均值、下界、G、每组图数、有效像素数、退化组比例、bootstrap seed、分位数实现（例如 NumPy `method="linear"`）、完整组差值。不能把像素或相邻视角当成独立 bootstrap 单元。

### 6.5 Confirm 使用次数与数据控制

最小执行默认：只对 **Mix-real H+A** 确认一次；CG-real/Mix-syn 固定迁移该层/head/beta，各自训练自己的 T0，不能声称它们分别通过 oracle 确认。B 骨干需要自己的相对层位选头，必须预留确认折或在开始前登记整个确认族，不能将同一 Confirm 无限制复用。

若三数据协议独立选头，必须在任何 Confirm 开封前登记完整确认族，采用 Holm 等预注册多重比较控制或独立确认折。H 的 B/C 解码器首轮迁移 A 的位置；重选头属于额外预算，需要同样的确认安排和重新配对。

确认失败后停止当前教师价值主张；修实现可重跑身份检查，但数据驱动重搜索需要新确认数据或预先设计的折。合成 oracle 阳性仅证明该固定模型、该合成分布上的特权干预；既不是部署性能，也不是学生严格上界。

## 7. 分阶段执行，不把完整论文预算塞进首轮

下表 S0–S9 是研究执行阶段；07 篇的 P0–P9 是工程实施任务，两套编号不混用。

| 阶段 | 任务及启动预算 | 进入下阶段的证据 | 未通过时 |
|---|---|---|---|
| S0 | M1/M2 数据身份、编码、group split | audit 与 QA 已验收 | 修数据 |
| S1 | H+A，每源 8–16 张固定训练图，300–1,000 更新；单卡 profile | 可拟合、梯度和形状正确、资源可承受 | 修模型／loss；不同时扩三架构 |
| S2 | Mix-real seed17 T0，约 2,000 更新 | 可用固定读出及 Q/S-dev 曲线 | 依据 Dev 重新预定公共预算 |
| S3 | M6 48 候选初筛、3 个复筛、1 个 Confirm | 独立确认与约束通过 | 保留普通 LoRA |
| S4 | Mix H0/H1/H2，约 2,000 T1 短跑；关系系数有限筛选 | H2 对 H1 Q-dev 改善，背景／边界受控 | 若 KL 降但深度不升，记录阴性 |
| S5 | 复用已锁 T0 预算，固定正式 T1、位置、系数；17/29/43 配对复核 | 每种子差值和稳定性 | 不用跨架构大表替代主配对 |
| S6 | CG-real H1/H2，再 Mix-syn H1/H2 | 数据差分可解释、预算相同 | 报曝光／真实监督效应 |
| S7 | B/H 六组补全，H+B、H+C 各自 0/1/2 | 各架构内部 FAR 配对与成本 | 保留有效架构和负结果 |
| S8 | C 有深度收益后做结构消融，补 FAR 机制及强基线 | 每项论文归因有控制 | 收窄主张 |
| S9 | RGB 导出、锁定、独立真实终评 | 本套 06 的全部 gate | 不用 Test 调参 |

最少优先做 Mix-H1/H2 三个配对种子。C 不必获胜才认可 A/B 上的 FAR。训练 MAE 降约 50% 是排错参考，不是对任意数据／架构的成功定理。约 2% 透明 MAE 相对改善是预注册实用门槛起点，不是已取得结果。

T0 的预算与 discovery `T0_ref` 必须在 oracle 前固定；不能确认后换一个更长训练的 T0，再把原确认当成该新固定模型的证据。T1 短跑若导致正式总步数或 LR 日程变化，应从同一个 T0 为 H0/H1/H2 重新启动正式配对；原短跑归为 discovery 成本。seed 29/43 使用相同 T0 预算，各自独立初始化训练。

## 8. 完整配置矩阵与差分含义

### 8.1 16 个独立配置

“B/16 骨干”与“B：U-Net 解码器”不同，文件名须同时包含 backbone 与 decoder 字段。

| 数据 | 骨干 | 解码器 | 0：冻结骨干继续读出 | 1：普通 Q/K LoRA | 2：同 LoRA + FAR |
|---|---|---|---|---|---|
| Mix-real | B/16 | A:DPT | B0 | B1 | B2 |
| Mix-real | H/16+ | A:DPT | H0=H-A0 | H1=H-A1 | H2=H-A2 |
| Mix-real | H/16+ | B:U-Net | H-B0 | H-B1 | H-B2 |
| Mix-real | H/16+ | C:Dual U-Net+DPT | H-C0 | H-C1 | H-C2 |
| CG-real | H/16+ | A:DPT | 不列新增主配置 | CG-H1 | CG-H2 |
| Mix-syn | H/16+ | A:DPT | 不列新增主配置 | Syn-H1 | Syn-H2 |

前四行 12 个主配置，后两行新增 4 个，总 16 个；全做三种子为 48 条学生训练结果。**不含**各组 T0、oracle、学习率／lambda 试跑、C 消融、机制控制和外部基线。H1/H2 在不同表复用一次结果，不重复计数。

核心数据差分：

\[
G_{FAR}^{mix}=e_{Mix-H1}-e_{Mix-H2},\quad
G_{62CAD}^{LoRA}=e_{CG-H1}-e_{Mix-H1},\quad
G_{real}^{LoRA}=e_{Syn-H1}-e_{Mix-H1},
\]
\[
I_{data\times FAR}=(e_{Mix-H1}-e_{Mix-H2})-(e_{CG-H1}-e_{CG-H2}).
\]

所有 e 使用同一预先确定的域和指标。第二／三项是在固定总预算下**替换一部分曝光**的效果；不是保持其他数据曝光完全不变的“追加数据收益”。若要后者，另做保持 CG 曝光不变的预算敏感性实验并报告增加的 GPU-hours。

### 8.2 FAR 与 C 的控制

| 控制 | 保持／改变 | 首轮优先级 |
|---|---|---|
| 普通 LoRA，lambda_rel=0 | 同位置、同参数、同预算 | 必须 |
| P=A0，lambda_rel>0 | 原关系保持正则 | 高 |
| 条件 q∝exp(-c) | 保持 m、补集、beta，去掉条件原 attention 偏好 | 高 |
| 2D 空间代价 | 同框架，测试局部平滑解释 | 完整机制主张前补 |
| 置换可靠 keys 内最终 P 的概率 | 保持概率集合、质量及熵，破坏对应关系 | 完整机制主张前补 |
| 同容量随机／固定末层头 | 检验选头价值 | 完整定位主张前补 |
| C-noSeg | 相同结构，lambda_seg=0 | C 有深度收益后 |
| A+Seg、B+Seg | 相同 mask、lambda_seg、匹配 SegTail | C 辅助监督归因必需 |
| C-noSemFuse | 保留分割，去掉 S→深度直接融合 | 双分支融合归因必需 |
| C-noDPT | 融合 /4 特征直达相同 Tail | DPT 归因必需 |
| 参数匹配 C | 含 reassembly 的读出参数约 ±10% | 容量差明显时 |

机制对照获得相同 lambda/beta 调优预算；不能只充分调 FAR。置换最终概率与“打乱代价再生成 P”不同，后者不保证最终熵保持。

### 8.3 配对审计输出

`pairing_report.json` 比较父 T0、LoRA 初始化、模型拓扑、冻结权重、data/split hash、全局样本表、增强键、loss、optimizer、LR 日程、successful updates、segmentation 权限、checkpoint 规则。H1/H2 允许差异清单仅包括 `lambda_rel`、关系计算开关、对应额外日志及运行标识。

不得将同一个 seed 下误用另一预处理版本、不同 T0-best、不同 Q 曝光或不同数据过滤的结果放进配对表。每个种子的 H1/H2 差值先列出，再讨论平均；不能将三种子的所有像素拼成伪样本量。

## 9. 训练资源与输出目录

profile 需要实际服务器：每架构及适配位置先 20 完整更新预热，再测至少 100 完整更新。记录各 rank 峰值 allocated/reserved、进程外 GPU 使用、update 中位/P95、加载占比、NaN/OOM、参数及 LoRA 梯度范数。峰值目标约保留 10% 单卡余量；DDP 8×24 GB 不构成单卡 192 GB。

OOM 处理顺序按计划：去意外完整 attention／重复教师 → suffix/必要 decoder checkpoint → 配对统一减 query → 必要时统一宽度／输入并重做相关对照。先测 H，不未经尝试就换 B。参数量、张量理论大小不能填入“实测峰值”栏。

建议产物结构：`run_dir` 是 01 篇本机路径配置中运行根目录下的一个运行，oracle 目录同样在配置的产物根目录内，不能硬编码仓库外未知路径。

```text
run_dir/
  resolved_config.yaml
  run_manifest.json
  parameter_groups.json
  sampling/plan_manifest.json
  sampling/exposure_by_sample.csv
  sampling/exposure_by_group.csv
  logs/train.jsonl
  logs/teacher_stats_by_source.jsonl
  metrics/dev_per_image.csv
  metrics/dev_per_group.csv
  metrics/dev_summary.json
  checkpoints/last.pt
  checkpoints/best.pt                 # 仅有合格候选时存在
  checkpoints/checkpoint_manifest.json
  checks/freeze_and_gradient.json
  checks/pairing_report.json
  checks/resume_equivalence.json
  resource/profile.json
oracle/<selection_id>/
  candidates.json
  select_per_image.csv
  select_per_group.csv
  shortlist.json
  oracle_choice.lock.json
  confirm_per_group.csv
  confirm_bootstrap.json
  confirm_decision.json
  oracle_lock.json
```

GPU-hours 按实际卡数 × 实际作业秒数/3600 统计，分别记录 T0、oracle、调参、正式种子、导出和终评；若以 `T * update_time` 估算须标 estimate。日志中 unknown/TBD 不写 0，以免被误解为零显存或零失败。

## 10. 未来 CLI 与阶段锁

以下命令是远端 Linux 项目完成后的目标接口，当前文档交付不等于它们已可执行。主键路径由环境篇和配置篇统一落地。

```bash
# 单进程数据/小样本配置已通过后，启动公共预热
torchrun --standalone --nproc_per_node=8 -m transdepth.cli.train \
  --config configs/experiments/t0.yaml --paths configs/paths.local.yaml --seed 17

# Oracle 身份 → Select → 锁定候选 → Confirm；每一步检查前一步输出
python -m transdepth.cli.oracle --phase identity --config configs/oracle/h_a.yaml --paths configs/paths.local.yaml
python -m transdepth.cli.oracle --phase select --config configs/oracle/h_a.yaml --paths configs/paths.local.yaml
python -m transdepth.cli.oracle --phase lock --config configs/oracle/h_a.yaml --paths configs/paths.local.yaml
python -m transdepth.cli.oracle --phase confirm --config configs/oracle/h_a.yaml --paths configs/paths.local.yaml

# 两个学生从同一 T0 分叉，不能用 --resume 代替
torchrun --standalone --nproc_per_node=8 -m transdepth.cli.train \
  --config configs/experiments/mix_h1.yaml --paths configs/paths.local.yaml \
  --seed 17 --fork-from <T0_ref>
torchrun --standalone --nproc_per_node=8 -m transdepth.cli.train \
  --config configs/experiments/mix_h2.yaml --paths configs/paths.local.yaml \
  --seed 17 --fork-from <T0_ref>

# 同一运行发生中断，完整恢复
torchrun --standalone --nproc_per_node=8 -m transdepth.cli.train \
  --config <resolved_config> --paths configs/paths.local.yaml --resume <last_complete_checkpoint>
```

启动断言至少检查：manifest/split 已锁、所有路径存在、训练角色白名单、来源配比、骨干身份、loss 版本、oracle 状态、LoRA 初值、T0 父 hash、确认族权限、world size、环境指纹。任何字段缺失返回非零退出码与可定位原因；不要用“先跑再说”绕过数据单位或 Confirm 权限缺口。

## 11. 故障与结论矩阵

| 现象 | 优先检查 | 允许结论／下一步 |
|---|---|---|
| 无法拟合 24–48 张固定图 | GT 单位/对齐、正值映射、有效域、梯度、LR | 修实现，不能否定 FAR |
| H1 的 LoRA 无深度梯度 | suffix no_grad、特征缓存、QKV 切片 | 梯度 gate 失败 |
| DDP 卡住 | rank 分支不同、空关系 continue、unused 参数 | 修同步，不关闭同步冒充成功 |
| 开累积后 LR 表现大变 | 多除 world size／batch、提前 clip/unscale | 修全局均值 |
| 真实 Train 好、Dev 恶化 | Q 曝光、组泄漏、记忆化 | 共同比较更短锁定预算，不用 Test 调参 |
| P=A0 也改变预测很多 | RoPE、head 拼接、SDPA dtype/scale/bias | 不进入 oracle 选择 |
| Confirm 不通过 | 实现已排除后检查统计支持 | 当前范围教师价值未建立 |
| Oracle 有益、KL 不降 | 梯度链、reduction、初始化、低秩容量 | 先诊断；r=8 另立单因素实验 |
| KL 下降、H2 不优于 H1 | 教师与任务不一致／不可预测部分 | 关系可拟合，任务收益未建立 |
| Mix 优于 CG，H2≈H1 | 数据替换曝光与真实监督 | 数据协议有益，不能归给 FAR |
| 只有一个种子有益 | 成对波动、训练差距 | 证据不足，保留全部结果 |
| C 被 A+Seg 追平 | 一般辅助监督解释 | 收窄双分支主张 |
| profile 未跑 | 服务器环境与设备尚未验收 | resource 状态只能是 pending |

本篇完成的证据应是可审查的实现、损失与实验协议；FAR 增量是否成立，只能由未来真实运行的 H1/H2 配对和封存终评证明。
