# TransDepth / FAR

基于冻结 DINOv3 H/16+、单层 Q/K LoRA 和 FAR 关系监督的 RGB-only 米制透明物体首表面深度研究。

当前阶段：技术设计文档。完整训练工程、数据验收、模型下载与远端部署尚未执行。

从 [实现文档总览](tech_documents/implementation/00_总览与模块实施路线.md) 开始阅读。整套文档覆盖项目结构与环境、RFTrans/ClearGrasp数据、DINOv3下载、三架构与FAR、训练实验、RGB推理评价和开发验收。

研究协议以 [2026-09-15修订计划](tech_documents/FAR_62CAD_ClearGrasp含真实训练_局部修订与逐模块最小验证计划.md) 为准；理论依据见 [核心理论独立版](tech_documents/FAR_核心理论与逐步数学推导_独立版.pdf)。
