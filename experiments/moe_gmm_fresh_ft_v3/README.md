# MoE × GMM fresh-ft-v3：原始 AgriFM 预训练初始化

本目录保存 2026-10-21 的独立实验版本：原始 AgriFM 编码器 + 全新 NativeFPN 分类头 + 全新时序 MoE。只做 seed42 的冻结编码器与全量微调编码器两组；不从旧 B1 第9轮或其他下游模型续训。

## 入口

- [模型架构与梯度路径](docs/ARCHITECTURE.md)：模块、张量形状、融合公式、损失和 Mermaid 架构图。
- [训练与评估协议](docs/TRAINING_AND_EVALUATION.md)：A/M/C 预热、早停、GTdev top3 与 CDL 主次结果规则。
- [复现、证据和限制](docs/REPRODUCIBILITY.md)：源码身份、依赖、n16r4 执行条件与已完成验证。
- [执行源码](code/README.md)、[冻结组配置](code/configs/joint_frozen_seed42.yaml)、[全量微调组配置](code/configs/joint_fullft_seed42.yaml)。

## 两组设置

|组别|初始化|编码器更新|分类头与 MoE|seed|
|---|---|---|---|---:|
|joint_frozen|原始 AgriFM 编码器 + 新任务模块|全部冻结，eval 模式|按 A/M/C 重新训练|42|
|joint_fullft|同一原始编码器 + 相同新任务模块|patch embedding、四个 Swin 层级及末端 norm 全量更新|按 A/M/C 重新训练|42|

两组初始整模型、分类头和 MoE 的哈希一致；全量微调组在 M/C 阶段允许可靠性损失经 F1 上下文更新编码器。未启用 FiLM、特征残差或 phase 监督。

## 发布时的状态

12 项 CPU 合同测试和两组真实瓦片 A/M/C GPU 验收已通过。正式任务已于 2026-10-21 启动；本次发布只读复核时两组仍在运行，均进入 M 阶段。训练完成后的六权重冻结、CDL 测试和汇总尚未完成，仓库没有本版最终精度结论。

此前审查包是在正式任务启动前导出的，因此包内“尚未提交正式训练”的字段是历史状态。原始证据保留在 `evidence/`，启动状态另存于 `training_launch_snapshot.*`；不得把 GPU 验收结果当作已收敛或分类精度提升。

## 本地核验

在本目录运行：

```bash
python verify_snapshot.py
cd code
python -B -m unittest -v test_v2
```

`code/` 是保持原字节的 n16r4 执行快照；`source_manifest.json` 锁定其中29份源码和配置。架构与发布文档在该快照之外。此版本的启动脚本和配置使用 n16r4 的 Linux 执行路径，不是通用的任意目录启动器；迁移步骤见复现说明。GitHub 上传不触发 GPU 训练。

仓库只含源码、配置、文档和小型验证记录。原始影像、伪标签、AgriFM.pth、训练检查点及完整预测不在 Git 中。
