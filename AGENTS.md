# Agent Working Agreement

## 必读顺序

开始任务前依次读取：

1. `README.md`
2. `docs/STATUS.md`
3. `docs/TASKS.md`
4. 与任务有关的 `docs/DECISIONS.md`、`docs/DATA_CONTRACT.md`、`docs/EXPERIMENTS.md`
5. `docs/handoffs/` 中最近一份交接（存在时）

涉及正式数据工作区时，还必须读取：

```text
E:\maize_paper_workspace\README.md
E:\maize_paper_workspace\00_docs\inventories\AGENT_FIXED_RULES.md
E:\maize_paper_workspace\00_docs\inventories\agent_project_handoff_2026-06-11.md
```

读取后先简要复述项目目标、当前状态、工作区修改和下一步，再开始实施。

## 非协商约束

- 正式数据工作区是 `E:\maize_paper_workspace`。不得擅自删除、移动或覆盖其中的主线资产。
- 本仓库不提交原始数据、处理数据、伪标签、模型权重、完整预测或本机绝对路径配置。
- `D:\cj_swcc\_external\Exact` 和 `D:\cj_swcc\_external\TimeMIL` 是已修改参考树。不得 reset、clean、覆盖或直接作为新模型开发目录。
- 使用用户自己的 Sentinel-2 数据和像元伪标签；第一阶段执行直接监督语义分割，不使用 Exact 的 CAM、prototype、TAAP 或弱监督专用损失。
- E0 与 E1 除 WPE 外必须保持数据、划分、伪标签、损失、解码器、增强、优化器、训练计划和随机种子一致。
- `pseudo_confidence` 只用于分割监督的 mask/权重，不得作为 WPE 的逐时相观测质量。
- 模型、阈值和 checkpoint 只能根据空间独立验证 AOI 选择；测试 AOI 只用于最终报告。
- 用户处于设计探索阶段时，修改代码或生成项目文件前先说明范围并获得确认。
- 不自动重跑已完成的 Xinjiang PEACE-Net EXP001-EXP004，也不自动重启已终止的 USA EXP002 全图推理。

## 文件更新规则

- 代码行为或实验定义变化时更新 `docs/DECISIONS.md`。
- 每次正式实验登记到 `docs/EXPERIMENTS.md`，记录配置、commit、seed、数据划分、输出路径、指标和结论。
- 结束一个阶段或切换长对话前更新 `docs/STATUS.md`，并在 `docs/handoffs/` 新建日期化交接。
- 对正式工作区生成、移动、删除或修改结果后，按其固定规则更新库存和交接文档。
