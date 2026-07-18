# Maize Model Research

本仓库是玉米遥感时序语义分割与模型创新的主论文代码仓库。它保存代码、配置、数据划分清单、实验索引和研究决策，不保存遥感原始数据、处理后数据、伪标签、模型权重或完整推理结果。

## 研究主线

```text
Sentinel-2 多时相数据
  -> 可追溯伪标签
  -> Exact 风格时空 Transformer 直接分割基线（E0）
  -> DOY 感知多尺度小波位置编码模型（E1）
  -> 空间独立 AOI 上的受控对比和消融
```

当前已完成可复现基线包括 Xinjiang 2021 PEACE-Net EXP002、Exact 风格 E0（TSViT+learned DOY）和受控 E1（E0+learnable WPE）。E0/E1 均完成正式 Train/Validation 训练、Validation early stopping 和独立 `best.pt` 重放；首轮 E1 的 Validation maize IoU 未超过 E0，因此当前仍以 E0 作为这组对照中的最佳模型。

## 本地数据连接

正式数据与实验资产保存在：

```text
E:\maize_paper_workspace
```

仓库提交 `configs/paths.example.yaml` 作为路径模板。本机使用 `configs/paths.local.yaml`，该文件被 Git 忽略。

## 环境

最小 Python 依赖记录在 `requirements.txt`，本项目验证环境和 CUDA 安装方式见 `docs/ENVIRONMENT.md`。正式实验使用 `D:\Anaconda3\envs\cawa\python.exe`，不要把本机环境路径写入共享模型配置。

## 仓库记录边界

- `docs/superpowers/specs/` 保存已批准的设计规范，`docs/superpowers/plans/` 保存可审计的实施计划；二者属于研究溯源记录，不是训练输出。
- `figures/` 只保留可复现的数据审计脚本、摘要和小型 PNG/PDF 证据图。
- 模型权重、完整预测、原始/处理数据和本机路径配置继续由 `.gitignore` 排除。

## 开始工作

新 Agent 或新对话按以下顺序读取：

1. `AGENTS.md`
2. `docs/STATUS.md`
3. `docs/TASKS.md`
4. `docs/DATA_CONTRACT.md`
5. 最近一份 `docs/handoffs/` 交接记录（存在时）

项目级技术文档和实验状态可直接通过 Obsidian 浏览，也可以在 VS Code 中直接打开本仓库。
