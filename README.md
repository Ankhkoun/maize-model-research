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

当前已完成可复现基线是 Xinjiang 2021 PEACE-Net EXP002。Exact 风格 E0 和 Exact+WPE E1 尚未实现或训练。

## 本地数据连接

正式数据与实验资产保存在：

```text
E:\maize_paper_workspace
```

仓库提交 `configs/paths.example.yaml` 作为路径模板。本机使用 `configs/paths.local.yaml`，该文件被 Git 忽略。

## 开始工作

新 Agent 或新对话按以下顺序读取：

1. `AGENTS.md`
2. `docs/STATUS.md`
3. `docs/TASKS.md`
4. `docs/DATA_CONTRACT.md`
5. 最近一份 `docs/handoffs/` 交接记录（存在时）

项目级技术文档和实验状态可直接通过 Obsidian 浏览，也可以在 VS Code 中直接打开本仓库。
