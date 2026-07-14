# Experiment Registry

本文件记录可复现实验索引。完整 checkpoint、概率图和推理结果保存在 `E:\maize_paper_workspace`，不提交 Git。

## 已完成基线：Xinjiang 2021 PEACE-Net

结果根目录：

```text
E:\maize_paper_workspace\06_models\retrain_outputs\peacenet_xinjiang_confidence_ablation_50ep_es
```

| ID | 置信度方案 | 权重 | Soft update | Val F1 @ 0.5 | Test F1 @ 0.5 | 状态 |
| --- | --- | --- | --- | ---: | ---: | --- |
| EXP001 | 原始 SAM raw | raw | 开 | 0.878807 | 0.874970 | 完成 |
| EXP002 | 保守 VVP/MMI | raw | 开 | 0.880629 | 0.876772 | 完成，正式推荐 |
| EXP003 | 保守 VVP/MMI | class-balanced | 开 | 0.879503 | 0.875353 | 完成 |
| EXP004 | 保守 VVP/MMI | class-balanced | 关 | 0.871252 | 0.870863 | 完成 |

EXP002 根据验证 AOI 选出。其验证导向阈值为 0.13，对应 Val F1=0.883803、Test F1=0.885334、Test IoU=0.794259、Test Kappa=0.692265。EXP004 的部分测试指标更高，但不能依据测试集改选模型。

## 计划模型对

| ID | Temporal encoding | WPE quality | 状态 | 目的 |
| --- | --- | --- | --- | --- |
| E0 | Exact-style DOY | 不适用 | 未实现 | 直接分割主基线 |
| E1 | DOY + 3-scale WPE | 有效帧均为 1 | 未实现 | 隔离 WPE 增益 |
| E2 | WPE only | 有效帧均为 1 | 暂缓 | 检验绝对 DOY 的互补性 |
| E3 | DOY + WPE | source-derived quality | 暂缓 | 质量感知扩展 |

## 新实验登记字段

每个正式实验至少记录：实验 ID、日期、Git commit、配置文件、数据版本、manifest、seed、环境、checkpoint、预测路径、评价规则、指标、结论和异常。
