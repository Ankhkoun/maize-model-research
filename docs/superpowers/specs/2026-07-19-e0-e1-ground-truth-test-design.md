# E0/E1 冻结 Ground-Truth Test 双口径设计

## 目标

在不改动 E0/E1 checkpoint、模型、阈值、训练、early stopping 或 checkpoint 选择的条件下，对 Test 305 个空间留出样本使用独立 30m 年度参考标签进行一次正式评价。此前 `test_evaluation` 的结果永久保留，但只能称为与训练伪标签的一致性诊断。

## 冻结输入

- Test manifest：`manifests/xinjiang_2021_e0_e1.csv`，仅 `split=test` 的 305 条。
- 真实标签根目录：`03_processed_data/labels_30m/xinjiang_2021/2021`（相对于 `paths.local.yaml` 的工作区根目录）。
- 每条记录的标签：`<root>/<sample_id>/y_patch_30m.npy`，shape `(85, 85)`，仅允许 `0`、`1`、`255`。
- E0/E1 checkpoint、manifest、normalization、seed42、24x24 window、patch2、stride24、CUDA AMP 与现有 Test-only 冻结资产完全不变。

## 已确认评价口径

所有统计仅使用真实标签 `!=255` 的像元；不使用耕地掩膜、不使用排除列表、不搜索阈值。二分类硬判定固定为 maize 概率 `>=0.5`（等价于两类概率 argmax）。模型每个样本只执行一次完整 256x256 滑窗拼接，随后同时生成两套指标。

### 原生 30m

裁掉模型网格的最右/最下各 1 个像元，将 `255x255` 类概率按无重叠 `3x3` 平均到 `85x85`。用平均后的两类概率计算 30m NLL/交叉熵、固定阈值预测、混淆矩阵和 OA/precision/recall/F1/IoU/mIoU/Kappa/area_ratio。

### 上采样到模型网格

将每个 `85x85` 真实标签用 `3x3` 最近邻复制成 `255x255`，仍丢弃模型最右/最下各 1 像元。直接用该网格上的模型概率与标签计算 10m-grid NLL/交叉熵、固定阈值预测和同一组指标。该方案是标签支持域的 3x3 复制，不宣称新增真实 10m 标注。

## 防错与审计

入口必须拒绝非 Test 样本、非 305 条、缺少标签、错误形状、标签域错误、已有输出目录和非 CUDA 运行。输出分别写入 checkpoint 父目录下的 `test_evaluation_ground_truth/native30m` 与 `test_evaluation_ground_truth/upsampled10m`，共享一次运行快照；文档记录两种尺度的监督像元数、完整混淆矩阵、checkpoint 与输出 SHA256。任何真实标签 Test 结果只作最终报告，不得反向改变实验设置。
