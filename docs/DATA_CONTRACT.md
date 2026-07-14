# Data And Evaluation Contract

## 输入概念

当前核心样本概念为：

```text
x_cube.npy              [T,C,H,W] = [26,10,256,256], float32
pseudo_label.npy        [H,W]
pseudo_confidence.npy   [H,W]
sam_object_id.npy       [H,W], optional
sam_boundary.npy        [H,W], optional
obs_quality.npy         [T,H,W], optional and phase two only
```

十个 Sentinel-2 波段顺序为 `B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12`。时间范围为 2021-04-01 至 2021-09-30，共 26 个有序时间窗。最终代码必须从 manifest 和数据元信息核验这些假设，不得仅依赖本文档硬编码。

## Dataset 返回值

```python
{
    "images": FloatTensor[T, C, H, W],
    "doy": FloatTensor[T],
    "valid_mask": BoolTensor[T],
    "label": LongTensor[H, W],
    "confidence": FloatTensor[H, W],
    # phase two only:
    "obs_quality": FloatTensor[T, H, W],
}
```

`pseudo_confidence` 表示伪标签类别可信度，只能用于监督 loss 的 mask 或权重。`obs_quality` 才能表达原始观测、双侧插值、单侧估计和无效时相的质量。

## 标签与 loss 原则

- 如果背景不是可靠语义类别，使用玉米/非玉米二分类，并将低置信、nodata、冲突和未标注像元设为 `ignore_index`。
- 当前候选起点为 `confidence < 0.30 -> ignore`，其余像元使用 `confidence^2` 加权交叉熵；该设置必须先通过数据分布核验。
- 可选 Dice 系数候选值为 0.2，但 E0/E1 必须一致。

## Manifest

提交到 Git 的 manifest 只记录样本 ID、区域、年份和相对于 `workspace_root` 的路径，例如：

```csv
sample_id,cube_path,label_path,region,year,split
001587,03_processed_data/cubes/xinjiang_2021/cubes/2021/001587.npy,05_pseudo_labels/example/001587.npy,xinjiang,2021,train
```

上例只说明字段和相对路径规则，不代表最终文件名。正式 manifest 必须根据磁盘现状生成并核验。

## 独立评价

- 10m 输出按现有规则聚合到 30m 年度参考网格。
- 仅在 cropland mask 与有效参考像元内统计。
- train/validation/test 按空间 AOI 留出。
- 所有模型、阈值和 checkpoint 决策只看 validation AOI。
- 最少报告 maize IoU、Precision、Recall、F1、Kappa、area ratio、macro-F1、mIoU、混淆矩阵和 per-AOI 指标。
