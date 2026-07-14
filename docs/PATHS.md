# Paths And Source Locations

## 正式工作区

```text
E:\maize_paper_workspace
```

| 用途 | 相对正式工作区的目录 |
| --- | --- |
| 原始数据 | `02_raw_data` |
| 处理数据 | `03_processed_data` |
| 指数 | `04_indices` |
| 伪标签 | `05_pseudo_labels` |
| 模型与 checkpoint | `06_models` |
| 推理输出 | `07_inference` |
| 评价结果 | `08_evaluation` |
| 图件 | `09_figures` |

## 已知参考代码

```text
D:\cj_swcc\_external\Exact
D:\cj_swcc\_external\TimeMIL
D:\cj_swcc\_external\PEACE-Net
D:\cj_swcc\maize_workspace_inventory
```

外部 Exact 和 TimeMIL 树已有本地修改，只读参考，不作为主仓库实现目标。

## 已知 Xinjiang 2021 路径

```text
Cube:
E:\maize_paper_workspace\03_processed_data\cubes\xinjiang_2021\cubes\2021

30m labels:
E:\maize_paper_workspace\03_processed_data\labels_30m\xinjiang_2021\2021

Cropland masks:
E:\maize_paper_workspace\03_processed_data\cropland_masks\xinjiang_2021\2021

Existing split candidate:
E:\maize_paper_workspace\01_code\spring_maize_paper_dataset\generated\splits_2021_top5_region_holdout.csv
```

这些路径在开始实现 dataset 前必须重新核验，并转换为相对于 `workspace_root` 的 manifest。
