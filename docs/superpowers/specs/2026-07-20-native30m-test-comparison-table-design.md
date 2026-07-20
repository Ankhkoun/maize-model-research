# 五方法独立原生 30 m Test 对比表设计

## 目标

生成一张可直接用于论文或汇报的静态表格图，将原始 SAM+VVP/MMI Otsu 伪标签、PEACENET EXP002、E0、E1 和 E2-W 放在同一评价口径下比较。

## 评价口径

- Xinjiang 2021 空间独立 Test AOI，共 305 个样本。
- 仅评价年度 30 m 参考标签中 `label != 255` 的 2,203,625 个原生网格单元。
- 不使用耕地掩膜或排除列表。
- 只展示 F1、IoU、Kappa、Precision、Recall 和 OA；不展示 Area ratio。
- 所有数值显示四位小数，每列最大值加粗。

## 视觉设计

- 采用截图对应的白底、细横线、无竖线表格样式。
- 第一列左对齐，数值列居中；表头使用浅灰底和粗体。
- E0、E1、E2-W 行使用极浅蓝色分组底色，基线行保持白色。
- 标题为 `Independent native 30 m Test comparison`，副标题说明 305 tiles 和 2,203,625 valid cells。
- 同时输出 300 DPI PNG 和矢量 PDF；保留生成脚本以便复现。

## 输出

- `figures/gen_fig_native30m_test_comparison_table.py`
- `figures/fig_native30m_test_comparison_table.png`
- `figures/fig_native30m_test_comparison_table.pdf`

本任务只新增仓库内的小型可复现图件，不修改正式工作区模型、Test 结果或实验选择，也不执行 stage、commit、push 或 PR。
