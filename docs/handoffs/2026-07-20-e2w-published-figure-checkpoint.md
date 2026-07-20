# 2026-07-20：E2-W 已推送与原生 30 m 对比图检查点

## 目的

记录 E2-W 核心实现已经推送、尚待用户创建 PR 的状态，并交接五方法独立原生 30 m Test 对比图的可复现资产。

## 已验证状态

- 当前分支：`codex/e2w-pt-mexican-hat`。
- E2-W 核心实现提交：`a2bd7ce9e771868e08327122dd31db4c58e3add7`（`feat: add E2-W Mexican-hat temporal encoding experiment`），已推送至同名 `origin` 分支。
- 该分支相对 `main` 为 `0 behind / 1 ahead`；`main` 已含 E0/E1 Test 评价合并提交 `ec76789`。E2-W 的 PR 尚未创建，发布与合并仍由用户决定。
- E2-W 的正式训练、独立 Validation 重放、伪标签 Test 和独立原生 30 m / 标签复制网格 Test 均已完成；Test 未参与模型选择。完整指标、混淆矩阵、哈希和解释见 `docs/EXPERIMENTS.md`。
- 结论不变：E2-W 的独立原生 30 m maize IoU/F1/Kappa 为 `0.781160/0.877136/0.779780`，略低于 E0 的 `0.785097/0.879613/0.782022`；保留 E0 为当前最佳模型。E2-W 同时更换时序位置编码并加入 WPE，不能将其解释为纯 WPE 单变量效应。

## 本次未提交的图件资产

下列文件是当前工作树内尚未提交的、可复现的五方法原生 30 m 主报告图件资产：

- `figures/gen_fig_native30m_test_comparison_table.py`
- `figures/fig_native30m_test_comparison_table.png`
- `figures/fig_native30m_test_comparison_table.pdf`
- `tests/test_native30m_test_comparison_figure.py`
- `docs/superpowers/specs/2026-07-20-native30m-test-comparison-table-design.md`
- `docs/superpowers/plans/2026-07-20-native30m-test-comparison-table.md`

复用规范已集中在 `docs/FIGURES.md`：固定同一 305 tiles、2,203,625 个 `label != 255` 的原生 30 m 支持集；只报 F1/IoU/Kappa/Precision/Recall/OA；按列加粗最高值；不混入伪标签或上采样 10 m 指标；不显示 area ratio。

## 建议的下一项有界任务

由用户审阅图件版式与文字后，提交上述图件资产及本次检查点文档；随后在 `codex/e2w-pt-mexican-hat` 创建 E2-W PR。建议提交信息：`docs: add reusable native 30m comparison figure guide`。不要在该过程中重新读取 Test、改动冻结 checkpoint 或依据 Test 调参。

## 验证待记录

完成本检查点后，应记录 fresh 图件单测、全量 pytest、`compileall` 与 `git diff --check` 的结果；未通过前不得声称图件提交已准备就绪。
