# D Drive Workspace Review

审阅日期：2026-07-14

本文件记录 `D:\cj_swcc` 的初步归属判断。当前没有执行删除、移动或覆盖操作。任何删除都必须先与正式工作区做内容/哈希核验，并由用户单独确认。

## 已合并到主仓库的知识

以下目录的有效事实已经整理进本仓库的项目规则、状态、数据契约、实验登记、路径说明和 Exact-WPE 规范：

```text
D:\cj_swcc\maize-agent-handoff
D:\cj_swcc\maize-wpe-exact-handoff
D:\cj_swcc\maize-peacenet-agent-handoff
```

暂时保留三个原目录，直到逐文件复核和主仓库稳定提交完成。之后优先归档，而不是直接删除。

## 必须保留

| 路径 | 原因 |
| --- | --- |
| `_external\Exact` | Exact 上游参考树，且已有本地修改。 |
| `_external\TimeMIL` | TimeMIL 复现与 WPE 概念参考树，且已有兼容补丁。 |
| `_external\PEACE-Net` | PEACE-Net 参考实现。 |
| `maize_workspace_inventory` | 数据完整性和可训练样本库存来源。 |
| `research-notes` | 已建立的 Obsidian 总索引与模板库，应保留并链接各项目仓库。 |

## 建议迁移或整合后归档

| 内容 | 建议归属 |
| --- | --- |
| 根目录 `build_*`、`prepare_*` 数据/伪标签脚本 | 与 `E:\maize_paper_workspace\01_code\spring_maize_paper_dataset` 同名或功能脚本对比后，将唯一有效版本纳入正式 pipeline；历史版本归档。 |
| `inspect_maize_workspace.py`、数据重组脚本和 manifest | 数据治理工具与迁移审计资料；合并到正式工作区 `01_code`/`00_docs/inventories` 后归档。 |
| `sam_report_selected_figures*`、`usa_bad_case_previews`、根目录 PNG | 若属于正式图件，迁移到 `E:\maize_paper_workspace\09_figures` 的明确子目录；否则作为可再生产物候选清理。 |
| `france_sam_tmp_verify` | 核验是否已有正式结果副本；若只是临时验证目录，可作为候选清理。 |
| `.vscode` | 若只包含个人编辑器设置，保留本地或并入具体项目，不应作为科研资产。 |
| `Exact` | 不是第二个源码仓库，而是早期 PASTIS24 demo 启动脚本、配置和数据准备工具；保留为复现记录，后续归档。 |

## 高概率可清理，但仍需确认

| 内容 | 条件 |
| --- | --- |
| `__pycache__` | 可由 Python 自动重建。 |
| `biaozhu\sam_vit_h.pth` | 与 `E:\maize_paper_workspace\06_models\foundation_models\sam\sam_vit_h.pth` 的 SHA-256 均为 `A7BF3B02F3EBF1267ABA913FF637D9A2D5C33D3173BB679E46D9F338C26F262E`；确认 E 盘备份策略后可删除 D 盘重复副本。 |
| 已确认复制到 `E:` 且 SHA256 相同的根目录图片/脚本副本 | 必须先输出逐文件哈希对照。 |
| 旧 dry-run 清单 | 仅在对应 apply 清单、迁移日志和最终库存都完整时归档或清理。 |

## 当前禁止直接删除

- `delete_d_copied_after_migration.ps1` 及迁移 manifest：名称不能证明迁移已经完整成功，它们也是审计线索。
- `D:\cj_swcc\Exact`：内容虽已确认是 demo 包装而非源码仓库，但在将其运行说明归档前不要删除。
- 三个 handoff：主仓库首次提交后仍需做逐文件覆盖检查。
- 所有用户标注、正式评价结果和不能重新生成的人工审查图件。
