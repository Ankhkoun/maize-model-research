# 图件目录与复用规范

本文件登记仓库内可复现的论文图件；图件脚本、数值口径和输出格式应一同维护。图仅用于呈现已冻结的结果，不能作为 Test 集调参或选择模型的依据。

## `fig_native30m_test_comparison_table`

### 用途与固定评价口径

该图为 Xinjiang 2021 的五方法独立原生 30 m Test 对比表，用于汇总已完成的 held-out 结果。评价对象固定为 305 个空间隔离 Test tiles 中全部 `label != 255` 的 2,203,625 个原生 30 m 参考像元；不使用耕地掩膜、排除列表或阈值搜索。它不是上采样到 10 m 的标签复制网格结果，也不是伪标签一致性诊断结果。

方法顺序固定为：原始 SAM + VVP/MMI Otsu 伪标签、PEACENET EXP002、E0 TSViT+DOY、E1 TSViT+DOY+WPE、E2-W learned P_T + Mexican-hat WPE。指标列固定为 F1、maize IoU、Kappa、Precision、Recall、OA，均保留四位小数；每一列最高值加粗。Area ratio 不进入该图，以免将面积偏差和分类准确性混为同一主结论。

### 数据来源与解释边界

表内数值来自已写入 `docs/EXPERIMENTS.md` 的冻结独立原生 30 m Test 输出。E0 是当前选择的最佳模型；E1 仅改变为 WPE 版本，E2-W 同时将 DOY lookup 改为 learned `P_T` 并加入 Mexican-hat WPE，因此 E2-W 不能被解释为“只增加 WPE”的单变量消融。

不要将本图与以下两类指标混在同一比较中：

- 伪标签 Test：仅诊断模型与伪标签的一致性；
- 标签复制到模型网格的上采样 10 m Test：用于尺度敏感性检查，不能替代原生 30 m 真实标签主报告。

### 生成与验证

脚本：`figures/gen_fig_native30m_test_comparison_table.py`。在仓库根目录使用项目 Python 环境运行：

```powershell
& 'D:\Anaconda3\envs\cawa\python.exe' figures\gen_fig_native30m_test_comparison_table.py
```

脚本同时生成下列版本：

- `figures/fig_native30m_test_comparison_table.png`：用于预览、Word 和幻灯片；
- `figures/fig_native30m_test_comparison_table.pdf`：用于论文排版和矢量导出。

修改数值、方法或版式后，先运行 `tests/test_native30m_test_comparison_figure.py`；测试会校验方法顺序、评价口径、六个指标、每列最大值的加粗规则，以及 PNG/PDF 是否能够生成。完成后应人工核对标题、305 tiles、2,203,625 valid cells、原生 30 m 说明和小数位数。

### 新增实验时的复用步骤

1. 先以冻结 checkpoint 完成独立原生 30 m Test，并把完整指标、混淆矩阵和证据路径写入 `docs/EXPERIMENTS.md`。
2. 仅将相同 Test 支持集和相同评价脚本得到的 F1、IoU、Kappa、Precision、Recall、OA 加入脚本中的数据表；不得从不同掩膜、不同尺度或不同年份拼接数值。
3. 保持方法顺序；如需新增方法，使用清晰实验名并让所有方法重新按列比较最高值。
4. 重跑图件测试与生成脚本，同时更新本文件和实验记录，说明新增方法是否为严格单变量消融。

