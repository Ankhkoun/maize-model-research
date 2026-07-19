# 2026-07-19 E0/E1 冻结独立真实标签 Test 双口径评价交接

## 边界与冻结资产

- 分支：`codex/e0-e1-test-evaluation`；HEAD：`d00e53d6cb3fb0eb8ccfdddd6c98757c2cb3a198`。
- E0：epoch13 `best.pt`，SHA256 `CAB74C64897DA7FEA8A1A458ED94DAC2E23C0054A1772EAF16CF7BB5C3F9DE86`。
- E1：epoch8 `best.pt`，SHA256 `60B7A2C0715D45B20F5723DAD8C5ED8CB33E785AAD2A15AD39797A512E0CDDC8`。
- manifest SHA256：`79DCAAF1270D48B99FA50AF2A57548B50D7FD3E232620D11DC53E5E64C1177A8`；normalization SHA256：`1401DC9AAFE9A30A8916AB1A3C738080DB63D433C04716CA27FA5CABB51EC142`。
- 未依据 Test 调参、搜索阈值、改模型、选 checkpoint、改 early stopping 或重训；未 stage/commit/push/PR。

## 独立参考标签契约

- 正式路径：`E:\maize_paper_workspace\03_processed_data\labels_30m\xinjiang_2021\2021`，而非旧代码树下已迁移的路径。
- Test manifest 的 305 条记录均有 `<sample_id>/y_patch_30m.npy`；标签为 85x85、值域 `{0,1,255}`。
- 只统计 `label !=255`，不应用耕地掩膜或 exclude list。
- 同一份 256x256 完整滑窗拼接 logits 同时导出：
  - `native30m`：裁为255x255，3x3 类概率平均至85x85，计算 NLL/loss 和指标。
  - `upsampled10m`：85x85 标签以3x3复制到255x255，模型最右/最下1像元忽略；这只是标签支持域复制，不是新增10m真值。
- 两类概率用 argmax 判定；这等价于0.5决策，精确平局归非玉米。

## 正式运行

- E0 新进程 exit0，112.8s；E1 新进程 exit0，118.7s。
- 每个 checkpoint 父目录下的新 `test_evaluation_ground_truth` 在启动前均不存在，运行后即锁定；每个目录有一个 `run_snapshot.json` 和两个尺度的 `test_evaluation.json`。

| 指标 | E0 原生30m | E1 原生30m | E1-E0 | E0 标签复制网格 | E1 标签复制网格 | E1-E0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loss | 0.381783613 | 0.399297600 | +0.017513987 | 0.434585314 | 0.470717486 | +0.036132172 |
| OA | 0.892183107 | 0.884158149 | -0.008024959 | 0.888062523 | 0.879650677 | -0.008411847 |
| maize precision | 0.891711023 | 0.885267201 | -0.006443822 | 0.887882859 | 0.884144265 | -0.003738594 |
| maize recall | 0.867838221 | 0.855663084 | -0.012175137 | 0.862249854 | 0.845646949 | -0.016602904 |
| maize F1 | 0.879612674 | 0.870213437 | -0.009399237 | 0.874878642 | 0.864467219 | -0.010411422 |
| maize IoU | 0.785096951 | 0.770245872 | -0.014851079 | 0.777586022 | 0.761287771 | -0.016298251 |
| mIoU | 0.803607777 | 0.790425604 | -0.013182173 | 0.796837943 | 0.782987650 | -0.013850293 |
| Kappa | 0.782021820 | 0.765665096 | -0.016356724 | 0.773650850 | 0.756338500 | -0.017312351 |
| area ratio | 0.973228096 | 0.966559116 | -0.006668980 | 0.971130195 | 0.956458106 | -0.014672089 |

- 原生30m：两模型均为 305 tiles / 2,203,625 supervised pixels；E0 CM `[[1098066,105406],[132182,867971]]`，E1 `[[1092559,110913],[144359,855794]]`。
- 标签复制网格：两模型均为 305 tiles / 19,832,625 supervised pixels；E0 CM `[[9851175,980073],[1239941,7761436]]`，E1 `[[9833795,997453],[1389390,7611987]]`。
- 四个矩阵元素和均严格等于对应 supervised-pixel 数。两种尺度均支持 E0 高于 E1；Test 结果没有改变先前仅凭 Validation 的模型选择。

## 哈希

| 资产 | E0 SHA256 | E1 SHA256 |
| --- | --- | --- |
| `native30m/test_evaluation.json` | `0A4A5F31DE61021A1AAED35884B31CB4AA41031417DE1ECD52DD0E3F8A040262` | `C291AD6E0B395D861CB7CC44C88C196417AB1D75229F32C0B74C312970B1B81B` |
| `upsampled10m/test_evaluation.json` | `29CF7830BBCF76A289834A46C5E6AA4B5D07BB8832D4F6C9178461A5DAACB73A` | `1D497BBF5E7DF3B3533839884D99F0B59498177DCA32F92E4C9F4884675BB2FE` |
| `run_snapshot.json` | `14DA6BFAF7F9B625E6DD20F7A5BA6EF5EEF348398786058E0A9D5ABD6133F645` | `B792B8942890DDF3E35A230E4005B526C41908E38A91630C2C64310F554D9A52` |

## 正式工作区记录与库存

- 实验说明：`00_docs/experiment_notes/maize_model_research_e0_e1_ground_truth_test_evaluation_2026-07-19.md`，SHA256 `45CE6985E06CA9C674622FA0F6A2AF00A87FCC2314C5DA478B36FE517CA38CD3`。
- 总 handoff 已追加一个章节，更新后 SHA256 `669808E4FAEF2F31A14ABA887DE2D7D649D184A4B268A37C0643D0F3654F5AD6`。
- 固定 `refresh_workspace_inventory.py` exit0，277.1s；刷新后 825,526 files / 359.189727 GB。逐文件清单对本轮 4 个尺度 JSON、2 个快照和 1 个实验说明共匹配 7 项。
## 伪标签 Test 的位置

此前 `test_evaluation` 中的 E0/E1 数值仍有效，但只衡量与训练监督伪标签的空间留出一致性；独立年度30m参考标签的本交接记录是最终 held-out reference 评价。不得用任一 Test 结果设计新的训练、消融或重训；任何后续实验必须独立预注册。

## 验证与后续

- 实现前 fresh pytest：`92 passed in 8.95s`。
- 合成 CUDA smoke：RTX 5060 Ti 通过；未读取正式 Test 样本或真实标签。
- 最终 fresh pytest：`92 passed in 9.65s`。
- `D:\Anaconda3\envs\cawa\python.exe -m compileall -q src scripts tests` exit0；`git diff --check` exit0（仅现有 CRLF 提示，无 whitespace error）；17 个未跟踪新文件的单独 whitespace 检查也通过。
- 正式工作区固定库存刷新 exit0，已完成。
- 用户负责审核后 stage/commit/push。
