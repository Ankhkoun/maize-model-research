# 2026-07-16 E0/E1 数据分布中间交接

## 根目录与目标

- 项目根目录：`D:\cj_swcc\maize-model-research`
- 目标：在空间独立 Xinjiang 2021 区域上公平比较 E0（TSViT+DOY）与 E1（E0+三个可学习 Mexican-hat WPE 基）。
- 本检查点性质：中间检查点。E0/E1 正式训练均未运行。

## 本阶段已验证工作

- 实现 TSViT 风格 temporal-spatial Transformer 直接分割核心。
- 实现三个全局可学习 Mexican-hat 基；WPE 在 DOY 后、temporal class token 前仅注入一次。
- 验证 WPE 参数约束、掩码、class token 隔离、`alpha=0` 等价性、CUDA 前反向有限性。
- 最新模型测试：`D:\Anaconda3\envs\cawa\python.exe -m pytest -q`，结果 `28 passed in 5.56s`。
- 对照 Exact 确认 24x24 是训练数据窗口，内部 Transformer patch 为 2x2；正式 E0/E1 设计已改为 `image_size=24`、`patch_size=2`，但当前代码配置仍待修改。
- 确认 Exact/TSViT 使用 one-hot+Linear 的可学习 DOY 表；当前主仓库仍是标量 `Linear(1,D)`，正式训练前必须替换为 E0/E1 共享表。
- 完成 1076 个样本的数据完整性、标签分布、区域划分、光谱和 NDVI 分布诊断，并生成可复现图表。

## 数据版本与完整性

- workspace root：`E:\maize_paper_workspace`
- cube：`03_processed_data/cubes/xinjiang_2021/cubes/2021`
- 标签：`05_pseudo_labels/sam_refined/xinjiang_vvp_mmi_confidence_corrected/conservative_v1`
- 外部 split manifest：`01_code/spring_maize_paper_dataset/generated/splits_2021_top5_region_holdout.csv`
- manifest SHA256：`59C2CF1D960194BE2BE46928FAEB06096461B2FAF071910090E01E7BD4A3972B`
- cube/标签/manifest：1076/1076/1076，样本集合完全一致，无缺失。
- cube：`[26,10,256,256]`、`float32`。
- 标签：`[256,256]`，值域 `0,1,255`。
- 波段：`B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12`。
- 时间日历：所有样本共用同一 26 时间点日历。
- 时间状态：27762 observed、214 two-sided interpolated、0 unresolved；863 个样本无插值、212 个有 1 个插值时间点、1 个有 2 个插值时间点。

## 区域划分

| Split | 区域 | 样本数 |
| --- | --- | ---: |
| Train | `region_r007_c021`, `region_r009_c021`, `region_r013_c017` | 495 |
| Validation | `region_r008_c021` | 276 |
| Test | `region_r008_c022` | 305 |

该划分按整个区域隔离。后续 24x24 窗口必须继承父 256x256 样本的 split，不能重新随机跨 split 分配。

## 标签分布重点证据

| Split | 非玉米 | 玉米 | ignore | 有监督占比 | 玉米/有监督 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 6,959,759 | 12,652,070 | 12,828,491 | 60.46% | 64.51% |
| Validation | 4,877,280 | 6,691,022 | 6,519,634 | 63.96% | 57.84% |
| Test | 4,352,588 | 7,942,837 | 7,693,055 | 61.51% | 64.60% |
| Total | 16,189,627 | 27,285,929 | 27,041,180 | 61.65% | 62.76% |

- Validation 玉米占有监督像元比例比 Train 低 6.67 个百分点。
- Test 与 Train 该比例只差 0.09 个百分点。
- 所有完整样本都有有效监督像元；无玉米正像元样本为 Train 9、Validation 5、Test 0。

## 光谱与 NDVI 分布重点证据

统计方法：每张完整影像根据 sample_id 确定性随机选一个连续 8x8 空间区域，读取全部 26 时间点和 10 波段。标签统计是全像元精确统计；光谱/NDVI 是抽样统计。

| 比较 | 平均标准化光谱偏移 | NDVI 曲线平均绝对差 | NDVI 最大差 | 峰值时间点 | 峰值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 0 | 0 | 0 | 18 | 0.804 |
| Validation vs Train | 0.332 | 0.038 | 0.098 | 17 | 0.756 |
| Test vs Train | 0.192 | 0.013 | 0.032 | 18 | 0.795 |

结论：Validation 有小到中等区域偏移且预期更难，Test 更接近 Train；三个集合逐样本分布仍明显重叠，不构成灾难性偏移。保留区域级划分，避免空间泄漏。

可复现证据：

- `figures/gen_fig_split_distribution_preview.py`
- `figures/split_distribution_summary.json`
- `figures/fig_split_distribution_preview.png` / `.pdf`
- `figures/fig_split_representative_tiles.png` / `.pdf`
- 代表样本：Train `sample_001391`、Validation `sample_002230`、Test `sample_002074`
- summary SHA256：`4C9E204BF07181F89CE326F8C887978303BAA4B0C1E6310630E2A388A9F75255`

## 已确认决策

- 正式训练窗口 24x24，内部 patch 2x2，形成 144 个空间 token。
- E0/E1 共享可学习 DOY 表；E1 唯一增加 WPE。
- 首轮损失为普通像元级交叉熵，`ignore_index=255`；`pseudo_confidence` 不进入 loss。
- 首轮保留 495/276/305 区域划分。
- 所有模型与 checkpoint 选择只看 Validation；Test 只在方案冻结后评价。
- 超参数固定后可考虑 Train+Validation=771 张进行最终重训练，但不得再用 Test 调参。

## 实验状态

### E0

- 状态：not run
- 配置：`configs/models/tsvit_baseline.yaml`，当前为工程 smoke 配置，待改 24/patch2 和 DOY 表。
- seed：42。
- command/output/checkpoint/metrics/conclusion：none。

### E1

- 状态：not run
- 配置：`configs/models/tsvit_wpe_basic.yaml`，当前为工程 smoke 配置，待改 24/patch2 和共享 DOY 表。
- seed：42。
- command/output/checkpoint/metrics/conclusion：none。

### DATA-AUDIT-2021

- 状态：completed
- 输出：`figures/` 下两组 PNG/PDF 和 JSON 摘要。
- 首次分散随机像元扫描在 304 秒超时；12 张影像计时确认随机磁盘页访问为根因，改为连续 8x8 块后 1076/1076 完整复跑成功。

## 本阶段会话文件

- `src/models/__init__.py`
- `src/models/tsvit_segmentation.py`
- `src/models/wavelet_position_encoding.py`
- `configs/models/tsvit_baseline.yaml`
- `configs/models/tsvit_wpe_basic.yaml`
- `tests/test_wavelet_position_encoding.py`
- `tests/test_tsvit_wpe_equivalence.py`
- `tests/test_e1_robustness.py`
- `tests/test_model_configs.py`
- `tests/test_wpe_review_regressions.py`
- `docs/superpowers/plans/2026-07-15-e1-learnable-wavelet-position-encoding.md`
- `docs/superpowers/specs/2026-07-15-e1-learnable-wavelet-position-encoding-design.md`
- `figures/` 下本交接列出的生成脚本、摘要和正式图件
- 本次更新的 `docs/STATUS.md`、`docs/TASKS.md`、`docs/DECISIONS.md`、`docs/EXPERIMENTS.md` 和本 handoff

未发现需要归属于其他人的未提交修改；仍应在发布前逐文件复核。未修改、重置或清理 Exact/TimeMIL 外部参考树。

## 验证命令与结果

- `D:\Anaconda3\envs\cawa\python.exe -m pytest -q` -> `28 passed in 5.56s`
- 分布脚本最终复跑 -> `1076/1076`，exit code 0
- PNG 完整性 -> 3428x2077 与 2666x1812 均可解码
- PDF 文件头和大小检查通过
- cube/label/manifest 集合差异均为空

## 阻塞与不确定性

- 当前 DOY 仍为标量 `Linear(1,D)`，与已确认正式设计不一致。
- 当前模型配置仍为 256/patch8，与已确认 24/patch2 正式训练设计不一致。
- 项目内正式相对路径 manifest、Dataset/DataLoader、Train-only 归一化、训练引擎、指标、checkpoint 和滑窗推理未实现。
- 未找到这 1076 个 Xinjiang 2021 样本的一一对应独立地面真值；当前 Test 只能衡量伪标签一致性。
- 24x24 窗口在 256x256 边缘的 padding/拼接规则尚待实现和测试。

## 下一项精确任务

先不训练：以 TDD 修改 E0/E1，使其共享 `padding=0, DOY=1..366` 的可学习 DOY 表；将模型与配置改为 `image_size=24, patch_size=2`；随后固化 1076 样本相对路径 manifest，并实现 Dataset/DataLoader 与 Train-only 归一化。完成单批次和小规模 smoke test 后展示结果，获得用户确认再启动正式 E0/E1。

## Git 状态

- branch：`main`
- HEAD：`b0a3463b1cbf506402b6d7b3034b53c7e2116240`
- remote：`origin git@github.com:Ankhkoun/maize-model-research.git`
- 工作树：dirty，包含上述本阶段文件。
- stage/commit/push：均未执行。

## 已读取来源

- 主仓库：`AGENTS.md`、`README.md`、四份研究记录、模型配置/代码/测试。
- 正式工作区：`README.md`、`AGENT_FIXED_RULES.md`、`agent_project_handoff_2026-06-11.md`，以及与本任务相关的 manifest、cube/标签元数据和 split 脚本。
- 参考代码：Exact 的 `TSViT_seg.py`、`Exact_cls.py`、PASTIS24 配置、数据变换和 24x24 window 生成脚本。

本阶段未写入正式数据工作区，因此没有运行其库存刷新脚本。
