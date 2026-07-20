# Current Status

更新时间：2026-07-19
检查点类型：E0/E1 冻结 Train/Validation 与一次性 Test 评价均已完成。

## 项目目标

使用 Sentinel-2 多时相数据和可追溯伪标签完成玉米直接语义分割，并在空间独立区域上比较 TSViT baseline（E0）与仅增加可学习 Mexican-hat 小波位置编码的 E1。

## 已验证完成

- 已在主仓库实现 TSViT 风格 temporal-spatial Transformer 直接分割核心，以及可独立开关的三基 Mexican-hat WPE。
- WPE 只作用于真实时间 token，注入位置为 DOY 编码之后、temporal class token 之前；支持有效时间掩码、边界参数约束和有限梯度检查。
- 在 `cawa` 环境中最新复跑模型测试：`28 passed in 5.56s`。
- 已核对 Xinjiang 2021 当前数据集：1076 个 cube、1076 个标签目录、1076 条空间划分记录完全一一对应，没有缺失样本。
- cube 契约为 `[T,C,H,W]=[26,10,256,256]`、`float32`；波段顺序为 `B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12`。
- 标签契约为 `[256,256]`，值域仅为 `0=非玉米`、`1=玉米`、`255=ignore`；全量标签和置信度数组形状、有限性检查通过。
- 26 个时间点使用同一套时间日历。27762 个样本-时间点为直接观测，214 个为双侧插值；863 个样本无插值、212 个样本有 1 个插值时间点、1 个样本有 2 个插值时间点，没有 unresolved 时间点。
- 已完成空间划分与分布诊断，结果保存在 `figures/`；诊断脚本已对全部 1076 个样本执行。

## 数据划分与分布

现有区域级划分：

| Split | 区域 | 完整样本数 | 占比 |
| --- | --- | ---: | ---: |
| Train | `region_r007_c021`, `region_r009_c021`, `region_r013_c017` | 495 | 46.0% |
| Validation | `region_r008_c021` | 276 | 25.7% |
| Test | `region_r008_c022` | 305 | 28.3% |

像元标签分布：

| Split | 非玉米 | 玉米 | ignore | 有监督像元占比 | 有监督像元中的玉米占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 6,959,759 | 12,652,070 | 12,828,491 | 60.46% | 64.51% |
| Validation | 4,877,280 | 6,691,022 | 6,519,634 | 63.96% | 57.84% |
| Test | 4,352,588 | 7,942,837 | 7,693,055 | 61.51% | 64.60% |
| Total | 16,189,627 | 27,285,929 | 27,041,180 | 61.65% | 62.76% |

分布偏移证据：

- Validation 相对 Train 的有监督玉米比例低 6.67 个百分点；Test 与 Train 仅差 0.09 个百分点。
- 以每张影像一个确定性随机连续 `8x8` 区域进行全数据集光谱抽样，Validation 相对 Train 的平均标准化光谱偏移为 0.332 个训练标准差，Test 为 0.192。
- Validation 与 Train 的 26 时间点 NDVI 中位数曲线平均绝对差为 0.038、最大差为 0.098；Validation 峰值在第 17 个时间点（0.756），Train 在第 18 个时间点（0.804）。
- Test 与 Train 的 NDVI 曲线平均绝对差为 0.013、最大差为 0.032；Test 峰值同样在第 18 个时间点（0.795）。
- 结论：Validation 存在小到中等区域偏移、预期更难；Test 与 Train 更接近。三个集合的逐样本标签覆盖和类别比例仍有明显重叠，不构成“无法学习”的灾难性偏移。为避免空间泄漏，当前区域级划分继续保留。

可复现证据：

- `figures/split_distribution_summary.json`
- `figures/fig_split_distribution_preview.png` / `.pdf`
- `figures/fig_split_representative_tiles.png` / `.pdf`
- `figures/gen_fig_split_distribution_preview.py`

## 已确认的正式设计

- 训练数据窗口采用 Exact 风格 `24x24` 像元窗口；模型内部空间 patch 为 `2x2`，形成 `12x12=144` 个空间 token。
- E0 使用可学习 DOY 查找表；E1 与 E0 完全一致，仅增加 WPE。
- E0/E1 使用同一 manifest、窗口、标签、增强、损失、优化器、训练计划和 seed。
- 使用普通像元级交叉熵与 `ignore_index=255`；`pseudo_confidence` 仅用于数据审计，不进入首轮 E0/E1 loss。
- 首轮开发保持 495/276/305 区域划分；最终超参数固定后，可在不再调参的前提下用 Train+Validation 共 771 张影像重训练，再仅在 Test 报告结果。

## 当前实验状态

- 正式 E0/E1 配置已改为 `24x24` 窗口、内部 patch 2；工程旧 `256/patch8` 不再作为现行规范。
- DOY 已改为共享 `nn.Embedding(367,D)`，0 为 padding、1..366 为日历日。
- 相对路径 manifest、严格 Dataset、Train-only 归一化、窗口/拼接、指标、AMP 同批重试、checkpoint/resume、E0/E1 CLI 和 Test-only CLI 已实现；正式 Test 前 fresh 全量测试为 81 项。
- E0 正式训练：completed；epoch 25 因 warmup 后连续 12 个 epoch 无提升正常早停，best 为 epoch 13。独立完整 Validation 重放逐项一致：maize IoU=0.938347、F1=0.968193、Kappa=0.923977，混淆矩阵 `[[4625760,251520],[176506,6514516]]`。
- E1 正式训练：completed；epoch 22 因 warmup 后连续 12 个 epoch 无提升正常早停，best 为 epoch 8。独立新进程完整 Validation 重放逐项一致：maize IoU=0.933572、F1=0.965645、Kappa=0.919046，混淆矩阵 `[[4680450,196830],[260718,6430304]]`。
- E1 训练共发生 4 次可审计 AMP 同批 backoff，均定位于 `wavelet.alpha` 的非有限梯度；loss scale 依次为 `8192→4096→2048→1024→512`。每次重试后的 epoch 均完成 3327 个 optimizer steps，没有跳过 effective batch。
- 受控 Validation 对比中，E1 maize IoU 比 E0 低 0.004776，F1 低 0.002548，Kappa 低 0.004930；首轮结果不支持“WPE 提升 E0”的假设，当前保留 E0 为这组对照中的最佳模型。
- 已以 TDD 实现独立、严格的 Test-only 入口：默认 evaluator 仍为 Validation-only；只有显式 `expected_split="test"` 的独立 CLI 才能加载 Test，并严格核验 305 records、冻结 checkpoint/epoch/SHA256、manifest 和 normalization。
- 正式 Test 前 fresh pytest 为 `81 passed in 8.58s`；真实 CUDA smoke 使用 physical/effective batch 16/16、checkpoint 精确重载，且 `test_records_loaded=0`。
- E0 冻结 Test 一次性评价：305 tiles、12295425 supervised pixels；loss=0.123839、OA=0.951711、precision=0.968469、recall=0.956387、F1=0.962390、maize IoU=0.927507、mIoU=0.900577、Kappa=0.894961、area ratio=0.987525；混淆矩阵 `[[4105268,247320],[346410,7596427]]`。
- E1 冻结 Test 一次性评价：305 tiles、12295425 supervised pixels；loss=0.127737、OA=0.948240、precision=0.976072、recall=0.942992、F1=0.959247、maize IoU=0.921686、mIoU=0.894624、Kappa=0.888390、area ratio=0.966109；混淆矩阵 `[[4168976,183612],[452803,7490034]]`。
- E1-E0 的 Test maize IoU/F1/Kappa 差值为 `-0.005821/-0.003143/-0.006571`；既有 Validation 差值为 `-0.004776/-0.002548/-0.004930`。Test 未用于任何反向选择，且结果未改变基于 Validation 保留 E0 的结论。
- 没有为这 1076 个 Xinjiang 2021 样本找到一一对应的独立人工/年度参考标签；当前 Test 只能报告空间留出伪标签一致性，不能表述为独立地面真值精度。

## 环境与仓库

- 环境：`D:\Anaconda3\envs\cawa\python.exe`
- Python 3.11.14；PyTorch 2.10.0+cu128；CUDA 可用；GPU 为 NVIDIA GeForce RTX 5060 Ti。
- 分支：`codex/e0-e1-test-evaluation`
- HEAD：`d00e53d6cb3fb0eb8ccfdddd6c98757c2cb3a198`（本阶段基础提交；尚未新增 commit）。
- 工作树包含本阶段尚未提交的 Test-only 代码、测试、设计/计划、实验记录和 handoff；未 stage、commit、push 或创建 PR。
- 未修改 `D:\cj_swcc\_external\Exact` 或 `D:\cj_swcc\_external\TimeMIL`。
- 正式 E0 输出已冻结在 `E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e0_tsvit_doy_seed42`；best.pt SHA256 为 `CAB74C64897DA7FEA8A1A458ED94DAC2E23C0054A1772EAF16CF7BB5C3F9DE86`；Test 结果位于其 `test_evaluation` 子目录。
- 正式工作区库存已按固定流程刷新：825519 个文件、359.18971 GB；E0/E1 两组 Test JSON/run snapshot、新实验说明和总 handoff 均已抽查入清单。
- 正式 E1 输出：`E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e1_tsvit_doy_wpe_seed42`；Test 结果位于其 `test_evaluation` 子目录。

## 下一项有界任务

1. 完成正式工作区实验说明、总 handoff、库存刷新与抽查，并 fresh 运行全量测试、compileall 和 git diff --check。
2. 用户审阅一次性 Test 结果与文件清单后，自行 stage/commit/push 和创建 PR。
3. 后续若考虑 Train+Validation 最终重训练或新消融，必须独立预注册；不得再次使用本轮 Test 调参或选择模型。

## 2026-07-19：独立真实标签 Test 已完成

- 冻结 E0 epoch13 与 E1 epoch8 各仅运行一次 Test 305；对正式工作区 `03_processed_data/labels_30m/xinjiang_2021/2021` 的 `y_patch_30m.npy` 同时输出原生30m和标签复制模型网格两套结果。
- 所有统计为真实参考标签 `!=255`，未应用耕地掩膜或排除列表；最右/最下 1 个模型像元不纳入标签复制网格。
- 原生30m E0/E1 maize IoU 为 `0.785097/0.770246`；标签复制网格为 `0.777586/0.761288`。两种尺度均支持 E0 高于 E1，Test 未参与任何选择。
- 旧 `test_evaluation` 仅为伪标签一致性诊断；独立30m参考标签双口径结果为当前最终 held-out reference 报告。
- 分支 `codex/e0-e1-test-evaluation`，HEAD `d00e53d6cb3fb0eb8ccfdddd6c98757c2cb3a198`；不 stage/commit/push。
- 收尾验证：fresh 全量 pytest `92 passed in 9.65s`；`compileall` 和 `git diff --check` 均 exit0；17 个新文件的 whitespace 检查通过。
## 2026-07-19 补充：独立30m Test 基线对照

在同一 305 Test tiles、全部 `label !=255` 的 2,203,625 个原生30m cells 上，只读重算了原始 SAM+VVP/MMI Otsu 伪标签和历史 PEACENET EXP002，以消除其原先耕地掩膜口径与 E0/E1 的差异。原始伪标签的 F1/IoU/Kappa 为 `0.839264/0.723044/0.720908`；EXP002 为 `0.863125/0.759208/0.760399`；E0 为 `0.879613/0.785097/0.782022`，E1 为 `0.870213/0.770246/0.765665`。因此 E0 相对原始伪标签提高 `+4.03/+6.21/+6.11 pp`，相对 EXP002 提高 `+1.65/+2.59/+2.16 pp`（F1/IoU/Kappa）；E1 也高于两条基线但低于 E0。此对照不改变冻结方案、阈值、checkpoint 或基于 Validation 的 E0 选择。


## 2026-07-19：E2-W 训练、Validation 重放与双口径 Test 完成

- E2-W 使用 learned 26-slot `P_T` 和 content-only five-point Mexican-hat WPE；Xinjiang2021、24x24/patch2、Train495/Validation276/Test305、seed42、batch16/16。
- epoch25 因 bad_epochs=12 正常早停；best 为 epoch13/global step43251；`best.pt` SHA256 `A74C8A33030172E94020410D1E46FB3439C4438ECC1E796F06CEA347DF859428`。
- Test 前独立 Validation 重放逐项一致：IoU/F1/Kappa `0.936912/0.967428/0.922812`；相对 E0 `-0.001436/-0.000765/-0.001164`。
- 一次性伪标签 Test IoU/F1/Kappa `0.923915/0.960453/0.890835`；相对 E0 `-0.003592/-0.001937/-0.004126`。
- 一次性独立原生30m Test IoU/F1/Kappa `0.781160/0.877136/0.779780`；相对 E0 `-0.003937/-0.002476/-0.002242`。标签复制网格 IoU `0.774401`。
- 两个 Test 入口各只执行一次；Test 未用于调参、阈值、模型或 checkpoint 选择。E2-W 未超过 E0，继续保留 E0 为当前最佳模型。
- 分支 `codex/e2w-pt-mexican-hat` 的 E2-W 核心实现已提交为 `a2bd7ce9e771868e08327122dd31db4c58e3add7` 并推送至同名 `origin` 分支；本检查点时相对 `main` 为 0 behind / 1 ahead，PR 尚待用户创建。五方法原生 30 m 对比图资产仍未提交，复用规范见 `docs/FIGURES.md`。
