# Experiment Registry

展示规范：本总表中的连续型评价指标（loss、OA、precision、recall、F1、IoU、mIoU、macro-F1、Kappa、area ratio 及其差值）统一显示到小数点后 5 位；原始 JSON/CSV、哈希、整数计数和命令参数保持原样，以保证可复现。

更新时间：2026-07-19。完整 checkpoint、概率图和正式推理结果保存在正式工作区，不提交 Git。

## 已完成参考基线：Xinjiang 2021 PEACE-Net

结果根目录：`06_models/retrain_outputs/peacenet_xinjiang_confidence_ablation_50ep_es`（相对于正式工作区）。

| ID | 置信度方案 | 权重 | Soft update | Val F1 @ 0.5 | Test F1 @ 0.5 | 状态 |
| --- | --- | --- | --- | ---: | ---: | --- |
| EXP001 | 原始 SAM raw | raw | 开 | 0.87881 | 0.87497 | completed |
| EXP002 | 保守 VVP/MMI | raw | 开 | 0.88063 | 0.87677 | completed，正式推荐 |
| EXP003 | 保守 VVP/MMI | class-balanced | 开 | 0.87950 | 0.87535 | completed |
| EXP004 | 保守 VVP/MMI | class-balanced | 关 | 0.87125 | 0.87086 | completed |

EXP002 根据 Validation 选择。其 Validation 导向阈值为 0.13，对应 Val F1=0.88380、Test F1=0.88533、Test IoU=0.79426、Test Kappa=0.69227。以上是既有实验记录，本检查点未重跑。

## DATA-AUDIT-2021：E0/E1 启动前数据与分布诊断

| 字段 | 记录 |
| --- | --- |
| ID | `DATA-AUDIT-2021` |
| 状态 | completed |
| 日期 | 2026-07-16 |
| Git commit | 基础 HEAD `b0a3463b1cbf506402b6d7b3034b53c7e2116240`；结果位于未提交工作树 |
| 配置/脚本 | `figures/gen_fig_split_distribution_preview.py`，每张完整影像确定性随机抽样一个连续 8x8 区域 |
| 命令 | `D:\Anaconda3\envs\cawa\python.exe figures\gen_fig_split_distribution_preview.py` |
| 环境 | `cawa`；Python 3.11.14；NumPy 2.3.5；Matplotlib 3.10.8 |
| 数据版本 | cube：`03_processed_data/cubes/xinjiang_2021/cubes/2021`；标签：`05_pseudo_labels/sam_refined/xinjiang_vvp_mmi_confidence_corrected/conservative_v1` |
| Split/manifest | `splits_2021_top5_region_holdout.csv`；SHA256 `59C2CF1D960194BE2BE46928FAEB06096461B2FAF071910090E01E7BD4A3972B` |
| Seed | 每样本以 sample_id CRC32 生成确定性空间抽样位置；模型 seed 不适用 |
| 输出 | `figures/split_distribution_summary.json`、两组 PNG/PDF 图 |
| Checkpoint | 不适用 |
| Metrics source | 全量标签像元计数；全部 1076 张 cube 的确定性 8x8 光谱/NDVI 抽样 |
| 失败/中断 | 首次分散随机像元读取在 304 秒超时；确认随机磁盘页访问为根因后改为连续 8x8 块，完整复跑 1076/1076 成功 |
| 结论 | Validation 有小到中等区域偏移、Test 更接近 Train；保留严格区域级划分 |

### 完整性与时间点

- cube/标签/manifest：1076/1076/1076，集合完全一致，缺失数均为 0。
- shape：cube `[26,10,256,256]`，标签和置信度 `[256,256]`。
- 标签只含 `0,1,255`；置信度只含 `0,0.4,0.7,0.8,1.0`（浮点表示有正常微小误差）。
- 时间日历版本数：1。
- 直接观测时间点：27762；双侧插值时间点：214；unresolved：0。
- 每样本插值数：863 个样本为 0，212 个为 1，1 个为 2。

### 区域划分

| Split | 区域 | 样本数 |
| --- | --- | ---: |
| Train | `r007_c021`, `r009_c021`, `r013_c017` | 495 |
| Validation | `r008_c021` | 276 |
| Test | `r008_c022` | 305 |

### 标签分布

| Split | 非玉米 | 玉米 | ignore | 有监督占比 | 玉米/有监督 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 6,959,759 | 12,652,070 | 12,828,491 | 0.60455 | 0.64512 |
| Validation | 4,877,280 | 6,691,022 | 6,519,634 | 0.63956 | 0.57839 |
| Test | 4,352,588 | 7,942,837 | 7,693,055 | 0.61513 | 0.64600 |
| Total | 16,189,627 | 27,285,929 | 27,041,180 | 0.61653 | 0.62761 |

- 所有样本均至少包含一个有监督像元。
- 无玉米正像元的完整样本：Train 9、Validation 5、Test 0。

### 光谱与时间分布

| 比较 | 平均标准化光谱偏移 | NDVI 曲线平均绝对差 | NDVI 最大差 | 峰值时间点 | 峰值 NDVI |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 0 | 0 | 0 | 18 | 0.804 |
| Validation vs Train | 0.332 | 0.038 | 0.098 | 17 | 0.756 |
| Test vs Train | 0.192 | 0.013 | 0.032 | 18 | 0.795 |

注意：标签分布为全像元精确统计；光谱/NDVI 为确定性空间抽样统计，不是全像元普查。

### 可视化与摘要

- `figures/fig_split_distribution_preview.png` / `.pdf`
- `figures/fig_split_representative_tiles.png` / `.pdf`
- 代表样本：Train `sample_001391`，Validation `sample_002230`，Test `sample_002074`
- `figures/split_distribution_summary.json` SHA256：`4C9E204BF07181F89CE326F8C887978303BAA4B0C1E6310630E2A388A9F75255`

## E0：TSViT baseline

| 字段 | 记录 |
| --- | --- |
| ID | E0 |
| 状态 | completed（epoch 25，warmup 后 patience=12 early stopping） |
| 日期 | 2026-07-17 |
| Git commit | 基础 HEAD `b0a3463b1cbf506402b6d7b3034b53c7e2116240`；分支 `experiment/e0-tsvit-xinjiang-2021`；运行快照记录 dirty 文件列表 |
| Config | `configs/models/tsvit_baseline.yaml`；正式 `24/patch2`；learned DOY；WPE off；最终 scaler scale=8192、growth_interval=1000000 |
| Command | `D:\Anaconda3\envs\cawa\python.exe scripts\train_e0.py --paths configs\paths.local.yaml --manifest manifests\xinjiang_2021_e0_e1.csv --normalization manifests\xinjiang_2021_train_normalization.json --config configs\models\tsvit_baseline.yaml --output-dir E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e0_tsvit_doy_seed42 --physical-batch-size 16 --resume` |
| Environment | `cawa`，Python 3.11.14，PyTorch 2.10.0+cu128，CUDA，RTX 5060 Ti；physical/effective batch 16/16 |
| Data version | 同 DATA-AUDIT-2021；仓库 manifest SHA256 `79DCAAF1270D48B99FA50AF2A57548B50D7FD3E232620D11DC53E5E64C1177A8` |
| Split | Train 495 / Validation 276；训练与重放过程不加载 Test 305 |
| Normalization | Train-only；SHA256 `1401DC9AAFE9A30A8916AB1A3C738080DB63D433C04716CA27FA5CABB51EC142` |
| Seed | 42 |
| Output | `06_models/retrain_outputs/maize_model_research/e0_tsvit_doy_seed42`（相对正式工作区） |
| Best checkpoint | epoch 13；`best.pt` SHA256 `CAB74C64897DA7FEA8A1A458ED94DAC2E23C0054A1772EAF16CF7BB5C3F9DE86` |
| Last checkpoint | epoch 25，global_step 83175，bad_epochs 12；`last.pt` SHA256 `7085320473FD7928A26B884EF4BFB095F56550A968C03FB606A2C8272A4D8B77` |
| Metrics source | 独立新进程从 best.pt 完整重放 Validation 276 张；确定性 24x24 滑窗 logits 重叠平均；伪标签一致性，不是独立地面真值精度 |
| Final Validation | loss 0.09189；OA 0.96300；maize precision 0.96283；recall 0.97362；F1 0.96819；IoU 0.93835；mIoU 0.92683；Kappa 0.92398；area ratio 1.01121 |
| Confusion matrix | `[[4625760,251520],[176506,6514516]]`；supervised pixels 11568302；evaluated tiles 276 |
| Validation replay | `validation_replay_best_epoch13_20260717/validation_replay.json`，SHA256 `9B4B128C9BD777D30280DF06AC847C8C9DEBDF14E3964E368D625C17D5B56A16`；逐项与 checkpoint epoch 13 记录完全一致 |
| AMP anomalies | 默认增长、65536、32768、16384 均暴露 scaled-gradient/FP16 边界问题；失败现场与迁移审计均保留。epoch 25 在仅 2 个监督像元 group 上以 16384 可复现全模型 NaN，8192 单变量完整重放并正式通过 |
| Conclusion | E0 正式运行 completed；仅依据 Validation 选择 epoch 13 best，未读取 Test。E1 启动前须冻结可审计的 AMP 同批降 scale 重试策略。 |

真实 CUDA smoke：63 项最终门禁前的 smoke 选择 physical batch 16；固定真实批次 loss `1.02078 -> 0.00639`；单张 Validation 整图拼接与 checkpoint logits 精确重载通过；Test 读取数为 0。

## E1：TSViT + learnable WPE

| 字段 | 记录 |
| --- | --- |
| ID | E1 |
| 状态 | completed；epoch 22 达到 patience=12 正常早停；进程正常退出，stderr 0 bytes |
| 日期 | 2026-07-18 |
| Git commit | 基础 HEAD `b0a3463b1cbf506402b6d7b3034b53c7e2116240`；分支 `experiment/e0-tsvit-xinjiang-2021`；代码与记录位于未提交工作树 |
| Config | `configs/models/tsvit_wpe_basic.yaml`；SHA256 `BAD5C5FB6821F0BC8F32F75C0FF0078DCD2B1E6787D1F9DF90D9D4A7F721229E`；正式 24/patch2；WPE 三基尺度初值 `[7,17.5,35]` 天、shift ±7 天、support radius 42 天 |
| AMP policy | init scale 8192、backoff 0.5、minimum 128、每 effective batch 最多 6 次降级、growth interval 1000000；同批重试且不跳 batch |
| Command | `D:\Anaconda3\envs\cawa\python.exe -u scripts\train_e0.py --config configs\models\tsvit_wpe_basic.yaml --output-dir E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e1_tsvit_doy_wpe_seed42 --physical-batch-size 16` |
| Environment | `cawa`，Python 3.11.14，PyTorch 2.10.0+cu128，CUDA，RTX 5060 Ti；physical/effective batch 16/16；3327 optimizer steps/epoch |
| Data version | 同 DATA-AUDIT-2021；manifest SHA256 `79DCAAF1270D48B99FA50AF2A57548B50D7FD3E232620D11DC53E5E64C1177A8` |
| Split | Train 495 / Validation 276；训练不加载 Test 305 |
| Normalization | Train-only；SHA256 `1401DC9AAFE9A30A8916AB1A3C738080DB63D433C04716CA27FA5CABB51EC142` |
| Seed | 42；从随机初始化训练，不加载 E0 权重 |
| Output | `06_models/retrain_outputs/maize_model_research/e1_tsvit_doy_wpe_seed42`（相对正式工作区） |
| CUDA smoke | passed；physical batch 16；1,707,531 parameters；峰值显存 1,609,693,696 bytes；固定真实批次 loss `1.02068 -> 0.00639`；checkpoint 精确重载；Test records loaded 0 |
| Verification | 71 项 pytest 全通过；compileall exit 0；git diff check exit 0；同批 backoff CUDA 回归通过 |
| Best checkpoint | epoch 8 / global step 26616；`best.pt` SHA256 `60B7A2C0715D45B20F5723DAD8C5ED8CB33E785AAD2A15AD39797A512E0CDDC8` |
| Terminal checkpoint | epoch 22 / global step 73194 / bad_epochs 12；`last.pt` SHA256 `AE5BE09108DCEA88EB07FB136EB0D6E8485CE54714408D0F4ED075B5FD6D1717` |
| Metrics source | 新进程从 `best.pt` 完整重放 Validation 276 张；重放结果与 checkpoint epoch 8 保存指标逐项一致；训练与重放均不加载 Test |
| Validation metrics | loss 0.10083；OA 0.96045；precision 0.97030；recall 0.96104；F1 0.96565；maize IoU 0.93357；mIoU 0.92226；Kappa 0.91905；area ratio 0.99045 |
| Confusion matrix | `[[4680450,196830],[260718,6430304]]`；supervised pixels 11568302；evaluated tiles 276 |
| Validation replay | `validation_replay_best_epoch8_20260718/validation_replay.json`；SHA256 `EE79C31EDAC4E785D763148C283469152C84B8FE5F9F8B6D892DB7706E67E051` |
| Learned WPE at best | scales `[7.22902,17.24530,35.00000]` 天；shifts `[-1.38885,1.02466,-0.40713]` 天；alpha `0.02265`；核为截断于 ±42 天并按绝对权重归一化的三基 Mexican-hat |
| AMP events | 4 次同批 backoff，均为 `wavelet.alpha` 非有限梯度：epoch 7/8/17/20；scale `8192→4096→2048→1024→512`。每个 epoch 仍为 3327 optimizer steps；`amp_events.jsonl` SHA256 `F791CE810ECBC80BC26133574E6A4E3946DA9B2760934037C528C7835752E684` |
| Training metrics file | 22 条 epoch 记录；`metrics.jsonl` SHA256 `86207D5D5346859D749F508244801E9DD46A8670FAD1E910ADFC8265160810F6` |
| Selection rule | 只依据完整 Validation maize IoU 保存 best 与 early stopping；达到终态后独立重放 best.pt；训练期间不读取 Test |
| Conclusion | completed negative result：E1 相对 E0 的 maize IoU/F1/Kappa 差值分别为 `-0.00478/-0.00255/-0.00493`；当前证据不支持 WPE 带来增益，保留 E0 为本组最佳模型 |

E1 模型仅相对 E0 启用已批准的 WPE。以上结论来自同一 split、seed 和训练计划下的独立 Validation 重放；它衡量与空间留出伪标签的一致性，不是独立地面真值精度，也没有使用 Test。

## E0/E1：冻结方案一次性 Test 评价

| 字段 | E0 | E1 |
| --- | --- | --- |
| 状态 | completed，冻结 epoch 13 best.pt 一次性评价 | completed，冻结 epoch 8 best.pt 一次性评价 |
| 日期 | 2026-07-19 | 2026-07-19 |
| Split | Test 305；region_r008_c022 | Test 305；region_r008_c022 |
| Checkpoint SHA256 | `CAB74C64897DA7FEA8A1A458ED94DAC2E23C0054A1772EAF16CF7BB5C3F9DE86` | `60B7A2C0715D45B20F5723DAD8C5ED8CB33E785AAD2A15AD39797A512E0CDDC8` |
| Output | `e0_tsvit_doy_seed42/test_evaluation` | `e1_tsvit_doy_wpe_seed42/test_evaluation` |
| Result JSON SHA256 | `C8A4F4A41DB32135224A6E0FF55BA81640DB68084A5C9B4D815AFD73A894089B` | `B4307DF7CE972E271BB38BB0663D78C926CE726F838FC6D85B66EFAB7A0A6E95` |
| Run snapshot SHA256 | `2C77964BF4935C53FE2EA1044E439895823FF6DA6670C81A221BE3454D586F82` | `A5B52424C701B374D32B2F8054262A925BD94B6C27A58D3067CF89DE7E4B98A8` |

正式命令均为独立新进程调用 `scripts/evaluate_test.py`，只解析 Test records，并在运行前严格验证 checkpoint 路径/SHA256/epoch、嵌入配置、manifest 与 normalization 哈希。两次运行均使用 24x24 window、stride 24、window batch 16、完整 256x256 logits 拼接、`ignore_index=255` 和 CUDA AMP；没有阈值搜索、checkpoint 选择、模型修改或重训练。

| 指标 | E0 Test | E1 Test | E1-E0 Test | E0 Validation | E1 Validation | E1-E0 Validation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loss | 0.12384 | 0.12774 | 0.00390 | 0.09189 | 0.10083 | 0.00893 |
| OA | 0.95171 | 0.94824 | -0.00347 | 0.96300 | 0.96045 | -0.00255 |
| maize precision | 0.96847 | 0.97607 | 0.00760 | 0.96283 | 0.97030 | 0.00747 |
| maize recall | 0.95639 | 0.94299 | -0.01339 | 0.97362 | 0.96103 | -0.01259 |
| maize F1 | 0.96239 | 0.95925 | -0.00314 | 0.96819 | 0.96564 | -0.00255 |
| maize IoU | 0.92751 | 0.92169 | -0.00582 | 0.93835 | 0.93357 | -0.00478 |
| mIoU | 0.90058 | 0.89462 | -0.00595 | 0.92683 | 0.92226 | -0.00457 |
| macro-F1 | 0.94748 | 0.94417 | -0.00331 | 0.96199 | 0.95952 | -0.00246 |
| Kappa | 0.89496 | 0.88839 | -0.00657 | 0.92398 | 0.91905 | -0.00493 |
| area ratio | 0.98752 | 0.96611 | -0.02142 | 1.01121 | 0.99045 | -0.02076 |

- E0 confusion matrix：`[[4105268,247320],[346410,7596427]]`。
- E1 confusion matrix：`[[4168976,183612],[452803,7490034]]`。
- 两者 supervised pixels 均为 12295425，evaluated tiles 均为 305；每个混淆矩阵元素之和均严格等于 supervised pixels。
- Test 与 Validation 的主要差值方向一致：E1 precision 更高，但 recall、F1、IoU、mIoU、Kappa 和 OA 更低。Test 不参与模型选择；它没有改变 D-021 基于 Validation 保留 E0 的结论。
- 所有 Test 指标衡量与空间留出伪标签的一致性，不是独立地面真值精度。
## 暂缓实验

| ID | 状态 | 目的 |
| --- | --- | --- |
| E2 | not run / deferred | WPE-only，检验绝对 DOY 的互补性 |
| E3 | not run / deferred | source-derived quality 的质量感知扩展 |

## E0/E1：冻结 Test 的独立 30m 参考标签双口径评价

真实标签根目录为正式工作区相对路径 `03_processed_data/labels_30m/xinjiang_2021/2021`；每个 Test 样本读取 `y_patch_30m.npy`（85x85，标签仅为 0/1/255）。305 个 Test 记录全部有对应参考标签。所有统计仅使用 `label != 255`，不使用耕地掩膜、排除列表或 Test 驱动阈值搜索。

- 原生 30m：将完整 256x256 拼接概率图裁至 255x255 后，固定 3x3 平均至 85x85；以两类概率 argmax（等价于 0.5 决策，精确平局归非玉米）计算 NLL/loss 与指标。
- 标签复制模型网格：将每个 30m 标签复制到其 3x3 支持域，形成 255x255 监督网格；模型最右/最下 1 像元不计入；直接在该网格计算 NLL/loss 与指标。此口径不声称新增 10m 真值。
- E0/E1 各在独立新进程中以冻结 best.pt 完整拼接一次，同时生成两种尺度；未更改模型、checkpoint、early stopping、训练、阈值或任何超参数。

| 指标 | E0 原生30m | E1 原生30m | E1-E0 | E0 标签复制网格 | E1 标签复制网格 | E1-E0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loss | 0.38178 | 0.39930 | 0.01751 | 0.43459 | 0.47072 | 0.03613 |
| OA | 0.89218 | 0.88416 | -0.00802 | 0.88806 | 0.87965 | -0.00841 |
| maize precision | 0.89171 | 0.88527 | -0.00644 | 0.88788 | 0.88414 | -0.00374 |
| maize recall | 0.86784 | 0.85566 | -0.01218 | 0.86225 | 0.84565 | -0.01660 |
| maize F1 | 0.87961 | 0.87021 | -0.00940 | 0.87488 | 0.86447 | -0.01041 |
| maize IoU | 0.78510 | 0.77025 | -0.01485 | 0.77759 | 0.76129 | -0.01630 |
| mIoU | 0.80361 | 0.79043 | -0.01318 | 0.79684 | 0.78299 | -0.01385 |
| macro-F1 | 0.89099 | 0.88281 | -0.00819 | 0.88681 | 0.87812 | -0.00868 |
| Kappa | 0.78202 | 0.76567 | -0.01636 | 0.77365 | 0.75634 | -0.01731 |
| area ratio | 0.97323 | 0.96656 | -0.00667 | 0.97113 | 0.95646 | -0.01467 |

- 原生30m supervised pixels：两模型均为 2,203,625；E0 confusion matrix `[[1098066,105406],[132182,867971]]`，E1 `[[1092559,110913],[144359,855794]]`。
- 标签复制网格 supervised pixels：两模型均为 19,832,625；E0 confusion matrix `[[9851175,980073],[1239941,7761436]]`，E1 `[[9833795,997453],[1389390,7611987]]`。
- 两个尺度的结果方向一致：E1 的 precision、recall、F1、IoU、mIoU、Kappa 和 OA 均低于 E0；这不改变先前仅凭 Validation 保留 E0 的选择，且没有依据 Test 结果做任何调整。
- 此处是独立年度30m参考标签的最终空间留出评价。此前 `test_evaluation` 的指标继续保留，但只解释为与训练监督伪标签的一致性诊断，不能作为独立真值精度。

| E0 输出 | SHA256 | E1 输出 | SHA256 |
| --- | --- | --- | --- |
| `test_evaluation_ground_truth/native30m/test_evaluation.json` | `0A4A5F31DE61021A1AAED35884B31CB4AA41031417DE1ECD52DD0E3F8A040262` | `test_evaluation_ground_truth/native30m/test_evaluation.json` | `C291AD6E0B395D861CB7CC44C88C196417AB1D75229F32C0B74C312970B1B81B` |
| `test_evaluation_ground_truth/upsampled10m/test_evaluation.json` | `29CF7830BBCF76A289834A46C5E6AA4B5D07BB8832D4F6C9178461A5DAACB73A` | `test_evaluation_ground_truth/upsampled10m/test_evaluation.json` | `1D497BBF5E7DF3B3533839884D99F0B59498177DCA32F92E4C9F4884675BB2FE` |
| `test_evaluation_ground_truth/run_snapshot.json` | `14DA6BFAF7F9B625E6DD20F7A5BA6EF5EEF348398786058E0A9D5ABD6133F645` | `test_evaluation_ground_truth/run_snapshot.json` | `B792B8942890DDF3E35A230E4005B526C41908E38A91630C2C64310F554D9A52` |
## 补充对照：原始 SAM+指数伪标签、历史 PEACENET 与 E0/E1 的独立30m Test

本节是对已保存预测/伪标签的只读后验汇总，不重新训练或重新推理 E0/E1，也不以 Test 改变 checkpoint、阈值或模型选择。所有方法使用同一 `xinjiang_2021_e0_e1.csv` 的 305 条 Test records（逐项与历史输出核对完整重叠）、同一年度30m参考标签 `03_processed_data/labels_30m/xinjiang_2021/2021/<sample_id>/y_patch_30m.npy`，仅评价 `label != 255` 的 2,203,625 个原生30m cells；不使用耕地掩膜或排除列表。

- 原始伪标签为 `05_pseudo_labels/sam_refined/xinjiang_vvp_otsu_mmi_otsu_auto_neg_clean/vvp_mmi_intersection`，即 SAM 与 VVP/MMI Otsu 约束；其与历史 EXP001 的 `sam_vvp_mmi_intersection` 像元采样清单 434,754 条标签逐项 100% 一致。按固定规则将左上 `255x255` 的 10m 伪标签按 3x3 聚合；正样本比例 `>=0.5` 判为正，ignore 像元按非正处理。
- 历史 PEACENET EXP002 为 `corrected_raw_soft_update`，使用 `conservative_v1` 监督来源和已保存的固定阈值 `0.5` 输出。该输出原始报告曾加耕地掩膜；此处只读重算为上方统一支持域，因而才可同 E0/E1 直接比较。
- E0/E1 是本文件上一节已冻结的原生30m结果；其模型选择仍只依据 Validation，以下 Test 对照不反向改变 D-021/D-023。

| 方法 | OA | precision | recall | F1 | maize IoU | Kappa | area ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始 SAM+VVP/MMI Otsu 伪标签 | 0.86308 | 0.89823 | 0.78756 | 0.83926 | 0.72304 | 0.72091 | 0.87680 |
| 历史 PEACENET EXP002（`conservative_v1`，0.5） | 0.88227 | 0.91372 | 0.81784 | 0.86312 | 0.75921 | 0.76040 | 0.89507 |
| E0 TSViT+DOY | 0.89218 | 0.89171 | 0.86784 | 0.87961 | 0.78510 | 0.78202 | 0.97323 |
| E1 TSViT+DOY+WPE | 0.88416 | 0.88527 | 0.85566 | 0.87021 | 0.77025 | 0.76567 | 0.96656 |

原始伪标签、EXP002、E0、E1 的原生30m confusion matrices（行是真实 0/1，列是预测 0/1）依次为 `[[1114224,89248],[212468,787685]]`、`[[1126230,77242],[182186,817967]]`、`[[1098066,105406],[132182,867971]]`、`[[1092559,110913],[144359,855794]]`，均和 2,203,625 个监督 cells 守恒。

结论：E0 相对原始 SAM+指数伪标签的 F1/IoU/Kappa 分别提高 `0.04035/0.06205/0.06111`（+4.03/+6.21/+6.11 pp）；E1 对应提高 `0.03095/0.04720/0.04476`。E0 相对历史 EXP002 仍提高 F1/IoU/Kappa `0.01649/0.02589/0.02162`，E1 也分别提高 `0.00709/0.01104/0.00527`。两模型的 precision 低于旧基线，但 recall 的提升足以使 OA、F1、IoU 和 Kappa 提升；当前证据支持 E0 提取到超出原始伪标签与旧 PEACENET 的空间泛化信息，E1 虽高于两条基线但低于 E0。


## E2-W：learned P_T + five-point Mexican-hat WPE

| 字段 | 记录 |
| --- | --- |
| ID | E2W |
| 状态 | completed；epoch 25 达到 patience=12 正常早停；best 为 epoch 13；stderr 0 bytes |
| 日期 | 2026-07-19 |
| Git | 分支 `codex/e2w-pt-mexican-hat`；基础 HEAD `ae7ccffaab8bf0d5f18d63b58c1657686f91e29c`；代码与记录未提交 |
| Config | `configs/models/tsvit_e2w_pt_mexican_hat_k5.yaml`；SHA256 `71E5FE24D29738A64D5138E7B796D573AED868D0A09EA8EC9D001364B9BFB247` |
| Architecture | learned temporal slot table `P_T=[1,26,128]`；content-only five-point Mexican-hat WPE，offsets `[-2,-1,0,1,2]`，三基 scale init `[0.75,1.0,1.25]`、bounds `[0.5,1.5]`，shift bounds `[-1,1]` |
| Controlled settings | Xinjiang 2021；Train 495 / Validation 276 / Test 305；24x24/patch2；seed42；physical/effective batch16/16；训练计划、标签、损失、优化器、manifest 与 normalization 沿用 E0/E1 |
| Parameters | 1,662,603 |
| Best checkpoint | epoch 13 / global step 43251；`best.pt` SHA256 `A74C8A33030172E94020410D1E46FB3439C4438ECC1E796F06CEA347DF859428` |
| Terminal checkpoint | epoch 25 / global step 83175 / bad_epochs 12；`last.pt` SHA256 `84054E53669B05D66FD23D48BCF3D0959591E5A7F65B933907CAFC2C8B7F36DA` |
| Training records | `metrics.jsonl` SHA256 `C2C38CDAA6142B926F7A4C462CD30B1A1DBE60DCFC6A03D8F2A273AA2C13460F`；`amp_events.jsonl` SHA256 `617D97EB84F044FAF1513491B314DA07E58D8DCA93CAAE3E8F340ABB0017DFE5`；`run_snapshot.json` SHA256 `3141F5115D2B6F7A0A9064454BEA195B14A13A03FFA4E97BC97EE3FE189A39C0` |
| AMP audit | 6 次同批 backoff，均为 `wavelet.gamma` Inf：epoch 1/17/21/23，scale `8192→4096→2048→1024→512→256→128`；同一失败 batch 成功后才推进，epoch optimizer steps 保持 3327 |
| Learned wavelet at best | scales `[0.824393,1.099981,1.321753]`；shifts `[0.245128,0.221329,0.201444]`；gamma `-0.097273`；sigmoid gates min/mean/max `0.291193/0.491921/0.701593` |

### 冻结 Validation 重放

Test 首次读取前，独立新进程从冻结 `best.pt` 完整重放 Validation 276；结果与 checkpoint epoch 13 记录逐项一致。

| 指标 | E2-W Validation | E0 Validation | E2-W−E0 |
| --- | ---: | ---: | ---: |
| loss | 0.095201 | 0.091894 | +0.003307 |
| OA | 0.962346 | 0.963000 | -0.000654 |
| precision | 0.968054 | 0.962826 | +0.005228 |
| recall | 0.966803 | 0.973620 | -0.006817 |
| F1 | 0.967428 | 0.968193 | -0.000765 |
| maize IoU | 0.936912 | 0.938347 | -0.001436 |
| mIoU | 0.925745 | 0.926827 | -0.001081 |
| Kappa | 0.922812 | 0.923977 | -0.001164 |

E2-W Validation confusion matrix 为 `[[4663806,213474],[222120,6468902]]`，supervised pixels `11568302`，evaluated tiles `276`。重放文件 `validation_replay_best_epoch13_20260719/validation_replay.json` SHA256 为 `FF8D39F054EC96AF351EA4EEBECE6CAEF4908FA2EEDCD60A1578E248F478E05F`。

### 冻结 Test 一次性评价

方案和 checkpoint 在 Test 前冻结。伪标签一致性 Test 与独立 30m Test 各只启动一个新进程；没有阈值搜索、模型修改、checkpoint 选择或重训练。

| 指标 | E2-W 伪标签 Test | E2-W 原生30m Test | E2-W 标签复制网格 |
| --- | ---: | ---: | ---: |
| loss | 0.131462 | 0.352727 | 0.392375 |
| OA | 0.949551 | 0.891294 | 0.887551 |
| precision | 0.972903 | 0.900506 | 0.896463 |
| recall | 0.948318 | 0.854949 | 0.850467 |
| F1 | 0.960453 | 0.877136 | 0.872859 |
| maize IoU | 0.923915 | 0.781160 | 0.774401 |
| mIoU | 0.896843 | 0.801763 | 0.795630 |
| Kappa | 0.890835 | 0.779780 | 0.772183 |
| area ratio | 0.974731 | 0.949410 | 0.948692 |

- 伪标签 Test：305 tiles / 12,295,425 supervised pixels；confusion matrix `[[4142798,209790],[410501,7532336]]`；相对 E0 的 IoU/F1/Kappa 为 `-0.003592/-0.001937/-0.004126`。
- 原生30m：305 tiles / 2,203,625 supervised cells；confusion matrix `[[1108997,94475],[145073,855080]]`；相对 E0 的 IoU/F1/Kappa 为 `-0.003937/-0.002476/-0.002242`。
- 标签复制网格：305 tiles / 19,832,625 supervised pixels；confusion matrix `[[9947087,884161],[1346005,7655372]]`；相对 E0 的 IoU/F1/Kappa 为 `-0.003185/-0.002019/-0.001468`。

| 输出 | SHA256 |
| --- | --- |
| `test_evaluation/test_evaluation.json` | `D4586E36003DDD8629E2F23F6165B02754A81F08C050865484ABBD3B1F4F850E` |
| `test_evaluation/run_snapshot.json` | `1458A3989FBDCCBFF9410A3650B24F4406F58DA355664BAFA68C046196390F3D` |
| `test_evaluation_ground_truth/native30m/test_evaluation.json` | `CC35CF6B812E8C2DA1C940B323961BE50F03F5C160BD55BFA36B7462BCC9F211` |
| `test_evaluation_ground_truth/upsampled10m/test_evaluation.json` | `4D8DD8BEEF70192DF4A8736F4CFEA2C80FFCD53FC2D258CB52452A7C905569AD` |
| `test_evaluation_ground_truth/run_snapshot.json` | `BA9496D1BA971692AE9C13D1B9737D4C837B7332015CDBD0B65B2C44DB32A916` |

结论：E2-W 比 E1 更接近 E0，但在冻结 Validation、伪标签 Test 和独立原生30m Test 的 maize IoU/F1/Kappa 上均未超过 E0，因此继续保留 E0 为当前最佳模型。E2-W 同时将 DOY lookup 替换为 learned `P_T` 并加入 five-point WPE，结果不能解释为“单独增加 WPE”的效应。

### 五方法独立原生30m Test 对比图

为后续论文绘图生成了 `figures/fig_native30m_test_comparison_table.png` 与 `.pdf`，其可复现脚本为 `figures/gen_fig_native30m_test_comparison_table.py`。该图仅汇总同一 305 Test tiles、2,203,625 个 `label !=255` 原生30m参考像元上的 F1、maize IoU、Kappa、Precision、Recall 与 OA；每列最高值加粗，不含 area ratio。图件复用规则、生成命令、验证步骤和不得混入伪标签/上采样10m结果的边界，见 `docs/FIGURES.md`。
