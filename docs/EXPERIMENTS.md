# Experiment Registry

更新时间：2026-07-17。完整 checkpoint、概率图和正式推理结果保存在正式工作区，不提交 Git。

## 已完成参考基线：Xinjiang 2021 PEACE-Net

结果根目录：`06_models/retrain_outputs/peacenet_xinjiang_confidence_ablation_50ep_es`（相对于正式工作区）。

| ID | 置信度方案 | 权重 | Soft update | Val F1 @ 0.5 | Test F1 @ 0.5 | 状态 |
| --- | --- | --- | --- | ---: | ---: | --- |
| EXP001 | 原始 SAM raw | raw | 开 | 0.878807 | 0.874970 | completed |
| EXP002 | 保守 VVP/MMI | raw | 开 | 0.880629 | 0.876772 | completed，正式推荐 |
| EXP003 | 保守 VVP/MMI | class-balanced | 开 | 0.879503 | 0.875353 | completed |
| EXP004 | 保守 VVP/MMI | class-balanced | 关 | 0.871252 | 0.870863 | completed |

EXP002 根据 Validation 选择。其 Validation 导向阈值为 0.13，对应 Val F1=0.883803、Test F1=0.885334、Test IoU=0.794259、Test Kappa=0.692265。以上是既有实验记录，本检查点未重跑。

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
| Train | 6,959,759 | 12,652,070 | 12,828,491 | 0.604551 | 0.645124 |
| Validation | 4,877,280 | 6,691,022 | 6,519,634 | 0.639559 | 0.578393 |
| Test | 4,352,588 | 7,942,837 | 7,693,055 | 0.615126 | 0.645999 |
| Total | 16,189,627 | 27,285,929 | 27,041,180 | 0.616533 | 0.627611 |

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
| Final Validation | loss 0.091894；OA 0.963000；maize precision 0.962826；recall 0.973620；F1 0.968193；IoU 0.938347；mIoU 0.926827；Kappa 0.923977；area ratio 1.011211 |
| Confusion matrix | `[[4625760,251520],[176506,6514516]]`；supervised pixels 11568302；evaluated tiles 276 |
| Validation replay | `validation_replay_best_epoch13_20260717/validation_replay.json`，SHA256 `9B4B128C9BD777D30280DF06AC847C8C9DEBDF14E3964E368D625C17D5B56A16`；逐项与 checkpoint epoch 13 记录完全一致 |
| AMP anomalies | 默认增长、65536、32768、16384 均暴露 scaled-gradient/FP16 边界问题；失败现场与迁移审计均保留。epoch 25 在仅 2 个监督像元 group 上以 16384 可复现全模型 NaN，8192 单变量完整重放并正式通过 |
| Conclusion | E0 正式运行 completed；仅依据 Validation 选择 epoch 13 best，未读取 Test。E1 启动前须冻结可审计的 AMP 同批降 scale 重试策略。 |

真实 CUDA smoke：63 项最终门禁前的 smoke 选择 physical batch 16；固定真实批次 loss `1.020775 -> 0.006387`；单张 Validation 整图拼接与 checkpoint logits 精确重载通过；Test 读取数为 0。

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
| CUDA smoke | passed；physical batch 16；1,707,531 parameters；峰值显存 1,609,693,696 bytes；固定真实批次 loss `1.020681 -> 0.006386`；checkpoint 精确重载；Test records loaded 0 |
| Verification | 71 项 pytest 全通过；compileall exit 0；git diff check exit 0；同批 backoff CUDA 回归通过 |
| Best checkpoint | epoch 8 / global step 26616；`best.pt` SHA256 `60B7A2C0715D45B20F5723DAD8C5ED8CB33E785AAD2A15AD39797A512E0CDDC8` |
| Terminal checkpoint | epoch 22 / global step 73194 / bad_epochs 12；`last.pt` SHA256 `AE5BE09108DCEA88EB07FB136EB0D6E8485CE54714408D0F4ED075B5FD6D1717` |
| Metrics source | 新进程从 `best.pt` 完整重放 Validation 276 张；重放结果与 checkpoint epoch 8 保存指标逐项一致；训练与重放均不加载 Test |
| Validation metrics | loss 0.100828；OA 0.960448；precision 0.970299；recall 0.961035；F1 0.965645；maize IoU 0.933572；mIoU 0.922260；Kappa 0.919046；area ratio 0.990452 |
| Confusion matrix | `[[4680450,196830],[260718,6430304]]`；supervised pixels 11568302；evaluated tiles 276 |
| Validation replay | `validation_replay_best_epoch8_20260718/validation_replay.json`；SHA256 `EE79C31EDAC4E785D763148C283469152C84B8FE5F9F8B6D892DB7706E67E051` |
| Learned WPE at best | scales `[7.229018,17.245300,35.000000]` 天；shifts `[-1.388848,1.024661,-0.407129]` 天；alpha `0.022648`；核为截断于 ±42 天并按绝对权重归一化的三基 Mexican-hat |
| AMP events | 4 次同批 backoff，均为 `wavelet.alpha` 非有限梯度：epoch 7/8/17/20；scale `8192→4096→2048→1024→512`。每个 epoch 仍为 3327 optimizer steps；`amp_events.jsonl` SHA256 `F791CE810ECBC80BC26133574E6A4E3946DA9B2760934037C528C7835752E684` |
| Training metrics file | 22 条 epoch 记录；`metrics.jsonl` SHA256 `86207D5D5346859D749F508244801E9DD46A8670FAD1E910ADFC8265160810F6` |
| Selection rule | 只依据完整 Validation maize IoU 保存 best 与 early stopping；达到终态后独立重放 best.pt；训练期间不读取 Test |
| Conclusion | completed negative result：E1 相对 E0 的 maize IoU/F1/Kappa 差值分别为 `-0.004776/-0.002548/-0.004930`；当前证据不支持 WPE 带来增益，保留 E0 为本组最佳模型 |

E1 模型仅相对 E0 启用已批准的 WPE。以上结论来自同一 split、seed 和训练计划下的独立 Validation 重放；它衡量与空间留出伪标签的一致性，不是独立地面真值精度，也没有使用 Test。

## 暂缓实验

| ID | 状态 | 目的 |
| --- | --- | --- |
| E2 | not run / deferred | WPE-only，检验绝对 DOY 的互补性 |
| E3 | not run / deferred | source-derived quality 的质量感知扩展 |
