# E0 正式训练设计

## 目标与范围

在 Xinjiang 2021 五区域空间留出数据上，实现并运行 E0（TSViT + 可学习 DOY）正式训练。E0 使用 `24x24` 训练窗口、模型内部 `2x2` patch 和直接像元级二分类监督。本阶段只训练 E0；不训练 E1，不使用 Test 选择模型、阈值、checkpoint 或训练轮数。

E0 与后续 E1 必须共享 manifest、归一化参数、窗口规则、增强、损失、优化器、学习率计划、seed 和评价代码。E1 后续唯一允许增加的结构因素是 WPE。

## 数据版本与空间划分

- 工作区根目录：`E:\maize_paper_workspace`
- cube：`03_processed_data/cubes/xinjiang_2021/cubes/2021`
- 标签：`05_pseudo_labels/sam_refined/xinjiang_vvp_mmi_confidence_corrected/conservative_v1`
- 源 split manifest：`01_code/spring_maize_paper_dataset/generated/splits_2021_top5_region_holdout.csv`
- 源 manifest SHA256：`59C2CF1D960194BE2BE46928FAEB06096461B2FAF071910090E01E7BD4A3972B`
- Train：495 张，区域 `region_r007_c021`、`region_r009_c021`、`region_r013_c017`
- Validation：276 张，区域 `region_r008_c021`
- Test：305 张，区域 `region_r008_c022`

仓库内正式 manifest 只保存相对于 `workspace_root` 的路径，并记录源 manifest 哈希。窗口必须继承完整影像的 split，不得重新随机跨 split 分配。

## 数据契约与归一化

每个样本包含：

- cube：`float32 [26,10,256,256]`
- 标签：`[256,256]`，值域仅为 `0=非玉米`、`1=玉米`、`255=ignore`
- 波段顺序：`B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12`
- DOY：26 个有序时间点
- valid mask：标识有效时间点

正式代码必须从 manifest 和数据元信息验证这些条件。任何缺失文件、shape、dtype、标签值域、DOY 或 split 隔离错误都立即终止，不静默丢弃样本。

十个波段的均值和标准差只使用 495 张 Train 影像的有效时间点统计。统计结果冻结到可追溯文件，E0 和 E1 共用；Validation 和 Test 不参与统计。

## 确定性窗口与整图恢复

Dataset 每次连续读取一张完整 `256x256` cube、标签、DOY 和有效时相掩码，再在内存中生成窗口。

- 窗口大小：`24x24`
- 常规步长：24
- 每个轴的起点：`0,24,48,...,216,232`
- 每张完整影像：`11x11=121` 个窗口
- 边缘规则：最后一个窗口锚定到坐标 232，因此完整覆盖 0–255；最后两个窗口有 8 像元重叠
- Train：跳过标签全为 255 的窗口
- Validation：保留所有窗口，在完整影像坐标中累加 logits 和覆盖计数，重叠位置取平均

不预先生成窗口文件。完整影像按确定性顺序读取，训练时只打乱影像顺序和影像内窗口顺序，以减少随机磁盘访问并避免新增大量重复数据。

## 数据增强

训练窗口仅使用空间增强：

- 随机水平翻转
- 随机垂直翻转
- 随机 `0/90/180/270` 度旋转

增强必须同步作用于图像和标签。Validation 不使用随机增强。不使用光谱扰动、时间抖动、MixUp 或会改变物候语义的增强。

## E0 模型

- `image_size=24`
- `patch_size=2`
- 每个窗口形成 `12x12=144` 个空间 token
- `num_frames=26`
- `num_channels=10`
- `num_classes=2`
- `dim=128`
- temporal depth 4
- spatial depth 4
- heads 4
- dropout 0
- WPE disabled

当前标量 `Linear(1,D)` DOY 编码替换为共享可学习查找表：索引 0 专用于 padding，索引 1–366 对应真实 DOY。无效时间点使用索引 0，并继续由 valid mask 屏蔽。分割头输出 `[B,2,24,24]` logits。

## 损失、优化和学习率

- 损失：普通像元级交叉熵，`ignore_index=255`
- `pseudo_confidence`：只用于数据审计，不进入首轮 E0 loss
- 优化器：AdamW
- weight decay：0
- seed：42
- 最大训练轮数：100 epochs
- warmup：前 10 epochs，学习率从 `1e-8` 线性增加至 `1e-3`
- 主计划：warmup 后使用 cosine schedule，最终学习率 `5e-6`
- AMP：CUDA FP16 autocast + GradScaler
- GradScaler：最终 init_scale=8192、growth_interval=1000000；E0 仍保留非有限梯度立即终止门禁。该固定 scale 策略仅记录 E0 实际完成状态，E1 前须另行冻结可审计的同批降 scale 重试。

GPU 物理窗口 batch size 按 `16,8,4,2` 顺序做显存 smoke test，选择第一个可完整前反向运行的值。若物理 batch 小于 16，使用梯度累积保持有效 batch size 16。学习率计划按 optimizer step 更新。

## 验证、checkpoint 和早停

每个 epoch 后只在完整 Validation 影像上运行确定性滑窗推理。Test 在训练、checkpoint 选择和超参数冻结期间保持未读。

记录以下 Validation 指标：

- loss
- maize IoU、Precision、Recall 和 F1
- mIoU 和 macro-F1
- Kappa
- area ratio
- 混淆矩阵
- per-region 指标

checkpoint 规则：

- `last.pt`：每个 epoch 更新，包含模型、优化器、scheduler、GradScaler、epoch、最佳指标、无提升计数和 Python/NumPy/PyTorch/CUDA 随机状态
- `best.pt`：Validation maize IoU 严格提高时更新
- 恢复训练：从 `last.pt` 恢复全部状态，不重置 scheduler 或早停计数
- 早停：warmup 结束后启用；Validation maize IoU 连续 12 个有效 epoch 未提高则停止

正式输出目录：

`E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e0_tsvit_doy_seed42`

输出包括配置快照、仓库 commit/dirty 状态、manifest 及哈希、归一化统计、训练日志、每 epoch 验证指标、`last.pt`、`best.pt` 和最终复算摘要。

## 实现前验证门槛

正式训练前必须依次通过：

1. manifest、路径、shape、dtype、波段、DOY、标签值域和 split 隔离测试。
2. 窗口坐标、完整覆盖、末端锚定、重叠平均和全 ignore 训练窗口过滤测试。
3. E0 `24/patch2` 配置、`[B,2,24,24]` 输出和 DOY 查找表测试。
4. 少量真实窗口单批次过拟合，loss 明显下降且梯度有限。
5. GPU 短训练/验证 smoke，验证 AMP、显存 batch 选择、指标、checkpoint 保存与恢复。
6. 全量自动化测试通过。

只有这些门槛全部通过，才能启动正式 E0。

## 异常处理

- CUDA OOM：清理失败步骤状态，降低物理 batch，并增加梯度累积以保持有效 batch 16。
- NaN/Inf：立即停止，保存故障 checkpoint、样本 ID、窗口坐标和诊断日志；不得跳过后继续。
- 数据契约错误：立即停止并报告具体样本和字段。
- 进程中断：从 `last.pt` 恢复。
- Validation 无提升：按 patience 12 早停并保留 `best.pt`。

## 完成判据

E0 正式训练完成必须同时满足：

- 训练达到 100 epochs 或按批准规则早停，且进程正常退出；
- `best.pt` 可在新进程重新加载；
- 完整 Validation 指标独立复算与训练记录一致；
- 没有使用 Test 进行任何选择；
- `docs/EXPERIMENTS.md`、`docs/STATUS.md`、`docs/TASKS.md` 和日期化 handoff 已更新；
- 正式工作区交接、实验说明和文件库存已按固定流程刷新。

