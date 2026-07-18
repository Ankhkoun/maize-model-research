# Tasks

更新时间：2026-07-18。仅勾选有验证证据的工作；E0/E1 正式 Train/Validation 实验均 completed。

## P0：数据与实验启动核验

- [x] 核验 `configs/paths.local.yaml` 指向 `E:\maize_paper_workspace`，并读取正式工作区固定规则。
- [x] 核验 1076 个 Xinjiang 2021 cube、伪标签、置信度和区域划分记录一一对应。
- [x] 确认当前标签语义为二分类：`0=非玉米`、`1=玉米`、`255=ignore`。
- [x] 记录波段顺序、26 个时间点日历和插值时间点分布。
- [x] 核验现有区域级划分：Train 495、Validation 276、Test 305。
- [x] 完成全量标签分布统计和确定性光谱/NDVI 抽样诊断，并生成 PNG/PDF 预览。
- [x] 将当前区域划分转换为本仓库内、相对于 `workspace_root` 的正式 manifest，并记录数据版本与哈希。
- [x] 仅使用 Train 区域计算并冻结 10 波段归一化参数。
- [ ] 确认可用于这 1076 个样本的独立参考评价资产；当前未找到一一对应的独立地面真值。
- [x] 明确 24x24 窗口在边缘的 padding/滑窗拼接规则，并测试完整 256x256 输出。

## P1：E0 TSViT baseline

- [x] 选择并核验 `cawa` 环境：Python 3.11.14、PyTorch 2.10.0+cu128、CUDA 可用。
- [x] 实现 temporal-spatial Transformer 直接分割模型核心和 patch-to-pixel 输出头。
- [x] 验证输出形状、前向/反向有限性和固定配置可运行性。
- [x] 对照 Exact 确认：Exact 的训练窗口为 24x24，内部 Transformer patch 为 2x2。
- [x] 将当前标量 `Linear(1,D)` DOY 编码替换为 E0/E1 共享的可学习 DOY 表。
- [x] 将正式配置从工程 smoke 设置 `256/patch8` 改为 `24/patch2`。
- [x] 实现 Dataset/DataLoader、标准交叉熵 `ignore_index=255`、训练循环、AMP、checkpoint 和恢复。
- [x] 实现完整影像滑窗验证/推理及每区域指标。
- [x] 冻结项目依赖清单。
- [x] 验证单批次过拟合和小规模训练 smoke test。

## P2：E1 WPE

- [x] 实现三个全局可学习 Mexican-hat 小波基。
- [x] 约束尺度、中心偏移和时间支持范围，保持可解释与有限。
- [x] 在 DOY 编码之后、temporal class token 之前仅注入一次 WPE。
- [x] 验证 `alpha=0`/关闭 WPE 时回到 E0 路径。
- [x] 验证 padding、全无效输入、class token 隔离、边界参数梯度和 CUDA 前反向有限性。
- [x] 在正式 `24/patch2` 配置和 Dataset 上重新运行 E0/E1 等价性及显存 smoke test。

## P3：受控实验

- [x] 运行 E0 小规模训练 smoke test。
- [x] 运行 E1 小规模训练 smoke test。
- [x] 用户审阅 smoke 结果后，使用同一 manifest、seed 和训练计划运行正式 E0。
- [x] E0 在 epoch 25 达到批准的 patience=12 early stopping；独立重放 best.pt 的完整 Validation 并核对混淆矩阵。
- [x] 以 TDD 实现可审计的 AMP 同批降 scale 重试并冻结 E1 训练策略。
- [x] 在完全相同数据、主体模型、seed 和训练计划下运行正式 E1；epoch 22 正常早停，并独立重放 epoch 8 `best.pt` 的完整 Validation。
- [x] E0/E1 训练与 checkpoint 选择只依据 Validation；两次正式训练和独立重放均未读取 Test。
- [ ] 方案冻结后只评价 Test 一次；不得依据 Test 反向选择模型、阈值或 checkpoint。
- [x] 登记 DATA-AUDIT-2021 与正式 E0 的 commit、配置、数据版本、seed、命令、输出、checkpoint、指标和异常。
- [ ] 超参数冻结后，评估是否用 Train+Validation 共 771 张影像进行最终重训练。

## P4：文档与发布

- [x] 创建 2026-07-16 中间检查点，明确正式实验未完成。
- [ ] 用户单独确认后再 stage/commit/push；当前不得自动发布。
