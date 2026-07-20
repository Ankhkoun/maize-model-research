# E0/E1 冻结 Test-only 评价设计

## 目标

在不改变 E0/E1 模型、checkpoint、阈值、训练计划或 Validation 选择结论的前提下，为 Xinjiang 2021 的冻结 Test split 增加一次性正式评价入口。E0 与 E1 分别只运行一次完整 305-tile Test 评价，结果仅解释为空间留出伪标签一致性。

## 冻结输入

- E0 checkpoint：`e0_tsvit_doy_seed42/best.pt`，epoch 13，SHA256 `CAB74C64897DA7FEA8A1A458ED94DAC2E23C0054A1772EAF16CF7BB5C3F9DE86`。
- E1 checkpoint：`e1_tsvit_doy_wpe_seed42/best.pt`，epoch 8，SHA256 `60B7A2C0715D45B20F5723DAD8C5ED8CB33E785AAD2A15AD39797A512E0CDDC8`。
- manifest SHA256：`79DCAAF1270D48B99FA50AF2A57548B50D7FD3E232620D11DC53E5E64C1177A8`。
- Train-only normalization SHA256：`1401DC9AAFE9A30A8916AB1A3C738080DB63D433C04716CA27FA5CABB51EC142`。
- Test split：305 tiles；24x24 window、stride 24、完整 256x256 logits 重叠平均拼接、`ignore_index=255`、physical window batch 16。

## 架构

保留 `scripts/train_e0.py --validation-only` 的训练/Validation 边界，不让训练 CLI 接触 Test。新增独立 `scripts/evaluate_test.py`：它只选择并解析 Test records，验证数量恰为 305，验证 checkpoint 路径、SHA256、epoch、嵌入的实验 ID、模型配置、数据配置、manifest/normalization 哈希、seed、batch 与 steps/epoch，然后仅加载模型权重进行评价，不恢复 optimizer、scheduler、scaler 或 RNG。

`src/training/evaluate.py::evaluate_tiles` 继续作为唯一的完整影像评价实现，但新增 `expected_split` 参数，默认值仍为 `validation`。调用者必须显式传入 `test` 才能评价 Test；数据集中任何混入的其他 split 都立即报错。滑窗、loss、混淆矩阵、per-region 指标和模型 train/eval 状态恢复逻辑保持共用。

## 输出与一次性保护

每个 checkpoint 的输出固定为其父目录下的 `test_evaluation/`，包含：

- `test_evaluation.json`：冻结资产标识、Test metrics、完整 confusion matrix、supervised pixels 和 evaluated tiles。
- `run_snapshot.json`：命令、Git、Python/PyTorch/CUDA/GPU、输入路径与哈希。

若目标 `test_evaluation/` 已存在，CLI 在加载 Test dataset 或运行模型前拒绝启动，避免无意重复评价。E0 与 E1 必须在两个独立新进程中依次运行。

## 错误处理

- 非 Test record、数量不是 305、输入哈希不一致、checkpoint 路径或 SHA 不一致、epoch/Validation best metric 不一致、嵌入配置不一致、非有限 logits、无监督像元或已有输出目录均为硬错误。
- 不提供阈值、split、checkpoint 选择、模型覆盖或训练参数覆盖选项。
- 出现异常时保留现场并按 systematic-debugging 定位；不得依据已看到的 Test 结果改变设置。

## 测试与运行顺序

1. TDD 覆盖 Test evaluator 的接受/拒绝行为、Test-only manifest 构造、冻结 checkpoint 验证和已有输出拒绝。
2. fresh focused/full pytest。
3. 不读取正式 Test 的最小 CUDA smoke。
4. 新进程运行 E0 Test；核对输出后不改变设置。
5. 新进程运行 E1 Test；核对输出后不改变设置。
6. 计算结果/checkpoint SHA256、比较 Test 与既有 Validation 差值、更新仓库和正式工作区文档、刷新库存并抽查。
7. fresh 全量 pytest、compileall、`git diff --check`。

## 发布约束

本阶段只创建并切换到 `codex/e0-e1-test-evaluation`。不 stage、commit、push 或创建 PR。
