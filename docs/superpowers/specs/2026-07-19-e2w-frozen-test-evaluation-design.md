# E2-W 冻结 Test 双口径评价设计

## 目标与边界

在不改变 E2-W 模型、训练配置、阈值、early stopping 或 checkpoint 选择的前提下，对已经仅凭 Validation 选出的 E2-W `best.pt` 执行一次正式 Test 评价。评价同时生成与 E0/E1 完全同口径的伪标签一致性结果和独立年度 30m 参考标签结果。Test 结果只用于最终报告，不得反向用于调参、选择 checkpoint 或修改模型。

## 冻结资产

- 实验：`E2W`，Xinjiang 2021，seed 42，Train 495 / Validation 276 / Test 305。
- 配置：`configs/models/tsvit_e2w_pt_mexican_hat_k5.yaml`；24x24 window、patch2、learned 26-slot `P_T` 与 content-only five-point Mexican-hat WPE。
- checkpoint：`06_models/retrain_outputs/maize_model_research/e2w_tsvit_pt_mexican_hat_k5_seed42/best.pt`，epoch 13，Validation maize IoU `0.9369115428555538`，SHA256 `A74C8A33030172E94020410D1E46FB3439C4438ECC1E796F06CEA347DF859428`。
- manifest、Train-only normalization、Test 305 记录和独立 30m 标签路径沿用 E0/E1 已冻结资产，不增加 Test 驱动阈值或排除规则。

在第一次 Test 启动前，必须用上述 `best.pt` 在独立新进程完整重放 Validation 276，并确认指标、混淆矩阵、epoch、配置和 SHA256 与冻结记录一致。若不一致，停止而不加载 Test。

## 最小代码扩展

保留 `scripts/train_e0.py --validation-only` 的 Train/Validation 边界。只扩展独立 Test 入口：

1. 在 `scripts/evaluate_test.py` 的冻结资产表中加入 `E2W`，并将用户可见错误信息从 E0/E1 泛化为 E0/E1/E2-W。
2. 复用现有 checkpoint 路径、SHA256、epoch、embedded config、manifest、normalization、seed、batch 和 Validation best metric 校验；不增加覆盖参数。
3. 复用现有 `evaluate_tiles(..., expected_split="test")` 完整 256x256 滑窗拼接与伪标签一致性指标。
4. 复用 `scripts/evaluate_test_ground_truth.py`，在同一个冻结 E2-W 模型推理过程中同时生成 `native30m` 和 `upsampled10m` 两套指标；原生 30m 是最终主要报告口径。

不修改 E0/E1 冻结资产，不修改模型实现、checkpoint、配置、阈值、数据、损失或训练代码。

## 输出与一次性保护

输出固定在 E2-W checkpoint 父目录：

- `test_evaluation/test_evaluation.json` 与 `run_snapshot.json`：Test 伪标签一致性结果。
- `test_evaluation_ground_truth/native30m/test_evaluation.json`：独立原生 30m 参考标签结果。
- `test_evaluation_ground_truth/upsampled10m/test_evaluation.json`：标签复制到模型网格的辅助结果。
- `test_evaluation_ground_truth/run_snapshot.json`：共享冻结资产和运行环境快照。

任一目标输出目录在启动前已存在时，相应入口必须拒绝运行，防止无意重复评价。两个入口各自只启动一次独立新进程；运行后计算输出 SHA256 并核对混淆矩阵元素和严格等于监督像元数、evaluated tiles 等于 305。

## TDD 与执行顺序

1. RED：增加 E2-W 冻结资产测试，要求路径为工作区相对路径、epoch/Validation 指标/SHA256 精确固定，并验证 E2-W embedded config 可通过、任何字段变化会被拒绝。
2. GREEN：对 `evaluate_test.py` 做最小扩展；focused tests 和 Test 相关回归全部通过。
3. 在不读取 Test 的前提下，独立重放 E2-W `best.pt` 完整 Validation；冻结重放 JSON 与 SHA256。
4. fresh 全量 pytest、compileall 和 `git diff --check`；必要的 CUDA smoke 只能使用 Train/Validation。
5. 独立新进程一次性运行伪标签 Test；核对输出、305 tiles、像元守恒和 SHA256。
6. 独立新进程一次性运行 ground-truth Test；同一次模型推理生成 native30m 与 upsampled10m，核对两个混淆矩阵和 SHA256。
7. 更新 `docs/EXPERIMENTS.md`、`docs/STATUS.md`、`docs/TASKS.md`、新 handoff、正式工作区实验说明和库存；最后 fresh 运行全量验证。

## 失败处理与发布约束

- checkpoint/config/hash/epoch/Validation 指标、Test 数量、标签形状或标签域不一致时，在读取或推理 Test 前硬失败。
- CUDA、非有限 logits、像元守恒或输出写入异常时保留现场，按 systematic debugging 定位；不得依据已看到的 Test 指标修改模型或重新选择 checkpoint。
- 本任务不 stage、commit、push、切换分支或创建 PR；用户负责后续 Git 操作。
