# AMP 同批降 Scale 重试设计

## 目标与边界

在 E1 正式训练前，为现有 TSViT 分割 trainer 增加可审计的 AMP 同一 effective batch 重试。该机制只处理 FP16 backward 产生的非有限梯度，不改变数据、增强、loss、optimizer、scheduler、模型结构或 Validation 选择规则。

正式 E1 继续使用 Xinjiang 2021 的 Train 495 / Validation 276，保持 `24x24` window、patch 2、physical/effective batch 16/16、seed 42、最多 100 epoch、warmup 10 和 patience 12。训练期间不得加载 Test。

## 方案与冻结参数

采用显式 group 重试循环：

```yaml
amp_init_scale: 8192.0
amp_backoff_factor: 0.5
amp_min_scale: 128.0
amp_max_backoffs_per_batch: 6
amp_growth_interval: 1000000
```

允许的 scale 序列为 `8192, 4096, 2048, 1024, 512, 256, 128`。一次 group 最多发生 6 次降级；在 128 仍出现非有限梯度时，以带完整诊断的 `FloatingPointError` 终止，不继续减小 scale，也不切换为 FP32。

## 数据流与状态语义

每个 effective batch 的窗口抽取、随机顺序和增强只执行一次。重试复用同一组 `group_images`、`group_labels`、DOY、valid mask、坐标和有效监督像元数，并重新执行该组全部 microbatch 的 forward/backward。

一次尝试的顺序为：

1. `optimizer.zero_grad(set_to_none=True)`；
2. 对该 group 的全部 microbatch 计算交叉熵并进行 scaled backward；
3. `scaler.unscale_(optimizer)`；
4. 扫描全部参数的未缩放梯度；
5. 梯度有限时才执行 `scaler.step`、`scaler.update`、`scheduler.step`，并推进 `global_step`；
6. 梯度非有限时记录事件、清梯度、调用 `scaler.update(new_scale)` 完成 scaler 状态复位，然后从同一 group 的第一个 microbatch 重试。

失败尝试不得改变 optimizer 参数或动量，不得推进 scheduler、global step、optimizer step 计数、累计 loss、窗口计数或监督像元计数。只有成功尝试的统计量进入 epoch metrics。

正式配置 dropout 为 0，因此重放不存在 dropout mask 差异；数据增强已在重试循环外冻结。

## 异常分类

- logits 或 objective 在 forward 阶段非有限：立即终止。loss scale 只影响 backward，降低 scale 无法修复前向异常。
- unscale 后梯度非有限且当前 scale 大于 128：执行批准的降级与同批重试。
- 当前 scale 已为 128，或已用完 6 次降级：写入终止事件并抛出 `FloatingPointError`。
- 无监督像元 group：沿用现有逻辑直接跳过，不算 optimizer step，也不产生 AMP 重试事件。

## 审计记录

每次非有限梯度尝试立即向输出目录的 `amp_events.jsonl` 追加一条 JSON，至少包含：

- `event`: `amp_gradient_backoff` 或 `amp_gradient_failure`；
- epoch、当前 global step、sample ID、group 首坐标及全部 group 坐标；
- attempt、old scale、new scale、backoff factor、min scale；
- learning rate、group loss sum、有效监督像元数；
- 每个异常参数的名称、NaN 数、Inf 数及有限梯度绝对值最大值。

写入后立即 flush。checkpoint 继续保存 GradScaler 状态，因此降级后的 scale 可随 `last.pt` 恢复。run snapshot 和配置记录四个冻结 AMP 参数。

## 正式 E1 入口

现有训练 CLI 扩展为同时验证 E0/E1 正式配置：

- E0 必须 `experiment.id=E0` 且 WPE disabled；
- E1 必须 `experiment.id=E1` 且 WPE enabled，并严格匹配已批准的小波核参数；
- 两者除 experiment 标识、名称和 `model.wavelet` 外，数据、主体模型、optimizer、schedule、seed 和评估规则必须一致；
- E1 使用独立输出目录 `E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e1_tsvit_doy_wpe_seed42`；
- CLI 构建 manifest 时只请求 `train` 和 `validation` split。

## 测试与验收

TDD 测试必须先失败，再实现最小代码，覆盖：

1. 第一次梯度非有限、降低 scale 后同一 group 成功；模型 forward 被再次调用；
2. 失败尝试不改变模型参数、optimizer/scheduler/global step 和 epoch 累计量；
3. 成功重试只产生一个 optimizer/scheduler/global step；
4. scale 按 `8192 -> 4096` 更新并保留；
5. 降至 128 后仍失败会终止，并产生 `amp_gradient_failure`；
6. `amp_events.jsonl` 字段完整且可逐行解析；
7. E1 正式配置可被 CLI 接受，E0/E1 非 WPE 条件一致，错误组合被拒绝；
8. 完整 pytest、compileall 和 `git diff --check` 通过；
9. CUDA smoke 使用真实 E1 `24/patch2`、physical batch 16 完成至少一个成功 optimizer step，且不读取 Test。

CUDA smoke 通过后才启动正式 E1。正式训练只依据 Validation 保存 best checkpoint 和 early stopping；Test 保持未读取。

## 非目标

本次不修改 WPE 数学定义，不调大模型、不改变 batch size、不增加 gradient clipping、不静默跳 batch、不使用 FP32 fallback、不从 E0 权重初始化 E1，也不运行 Test。
