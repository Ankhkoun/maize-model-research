# 2026-07-18 E1 正式训练启动交接

## 根目录与状态

- 仓库：`D:\cj_swcc\maize-model-research`
- 当前分支：`experiment/e0-tsvit-xinjiang-2021`；基础 HEAD `b0a3463b1cbf506402b6d7b3034b53c7e2116240`。
- E0 completed；E1 running，PID 28820，启动时间 2026-07-18 17:39:01。
- 输出：`E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e1_tsvit_doy_wpe_seed42`。
- 未 stage、commit、push 或创建 PR；用户负责后续 E1 分支与发布。

## 启动前验证

- TDD 实现 AMP 同一 effective batch 重试：失败尝试不推进参数、optimizer、scheduler、global step 或 epoch 统计；成功重试只推进一次。
- 冻结策略：init scale 8192、backoff 0.5、minimum scale 128、每批最多 6 次降级、growth interval 1000000；不跳 batch、不切 FP32。
- CUDA 回归覆盖瞬时 Inf 后 8192→4096 成功，以及 256→128 后仍 Inf 的终止路径与 JSONL 审计。
- 最终 prelaunch：`71 passed in 9.86s`；compileall exit 0；git diff check exit 0。
- 真实 E1 CUDA smoke：passed；physical/effective batch 16/16；1,707,531 parameters；峰值显存 1,609,693,696 bytes；固定真实 batch loss 1.020681→0.006386；checkpoint 精确重载；Test records loaded 0。

## 正式配置

- 模型：24x24 window、patch2、dim128、temporal/spatial depth 4/4、heads4、10 bands、26 frames。
- E1 唯一模型增量：三个全局可学习 Mexican-hat WPE 基；scale init `[7,17.5,35]` 天、scale bounds `[3.5,35]` 天、shift bounds ±7 天、support radius 42 天、alpha init 0.01。
- 数据：Xinjiang 2021 Train 495 / Validation 276；seed42；Train-only normalization；训练不加载 Test 305。
- optimizer/schedule：AdamW、weight decay0、3327 steps/epoch、最多100 epoch、warmup10、patience12，按完整 Validation maize IoU 选 best。
- manifest SHA256：`79DCAAF1270D48B99FA50AF2A57548B50D7FD3E232620D11DC53E5E64C1177A8`。
- normalization SHA256：`1401DC9AAFE9A30A8916AB1A3C738080DB63D433C04716CA27FA5CABB51EC142`。
- config SHA256：`BAD5C5FB6821F0BC8F32F75C0FF0078DCD2B1E6787D1F9DF90D9D4A7F721229E`。

## 当前运行状态

- 启动记录时完成 epoch 3；最新核验已完成 epoch 5/global step 16635；stderr 0 bytes；无 `amp_events.jsonl`，即尚未触发 backoff。
- 当前 best 为 epoch 4：Val loss 0.106745、OA 0.958688、maize precision 0.962097、recall 0.966657、F1 0.964372、IoU 0.931195、mIoU 0.918747、Kappa 0.915219、area ratio 1.004739。
- 当前 best 混淆矩阵：`[[4622469,254811],[223100,6467922]]`；supervised pixels 11568302；Validation tiles 276。
- 当前仍在 warmup，以上不是最终结果，不用于判定 WPE 增益。

## 监控与终态动作

1. 监控 PID 28820、`train.stderr.log`、`metrics.jsonl`、`amp_events.jsonl`、GPU 状态。
2. 若发生 backoff，核对事件中的 group、old/new scale，并确认该 group 后续只推进一次状态。
3. 若在 scale128 终止，保留现场，按 systematic debugging 定位；不得跳 batch 或直接改配置恢复。
4. 达到 patience early stopping 或 epoch100 后，使用 best.pt 在新进程独立重放完整 Validation并逐项核对混淆矩阵。
5. E1 终态前不读取 Test；不执行 Git 发布操作。
