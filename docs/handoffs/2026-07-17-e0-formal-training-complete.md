# 2026-07-17 E0 正式训练完成交接

## 根目录与目标

- 根目录：`D:\cj_swcc\maize-model-research`
- 目标：在 Xinjiang 2021 区域级 holdout 上完成 E0（TSViT+learned DOY），为 E1（仅增加 WPE）建立正式 Validation baseline。
- 状态：E0 completed；E1 not run。

## 已验证结果

- E0 在 epoch 25 达到 warmup 后 patience=12 early stopping，进程正常退出，stderr 为 0。
- best 为 epoch 13；独立新进程完整重放 Validation 276 张，与 checkpoint 记录逐项一致。
- Val loss 0.091894；OA 0.963000；maize precision 0.962826、recall 0.973620、F1 0.968193、IoU 0.938347；mIoU 0.926827；Kappa 0.923977；area ratio 1.011211。
- 混淆矩阵：`[[4625760,251520],[176506,6514516]]`；有监督像元 11568302。
- 全程只加载 Train/Validation，未读取 Test。

## 资产与校验和

- 输出：`E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e0_tsvit_doy_seed42`
- `best.pt`：epoch 13；SHA256 `CAB74C64897DA7FEA8A1A458ED94DAC2E23C0054A1772EAF16CF7BB5C3F9DE86`
- `last.pt`：epoch 25/global_step 83175/bad_epochs 12；SHA256 `7085320473FD7928A26B884EF4BFB095F56550A968C03FB606A2C8272A4D8B77`
- `metrics.jsonl` SHA256：`3D2793C3E1829B3F6BF04A968D35563C6F2FFBCFE2BB9E25513AE10EE2E7C840`
- 重放：`validation_replay_best_epoch13_20260717/validation_replay.json`；SHA256 `9B4B128C9BD777D30280DF06AC847C8C9DEBDF14E3964E368D625C17D5B56A16`

## 配置与数据

- 模型：24x24 window、patch2、dim128、temporal/spatial depth 4/4、heads4、10 bands、26 frames、WPE off；1,658,372 parameters。
- optimizer：AdamW，weight_decay 0；physical/effective batch 16/16；3327 steps/epoch；seed42。
- split：Train 495、Validation 276、Test 305；训练与重放不加载 Test。
- manifest SHA256：`79DCAAF1270D48B99FA50AF2A57548B50D7FD3E232620D11DC53E5E64C1177A8`。
- Train-only normalization SHA256：`1401DC9AAFE9A30A8916AB1A3C738080DB63D433C04716CA27FA5CABB51EC142`。

## AMP 异常与决策

- 默认增长、固定 65536、32768、16384 均暴露 scaled-gradient/FP16 边界问题；每次失败现场和 checkpoint 迁移记录均保留。
- epoch 25 的 `sample_004264`/`(72,192)` group 仅有 2 个监督像元，16384 可复现全模型反向 NaN；8192 从同一 epoch 24 checkpoint 完整通过 3327 steps，随后正式运行通过并早停。
- E0 最终 checkpoint 记录 scale=8192、growth_interval=1000000。
- 固定 scale 反复迁移是训练架构缺陷；E1 前须以 TDD 实现“同一 batch 降 scale 重试”，不得静默跳 batch，也不得推进 optimizer/scheduler/global_step。

## 验证命令与结果

- `D:\Anaconda3\envs\cawa\python.exe -m pytest -q --basetemp=.smoke\pytest-scale8192-full` -> `63 passed in 6.59s`。
- 最终收尾复跑：`D:\Anaconda3\envs\cawa\python.exe -m pytest -q --basetemp=.smoke\pytest-e0-final` -> `63 passed in 6.44s`。
- 独立 Validation：`scripts/train_e0.py ... --validation-only ...\best.pt` -> exit 0，276 tiles，指标/混淆矩阵逐项一致。
- 正式工作区库存刷新 -> exit 0；825503 个文件、359.151175 GB，四份 CSV/JSON 清单已生成。
- `git diff --check` -> exit 0。

## 工作树与归属

- branch：`experiment/e0-tsvit-xinjiang-2021`
- HEAD：`b0a3463b1cbf506402b6d7b3034b53c7e2116240`
- 本阶段代码、配置、测试、设计、实验记录和图件均尚未提交；部分修改始于本次恢复前的同一 E0/E1 工作序列，无法进一步按单次会话拆分，全部保留待用户审阅。
- 未修改或清理 `D:\cj_swcc\_external\Exact`、`D:\cj_swcc\_external\TimeMIL`。
- 未 stage、commit、push 或创建 PR。

## 下一项精确任务

先不要启动 E1：以 TDD 实现并验证可审计的 AMP 同批降 scale 重试，冻结起始 scale/backoff 下限；完成 CUDA smoke 后向用户展示 E1 启动配置，获得确认再训练。

## 已读取来源

- 仓库：`AGENTS.md`、`README.md`、`docs/STATUS.md`、`docs/TASKS.md`、`docs/DECISIONS.md`、`docs/DATA_CONTRACT.md`、`docs/EXPERIMENTS.md`、上一份 handoff、正式配置/训练/评估代码与测试。
- 正式工作区：`README.md`、`AGENT_FIXED_RULES.md`、总 handoff、相关输出/迁移审计/Validation 重放。
- 方法参考：Exact/TSViT 结构与配置审计记录；本次结果不依赖 Test 或外部在线数据。
