# 2026-07-18 E1 正式训练完成交接

## 根目录与目标

- 根目录：`D:\cj_swcc\maize-model-research`
- 目标：在 Xinjiang 2021 空间 holdout 上，以完全相同的 Train/Validation、seed42 和训练计划比较 E0（TSViT+learned DOY）与仅新增三基 learnable Mexican-hat WPE 的 E1。
- 状态：E0/E1 正式 Train/Validation 训练和独立 `best.pt` 重放均 completed；Test 未读取。

## E1 已验证结果

- 正式训练从随机初始化开始，在 epoch 22 达到 warmup 后 patience=12 early stopping；global step 73194，进程正常退出，stderr 0 bytes。
- best 为 epoch 8 / global step 26616；独立新进程完整重放 Validation 276 张，结果与 checkpoint 保存指标逐项一致。
- Val loss 0.100828；OA 0.960448；precision 0.970299；recall 0.961035；F1 0.965645；maize IoU 0.933572；mIoU 0.922260；Kappa 0.919046；area ratio 0.990452。
- 混淆矩阵：`[[4680450,196830],[260718,6430304]]`；有监督像元 11568302。
- 相对 E0，E1 的 maize IoU/F1/Kappa 分别低 `0.004776/0.002548/0.004930`。首轮结果不支持 WPE 增益，当前保留 E0 为本组最佳模型。

## WPE 与 AMP 审计

- best checkpoint 学到的尺度为 `[7.229018,17.245300,35.000000]` 天，平移为 `[-1.388848,1.024661,-0.407129]` 天，alpha 为 `0.022648`。
- 小波核为三基 Mexican-hat：`a^-1/2 * (1-u^2) * exp(-u^2/2)`，其中 `u=(ΔDOY-shift)/scale`；时间支持截断于 ±42 天，并按每个 query 的绝对权重和归一化。
- 训练在 epoch 7/8/17/20 共发生 4 次 `wavelet.alpha` 非有限梯度；同批重试将 scale 从 8192 依次降至 4096/2048/1024/512。每个完成 epoch 均有 3327 optimizer steps，没有跳过 effective batch。

## 资产与哈希

- 输出：`E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e1_tsvit_doy_wpe_seed42`
- `best.pt` SHA256：`60B7A2C0715D45B20F5723DAD8C5ED8CB33E785AAD2A15AD39797A512E0CDDC8`
- `last.pt` SHA256：`AE5BE09108DCEA88EB07FB136EB0D6E8485CE54714408D0F4ED075B5FD6D1717`
- `metrics.jsonl` SHA256：`86207D5D5346859D749F508244801E9DD46A8670FAD1E910ADFC8265160810F6`
- `amp_events.jsonl` SHA256：`F791CE810ECBC80BC26133574E6A4E3946DA9B2760934037C528C7835752E684`
- `validation_replay_best_epoch8_20260718/validation_replay.json` SHA256：`EE79C31EDAC4E785D763148C283469152C84B8FE5F9F8B6D892DB7706E67E051`

## 配置与环境

- E1 config：`configs/models/tsvit_wpe_basic.yaml`；运行时 SHA256 `BAD5C5FB6821F0BC8F32F75C0FF0078DCD2B1E6787D1F9DF90D9D4A7F721229E`。
- manifest SHA256：`79DCAAF1270D48B99FA50AF2A57548B50D7FD3E232620D11DC53E5E64C1177A8`；Train-only normalization SHA256：`1401DC9AAFE9A30A8916AB1A3C738080DB63D433C04716CA27FA5CABB51EC142`。
- Train 495 / Validation 276；24x24/patch2；physical/effective batch 16/16；3327 steps/epoch；1,707,531 parameters。
- `cawa`：Python 3.11.14、PyTorch 2.10.0+cu128、RTX 5060 Ti。

## 验证与工作树

- 独立 Validation 命令 exit 0，耗时 107.6 秒；276 tiles，指标和混淆矩阵逐项一致。
- 正式工作区库存刷新 exit 0：825514 个文件、359.189696 GB；实验说明、`best.pt`、`amp_events.jsonl` 和重放 JSON 已抽查入清单。
- 收尾后 fresh 验证：`D:\Anaconda3\envs\cawa\python.exe -m pytest -q --basetemp=.smoke\pytest-e1-final-20260718` -> `71 passed in 8.23s`；`python -m compileall -q src scripts tests` -> exit 0；`git diff --check` -> exit 0。
- 分支：`experiment/e0-tsvit-xinjiang-2021`；基础 HEAD：`b0a3463b1cbf506402b6d7b3034b53c7e2116240`。
- 工作树包含尚未提交的 E0/E1 代码、配置、测试、设计、实验记录和图件；未 stage、commit、push、切换分支或创建 PR。
- 未修改 `D:\cj_swcc\_external\Exact` 或 `D:\cj_swcc\_external\TimeMIL`。

## 下一项精确任务

用户审阅 E0/E1 负结果后，决定是否冻结 E0 为当前主模型，并由用户自行建立/整理实验分支、提交和 PR。只有方案明确冻结且用户授权后，才进行一次性 Test 评价或 Train+Validation 最终重训练；不得用 Test 反向选择模型。

## 已读来源

- 仓库：`AGENTS.md`、`README.md`、`docs/STATUS.md`、`docs/TASKS.md`、`docs/DECISIONS.md`、`docs/DATA_CONTRACT.md`、`docs/EXPERIMENTS.md`、E1 启动 handoff、训练/评估实现与计划。
- 正式工作区：`README.md`、`AGENT_FIXED_RULES.md`、固定 workflow、总 handoff、E1 实验说明与输出资产。
