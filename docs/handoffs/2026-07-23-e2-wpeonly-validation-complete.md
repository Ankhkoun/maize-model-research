# 2026-07-23 E2-WPEonly Train/Validation 完成交接

## 项目与目标

- 根目录：`D:\cj_swcc\maize-model-research`。
- 目标：检验 E2-W 未超过 E0 是由 K5 WPE 本身，还是由同时用 learned `P_T` 替换 calendar DOY 所致。
- 受控路径：content-only K5 WPE 后加 E0 `nn.Embedding(367,128)` calendar DOY；无 learned `P_T`；class token 不进入 WPE。

## 已验证工作

- 正式训练 completed：epoch 23 / global step 76521 / bad_epochs 12 正常早停；best epoch 11 / global step 36597。
- 独立新进程完整重放 Validation 276，与 checkpoint 逐项一致：IoU/F1/Kappa `0.9375021027/0.9677430557/0.9241060406`。
- 混淆矩阵 `[[4702039,175241],[253886,6437136]]`，总和与 supervised pixels `11568302` 守恒。
- 4 次 `wavelet.gamma` 同批 AMP backoff，scale `8192→4096→2048→1024→512`；失败尝试未推进训练状态。
- 已在 `docs/EXPERIMENTS.md` 汇总 E0/E1/E2-W/E2-WPEonly 的统一独立 Validation 对比。

## 实验证据

- Git：分支 `codex/e2-wpeonly`；训练 commit `dfbff1bf91c4e80e7077f2686e6ecd155b20ae2e`。
- Config：`configs/models/tsvit_e2_wpeonly_doy_mexican_hat_k5.yaml`，SHA256 `DC02660A8CE4B32D8189875B1C79E2AB7DB617D1262276E83EF5E2FF3867E28B`。
- 输出：`E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e2_wpeonly_tsvit_doy_mexican_hat_k5_seed42`。
- `best.pt` SHA256：`1C5F697D5FF12D114295D1FD359ADE1AA90342DE71DEAA3D5A3A1BD69130EBD8`。
- `last.pt` SHA256：`8D3326BD1A7A67B086F5DD10C7B08E08F87163988E3DCE1A5DF28D6DA7A25D03`。
- `metrics.jsonl` SHA256：`CFE98CC3CC282E55CE79A25320BEFB742D56BD53FCB2A35ECBEAC65FBD8D11D7`。
- `amp_events.jsonl` SHA256：`3300700A4CA0BF21DA97906C19D8D5696EAA54AB27E7650921170BB55A591395`。
- Validation replay JSON SHA256：`A8F4C028F19F2CF48C3661A143A8AE5DA49A48783AF21E8BE1DDDE546BD6C2A3`。

## 决策与解释

- E2-WPEonly−E0 IoU/F1/Kappa：`-0.000845/-0.000450/+0.000129`。
- E2-WPEonly−E2-W IoU/F1/Kappa：`+0.000591/+0.000315/+0.001294`。
- 恢复 calendar DOY 缩小了 E2-W 与 E0 的差距，但 K5 WPE 的 WPE-only 消融仍未提高预注册 maize IoU；继续保留 E0。
- 单 seed 结果不声明显著性。配置中的 `experiment.status: not_run` 属于训练时冻结哈希的一部分，不修改；实际状态以实验注册表和本 handoff 为准。

## Test 与工作树边界

- 未注册 E2-WPEonly Test asset，未解析、加载或运行 Test；输出中不存在 `test_evaluation` 或 `test_evaluation_ground_truth`。
- 本次会话修改项目记录与正式工作区实验说明；没有 stage、commit、push、切换分支或创建 PR。
- 会话开始时工作树 clean；本 handoff 与文档更新均为本次会话产生，无已知 pre-existing/unattributed changes。

## 验证与下一项有界任务

- 已完成：结果 JSON/检查点/日志 SHA256、混淆矩阵守恒、独立重放逐项一致、Test 目录不存在。
- 正式工作区库存已刷新：825,553 个文件、359.282401 GB；关键训练、重放和实验说明均抽查入清单。
- 收尾验证：fresh 全量 pytest `103 passed in 8.00s`；compileall 与 `git diff --check` exit0；文档 diff、UTF-8 和换行已审阅。
- 下一项任务：用户审核记录后单独确认是否 stage/commit/push；只有另行明确批准后才能设计一次性 E2-WPEonly Test。
