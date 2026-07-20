# 2026-07-19 E0/E1 冻结 Test 一次性评价完成交接

## 根目录与任务边界

- 根目录：`D:\cj_swcc\maize-model-research`
- 分支：`codex/e0-e1-test-evaluation`
- 基础 HEAD：`d00e53d6cb3fb0eb8ccfdddd6c98757c2cb3a198`
- 目标：在方案、checkpoint、manifest、normalization 与评价口径均冻结后，对 E0 epoch 13 和 E1 epoch 8 `best.pt` 各进行一次完整 Test 评价。
- 禁止事项：没有依据 Test 调参、搜索阈值、修改模型、选择 checkpoint、改变 early stopping 或重跑训练；未 stage、commit、push 或创建 PR。

## Test-only 实现

- 新增独立 `scripts/evaluate_test.py`；训练/Validation CLI 不接触 Test。
- `evaluate_tiles` 默认仍严格要求 Validation；只有显式 `expected_split="test"` 才允许 Test，混入其他 split 立即报错。
- Test CLI 只选择并构造 Test records，严格要求 305 条；在读取 Test dataset 前验证 checkpoint 路径/SHA256/epoch/Validation best、嵌入 experiment/model/data、manifest/normalization 哈希、seed42、physical batch16 和 3327 steps/epoch。
- 只加载模型权重，不恢复 optimizer、scheduler、scaler 或 RNG；输出目录固定为 checkpoint 父目录的 `test_evaluation`，若已存在则拒绝重复评价。
- 保持 Validation 评价口径：24x24 window、stride24、完整 256x256 logits 重叠平均拼接、`ignore_index=255`、二分类混淆矩阵、window batch16 与 CUDA AMP。

## TDD 与 Test 前门禁

- 基线：`71 passed in 9.03s`。
- RED 1：`expected_split` 不存在，2 项按预期失败；GREEN：新旧 evaluator 聚焦测试 `4 passed`。
- RED 2：`scripts.evaluate_test` 不存在，收集按预期失败；GREEN：dataset/checkpoint/output guard 聚焦测试通过。
- RED 3：CLI 参数/结果文档函数不存在，收集按预期失败；GREEN：全部聚焦测试 `12 passed in 1.91s`。
- 正式 Test 前 fresh 全量 pytest：`81 passed in 8.58s`。
- 真实 CUDA smoke：RTX 5060 Ti；physical/effective batch16/16；checkpoint 精确重载；`test_records_loaded=0`；E0 固定真实批次 loss `1.020775 -> 0.006388`。
- 两个正式 `test_evaluation` 目录在启动前均不存在；真实 E0/E1 checkpoint CPU preflight 逐项通过。

## 冻结资产

- E0：`best.pt` epoch13/global step43251；Validation maize IoU `0.9383473661376481`；SHA256 `CAB74C64897DA7FEA8A1A458ED94DAC2E23C0054A1772EAF16CF7BB5C3F9DE86`。
- E1：`best.pt` epoch8/global step26616；Validation maize IoU `0.9335717434114438`；SHA256 `60B7A2C0715D45B20F5723DAD8C5ED8CB33E785AAD2A15AD39797A512E0CDDC8`。
- manifest SHA256：`79DCAAF1270D48B99FA50AF2A57548B50D7FD3E232620D11DC53E5E64C1177A8`。
- normalization SHA256：`1401DC9AAFE9A30A8916AB1A3C738080DB63D433C04716CA27FA5CABB51EC142`。

## 正式 Test 结果

E0 新进程 exit0、耗时123.5秒；E1 新进程 exit0、耗时110.2秒。两者均评价 Test `region_r008_c022` 的305 tiles，有监督像元均为12295425，混淆矩阵之和逐项守恒。

| 指标 | E0 Test | E1 Test | E1-E0 Test | E0 Validation | E1 Validation | E1-E0 Validation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loss | 0.123838690 | 0.127736956 | +0.003898266 | 0.091894118 | 0.100827849 | +0.008933731 |
| OA | 0.951711307 | 0.948239691 | -0.003471616 | 0.963000101 | 0.960448128 | -0.002551973 |
| maize precision | 0.968469151 | 0.976072391 | +0.007603240 | 0.962826092 | 0.970299378 | +0.007473286 |
| maize recall | 0.956387120 | 0.942992283 | -0.013394836 | 0.973620472 | 0.961034652 | -0.012585820 |
| maize F1 | 0.962390217 | 0.959247226 | -0.003142991 | 0.968193196 | 0.965644793 | -0.002548403 |
| maize IoU | 0.927506884 | 0.921685966 | -0.005820918 | 0.938347366 | 0.933571743 | -0.004775623 |
| mIoU | 0.900577207 | 0.894624126 | -0.005953081 | 0.926826619 | 0.922259969 | -0.004566650 |
| macro-F1 | 0.947476795 | 0.944166238 | -0.003310557 | 0.961986780 | 0.959521973 | -0.002464807 |
| Kappa | 0.894961011 | 0.888390183 | -0.006570827 | 0.923976845 | 0.919046473 | -0.004930372 |
| area ratio | 0.987524609 | 0.966108961 | -0.021415648 | 1.011211142 | 0.990451683 | -0.020759459 |

- E0 confusion matrix：`[[4105268,247320],[346410,7596427]]`。
- E1 confusion matrix：`[[4168976,183612],[452803,7490034]]`。
- Test 与 Validation 的主要方向一致：E1 precision 更高，但 recall、F1、IoU、mIoU、Kappa 和 OA 更低。
- Test 不参与模型选择，没有改变此前仅依据 Validation 保留 E0 的结论。
- 所有指标衡量与空间留出伪标签的一致性，不是独立地面真值精度。

## 输出与哈希

- E0 `test_evaluation/test_evaluation.json`：SHA256 `C8A4F4A41DB32135224A6E0FF55BA81640DB68084A5C9B4D815AFD73A894089B`。
- E0 `test_evaluation/run_snapshot.json`：SHA256 `2C77964BF4935C53FE2EA1044E439895823FF6DA6670C81A221BE3454D586F82`。
- E1 `test_evaluation/test_evaluation.json`：SHA256 `B4307DF7CE972E271BB38BB0663D78C926CE726F838FC6D85B66EFAB7A0A6E95`。
- E1 `test_evaluation/run_snapshot.json`：SHA256 `A5B52424C701B374D32B2F8054262A925BD94B6C27A58D3067CF89DE7E4B98A8`。
- 正式实验说明：`00_docs/experiment_notes/maize_model_research_e0_e1_test_evaluation_2026-07-19.md`，SHA256 `E3EECF7D45936D65106A7A550643A5BBECE52CF36B7078FA4A1325D85C16FA31`。
- 正式总 handoff 更新后 SHA256：`200BE82A77A2DA42E1FE3E3C18A5711D782330ECE24A8E61B78CE6A5B6767AC6`。

## 正式工作区库存

- 固定脚本 exit0，耗时277.3秒。
- 刷新后：825519文件、359.18971GB。
- 逐文件库存已抽查到5个新增文件：E0/E1各两个 Test JSON，加一份实验说明；总 handoff 的新章节计数为1。

## 收尾验证

- `D:\Anaconda3\envs\cawa\python.exe -m pytest -q --basetemp=.smoke\pytest-test-eval-final2-20260719` -> `83 passed in 7.72s`（含冻结 checkpoint 工作区相对路径可移植性回归）。
- `D:\Anaconda3\envs\cawa\python.exe -m compileall -q src scripts tests` -> exit0。
- `git diff --check` -> exit0；仅报告既有 CRLF 转换提示，无 whitespace error。

## 仓库文件

- 修改：`src/training/evaluate.py`、`docs/EXPERIMENTS.md`、`docs/STATUS.md`、`docs/TASKS.md`、`docs/DECISIONS.md`。
- 新增：`scripts/evaluate_test.py`、3个 Test-only 测试文件、Test-only 设计与实施计划、本 handoff。
- 未修改外部参考树 `D:\cj_swcc\_external\Exact` 或 `D:\cj_swcc\_external\TimeMIL`。

## 下一步

用户审阅结果和文件清单后自行 stage/commit/push/创建 PR。任何后续 Train+Validation 重训练或新消融必须独立预注册，且不得再次使用本轮 Test 调参或选择模型。
