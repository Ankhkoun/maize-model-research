# Decision Log

| ID | 日期 | 决策 | 理由与后果 |
| --- | --- | --- | --- |
| D-001 | 2026-07-14 | 建立独立主仓库 `maize-model-research`。 | 将论文代码与大型数据工作区、外部参考树隔离。 |
| D-002 | 2026-07-14 | 大型数据和正式输出继续保存在 `E:\maize_paper_workspace`。 | 避免 Git 误提交大型或受限资产，并保持现有数据路径稳定。 |
| D-003 | 2026-07-14 | 只提交 `paths.example.yaml`，忽略 `paths.local.yaml`。 | 保持代码可迁移，不固化个人机器路径。 |
| D-004 | 2026-07-14 | Exact 只作为 temporal-spatial Transformer baseline 结构来源。 | 当前使用像元伪标签直接分割，不采用 CAM、prototype、TAAP 或弱监督专用损失。 |
| D-005 | 2026-07-14 | 首个模型对为 E0（DOY）和 E1（DOY+WPE）。 | E1 相对 E0 只增加 WPE，形成受控对照。 |
| D-006 | 2026-07-14 | 首版 WPE 不使用观测质量权重。 | `pseudo_confidence` 是标签置信度，不能替代逐时间点观测质量。 |
| D-007 | 2026-07-14 | 所有模型选择只依据 Validation 区域。 | 避免用 Test 选择模型、阈值或 checkpoint。 |
| D-008 | 2026-07-15 | E1 使用三个全局可学习 Mexican-hat 基，在 DOY 编码后、temporal class token 前仅注入一次；尺度、中心偏移和支持范围均受约束。 | 当前实现以真实 DOY 差计算；在本数据固定 7 天时间点上，初始尺度对应约 1、2.5、5 个时间点，中心偏移约束为 ±1 个时间点，支持为前后各 6 个时间点。 |
| D-009 | 2026-07-15 | 工程 smoke 配置曾采用 `256x256` 输入与内部 patch 8。 | 该配置用于验证完整影像前反向和显存，不再作为正式训练配置；由 D-010 取代。 |
| D-010 | 2026-07-16 | 正式 E0/E1 采用 Exact 风格 `24x24` 训练窗口，模型内部 patch 为 `2x2`。 | Exact 中 24x24 是数据窗口而非单个 token；该设置形成 `12x12=144` 个空间 token，并通过 patch-to-pixel head 恢复 24x24 像元输出。 |
| D-011 | 2026-07-16 | E0/E1 共享可学习 DOY 查找表，替换当前标量 `Linear(1,D)`。 | Exact/TSViT 的 one-hot+Linear 数学上等价于查找表；共享该表可确保 E1 唯一新增因素是 WPE。正式实现计划使用索引 0 作为 padding、1..366 表示 DOY。 |
| D-012 | 2026-07-16 | 首轮保留现有区域级 495/276/305 Train/Validation/Test 划分。 | 两个 `r008` 区域均不进入 Train，降低空间邻近泄漏风险；虽然 Validation 占比偏大，但 495 张完整影像切成 24x24 窗口后训练样本量充足。 |
| D-013 | 2026-07-16 | 首轮 E0/E1 使用普通像元级交叉熵与 `ignore_index=255`，不以 `pseudo_confidence` 加权。 | 遵守“不使用弱监督损失”的约束，且保证 E0/E1 损失完全一致。 |
| D-014 | 2026-07-16 | 当前分布偏移不触发重新随机划分。 | Validation 相对 Train 有小到中等光谱/时序偏移且玉米比例低 6.67 个百分点，Test 更接近 Train；这种差异可用于检验区域泛化，随机拆分反而会造成空间泄漏。 |
| D-015 | 2026-07-17 | E0 正式训练使用物理/有效 batch 16、每 epoch 3327 optimizer steps，Test 在训练期间不加载。 | RTX 5060 Ti 真实 CUDA smoke 通过 batch 16；manifest split 选择接口只解析 Train/Validation 路径。 |
| D-016 | 2026-07-17 | GradScaler 曾将 init_scale 固定为 65536 且禁止自动增长；该值随后由 D-017 取代。 | 首次正式尝试在默认 2000-step 增长后于 step 2228 出现 scaled-gradient overflow；这是第一阶段诊断假设；epoch 4 的确定性复现证明 65536 仍可能使 patch embedding 的 FP16 反向溢出。 |
| D-017 | 2026-07-17 | GradScaler 曾使用 init_scale=32768、growth_interval=1000000；该值随后由 D-018 取代。 | 从 epoch 3 checkpoint 的单变量确定性重放显示：65536 在 step 10815 仅使 patch embedding 梯度 Inf，而 32768 完整通过 epoch 4；但它在训练后期仍不够保守。 |
| D-018 | 2026-07-17 | GradScaler 曾使用 init_scale=16384、growth_interval=1000000；该值随后由 D-019 取代。 | 32768 在 epoch 10 可复现分割头 Inf；16384 虽通过 epoch 10 单变量重放，但在 epoch 25 的 2 个监督像元 group 上可复现全模型 FP16 反向 NaN。 |
| D-019 | 2026-07-17 | E0 以 init_scale=8192、growth_interval=1000000 完成 epoch 25 并早停；E1 启动前必须实现可审计的同批降 scale 重试。 | 从同一 epoch 24 checkpoint 的单变量重放证明 8192 完整通过 3327 steps。固定 scale 反复迁移说明策略本身脆弱；后续不得静默跳 batch，非有限 scaled gradient 应在不推进 optimizer/scheduler/global_step 的前提下降 scale 并重放同一 batch。 |
| D-020 | 2026-07-18 | E1 冻结 AMP 同批重试：init_scale=8192、backoff=0.5、min_scale=128、每批最多 6 次降级、growth_interval=1000000。 | CUDA 测试证明失败尝试不改变参数或推进 optimizer/scheduler/global_step；成功重试只推进一次，并向 amp_events.jsonl 写入逐批审计。前向非有限立即终止，scale=128 仍失败则保留诊断并终止，不跳 batch、不切 FP32。 |
| D-021 | 2026-07-18 | 将首轮 E1 登记为完成但未优于 E0 的受控负结果；当前不以 WPE 替换 E0。 | 独立完整 Validation 重放显示 E1 maize IoU 0.933572，低于 E0 的 0.938347（差 -0.004776）；F1 与 Kappa 也分别低 0.002548 和 0.004930。该结论只适用于当前 seed42、三基 WPE 与已批准训练方案，不使用 Test，也不排除后续预注册消融。 |
| D-022 | 2026-07-19 | 使用独立、严格的 Test-only 入口对冻结 E0 epoch 13 与 E1 epoch 8 `best.pt` 各进行一次完整 Test 评价；Test 只用于最终报告，不改变 D-021。 | 入口只解析 305 条 Test records，并强制 checkpoint/manifest/normalization 哈希与配置一致。E1-E0 Test maize IoU/F1/Kappa 为 `-0.005821/-0.003143/-0.006571`，与 Validation 方向一致；不据此调参、搜索阈值、选择 checkpoint 或重跑训练，指标仅解释为空间留出伪标签一致性。 |

| D-023 | 2026-07-19 | 冻结 E0/E1 的最终 Test 同时报告独立年度30m参考标签的原生30m和标签复制模型网格两种口径；两者均仅评价 `label !=255`，不使用耕地掩膜或排除列表。 | 30m参考标签原生支持域为85x85，而模型网格为256x256；固定3x3概率平均提供原生30m指标，固定3x3标签复制展示模型网格内的预测异质性。最右/最下1像元无对应30m支持域，明确忽略。E1相对E0的原生30m IoU/F1/Kappa差值为 `-0.014851/-0.009399/-0.016357`，标签复制网格为 `-0.016298/-0.010411/-0.017312`；只作最终报告，不反向改变 D-021。 |

| D-024 | 2026-07-19 | 将 E2-W（learned `P_T` + content-only five-point Mexican-hat WPE）登记为完成但未超过 E0 的负结果；冻结 epoch13 后仅执行一次伪标签 Test 和一次独立30m Test。 | E2-W 相对 E0 的 Validation IoU/F1/Kappa 为 `-0.001436/-0.000765/-0.001164`，伪标签 Test 为 `-0.003592/-0.001937/-0.004126`，原生30m Test 为 `-0.003937/-0.002476/-0.002242`。三种口径方向一致，继续保留 E0。E2-W 同时替换 temporal position table 和加入 WPE，不能解释为单独 WPE 效应；Test 不反向改变选择。 |
| D-025 | 2026-07-22 | 将原预留 E2 的 WPE-only/绝对 DOY 互补性实验正式命名为 `E2-WPEonly`；保留 E0 calendar DOY lookup，只在 content-only patch tokens 上增加 E2-W 的 five-point Mexican-hat WPE。原预留 E3 继续专属于 source-derived quality 扩展，不复用。 | E2-W 同时替换 `P_T` 并加入 WPE，不能解释纯 WPE 效应；E2-WPEonly 相对 E0 的唯一结构差异是 five-point WPE。代码与 Train/Validation-only smoke 已通过，但正式训练仍须用户另行批准，且不得读取 Test。 |
| D-026 | 2026-07-23 | 将 E2-WPEonly 登记为完成但未超过 E0 的单变量结构消融；继续以 E0 为当前最佳，并保持 E2-WPEonly Test 未运行。 | 独立 Validation 重放显示 E2-WPEonly IoU/F1/Kappa 为 `0.937502/0.967743/0.924106`，相对 E0 为 `-0.000845/-0.000450/+0.000129`，相对 E2-W 为 `+0.000591/+0.000315/+0.001294`。恢复 calendar DOY 缩小了与 E0 的差距，但预注册选择指标 maize IoU 仍未超过 E0，因此不支持 K5 WPE 在当前 seed42 冻结设置下改善 E0；单 seed 不声明显著性。 |

## 当前已解决问题

- 标签语义：二分类，`0=非玉米`、`1=玉米`、`255=ignore`。
- 首轮空间划分：采用已核验的五区域 holdout manifest，正式项目副本仍待固化。
- 正式空间结构：24x24 数据窗口、内部 patch 2、共享 patch-to-pixel head。
- 首轮损失：普通交叉熵，ignore 255，不使用置信度权重。

## 待决策或待实现

| ID | 问题 | 当前原则 |
| --- | --- | --- |
| O-001 | 24x24 滑窗在 256x256 边缘如何 padding 和拼接？ | Train/Validation/Test 必须使用明确、可测试且无标签泄漏的统一空间规则。 |
| O-002 | 训练集 10 波段归一化参数是什么？ | 只能用 Train 三个区域计算，冻结后 E0/E1 共用。 |
| O-003 | 1076 个 Xinjiang 2021 样本是否存在独立参考标签？ | 已确认 Test 305 条均有年度30m `y_patch_30m.npy`；最终报告使用原生30m与标签复制网格双口径，伪标签 Test 仅保留为一致性诊断。 |
| O-004 | 最终重训练是否合并 Train+Validation？ | 仅在所有超参数和 epoch 规则由开发阶段冻结后考虑；不得再用 Test 调参。 |

| D-027 | 2026-07-26 | After explicit user authorization, register E2-WPEonly as a frozen Test asset and run exactly one pseudo-label Test plus one independent annual 30m ground-truth Test; retain E0 as the current best model. | The frozen best checkpoint (epoch 11) passed strict hash/config/manifest/normalization checks. E2-WPEonly pseudo Test IoU/F1/Kappa was `0.926351/0.961768/0.894869`; native30m reference was `0.780135/0.876490/0.777386`; upsampled10m label-grid was `0.771875/0.871252/0.768449`. All remain below E0 on the reported IoU/F1/Kappa comparisons. Test is final reporting only and does not change the pre-registered Validation selection rule, checkpoint, threshold, model, or training. |
| D-028 | 2026-07-27 | 新增 E3-ImageWPE：严格纯影像 K5 Mexican-hat 小波消融；不注入 calendar DOY 或 learned temporal slot。 | E1 三基连续小波以真实 DOY 差为核坐标，不能作为严格无 DOY 对照；E3 仅使用五点内容型小波、有效帧掩码和影像 patch token。E3 与 E0/E2-WPEonly 保持相同的 495/276 划分、seed42、24x24/patch2、损失、优化器和 Validation maize IoU 选择规则；不读取或注册 Test。 |
| D-029 | 2026-07-27 | E3-ImageWPE epoch15 checkpoint is frozen after independent Validation replay and evaluated exactly once on pseudo-label Test and annual 30m reference Test; E3 is the numerical Validation-IoU leader for this single-seed comparison. | The frozen checkpoint SHA256 is `83F9128719C60C87CA07A044B22DCA3FF4264E23316CD0CA8D088BF7FDC9FB85`; Validation maize IoU is 0.938460, only +0.000113 above E0. Pseudo Test IoU/F1/Kappa is `0.925872/0.961509/0.895082`; native30m is `0.776375/0.874112/0.774001`; upsampled10m is `0.767169/0.868247/0.764869`. The Test results are reporting-only and do not alter checkpoint, threshold, training, or selection. |
| D-030 | 2026-07-28 | After independent Validation replay, register E4-ImageWPE-K9 as a frozen Test asset and execute exactly one pseudo-label Test and one annual 30m reference Test. | E4 binds strict image-only K9 (`kernel_radius=4`) to epoch-11 `best.pt`, SHA256 `E30E912C7ECEA90FAEFCCCF8444849F7F36A1137084FDDC9F70EC0F1204ABE14`, Validation IoU `0.938988`. One-time pseudo Test IoU/F1/Kappa is `0.927715/0.962502/0.896610`; native30m is `0.776925/0.874460/0.772287`; upsampled10m is `0.769763/0.869905/0.764588`. Test is reporting-only and does not change checkpoint selection or establish significance. |
| D-031 | 2026-07-28 | After independent Validation replay, register E5-DOY-K9WPE as a frozen Test asset and execute exactly one pseudo-label Test and one annual 30m reference Test. | E5 binds calendar DOY + K9 WPE to epoch-14 `best.pt`, SHA256 `1648958D1563941EEE427B6222FCB4ACB283E6E2A84C76B83E1616B8C2181561`, Validation IoU `0.934533`. One-time pseudo Test IoU/F1/Kappa is `0.926526/0.961862/0.891992`; native30m is `0.784859/0.879463/0.778732`; upsampled10m is `0.776534/0.874212/0.770151`. Test is reporting-only and does not change checkpoint selection or establish significance. |

## 2026-10-19：以独立子项目发布USA B1-repro-K4

用户指定发布B1-repro-K4。将其置于`baselines/b1_repro_k4/`，不替换Xinjiang模型或数据协议。冻结核心代码字节，补充可移植入口、预检和文档；模型已知限制保持公开。数据、伪标签、checkpoint、本地路径配置不提交。历史结果仅用于协议参照，不据此宣称MoE分类融合有效。
