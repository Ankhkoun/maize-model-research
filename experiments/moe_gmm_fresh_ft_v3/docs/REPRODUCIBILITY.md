# 复现、验证与来源

## 源码身份

`code/` 的29份锁定源码/配置与本地审查包和n16r4封存版本对应；原 `source_manifest.json` 保持不变。`verify_snapshot.py` 使用标准库检查每个锁定文件，以及GPU验收记录和审查包记录中的源码锁。`PUBLICATION_MANIFEST.json`另记录发布文件哈希。`.gitattributes`禁止Git改写封存代码与证据的换行。

新增架构、训练说明、发布验证和许可证放在封存代码目录之外，因此不改变正在训练的执行源锁。旧B1代码和历史结果单独保留，不把V2续训结果标成V3结果。

## 验证层次

- 审查包：12项CPU合同测试通过。
- GPU作业1336973：两种编码器模式各4个真实Train瓦片，A/M/C各4次Adam更新；验证初始化一致、冻结组编码器不变、fullft patch embedding及四个Swin层级有梯度和参数更新，以及分类推理等价性。
- 首次作业1336952在A/M/C更新后因末尾推理检查的str/Path参数类型失败；修复后完整重新验收通过。首次日志和当时源码保留在 `evidence/validation/attempt_1` 与 `logs` 中，不能当作最终封存版本。
- 发布时会在独立Git克隆中重新跑12项CPU测试并检查哈希；实际结果见 `../evidence/publication_cpu_tests.txt` 与 `../evidence/publication_checks.json`。

真实GPU验收环境：Python3.12.15、PyTorch2.13.0+cu129、numpy2.3.5、PyYAML6.0.3、rasterio1.5.2、affine3.0.1。以上是历史执行环境，`requirements.txt`并非严格依赖锁。

## 运行前提

本仓库发布的是**n16r4执行快照**。两份Linux配置、`order.json`和Slurm脚本保存了目标集群的绝对路径；它们不是Windows本机配置。源码中BUNDLE只用于定位原始AgriFM权重及协议资产，不会加载该目录下的B1任务检查点。

离线审查只需CPU依赖：在本目录执行 `python verify_snapshot.py`，再进入 `code` 执行 `python -B -m unittest -v test_v2`；这些命令不会提交训练。

完整训练还需要：原始AgriFM.pth、Train影像与伪标签、Train1296归一化、冻结的空间划分、GTdev参考、CDL2021原始栅格、n16r4兼容CUDA/Slurm及足够存储。它们不随Git发布。协议清单和输入文件身份由 `source_snapshot/data.py` 检查，AgriFM和CDL分别有固定SHA256检查。

在既有n16r4目录中已有正式任务，不能重新执行prepare或submit来覆盖它。迁移到新目录/机器时，应创建独立副本，更新common/prepare/配置/启动脚本中的部署路径，再重新生成源码锁并执行CPU与真实GPU验收；这会形成新的执行版本。不能修改正在运行的封存代码或仅替换YAML路径后沿用旧验收。

脚本流程为 `prepare.py → submit.sh validate → 验收 → submit.sh train → 两组训练 → 六权重冻结 → CDL → 汇总`。本次归档已经包含生成的配置和源锁，prepare会按设计拒绝重复封存。

## 研究限制

单seed、两组；没有普通CE/GMM独立消融，不能单独归因于联合融合。两组可能不同训练轮数。MoE和损失证据相关，q*未校准，跨轮自我强化风险仍在。工程验收不等于完整训练收敛或效果提升；发布时尚无V3最终CDL结果。

## 第三方来源

`code/vendor/AgriFM_source/AgriFM/models/video_swin_transformer.py`为现有研究适配版本，按执行快照保存，不声称与上游逐字一致。AgriFM归档许可证Apache-2.0全文见 `../AgriFM_LICENSE`；既有项目说明见仓库 `baselines/b1_repro_k4/THIRD_PARTY_NOTICES.md`。本次不另行指定整个仓库的统一许可证，也不分发预训练权重。
