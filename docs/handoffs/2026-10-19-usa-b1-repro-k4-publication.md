# 2026-10-19 USA B1-repro-K4代码发布

## 用户目标与交付

将指定的B1-repro-K4完整模型、训练、测试代码发布到本仓库。新增目录为`baselines/b1_repro_k4/`，不替换现有Xinjiang模型。执行所有B1命令前先进入该目录。主README已有入口，CPU契约工作流在根目录`.github/workflows/b1-repro.yml`。

## 版本与结果边界

历史实验ID为`USA-A4-Ablation-B1-Repro-K4-Seed42-20261012`。seed42、K4、A3/M3/J3共9轮。历史GTdev mIoU=0.8791804512971124，CDL mIoU=0.8774247062503876。8个核心文件按原hash保留。MoE只用于可靠性辅助监督；FiLM、特征残差和weighted CE关闭。上传代码不是新GPU实验，不能据此宣称新环境复现成功或MoE分类融合有效。

## 验证

在本仓库的B1子目录实际运行16项CPU测试，全部通过，包括冻结hash、配置、固定split及评估入口契约。train.py/test.py帮助入口通过。Python/Bash等其他预发布检查见子项目`docs/VALIDATION.md`。GPU端到端评估和新环境精度仍需实际数据、权重及CUDA；依赖列表没有锁定历史环境。GitHub CI结果以实际工作流为准。

## 文件范围与后续

只纳入代码、通用配置、协议元数据、文档和第三方许可证；不纳入原始数据、伪标签、预训练/训练后权重、完整预测、凭据或本地配置。既有未提交工作区未参与本次发布。后续如修改模型训练语义，应新建配置与实验记录，不能修改历史hash来掩盖变更。
