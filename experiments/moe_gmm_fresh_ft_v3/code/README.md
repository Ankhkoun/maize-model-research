# MoE × GMM fresh-ft-v3：AgriFM预训练起点的两组完整训练

## 用户指定范围

仅seed42，两组：`joint_frozen`冻结编码器；`joint_fullft`全量微调编码器。两组均使用相同原始AgriFM.pth，分类头、MoE专家/路由门控全新初始化，不加载B1或其他任务检查点。
同时最多2张GPU。默认`bash submit.sh validate`只提交工程验证，不启动正式训练；审查后才使用`bash submit.sh train`。

## 初始化与全量微调

`source_snapshot/agrifm.py`在创建NativeFPN256之前加载预训练，避免从AgriFM.pth加载任务分类头。随后创建A4 MoE和门控。预训练参数覆盖率必须>=90%，实际覆盖及缺失项记录于initialization.json。
`common.make_model`从构造函数创建模型，不加载epoch_009.pt，不调用旧整模型state_dict；`prepare.py`只验证原始AgriFM.pth的SHA256。
相同seed使两组初始模型、分类头、MoE参数哈希一致。结构中固定初始化的物候中心仍按原架构初始化。
frozen组编码器requires_grad=False、eval；fullft组patch embedding、全部四个Swin层级和末端norm均requires_grad=True并进入优化器/train模式，从A阶段开始更新。
两组都重新训练分类头和MoE。未启用FiLM、特征残差、相位监督。

## 自适应阶段与停止规则

|阶段|最少/最多轮|分类目标|MoE目标|停止条件|
|---|---|---|---|---|
|A|3 / 10|普通CE|不训练|GTdev连续3轮无实质提升|
|M|3 / 10|普通CE|冻结快照GMM后验qG的FP32 BCE|GTdev及固定Train探针excess-BCE均连续3轮无实质改善|
|C|8 / 30|联合后验q*加权CE，3轮权重渐进|冻结快照q*的FP32 BCE|GTdev连续6轮无实质提升|

GTdev实质改善阈值：mIoU绝对值0.0002，即0.02个百分点；探针excess-BCE改善阈值0.0001。恰好达到阈值不算超过阈值。
M入口从Train选最多32个具有有效GMM目标的固定瓦片，冻结其qG和掩码；后续在eval模式评估同一软目标的BCE减目标熵。该值仅衡量目标拟合，不代表伪标签正确率；探针无有效样本时不会触发M早停，只能到预算上限。
各阶段至少运行最少轮数；A/M选取达到最少轮数后的GTdev最佳整模型作为下一阶段起点，并重建优化器。C不回退继续训练，直至早停或上限。
`phase_summary.json`明确记录early_stopping或maximum_budget，触及上限不声称收敛。早停不能保证全局最优。
两组采用相同上限和规则，但实际轮数/更新数可能不同；不能把此结果称为严格同更新数消融。

## 融合、梯度与配置

q*=r*fL/(r*fL+(1-r)*fH)，高斯PDF不含GMM混合权重；FP64对数域融合、FP32概率输出。
M用qG预热MoE；C每轮从当前冻结快照重新生成q*，用于分类权重和BCE目标；目标和权重均detach，默认EMA0、eta1、eps1e-4。
GMM失败不使用rank：q*回退r，跳过无统计证据BCE；缺直接观测/混合/无效patch保留单位CE并跳过BCE。原结构正则保留全局定义。
学习率：encoder=1e-5（仅fullft）、decoder=1.5e-4、MoE=1e-4；AdamW，weight_decay1e-4。各参数组独立clip1。
fullft的CE与已有F1 context路径上的可靠性损失均可更新编码器；未额外detach该上下文。训练/探针/刷新模式切换会正确恢复全量编码器train模式。
micro_batch1、累积8，有效batch8，每轮Train1296共162次更新，BF16。编码器激活检查点开启。
按(seed,stage,stage_epoch,sample_id)固定增强，同一阶段同一局部轮次的数据轨迹可核验，实际阶段长度可以不同。

## Top3、测试和检查点

C阶段按GTdev原始mIoU排序，完全并列时优先更早轮次；top3排名不使用早停的min_delta。
每组保留三个C阶段最佳权重、last.pt及A/M最佳权重。只修剪本次run/checkpoints下被新top3替换的旧候选，不触碰其他实验。
两组训练均完成后才冻结全部6份top3权重哈希。全部6份均做CDL测试；GTdev rank1固定为主结果，rank2/3为次要结果，不按CDL重新选最佳。
评估几何使用字节未变的Native30m-v1：500瓦片、1959211去重有效30米像元，面积平均重投影，阈值>=0.5，无农田掩码、不读取测试伪标签。
Test500已被历史实验复用，不称全新盲测。不能根据本轮CDL调整阈值、选模型或追加预算。
检查点保存模型、优化器、目标库、RNG、配置、源码锁和停止计数；未提供自动断点恢复，入口拒绝覆盖已有run。

## 验证与边界

test_v2.py检查融合、PDF、失败回退、掩码、detach/EMA、BCE、配置、阶段策略、早停计数和top3并列排序。
validate_gpu.py对两组各用4个真实Train瓦片运行A/M/C前后向与Adam更新，验证patch embedding及全部Swin层级的梯度/参数更新、冻结组不变、相同初始参数，并检查分类推理等价性；显存峰值记录。
验证只证明实现可运行，不证明阶段已充分训练或效果提升。单seed且缺少GMM/普通CE消融，不能独立归因于联合融合机制。

入口：prepare.py → submit.sh validate → 审查 → submit.sh train → 两组训练 → 六权重冻结 → 六次CDL → summarize.py。
