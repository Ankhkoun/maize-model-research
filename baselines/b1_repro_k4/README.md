# B1-repro-K4

基于AgriFM、NativeFPN和时序可靠性专家的玉米遥感分类模型。

## 模型结构

- 输入：26期、10波段遥感影像，空间大小为256×256。
- 编码器：AgriFM Video Swin Transformer，提取三级多尺度特征。
- 解码器：NativeFPN，输出256×256的两类logits。
- 时序专家：A4时序编码、K=4软路由专家与importance gate，输出混合可靠性。

结构细节见[模型架构](docs/ARCHITECTURE.md)。

## 代码目录

| 路径 | 内容 |
|---|---|
| `model.py` | 模型架构导出入口 |
| `train.py` | 训练入口 |
| `af2_usa_a4_rf_20261005/agrifm.py` | AgriFM编码器与NativeFPN解码器 |
| `af2_usa_a4_rf_20261005/model.py` | 总模型与前向计算 |
| `af2_usa_a4_rf_20261005/reliability.py` | 时序编码、专家与门控 |
| `af2_usa_a4_rf_20261005/objective.py` | 训练损失 |
| `af2_usa_a4_rf_20261005/targets.py` | GMM目标与EMA更新 |
| `configs/b1.yaml` | 训练配置模板 |
| `protocol_assets/` | 数据划分和归一化参数 |

## 环境与数据

进入本目录后安装依赖。需要Python≥3.10和CUDA版PyTorch。

```bash
python -m pip install -r requirements.txt
```

将AgriFM预训练权重放入`weights/AgriFM.pth`，按[数据要求](docs/DATA.md)准备影像、伪标签、时间记录及验证标签。生成本地配置：

```bash
python scripts/configure.py --data-root /path/to/maize_paper_workspace --output /path/to/results/b1_repro_k4_seed42

python scripts/preflight.py --config configs/local.yaml --full-train-hashes
```

## 训练

| 参数 | 设置 |
|---|---|
| 专家数 | K=4 |
| 训练阶段 | A3 → M3 → J3，共9轮 |
| 编码器 | 冻结 |
| 随机种子 | 42 |
| 批量 | 单GPU，micro batch=1，梯度累积8步 |
| 精度 | BF16 |
| 优化器 | AdamW，weight decay=1e-4 |
| 学习率 | decoder=1.5e-4，reliability=1e-4 |
| 可靠性目标 | GMM 40步，EMA=0.95，λ=1 |

仅暴露一张GPU，先检查模型和训练流程，再启动正式训练：

```bash
python scripts/check_model.py --config configs/local.yaml --device cuda
python scripts/smoke.py --config configs/local.yaml
python train.py --config configs/local.yaml
```

训练输出目录必须为空，结果中保存checkpoint、best、训练历史及配置。配置示例与脚本均以本目录为工作目录。

Slurm提交示例：

```bash
# 按平台填写实际分区，提交前激活Python环境
sbatch --partition YOUR_PARTITION scripts/slurm_train.sh configs/local.yaml
```

第三方代码来源与许可见[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
