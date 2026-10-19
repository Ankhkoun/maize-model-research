# USA B1-repro

AgriFM + NativeFPN 分类基线，附带 A4 时序可靠性专家的辅助监督。这里整理的是 **B1-repro-K4**，用于复现2026-10-12的统一源码基线。

**本版本的MoE只预测可靠性并接受辅助损失监督。它没有通过残差、FiLM或加权CE直接进入分类决策。** 不应将本仓库描述为“已证明专家分类融合有效”的模型。代码整理保持核心实现字节不变，审查发现和限制见[实现说明](docs/IMPLEMENTATION.md)。

这不是所有历史实验中测试数字最高的架构；与历史原B1、R1的区别见[版本与指标对应关系](docs/VERSION_SELECTION.md)。

## 目录

```text
model.py               模型架构导出入口
train.py               统一训练入口
test.py                GTdev / CDL / 单瓦片预测入口
af2_usa_a4_rf_20261005/  冻结的模型、数据、损失、训练与CDL评估实现
configs/b1.yaml        通用路径配置模板；训练设置与原B1-repro一致
protocol_assets/       固定划分、训练集归一化及训练影像hash
scripts/               配置、预检、GPU检查、训练与Slurm入口
tests/                 原有关键契约与仓库完整性测试
vendor/AgriFM_source/   所需Video Swin源码及Apache-2.0许可证
docs/                  数据要求、实现限制、历史参照与源码hash
```

完整网络结构、张量形状和作用路径见[模型架构说明](docs/ARCHITECTURE.md)。

保留历史Python包名，方便按原源码hash核验。仓库中不包含影像、伪标签、预训练权重和训练后checkpoint。

## 固定设置

| 项目 | 配置 |
|---|---|
| 初始化 | 原始 `AgriFM.pth`，从头训练下游模型 |
| 阶段 | A3 → M3 → J3，共9轮 |
| 专家数 / 输入 | K=4 / `a4_cube` |
| 分类 | 冻结encoder，训练NativeFPN decoder，普通CE |
| 可靠性监督 | GMM 40步，EMA=.95，λ=1 |
| 优化 | AdamW，decoder lr=1.5e-4，reliability lr=1e-4，wd=1e-4 |
| 批量 / 精度 | 单GPU，micro batch=1，累积8步，BF16 |
| 种子 | 42 |
| FiLM / feature residual / phase / weighted CE | 均关闭 |

## 环境

Python ≥3.10，CUDA版PyTorch、NumPy、PyYAML；最终CDL评估另需rasterio、affine。优先使用计算平台已经验证的CUDA环境。

```bash
python -m pip install -r requirements.txt
# 可选：仅在当前仓库中以editable方式安装
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
```

requirements是依赖列表，未声称锁定原训练环境。CI只验证CPU契约，不表示完成GPU训练。AgriFM加载器已包含所需轻量兼容实现，无需另装完整mmseg/mmengine。

## 准备权重与数据

将原始AgriFM预训练权重自行放入`weights/AgriFM.pth`。期望SHA256：

```text
3a8f4b43f466933c5607916c48f8cf9f17230f449f09e4efb7790ece3278d6a8
```

数据目录必须符合[数据要求](docs/DATA.md)，包括Train1296和GTdev60所需影像、source_map、时间记录、训练伪标签及GTdev参考标签。保持`protocol_assets/`原字节内容，不重新划分或归一化。测试集不参与训练和选模。

从仓库根目录生成本地配置：

```bash
python scripts/configure.py \
  --data-root /path/to/maize_paper_workspace \
  --output /path/to/results/b1_repro_k4_seed42

python scripts/preflight.py --config configs/local.yaml --full-train-hashes
```

配置工具只替换五个数据/输出路径；`configs/local*.yaml`不提交Git。已有配置不会被覆盖，可使用`--save configs/local_other.yaml`。

## GPU检查与训练

在分配到的一张GPU上，从仓库根目录运行：

```bash
python scripts/check_model.py --config configs/local.yaml --device cuda
python scripts/smoke.py --config configs/local.yaml
bash scripts/train.sh configs/local.yaml
```

也可直接运行`python train.py --config configs/local.yaml`，此前先执行预检。入口要求仅暴露一张GPU，使用spawn worker以接近原Windows数据加载行为；不要直接用torchrun把此基线改为8卡DDP。

Slurm示例：

```bash
# 先激活环境，再根据站点实际规则填写分区/账户参数
sbatch --partition YOUR_PARTITION scripts/slurm_train.sh configs/local.yaml
```

资源模板为1 GPU、8 CPU、48GB内存、24小时，可按站点调整资源。输出目录必须为空。每轮权重、best、history、配置hash和环境记录均保留在结果目录；需要长期运行时使用站点调度器。

## 选模与最终评估

GTdev60采用历史Derived85网格、固定阈值0.5选best。最终CDL采用Native30m-v1，全局去重；两种评估网格不能混用。冻结checkpoint和配置后执行：

```bash
# 重放GTdev60：FP32，使用原历史选模口径
python test.py --mode gtdev \
  --config configs/local.yaml \
  --checkpoint /path/to/results/b1_repro_k4_seed42/best.pt \
  --output /path/to/results/gtdev_replay

# 最终CDL500：调用字节未变的原始重投影、掩码与去重评估器
python scripts/preflight.py --config configs/local.yaml --final-test
python test.py --mode cdl \
  --config configs/local.yaml \
  --checkpoint /path/to/results/b1_repro_k4_seed42/best.pt \
  --output /path/to/results/b1_repro_k4_seed42/cdl_frozen_best \
  --cdl /path/to/2021_30m_cdls.tif
```

CDL结果应包含500瓦片、1,959,211个去重有效像元，不加载测试伪标签，不使用crop mask，不按测试结果重新选权重。

单瓦片预测不读取伪标签或参考标签，输出原10m网格的概率及二值`.npy`文件：

```bash
python test.py --mode predict \
  --config configs/local.yaml \
  --checkpoint /path/to/results/b1_repro_k4_seed42/best.pt \
  --sample-id YOUR_SAMPLE_ID \
  --output /path/to/results/one_tile
```

`sample-id`应在固定候选清单中。三个模式均记录checkpoint SHA256，拒绝不同模型/训练配置混用，且不会覆盖非空输出目录。GTdev模式遵循原历史加载方式，仍需要GTdev伪标签文件；CDL和predict模式不需要测试伪标签。CPU推理可通过`--device cpu`用于诊断，原CDL评估要求CUDA。

## 历史参照

| 模型 | GTdev mIoU | CDL mIoU | 最佳轮次 |
|---|---:|---:|---:|
| B1-repro-K4，seed42 | 87.9180% | 87.7425% | 9 |

这是既有单种子结果，不是本仓库整理后新测得的成绩，也不是多种子显著性结论。不同GPU、PyTorch版本和随机行为可能改变轨迹。历史checkpoint hash及源码hash见[来源与结果](docs/PROVENANCE.md)。

## 发布范围与第三方代码

`.gitignore`排除数据、权重、运行日志、本地配置及常见密钥文件。固定划分与归一化元数据属于仓库复现资料，保留跟踪。

第三方AgriFM代码保留Apache-2.0许可证与[来源说明](THIRD_PARTY_NOTICES.md)。主项目开源许可证尚未指定；作者可在发布时另行选择，不能把第三方许可证自动视为全部项目文件的授权。
