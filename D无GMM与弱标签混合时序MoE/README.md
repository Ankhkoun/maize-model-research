# D＋双时序 MoE 的两种无损失 GMM 监督版本

用于审阅与发布的独立代码包。两种配置共用冻结 AgriFM＋FPN 分类分支、无损失 GMM 的 D 教师和双时序专家，通过弱标签混合系数选择监督目标。

| 配置 | 系数 | 专家拟合软监督 | 学生软目标 |
|---|---|---|---|
| `pure_moe_lambda0.json` | 0 | 教师概率 pT | MoE 后验 pM |
| `weak_mix_lambda025.json` | 0.25 | 0.25y＋0.75pT | 0.25y＋0.75pM |

其中 y 为原弱标签的 one-hot 向量。两版均执行普通 CE3 → D 教师12轮 → 双时序专家30轮 → 学生12轮。日历专家与时间对齐专家各有0.5先验，通过观测证据计算后验。原版的高斯光谱似然和多模板结构保留。

主要训练目标与公式的逐项对应见 [训练与公式](docs/训练与公式.md)。本包供方法审阅，核心公式经过 CPU 数值对照，完整 GPU 流程尚待复现。

## 历史实验结果

| 版本 | Dev mIoU | Test mIoU | 学生选中轮次 |
|---|---|---|---|
| 系数0 | 89.0877% | 88.8688% | 0，保留D教师 |
| 系数0.25 | 89.1905% | 88.7568% | 1 |

系数0版实际训练的最佳学生为第1轮，Dev87.4175%。两版都完成专家30轮及学生12轮。`results/` 保存历史指标、区域结果和逐轮记录，`provenance/` 保存原文件哈希。这里的指标来自归档实验。整理后的代码需要在目标GPU环境重新运行，才能得到该代码包自身的复现实测结果。评估为seed42、历史复用Test500。

## 运行环境与数据

使用Python3.10及以上，在目标机器安装匹配CUDA的PyTorch，然后安装依赖。

```bash
pip install -r requirements.txt
python -m pytest -q
python -m d_temporal_moe stages
```

数据包括Train1296、Dev60、Test500，影像26×10×256×256，类别0/1及忽略255。按两个配置的 `paths` 填写数据与官方AgriFM权重路径。相对路径相对于配置文件所在目录。请同步填写两份配置中相同的数据路径。第三方骨干源码和许可证在 `third_party/AgriFM/`。

## 先训练共享教师

使用系数0配置完成以下四步。两份配置的 `teacher_run_dir` 指向同一个目录，保证随后使用同一个教师。教师训练阶段与混合系数独立。

```bash
python -m d_temporal_moe --config configs/pure_moe_lambda0.json check-data
python -m d_temporal_moe --config configs/pure_moe_lambda0.json warmup
python -m d_temporal_moe --config configs/pure_moe_lambda0.json neighbors
python -m d_temporal_moe --config configs/pure_moe_lambda0.json train-teacher
```

教师保存于 `outputs/shared_d_teacher/`。已存在阶段会进行来源检查或拒绝覆盖，请按阶段顺序执行。

## 分别运行两版后续训练

下面给出系数0的全部步骤。运行混合版时，将每条命令的配置替换为 `configs/weak_mix_lambda025.json`，从 `prepare-experts` 开始执行即可。

```bash
python -m d_temporal_moe --config configs/pure_moe_lambda0.json prepare-experts
python -m d_temporal_moe --config configs/pure_moe_lambda0.json fit-experts
python -m d_temporal_moe --config configs/pure_moe_lambda0.json make-targets
python -m d_temporal_moe --config configs/pure_moe_lambda0.json train-student
python -m d_temporal_moe --config configs/pure_moe_lambda0.json seal
python -m d_temporal_moe --config configs/pure_moe_lambda0.json test
```

学生输出分别位于 `outputs/pure_moe_lambda0/student/` 与 `outputs/weak_mix_lambda025/student/`。教师与学生均在epoch0..12按Dev mIoU选最早最优，固定阈值0.5，专家固定使用第30轮。学生保留epoch0时，报告教师保留及最佳已训练轮次。

## 独立影像推理

```bash
python -m d_temporal_moe.predict --config configs/pure_moe_lambda0.json --checkpoint outputs/pure_moe_lambda0/student/checkpoints/best.pt --cube data/cubes/SAMPLE_ID/x_cube.npy --output prediction.npy
```

推理加载官方冻结骨干和选中的FPN检查点。输出为256×256玉米概率图。

## 审阅入口

| 文件 | 内容 |
|---|---|
| `d_temporal_moe/runtime.py` | 配置与共享教师路径、源码指纹 |
| `d_temporal_moe/training/teacher.py` | 等权邻域联合似然训练D |
| `d_temporal_moe/training/experts.py` | 混合锚点、专家拟合及学生目标生成 |
| `d_temporal_moe/models/temporal_moe.py` | 光谱似然、时间位移边缘化及固定混合目标 |
| `d_temporal_moe/training/student.py` | FPN训练、Dev选择与封存 |
| `docs/数据说明.md` | 数据布局与评估协议 |
| `docs/审阅说明.md` | 整理改动、历史来源与验证范围 |

原始数据与训练权重通过配置提供。正式训练与整图评估要求CUDA。CPU测试覆盖数学和模块接口。研究代码的项目级许可证由作者选择，第三方Apache2.0许可和原始声明已保留。
