# D＋GMM＋双时序 MoE

用于 USA 玉米二分类的完整模型代码。项目只实现下面这一条流程：

```text
时序影像与弱标签
    ↓
冻结 AgriFM ＋ FPN 普通 CE 预热 3 轮
    ↓
跨区域近邻准备 → D＋GMM 教师训练 12 轮 → 按 Dev 选择教师
    ↓
固定教师概率与 GMM 可靠性 q
    ↓
日历专家 ＋ 时间对齐专家拟合 30 轮
    ↓
生成固定软标签 → 从教师 FPN 继续训练学生 12 轮
    ↓
Dev 选模 → 封存 → Test
```

最终推理只需要 **AgriFM＋学生 FPN**。D 噪声专家、GMM 和双时序专家用于训练，不进入最终分类前向。

## 先看哪些代码

| 文件 | 负责什么 |
|---|---|
| `dgmm_moe/models/network.py` | 完整分类网络与教师噪声头；冻结和训练参数的边界 |
| `dgmm_moe/models/classifier.py` | AgriFM 接入、三尺度特征、128 通道 FPN |
| `dgmm_moe/models/noise.py` | D 的三个条件噪声矩阵与邻域联合似然 |
| `dgmm_moe/reliability/gmm.py` | 分类 CE 的两成分 GMM 与低损失责任度 q |
| `dgmm_moe/models/temporal_moe.py` | 日历专家、时间对齐专家、证据路由与软标签 |
| `dgmm_moe/training/warmup.py` | 普通 CE3 初始化 |
| `dgmm_moe/training/neighbors.py` | 仅使用 Train 的冻结特征建立跨区域近邻 |
| `dgmm_moe/training/teacher.py` | D＋GMM 教师训练 |
| `dgmm_moe/training/experts.py` | 冻结教师、拟合时序专家、生成固定软标签 |
| `dgmm_moe/training/student.py` | 学生 FPN 继续训练、Dev 选模与封存 |

数据组织见 [数据说明](docs/数据说明.md)。`tests/` 是 CPU 数学和模块集成检查；`third_party/` 仅包含实际需要的 AgriFM 骨干文件及许可。

## 模型

输入影像为 `B×26×10×256×256`，两类分别是非玉米和玉米。AgriFM 提取 `1024×64×64`、`1024×32×32`、`1024×16×16` 三尺度特征。FPN 将各层变为 128 通道，逐级上采样相加，输出 `B×2×256×256` logits。AgriFM 始终冻结，FPN 有 689,666 个可训练参数。

### D＋GMM 教师

D 使用中心弱标签与两个跨区域相似近邻弱标签。三个专家分别是可学习的二分类噪声矩阵：

\[
T_k=\begin{bmatrix}1-\alpha_k&\alpha_k\\\beta_k&1-\beta_k\end{bmatrix},\qquad k=1,2,3.
\]

矩阵行表示潜在真实类别，列表示观测弱标签。对平均观测比例、最长缺测比例和前中后三段观测比例共五个条件，使用 Train KMeans 中心作距离 softmax 路由：

\[
r_{ivk}=\operatorname{softmax}_k(-\|z_{iv}-\mu_k\|^2/5),\quad T_{iv}=\sum_k r_{ivk}T_k.
\]

近邻按冻结 F1 特征的余弦相似度检索，不使用 NDRE。中心和两个近邻共用一个潜在类别的建模假设，得到：

\[
\ell_i=-\log\sum_{c=0}^1p_\theta(c\mid X_i)\prod_{v=0}^2T_{iv}[c,\widetilde y_{iv}].
\]

GMM 对同一影像块、同一伪标签类别的 `2×2` patch 平均**分类 CE**拟合两成分高斯混合，取低损失成分责任度 q。教师的联合损失权重为 `w=0.05+0.95q`；不可用位置用单位权重。第 1、4、7、10 轮刷新，前三轮逐步启用：

\[
\mathcal L_T=(1-\lambda_e)\operatorname{mean}_i\ell_i+
\lambda_e\frac{\sum_iw_i\ell_i}{\sum_iw_i},\qquad\lambda_e=\min(1,e/3).
\]

GMM 不参与反向传播。FPN 和三个噪声矩阵一起更新。q 是弱标签可信度的统计估计，不是玉米概率或已知的正确率。

### 双时序 MoE

冻结 Dev 最优教师后，重新计算一次 Train GMM q 并固定。这里使用原始 q，不加教师损失中的 0.05 下限；不可用位置 q=1。拟合专家的软锚点是：

\[
a_i(c)=q_i\mathbf1[\widetilde y_i=c]+(1-q_i)p_T(c\mid X_i).
\]

专家只使用 `source_map==1` 的直接观测光谱以及周合成窗口的日期中点，不把插值位置当作真实观测。两个专家每类各有 3 条 `26×10` 光谱模板，共享 10 个可学习波段尺度，总计 3,130 个参数。

单模板的观测对数似然为：

\[
G_{ecr\delta}=-\frac12\sum_{t,b}m_{it}
\left[\frac{(x_{itb}-\mu_{ecrb}(d_t-\delta))^2}{\sigma_b^2}+\log(2\pi\sigma_b^2)\right].
\]

日历专家使用 `δ=0`；时间对齐专家对 `δ∈{−14,−7,0,7,14}` 天按 `(1,2,3,2,1)/9` 边缘化。模板也按均匀先验边缘化。两个专家先验均为 0.5，类别先验为全 Train 软锚点均值。

\[
 p_M(c\mid x_i)\propto\rho_c\sum_e\pi_e\exp L_{eic}.
\]

专家权重由观测的概率证据产生，没有额外 MLP 门控。模板采用软锚点加权 KMeans 初始化，再用类条件混合似然拟合 30 轮；不按 Dev 挑选专家轮次。

### 学生分类器

最终固定软标签为：

\[
t_i(c)=q_i\mathbf1[\widetilde y_i=c]+(1-q_i)\operatorname{stopgrad}p_M(c\mid x_i).
\]

没有直接观测的位置，将公式中的 MoE 后验换成教师概率；原弱标签为 255 的位置不参与损失。学生 FPN 从所选教师初始化，使用软标签 CE 继续训练：

\[
\mathcal L_S=-\operatorname{mean}_{i\in\mathrm{valid}}\sum_ct_i(c)\log p_\theta(c\mid X_i).
\]

这一阶段只更新 FPN，不再额外乘 q 权重，也不再使用 D 的邻域联合损失。

## 运行

Python 3.10 及以上；在训练机器安装与其 CUDA 版本匹配的 PyTorch，再安装其余依赖：

```bash
pip install -r requirements.txt
python -m dgmm_moe stages
python -m pytest -q
```

先按 `configs/usa_seed42.json` 填写数据与官方预训练权重路径。相对路径按配置文件所在目录解析。方法预算固定在各训练模块中，本版本不提供多模型或超参数搜索入口。

```bash
python -m dgmm_moe check-data
python -m dgmm_moe warmup
python -m dgmm_moe neighbors
python -m dgmm_moe train-teacher
python -m dgmm_moe prepare-experts
python -m dgmm_moe fit-experts
python -m dgmm_moe make-targets
python -m dgmm_moe train-student
python -m dgmm_moe seal
python -m dgmm_moe test
```

以上命令应按顺序执行；不会自动启动后续阶段。真实训练与影像推理要求 CUDA。检查、预热、教师训练、专家准备、拟合、目标生成和学生训练均使用本项目内的代码，不需要其他实验文件夹。输出保存在配置的 `run_dir`，已有训练目录会拒绝覆盖。

| 阶段 | 优化器与预算 | 更新对象 |
|---|---|---|
| CE3 | AdamW；3 轮；FPN lr=1.5e−4 | FPN |
| D＋GMM 教师 | AdamW；12 轮；FPN lr=1.5e−4；噪声 lr=1e−4 | FPN 与三个噪声矩阵 |
| 双时序专家 | Adam；30 轮；lr=0.01；batch=8192 | 模板与共享波段尺度 |
| 学生 | AdamW；12 轮；lr=1.5e−4 | FPN |

分类器阶段使用 micro batch=1、梯度累积=8、weight decay=1e−4、梯度裁剪=1、BF16；Train1296 每轮 162 次更新。教师和学生分别在第 0 至 12 轮按 Dev mIoU 选择最早达到最大值的权重，固定阈值 0.5。

## 单独推理

学生训练完成后，可以单独加载学生权重和官方 AgriFM 权重预测，不需要教师权重、近邻缓存、GMM 或时序专家文件：

```bash
python -m dgmm_moe.predict --checkpoint outputs/usa_seed42/student/checkpoints/best.pt --cube data/cubes/样本编号/x_cube.npy --output prediction/maize_probability.npy
```

输出是 `256×256` 玉米概率；使用与训练一致的编码器归一化和 CUDA BF16 前向。

## 结果与验证范围

对应历史 USA 实验的最终结果为：Dev mIoU **90.1497%**，Test mIoU **90.5939%**，玉米 Precision **96.2474%**、Recall **92.0204%**、F1 **94.0864%**、OA **95.2460%**、Kappa **0.901151**。历史学生选中继续训练的第 3 轮。

这些是 seed42 的历史归档指标，Test500 已被历史研究使用。整理版经过 CPU 数学与模块检查，尚未重新完成端到端 GPU 训练及数据集评估，因此这些数值不能写成整理后代码已经重新复现的结果。完整方法继承教师并增加了训练预算，不能仅用该数值归因于某一个模块。

数据、预训练权重和训练产物不随代码发布。依赖清单不是历史运行环境的逐版本锁文件。第三方许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)；项目自身代码的发布许可尚未指定。
