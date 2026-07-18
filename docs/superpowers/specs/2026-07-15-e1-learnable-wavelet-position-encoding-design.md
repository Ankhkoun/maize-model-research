# E1 可学习小波位置编码设计

## 目标与范围

E1 在 Exact 风格的 TSViT 直接语义分割基线中加入三个可学习 Mexican-hat 小波基。模型使用真实观测 DOY，同时学习小波尺度和受约束的中心偏移。E1 仅验证小波位置编码能否稳定接入直接监督分割，不引入 CAM、TAAP、原型学习、弱监督损失或观测质量门控，也不启动正式训练。

外部参考树 `D:\cj_swcc\_external\Exact` 和 `D:\cj_swcc\_external\TimeMIL` 保持只读。新实现全部位于主仓库。

## 方案选择

采用三个全局可学习 Mexican-hat 基，而不是固定尺度或逐样本动态生成尺度：

- 固定尺度不能满足自适应学习目标。
- 逐样本动态预测尺度会把首轮工程验证扩大为条件超网络问题，难以解释消融结果。
- 全局可学习尺度和偏移能够随训练自适应，同时保持参数少、可解释和可复现。

三个尺度响应逐通道计算，拼接为 `[B,N,T,3D]`，再使用线性投影融合回 `[B,N,T,D]`。

## 数学定义与约束

输入 token 为 `Z ∈ R^[B,N,T,D]`，真实日序为 `doy ∈ R^[B,T]`，有效帧掩码为 `valid_mask ∈ {0,1}^[B,T]`。

第 `r` 个 Mexican-hat 基定义为：

```text
delta_d(i,j) = doy_i - doy_j
a_r = clamp(raw_scale_r, a_min, a_max)
b_r = b_max * tanh(raw_shift_r)
u_r(i,j) = (delta_d(i,j) - b_r) / a_r
w_r(i,j) = (1 / sqrt(a_r)) * (1 - u_r(i,j)^2) * exp(-u_r(i,j)^2 / 2)
```

参数和支持范围固定为：

```yaml
wavelet: mexican_hat
num_bases: 3
scale_init_days: [7.0, 17.5, 35.0]
scale_min_days: 3.5
scale_max_days: 35.0
shift_init_days: [0.0, 0.0, 0.0]
shift_max_abs_days: 7.0
support_radius_days: 42.0
alpha_init: 0.01
normalization_eps: 1.0e-6
```

中心偏移 `b_r` 是可学习参数，但被限制在 `[-7,7]` 天；尺度被限制在 `[3.5,35]` 天。仅保留 `|delta_d(i,j) - b_r| <= 42` 天的核权重，因此三个核不会扩张到整条 26 帧序列。

对每个尺度，采用有效帧和绝对核权重归一化：

```text
numerator_r(i)   = sum_j valid_j * support_r(i,j) * w_r(i,j) * Z_j
denominator_r(i) = sum_j valid_j * support_r(i,j) * abs(w_r(i,j))
R_r(i) = numerator_r(i) / clamp_min(denominator_r(i), eps)
```

若查询帧无效，或没有有效邻居，则对应 `R_r(i)` 强制为零。三个响应拼接并投影：

```text
P_wave = Linear(concat(R_1, R_2, R_3))
Z_out = Z + alpha * P_wave
```

`alpha` 为可学习标量，初始化为 `0.01`。`alpha=0` 时必须严格回到无 WPE 路径。

## TSViT 接入位置与数据流

模型输入接口使用项目数据契约，不把 DOY 作为额外影像通道：

```text
images:     FloatTensor[B,T,C,H,W]
doy:        FloatTensor[B,T]
valid_mask: BoolTensor[B,T]
```

完整路径为：

```text
images
  -> patch embedding
  -> Z [B,N,T,D]
  -> add learned DOY embedding [B,1,T,D]
  -> learnable three-basis WPE [B,N,T,D]
  -> reshape [B*N,T,D]
  -> prepend K temporal class tokens
  -> masked temporal Transformer
  -> retain K class-token outputs per patch
  -> spatial Transformer per class
  -> segmentation head
  -> logits [B,K,H,W]
```

WPE 只接收真实时间 token，不能看到或生成 temporal class token。时间 padding 同时用于 WPE 聚合掩码和 temporal Transformer 的 key padding mask；class token 始终有效。

DOY embedding 继续表示每个观测的真实日序。小波核也使用真实 DOY 差，而不是假设相邻帧恒定间隔七天。

## 组件和文件边界

计划新增以下文件：

```text
src/models/wavelet_position_encoding.py
src/models/tsvit_segmentation.py
configs/models/tsvit_baseline.yaml
configs/models/tsvit_wpe_basic.yaml
tests/test_wavelet_position_encoding.py
tests/test_tsvit_wpe_equivalence.py
```

`wavelet_position_encoding.py` 只负责参数约束、核构造、掩码归一化、多尺度融合和 WPE 残差。`tsvit_segmentation.py` 负责 patch embedding、DOY embedding、temporal/spatial Transformer 和分割头。E0 与 E1 共用同一 TSViT 实现；配置中的 `wavelet.enabled` 是唯一结构差异。

## 张量接口

`LearnableWaveletPositionEncoding.forward`：

```python
forward(
    tokens: torch.Tensor,      # [B,N,T,D], floating point
    doy: torch.Tensor,         # [B,T], floating point day-of-year
    valid_mask: torch.Tensor,  # [B,T], bool
) -> torch.Tensor              # [B,N,T,D]
```

`TSViTSegmentation.forward`：

```python
forward(
    images: torch.Tensor,      # [B,T,C,H,W]
    doy: torch.Tensor,         # [B,T]
    valid_mask: torch.Tensor,  # [B,T], bool
) -> torch.Tensor              # [B,K,H,W]
```

模块显式校验 rank、批次、时间长度、特征维度、布尔掩码、图像尺寸整除关系，以及有效 DOY 的有限性。非法接口抛出 `ValueError`；全无效样本不抛异常，WPE 残差为零并保持数值有限。

## 配置关系

`tsvit_baseline.yaml` 定义 E0，关闭 WPE。`tsvit_wpe_basic.yaml` 继承相同的数据、模型宽度、深度、解码器、损失、优化器、训练计划和随机种子，只打开上述 WPE 参数。正式对照时除 WPE 外不得改变任何条件。

## 测试与验收

WPE 单元测试至少覆盖：

- 输入输出形状均为 `[B,N,T,D]`。
- 三个尺度和三个偏移始终位于配置边界内。
- 修改 padding token 的数值不会改变有效查询帧的 WPE 输出。
- 全无效输入返回原 token，WPE 残差为零且无 `NaN/Inf`。
- 前向和反向传播中的 token、核、归一化分母、尺度、偏移和梯度均无 `NaN/Inf`。
- 小波模块从未接收 class token；模型接入点通过 hook 或输入形状断言验证时间长度严格等于 `T`。

模型测试至少覆盖：

- `alpha=0` 的 E1 与权重相同的 E0 输出严格一致。
- 启用 WPE 后输出为 `[B,K,H,W]`。
- padding 时间步不进入 WPE 聚合或 temporal attention 的有效 key 集合。
- 非法尺寸、掩码类型和不一致时间长度产生明确异常。
- 小型输入可以完成一次前向、损失计算和反向传播。

本轮只运行单元测试和小型合成数据 smoke test，不下载数据、不安装依赖、不运行正式训练，也不登记正式实验指标。

## 非目标与后续升级

本轮不实现逐样本动态尺度、观测质量权重、通道门控、重复 block 注入或 WPE-only 消融。若 E1 工程验证通过，后续实验可以单独评估更宽的尺度边界、样本条件门控和观测质量感知，但不得静默改变本设计对应的 E1 配置。
