# Exact-Style Baseline And WPE Specification

## E0：Exact 风格直接分割基线

保留 Exact 的空间 patch embedding、temporal Transformer、spatial Transformer 和 DOY embedding。使用用户自己的时序 cube 与像元伪标签，从时空特征直接输出 `[B,K,H,W]` logits。

首版排除 CAM、prototype、TAAP、原弱监督辅助损失，以及只服务于 CAM 的分类头。

## E1：DOY 感知多尺度 Mexican-hat WPE

首个受控创新模型为：

```text
Z_doy = Z + P_DOY
Z_tilde = Z_doy + alpha * P_wave
```

Mexican-hat 核：

```text
psi(u) = (1 - u^2) * exp(-u^2 / 2)
w_r(delta_d) = 1/sqrt(a_r) * psi((delta_d - b_r) / a_r)
```

`a_r = clamp(raw_a_r, 3.5, 35.0)`，`b_r = 7 * tanh(raw_b_r)`，且 `b_r` 初始为 0。使用真实 DOY 差值。对 `Z[B,N,T,D]` 的每个尺度，使用 valid mask 做绝对核权归一化聚合，并将支持范围限制为 `|delta_d-b_r| <= 42` 天；首版质量权重 `q=1`。

三个尺度的响应在通道维拼接，经无偏置线性层投影回 `D`。首版只使用一个全局可学习残差系数 `alpha`，不使用样本级或通道级 gate。

## 冻结参数

```yaml
wavelet:
  enabled: true
  scale_init_days: [7.0, 17.5, 35.0]
  scale_min_days: 3.5
  scale_max_days: 35.0
  shift_init_days: [0.0, 0.0, 0.0]
  shift_max_abs_days: 7.0
  support_radius_days: 42.0
  alpha_init: 0.01
  eps: 1.0e-6
```

在固定 7 天时间槽上，三个初始尺度约对应 1、2.5、5 个时间点；中心平移约束为正负 1 个时间点；支持半径约为前后各 6 个时间点。早期草案中的 `[7.0,21.0,42.0]` 已由本冻结参数取代，不得用于正式 E1。

## 唯一注入位置

```text
Z [B,N,T,D]
  -> add learned DOY embedding
  -> inject alpha-scaled WPE once
  -> reshape [B*N,T,D]
  -> prepend temporal class tokens
  -> temporal Transformer
```

WPE 不得作用于 temporal class token，首版不得在每个 temporal block 重复注入。

## 强制验收

- `wavelet.enabled=false` 或 `alpha=0` 时严格回到 E0 路径。
- 改变 padding 帧不影响有效帧 WPE 输出。
- 全无效输入返回有限数值且 WPE 为零。
- class token 不进入 WPE 聚合。
- WPE 输出、归一化分母和梯度无 NaN/Inf。
