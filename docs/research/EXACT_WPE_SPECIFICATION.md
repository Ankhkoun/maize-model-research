# Exact-Style Baseline And WPE Specification

## E0：Exact 风格直接分割基线

保留 Exact 的空间 patch embedding、temporal Transformer、spatial Transformer 和 DOY embedding。使用用户自己的时序 cube 与像元伪标签，从时空特征直接输出 `[B,K,H,W]` logits。

首版排除 CAM、prototype、TAAP、原弱监督辅助损失，以及只服务于 CAM 的分类头。

## E1：DOY 感知多尺度 Mexican-hat WPE

首个受控创新模型为：

```text
Z_tilde = Z + P_DOY + alpha * gate(Z) * P_wave
```

Mexican-hat 核：

```text
psi(u) = (1 - u^2) * exp(-u^2 / 2)
w_r(delta_d) = 1/sqrt(a_r) * psi((delta_d - b_r) / a_r)
```

`a_r = softplus(raw_a_r) + eps`，`b_r` 初始为 0。优先使用真实 DOY 差值。对 `Z[B,N,T,D]` 的每个尺度，使用 valid mask 做绝对核权归一化聚合；首版质量权重 `q=1`。

多尺度响应拼接并投影回 `D`，门控为：

```text
gate(Z) = sigmoid(W_gate(LayerNorm(Z)))
```

## 首版参数

```yaml
wavelet_enabled: true
wavelet_num_scales: 3
wavelet_scales_days: [7.0, 21.0, 42.0]
wavelet_shift_init_days: [0.0, 0.0, 0.0]
wavelet_radius_days: 42.0
wavelet_alpha_init: 0.01
wavelet_gate: channel
wavelet_injection: temporal_input
wavelet_use_doy: true
wavelet_use_quality: false
```

## 唯一注入位置

```text
Z [B,N,T,D]
  -> add DOY and gated WPE
  -> reshape [B*N,T,D]
  -> prepend temporal class tokens
  -> temporal Transformer
```

WPE 不得作用于 temporal class token，首版不得在每个 temporal block 重复注入。

## 强制验收

- `wavelet_enabled=false` 或 `alpha=0` 时严格回到 E0 路径。
- 改变 padding 帧不影响有效帧 WPE 输出。
- 全无效输入返回有限数值且 WPE 为零。
- class token 不进入 WPE 聚合。
- WPE 输出、归一化分母和梯度无 NaN/Inf。
