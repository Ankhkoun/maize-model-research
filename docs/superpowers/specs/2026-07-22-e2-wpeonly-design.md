# E2-WPEonly 预注册设计

## 研究问题

E2-W 同时把 E0 的 calendar-day DOY lookup 替换为 learned temporal slot `P_T`，并加入 content-only five-point Mexican-hat WPE，因此它不是纯 WPE 消融。E2-WPEonly 用严格单变量对照检验 five-point WPE 在保留 E0 物理 DOY 编码时是否改善 E0。

## 正式标识与状态

- 正式 ID：`E2-WPEonly`
- 初始状态：`not_run`
- 不复用已预留给 source-derived quality 扩展的 `E3`
- 在用户再次明确批准前，不启动正式训练

## 冻结架构

```text
patch_tokens
→ content-only FivePointMexicanHatWaveletEncoding(patch_tokens, valid_mask)
→ + E0 nn.Embedding(367, 128, padding_idx=0) calendar DOY
→ temporal class tokens
→ temporal Transformer
→ spatial Transformer
→ segmentation logits
```

不创建或使用 learned temporal slot `P_T`。Five-point WPE 不接收 DOY token，class token 不进入 WPE。无效时相不能作为 key 参与响应，无效 query 的 WPE residual 为零；边缘 offset 允许 clip。

## 唯一结构差异

E2-WPEonly 相对 E0 的唯一结构差异是启用以下 WPE：

- offsets `[-2,-1,0,1,2]`
- three bases
- scales init `[0.75,1.0,1.25]`，bounds `[0.5,1.5]`
- shifts init `[0,0,0]`，bounds `[-1,1]`
- `gamma_init=0.001`
- input/output LayerNorm
- per-base/per-channel sigmoid gates
- Mexican-hat kernel 零均值并按 L1 归一化

配置显式声明 `temporal_position_encoding.kind: doy_lookup`，该字段只是把 E0 的既有默认语义写明，不构成结构变化。

## 冻结数据与训练设置

Xinjiang 2021；Train/Validation/Test 为 495/276/305 的区域级空间隔离划分；24×24 window、patch 2、26 frames、10 bands；dim 128、temporal/spatial depth 4/4、heads 4、MLP ratio 4；Train-only normalization；seed 42；physical/effective batch 16/16；AdamW、weight decay 0；LR `1e-8 → 1e-3 → 5e-6`；warmup 10、最多 100 epoch、patience 12；普通 pixel-wise CE、`ignore_index=255`；`pseudo_confidence` 不进入 loss 或 WPE。

AMP 固定为 init scale 8192、backoff 0.5、min scale 128、每 batch 最多 6 次 backoff、growth interval 1000000。失败尝试必须重试同一 effective batch，且不推进 optimizer、scheduler 或 global step。

## 选择规则与 Test 边界

唯一选择指标为完整 Validation maize IoU。训练、early stopping、checkpoint 选择和独立 Validation 重放不得解析或加载 Test records。Test 入口不会为 E2-WPEonly 注册冻结资产；只有用户在正式训练结束、Validation 独立重放核对完成后另行明确批准，才可设计后续 Test 步骤。

## 预注册解释

- 若 E2-WPEonly Validation maize IoU 高于 E0 的 `0.938347`，支持 five-point WPE 与 calendar DOY 互补；E2-W 未超过 E0 可能与移除 DOY 或交互效应有关。
- 若不高于 E0，则不支持 five-point WPE 在此冻结设置下改善 E0；E2-W 的负结果不能只归因于 `P_T`。
- E2-WPEonly 与 E2-W 的差异可描述同一 K5 WPE 下 DOY 与 `P_T` 的条件差别，但主纯消融始终是 E2-WPEonly 对 E0。
- 单 seed 结果是受控描述性证据，不声明跨 seed 显著性。

## 验证门禁

先写失败测试，验证配置冻结、`doy_lookup`、无 `P_T`、content-only WPE 输入、WPE 后加 DOY、padding/query/class token 隔离、关闭 WPE 时与 E0 等价、CUDA 前反向有限，以及 AMP 同批重试不错误推进状态。之后运行目标单测、全量 pytest、compileall、`git diff --check`、CUDA smoke、固定 batch 过拟合和小规模 Train/Validation smoke；不运行正式训练或 Test。
