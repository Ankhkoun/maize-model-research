# E5-DOY-K9WPE 受控实验设计

## 目标

检验 E0 的 calendar DOY 位置编码与 E4 的内容驱动 K9 Mexican-hat 小波编码能否互补，并在不改变数据、优化或选择规则的前提下，改善冻结 Validation maize IoU。

## 实验定义

- ID：`E5-DOY-K9WPE`
- 时间编码：E0 的 `doy_lookup`，保留 `nn.Embedding(367, dim)` calendar DOY 路由。
- 小波：内容驱动的三尺度 Mexican-hat，`kind: five_point_mexican_hat`，`kernel_radius: 4`，即 K9 的 `[-4,-3,-2,-1,0,1,2,3,4]` 帧偏移。
- 小波参数：`scales_init=[0.75,1.0,1.25]`、`scale_bounds=[0.5,1.5]`、`shifts_init=[0,0,0]`、`shift_bounds=[-1,1]`、`gamma_init=0.001`、`eps=1e-6`；与 E2-WPEonly/E4 一致。

## 对照与固定条件

| 实验 | DOY | WPE |
| --- | --- | --- |
| E0 | calendar DOY | off |
| E2-WPEonly | calendar DOY | content-only K5 |
| E4-ImageWPE-K9 | none | content-only K9 |
| E5-DOY-K9WPE | calendar DOY | content-only K9 |

除 E5 启用 K9 WPE 外，固定 Xinjiang 2021 Train495/Validation276/Test305 划分、24x24 窗口、patch2、seed42、batch16、manifest、Train-only normalization、损失、优化器、学习率计划、AMP 和 early stopping。

## 评估与边界

1. 训练阶段只读取 Train 与 Validation，禁止 Test records。
2. 按完整 Validation maize IoU 选择 checkpoint；E0 的冻结 Validation IoU `0.9383473661376481` 为主要参考。
3. 训练结束后，在独立新进程完整重放 Validation；须与冻结最佳 checkpoint 记录逐项一致。
4. Test 不在本设计中自动授权。仅在用户于 Validation 重放后明确授权时，登记冻结资产并分别运行一次伪标签 Test 与年度 30m reference Test。
5. 所有结果为单 seed 受控比较；不得将细小差异表述为统计显著。

## 实现与验证边界

- 添加一个冻结 YAML、正式输出目录映射与 guard：E5 必须为 `doy_lookup` 且 WPE 字典必须精确匹配 K9 参数。
- 保持 K5 默认与既有 E0/E2/E3/E4 冻结资产不变。
- 以失败测试先验证 E5 配置守卫与 DOY 路由，再实现最小改动。
- 在启动训练前运行目标测试、完整 pytest、compileall 和 `git diff --check`。
