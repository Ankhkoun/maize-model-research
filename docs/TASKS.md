# Tasks

## P0：启动前核验

- [ ] 核验 `configs/paths.local.yaml` 指向的工作区结构。
- [ ] 确认二分类或三分类语义；低置信、nodata 和未标注像元优先使用 `ignore_index`。
- [ ] 确认 Xinjiang 2021 cube、伪标签、置信度、农田掩膜和 30m 参考标签的精确路径。
- [ ] 将空间独立 train/validation/test 样本清单固化到 `manifests/`。
- [ ] 记录波段顺序、归一化参数、时间槽 DOY 和有效帧定义。

## P1：E0 基线

- [ ] 建立独立 Python 环境并冻结依赖。
- [ ] 实现数据适配器和数据契约测试。
- [ ] 实现 Exact 风格 temporal-spatial Transformer 直接分割基线。
- [ ] 验证单批次过拟合、输出尺寸、ignore mask 和固定 seed。
- [ ] 接入现有 10m 到 30m 空间评价流程。

## P2：E1 WPE

- [ ] 实现 DOY 感知多尺度 Mexican-hat WPE 独立模块。
- [ ] 验证 `alpha=0`/关闭 WPE 时严格回到 E0 路径。
- [ ] 验证 padding、全无效输入、class token 隔离和有限梯度。
- [ ] 在 temporal Transformer class-token 拼接前仅注入一次 WPE。

## P3：受控实验

- [ ] 先运行 E0/E1 小规模调试实验。
- [ ] 使用同一 manifest、配置和 seed 运行正式 E0/E1。
- [ ] 在验证 AOI 选择 checkpoint 和阈值，在测试 AOI 只报告一次最终结果。
- [ ] 登记每次实验的 commit、配置、数据版本、seed、结果路径和指标。
