# 超算已有数据必须满足的协议

本仓库包含固定划分、归一化和1296个训练影像的历史SHA256；不包含原始影像、伪标签、GTdev参考数组或全国CDL栅格。已有影像不代表这些配套文件齐全，先执行scripts/preflight.py确认。

设`DATA_ROOT`为超算上与原maize_paper_workspace对应的目录。代码按sample_id构造以下相对路径，不使用CSV中可能出现的原Windows路径。请保持原数据布局或在数据区建立符号链接，不修改冻结代码和清单。

| 相对DATA_ROOT路径 | 内容与要求 |
|---|---|
| `03_processed_data/cubes/usa_2021/cubes_linear_interp/2021/<sample_id>/x_cube.npy` | float32，[26,10,256,256]，反射率原始尺度约[0,1]，不得预先再归一化或乘10000 |
| 同目录`source_map.npy` | uint8，[26,256,256]，编码1/2/3/4；仅1是直接观测支持 |
| 同目录`metadata.json` | sample_id一致、crs=EPSG:3857；最终CDL还需要原空间变换等完整元数据 |
| 同目录`time_quality.json` | 原26期时相和日期记录；不能只按顺序伪造 |
| `05_pseudo_labels/sam_refined/usa_2021/vvp_mmi_intersection/<sample_id>/pseudo_label.npy` | [256,256]，0非玉米、1玉米、255无效；必须对应原intersection版本 |
| `03_processed_data/labels_30m/usa_typical_2021/2021/<sample_id>/y_patch_30m.npy` | GTdev参考首选路径，原85×85数组，或代码支持的256×256数组 |
| `03_processed_data/labels_10m/usa_2021/2021/<sample_id>/y_patch.npy` | GTdev参考备选；仅当前面路径不存在时使用，代码按原规则转85×85 |

训练必须具有Train1296和GTdev60的影像四件套及伪标签，共1356个ID。虽然GTdev最终用参考标签评分，原冻结代码仍加载GTdev伪标签以构造输入，不能省略。Validation330不用于本次固定0.5阈值选模，勿重新校准阈值。

`protocol_assets/normalization_train1296.json`含Train1296原x_cube.npy的逐文件SHA256。默认预检核验首个训练影像；加`--full-train-hashes`逐个读取1296个影像验证，读取量较大。五份协议文件均按原hash严格核验。

现有缓存没有原伪标签、source_map和GTdev参考逐文件hash，因此预检对这些只能确认存在、形状/编码及实际加载；它不能证明这些文件与原训练逐字节相同。首次超算预检会输出GTdev参考hash，供该环境后续复验。若目标分数偏离，应回原数据源补齐逐文件核验，不能偷偷重新制作标签。

数据hash默认不读取Test500；仅最终测试才加`--final-test`。CDL测试需要Test500影像四件套和2021_30m_cdls.tif，原栅格SHA256为`100f0b664367453ec38e28e1e72812af66b6308f29b5c2f166acdb708940b922`。冻结评估器不加载测试伪标签、不使用crop mask，结果应覆盖500瓦片、1959211个去重有效像元。

训练时同时使用两种归一化：AgriFM按发布统计对raw×10000归一化，A4按Train1296统计归一化；这些已在冻结data.py中实现，不要提前处理输入。
