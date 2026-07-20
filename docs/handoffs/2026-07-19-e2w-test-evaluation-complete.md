# 2026-07-19 E2-W 训练与冻结 Test 完成交接

## 冻结资产

- 分支：`codex/e2w-pt-mexican-hat`；基础 HEAD：`ae7ccffaab8bf0d5f18d63b58c1657686f91e29c`。
- 配置：`configs/models/tsvit_e2w_pt_mexican_hat_k5.yaml`；SHA256 `71E5FE24D29738A64D5138E7B796D573AED868D0A09EA8EC9D001364B9BFB247`。
- 输出：`E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e2w_tsvit_pt_mexican_hat_k5_seed42`。
- best：epoch13/global_step43251，SHA256 `A74C8A33030172E94020410D1E46FB3439C4438ECC1E796F06CEA347DF859428`。
- terminal：epoch25/global_step83175/bad_epochs12；`last.pt` SHA256 `84054E53669B05D66FD23D48BCF3D0959591E5A7F65B933907CAFC2C8B7F36DA`。
- 数据：Xinjiang2021 Train495/Validation276/Test305；24x24/patch2；seed42；physical/effective batch16/16。

## Validation 冻结与 Test 边界

Test 前新进程完整重放 Validation 276：IoU/F1/Kappa `0.936911543/0.967428323/0.922812490`，混淆矩阵 `[[4663806,213474],[222120,6468902]]`，与 checkpoint 逐项一致。重放 JSON SHA256 `FF8D39F054EC96AF351EA4EEBECE6CAEF4908FA2EEDCD60A1578E248F478E05F`。

随后只运行一次伪标签 Test 进程和一次独立30m Test 进程。没有 Test 驱动阈值、调参、模型修改、checkpoint 选择或重训练。

## Test 结果

| 口径 | IoU | F1 | Kappa | E2-W−E0 IoU/F1/Kappa |
| --- | ---: | ---: | ---: | --- |
| 伪标签一致性 | 0.923915 | 0.960453 | 0.890835 | `-0.003592/-0.001937/-0.004126` |
| 独立原生30m | 0.781160 | 0.877136 | 0.779780 | `-0.003937/-0.002476/-0.002242` |
| 标签复制网格 | 0.774401 | 0.872859 | 0.772183 | `-0.003185/-0.002019/-0.001468` |

- 伪标签 Test：305 tiles / 12,295,425 pixels；CM `[[4142798,209790],[410501,7532336]]`；JSON SHA256 `D4586E36003DDD8629E2F23F6165B02754A81F08C050865484ABBD3B1F4F850E`。
- 原生30m：305 tiles / 2,203,625 cells；CM `[[1108997,94475],[145073,855080]]`；JSON SHA256 `CC35CF6B812E8C2DA1C940B323961BE50F03F5C160BD55BFA36B7462BCC9F211`。
- 标签复制网格：305 tiles / 19,832,625 pixels；CM `[[9947087,884161],[1346005,7655372]]`；JSON SHA256 `4D8DD8BEEF70192DF4A8736F4CFEA2C80FFCD53FC2D258CB52452A7C905569AD`。

## 可解释参数与 AMP

- learned scales `[0.824393,1.099981,1.321753]`，shifts `[0.245128,0.221329,0.201444]`，gamma `-0.097273`。
- sigmoid gates min/mean/max `0.291193/0.491921/0.701593`；参数量 1,662,603。
- 6 次 `wavelet.gamma` 同批 backoff，scale `8192→4096→2048→1024→512→256→128`，未跳 effective batch。

## 结论与 Git 边界

E2-W 比 E1 更接近 E0，但 Validation、伪标签 Test 和独立原生30m Test 均未超过 E0，继续保留 E0。E2-W 同时替换 learned `P_T` 并加入 five-point WPE，不能解释为单独 WPE 效应。

当前未 stage、commit、push、切换分支或创建 PR；用户负责后续 Git 操作。
