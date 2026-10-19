# GitHub中的位置与更新方式

B1-repro-K4位于主仓库的`baselines/b1_repro_k4/`目录：

https://github.com/Ankhkoun/maize-model-research/tree/main/baselines/b1_repro_k4

先进入此目录，再运行README中的环境、配置、训练和测试命令。这里所说的“仓库根目录”均指本B1子项目目录；不要用主仓库的Xinjiang配置替代USA固定协议。

主仓库根目录的`.github/workflows/b1-repro.yml`在此目录变更时运行CPU契约测试。数据、伪标签、预训练和训练后权重、完整预测、本地路径配置均不上传。固定split和训练归一化元数据纳入版本管理。

更新前运行`python -m unittest discover -s tests -v`；从主仓库根目录检查`git status --short`及`git diff --cached --stat`，仅提交有意修改。不要强行加入被忽略的数据或凭据，也不要改写历史核心源码hash来掩盖实现变更。

本B1子项目未另行指定主项目开源许可证；`vendor`中的Apache-2.0许可只覆盖对应第三方代码。历史指标是原实验参照，上传及CPU测试不等于在新环境重新得到该指标。
