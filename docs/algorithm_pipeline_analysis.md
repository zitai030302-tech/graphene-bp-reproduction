# Graphene_BP 算法管线分析

## 1. 复现边界

本目录复现的是特征级流程：使用 GitHub 仓库公开的预提取特征表，将 Bio-Z 心搏特征映射到 DBP/SBP。它不复建石墨烯贴片制备、模拟前端、原始阻抗解调、滤波、峰点检测或论文图表排版。

主要来源：

- 上游代码：https://github.com/TAMU-ESP/Graphene_BP
- 论文页面：https://www.nature.com/articles/s41565-022-01145-w
- PhysioNet 原始数据页：https://physionet.org/content/bp-graphene-bioimpedance/1.0.0/

## 2. 信号与特征

这篇工作的核心思想不是直接用单个 PTT 替代袖带血压，而是用颈部 Bio-Z 波形里的多点形态变化来回归血压。公开特征表里可以看到几类特征：

- 多通道 Bio-Z 特征：`BM1` 到 `BM4` 代表多个测量通道或通道组合。
- 时间特征：`__T`、`__PTT`、`__IBI` 等，描述波形关键点时间、脉搏传导时间和心搏间期。
- 形态特征：`__A`、`__AR`、`__IPA`、`__IPAR` 等，描述幅值、面积以及归一化形态比例。
- 生理标签：`DBP`、`SBP` 来自同步参考血压。
- 协议标签：`exp_setup_name` 区分 baseline、HGCP、Valsalva、餐后、运动后、出汗后、跨天等实验段。

配置名 `f2_ma20_mn1` 中的 `ma20` 在上游脚本里被解析为 20，用于 shuffled CV 场景下的 50% overlap 下采样。默认复现使用 `f2_ma20_mn1_mean_all.csv`，也就是 20 beat 窗口均值特征，而不是逐搏特征。

## 3. 模型与实验协议

上游默认启用 AdaBoost，分别训练 DBP 和 SBP 两个 subject-specific 模型。每个受试者单独建模，而不是跨人泛化，这是指标好看的重要前提：模型可以吸收个体血管几何、贴片位置、皮肤接触和阻抗基线差异。

七个 `training_select` 场景的含义：

- `0 ShuffleCV_hgcp`：HGCP 数据随机 10 折，带下采样。
- `1 NoShuffleCV_hgcp`：HGCP 数据顺序 10 折。
- `2 SingleTrain_hgcp`：用 HGCP 训练，测试 postExercise/postMeal/postSweat。
- `3 SingleTrain_WithBaseValsalva`：用 HGCP、baseline、Valsalva 训练，测试 postExercise/postMeal/postSweat。
- `4 SingleTrain_S1All_TestNextDay`：subject 1 day1 训练，day4 测试，检验跨天漂移。
- `5 ShuffleCV_valsalva`：Valsalva 数据随机 10 折。
- `6 NoShuffleCV_valsalva`：Valsalva 数据顺序 10 折。

忠实复现默认保留了上游一个关键实现细节：特征缺失值的均值插补在切分前对整张 subject 表拟合。这会让测试集分布进入插补均值，属于轻微数据泄漏风险。因此 runner 提供 `--imputer train_only`，用于做更保守的敏感性分析。

## 4. 对后续项目的启示

如果后续项目想借鉴相似原理，建议优先验证这几件事：

- 同步质量：Bio-Z、PPG/ECG 和参考 BP 的时间戳必须可靠。亚秒级错位都会污染 PTT 和关键点特征。
- 血压动态范围：只采安静 baseline 很容易训练出看似稳定、实际只学到均值的模型。HGCP、Valsalva、运动后、餐后等扰动是扩大标签范围的关键。
- 特征鲁棒性：峰点、谷点、面积和幅值比例比单点 PTT 更丰富，但也更依赖波形质量；运动、汗液、贴片接触阻抗和跨天位置变化都要单独测试。
- 评估方式：随机 CV 适合快速确认信号相关性，但项目落地更应该看时间分离、协议分离和跨天测试。
- 个体化校准：这条路线更像“连续趋势监测 + 个体校准”，不要过早承诺无校准跨人绝对血压。

## 5. 复现结果读取

运行 `scripts/run_reproduction.py` 后会得到：

- `predictions.csv`：逐样本预测、误差、fold、协议名和模型参数。
- `summary_by_subject.csv`：每个 subject/target/split 的 CC、ME、STD、MAE、RMSE。
- `summary_overall.csv`：跨 subject 的均值汇总。
- `pooled_metrics.csv` 和 `paper_comparison.csv`：由 `scripts/summarize_results.py` 生成，用于和论文摘要级指标比较。

