# Graphene_BP 文献特征级复现与算法管线分析组会汇报

汇报日期：2026-05-21  
汇报主题：基于 Bio-Z 特征的无袖带血压估计复现与后续项目启示

## 1. 汇报摘要

本次工作围绕 TAMU-ESP/Graphene_BP 公开代码，对 Nature Nanotechnology 论文中“Bio-Z 特征到血压估计”的核心算法流程做了特征级复现。当前复现不涉及石墨烯贴片制备、硬件采集和原始信号滤波，而是从公开仓库提供的预提取特征表出发，跑通 subject-specific AdaBoost 回归流程，并分析该路线对后续四电极阻抗项目的启发。

本轮已经完成：

- 下载并归档上游代码、配置和预提取特征文件。
- 编写现代 Python 环境兼容的复现脚本，绕开上游旧版 `pandas` 和 `scikit-learn` API 问题。
- 完成 subject 1、`training_select=3` 的 smoke test。
- 完成无测试集泄漏插补的敏感性检查。
- 完成固定随机种子的最小复跑验证。

当前最重要的结论是：该方法在特征级复现上可以跑通，误差标准差接近论文摘要级指标，但单个 smoke slice 存在明显系统偏差，尤其 SBP 的 ME 为 `-7.285 mmHg`。此外，上游默认均值插补方式存在测试集信息进入训练前处理的风险，后续项目应采用 train-only preprocessing 作为正式评估策略。

## 2. 文献与任务背景

原论文关注连续、无袖带血压监测，核心方案是使用颈部石墨烯 Bio-Z 传感器捕捉动脉脉搏相关阻抗变化，再通过机器学习模型估计 DBP 和 SBP。

与传统单一 PTT 方法相比，这篇工作的特点是：

- 使用多通道 Bio-Z 波形，而不是只依赖单一 PTT。
- 从每个心搏中提取时间、幅值、面积、比例和 IBI 等多类特征。
- 用个体化模型做 DBP/SBP 回归。
- 通过 HGCP、Valsalva、运动后、餐后、出汗后等实验段扩大血压变化范围。

本次复现的定位是算法管线层复现，重点回答三个问题：

- 公开特征和代码是否能跑通可重复实验。
- 复现结果与论文摘要级指标是否同量级。
- 后续项目使用类似四电极阻抗原理时，哪些环节最值得优先验证。

主要参考：

- GitHub 仓库：https://github.com/TAMU-ESP/Graphene_BP
- 论文页面：https://www.nature.com/articles/s41565-022-01145-w
- PhysioNet 数据页：https://physionet.org/content/bp-graphene-bioimpedance/1.0.0/

## 3. 数据与复现范围

本次使用的数据是 GitHub 仓库中的预提取特征文件：

`upstream/Data/features/2020-11-05/f2_ma20_mn1_mean_all.csv`

数据概况：

| 项目 | 数值 |
| --- | ---: |
| 文件大小 | 36 MB |
| CSV 行数 | 29810 行，含表头 |
| 样本行数 | 29809 |
| 本次匹配特征数 | 54 |
| 默认窗口 | 20 beat 均值特征 |
| 标签 | DBP、SBP |

复现边界：

- 包含：预提取 Bio-Z 特征读取、缺失值处理、subject-specific AdaBoost、DBP/SBP 分别建模、指标统计。
- 不包含：原始 Bio-Z 解调、滤波、关键点检测、石墨烯贴片工艺、硬件前端设计、逐图复刻论文图表。

## 4. 算法管线

本次复现的算法流程如下：

```text
公开特征 CSV
  -> 按 subject_id 选择单个受试者
  -> 按 training_select 选择实验协议
  -> 根据正则选择 Bio-Z 特征列
  -> 缺失值均值插补
  -> 构造训练/测试 split
  -> DBP 和 SBP 分别训练 AdaBoost 回归模型
  -> 输出逐样本预测和误差
  -> 汇总 CC、ME、STD、MAE、RMSE
```

模型设置：

- 模型：AdaBoostRegressor + DecisionTreeRegressor。
- 训练方式：每个 subject 单独建模。
- 目标变量：DBP 和 SBP 分开训练。
- 默认复现策略：保留上游逻辑，先对整张 subject 表做均值插补，再切分训练/测试。
- 对照策略：新增 `--imputer train_only`，仅用训练集拟合插补器，再应用到测试集。

本次 smoke test 使用的实验场景：

| 字段 | 设置 |
| --- | --- |
| `training_select` | 3 |
| 场景名 | `SingleTrain_WithBaseValsalva` |
| 训练数据 | HGCP、baseline、Valsalva |
| 测试数据 | postExerciseHgcp |
| subject | 1 |
| 测试样本数 | 94 |

## 5. 复现实现

新建复现目录：

`/Users/lizitai/Desktop/四电极阻抗/graphene_bp_reproduction`

核心文件：

| 文件 | 作用 |
| --- | --- |
| `scripts/run_reproduction.py` | 现代环境兼容的复现 runner |
| `scripts/summarize_results.py` | 汇总指标并生成 Markdown 报告 |
| `scripts/download_assets.py` | 下载最小上游资产 |
| `docs/algorithm_pipeline_analysis.md` | 算法管线和迁移分析 |
| `docs/reproduction_log.md` | 本次复现实验日志 |
| `results/smoke/` | 忠实复现 smoke test 输出 |
| `results/smoke_train_only_imputer/` | train-only 插补敏感性检查输出 |

当前机器环境：

| 项目 | 版本 |
| --- | --- |
| Python | `/opt/miniconda3/bin/python3` |
| numpy | 2.3.5 |
| pandas | 3.0.0 |
| scikit-learn | 1.7.2 |

注意：当前工作目录中的普通 `python3` 指向另一个 Python 3.11 环境，没有安装 numpy。因此运行复现时需要显式使用：

```bash
/opt/miniconda3/bin/python3 scripts/run_reproduction.py --smoke
```

## 6. 主要复现结果

忠实复现 smoke test，也就是使用上游同类的 global mean imputation，结果如下：

| Target | N | CC | ME | STD | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DBP | 94 | 0.704 | 3.667 | 4.522 | 4.825 | 5.803 |
| SBP | 94 | 0.708 | -7.285 | 5.910 | 8.021 | 9.361 |

论文摘要级参考值：

| Target | Paper ME | Paper STD |
| --- | ---: | ---: |
| DBP | 0.2 | 4.5 |
| SBP | 0.2 | 5.8 |

与论文参考的直观比较：

| Target | 本次 ME | 本次 STD | Paper ME | Paper STD | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| DBP | 3.667 | 4.522 | 0.2 | 4.5 | STD 接近，但存在正偏差 |
| SBP | -7.285 | 5.910 | 0.2 | 5.8 | STD 接近，但负偏差明显 |

解读：

- 本次 smoke test 的误差离散度与论文摘要级 STD 接近，说明特征和模型可以捕捉一定 Bio-Z 与血压变化关系。
- ME 明显偏离论文整体报告值，说明单个 subject、单个测试段不足以代表论文全集结果。
- SBP 系统低估更明显，提示模型跨协议泛化或训练/测试血压分布差异可能较大。

## 7. 插补敏感性分析

上游代码的默认逻辑是在切分训练/测试前，对整个 subject 表进行均值插补。这会使测试集分布参与预处理统计量。为了评估该问题，本次增加 train-only imputer 对照：

| Target | N | CC | ME | STD | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DBP | 94 | 0.699 | 7.359 | 4.281 | 7.683 | 8.502 |
| SBP | 94 | 0.754 | -6.923 | 5.548 | 7.309 | 8.853 |

与忠实复现相比：

| Target | Global ME | Train-only ME | Global RMSE | Train-only RMSE |
| --- | ---: | ---: | ---: | ---: |
| DBP | 3.667 | 7.359 | 5.803 | 8.502 |
| SBP | -7.285 | -6.923 | 9.361 | 8.853 |

结论：

- DBP 对插补策略非常敏感，去掉测试集信息后偏差和 RMSE 都变大。
- SBP 在这个 smoke slice 中略有改善，但仍有较大系统偏差。
- 后续项目正式评估必须把所有预处理都放在训练集内拟合，测试集只做 transform。

## 8. 可重复性检查

为了确认 runner 的可重复性，使用 shuffled CV 场景做最小复跑：

```bash
/opt/miniconda3/bin/python3 scripts/run_reproduction.py --training-select 0 --subjects 1 --max-folds 1 --output-dir results/repro_shuffle_a
/opt/miniconda3/bin/python3 scripts/run_reproduction.py --training-select 0 --subjects 1 --max-folds 1 --output-dir results/repro_shuffle_b
cmp -s results/repro_shuffle_a/summary_by_subject.csv results/repro_shuffle_b/summary_by_subject.csv
cmp -s results/repro_shuffle_a/predictions.csv results/repro_shuffle_b/predictions.csv
```

结果：

- 两次 `summary_by_subject.csv` 字节级一致。
- 两次 `predictions.csv` 字节级一致。
- 固定 seed 后，当前最小随机切分实验可重复。

## 9. 对后续项目的启示

### 9.1 不要只依赖单一 PTT

这篇工作的优势在于使用了多维 Bio-Z 波形形态特征，而不仅是 PTT。后续项目建议同时保留：

- 关键点时间特征。
- 多通道 PTT 或相对延迟。
- 幅值、面积、比例类形态特征。
- IBI 或心率相关特征。

### 9.2 实验协议比模型更关键

模型能否学到血压变化，首先取决于训练数据是否覆盖足够大的血压动态范围。只采静息 baseline 很可能得到一个“看似稳定、实则只学均值”的模型。

后续采集建议包含：

- 静息 baseline。
- 呼吸或 Valsalva 类扰动。
- 运动后恢复过程。
- 餐后或姿态变化。
- 出汗、贴片接触变化等鲁棒性条件。
- 跨天重复佩戴测试。

### 9.3 个体化校准是短期更现实路线

该仓库采用 subject-specific 模型。对我们后续项目而言，短期更现实的定位可能是：

- 个体校准后的连续趋势监测。
- 相对血压变化追踪。
- 特定场景内的 DBP/SBP 辅助估计。

不宜过早宣称：

- 无校准跨人绝对血压预测。
- 医疗级袖带替代。
- 任意运动状态下稳定估计。

### 9.4 评估必须时间分离

随机 CV 容易高估性能，因为相邻窗口、同一协议和同一佩戴状态高度相关。正式评估应优先看：

- 时间分离测试。
- 协议分离测试。
- 跨天测试。
- 重新佩戴测试。
- 不同血压扰动来源之间的泛化。

## 10. 当前局限

本轮结果仍有几个限制：

- 只完成了 subject 1、`training_select=3` 的 smoke test，没有完整跑完 7 个协议和 6 个 subject。
- 当前比较只对照论文摘要级指标，不能等同于完整复现论文全部结果。
- 使用的是公开预提取特征，无法评估原始信号处理和关键点检测误差。
- 忠实复现路径保留了全表均值插补，存在数据泄漏风险。
- 没有做跨天完整评估，尚不能判断长期佩戴漂移。

## 11. 下一步计划

建议后续按以下顺序推进：

1. 跑完整矩阵
   - 执行 `training_select=0..6`、subject 1 到 6。
   - 生成完整 summary 和 protocol-level 对照表。

2. 增加严谨评估版本
   - 所有 scaler、imputer、feature selector 都只在训练集 fit。
   - 增加 grouped split，避免相邻窗口泄漏。

3. 做误差分解
   - 按 protocol、subject、血压范围、时间段分层统计。
   - 重点看 SBP 系统偏差来源。

4. 转向原始信号端到端复现
   - 从 PhysioNet 原始 Bio-Z 数据开始。
   - 复建滤波、心搏分割、关键点检测、特征提取。

5. 映射到自研四电极阻抗方案
   - 明确电极布置、激励频率、电流幅度、采样率、同步方式。
   - 设计覆盖血压动态范围的采集协议。
   - 先做个体内趋势验证，再考虑跨人泛化。

## 12. 组会可讲的 1 分钟总结

这次我们完成了 Graphene_BP 的特征级复现，把公开 Bio-Z 特征表接到现代 Python 环境下的 AdaBoost 回归流程，并跑通了 subject 1 的代表性 smoke test。结果显示，DBP 和 SBP 的误差标准差分别约为 `4.52 mmHg` 和 `5.91 mmHg`，与论文摘要中的 `4.5 mmHg` 和 `5.8 mmHg` 在量级上接近，但 ME 存在明显偏差，特别是 SBP 低估约 `7.29 mmHg`。敏感性分析还发现，上游默认全表均值插补会影响结果，后续项目正式评估必须采用训练集内拟合的预处理策略。总体上，这篇工作的可借鉴点不是某一个模型，而是多通道 Bio-Z 形态特征、个体化校准、血压扰动协议和严格时间分离评估这一整套管线。

