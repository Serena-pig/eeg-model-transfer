# 条件融合与协方差对齐实验（2026-10-30）

本轮最高内部合并 AUC **84.5856%**，抑郁组 AUC **75.6600%**。原候选为84.5656% /75.6100%。提升仅约0.02个百分点，置信区间包含0，未达到85%，不能据此认定真实泛化提高。

最高实验候选：原83个来源模型 +3个DE/raw神经网络来源 +1个内层选择的欧氏对齐协方差SVM；融合损失为二分类交叉熵与8条记录条件似然各50%，C沿用.01，源临床分组等权。

协方差表示和SVM C只在每个外层训练集的3个内层验证中选择；所有融合参数只用源训练集的内层预测拟合。新损失系数和温度范围预先固定，未用外层结果调参。沿用旧候选的内层校准策略，本轮没有声称完成了学习型融合器全流程的更深层校准搜索。

验证覆盖60名开发被试、2400条10秒记录。没有把50秒聚合分数当作竞赛分数。原候选和84.42%基线均保留。

| 实验 | 变体 | 合并 AUC | 抑郁组 AUC |
|---|---|---:|---:|
| cached_objectives_20261030 | focal_gamma1 | 84.2633% | 75.5694% |
| cached_objectives_20261030 | conditional_bce | 84.4567% | 75.6369% |
| cached_objectives_20261030 | class_mean_align | 82.0531% | 74.6025% |
| cached_objectives_20261030 | focal_gamma1_calibrated | 84.3584% | 75.6744% |
| cached_objectives_20261030 | conditional_bce_calibrated | 84.5794% | 75.6575% |
| cached_objectives_20261030 | class_mean_align_calibrated | 82.2045% | 74.7106% |
| cached_posterior_rank_20261030 | conditional_only | 84.4440% | 75.6225% |
| cached_posterior_rank_20261030 | posterior_pair_conditional | 84.4245% | 75.5994% |
| cached_posterior_rank_20261030 | conditional_only_calibrated | 84.5785% | 75.7100% |
| cached_posterior_rank_20261030 | posterior_pair_conditional_calibrated | 84.5691% | 75.7194% |
| cached_joint_temperature_20261030 | uncertainty_temperature | 84.4530% | 75.5931% |
| cached_joint_temperature_20261030 | uncertainty_clinical_temperature | 84.3912% | 75.7338% |
| cached_joint_temperature_20261030 | uncertainty_temperature_calibrated | 84.5569% | 75.6425% |
| cached_joint_temperature_20261030 | uncertainty_clinical_temperature_calibrated | 84.4932% | 75.7650% |
| cached_nonlinear_fusion_20261030 | rbf_kernel_fusion | 81.9572% | 73.7919% |
| cached_nonlinear_fusion_20261030 | histogram_boosted_fusion | 82.8753% | 73.7688% |
| cached_nonlinear_fusion_20261030 | rbf_kernel_fusion_calibrated | 82.1762% | 73.8125% |
| cached_nonlinear_fusion_20261030 | histogram_boosted_fusion_calibrated | 83.1066% | 73.9406% |
| aligned_covariance_fusion_20261030 | BCE87 | 84.4828% | 75.6063% |
| aligned_covariance_fusion_20261030 | conditional_BCE87 | 84.4818% | 75.6469% |
| aligned_covariance_fusion_20261030 | BCE87_calibrated | 84.5791% | 75.6188% |
| aligned_covariance_fusion_20261030 | conditional_BCE87_calibrated | 84.5856% | 75.6600% |
| regional_covariance_fusion_20261030 | BCE87 | 84.4503% | 75.5719% |
| regional_covariance_fusion_20261030 | conditional_BCE87 | 84.4390% | 75.5919% |
| regional_covariance_fusion_20261030 | BCE87_calibrated | 84.5472% | 75.5413% |
| regional_covariance_fusion_20261030 | conditional_BCE87_calibrated | 84.5652% | 75.6256% |
| csp_covariance_fusion_20261030 | BCE87 | 84.4153% | 75.4681% |
| csp_covariance_fusion_20261030 | conditional_BCE87 | 84.4038% | 75.4925% |
| csp_covariance_fusion_20261030 | BCE87_calibrated | 84.5059% | 75.4906% |
| csp_covariance_fusion_20261030 | conditional_BCE87_calibrated | 84.5122% | 75.5206% |
| cached_temporal_prior_20261030 | temporal_prior_BCE | 84.4006% | 75.5106% |
| cached_temporal_prior_20261030 | temporal_prior_conditional_BCE | 84.3801% | 75.5144% |
| cached_temporal_prior_20261030 | temporal_prior_BCE_calibrated | 84.5078% | 75.5594% |
| cached_temporal_prior_20261030 | temporal_prior_conditional_BCE_calibrated | 84.4862% | 75.6025% |
| cached_evidence_logistic_20261030 | evidence_MAP | 83.9228% | 74.3656% |
| cached_evidence_logistic_20261030 | evidence_posterior_group | 84.0151% | 74.3725% |
| cached_evidence_logistic_20261030 | evidence_MAP_calibrated | 84.0025% | 74.4219% |
| cached_evidence_logistic_20261030 | evidence_posterior_group_calibrated | 84.0151% | 74.3725% |

## 核验与限制

新最高候选已独立重载5折融合模型，隐藏标签、诊断、训练标记及trial编号，替换被试ID并打乱行顺序。最大预测差5e-13；保留10、12、14位小数时AUC均一致。

协方差对齐只使用同一被试同一crop的8条无标签信号。真实数据上的顺序置换和幅值乘3不变性检查通过。CSP空间滤波只在相应训练折使用标签。新训练仅为浅层分类器和融合器，未重训原2080项基础实验。

开发集已经反复使用，结果为探索性比较，置信区间未作多重尝试校正。封存集和官方测试未读取；未进行比赛提交。候选已冻结但未自动替换稳定基线。研究代码依赖原工程及已有特征/模型，并非可直接提交的完整推理应用。

## 参考来源

- [Focal Loss原论文](https://arxiv.org/abs/1708.02002)：仅借鉴难样本加权思想，原任务为目标检测；本轮Focal结果未改善。
- [Euclidean Alignment原论文](https://arxiv.org/abs/1808.05464)：本轮采用独立实现的滤波器组、协方差收缩与矩阵对数，不声称完全复现原文。