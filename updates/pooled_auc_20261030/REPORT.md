# 合并AUC定向优化实测（2026-10-30）

完整五折、同一60名开发被试、2400条10秒记录的最高探索性候选为 **84.752986% AUC**；本轮起点84.585556%，增加0.167431个百分点。抑郁组由75.660000%至75.934375%。尚未达到85%，不是官方测试分数。

主要变化是把源训练数据的健康/抑郁强制50:50权重改为实际人数比例，并在条件后验之后直接优化全体正负记录对的排序损失。保留87来源模型，另加按源数据推断临床状态控制的模型族残差。推理不读取诊断、标签或trial编号；模型族只由已冻结成员类型确定。

三个近似并列候选均保存：

|候选|合并AUC|抑郁组AUC|
|---|---:|---:|
|clinical_family_pair|84.752986%|75.934375%|
|pooled_posterior_pair|84.752431%|75.940625%|
|subbag32_rawmean|84.752639%|75.833125%|

三者差异只涉及少量排序对，不能把这点数值差认定为可靠泛化优势。保存最高数值候选，同时保留更简单的排序融合和被试级重采样方案。

## 所有尝试的记录

本轮共记录59个输出，包括原始和校准版本；下表是每条路径内部数值最高结果，全部结果见CSV。多次尝试已经使用同一开发集，表格不是独立确认。

|路径|该路径最高输出|合并AUC|抑郁组AUC|
|---|---|---:|---:|
|pooled_weight_fusion_20261030|pooled_conditional_BCE87_calibrated|84.748819%|75.803750%|
|pooled_posterior_rank_20261030|pooled_posterior_pair_calibrated|84.752431%|75.940625%|
|source_reliability_fusion_20261030|reliability_conditional_BCE87_calibrated|84.687083%|75.580000%|
|pooled_86_control_20261030|pooled_conditional_BCE86_calibrated|84.751736%|75.779375%|
|score_views_fusion_20261030|posterior_control_calibrated|84.748819%|75.803750%|
|pooled_temperature_fusion_20261030|pooled_confidence_temperature_calibrated|84.695833%|75.720625%|
|ordinal_svm_20261030|ordinal_svm_20261030|75.597847%|68.644375%|
|ordinal_fusion_20261030|pooled_conditional_BCE87_calibrated|84.707431%|75.838125%|
|ambiguity_residual_fusion_20261030|linear88_conditional_calibrated|84.702778%|75.818750%|
|pooled_patient_bag_20261030|subbag32_rawmean_calibrated|84.752639%|75.833125%|
|family_gate_fusion_20261030|clinical_family_pair_calibrated|84.752986%|75.934375%|

## 验证与适用范围

保存的五折融合模型已独立重载，在隐藏标签、诊断、训练标记和trial编号后，替换被试ID并打乱记录顺序，最大预测差小于5e-13；10/12/14位小数的AUC相同。原基础模型此前已经核验，本轮不宣称重新训练或独立重放所有深度编码器。

融合参数只拟合各外层源训练被试的内层预测。新增参数固定后再评价外层；沿用原源内层校准策略，没有声称额外完成严格更深层的元模型校准交叉验证。不能在已有OOF特征上随意再切元验证集并称为完整嵌套验证。

最高候选相对84.585556%的按被试、健康/抑郁分层配对重采样95%差值区间为[-0.0527, 0.3819]个百分点，包含0，且未校正多重尝试。因此数值提高已复现，真实泛化改善仍未获统计确认。

封存集及官方公开测试均未读取，未提交任何竞赛结果。该验证是60人开发集的合并AUC，不能当作官方80条测试结果。

## 冻结与复现

冻结位置：`frozen_baselines/candidate_847530_20261030`；`FREEZE_COMPLETE.json`记录每个文件SHA256及旧基线依赖。原84.42%、84.5656%、84.5856%冻结版本均保留。

本地完整归档含本轮融合头、源码、预测、配置、检查结果和CSV；私有仓库的轻量更新含最高候选五折融合头、源码、各路径摘要、CSV和核验记录，另两候选也保留在本地完整归档中。它们依赖原工程、87来源特征及既有基础模型，**不是可独立运行的完整比赛推理包**。没有重写原有提交文件。

## 方法参考

- [RankNet原论文](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)：借鉴成对排序损失，本实现通过条件概率的雅可比反向传播，不声称复现其网络。
- [Bandt与Pompe，Permutation Entropy](https://doi.org/10.1103/PhysRevLett.88.174102)：本轮独立实现排列熵、加权排列熵及Hjorth特征；其新增SVM未带来融合收益。
