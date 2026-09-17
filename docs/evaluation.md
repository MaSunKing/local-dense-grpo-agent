# 评测与独立复核

区分结构验证、真实执行和语义正确。Unit tests 或 HTTP 成功都不能证明引用被支持或策略效果更好。

采用留出题目，冻结 requirements，不向策略暴露私有评价要点。固定 SFT/RL adapter，在相同 backbone、检索配置与声明的采样条件下对照；保存首次输出、格式失败、工具事件和精确 capture。

分别报告局部评分、Final completeness / fidelity / citation、工具使用、有效生成比例、未解决判断和证据 provenance。失败样本不能悄悄丢掉，不可观察奖励不能补零。小样本、一次更新只是集成验证，不构成稳定效果提升的统计证据。

Reviewer 可检查原题、固定 requirements、answer units、实际 attached evidence、完整 Judge input 和原始判断。需区分被支持的改写、部分支持、无根据推论和明确反证；“没有证据”不是 contradicted。

独立审核结果和修正需不可变保存并版本化，通过可信评分 authority 再进入原有 compiler gate。人工或 ChatGPT 复核不能绕过 token identity、provenance、完整题组和 behavior-policy 检查。
