# Trusted authority bundle

`training/train.py` 必须用 `--authority PATH` 加载由可信 collector/Judge assembler 生成的只读 JSON。

顶层字段固定为：

```json
{
  "version": "trusted_reward_authority_bundle_v1",
  "authority_id": "canonical business digest",
  "batch_sha256": "exact batch file SHA-256",
  "registries": {
    "tool_executions": {},
    "stage_scores": {},
    "evidence_gains": {},
    "terminations": {},
    "execution_manifests": {},
    "policy_violations": {},
    "gain_dispositions": {},
    "coverage_receipts": {},
    "evidence_transitions": {}
  }
}
```

安全要求：

- authority 文件不得位于 acting Agent 可写目录；
- assembler 必须从原始 capture、工具执行日志和保存的 Judge 回执重建，不得复制 batch 自报字段；
- 修改 batch 后必须重新生成 authority；
- 缺失 registry entry 返回 `None` 并由 compiler fail closed；
- 每次 Browse 必须有 `gain_dispositions` 条目；`pending/unobservable/needs_review` 不得伪装为零 gain；
- Judge 请求、coverage key、receipt 和 compiler 必须共用 `canonical_coverage_view()`；空 source metadata entry 不构成新输入；
- 相同 scorer/task scope/coverage evidence basis 必须复用同一 `coverage_receipts` 结果，identity 不得包含 step；
- 每次 Browse 必须存在以 `tool_execution_id` 为键的 `evidence_transitions`；
- 同一 rollout 的前一 transition.after 必须等于后一 transition.before，模型上下文裁剪不构成账本重置；
- `no_change` 和 `failed_no_change` 允许没有相邻 State，但仍必须绑定真实执行、读前/读后快照及规范化 source headers；
- `training/train.py` 对 batch 只读取一次 bytes，同一 bytes 同时用于 SHA、JSON 解析和运行身份；
- `build_for_test` 只用于明确标记的离线夹具，不是生产 assembler。
