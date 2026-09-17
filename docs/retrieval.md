# 检索与证据漏斗

```text
Search 候选 → Browse 选源 → HTML/XML section
→ 通用模板噪声过滤 → 保留坐标的 chunks
→ BM25/BGE 召回 → MiniLM 重排
→ 按预算返回正文片段 → State / Final context
```

后端支持网页与医学 XML。排序前做通用结构/模板清洗，而不是针对某个域名写特例。保留正文、短证据陈述及原始坐标/哈希；不按“参考文献”标题一刀切删除内容。

Chunk 边界兼顾中文/全角标点、闭合符号、省略号、URL 与小数。这是多语言切分支持，不代表所有语言的检索质量已验证。默认 BGE-small-en 和 MS-MARCO MiniLM 偏英语，应按实际部署语言评价效果。

BM25 与 BGE 提供候选，MiniLM 重排后执行片段/token 预算。PubMed、Semantic Scholar、web search 和文档解析服务需自行配置；公开版不携带凭证或权重。

排查证据质量时，逐层比较原始/解析文档、全部 section/chunk、实际选中片段和 Final 真正收到的输入。合法引用 ID 或干净正文都不等于实际主张被支持。
