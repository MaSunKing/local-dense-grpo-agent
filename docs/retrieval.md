# Evidence funnel

```text
Search candidates → selected Browse source → parsed HTML/XML sections
→ generic template filtering → coordinate-preserving chunks
→ BM25/BGE candidate retrieval → MiniLM reranking
→ budgeted evidence chunks → State and Final context
```

The backend supports web pages and biomedical XML sources. Generic structural/template filtering is applied before ranking, rather than domain-specific bans on a particular site. Substantive prose and short evidence-bearing statements must remain available; bibliographies are not blanket-removed. Original text coordinates and chunk hashes support provenance inspection.

Chunk boundaries handle Chinese/fullwidth punctuation, closing symbols, ellipses, URLs and decimals. This is multilingual segmentation support, not a claim of universal language quality. The default BGE-small-en and MS-MARCO MiniLM models are English-oriented; retrieval effectiveness must be tested on the deployment's languages.

BM25 lexical retrieval and BGE semantic retrieval supply candidates; MiniLM reranks candidates before passage/token budgets are enforced. Tools may use PubMed, Semantic Scholar, web search and document extraction providers. Configure service credentials locally; no keys or weights are shipped here.

Debug evidence provenance by comparing fetched/parsed document, all sections/chunks, selected passages and actual Final input. A legal citation ID or clean passage is not proof that the final attached claim is supported.
