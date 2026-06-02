# PDF→MD conversion drops table CONTENT (tables become image placeholders)

**Date:** 2026-06-02
**Severity:** high (data loss in ingestion — affects both `document_get_md` and `query`)
**Component:** PDF→Markdown conversion / ingestion

## Symptom

`document_get_md` returns prose, headings, and equations faithfully, but **tables are not
extracted as text** — each table is replaced by a placeholder like:

```
![tbl_6_1](edgequake-figure)
<div style="text-align: center;">Table 1: Test accuracy results (%) …</div>
```

The caption survives; the table's **numeric cells do not**. Since a large fraction of the
load-bearing quantitative results in these papers (communication cost in GB/MB, per-component
latency breakdowns, accuracy-vs-baseline tables) live *only* in tables, those numbers are
unrecoverable from the converted MD — even though the conversion is otherwise faithful (prose
hardware lines, for example, are preserved; verified against the ar5iv PDF for arXiv 2407.02960).

## Reproduction

- `document_get_md` on CipherFormer (`e89ec545-…`): the body references "Table IV … reducing
  communication overhead by 40%" but Table IV itself is `![tbl_6_3](edgequake-table)` — the
  absolute latency/communication numbers in it are gone.
- Same pattern in ObfuscaTune (`4f7ce548`, Tables 1–2), SecFormer (`87b98329`, "Table 1"
  communication), Fission (`9acbd0c0`, "Table 2: Latency (seconds) and communication (GB)"),
  CryptoMoE (`47db1317`, Table 2 per-token latency/comm). In each, the per-scheme absolute
  communication cost is table-bound and therefore absent from the MD.

## Impact

This is the root cause of many false "not reported" extractions for **communication cost** and
**fine-grained latency** (distinct from the hardware/query-recall issue in the companion report
`2026-06-02-hybrid-query-misses-hardware-chunks.md`, where the data was in prose but query didn't
return it). Here the data is genuinely *absent from the MD*, so neither `query` nor
`document_get_md` can recover it. Downstream, a structured per-paper table cannot be reliably
populated for any table-bound metric using EdgeQuake alone — one must fall back to the original
PDF (e.g. ar5iv HTML).

## Suggestion

- Extract tables to Markdown/HTML tables (or CSV) during conversion rather than rasterizing to
  `![tbl_…]` placeholders. Even a lossy text dump of cell values would recover the numbers.
- If a table-to-text model is already used for some docs (a few tables DID come through as
  `<table>` HTML — e.g. PIR-RAG, ObfuscaTune Algorithm tables), make it consistent; the
  image-placeholder path silently drops data.

## Note

Verified the *prose* fidelity separately (ar5iv cross-check of ObfuscaTune hardware lines matched
the MD verbatim), so this is specifically a **table-extraction** gap, not a general conversion
failure.
