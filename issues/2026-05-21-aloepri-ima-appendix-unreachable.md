# 2026-05-21 — AloePri paper Appendix D.1 / Table 2 cells unreachable via edgequake

## Context

Investigating IMA-EmbedRow-transformer paper-vs-us disparity. The handoff claims paper reports `IMA = 0 %` on Qwen2.5-14B-Instruct (Table 2). Need to verify (a) the actual cell value, (b) paper's IMA inverter architecture, (c) paper's IMA epoch/corpus settings.

Target document: `Yu Lin et al. - TOWARDS PRIVACY-PRESERVING LLM INFERENCE VIA COVARIANT OBFUSCATION (TECHNICAL REPORT).pdf` (id `812146cf-5c60-449b-855b-60e10ea8e003`, 28 chunks).

## Failures

1. **`mcp__edgequake__document_get_md`** errors immediately:
   `Error: Cannot read properties of undefined (reading 'trim')`
   Tried with the doc UUID above. No content returned. This is a hard blocker — I cannot pull the full markdown to scan appendices by section.

2. **`mcp__edgequake__query`** with `hybrid` mode keeps returning the same §7.1 / §7.2 / §1 chunks for any IMA-related query. Specifically tried:
   - `"IMA inversion model attack hyperparameters: epochs, sequence length…"`
   - `"Appendix D.1 attack details: Inversion Model Attack IMA train neural network inverter…"`
   - `"Table 9 Table 8 detailed hyperparameters epoch settings sequence number…"`
   - `"D.1 IMA Inversion Model Attack appendix definition train inverter network input observed obfuscated embedding output recovered…"`
   - `"F.1 reproducibility setup inverter network hidden layers Qwen2 architecture training data public corpus sequence batch size epoch IMA"`
   - `"Algorithm 1 inverse key matrix generation B_inv F D null space hidden_size Z.T sample_null_rows construction embedding obfuscation"`
   - `"D.1 details about VMA IA ISA NN attack training set adversary specification permutation tau recovery success rate"`
   - `"Table 9 Table 8 detailed hyperparameters epoch settings sequence number tokens evaluated reproducibility"`

   The last four returned `No relevant context found in the knowledge base.` The first four kept surfacing the abstract / §7 / §F.3 (baselines hyperparams for SGT/DP-Forward — not the IMA attack hyperparams).

3. **Same with `local` mode** — same surface chunks.

## What I can confirm from edgequake

- Paper §7.2: *"AloePri shows strong resistance to VMA, IMA, IA, and ISA, i.e., TTRSR is **less than 15%**."*
- Paper §7.2: DP-Forward → IMA >75% TTRSR; SGT → IMA >90% TTRSR.
- Paper §F.3 contains baseline (SGT, DP-Forward, SANTEXT, RANTEXT) hyperparams — but not AloePri's IMA-attack hyperparams.

## What I cannot confirm

- The exact Table 2 cell value for AloePri vs IMA on Qwen2.5-14B-Instruct (just the "< 15 %" upper bound).
- Whether the paper explicitly specifies the IMA inverter architecture (the reference impl's `_PaperLikeIMAInverter` uses "2 hidden layers + 8 heads + AdamW lr=3e-4 wd=0 batch_size=8 epochs=2" — but I can't verify whether these are paper-authored or reference-impl-authored defaults).
- Which corpus the paper used to train the IMA inverter (CCI3? Huatuo26M-Lite? something else?).

## Workaround

Falling back to the reference impl (`vendor/aloepri-py/src/security_qwen/ima.py`) as the operational definition of "paper's IMA." Documenting in the report that the appendix is unreachable so the next session can re-pull if/when edgequake is fixed.

## Suggested fix

- Investigate `document_get_md` crash on this doc id (likely a missing field in the markdown payload).
- Improve chunking / retrieval for appendix sections — terms like "Appendix D.1" or "F.1" seem to dilute against the more numerous §7 chunks.

---

## 2026-05-21 — Diagnosis (Claude, /diagnose)

### Failure 1 — `document_get_md` trim crash: ROOT CAUSE + FIX

Backend (`crates/edgequake-api/src/handlers/pdf_upload/content.rs:181-188`) serializes:

```
{ pdf_id, filename, file_size_bytes, content_type, markdown_content: Option<String>, is_processed }
```

But every SDK (TS / Python / Go) declared the wire field as `markdown: string`. So the TS SDK's `pdfContent.markdown` was *always* `undefined`, and the MCP handler at `mcp/src/tools/document.ts:404` called `.trim()` on it unconditionally → `TypeError: Cannot read properties of undefined (reading 'trim')`. The bug shipped because the only e2e test for `document_get_md` (tests/e2e/document.test.ts:121) uses a text document, never a PDF — so the PDF code path had zero coverage.

**Patched** (this commit):
- `sdks/typescript/src/types/documents.ts:340` — `PdfContentResponse` aligned to the real wire shape; `markdown_content` is `string | undefined` to reflect `Option<String>`.
- `mcp/src/tools/document.ts:404` — read `markdown_content`, guard against `undefined`. Type system now also catches this at compile time.
- `mcp/tests/document-get-md.test.ts` — 3 regression tests covering (a) `markdown_content` present, (b) `markdown_content` missing (the original crash), (c) text-doc fallback to `doc.content`. All pass.

Python and Go SDKs have the **same drift** (`markdown: str | None = None` in `sdks/python/edgequake/types/documents.py:298`; `Markdown string` in `sdks/go/types.go:667`). Not patched here — flag for a follow-up SDK alignment PR. They are not on the MCP hot path.

### Failure 2 — Appendix unreachable: HIGH-PRIOR HYPOTHESIS, not auto-fixed

Commit `5eb16fcf` (2026-05-18) wired `edgequake_pdf::strip_references_section` into `crates/edgequake-api/src/processor/pdf_processing.rs:900`, truncating the markdown at the first line matching `^\s*(?:\d+\.?\s+)?(?:references|bibliography|works\s+cited|literature\s+cited)\s*$` *before* chunking/embedding/entity-extraction. Academic papers almost universally place References between §7 and the Appendix, so this strip silently amputates the entire appendix from retrieval. Tables 8/9 and Appendix D.1/F.1 would be physically absent from `document_chunks`.

The full markdown is still persisted to `pdf_documents.markdown_content` (the strip only affects what goes downstream), so once the `document_get_md` fix above ships, the user can pull the markdown and confirm whether the appendix survived in `markdown_content` vs whether it appears in any `document_chunks` row.

**Fix shipped (option 1)**: `strip_references_section` now finds both the start of the References block *and* the start of the next section that follows it (Appendix / Supplementary / lettered `A Notation`-style heading / any markdown heading at the same-or-higher importance level than the References heading), and splices out only the References block. When References runs to EOF (no surviving appendix in the file), behavior is unchanged.

Detection arms in `non_refs_section_heading_regex`:
- `^[AS](?i:ppendix|upplement(?:ary)?)\b` — `Appendix`, `APPENDIX`, `Supplementary Material`, etc. Capital first letter only, so lowercase `appendix` mid-paragraph won't match.
- `^[A-Z](?:\.\d+\.?|[.)])?\s+[A-Z][A-Za-z\d\- ]{0,120}$` — letter-headed appendix: `A Notation`, `A. Notation`, `A) Proofs`, `D.1 IMA inverter`, `D.1. IMA inverter`. The body class excludes commas/parens, so bibliography author-list entries (`A. Smith, J. Doe (2024). Title.`) cannot false-match.
- Plus: any markdown `#`-heading at same-or-higher level than the References heading, *except* a duplicated References heading (which keeps the scan going).

The signature changed from `fn(&str) -> &str` to `fn(&str) -> Cow<'_, str>`. Caller in `pdf_processing.rs:900` does `.to_string()` which still works via deref.

7 new unit tests cover: markdown-heading appendix, bare-heading appendix, dotted-letter `D.1` form, numbered References (`6. References`), sub-headings inside the References block (`### Primary Sources`), and the no-false-positive guard for bibliography author-list entries. All 10 existing tests still pass.

**To re-populate missing appendix chunks for the AloePri PDF**: rebuild the workspace, then re-process the document (the markdown stays intact in `pdf_documents.markdown_content` — only the downstream pipeline needs to re-run).

**Parallel issue (not auto-fixed)**: `LinkExtraction::is_past_refs` (`links.rs:51`) still uses the original "everything past References is out-of-scope" rule for PDF-coordinate link filtering (used in `repos.rs:102` for reference-implementation detection). That logic will now incorrectly drop GitHub/GitLab URLs that appear in the appendix. Lower priority than chunks/embeddings — flagged for follow-up.

