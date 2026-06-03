# Hybrid query omits experimental-setup / hardware chunks that exist in the indexed doc

**Date:** 2026-06-02
**Severity:** medium (retrieval completeness)
**Component:** `mcp__edgequake__query` (mode=hybrid)
**Status:** WON'T FIX (2026-06-03) — root cause is a multi-aspect query, not a retrieval bug. See Resolution.

> ## Resolution — WON'T FIX (query design, not a system bug)
>
> Diagnosed to **query-embedding dilution**: the repro query bundles ~6 unrelated
> sub-questions (threat model + trusted hardware + TEE-vs-GPU split + what's
> protected + security basis + 1.5–4.3× overhead). Embedding-based retrieval
> represents that as ONE averaged vector dominated by the majority aspects, so
> the minority "hardware" aspect ranks below the cut. Proven: the same hardware
> chunk (`chunk-7`, "2 GPU devices… simulates the TEE") ranks **#1 at cosine
> 0.98** for a hardware-focused query, but is absent for the 6-facet query.
> (Compounded by chunk-side dilution: that GPU sentence is buried in a chunk
> that is mostly experimental-methodology prose.)
>
> The fixes considered — per-keyword multi-query, LLM query decomposition, MMR,
> re-chunking — each add latency, an LLM hop, or a corpus reprocess to paper over
> what is fundamentally a malformed query. Not worth it.
>
> **Mitigation instead:** steer callers to issue **one focused query per aspect**
> (and combine results themselves) via guidance in the `mcp__edgequake__query`
> tool/param descriptions. Multi-aspect "kitchen-sink" queries are documented as
> an anti-pattern.
>
> Note: the two *related* sub-causes from the original report ARE fixed —
> table-bound numbers (commit `feb1e7cf`) and appendix-after-References chunking
> (commit `a71b7fd4`). This Resolution applies only to the prose-hardware
> ranking case.

## Symptom

A hybrid `query` about a paper's method/threat-model/overhead does **not** surface the
experimental-setup / hardware chunks, even though those chunks are present in the same
indexed document (confirmed via `document_get_md` and against the original PDF). Downstream
this made me record "hardware not reported" for papers that **do** report it — a false negative.

## Concrete reproduction

- **Document:** `4f7ce548-5021-4985-9602-51daa95cb31d` — "ObfuscaTune: Obfuscated Offsite
  Finetuning and Inference of Proprietary LLMs on Private Datasets" (Frikha et al., arXiv 2407.02960).
  Status `completed`, 9 chunks, 122 entities.
- **Query (hybrid):** asked for ObfuscaTune's threat model, TEE fraction, overhead (1.5–4.3×),
  security basis. Returned good chunks for those, but **none containing the hardware setup**.
- **Ground truth in the same doc** (`document_get_md` on `4f7ce548…`, §5 + Appendix A):
  - "In each ObfuscaTune experiment, we use **2 GPU devices**, one that is placed outside of
    TEE and another that **simulates the TEE**."
  - "We did all experiments on **middle-range GPUs**. Each experiment took between **1 and 8
    GPU hours**."
  - "model obfuscation … **less than 10 seconds on a middle range GPU** for a GPT2-XL model."
- **PDF cross-check** (ar5iv `https://ar5iv.labs.arxiv.org/abs/2407.02960`): matches the
  converted MD verbatim — so the PDF→MD conversion is faithful; the gap is purely in query
  retrieval, not ingestion.

## Expected vs actual

- **Expected:** a hybrid query whose intent could include setup details (or a follow-up query
  naming "hardware"/"experimental setup") should retrieve the §5 / Appendix chunks that contain
  the GPU/runtime info.
- **Actual:** those chunks were not in the returned set; only method/abstract/results chunks came
  back. Relying on `query` output alone yields incomplete coverage of the document.

## Impact

Anyone using `query` (rather than `document_get_md`) to populate a structured per-paper table
(hardware, comm cost, etc.) will get false "not reported" cells. Workaround: use
`document_get_md` for completeness-sensitive extraction; treat `query` as recall-incomplete.

## Suspected cause / suggestions

- Chunk ranking under-weights experimental-setup/appendix sections for method-oriented queries
  (possible: appendix chunks scored low; or top-k too small for multi-facet questions).
- Consider: (a) a higher/configurable top-k for hybrid; (b) section-aware retrieval so
  "experimental setup"/"appendix" chunks are reachable; (c) surfacing a "doc coverage" hint when
  a query touches a doc but returns < N of its chunks.

## Notes — confirmed scope of the false-negative (re-checked via `document_get_md`)

Re-pulled the full MD for 6 more query-grounded docs and grepped for the experimental-setup
hardware line. **4 of 7 (incl. ObfuscaTune) are confirmed query false-negatives** — hardware
present in the full MD but absent from the hybrid-query result:

| Doc | Hardware present in full MD (missed by query) |
|---|---|
| ObfuscaTune `4f7ce548` | "2 GPU devices, one simulating the TEE"; "middle-range GPUs"; 1–8 GPU-h (§5 + App. A) |
| DP-Forward `4d8024c6` | "We run experiments on a cluster with **Tesla P100 GPUs**" (§5.1) |
| RemoteRAG `82e9a235` | "Ubuntu 22.04 server … two 28-core **Intel Xeon Gold 5420+** … two **Nvidia A40 48GB GPUs**" (§5.1) |
| Compass `7b372edb` | "Google Cloud … **n2-standard-8** client (8 vCPU/32 GB) … **n2-highmem-64** server (64 vCPU/512 GB)"; tc-simulated 3 Gbps/1 ms ↔ 400 Mbps/80 ms (§6.1) |
| SGT (Stained Glass) `0a63e015` | "trained on a **single Nvidia A100 80GB** for roughly six hours"; large models "up to **64 nodes of 8 Nvidia A100 80GB**, up to 2 days, FSDP2 × Tensor-Parallelism" |
| TwinShield `d5e3c42a` | "a server powered by an **Intel Xeon Gold 6342 CPU @2.8GHz**, **512GB DRAM**, **NVIDIA A40 48GB GPU**; SGX" (also Xilinx Alveo U280 FPGA, Google TPU v3-8) — query had returned only generic "SGX + GPU" |

**6 of ~15** query-grounded docs re-checked are confirmed hardware false-negatives (the hardware
is in a prose §Experimental-Setup / §System-Setup line that hybrid query did not return).
A related but distinct ingestion bug — **tables (which hold communication cost and per-component
latency) are dropped to `![tbl_…]` placeholders during PDF→MD conversion** — is filed separately
in `2026-06-02-md-conversion-drops-table-content.md`; that one is unrecoverable via either
`query` or `document_get_md`. The other 3
(OSNIP `2d50f87d`, SPARSE `e2e3fd43`, DP-KSA `253897c2`) genuinely do **not** report compute
hardware anywhere in the full MD — correct negatives. (SCX `c82a73e5` checked separately: full
MD has only related-work GPU mentions, no own testbed machine — correct negative / thin.)

**Takeaway:** hybrid-query recall is incomplete for experimental-setup / hardware content even
when the query intent could cover it; for completeness-sensitive extraction, `document_get_md`
must be used. The 4 false-negatives all had the hardware in a §Experimental-Setup or §Appendix
chunk that hybrid retrieval did not return.

## Exact queries used (for reproduction)

All run with `mode: hybrid` against the default workspace. Each returned good method/threat/
results chunks but **no chunk containing the hardware** that exists in the doc.

- **ObfuscaTune** (`4f7ce548`) — *strongest repro: the query explicitly asks for hardware and
  still gets none back:*
  > "ObfuscaTune threat model and design: is the server honest-but-curious or malicious, **what
  > hardware is trusted (TEE/confidential VM)**, what fraction of computation runs inside the TEE
  > versus the GPU, what is protected (proprietary model weights and the private data), the
  > security basis, and the reported inference overhead (1.5x to 4.3x)?"
  Returned: problem-statement / method / Table-1 chunks. Missed §5 "2 GPU devices, one simulates
  the TEE" and Appendix A "middle-range GPUs, 1–8 GPU hours."

- **SGT / Stained Glass** (`0a63e015`):
  > "In the Stained Glass Transform (SGT) paper by Roberts et al., what is the threat model
  > (honest-but-curious or malicious; what is trusted; what attacker knowledge is assumed), what
  > exactly is protected and what leaks, what is the security basis (formal guarantee vs
  > heuristic/DP), and what are the reported performance overhead and utility/quality numbers
  > (latency, accuracy delta vs plaintext)?"
  Returned: intro/method/utility chunks. Missed "trained on a single Nvidia A100 80GB … up to 64
  nodes of 8 Nvidia A100 80GB GPUs."

- **RemoteRAG** (`82e9a235`):
  > "In RemoteRAG (Cheng et al., privacy-preserving LLM cloud RAG), what is the threat model,
  > what is protected (query privacy) and what leaks, the security basis (DistanceDP plus PHE),
  > the privacy parameters (n, epsilon), and reported latency (e.g. 0.67s) and retrieval-quality
  > numbers vs plaintext?"
  Returned: abstract / threat-model / communication-cost chunks. Missed §5.1 "Ubuntu 22.04 server
  … two 28-core Intel Xeon Gold 5420+ … two Nvidia A40 48GB GPUs."

**Caveat on the other two false-negatives (not directly query-attributable):**
- **DP-Forward** (`4d8024c6`) — never had a *dedicated* query; its cells were populated from the
  corpus plus DP-Forward chunks that surfaced *incidentally* in other schemes' query results.
  Those incidental chunks omitted the §5.1 "cluster with Tesla P100 GPUs" line. (Same recall gap;
  just no single attributable query string.)
- **Compass** (`7b372edb`) — grounded by a sub-agent (round 1) whose exact query string isn't
  captured here; the full MD has §6.1 "Google Cloud … n2-standard-8 client / n2-highmem-64
  server" which the agent's retrieval did not surface.
