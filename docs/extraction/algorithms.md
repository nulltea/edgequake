---
title: "Algorithm Extraction and Search"
---

# Algorithm Extraction and Search

This document describes how EdgeQuake turns paper algorithms into reviewable structured records, how those records become searchable, and what tradeoffs the system makes.

## Why this exists

Papers often describe algorithms in semi-structured ways:

- numbered procedures
- inline pseudocode
- mathematical definitions
- prose that mixes assumptions, inputs, and steps

For retrieval and downstream automation, EdgeQuake needs a normalized representation:

- algorithm identity
- description
- steps
- inputs and outputs
- tags and pseudocode

The system is intentionally review-first. Extraction is allowed to be generous. Search is not. Only approved algorithms participate in workspace-wide discovery.

## Extraction design

Algorithm extraction is a 3-pass pipeline.

### Pass 1: Inventory

Goal: detect candidate algorithm blocks without committing to a final schema yet.

There are two entry paths:

- PDF path: layout detection plus vision recognition identifies actual algorithm blocks on the page
- Text path: the document is chunked and scanned in overlapping chunk pairs

The PDF path is preferred when available because it follows the visual structure of the paper instead of inferring it from flattened text.

Pass 1 is intentionally lightweight in output. It produces an inventory of likely algorithm candidates rather than full definitions. That keeps recall high and pushes the expensive structuring work into the next stage.

### Pass 2: Structured extraction

Goal: convert each candidate into a detailed, implementable algorithm record.

For each candidate block, the extractor produces fields such as:

- name
- algorithm type
- description
- steps
- inputs and outputs
- preconditions
- complexity
- mathematical notation
- pseudocode
- tags
- confidence

On text documents, chunk pairs are used because a single algorithm can straddle chunk boundaries. On layout-detected PDF blocks, each block is processed independently to avoid duplicate extractions from overlapping windows.

After extraction, results are deduplicated by normalized algorithm name. This is a deliberate simplification: it is cheap, predictable, and removes most duplicate artifacts caused by overlapping text windows. The tradeoff is that two truly distinct algorithms with nearly identical names may collapse during extraction and require re-extraction or manual correction later.

### Pass 3: Verification

Goal: quality-check the extracted set as a whole.

Verification is advisory, not blocking. It records a verification status such as:

- `pass`
- `warn`
- `fail`

This stage is meant to surface extraction quality concerns, not to replace human review. A warning or failure does not silently discard algorithms.

## Review and storage model

Extracted algorithms are stored per document and start in one of two review modes:

- `pending` by default
- `approved` immediately when the workspace is configured for auto-review

Normal operator flow is:

1. extract algorithms for a document
2. inspect the extracted set
3. approve or reject each algorithm
4. submit the reviewed document so approved algorithms are embedded

Important behavior:

- re-extraction replaces the document’s previous algorithm set
- rejected algorithms stay out of search
- approved algorithms can be embedded for semantic retrieval

This keeps extraction iterative without mixing old and new generations.

## Search design

`algorithm_search` is optimized for finding approved algorithms, not for broad document discovery.

### Scope rules

- only `approved` algorithms are searched
- if `document_id` is provided, search is restricted to that document
- if `document_id` is not provided, the system first resolves a single focus document

If the system cannot resolve exactly one source document, it fails with guidance to use `document_list(search=...)`. This is intentional. Algorithm search is meant to return algorithms from one resolved paper, not a noisy union across the workspace.

### Internal document resolution

When `document_id` is absent, the query is resolved to one paper using multiple signals:

- dense semantic retrieval over document/chunk context
- graph entity matches from the query
- supporting title and metadata evidence

Entity evidence is the highest-value signal. The design assumption is that algorithm queries often contain scheme, protocol, or paper-specific identifiers, and those should dominate generic terms like “encryption” or “distance”.

If there is no clear single-document winner, search fails instead of guessing.

### Ranking signals inside the resolved document

Once a document is resolved, candidate algorithms are ranked with three signals:

- semantic similarity over approved algorithm embeddings
- lexical scoring over algorithm identity and content fields
- graph-derived document evidence

The lexical layer is intentionally not BM25+RRF. It uses a deterministic tokenized scoring function over fields such as:

- name
- tags
- description
- type
- pseudocode
- steps
- inputs and outputs

This keeps ranking explainable and biased toward exact algorithm identity matches.

The semantic layer improves recall for long natural-language descriptions.

The graph layer is a small supporting signal. It helps when the query names entities strongly associated with one paper, but it does not override the resolved-document constraint.

## Tradeoffs

### Why three extraction passes instead of one

Separating inventory, extraction, and verification improves controllability:

- Pass 1 focuses on recall
- Pass 2 focuses on structure
- Pass 3 focuses on quality

A single-pass extraction prompt is simpler, but it is harder to tune and harder to debug when results are incomplete or duplicated.

### Why document resolution is strict

Workspace-wide algorithm search creates misleading totals and irrelevant matches when papers share broad terminology. Requiring one resolved paper makes `algorithm_search` behave like a focused lookup tool instead of a fuzzy cross-paper browser.

### Why approved-only search

Pending and rejected algorithms are editorial state, not trusted retrieval material. Keeping them out of search prevents low-quality extractions from affecting agent behavior.

## Usage guidance

Use `algorithm_search` when the query already points to one paper or one named scheme.

Good queries:

- `CAPRISE distance preserving encryption`
- `SAP Scale and Perturb approximate distance comparison preserving symmetric encryption`
- `EncSAP encryption procedure`

If the source paper is unclear, search documents first:

- `document_list(search="paper title")`
- `document_list(search="author name")`
- `document_list(search="scheme or protocol name")`

Then call `algorithm_search` with `document_id`.

Use `algorithm_list(document_id)` when you already know the paper and want the extracted set without ranking.
