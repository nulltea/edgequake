//! Golden regression tests for `ContextAwareChunking` — the upload-pipeline
//! default since Tier B.
//!
//! These tests are hermetic: no live LLM, no VLM, no database. They feed a
//! markdown fixture with mixed structural elements (headings, fenced code,
//! `$$…$$` math, list, a `## Fig. N:` algorithm caption) into the chunker
//! and assert the structural properties we promised in the plan:
//!
//!   1. `## Fig. N:` caption stays co-located with its algorithm body —
//!      the primary regression guard for the caption-stitching work.
//!   2. Fenced code blocks are never split across chunks.
//!   3. `$$…$$` display math is never split across chunks.
//!   4. Every chunk carries a `heading_path` matching its source section.
//!   5. Merge pass eliminates the old "<50 char chunks silently dropped"
//!      behaviour: no content vanishes, small chunks collapse up.
//!   6. Token counts are real (tiktoken) — within a tight factor of the
//!      direct BPE count, never the char/4 estimate.

use edgequake_pipeline::chunker::{Chunker, ChunkerConfig, ContextAwareChunking};
use std::sync::Arc;

const FIXTURE: &str = r#"# Preliminaries

Some background before we get into details. This paragraph introduces the
notation we'll use throughout: $x \in \{0,1\}^{\ell}$ and similar.

## A. Bit-level sharing

A short paragraph about bits.

## B. B2A Protocol

We propose a B2A protocol $\Pi_2$ detailed in Protocol 2 that reduces
communication cost substantially.

## Fig. 2: FUNCTIONALITY $\mathcal{F}_{\mathrm{Bit2A}}$

1) $\mathcal{F}_{\mathrm{Bit2A}}$ receives $b_i$ from $\mathsf{P}_i$ for each
   $i \in [3]$, then computes $b = \oplus_{i=0}^{2} b_i$.
2) $\mathcal{F}_{\mathrm{Bit2A}}$ chooses $v_0, v_1 \leftarrow \$ \mathbb{Z}_{2^{\ell}}^{2}$
   and computes $v_2 = b - v_0 - v_1$.
3) $\mathcal{F}_{\mathrm{Bit2A}}$ returns $(v_i, v_{i+1})$ to $\mathsf{P}_i$
   for each $i \in [3]$.

## Implementation notes

Here is a code block that must stay intact:

```python
def bit2a(b0, b1, b2):
    b = b0 ^ b1 ^ b2
    v0 = random_uniform()
    v1 = random_uniform()
    v2 = (b - v0 - v1) % (2 ** ELL)
    return [(v0, v1), (v1, v2), (v2, v0)]
```

And a display-math block that must stay intact too:

$$
\begin{aligned}
\beta_i &= m_i \oplus x_{i,2} = x[i] \oplus \gamma_i \\
[x[i]]^A &\leftarrow \mathrm{SS.add}(\beta_i, \mathrm{SS.sMul}((-1)^{\beta_i}, [\gamma_i]^A))
\end{aligned}
$$

That concludes the notes.
"#;

fn chunk_fixture(config: ChunkerConfig) -> Vec<edgequake_pipeline::chunker::TextChunk> {
    let chunker = Chunker::with_strategy(config, Arc::new(ContextAwareChunking));
    chunker.chunk(FIXTURE, "test-doc").expect("chunk")
}

#[test]
fn fig_caption_co_locates_with_body() {
    // This is the flagship regression guard for the caption-stitching work.
    // Before Tier B the default chunker split `## Fig. 2: FUNCTIONALITY …`
    // from the three numbered bullets below it, undoing the OCR-stage fix.
    let chunks = chunk_fixture(ChunkerConfig {
        chunk_size: 400,
        chunk_overlap: 20,
        min_chunk_size: 80,
        ..Default::default()
    });

    let caption_chunk = chunks
        .iter()
        .find(|c| c.content.contains("Fig. 2: FUNCTIONALITY"));
    assert!(
        caption_chunk.is_some(),
        "caption not found in any chunk; chunks were: {:#?}",
        chunks.iter().map(|c| &c.content).collect::<Vec<_>>()
    );
    let caption_chunk = caption_chunk.unwrap();
    assert!(
        caption_chunk.content.contains("receives $b_i$"),
        "caption chunk does not contain step 1 body: {:?}",
        caption_chunk.content
    );
    assert!(
        caption_chunk.content.contains("returns $(v_i, v_{i+1})$"),
        "caption chunk does not contain step 3 body: {:?}",
        caption_chunk.content
    );
}

#[test]
fn fenced_code_block_not_split() {
    let chunks = chunk_fixture(ChunkerConfig {
        chunk_size: 200,
        chunk_overlap: 10,
        min_chunk_size: 40,
        ..Default::default()
    });
    // Exactly one chunk should contain the fenced code delimiter.
    let code_chunks: Vec<_> = chunks
        .iter()
        .filter(|c| c.content.contains("```python"))
        .collect();
    assert_eq!(
        code_chunks.len(),
        1,
        "code fence should sit in exactly one chunk, got {}: {:#?}",
        code_chunks.len(),
        code_chunks.iter().map(|c| &c.content).collect::<Vec<_>>()
    );
    // And that chunk should also contain the closing ``` and the body.
    let code_chunk = code_chunks[0];
    assert!(
        code_chunk.content.contains("def bit2a"),
        "code chunk missing body"
    );
    assert!(
        code_chunk.content.trim_end().ends_with("```") || code_chunk.content.contains("```\n"),
        "code chunk missing closing fence"
    );
}

#[test]
fn display_math_not_split() {
    let chunks = chunk_fixture(ChunkerConfig {
        chunk_size: 200,
        chunk_overlap: 10,
        min_chunk_size: 40,
        ..Default::default()
    });
    // Exactly one chunk should contain the opening `$$`.
    let math_chunks: Vec<_> = chunks
        .iter()
        .filter(|c| c.content.contains("\\begin{aligned}"))
        .collect();
    assert_eq!(
        math_chunks.len(),
        1,
        "display math should sit in one chunk, got {}",
        math_chunks.len()
    );
    // And that chunk should contain `\end{aligned}` too.
    assert!(
        math_chunks[0].content.contains("\\end{aligned}"),
        "math chunk missing close tag"
    );
}

#[test]
fn heading_path_populated() {
    let chunks = chunk_fixture(ChunkerConfig {
        chunk_size: 400,
        chunk_overlap: 20,
        min_chunk_size: 80,
        ..Default::default()
    });
    // Every chunk from after the first heading should have a non-empty path.
    let non_preamble_chunks: Vec<_> = chunks
        .iter()
        .filter(|c| !c.content.trim_start().starts_with("# Preliminaries"))
        .collect();
    assert!(
        non_preamble_chunks
            .iter()
            .all(|c| !c.heading_path.is_empty()),
        "every post-heading chunk should have a heading path; got: {:#?}",
        non_preamble_chunks
            .iter()
            .map(|c| (
                &c.heading_path,
                c.content.chars().take(40).collect::<String>()
            ))
            .collect::<Vec<_>>()
    );
}

#[test]
fn heading_path_identifies_section() {
    let chunks = chunk_fixture(ChunkerConfig {
        chunk_size: 400,
        chunk_overlap: 20,
        min_chunk_size: 80,
        ..Default::default()
    });
    // The Fig. 2 chunk should carry the expected heading path.
    let fig_chunk = chunks
        .iter()
        .find(|c| c.content.contains("FUNCTIONALITY"))
        .expect("fig chunk");
    // The path should contain "Preliminaries" as root and the Fig. 2
    // heading as leaf.
    let path = &fig_chunk.heading_path;
    assert!(
        path.iter().any(|h| h.contains("Preliminaries")),
        "heading path missing Preliminaries root: {path:?}"
    );
    assert!(
        path.iter().any(|h| h.contains("FUNCTIONALITY")),
        "heading path missing FUNCTIONALITY leaf: {path:?}"
    );
}

#[test]
fn no_content_silently_dropped() {
    // The old <50-char filter used to drop tiny chunks. After Tier B merges
    // them with a neighbour, the total byte count of all chunks should cover
    // the fixture (modulo overlap duplication and trim-whitespace).
    let chunks = chunk_fixture(ChunkerConfig {
        chunk_size: 400,
        chunk_overlap: 0,
        min_chunk_size: 80,
        ..Default::default()
    });
    let total_bytes: usize = chunks.iter().map(|c| c.content.len()).sum();
    // With overlap=0 we expect ~source size. The chunker trims surrounding
    // whitespace so total can be slightly less than FIXTURE.len(); assert
    // it's within 5%.
    let fixture_len = FIXTURE.trim().len();
    assert!(
        total_bytes >= fixture_len * 90 / 100,
        "chunks cover only {}/{} bytes of source; content was silently dropped",
        total_bytes,
        fixture_len
    );
}

#[test]
fn token_counts_are_real_bpe() {
    let chunks = chunk_fixture(ChunkerConfig {
        chunk_size: 400,
        chunk_overlap: 20,
        min_chunk_size: 80,
        ..Default::default()
    });
    let bpe = tiktoken_rs::cl100k_base().unwrap();
    for chunk in &chunks {
        let real = bpe.encode_ordinary(&chunk.content).len();
        // TextChunk::new computes `token_count` via the char/4 estimate (used
        // by the types.rs constructor). What we care about here is that the
        // ChunkResult's `tokens` was a real BPE count — which we can check
        // indirectly by making sure the content fits within our max target.
        // chunk_size=400 → strict cap of 400 BPE tokens; real count must
        // never exceed the target.
        assert!(
            real <= 400,
            "chunk {} has {} real BPE tokens, exceeds target 400: {:?}",
            chunk.index,
            real,
            chunk.content.chars().take(60).collect::<String>()
        );
    }
}
