# EdgeQuake — project conventions

## Building & checking Rust crates

**Always pass `--features postgres` when running `cargo check` / `cargo test` /
`cargo build` against the workspace or any crate that exposes a `postgres`
feature (currently: `edgequake-api`, `edgequake-agents`, `edgequake-algorithms`,
`edgequake-core`, `edgequake-storage`, `edgequake-tasks`, and the umbrella
`edgequake` crate).**

Without the feature, several modules `#[cfg(feature = "postgres")]` get
excluded from compilation, which leaves dangling references in callers and
makes the build look broken when it isn't. The default `cargo check` output
will show errors like:

- `cannot find value algorithm_counts in this scope` (handlers/algorithms/mod.rs)
- `no field pdf_storage on type DocumentTaskProcessor` (processor/algorithm_extraction.rs)
- type-annotation errors downstream of those

These are not real errors — they disappear with `--features postgres`.

Examples:

```bash
# Single crate (must support the feature)
cargo check -p edgequake-api --features postgres
cargo test  -p edgequake-core --features postgres

# Whole workspace
cargo check --workspace --features postgres
```

Crates that do **not** expose a `postgres` feature (e.g. `edgequake-pdf`,
`edgequake-pipeline`) must be checked / tested without the flag — passing it
fails with "the package does not contain this feature".

The CI / packaging path uses this flag; local checks should too.

## Known pre-existing test-fixture breakage

`cargo test -p edgequake-core --features postgres` currently fails to build
its test binary with three `E0063` errors in `src/workspace_service.rs` —
`CreateWorkspaceRequest` test fixtures missing `chunk_min_score`,
`embedding_query_instruction`, and `enable_rerank` fields. Those fields were
added in commits 5ea7f3a4 and b4b84807 without updating the fixtures. The
production code compiles cleanly; only the test binary is affected. Fixing
the fixtures is a separate cleanup task — until then, run unit tests for
unrelated modules via crates that don't depend on edgequake-core's test
binary (e.g. `cargo test -p edgequake-pdf --lib figure_extract::`).
