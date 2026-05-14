//! Direct HTTP clients for embedding paths that bypass the
//! [`edgequake_llm::traits::EmbeddingProvider`] trait.
//!
//! The trait abstracts text-only embedding providers (`embed(texts: &[String])`),
//! which is fine for the regular ingest path. Multimodal inputs — caption text
//! fused with image bytes in a single `/v1/embeddings` call — don't fit the
//! trait shape. Rather than fork the external `edgequake-llm` crate, this
//! module exposes a parallel, narrow client used only where multimodal
//! semantics are needed (today: PDF figure chunks).

pub mod multimodal;

pub use multimodal::{
    EmbeddingInput, EmbeddingRole, MultimodalEmbeddingClient, MultimodalEmbeddingError,
};
