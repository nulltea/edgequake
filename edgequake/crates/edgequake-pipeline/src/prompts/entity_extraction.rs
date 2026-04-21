//! SOTA Entity Extraction Prompts ported from LightRAG.
//!
//! Implements tuple-based extraction format with completion signals
//! and comprehensive extraction instructions.

use super::{DEFAULT_COMPLETION_DELIMITER, DEFAULT_TUPLE_DELIMITER};

/// SOTA Entity Extraction Prompts configuration.
#[derive(Debug, Clone)]
pub struct EntityExtractionPrompts {
    /// Tuple delimiter for parsing.
    pub tuple_delimiter: String,
    /// Completion signal for detection.
    pub completion_delimiter: String,
}

impl Default for EntityExtractionPrompts {
    fn default() -> Self {
        Self {
            tuple_delimiter: DEFAULT_TUPLE_DELIMITER.to_string(),
            completion_delimiter: DEFAULT_COMPLETION_DELIMITER.to_string(),
        }
    }
}

impl EntityExtractionPrompts {
    /// Create with custom delimiters.
    pub fn new(tuple_delimiter: &str, completion_delimiter: &str) -> Self {
        Self {
            tuple_delimiter: tuple_delimiter.to_string(),
            completion_delimiter: completion_delimiter.to_string(),
        }
    }

    /// Build the system prompt for entity extraction.
    ///
    /// This prompt instructs the LLM on how to extract entities and relationships
    /// in a structured tuple format.
    pub fn system_prompt(&self, entity_types: &[impl AsRef<str>], language: &str) -> String {
        let entity_types_str = entity_types
            .iter()
            .map(|s| s.as_ref())
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            r#"---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** Identify clearly defined and meaningful entities in the input text.
    *   **Entity Details:** For each identified entity, extract the following information:
        *   `entity_name`: The name of the entity. If the entity name is case-insensitive, capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
        *   `entity_type`: Categorize the entity using one of the following types: `{entity_types}`. If none of the provided entity types apply, classify it as `Other`.
        *   `entity_description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.
    *   **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **Relationship Extraction & Output:**
    *   **Identification:** Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
    *   **N-ary Relationship Decomposition:** If a single statement describes a relationship involving more than two entities (an N-ary relationship), decompose it into multiple binary (two-entity) relationship pairs for separate description.
        *   **Example:** For "Alice, Bob, and Carol collaborated on Project X," extract binary relationships such as "Alice collaborated with Project X," "Bob collaborated with Project X," and "Carol collaborated with Project X."
    *   **Relationship Details:** For each binary relationship, extract the following fields:
        *   `source_entity`: The name of the source entity. Ensure **consistent naming** with entity extraction.
        *   `target_entity`: The name of the target entity. Ensure **consistent naming** with entity extraction.
        *   `relationship_keywords`: One or more high-level keywords summarizing the overarching nature of the relationship. Multiple keywords separated by comma.
        *   `relationship_description`: A concise explanation of the nature of the relationship between the source and target entities.
    *   **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
        *   Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **Delimiter Usage Protocol:**
    *   The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
    *   **Correct Example:** `entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.`

4.  **Relationship Direction & Duplication:**
    *   Treat all relationships as **undirected** unless explicitly stated otherwise.
    *   Avoid outputting duplicate relationships.

5.  **Output Order & Prioritization:**
    *   Output all extracted entities first, followed by all extracted relationships.
    *   Within the list of relationships, prioritize those that are **most significant** to the core meaning of the input text.

6.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `our company`, `I`, `you`.

7.  **Language & Proper Nouns:**
    *   The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
    *   Proper nouns should be retained in their original language if translation would cause ambiguity.

8.  **Completion Signal:** Output the literal string `{completion_delimiter}` only after all entities and relationships have been completely extracted.

---Exclusions---
Do NOT extract the following, even when they appear named in the text:

*   **Document structure references:** `Table 1`, `Figure 3`, `Section 4`, `Appendix A`, `Equation (2)`, `Chapter 5`, `§3`, page numbers.
*   **Formal objects and their numbered labels:** `Theorem 2.1`, `Lemma 4.3`, `Protocol 4`, `Algorithm 5`, `Corollary 1`, `Proposition 3`, `Definition 2`, `Claim A.1`, `Remark 5`, `Case 3`. These are proof-structure artifacts, not entities.
*   **Single-letter variable names or symbols:** `S`, `Q`, `P`, `A`, `x`, `π`. These are mathematical notation, not entities.
*   **Generic protocol-role names used alone:** `Adversary`, `Challenger`, `Verifier`, `Prover`, `Simulator`, `Client`, `Server` — including variants with a parenthesised one-letter suffix like `Adversary (A)`. They describe a role, not a thing. Extract them only when they appear with a proper name attached (e.g. `Adversary Eve`, `Server Alice`).
*   **Authors, reviewers, or any person mentioned in the text.** Person-level granularity is captured at the document level; do not emit PERSON entities. This is why `PERSON` is absent from the allowed `entity_types` list.
*   **Real-world locations and dated events** unless they are the explicit subject of the paper (rare for technical / research material). This is why `LOCATION` and `EVENT` are absent from the allowed list.
*   **Bare dates, years, months, or publication timestamps** (`2024`, `2026-03-14`, `Q3 2025`, `last year`). Document-level dates are stored in metadata; per-mention dates rarely stand as useful entities. This is why `DATE` is absent from the allowed list.

Extract ONLY: named organisations, named systems / products / protocols / datasets / libraries / algorithms, and well-defined concepts with descriptive names. If an item is borderline, prefer omission over including it.

---Examples---
{examples}"#,
            entity_types = entity_types_str,
            tuple_delimiter = self.tuple_delimiter,
            language = language,
            completion_delimiter = self.completion_delimiter,
            examples = self.get_examples()
        )
    }

    /// Build the user prompt for extraction.
    pub fn user_prompt(
        &self,
        input_text: &str,
        entity_types: &[impl AsRef<str>],
        language: &str,
    ) -> String {
        let entity_types_str = entity_types
            .iter()
            .map(|s| s.as_ref())
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            r#"---Task---
Extract entities and relationships from the input text below.

---Instructions---
1. Strictly adhere to all format requirements for entity and relationship lists.
2. Output *only* the extracted list of entities and relationships. No introductory or concluding remarks.
3. Output `{completion_delimiter}` as the final line after all extractions.
4. Ensure the output language is {language}.

---Data to be Processed---
<Entity_types>
[{entity_types}]

<Input Text>
```
{input_text}
```

<Output>"#,
            completion_delimiter = self.completion_delimiter,
            language = language,
            entity_types = entity_types_str,
            input_text = input_text
        )
    }

    /// Build the continue extraction (gleaning) prompt.
    ///
    /// Used after initial extraction to find missed entities.
    pub fn continue_extraction_prompt(&self, language: &str) -> String {
        format!(
            r#"---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly formatted** entities and relationships from the input text.

---Instructions---
1.  **Strict Adherence to System Format:** Follow all format requirements from the system instructions.
2.  **Focus on Corrections/Additions:**
    *   **Do NOT** re-output entities and relationships that were **correctly and fully** extracted.
    *   If an entity or relationship was **missed**, extract and output it now.
    *   If an entity or relationship was **truncated or malformed**, re-output the *corrected and complete* version.
3.  **Output Format - Entities:** 4 fields per entity, delimited by `{tuple_delimiter}`.
4.  **Output Format - Relationships:** 5 fields per relationship, delimited by `{tuple_delimiter}`.
5.  **Output Content Only:** No introductory or concluding remarks.
6.  **Completion Signal:** Output `{completion_delimiter}` as the final line.
7.  **Output Language:** Ensure the output language is {language}.

<Output>"#,
            tuple_delimiter = self.tuple_delimiter,
            completion_delimiter = self.completion_delimiter,
            language = language
        )
    }

    /// Get the few-shot examples for the prompt.
    ///
    /// Examples deliberately avoid PERSON / LOCATION / EVENT output and
    /// include passages that tempt the model toward structural refs
    /// (Theorem 2.1, Table 1, Adversary A, §3) with the correct answer
    /// being to *skip* those items.
    fn get_examples(&self) -> String {
        format!(
            r#"
Example 1:
<Input Text>
We evaluate the GraphRAG pipeline on the HotpotQA benchmark using the Mistral 7B model hosted on the Ollama runtime. Indexing is backed by pgvector with an HNSW graph, and the vector store is exposed through the LangChain API wrapper.

<Output>
entity{td}GraphRAG{td}TECHNOLOGY{td}GraphRAG is the retrieval-augmented generation pipeline being evaluated.
entity{td}HotpotQA{td}PRODUCT{td}HotpotQA is a multi-hop question-answering benchmark dataset used for evaluation.
entity{td}Mistral 7B{td}PRODUCT{td}Mistral 7B is the open-weights language model used by the pipeline under test.
entity{td}Ollama{td}PRODUCT{td}Ollama is the local model-serving runtime hosting Mistral 7B.
entity{td}pgvector{td}TECHNOLOGY{td}pgvector is the Postgres extension providing vector-similarity indexing.
entity{td}HNSW{td}CONCEPT{td}HNSW is the approximate-nearest-neighbour graph algorithm used by the vector index.
entity{td}LangChain{td}PRODUCT{td}LangChain is the API wrapper through which the vector store is accessed.
relation{td}GraphRAG{td}HotpotQA{td}evaluation{td}GraphRAG is evaluated against the HotpotQA benchmark.
relation{td}GraphRAG{td}Mistral 7B{td}uses{td}The GraphRAG pipeline uses Mistral 7B as its language model.
relation{td}Mistral 7B{td}Ollama{td}hosted-on{td}Mistral 7B runs on the Ollama runtime in this setup.
relation{td}pgvector{td}HNSW{td}uses-index{td}pgvector builds an HNSW graph to accelerate similarity search.
relation{td}LangChain{td}pgvector{td}wraps{td}LangChain exposes the pgvector-backed store via its API wrapper.
{cd}

Example 2:
<Input Text>
In Theorem 2.1 we show that the protocol is secure against a malicious Adversary (A). Table 1 summarizes the concrete bounds for the AES-GCM ciphersuite under the TLS 1.3 handshake. See §3 for the proof sketch.

<Output>
entity{td}AES-GCM{td}TECHNOLOGY{td}AES-GCM is an authenticated-encryption ciphersuite whose concrete security bounds are analysed.
entity{td}TLS 1.3{td}TECHNOLOGY{td}TLS 1.3 is the transport-layer handshake under which the AES-GCM bounds are evaluated.
relation{td}AES-GCM{td}TLS 1.3{td}ciphersuite-of{td}AES-GCM is one of the ciphersuites used within the TLS 1.3 handshake.
{cd}
(Note: `Theorem 2.1`, `Adversary (A)`, `Table 1`, and `§3` are structure / role references and are intentionally omitted.)

Example 3:
<Input Text>
The Quantum Dynamics Lab published a paper on quantum entanglement in the Nature Physics journal on 2026-03-14. The research was funded by the National Science Foundation and conducted on the IBM Quantum hardware platform.

<Output>
entity{td}Quantum Dynamics Lab{td}ORGANIZATION{td}Quantum Dynamics Lab is the research institution behind the quantum-entanglement study.
entity{td}Nature Physics{td}ORGANIZATION{td}Nature Physics is the scientific journal that published the study.
entity{td}Quantum Entanglement{td}CONCEPT{td}Quantum entanglement is the physical phenomenon that is the subject of the study.
entity{td}National Science Foundation{td}ORGANIZATION{td}The National Science Foundation is the funding body for the research.
entity{td}IBM Quantum{td}PRODUCT{td}IBM Quantum is the quantum-computing hardware platform on which the experiments ran.
relation{td}Quantum Dynamics Lab{td}Nature Physics{td}publication{td}Quantum Dynamics Lab published their study in Nature Physics.
relation{td}Quantum Dynamics Lab{td}Quantum Entanglement{td}studies{td}The lab's research subject is quantum entanglement.
relation{td}National Science Foundation{td}Quantum Dynamics Lab{td}funding{td}The National Science Foundation funds the lab's research.
relation{td}Quantum Dynamics Lab{td}IBM Quantum{td}uses-platform{td}Experiments were carried out on the IBM Quantum platform.
{cd}
(Note: individual researchers are not extracted as PERSON entities; authorship is captured at the document level. The `2026-03-14` publication date is also omitted — it belongs in document metadata, not as a graph entity.)
"#,
            td = self.tuple_delimiter,
            cd = self.completion_delimiter
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_prompt_generation() {
        let prompts = EntityExtractionPrompts::default();
        let system = prompts.system_prompt(&["PERSON", "ORGANIZATION"], "English");

        assert!(system.contains("Knowledge Graph Specialist"));
        assert!(system.contains("PERSON, ORGANIZATION"));
        assert!(system.contains("<|#|>"));
        assert!(system.contains("<|COMPLETE|>"));
    }

    #[test]
    fn test_user_prompt_generation() {
        let prompts = EntityExtractionPrompts::default();
        let user = prompts.user_prompt("Test text here", &["PERSON"], "English");

        assert!(user.contains("Test text here"));
        assert!(user.contains("<|COMPLETE|>"));
        assert!(user.contains("PERSON"));
    }

    #[test]
    fn test_continue_extraction_prompt() {
        let prompts = EntityExtractionPrompts::default();
        let continue_prompt = prompts.continue_extraction_prompt("English");

        assert!(continue_prompt.contains("missed or incorrectly formatted"));
        assert!(continue_prompt.contains("<|#|>"));
        assert!(continue_prompt.contains("<|COMPLETE|>"));
    }

    #[test]
    fn test_examples_in_prompt() {
        let prompts = EntityExtractionPrompts::default();
        let system = prompts.system_prompt(&["ORGANIZATION", "CONCEPT"], "English");

        assert!(system.contains("Example 1:"));
        assert!(system.contains("Example 2:"));
        assert!(system.contains("Example 3:"));
        // Domain-appropriate anchors from the rewritten examples:
        assert!(system.contains("GraphRAG"));
        assert!(system.contains("Quantum Dynamics Lab"));
    }

    #[test]
    fn test_exclusions_section_present() {
        let prompts = EntityExtractionPrompts::default();
        let system = prompts.system_prompt(&["ORGANIZATION"], "English");

        // The exclusions are the whole point of this prompt revision —
        // if the section regresses, over-extraction returns.
        assert!(system.contains("---Exclusions---"));
        assert!(system.contains("Theorem 2.1"));
        assert!(system.contains("Adversary (A)"));
        assert!(system.contains("Table 1"));
        assert!(system.contains("PERSON entities"));
    }
}
