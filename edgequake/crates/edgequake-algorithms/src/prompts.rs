//! 3-pass prompt templates for algorithm extraction.
//!
//! Ported from RAGSearcher's `algorithm_prompts.rs`.
//! These are pure functions returning prompt strings — no LLM provider dependency.

/// Pass 1: Algorithm Inventory — identify algorithms/protocols/schemes in a document.
///
/// The document text should be prepended to this prompt by the caller.
pub fn algorithm_inventory_prompt() -> String {
    r#"You are identifying algorithms, protocols, and schemes in an excerpt from a research paper.

## Your Task
Find distinct algorithms, protocols, schemes, or procedures described **within this excerpt only**.
The excerpt may be a section, page, or chunk — not the whole paper.
Only include items that have procedural/step-by-step content VISIBLE in the excerpt — skip vague mentions or things only referenced.
If no algorithms are present in this excerpt, return an empty `algorithms` array.

## What Counts as an Algorithm
- Named algorithms with defined steps (e.g., "Algorithm 1: FedAvg")
- Protocols with message flows or phases
- Schemes with defined construction/execution procedures
- Training/optimization procedures with specific update rules
- Data processing pipelines with defined stages

## What Does NOT Count
- General concepts mentioned without procedural detail
- Related work references to other papers' algorithms
- Evaluation metrics or benchmarks (unless they define a procedure)

## Output Format
Return ONLY a JSON object (no markdown fences):
{
    "paper_title": "Title of the paper",
    "algorithms": [
        {
            "id": "A1",
            "name": "Algorithm Name",
            "description": "1-2 sentence summary of what it does",
            "location": "Section name or number",
            "type": "algorithm|protocol|scheme|procedure"
        }
    ],
    "paper_type": "empirical|theoretical|system|survey"
}

## Guidelines
- Use sequential IDs: A1, A2, A3, etc.
- Keep the total response under 2000 tokens

The full paper text is available in the system prompt above."#
        .to_string()
}

/// Pass 2: Algorithm Definition Extraction — produce detailed, implementable definitions.
///
/// The document text should be prepended to this prompt by the caller.
pub fn algorithm_extraction_prompt(inventory_json: &str) -> String {
    format!(
        r#"You are extracting algorithm / protocol / functionality definitions that a software engineer will use to implement them WITHOUT access to the original paper.

## CRITICAL REQUIREMENTS
1. Each definition must be **self-contained and implementable**. The reader has NO access to the paper. Every mathematical formula, threshold, hyperparameter, and decision rule must be explicit.
2. **Mathematical precision above all.** Copy symbols from the source text VERBATIM. Do NOT paraphrase operators into English: keep `⊕` as `\oplus` (XOR), `⊗` as `\otimes`, `·` as `\cdot`, `⟨·,·⟩` as `\langle·,·\rangle`, `⊆` as `\subseteq`, etc. Never write "sum" when the source uses XOR (`⊕`). Never invent numeric values or ring moduli. If the source is ambiguous, say so in `description` rather than guessing.
3. **Valid LaTeX always.** Every `$...$` and `$$...$$` you output must parse. If the paper excerpt you received has malformed LaTeX (unclosed braces, stray backslashes, missing `$` delimiters, `\mathsf` without `{{...}}`, etc.), FIX it in your output. Balance every `{{`, close every math environment, and make sure subscripts/superscripts have braces when they span more than one character (`x_{{i,j}}`, not `x_i,j`).
4. **Exact step count.** If the source box is numbered 1..N, output exactly N steps with matching numbers and matching substance. Do not split a single numbered step into multiple imperatives; do not merge two steps into one.

## Your Task
For each algorithm in the inventory below, produce a complete structured definition.
The paper excerpt is provided above — use it to extract precise details.

## `type` field (REQUIRED)
Label each entry with the kind of construct it actually is in the paper, taken literally from the source heading/caption when present. Common values:
- `"Protocol"` — a multi-party interactive protocol (e.g. "Protocol 1 Bit2A protocol (Π₁)")
- `"Functionality"` — an idealised/ideal functionality box (e.g. "Fig. 2: FUNCTIONALITY F_Bit2A")
- `"Algorithm"` — a generic algorithm block (default when nothing more specific fits)
- `"Theorem"`, `"Definition"`, `"Lemma"`, `"Scheme"` — use when the source uses those words
Custom values are allowed when the paper uses a specific term.

## Step Format Rules
- Each step is an **imperative action** ("Compute X", "Initialize Y", "For each Z, do W")
- Include implementation-level detail in the `details` field
- Use LaTeX for ALL math: `$...$` inline, `$$...$$` display

## LaTeX Conventions (token-efficient)
- Vectors: `$\mathbf{{x}}$` or `$x_i$`
- Sums: `$\sum_{{i=1}}^{{n}}$`
- Fractions: `$\frac{{a}}{{b}}$`
- Greek: `$\alpha, \beta, \theta, \nabla$`
- Sets: `$\mathcal{{D}}, \mathbb{{R}}^d$`
- Norms: `$\|x\|_2$`
- XOR: `$\oplus$` (NEVER write "sum" for `\oplus`)

## Output Format
Return ONLY a JSON object (no markdown fences):
{{
    "algorithms": [
        {{
            "rank": 1,
            "name": "Full Algorithm Name",
            "type": "Protocol",
            "description": "2-3 sentence overview of purpose and approach",
            "steps": [
                {{
                    "number": 1,
                    "action": "Initialize model parameters",
                    "details": "Set $\\theta_0 \\sim \\mathcal{{N}}(0, 0.01)$ for all layers. Initialize learning rate $\\eta = 0.01$.",
                    "math": "$\\theta_0 \\in \\mathbb{{R}}^d$"
                }}
            ],
            "inputs": [
                {{
                    "name": "training_data",
                    "type": "Dataset of (x, y) pairs",
                    "description": "Labeled samples where $x \\in \\mathbb{{R}}^d$, $y \\in \\{{0,1\\}}$"
                }}
            ],
            "outputs": [
                {{
                    "name": "trained_model",
                    "type": "Model parameters $\\theta^*$",
                    "description": "Optimized parameters after convergence"
                }}
            ],
            "preconditions": [
                "Data is IID sampled from distribution $\\mathcal{{D}}$",
                "Loss function $\\ell$ is differentiable"
            ],
            "complexity": "O(T \\cdot n \\cdot d) where T=rounds, n=samples, d=dimensions",
            "mathematical_notation": "$$\\theta_{{t+1}} = \\theta_t - \\eta \\nabla \\ell(\\theta_t; x, y)$$",
            "pseudocode": "function train(data, T, eta):\n  theta = init_params()\n  for t in 1..T:\n    for (x, y) in data:\n      grad = compute_gradient(theta, x, y)\n      theta = theta - eta * grad\n  return theta",
            "tags": ["optimization", "gradient-descent"],
            "confidence": "high"
        }}
    ]
}}

## Self-Containment + Precision Checklist
Before finalizing each entry, verify:
1. Could someone implement this from ONLY your definition?
2. Are all variables defined before use?
3. Are all hyperparameters/thresholds/ring moduli specified with exact values?
4. Are termination conditions explicit?
5. Are edge cases mentioned in preconditions?
6. Does every `$...$` and `$$...$$` parse as valid LaTeX (balanced braces, closed environments)?
7. Are `\oplus`, `\otimes`, `\cdot`, `\langle`, `\rangle`, etc. preserved verbatim from the source (not paraphrased)?
8. Does the step count match the numbered box in the source?

If any answer is "no", fix before returning.

## Guidelines
- Extract 1-5 algorithms, ranked by importance
- Keep step count faithful to the source — do not pad or merge
- Pseudocode should be language-agnostic (no specific syntax)
- Keep the total response under 6000 tokens

## Algorithm Inventory
{inventory_json}"#
    )
}

/// Pass 3: Algorithm Verification — quality-check extracted definitions.
pub fn algorithm_verification_prompt(algorithms_json: &str) -> String {
    format!(
        r#"You are verifying the quality and implementability of extracted algorithm definitions.

## Your Task
Review each algorithm definition for completeness, correctness, and self-containment.

## Checks to Perform

### 1. Implementability
- Can each algorithm be implemented from the definition alone (no paper access)?
- Are all variables defined before use?
- Are termination conditions explicit?
- Are hyperparameters specified with concrete values or ranges?

### 2. Step Completeness
- Are steps ordered correctly?
- Are there missing intermediate steps?
- Is mathematical notation consistent across steps?

### 3. Mathematical Correctness
- Is LaTeX well-formed?
- Are dimensions/types consistent?

## Output Format
Return ONLY a JSON object (no markdown fences):
{{
    "verification_status": "pass|warn|fail",
    "completeness_issues": [
        {{
            "algorithm_rank": 1,
            "issue": "Step 3 references $\\alpha$ but it is not defined in inputs or earlier steps",
            "severity": "error|warning"
        }}
    ],
    "citation_issues": [],
    "overall_quality": "high|medium|low",
    "improvement_suggestions": ["Add learning rate schedule to Algorithm 1"]
}}

## Verification Status Guidelines
- pass: All algorithms are implementable, no errors
- warn: Minor issues (warnings only) that don't block implementation
- fail: Critical issues — undefined variables, missing steps, or invalid citations

## Guidelines
- Be concise. Keep the total response under 2000 tokens.

## Extracted Algorithms
{algorithms_json}"#
    )
}
