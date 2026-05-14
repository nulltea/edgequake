"""
title: EdgeQuake RAG
author: EdgeQuake
version: 0.7.1
description: Query the EdgeQuake knowledge graph, upload documents, and explore entities and relationships. v0.7.1 instructs the chat model to compose natural-language queries (re-querying only on new terms) and to render retrieved figure images inline.
"""

import json
import os
from typing import Optional

import requests
from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        edgequake_base_url: str = Field(
            default="http://host.docker.internal:8080",
            description="EdgeQuake API base URL — used by the tool's server-side HTTP client",
        )
        public_base_url: str = Field(
            default="",
            description=(
                "Public base URL used in markdown image links the chat model emits "
                "(e.g. https://edgequake.tail59ea6b.ts.net). The URL must be reachable "
                "from the END USER'S browser, not the OpenWebUI container — so this is "
                "typically the Tailscale / public proxy URL, NOT host.docker.internal. "
                "If empty, falls back to edgequake_base_url which usually only resolves "
                "inside the Docker network."
            ),
        )
        workspace_id: str = Field(
            default="00000000-0000-0000-0000-000000000003",
            description="EdgeQuake workspace ID",
        )
        tenant_id: str = Field(
            default="00000000-0000-0000-0000-000000000002",
            description="EdgeQuake tenant ID",
        )
        query_mode: str = Field(
            default="hybrid",
            description="Default RAG query mode: naive, local, global, hybrid, mix",
        )

    def __init__(self):
        self.valves = self.Valves()

    def _headers(self) -> dict:
        return {
            "X-Workspace-ID": self.valves.workspace_id,
            "X-Tenant-ID": self.valves.tenant_id,
        }

    def _api(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.valves.edgequake_base_url}/api/v1{path}"
        headers = {**self._headers(), **kwargs.pop("headers", {})}
        r = requests.request(method, url, headers=headers, timeout=120, **kwargs)
        r.raise_for_status()
        return r.json()

    def query_knowledge_base(
        self,
        query: str,
        mode: Optional[str] = None,
        __user__: dict = {},
    ) -> str:
        """
        Search the user's documents. Returns text chunks, figure images
        with captions, entities, and relationships.

        Compose the query as a short natural-language sentence reflecting
        what is being asked. Keep concrete terms (names, technical
        terminology, numbers, figure references) from the user's phrasing;
        strip conversational filler. Full phrasing retrieves better than a
        list of disjoint keywords.

        Re-query on a follow-up turn when the user introduces
        substantially new terms or topics. Rephrasing the same question
        rarely surfaces new content.

        When the response contains a **Figures** section, each figure is a
        markdown image `![caption](url)`. Copy those image lines verbatim
        into your answer where the figure is discussed — do not strip,
        paraphrase, or replace them with text. The image renders inline
        for the user.

        :param query: Natural-language question with concrete terms
            preserved from the user's phrasing.
        :param mode: One of: naive, local, global, hybrid, mix. Empty for
            default.
        """
        body = {
            "query": query,
            "mode": mode or self.valves.query_mode,
            "context_only": True,
        }
        try:
            data = self._api("POST", "/query", json=body)
        except requests.HTTPError as e:
            return f"EdgeQuake query failed: {e}"

        sources = data.get("sources", [])
        reference_code = data.get("reference_code", [])
        approved_algorithms = data.get("approved_algorithms", [])
        if not sources and not reference_code and not approved_algorithms:
            return "No relevant context found in the knowledge base."

        chunks = []
        figures = []
        entities = []
        relationships = []
        seen_docs = set()
        # WHY public_base_url: image URLs go into the chat response and are
        # loaded by the user's browser. edgequake_base_url is typically
        # host.docker.internal which only resolves inside Docker. Fall back
        # to it only as a last resort — usually the user must set the valve
        # to the publicly-reachable URL.
        public_base = (
            self.valves.public_base_url or self.valves.edgequake_base_url
        ).rstrip("/")

        for src in sources:
            stype = src.get("source_type", "")
            if stype == "chunk":
                # VLM-OCR figure chunk: render as inline markdown image so
                # OpenWebUI displays the PNG in the chat thread. The
                # /api/v1/documents/{doc}/figures/{figure_id} endpoint streams
                # the bytes the figure-extractor captured during PDF ingest.
                if src.get("kind") == "figure":
                    doc_id = src.get("document_id", "")
                    fid = src.get("figure_id", "")
                    if doc_id and fid:
                        caption = src.get("caption") or src.get("snippet") or fid
                        img_url = (
                            f"{public_base}/api/v1/documents/{doc_id}/figures/{fid}"
                        )
                        figures.append(f"![{caption}]({img_url})\n\n*{caption}*")
                        if src.get("file_path"):
                            seen_docs.add(src["file_path"])
                        continue
                snippet = src.get("snippet", "")
                doc = src.get("file_path", "")
                if snippet:
                    chunks.append(f"[{doc}]: {snippet}")
                if doc:
                    seen_docs.add(doc)
            elif stype == "entity":
                name = src.get("id", "")
                desc = src.get("snippet", "")
                entities.append(f"- {name}: {desc}")
            elif stype == "relationship":
                desc = src.get("snippet", "")
                relationships.append(f"- {desc}")

        parts = []
        if figures:
            # WHY this banner comes first: many open-weight chat models
            # (Qwen3.6, Gemma, Llama) drop image-markdown when summarising
            # tool output. Front-loading an explicit rule, in imperative
            # form, before any other content steers them to copy the
            # `![alt](url)` snippets into the final answer verbatim.
            parts.append(
                "**RENDERING RULE — important:** the **Figures** section "
                "below contains retrieved figure images as markdown "
                "`![caption](url)`. Copy each such image-markdown line "
                "verbatim into your answer where you discuss the figure. "
                "OpenWebUI will inline-render the image to the user. Do "
                "NOT strip, paraphrase, or replace these image lines with "
                "text descriptions."
            )
        if chunks:
            parts.append("**Text chunks:**\n" + "\n\n".join(chunks[:10]))
        if figures:
            # Cap at 6 — more would clutter the chat thread and bloat tokens
            # going into the chat model. OpenWebUI renders ![alt](url) inline.
            parts.append("**Figures:**\n\n" + "\n\n".join(figures[:6]))
        if entities:
            parts.append("**Entities:**\n" + "\n".join(entities[:15]))
        if relationships:
            parts.append("**Relationships:**\n" + "\n".join(relationships[:10]))
        if approved_algorithms:
            parts.append(_render_algorithms(approved_algorithms))
        if reference_code:
            parts.append(_render_reference_code(reference_code))
        if seen_docs:
            parts.append("**Source documents:** " + ", ".join(seen_docs))

        return "\n\n".join(parts)

    def list_documents(
        self,
        search: Optional[str] = None,
        __user__: dict = {},
    ) -> str:
        """
        List documents in the EdgeQuake knowledge base. Optionally filter by search term.

        :param search: Optional search term to filter documents by title or content.
        """
        params = {"page_size": 20}
        if search:
            params["search"] = search
        try:
            data = self._api("GET", "/documents", params=params)
        except requests.HTTPError as e:
            return f"Failed to list documents: {e}"

        docs = data.get("documents", data.get("items", []))
        if not docs:
            return "No documents found."

        lines = [f"**{len(docs)} document(s):**\n"]
        for doc in docs:
            title = doc.get("title", doc.get("filename", "Untitled"))
            status = doc.get("status", "unknown")
            doc_id = doc.get("id", doc.get("document_id", ""))
            lines.append(f"- **{title}** ({status}) `{doc_id}`")
        return "\n".join(lines)

    def search_entities(
        self,
        search: str,
        label: Optional[str] = None,
        __user__: dict = {},
    ) -> str:
        """
        Search for entities (people, organizations, technologies, concepts) in the EdgeQuake knowledge graph.

        :param search: Search term to find entities by name.
        :param label: Optional entity type filter: PERSON, ORGANIZATION, TECHNOLOGY, CONCEPT, EVENT, LOCATION, PRODUCT.
        """
        params = {
            "search": search,
            "limit": 15,
        }
        if label:
            params["label"] = label
        try:
            data = self._api("GET", "/graph/entities", params=params)
        except requests.HTTPError as e:
            return f"Entity search failed: {e}"

        entities = data.get("entities", data.get("items", []))
        if not entities:
            return f"No entities found for '{search}'."

        lines = [f"**{len(entities)} entities matching '{search}':**\n"]
        for ent in entities:
            name = ent.get("name", "")
            etype = ent.get("label", ent.get("type", ""))
            desc = ent.get("description", "")[:120]
            lines.append(f"- **{name}** [{etype}] — {desc}")
        return "\n".join(lines)

    def explore_entity(
        self,
        entity_name: str,
        __user__: dict = {},
    ) -> str:
        """
        Explore an entity's neighborhood in the knowledge graph — shows all directly connected entities and their relationships.

        :param entity_name: The entity name to explore (e.g. RUST, OPENAI, MACHINE_LEARNING).
        """
        try:
            data = self._api(
                "GET",
                f"/graph/entities/{requests.utils.quote(entity_name, safe='')}/neighborhood",
                params={},
            )
        except requests.HTTPError as e:
            return f"Entity exploration failed: {e}"

        entity = data.get("entity", {})
        neighbors = data.get("neighbors", data.get("relationships", []))

        parts = [f"**{entity.get('name', entity_name)}** [{entity.get('label', '')}]"]
        desc = entity.get("description", "")
        if desc:
            parts.append(desc[:300])

        if neighbors:
            parts.append(f"\n**Connections ({len(neighbors)}):**")
            for rel in neighbors[:15]:
                src = rel.get("source", rel.get("source_name", ""))
                tgt = rel.get("target", rel.get("target_name", ""))
                lbl = rel.get("label", rel.get("relationship", ""))
                other = tgt if src.upper() == entity_name.upper() else src
                parts.append(f"- {lbl} → **{other}**")
        else:
            parts.append("\nNo connections found.")
        return "\n".join(parts)

    def upload_text_document(
        self,
        content: str,
        title: Optional[str] = None,
        __user__: dict = {},
    ) -> str:
        """
        Upload a text document to the EdgeQuake knowledge base for entity extraction and indexing.

        :param content: The full text content of the document.
        :param title: Optional title for the document.
        """
        body = {
            "content": content,
            "async_processing": True,
        }
        if title:
            body["title"] = title
        try:
            data = self._api("POST", "/documents", json=body)
        except requests.HTTPError as e:
            return f"Upload failed: {e}"

        doc_id = data.get("document_id", "")
        status = data.get("status", "unknown")
        return f"Document uploaded. ID: `{doc_id}`, status: {status}. Processing will continue in the background."

    def upload_pdf(
        self,
        file_path: str,
        title: Optional[str] = None,
        __user__: dict = {},
    ) -> str:
        """
        Upload a PDF file from the server filesystem to the EdgeQuake knowledge base.

        :param file_path: Absolute path to the PDF file on the server.
        :param title: Optional title for the document.
        """
        if not os.path.isfile(file_path):
            return f"File not found: {file_path}"

        url = f"{self.valves.edgequake_base_url}/api/v1/documents/pdf"
        try:
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f, "application/pdf")}
                data = {}
                if title:
                    data["title"] = title
                r = requests.post(
                    url, files=files, data=data, headers=self._headers(), timeout=120
                )
                r.raise_for_status()
                resp = r.json()
        except requests.HTTPError as e:
            return f"PDF upload failed: {e}"
        except Exception as e:
            return f"PDF upload error: {e}"

        pdf_id = resp.get("pdf_id", "")
        status = resp.get("status", "unknown")
        pages = resp.get("metadata", {}).get("page_count", "?")
        return f"PDF uploaded ({pages} pages). ID: `{pdf_id}`, status: {status}. Processing in background."

    def upload_pdf_from_url(
        self,
        url: str,
        title: Optional[str] = None,
        __user__: dict = {},
    ) -> str:
        """
        Upload a PDF by URL. EdgeQuake fetches it server-side (HEAD
        content-type check + streaming 100 MB cap) and queues it for
        extraction. After VLM-OCR completes, the filename is renamed
        automatically to academic citation format ("Author et al. -
        Year - Title.pdf") when the paper's front-matter is extractable.
        Year is derived from the arxiv ID when the URL points to arxiv.

        Prefer this over `upload_pdf` for arxiv / publicly-hosted PDFs —
        the caller doesn't need to download or name the file.

        :param url: Direct http(s) URL to a PDF.
        :param title: Optional initial filename override (the post-OCR
            rename still runs; use this only when auto-rename is unwanted).
        """
        body: dict = {"url": url}
        if title:
            body["title"] = title
        try:
            resp = self._api("POST", "/documents/pdf/from-url", json=body)
        except requests.HTTPError as e:
            # Surface the server's readable error body when present —
            # typical causes: bad URL scheme, upstream 404, content-type
            # not PDF.
            try:
                detail = e.response.json().get("message") or e.response.text
            except Exception:
                detail = str(e)
            return f"URL upload failed: {detail}"

        pdf_id = resp.get("pdf_id", "")
        status = resp.get("status", "unknown")
        filename = resp.get("metadata", {}).get("filename", "")
        pages = resp.get("metadata", {}).get("page_count", "?")
        if status == "duplicate":
            return (
                f"PDF already in workspace ({pages} pages). "
                f"ID: `{pdf_id}`, filename: {filename}."
            )
        return (
            f"PDF fetched ({pages} pages) and queued. ID: `{pdf_id}`, "
            f"filename: {filename}. Processing in background; the title "
            f"will auto-rename to citation format once extraction completes."
        )

    def search_relationships(
        self,
        relationship_type: Optional[str] = None,
        __user__: dict = {},
    ) -> str:
        """
        Search relationships between entities in the EdgeQuake knowledge graph.

        :param relationship_type: Optional filter by relationship type (e.g. USES, IMPLEMENTS, PART_OF, RELATED_TO).
        """
        params = {"page_size": 20}
        if relationship_type:
            params["relationship_type"] = relationship_type
        try:
            data = self._api("GET", "/graph/relationships", params=params)
        except requests.HTTPError as e:
            return f"Relationship search failed: {e}"

        rels = data.get("items", [])
        total = data.get("total", len(rels))
        if not rels:
            return "No relationships found."

        lines = [f"**{total} relationship(s)** (showing {len(rels)}):\n"]
        for rel in rels:
            src = rel.get("src_id", "")
            tgt = rel.get("tgt_id", "")
            rtype = rel.get("relation_type", rel.get("label", ""))
            desc = rel.get("description", "")[:100]
            lines.append(
                f"- **{src}** —[{rtype}]→ **{tgt}**{f': {desc}' if desc else ''}"
            )
        return "\n".join(lines)


def _render_algorithms(algorithms: list) -> str:
    """Render curated algorithm definitions from the EdgeQuake query API
    into a Markdown block the chat model can treat as authoritative.

    EdgeQuake matches algorithm embeddings against the query via cosine
    search over the workspace vector store. Structured pseudocode + step
    lists are carried through explicitly so the LLM doesn't have to
    reconstruct them from paraphrased chunk text — especially useful for
    crypto protocols where step order matters. (The backend JSON field is
    still `approved_algorithms`; that workflow term is kept server-side.)"""
    lines = ["**Algorithms:**"]
    for i, a in enumerate(algorithms, start=1):
        name = a.get("name") or a.get("algorithm_id", "?")
        algo_type = a.get("algorithm_type", "Algorithm")
        confidence = a.get("confidence", "")
        header = f"[A{i}] **{name}** ({algo_type}"
        if confidence:
            header += f", confidence: {confidence}"
        header += ")"
        lines.append(header)

        desc = a.get("description")
        if desc:
            lines.append(f"_description:_ {desc}")
        complexity = a.get("complexity")
        if complexity:
            lines.append(f"_complexity:_ {complexity}")

        steps = a.get("steps") or []
        if steps:
            lines.append("_steps:_")
            for s in steps:
                num = s.get("number", "?")
                action = (s.get("action") or "").strip()
                details = (s.get("details") or "").strip()
                lines.append(f"  {num}. {action} {details}".rstrip())

        pseudocode = a.get("pseudocode")
        if pseudocode:
            lines.append("_pseudocode:_")
            lines.append(f"```\n{pseudocode.rstrip()}\n```")

    return "\n".join(lines)


def _render_reference_code(snippets: list) -> str:
    """Render reference-code snippets from the EdgeQuake query API into a
    Markdown block the chat model can treat as authoritative.

    EdgeQuake matches code snippets against the query via HNSW cosine
    search over jina-code-embeddings. These are actual implementations of
    algorithms in the retrieved papers — higher fidelity than the paper's
    prose description of the same algorithm."""
    lines = ["**Reference code implementations:**"]
    for i, s in enumerate(snippets, start=1):
        algo = s.get("algorithm_name") or s.get("algorithm_id", "?")
        file_path = s.get("file_path", "?")
        start = s.get("start_line", 0)
        end = s.get("end_line", 0)
        lang = s.get("language", "")
        snippet = s.get("snippet", "")
        repo_url = s.get("repo_url")
        repo_commit = s.get("repo_commit", "")
        rationale = s.get("match_rationale")

        header = f"[C{i}] **{algo}** — `{file_path}:{start}-{end}`"
        if repo_url and repo_commit:
            header += f" ([source]({repo_url}/blob/{repo_commit}/{file_path}#L{start}-L{end}))"
        lines.append(header)
        if rationale:
            lines.append(f"_rationale:_ {rationale}")
        fence_lang = lang if lang else ""
        lines.append(f"```{fence_lang}\n{snippet}\n```")
    return "\n".join(lines)
