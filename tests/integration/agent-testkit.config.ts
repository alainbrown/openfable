// Deterministic provider behaviours for the LLM boundary tests.
//
// Each behaviour matches on a fragment of the prompt that identifies which
// call site is asking, then returns JSON that instructor parses into the
// corresponding Pydantic model. Matching on a fragment rather than the whole
// prompt keeps this from breaking every time prompt wording is edited -- but
// the fragments below are load-bearing, so if a prompt's opening changes,
// change it here too.
//
// RETRY BEHAVIOURS: instructor re-prompts on a Pydantic validation failure,
// and on that second request `latestUserText` is the validation error, not
// the original prompt. So a retry is matched by the error text, which also
// proves the validator messages in schemas/tree.py actually reach the model.
//
// unmatchedRequest is "error" so an unexpected call fails loudly rather than
// returning something plausible.
//
// PRIORITY: the server errors when several behaviours match, rather than
// taking the first. Case-specific behaviours therefore sit at priority 10 and
// the general per-call-site ones stay at the default 0.
import type { AgentTestkitDocument } from "@agent-testkit/schema";

const leaf = (i: number) => ({ type: "leaf", chunk_index: i });

const json = (value: unknown) => ({
  type: "content" as const,
  blocks: [{ type: "text" as const, text: JSON.stringify(value) }],
});

export default {
  schemaVersion: "1",
  models: ["gpt-4.1-mini"],
  behaviors: [
    // --- chunking -------------------------------------------------------
    {
      id: "chunking",
      match: { fact: "latestUserText", operator: "contains", value: "Segment the following document text" },
      respond: json({
        chunks: [
          { chunk_text: "# Alpha\n\nFirst section body.\n\n", start_idx: 0, end_idx: 30 },
          { chunk_text: "## Beta\n\nSecond section body.\n", start_idx: 30, end_idx: 60 },
        ],
      }),
    },

    {
      id: "chunking-always-invalid",
      priority: 10,
      // Matches the marked prompt, and also the validation error it provokes,
      // so every retry stays invalid and instructor exhausts them.
      match: {
        any: [
          { fact: "latestUserText", operator: "contains", value: "CHUNK-FAIL" },
          { fact: "latestUserText", operator: "contains", value: "start_idx" },
        ],
      },
      // Missing start_idx / end_idx -- ChunkResult rejects it.
      respond: json({ chunks: [{ chunk_text: "orphan" }] }),
    },

    // --- tree build: valid, and the invalid/retry pair -------------------
    {
      id: "tree-build-invalid-title",
      priority: 10,
      match: {
        all: [
          { fact: "latestUserText", operator: "contains", value: "chunks (indexes" },
          { fact: "latestUserText", operator: "contains", value: "RETRY-CASE" },
        ],
      },
      // Empty title violates the validator in schemas/tree.py.
      respond: json({
        root: { type: "internal", node_type: "root", title: "", summary: "s", children: [leaf(0)] },
      }),
    },
    {
      id: "tree-build-retry-corrected",
      priority: 10,
      match: { fact: "latestUserText", operator: "contains", value: "title must not be empty" },
      respond: json({
        root: { type: "internal", node_type: "root", title: "Recovered", summary: "s", children: [leaf(0)] },
      }),
    },
    {
      id: "tree-build-always-invalid",
      priority: 10,
      match: {
        any: [
          {
            all: [
              { fact: "latestUserText", operator: "contains", value: "chunks (indexes" },
              { fact: "latestUserText", operator: "contains", value: "EXHAUST-CASE" },
            ],
          },
          { fact: "latestUserText", operator: "contains", value: "node_type must be" },
        ],
      },
      // Invalid node_type, and stays invalid on every retry.
      respond: json({
        root: { type: "internal", node_type: "chapter", title: "T", summary: "s", children: [leaf(0)] },
      }),
    },
    {
      id: "tree-build-missing-chunk",
      priority: 10,
      match: {
        all: [
          { fact: "latestUserText", operator: "contains", value: "chunks (indexes" },
          { fact: "latestUserText", operator: "contains", value: "COVERAGE-CASE" },
        ],
      },
      // Schema-valid, but omits chunk_index 1 -- caught by _validate_chunk_coverage.
      respond: json({
        root: { type: "internal", node_type: "root", title: "T", summary: "s", children: [leaf(0)] },
      }),
    },
    {
      id: "tree-build-valid",
      match: { fact: "latestUserText", operator: "contains", value: "chunks (indexes" },
      respond: json({
        root: {
          type: "internal",
          node_type: "root",
          title: "Document",
          summary: "Two sections.",
          children: [
            leaf(0),
            { type: "internal", node_type: "section", title: "Beta", summary: "Second section.", children: [leaf(1)] },
          ],
        },
      }),
    },

    // --- tree merge -----------------------------------------------------
    {
      id: "tree-merge",
      match: { fact: "latestUserText", operator: "contains", value: "Part 0:" },
      respond: json({ merged_title: "Merged", merged_summary: "Covers all parts." }),
    },

    // --- retrieval ------------------------------------------------------
    // Both return one real id and one fabricated id, so the tests assert the
    // hallucination guards drop the fabricated one.
    {
      id: "llmselect",
      match: { fact: "latestUserText", operator: "contains", value: "Documents:" },
      respond: json({
        selected_documents: [
          { document_id: "11111111-1111-1111-1111-111111111111", relevance_score: 0.9 },
          { document_id: "99999999-9999-9999-9999-999999999999", relevance_score: 0.8 },
        ],
      }),
    },
    {
      id: "llmnavigate",
      match: { fact: "latestUserText", operator: "contains", value: "Document tree:" },
      respond: json({
        selected_nodes: [
          { node_id: "22222222-2222-2222-2222-222222222222", relevance_score: 0.9 },
          { node_id: "88888888-8888-8888-8888-888888888888", relevance_score: 0.7 },
        ],
      }),
    },
  ],
  unmatchedRequest: "error",
} satisfies AgentTestkitDocument;
