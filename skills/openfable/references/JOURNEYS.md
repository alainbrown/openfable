# Reference journeys

Runnable examples. Each is a complete, commented user journey — read the one
that matches your situation rather than all of them.

| Script | Journey | Read it when |
|--------|---------|--------------|
| `01-simple-document.sh` | Index one short, flat document | First time. Everything else varies these three steps. |
| `02-query-existing-corpus.sh` | Query without indexing; the budget ladder | You have a question and the corpus may already hold the answer. |
| `03-nested-document.sh` | Deep headings → `section` / `subsection`, depth-4 cap | The document nests more than two levels. |
| `04-error-recovery.sh` | Both failure modes and how to recover | A step returned `{"error": ...}`. |
| `05-multi-document-synthesis.sh` | Two sources that disagree; cross-document retrieval | The answer spans documents, or sources may conflict. |
| `06-chunk-granularity.sh` | Coarse vs fine boundaries, and why it decides everything | Retrieval returns something related but not the fact you asked for. |

These assume onboarding has run, so `.openfable/` exists. If it does not, ask
the agent to set OpenFable up first — or generate it by hand from
`compose/all-local.yml`, substituting `__PROJECT__` with your project directory
name, plus a `.env` built from `compose/env.example`.

Then, from the repository root:

```bash
docker compose -f .openfable/docker-compose.yml up -d
bash skills/openfable/scripts/01-simple-document.sh
```

`compose/` holds the four templates onboarding picks between, and the `.env`
and `preferences.json` examples it fills in.

`fixtures/` holds the documents 03, 05 and 06 index. 01, 02 and 04 use the
repository's own `CONTRIBUTING.md` and `README.md`.

## Two things to know before reading the output

**These scripts share one database.** Running them in sequence leaves documents
behind, and that is realistic — but it means a query in one script can retrieve
a document another script indexed. Re-running a script is safe: identical
content re-registers as `reingest: true`, clearing the previous index rather
than duplicating it.

**Across documents, ranking follows the document root more than the chunk.**
Node scores are globally comparable — min-max normalised across every candidate
leaf at once — so the distortion is not in the normalisation. It is upstream.

FABLE is bi-path: an LLM selects documents, and structure-aware scoring recovers
passages within that selection. `--vector-only` removes the LLM path, leaving
the structural score to do document selection unaided, which it was never meant
to do.

The scoring is `S(v) = ⅓[S_sim + S_inh + S_child]`, and a leaf has no children,
so `S_child = 0`. `S_sim` is the leaf's own similarity divided by its depth;
`S_inh` is the highest ancestor `S_sim`, and since the root sits at depth 1 its
similarity enters undivided and is inherited identically by every leaf below it.
So a per-document constant can outweigh what the chunk actually says. Measured
here: a chunk with cosine 0.5873 at depth 3 scored 0.1655 and lost to one with
cosine 0.3355 at depth 2 scoring 0.2064, because their document roots scored
0.3007 and 0.4514 respectively.

This is faithful to the paper, not an implementation defect — the equations
match §3.2 of [arXiv:2601.18116v1](https://arxiv.org/abs/2601.18116v1). The v2
revision generalises the weights to `α·S_sim + β·S_inh + γ·S_child` and calls
the equal ⅓ split a "simple untuned default", noting the method is sensitive to
them by query type.

Practically: trust the document-level result first, treat node scores as
reliable only within a single document, and narrow rather than widen the budget
when a query returns plausible-but-wrong content. Within one document all leaves
inherit the same root value, so it cancels and ranking behaves normally — keep
building proper hierarchy. `06` avoids the issue entirely by asserting on chunk
token counts, which are deterministic.
