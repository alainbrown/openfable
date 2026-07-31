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

**Node scores are normalised per document and are not comparable across
documents.** In a multi-document corpus, an unrelated document's best chunk can
show a higher score than the chunk that actually answers your question, and a
larger budget can pull that noise in ahead of the answer. Trust the
document-level ranking first; treat node scores as meaningful only within a
single document. `06` demonstrates this and works around it by asserting on
chunk token counts, which are deterministic, rather than on scores.
