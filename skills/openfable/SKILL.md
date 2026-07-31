---
name: openfable
description: Index long documents into a semantic forest and retrieve from them within a token budget. Use when the user wants to ingest, index, or search documents (specs, papers, contracts, manuals, transcripts) too long to read whole, or asks a question about a corpus already indexed in OpenFable. You supply the chunking and tree structure yourself — no LLM API key needed.
compatibility: Requires Docker with Compose v2. No LLM API key needed.
license: Apache-2.0
---

<skill name="openfable">

<overview>
Retrieval over long, structured documents. Each document becomes a tree
(root → sections → subsections → leaves); queries return the most relevant
nodes within a token budget.

You do the indexing reasoning: you read the document, choose chunk boundaries,
and build the tree. OpenFable persists, embeds, scores, and enforces the budget.
No LLM API key is involved — embeddings run in a local container.
</overview>

<onboarding trigger="missing-preferences" skippable="true" rerun-with="setup">
  <gate>
    <check order="1">If .openfable/preferences.json exists, onboarding is DONE. Do not ask anything; go straight to setup.</check>
    <check order="2">If it exists but its skill_version differs from 1, the compose template has moved on. Offer to regenerate .openfable/docker-compose.yml; do not regenerate .env.</check>
    <warning>Preferences existing means the configuration is known, NOT that the stack is running. Always run health afterwards.</warning>
  </gate>

  <skip-path>
    <offer>Offer this BEFORE asking anything: "I can set up OpenFable with a local database and local embeddings — nothing to configure, no API keys. Or I can ask two questions to point it at infrastructure you already have."</offer>
    <action>On skip, generate the all-local template, write preferences.json with skipped: true, and proceed. Never ask again.</action>
    <note>`setup` re-runs the interview later with current values as defaults.</note>
  </skip-path>

  <question id="database" order="1" default="container">
    <ask>Should OpenFable run its own PostgreSQL, or connect to one you already have?</ask>

    <option value="container">A pgvector container, isolated to this project directory. Nothing to configure.</option>

    <option value="external" collects="OPENFABLE_DATABASE_URL">
      <when>They have managed pgvector (Neon, Supabase, RDS, Cloud SQL) or a local server.</when>
      <side-effect>Several project directories pointed at one database share a corpus. This is how to opt into a shared corpus rather than the per-directory default.</side-effect>
      <validate id="reachable">Connect with the supplied URL before writing anything. Managed providers usually need ?sslmode=require.</validate>
      <validate id="extensions">Confirm `vector` and `ltree` are installed, or creatable by this role. CREATE EXTENSION typically needs superuser and managed providers vary — fail here rather than at first ingest.</validate>
      <consent id="migrations">State plainly that migrations will CREATE TABLES in a database they own, and get agreement before proceeding.</consent>
    </option>
  </question>

  <question id="embeddings" order="2" default="container">
    <ask>Should OpenFable run its own embedding server, or use an endpoint you already have?</ask>

    <option value="container">A local TEI container serving bge-m3. Downloads ~2GB once, then runs offline. No key.</option>

    <option value="external" collects="OPENFABLE_EMBEDDING_URL,OPENFABLE_EMBEDDING_MODEL,OPENFABLE_EMBEDDING_API_KEY">
      <when>They already run TEI, vLLM or Ollama, or want OpenAI embeddings.</when>
      <validate id="responds">POST a probe to /v1/embeddings before writing anything.</validate>
      <validate id="dimensions">The returned vector MUST be 1024 long. The column is fixed at that size, so a mismatch fails every insert. text-embedding-3-small truncates to 1024; -3-large at 3072 does not fit.</validate>
      <validate id="corpus-conflict">If a corpus already exists and the model differs from preferences.embeddings.model, WARN LOUDLY. Stored vectors are from the old model and retrieval silently degrades — nothing errors. Re-index rather than mixing.</validate>
    </option>
  </question>

  <not-asked>
  Do not collect behavioural preferences here. A user cannot sensibly choose a
  default token budget before running a query. Deployment config is the only
  thing needed up front, because it is the only thing that cannot be changed
  for free afterwards.
  </not-asked>

  <template-selection>
    <template name="all-local" when="database=container AND embeddings=container"/>
    <template name="external-db" when="database=external AND embeddings=container"/>
    <template name="external-embeddings" when="database=container AND embeddings=external"/>
    <template name="external-both" when="database=external AND embeddings=external"/>
  </template-selection>

  <outputs directory=".openfable/">
    <file name="docker-compose.yml" from="assets/compose-{template}.yml">
    Substitute __PROJECT__ with the project directory name. Every project's file
    lives in a directory called .openfable, so without this each would derive
    the same compose project and one project's `up` would recreate another's
    containers.
    </file>
    <file name=".env" from="assets/env.example">
    Write the BASE block plus the blocks for the chosen template. Do not rely on
    omission — the code defaults target the hosted server, so an empty .env
    sends a local stack to OpenAI and fails with 401.
    </file>
    <file name="preferences.json" from="assets/preferences.example.json">
    Record the template, both answers, and the embedding model and dimensions,
    so a later change can be detected.
    </file>
    <file name=".gitignore" content=".env">
    Written every time. Without it a managed-database password gets committed.
    </file>
  </outputs>
</onboarding>

<setup>
  <precondition>.openfable/ exists. If not, run onboarding first.</precondition>

  <invoke><![CDATA[
OF="docker compose -f .openfable/docker-compose.yml"
$OF up -d
$OF exec openfable openfable health
  ]]></invoke>

  <expect>{"status": "healthy", "components": {"database": "healthy", "embeddings": "healthy"}}</expect>

  <note topic="first-run">
  The bge-m3 model downloads on first start (~2 min). Until it finishes, health
  reports embeddings: unhealthy. Wait and re-run; do not proceed.
  </note>

  <note topic="working-directory">
  Run from the directory holding your documents. It mounts read-only at /work,
  so ./spec.md is /work/spec.md inside the container. Override with an absolute
  OPENFABLE_WORKDIR.
  </note>

  <invariant id="one-json-object">
  Every command prints exactly one JSON object to stdout. Failure prints
  {"error": "..."} and exits non-zero.
  </invariant>

  <invariant id="retry-in-place">
  On error, correct your input and re-run the SAME step. Never restart the
  workflow. A rejected step persists nothing.
  </invariant>
</setup>

<workflow name="index-document">
  <preconditions>
    <precondition>health reports status: healthy</precondition>
    <precondition>You have read the document yourself and understand its structure.</precondition>
  </preconditions>

  <description>
  Three ordered steps. Each consumes the previous step's output — you cannot
  skip or reorder. The document is NOT queryable until step 3 returns.
  </description>

  <step id="1" name="plan" produces="document_id">
    <invoke><![CDATA[
$OF exec -T openfable openfable plan /work/spec.md
    ]]></invoke>

    <returns>
      <field name="document_id">Required by steps 2 and 3. Carry it forward.</field>
      <field name="token_count">Total size of the document.</field>
      <field name="reingest">True if this exact content was indexed before; the old index is cleared and rebuilt.</field>
    </returns>

    <errors>
      <error match="No such file" retry="step-1">
      The path is wrong. It must be /work/..., not a host path.
      Check what is mounted: $OF exec openfable ls /work
      </error>
    </errors>
  </step>

  <step id="2" name="apply-chunks" requires="document_id" produces="chunk_index-set">
    <agent_task>
    Choose the chunk boundaries. Split where the TOPIC CHANGES, not at a fixed
    size. Each chunk covers one idea and begins at a sentence or heading.

    Produce a JSON array of the verbatim opening snippet of each chunk, in
    document order — roughly 5–10 words each, enough to be unique. You supply
    openings, not character offsets. Text before the first marker becomes its
    own chunk, so you may start at the second boundary.
    </agent_task>

    <invoke><![CDATA[
DOC_ID=<the document_id from step 1>
echo '["# Project Chimera", "## Section 2: Neural Mapping", "#### 2.2.1 Measurement Methodology"]' \
  | $OF exec -T openfable openfable apply-chunks "$DOC_ID" -
    ]]></invoke>

    <constraints>
      <constraint id="stdin">The trailing `-` reads stdin. /work is read-only, so there is nowhere to write a temp file.</constraint>
      <constraint id="no-tty">`exec -T` is required. Without it docker allocates a TTY and stdin never reaches the command.</constraint>
      <constraint id="all-or-nothing">Re-send the whole array when correcting. This step is not incremental.</constraint>
    </constraints>

    <returns>
      <field name="chunk_count">Number of chunks created.</field>
      <field name="chunks[].chunk_index">The exact set step 3 must cover.</field>
      <field name="chunks[].token_count">Chunk size. A chunk larger than your query budget can never be returned at that budget.</field>
      <field name="chunks[].preview">First 160 characters, to confirm boundaries landed where you intended.</field>
    </returns>

    <errors>
      <error match="Boundary marker N not found at or after offset X" retry="step-2">
      That marker is not verbatim, or is out of document order. The message
      quotes the offending marker. Copy the text out of the file exactly.
      </error>
    </errors>
  </step>

  <step id="3" name="apply-tree" requires="chunk_index-set" produces="indexed">
    <agent_task>
    Organise the chunk_index values from step 2 into a hierarchy. Internal
    nodes carry a title and a summary. The summary is what retrieval searches
    over — describe what the subtree SAYS, not that it exists. Write "Synapse
    integration rates and measured latency figures", not "This section covers
    neural mapping".
    </agent_task>

    <rules>
      <rule>node_type is one of: root, section, subsection.</rule>
      <rule>Maximum depth 4 (root=1, section=2, subsection=3, leaf=4).</rule>
      <rule>Every chunk_index from step 2 appears exactly once as a leaf.</rule>
      <rule>Not every chunk needs a section wrapper; leaves may hang off the root.</rule>
    </rules>

    <invoke><![CDATA[
cat <<'JSON' | $OF exec -T openfable openfable apply-tree "$DOC_ID" -
{"root": {"type": "internal", "node_type": "root",
  "title": "Project Chimera Q1 Report",
  "summary": "Q1 results across neural mapping, ethics, and budget.",
  "children": [
    {"type": "leaf", "chunk_index": 0},
    {"type": "internal", "node_type": "section",
     "title": "Neural Mapping",
     "summary": "Synapse integration rates and measured latency figures.",
     "children": [{"type": "leaf", "chunk_index": 1}, {"type": "leaf", "chunk_index": 2}]}
  ]}}
JSON
    ]]></invoke>

    <returns>
      <field name="nodes_embedded">Total nodes written and embedded.</field>
      <field name="status">"indexed" — the document is now queryable.</field>
    </returns>

    <errors>
      <error match="Missing chunk indexes: [2, 3]" retry="step-3">Add those leaves.</error>
      <error match="Duplicate chunk_index values found" retry="step-3">A chunk appears twice; each may appear only once.</error>
      <error match="Tree does not match the expected schema" retry="step-3">A node is malformed. Check node_type is root/section/subsection and every leaf has both "type" and "chunk_index".</error>
    </errors>
  </step>

  <postcondition>The document is queryable. Verify with: $OF exec openfable openfable list</postcondition>
</workflow>

<workflow name="query">
  <preconditions>
    <precondition>At least one document has completed index-document.</precondition>
  </preconditions>

  <step id="1" name="query">
    <invoke><![CDATA[
$OF exec openfable openfable query "which env var sets the embedding model" --budget 2000 --vector-only
    ]]></invoke>

    <constraints>
      <constraint id="vector-only">
      Always pass --vector-only on this stack. No LLM is configured; without it
      the LLM paths are attempted, fail, and fall back to vector search anyway
      — same result, slower, plus stderr noise.
      </constraint>
    </constraints>

    <budgets min="100" max="32000" default="2000">
      <budget range="150-500" use-for="one specific buried fact"/>
      <budget range="2000" use-for="a question about one topic"/>
      <budget range="8000+" use-for="cross-document synthesis, comparing sources"/>
      <tuning>Start at 2000. Narrow if you get noise, widen if the answer looks truncated.</tuning>
    </budgets>

    <response_fields>
      <field name="routing" value="document_level">Whole documents fit the budget. Read `documents`; `chunks` is absent.</field>
      <field name="routing" value="node_level">It drilled into the tree. Read `chunks` and `node_results`, which carry scores.</field>
      <field name="over_budget" value="true">The budget could not fit even the top result. Raise it and re-query rather than interpreting a truncated answer.</field>
      <field name="total_tokens_used">How much of the budget was actually spent.</field>
    </response_fields>

    <caveat id="cross-document-scores">
    Node scores are normalised per document and are NOT comparable across
    documents. In a multi-document corpus an unrelated document's best chunk
    can outscore the chunk that answers your question, and a larger budget can
    pull that noise in ahead of the answer. Trust the document-level ranking
    first; treat node scores as meaningful only within one document.
    </caveat>
  </step>
</workflow>

<commands prefix="$OF exec [-T] openfable openfable">
  <command name="plan" args="PATH">Register a document; returns document_id.</command>
  <command name="apply-chunks" args="DOC_ID -" stdin="required">Persist chunk boundaries. Needs exec -T.</command>
  <command name="apply-tree" args="DOC_ID -" stdin="required">Persist tree, then embed. Needs exec -T.</command>
  <command name="query" args="TEXT --budget N [--vector-only]">Bi-path retrieval within a token budget.</command>
  <command name="list">What is indexed.</command>
  <command name="get" args="DOC_ID [--meta-only]">One document.</command>
  <command name="health">Database and embedding server.</command>
  <command name="index" args="PATH" available="false">Full LiteLLM ingest. NOT available here — no LLM is configured.</command>
</commands>

<examples location="scripts/">
  <instruction>Read the ONE example matching your situation. Do not read them all.</instruction>

  <example file="scripts/01-simple-document.sh" complexity="simple">
    <journey>Index one short, flat document. The minimal complete case.</journey>
    <read-when>You have not indexed before. Everything else varies these three steps.</read-when>
  </example>

  <example file="scripts/02-query-existing-corpus.sh" complexity="simple">
    <journey>Query without indexing, and how one question behaves across budgets.</journey>
    <read-when>Before indexing anything — the corpus may already hold the answer.</read-when>
  </example>

  <example file="scripts/03-nested-document.sh" complexity="medium">
    <journey>Mapping heading depth onto root/section/subsection, and choosing which level to collapse when a document nests past the depth-4 cap.</journey>
    <read-when>The document nests more than two levels.</read-when>
  </example>

  <example file="scripts/04-error-recovery.sh" complexity="medium">
    <journey>Both failure modes worked through: a non-verbatim marker, and a tree with missing chunk indexes.</journey>
    <read-when>A step returned {"error": ...}.</read-when>
  </example>

  <example file="scripts/05-multi-document-synthesis.sh" complexity="complex">
    <journey>Two documents that disagree, and retrieving across both so the contradiction is visible.</journey>
    <read-when>The answer spans documents, or sources may conflict.</read-when>
  </example>

  <example file="scripts/06-chunk-granularity.sh" complexity="complex">
    <journey>Why a fact inside a 298-token chunk is unreachable at a 140-token budget, and how boundary placement fixes it.</journey>
    <read-when>Retrieval returns something related but not the fact you asked for.</read-when>
  </example>

  <index file="references/JOURNEYS.md">Lists these and records two behaviours worth knowing before interpreting any output.</index>
</examples>

<usage_guidance>
  <guidance id="check-first">
  Run `list` before indexing. The corpus may already hold what you need, and
  re-indexing identical content is wasted work (it returns reingest: true).
  </guidance>

  <guidance id="dont-over-reach">
  Indexing costs real work per document. Do not use OpenFable for a file you
  can simply read, or for a known string you could grep.
  </guidance>

  <guidance id="where-it-pays">
  It earns its cost on documents too long to hold at once, questions spanning
  several sections, and repeated queries against a stable corpus.
  </guidance>
</usage_guidance>

</skill>
