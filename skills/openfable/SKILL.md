---
name: openfable
description: Index long documents into a semantic forest and retrieve from them within a token budget. Use when the user wants to ingest, index, or search documents (specs, papers, contracts, manuals, transcripts) too long to read whole, or asks a question about a corpus already indexed in OpenFable. You supply the chunking and tree structure yourself.
compatibility: Requires Docker with Compose v2.
license: Apache-2.0
---

<skill name="openfable">

<overview>
Each document becomes a tree: root → sections → subsections → leaves. Leaves hold
chunk text; internal nodes hold a title and summary. Every node is embedded.
Queries return the highest-scoring nodes that fit a token budget.

You choose the chunk boundaries and build the tree. OpenFable resolves boundaries
to character offsets, persists, embeds, scores, and enforces the budget.
</overview>

<onboarding trigger="missing-preferences" skippable="true" rerun-with="setup">
  <gate>
    <check order="1">.openfable/preferences.json exists → skip onboarding, go to setup.</check>
    <check order="2">Exists but skill_version != 1 → offer to regenerate docker-compose.yml only. Never regenerate .env.</check>
    <check order="3">Preferences existing does not mean the stack is running. Always run health after.</check>
  </gate>

  <skip-path>
    <offer>Offer before asking: local database and local embeddings, nothing to configure — or two questions to use existing infrastructure.</offer>
    <action>Generate all-local, write preferences.json with skipped=true, proceed. Do not ask again.</action>
  </skip-path>

  <question id="database" order="1" default="container">
    <ask>Run PostgreSQL as a container, or connect to an existing one?</ask>
    <option value="container">pgvector container, scoped to this directory.</option>
    <option value="external" collects="OPENFABLE_DATABASE_URL">
      <validate id="reachable">Connect before writing anything. Managed providers usually need ?sslmode=require.</validate>
      <validate id="extensions">`vector` and `ltree` must exist or be creatable by this role. CREATE EXTENSION usually needs superuser. Fail here, not at first ingest.</validate>
      <consent id="migrations">Migrations CREATE TABLES in a database the user owns. Get explicit agreement.</consent>
      <side-effect>Directories sharing a database share a corpus. Default is one corpus per directory.</side-effect>
    </option>
  </question>

  <question id="embeddings" order="2" default="container">
    <ask>Run an embedding server as a container, or use an existing endpoint?</ask>
    <option value="container">TEI serving bge-m3. Downloads ~2GB once, then offline.</option>
    <option value="external" collects="OPENFABLE_EMBEDDING_URL,OPENFABLE_EMBEDDING_MODEL,OPENFABLE_EMBEDDING_API_KEY">
      <validate id="responds">POST a probe to /v1/embeddings before writing anything.</validate>
      <validate id="dimensions">Returned vector length MUST be 1024; the column is fixed. text-embedding-3-small truncates to 1024; -3-large at 3072 does not fit.</validate>
      <validate id="corpus-conflict">If a corpus exists and the model differs from preferences.embeddings.model, warn and offer re-index. Old vectors stay in the old model's space; retrieval degrades with no error.</validate>
    </option>
  </question>

  <not-asked>Behavioural preferences (token budget, vector_only). Deployment config only — it is the part that cannot be changed for free later.</not-asked>

  <template-selection>
    <template name="all-local" when="database=container AND embeddings=container"/>
    <template name="external-db" when="database=external AND embeddings=container"/>
    <template name="external-embeddings" when="database=container AND embeddings=external"/>
    <template name="external-both" when="database=external AND embeddings=external"/>
  </template-selection>

  <outputs directory=".openfable/">
    <file name="docker-compose.yml" from="assets/compose-{template}.yml">
    Replace __PROJECT__ with the project directory name. Required: every copy
    lives in a directory named .openfable, so unnamed projects all collide and
    one `up` recreates another's containers.
    </file>
    <file name=".env" from="assets/env.example">
    Write the BASE block plus the chosen template's blocks. Never rely on
    omission: OPENFABLE_EMBEDDING_URL defaults to https://api.openai.com, so an
    empty .env sends a local stack to OpenAI and fails 401.
    </file>
    <file name="preferences.json" from="assets/preferences.example.json">
    Record skill_version, compose_template, both answers, and
    embeddings.{model,dimensions} so a later change is detectable.
    </file>
    <file name=".gitignore" content=".env">Always. Prevents committing a database password.</file>
  </outputs>
</onboarding>

<setup>
  <precondition>.openfable/ exists, else run onboarding.</precondition>

  <invoke><![CDATA[
OF="docker compose -f .openfable/docker-compose.yml"
$OF up -d
$OF exec openfable openfable health
  ]]></invoke>

  <expect>{"status": "healthy", "components": {"database": "healthy", "embeddings": "healthy"}}</expect>

  <note topic="first-run">bge-m3 downloads on first start (~2 min). health reports embeddings unhealthy until it completes. Wait; do not proceed.</note>
  <note topic="paths">The invoking directory mounts read-only at /work. ./spec.md is /work/spec.md. Override with absolute OPENFABLE_WORKDIR.</note>

  <invariant id="output">One JSON object on stdout per command. Failure: {"error": "..."} and exit 1.</invariant>
  <invariant id="retry">On error, fix the input and re-run the SAME step. A rejected step persists nothing.</invariant>
  <invariant id="tty">Commands reading stdin require `exec -T`. Without it docker allocates a TTY and stdin never arrives.</invariant>
</setup>

<workflow name="index-document">
  <preconditions>
    <precondition>health is healthy</precondition>
    <precondition>You have read the document and know its structure.</precondition>
  </preconditions>

  <description>Three ordered steps; each consumes the previous output. Not queryable until step 3 returns.</description>

  <step id="1" name="plan" produces="document_id">
    <invoke><![CDATA[
$OF exec -T openfable openfable plan /work/spec.md
    ]]></invoke>

    <returns>
      <field name="document_id" type="uuid">Required by steps 2 and 3.</field>
      <field name="token_count" type="int">Whole-document token count.</field>
      <field name="content_hash" type="sha256">Identity key. Re-planning identical bytes reuses the row.</field>
      <field name="reingest" type="bool">True when content_hash already existed; chunks, nodes and embeddings are cleared first.</field>
    </returns>

    <errors>
      <error match="No such file" retry="step-1">Path must be under /work. List it: $OF exec openfable ls /work</error>
    </errors>
  </step>

  <step id="2" name="apply-chunks" requires="document_id" produces="chunk_index-set">
    <agent_task>Split where the topic changes, not at a fixed size. One chunk per idea, beginning at a sentence or heading.</agent_task>

    <input type="json-array-of-string">
    The verbatim opening of each chunk, in document order. Roughly 5-10 words —
    long enough to be unique. Offsets are computed from these; never supply them.
    </input>

    <marker-rules>
      <rule id="verbatim">Matched literally first. Characters must match exactly: an em-dash is not a hyphen.</rule>
      <rule id="whitespace">Falls back to a whitespace-tolerant match, so differing line breaks or repeated spaces still resolve.</rule>
      <rule id="order">Each search starts after the previous match. Markers must be in document order; a backwards marker cannot resolve.</rule>
      <rule id="repeats">A phrase occurring twice resolves to successive occurrences, not the same one.</rule>
      <rule id="leading-text">If the first marker is not at offset 0, the text before it becomes chunk 0. You may start at the second boundary.</rule>
      <rule id="whitespace-only">Chunks that are entirely whitespace are dropped.</rule>
      <rule id="atomic">Re-send the whole array when correcting. Not incremental.</rule>
    </marker-rules>

    <invoke><![CDATA[
DOC_ID=<document_id from step 1>
echo '["# Project Chimera", "## Section 2: Neural Mapping", "#### 2.2.1 Measurement Methodology"]' \
  | $OF exec -T openfable openfable apply-chunks "$DOC_ID" -
    ]]></invoke>

    <returns>
      <field name="chunk_count" type="int"/>
      <field name="chunks[].chunk_index" type="int">0-based, document order. Step 3 must cover this exact set.</field>
      <field name="chunks[].token_count" type="int">A chunk larger than a query budget can never be returned at that budget.</field>
      <field name="chunks[].preview" type="string">First 160 chars. Confirm boundaries landed as intended.</field>
    </returns>

    <errors>
      <error match="Boundary marker N not found at or after offset X" retry="step-2">Not verbatim, or out of order. The message quotes the marker. Copy it from the file exactly.</error>
    </errors>
  </step>

  <step id="3" name="apply-tree" requires="chunk_index-set" produces="indexed">
    <agent_task>
    Group the chunk_index values into a hierarchy. Summaries are embedded and
    searched: state what the subtree SAYS. "Synapse integration rates and
    measured latency figures", not "This section covers neural mapping".
    </agent_task>

    <schema>
      <node type="internal">
        <field name="type" const="internal"/>
        <field name="node_type" enum="root|section|subsection"/>
        <field name="title" type="string" constraint="non-empty"/>
        <field name="summary" type="string" constraint="non-empty"/>
        <field name="children" type="array">internal or leaf nodes</field>
      </node>
      <node type="leaf">
        <field name="type" const="leaf"/>
        <field name="chunk_index" type="int"/>
      </node>
      <root>Top-level object is {"root": INTERNAL_NODE}.</root>
    </schema>

    <rules>
      <rule id="coverage">Every chunk_index from step 2 appears exactly once. Gaps and duplicates are rejected by index.</rule>
      <rule id="depth">Depth 4 max (root=1, section=2, subsection=3, leaf=4). Deeper nodes are NOT rejected — they are re-parented to their grandparent and a warning is logged. Choose the collapse yourself rather than letting it happen.</rule>
      <rule id="flat-ok">Leaves may hang directly off the root; not every chunk needs a section.</rule>
      <rule id="paths">toc_path is computed from ancestor titles. Never supply it.</rule>
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
      <field name="nodes_embedded" type="int">Internal nodes plus leaves.</field>
      <field name="status" const="indexed"/>
    </returns>

    <errors>
      <error match="Missing chunk indexes: [2, 3]" retry="step-3">Add those leaves.</error>
      <error match="Duplicate chunk_index values found" retry="step-3">Each index may appear once.</error>
      <error match="Tree does not match the expected schema" retry="step-3">Malformed node. Check node_type against the enum and that every leaf has type and chunk_index.</error>
    </errors>
  </step>

  <postcondition>Queryable. Verify: $OF exec openfable openfable list</postcondition>
</workflow>

<workflow name="query">
  <preconditions><precondition>At least one document indexed.</precondition></preconditions>

  <description>
  Choose the documents, then query within them. You are doing the document
  selection the service would otherwise ask an LLM to do — see the caveat on
  step 2 for what happens when nothing does it.
  </description>

  <step id="1" name="candidates" optional="true" produces="document-selection">
    <skip-when>The corpus holds one document, or you already know which document applies. Check with `list`.</skip-when>

    <invoke><![CDATA[
$OF exec -T openfable openfable candidates
    ]]></invoke>

    <returns>
      <field name="documents[].document_id" type="uuid"/>
      <field name="documents[].token_count" type="int">Whole-document size. Below the budget means step 2 returns it entire.</field>
      <field name="documents[].sections" type="array">toc_path and summary per internal node, down to depth 2 — the document's shallow table of contents.</field>
    </returns>

    <agent_task>
    Pick the documents whose sections could answer the question. Judge on subject
    overlap, not wording. Partial relevance counts — include a document covering
    part of the question. Prefer too few over too many: each extra document adds
    competing chunks.
    </agent_task>
  </step>

  <step id="2" name="query" requires="document-selection">
    <invoke><![CDATA[
$OF exec openfable openfable query "which env var sets the embedding model" \
  --budget 2000 --vector-only --documents <id>[,<id>...]
    ]]></invoke>

    <constraints>
      <constraint id="vector-only">Always pass --vector-only. No LLM is configured; without it the LLM paths are attempted, fail, and fall back to the same vector result — slower, with stderr noise.</constraint>
      <constraint id="documents">Omitting --documents searches the whole corpus. Correct for a single-document corpus; for several documents see the caveat.</constraint>
    </constraints>

    <errors>
      <error match="Unknown document_id(s)" retry="step-2">An id is not in the corpus. The message names it. Re-check against `list` or `candidates`.</error>
    </errors>

    <budgets min="100" max="32000" default="2000">
      <budget range="150-500" use-for="one specific buried fact"/>
      <budget range="2000" use-for="one topic"/>
      <budget range="8000+" use-for="cross-document synthesis"/>
      <tuning>Start at 2000. Narrow on noise, widen on truncation.</tuning>
    </budgets>

    <response_fields>
      <field name="routing" value="document_level">Whole documents fit. Read `documents`; `chunks` absent.</field>
      <field name="routing" value="node_level">Drilled into the tree. Read `chunks` and `node_results`, which carry score and depth.</field>
      <field name="over_budget" type="bool">True means the budget could not fit even the top result. Raise and re-query.</field>
      <field name="total_tokens_used" type="int">Budget actually spent.</field>
    </response_fields>

    <caveat id="cross-document-ranking">
    Why step 1 exists. S(v) = 1/3[S_sim + S_inh + S_child]. For a leaf:
    S_child=0, S_sim = cosine/depth, S_inh = highest ancestor S_sim. Root depth
    is 1, so the root's similarity enters undivided and every leaf inherits it.

    Without a selection, ranking across documents therefore follows
    document-root similarity more than chunk relevance, and a larger budget
    pulls in more of another document's chunks. Within one document S_inh is
    constant and cancels, so ranking is normal — which is what --documents
    restores. Keep building hierarchy either way.

    Measured on a 4-document corpus: the answering chunk ranked 8th, behind
    three chunks of an unrelated document. Selecting its document first
    returned it as the whole document at the same budget.

    Faithful to arXiv:2601.18116v1 §3.2. This is the algorithm running without
    the document-selection step it assumes, not a defect.
    </caveat>
  </step>
</workflow>

<commands prefix="$OF exec [-T] openfable openfable">
  <command name="plan" args="PATH">Register a document; returns document_id.</command>
  <command name="apply-chunks" args="DOC_ID -" stdin="required">Persist boundaries.</command>
  <command name="apply-tree" args="DOC_ID -" stdin="required">Persist tree, then embed.</command>
  <command name="candidates">Every document with its toc paths and summaries, for choosing before a query.</command>
  <command name="query" args="TEXT --budget N [--vector-only] [--documents ID,ID]">Retrieval within a token budget, optionally restricted to chosen documents.</command>
  <command name="list">Indexed documents with id, hash, token_count.</command>
  <command name="get" args="DOC_ID [--meta-only]">One document; --meta-only omits content.</command>
  <command name="health">Database and embedding server.</command>
  <command name="index" args="PATH" available="false">LiteLLM ingest. Unavailable here — no LLM configured.</command>
</commands>

<examples location="scripts/">
  <instruction>Read the ONE example matching the situation, not all of them.</instruction>
  <example file="scripts/01-simple-document.sh" complexity="simple" read-when="first time indexing">One short flat document; the minimal case.</example>
  <example file="scripts/02-query-existing-corpus.sh" complexity="simple" read-when="before indexing anything">Query only, and budget behaviour.</example>
  <example file="scripts/03-nested-document.sh" complexity="medium" read-when="document nests past two levels">Heading depth to node_type, and choosing which level to collapse.</example>
  <example file="scripts/04-error-recovery.sh" complexity="medium" read-when="a step returned an error">Non-verbatim marker and missing chunk indexes, both recovered.</example>
  <example file="scripts/05-multi-document-synthesis.sh" complexity="complex" read-when="answer spans sources">Two documents that disagree.</example>
  <example file="scripts/06-chunk-granularity.sh" complexity="complex" read-when="retrieval returns related but wrong content">Why a 298-token chunk is unreachable at a 140-token budget.</example>
  <index file="references/JOURNEYS.md">Indexes the above; records behaviours affecting how output should be read.</index>
</examples>

<usage_guidance>
  <guidance id="check-first">Run `list` before indexing; the corpus may already hold the answer. Re-indexing identical bytes returns reingest=true and rebuilds from scratch.</guidance>
  <guidance id="scope">Not for a file you can read directly, or a string you could grep. It pays on documents too long to hold at once, questions spanning sections, and repeated queries against a stable corpus.</guidance>
</usage_guidance>

</skill>
