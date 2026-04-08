"""Golden document fixtures, mock LLM responses, and deterministic vector helpers.

All test data here is deterministic and curated per D-04 (2-3 small documents),
D-06 (stored as Python fixtures, not files), and D-08 (deterministic vectors
with known cosine similarities).

Usage:
    from tests.integration.fixtures import (
        GOLDEN_DOC_SHORT, GOLDEN_DOC_LONG,
        MOCK_CHUNKS_SHORT, MOCK_CHUNKS_LONG,
        MOCK_TREE_SHORT, MOCK_TREE_LONG,
        make_query_vector, make_relevant_vector, make_irrelevant_vector,
        EMBEDDING_DIM, QUERY_RETRIEVAL, RETRIEVAL_RELEVANT_CHUNK_INDEX,
        CHUNK_VECTORS_SHORT, CHUNK_VECTORS_LONG, ROOT_VECTOR,
    )
"""

import math

# ---------------------------------------------------------------------------
# Embedding dimension (BGE-M3 dense, must match Node.embedding = Vector(1024))
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 1024

# ---------------------------------------------------------------------------
# Golden documents (D-04: 2 small curated documents)
# ---------------------------------------------------------------------------

# Short document: 2 clear topical sections (~100 words)
# Section 1 covers retrieval systems (relevant to QUERY_RETRIEVAL)
# Section 2 covers vector search (not relevant to QUERY_RETRIEVAL)
GOLDEN_DOC_SHORT = """\
Retrieval Systems Overview

Retrieval systems are software components that locate and return relevant information \
from a large corpus in response to a user query. Modern retrieval systems use inverted \
indexes, ranking algorithms, and contextual signals to identify the most pertinent \
documents. They form the backbone of search engines, question-answering pipelines, \
and recommendation platforms. Effective retrieval requires balancing precision, recall, \
and latency within the constraints of the target application.

Vector Search Fundamentals

Vector search encodes both documents and queries as dense numeric vectors in a shared \
embedding space. Similarity between a query and a document is measured using cosine \
distance or dot product. Approximate nearest neighbor algorithms such as HNSW and \
IVFFlat allow vector search to scale to millions of documents while maintaining \
sub-second query latency. The quality of the embedding model is the primary determinant \
of retrieval accuracy in vector-based systems.\
"""

# Long document: 3 sections (~150 words)
# Section 1: Introduction to Language Models (relevant to QUERY_LLM)
# Section 2: Transformer Architecture
# Section 3: Applications in Information Retrieval
GOLDEN_DOC_LONG = """\
Introduction to Language Models

Language models are probabilistic systems that assign a probability distribution over \
sequences of tokens. Large language models trained on vast corpora of text develop rich \
internal representations of linguistic structure, world knowledge, and reasoning patterns. \
They can generate coherent prose, answer questions, and follow complex instructions by \
predicting the most likely continuation given a context window of preceding tokens.

Transformer Architecture

The transformer architecture introduced the self-attention mechanism, which allows each \
token to attend to every other token in the sequence simultaneously. This parallel \
computation replaced the sequential processing of recurrent networks, enabling training \
on much larger datasets with greater efficiency. Positional encodings preserve token \
order within the fixed-length context window used by the attention layers.

Applications in Information Retrieval

Language models have transformed information retrieval by enabling semantic search beyond \
keyword matching. Dense retrieval models encode queries and passages into a shared vector \
space, allowing retrieval by semantic similarity. Retrieval-augmented generation pipelines \
combine a retrieval engine with a language model to produce grounded, factual responses \
to open-domain questions using evidence drawn from a document corpus.\
"""

# ---------------------------------------------------------------------------
# Mock chunk responses (D-06: stored as plain dicts matching ChunkResult schema)
# Each chunk_text must be >= 20 tokens (ChunkResult validator rejects < 20)
# start_idx and end_idx are character offsets into the respective golden document
# ---------------------------------------------------------------------------

# Split at the blank line between sections in GOLDEN_DOC_SHORT
_SHORT_SPLIT = GOLDEN_DOC_SHORT.index("\n\nVector Search Fundamentals")

MOCK_CHUNKS_SHORT: list[dict] = [
    {
        "chunk_text": (
            "Retrieval systems are software components that locate and return relevant "
            "information from a large corpus in response to a user query. Modern retrieval "
            "systems use inverted indexes, ranking algorithms, and contextual signals to "
            "identify the most pertinent documents. They form the backbone of search engines, "
            "question-answering pipelines, and recommendation platforms. Effective retrieval "
            "requires balancing precision, recall, and latency within the constraints of the "
            "target application."
        ),
        "start_idx": GOLDEN_DOC_SHORT.index("Retrieval systems are software"),
        "end_idx": GOLDEN_DOC_SHORT.index("target application.") + len("target application."),
    },
    {
        "chunk_text": (
            "Vector search encodes both documents and queries as dense numeric vectors in a "
            "shared embedding space. Similarity between a query and a document is measured "
            "using cosine distance or dot product. Approximate nearest neighbor algorithms "
            "such as HNSW and IVFFlat allow vector search to scale to millions of documents "
            "while maintaining sub-second query latency. The quality of the embedding model "
            "is the primary determinant of retrieval accuracy in vector-based systems."
        ),
        "start_idx": GOLDEN_DOC_SHORT.index("Vector search encodes"),
        "end_idx": len(GOLDEN_DOC_SHORT),
    },
]

# Split GOLDEN_DOC_LONG at the two section boundaries
_LONG_SPLIT1 = GOLDEN_DOC_LONG.index("\n\nTransformer Architecture")
_LONG_SPLIT2 = GOLDEN_DOC_LONG.index("\n\nApplications in Information Retrieval")

MOCK_CHUNKS_LONG: list[dict] = [
    {
        "chunk_text": (
            "Language models are probabilistic systems that assign a probability distribution "
            "over sequences of tokens. Large language models trained on vast corpora of text "
            "develop rich internal representations of linguistic structure, world knowledge, "
            "and reasoning patterns. They can generate coherent prose, answer questions, and "
            "follow complex instructions by predicting the most likely continuation given a "
            "context window of preceding tokens."
        ),
        "start_idx": GOLDEN_DOC_LONG.index("Language models are probabilistic"),
        "end_idx": GOLDEN_DOC_LONG.index("context window of preceding tokens.")
        + len("context window of preceding tokens."),
    },
    {
        "chunk_text": (
            "The transformer architecture introduced the self-attention mechanism, which "
            "allows each token to attend to every other token in the sequence simultaneously. "
            "This parallel computation replaced the sequential processing of recurrent "
            "networks, enabling training on much larger datasets with greater efficiency. "
            "Positional encodings preserve token order within the fixed-length context window "
            "used by the attention layers."
        ),
        "start_idx": GOLDEN_DOC_LONG.index("The transformer architecture"),
        "end_idx": GOLDEN_DOC_LONG.index("used by the attention layers.")
        + len("used by the attention layers."),
    },
    {
        "chunk_text": (
            "Language models have transformed information retrieval by enabling semantic "
            "search beyond keyword matching. Dense retrieval models encode queries and "
            "passages into a shared vector space, allowing retrieval by semantic similarity. "
            "Retrieval-augmented generation pipelines combine a retrieval engine with a "
            "language model to produce grounded, factual responses to open-domain questions "
            "using evidence drawn from a document corpus."
        ),
        "start_idx": GOLDEN_DOC_LONG.index("Language models have transformed"),
        "end_idx": len(GOLDEN_DOC_LONG),
    },
]

# ---------------------------------------------------------------------------
# Mock tree structures (list of dicts per LLM tree schema)
# Used to configure mock_llm_service.complete_structured side_effect
# ---------------------------------------------------------------------------

# Tree for GOLDEN_DOC_SHORT:
# root
#   leaf (chunk_index=0, retrieval systems section)
#   leaf (chunk_index=1, vector search section)
MOCK_TREE_SHORT: list[dict] = [
    {
        "node_type": "root",
        "depth": 0,
        "position": 0,
        "title": "Retrieval and Vector Search",
        "summary": "Overview of retrieval systems and vector search fundamentals",
        "children": [
            {
                "node_type": "leaf",
                "depth": 1,
                "position": 0,
                "title": None,
                "summary": None,
                "children": [],
                "chunk_index": 0,
            },
            {
                "node_type": "leaf",
                "depth": 1,
                "position": 1,
                "title": None,
                "summary": None,
                "children": [],
                "chunk_index": 1,
            },
        ],
        "chunk_index": None,
    }
]

# Tree for GOLDEN_DOC_LONG:
# root
#   leaf (chunk_index=0, language models section)
#   leaf (chunk_index=1, transformer architecture section)
#   leaf (chunk_index=2, applications in IR section)
MOCK_TREE_LONG: list[dict] = [
    {
        "node_type": "root",
        "depth": 0,
        "position": 0,
        "title": "Language Models, Transformers, and Information Retrieval",
        "summary": (
            "Introduction to language models, transformer"
            " architecture, and their applications in"
            " information retrieval"
        ),
        "children": [
            {
                "node_type": "leaf",
                "depth": 1,
                "position": 0,
                "title": None,
                "summary": None,
                "children": [],
                "chunk_index": 0,
            },
            {
                "node_type": "leaf",
                "depth": 1,
                "position": 1,
                "title": None,
                "summary": None,
                "children": [],
                "chunk_index": 1,
            },
            {
                "node_type": "leaf",
                "depth": 1,
                "position": 2,
                "title": None,
                "summary": None,
                "children": [],
                "chunk_index": 2,
            },
        ],
        "chunk_index": None,
    }
]

# ---------------------------------------------------------------------------
# Golden query constants
# ---------------------------------------------------------------------------

# Query relevant to GOLDEN_DOC_SHORT chunk 0 (retrieval systems section)
QUERY_RETRIEVAL = "What are retrieval systems used for?"
RETRIEVAL_RELEVANT_CHUNK_INDEX = 0  # chunk at position 0 of GOLDEN_DOC_SHORT

# Query relevant to GOLDEN_DOC_LONG chunk 0 (language models section)
QUERY_LLM = "How do language models work?"
LLM_RELEVANT_CHUNK_INDEX = 0  # chunk at position 0 of GOLDEN_DOC_LONG

# ---------------------------------------------------------------------------
# Deterministic vector helpers (D-08: known cosine similarities)
# All vectors are exactly EMBEDDING_DIM (1024) floats.
# ---------------------------------------------------------------------------


def make_query_vector() -> list[float]:
    """Unit vector along the first axis: [1.0, 0.0, 0.0, ..., 0.0].

    This is the canonical query direction against which relevant/irrelevant
    vectors are defined.
    """
    v = [0.0] * EMBEDDING_DIM
    v[0] = 1.0
    return v


def make_relevant_vector(similarity: float = 0.95) -> list[float]:
    """Vector with cosine similarity approximately equal to `similarity` to make_query_vector().

    Construction: v[0] = similarity, v[1] = sqrt(1 - similarity^2), rest zeros.
    The resulting vector is already unit-length by construction (verified below).

    cos(query, v) = dot([1,0,...], [sim, sqrt(1-sim^2), 0,...]) = sim
    """
    v = [0.0] * EMBEDDING_DIM
    v[0] = similarity
    complement = math.sqrt(max(0.0, 1.0 - similarity**2))
    v[1] = complement
    # Vector is unit-length by construction: v[0]^2 + v[1]^2 = sim^2 + (1-sim^2) = 1
    # Normalize for numerical robustness
    magnitude = math.sqrt(sum(x * x for x in v))
    if magnitude > 0:
        v = [x / magnitude for x in v]
    return v


def make_irrelevant_vector() -> list[float]:
    """Unit vector orthogonal to make_query_vector(): [0.0, 1.0, 0.0, ..., 0.0].

    Cosine similarity to make_query_vector() = dot([1,0,...], [0,1,...]) = 0.0
    """
    v = [0.0] * EMBEDDING_DIM
    v[1] = 1.0
    return v


# ---------------------------------------------------------------------------
# Pre-computed vector assignments (module-level constants for deterministic tests)
# ---------------------------------------------------------------------------

# GOLDEN_DOC_SHORT vectors:
# - Chunk 0 (retrieval systems, relevant to QUERY_RETRIEVAL): high similarity
# - Chunk 1 (vector search, not relevant to QUERY_RETRIEVAL): orthogonal
CHUNK_VECTORS_SHORT: list[list[float]] = [
    make_relevant_vector(0.95),
    make_irrelevant_vector(),
]

# GOLDEN_DOC_LONG vectors:
# All chunks are "irrelevant" to QUERY_RETRIEVAL (this doc is about language models)
# Tests involving GOLDEN_DOC_LONG use QUERY_LLM which is configured separately
CHUNK_VECTORS_LONG: list[list[float]] = [
    make_irrelevant_vector(),
    make_irrelevant_vector(),
    make_irrelevant_vector(),
]

# Default root vector for documents
ROOT_VECTOR: list[float] = make_irrelevant_vector()
