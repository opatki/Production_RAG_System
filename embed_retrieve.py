"""
embed_retrieve.py — Milestone 4: Embedding + Retrieval for "The Unofficial Guide".

Consumes the chunks produced by ingest_pipeline.build_corpus(), embeds them with the
local HuggingFace `all-MiniLM-L6-v2` sentence-transformer (384-dim vectors), and
populates a PERSISTENT ChromaDB collection. Exposes:

    retrieve_context(query, k=5) -> list[dict]
        Each result: {"text": ..., "metadata": {...}, "distance": float}

Design choices (see planning.md > Retrieval Approach):
  - Model: all-MiniLM-L6-v2 — 384-dim, fast, free, runs locally (no API cost).
  - Distance: COSINE. Embeddings are L2-normalized, so on-topic hits score well
    below the 0.5 threshold from the Evaluation Plan, and off-topic queries stay high.
  - Persistence: ChromaDB PersistentClient at ./chroma_db, so we embed once and reuse.

Run:
    python embed_retrieve.py            # build (if needed) + run eval queries
    python embed_retrieve.py --rebuild  # force a fresh re-embed
"""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ingest_pipeline import Chunk, build_corpus

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "ucd_food"
DEFAULT_K = 5
EMBED_BATCH = 256

# --------------------------------------------------------------------------- #
# Lazy singletons (model load + chroma client are expensive; build once)
# --------------------------------------------------------------------------- #

_model: SentenceTransformer | None = None
_client: chromadb.api.ClientAPI | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[model] loading {MODEL_NAME} ...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_client() -> "chromadb.api.ClientAPI":
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    return _client


def get_collection():
    """Get-or-create the collection, pinned to cosine distance."""
    return get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #


def _embed(texts: list[str]):
    """Encode texts to L2-normalized 384-dim vectors (so cosine distance is clean)."""
    return get_model().encode(
        texts,
        batch_size=EMBED_BATCH,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def _chunk_id(chunk: Chunk, i: int) -> str:
    stem = chunk.metadata["source_file"].removesuffix(".txt")
    return f"{stem}-{i}"


def _embed_text(chunk: Chunk) -> str:
    """Text actually embedded + stored as the document.

    Track A keeps the restaurant name only in metadata (per the spec), but semantic
    search sees only text — so an `Academic Year Hours` chunk never matches
    "Segundo hours". We prepend the restaurant name to the embedded body to bridge
    that entity-resolution gap (planning.md > Anticipated Challenges #1). Track B
    already carries its `[Source: ...]` prefix, so it is left untouched.
    """
    m = chunk.metadata
    if m.get("track") == "A":
        restaurant = m.get("restaurant", "")
        if restaurant and not chunk.text.startswith(restaurant):
            return f"{restaurant} — {chunk.text}"
    return chunk.text


def build_index(rebuild: bool = False):
    """Embed the corpus and populate ChromaDB. Idempotent unless rebuild=True.

    Re-embeds only when the persisted count differs from the current corpus
    (e.g. after the chunker changes), so repeat runs are fast.
    """
    client = get_client()
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("[index] dropped existing collection (--rebuild)")
        except Exception:
            pass

    collection = get_collection()
    corpus = build_corpus()

    if not rebuild and collection.count() == len(corpus):
        print(f"[index] up to date: {collection.count()} vectors — skipping embed.")
        return collection

    if collection.count() != 0:
        # Stale/partial index -> rebuild cleanly.
        client.delete_collection(COLLECTION_NAME)
        collection = get_collection()

    print(f"[index] embedding {len(corpus)} chunks with {MODEL_NAME} ...")
    texts = [_embed_text(c) for c in corpus]
    embeddings = _embed(texts)
    ids = [_chunk_id(c, i) for i, c in enumerate(corpus)]
    metadatas = [c.metadata for c in corpus]

    for i in range(0, len(corpus), EMBED_BATCH):
        sl = slice(i, i + EMBED_BATCH)
        collection.add(
            ids=ids[sl],
            documents=texts[sl],
            embeddings=embeddings[sl].tolist(),
            metadatas=metadatas[sl],
        )
    print(f"[index] done. {collection.count()} vectors persisted at {PERSIST_DIR}")
    return collection


# --------------------------------------------------------------------------- #
# Retrieval (the Milestone 4 public API)
# --------------------------------------------------------------------------- #


def retrieve_context(query: str, k: int = DEFAULT_K) -> list[dict]:
    """Return the top-k chunks for `query`, with metadata and cosine distance.

    Each element: {"text": str, "metadata": dict, "distance": float}.
    Lower distance = closer match (cosine; ~0 identical, ~1 unrelated).
    """
    collection = get_collection()
    query_emb = get_model().encode([query], normalize_embeddings=True)[0].tolist()
    res = collection.query(query_embeddings=[query_emb], n_results=k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    return [
        {"text": doc, "metadata": meta, "distance": float(dist)}
        for doc, meta, dist in zip(docs, metas, dists)
    ]


# --------------------------------------------------------------------------- #
# Verification (Evaluation Plan queries)
# --------------------------------------------------------------------------- #

EVAL_QUERIES = [
    "What are the hours for Segundo Dining Commons on a Tuesday?",
    "Does The Gunrock accept meal swipes or AggieCash?",
    "Where can I get good garlic knots and a spinach stromboli near campus?",
    "Which spots in Davis have the best spicy mango habanero wings?",
    "Are there any places near the dorms that serve Indian street food like pav bhaji?",
]

# An intentionally out-of-domain query: distances should be noticeably higher.
OFF_TOPIC_QUERY = "Where can I park my car and find a parking permit on campus?"


def _print_results(query: str, results: list[dict]) -> None:
    print(f"\n{'=' * 80}\nQUERY: {query}\n{'-' * 80}")
    for rank, r in enumerate(results, start=1):
        m = r["metadata"]
        src = m.get("source_file", "?")
        section = m.get("section") or m.get("platform") or "-"
        flag = "  <-- on-topic (<0.5)" if r["distance"] < 0.5 else ""
        print(f"[{rank}] dist={r['distance']:.3f}  {src}  | {section}{flag}")
        snippet = " ".join(r["text"].split())[:200]
        print(f"     {snippet}")


def run_evaluation() -> None:
    print("\n" + "#" * 80)
    print("# MILESTONE 4 VERIFICATION — retrieval over the Evaluation Plan queries")
    print("#" * 80)
    for q in EVAL_QUERIES:
        _print_results(q, retrieve_context(q, k=DEFAULT_K))
    print("\n" + "#" * 80)
    print("# OUT-OF-DOMAIN sanity check (distances should be higher / less confident)")
    print("#" * 80)
    _print_results(OFF_TOPIC_QUERY, retrieve_context(OFF_TOPIC_QUERY, k=DEFAULT_K))


if __name__ == "__main__":
    build_index(rebuild="--rebuild" in sys.argv)
    run_evaluation()
