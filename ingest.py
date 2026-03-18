"""
Ingest scraped blog posts into a ChromaDB vector store.

Supports multiple sources (Die, Workwear!, Permanent Style, etc.)
in a single unified collection for RAG queries.

Usage:
    python ingest.py                          # Ingest all known sources
    python ingest.py path/to/posts.json       # Ingest a single file
    python ingest.py --reset                  # Wipe DB and re-ingest all

Defaults to ingesting both:
    - ~/Downloads/dieworkwear_posts.json
    - ~/Downloads/permanentstyle_posts.json
"""

import json
import sys
import os
import chromadb
from chromadb.utils import embedding_functions

CHUNK_SIZE = 800  # words per chunk
CHUNK_OVERLAP = 100  # word overlap between chunks
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "style_advice"

# Known data sources (relative to ~/Downloads)
KNOWN_SOURCES = [
    os.path.expanduser("~/Downloads/dieworkwear_posts.json"),
    os.path.expanduser("~/Downloads/permanentstyle_posts.json"),
    os.path.expanduser("~/Downloads/putthison_posts.json"),
]


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks


def ingest_file(json_path, collection, source_prefix):
    """Ingest a single JSON file into the collection. Returns (posts, chunks) counts."""
    print(f"\nLoading posts from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    posts = data["posts"]
    source = data.get("source", os.path.basename(json_path))
    author = data.get("author", "Unknown")
    print(f"  Source: {source} | Author: {author}")
    print(f"  Found {len(posts)} posts, {data.get('total_words', 0):,} total words")

    documents = []
    metadatas = []
    ids = []

    for i, post in enumerate(posts):
        if not post.get("text"):
            continue

        chunks = chunk_text(post["text"])
        for j, chunk in enumerate(chunks):
            doc_id = f"{source_prefix}_post_{i:04d}_chunk_{j:03d}"
            meta = {
                "source": source,
                "author": author,
                "title": post.get("title", ""),
                "date": post.get("date", ""),
                "url": post.get("url", ""),
                "categories": ", ".join(post.get("categories", [])),
                "chunk_index": j,
                "total_chunks": len(chunks),
                "word_count": len(chunk.split()),
            }
            documents.append(chunk)
            metadatas.append(meta)
            ids.append(doc_id)

    # ChromaDB has a batch limit, so add in batches of 500
    batch_size = 500
    for start in range(0, len(documents), batch_size):
        end = min(start + batch_size, len(documents))
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        print(f"  Indexed {end}/{len(documents)} chunks...")

    post_count = sum(1 for p in posts if p.get("text"))
    print(f"  ✓ Indexed {len(documents)} chunks from {post_count} posts")
    return post_count, len(documents)


def main():
    reset = "--reset" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--reset"]

    # Determine which files to ingest
    if args:
        files = [(os.path.abspath(a), os.path.splitext(os.path.basename(a))[0]) for a in args]
    else:
        files = []
        for path in KNOWN_SOURCES:
            if os.path.exists(path):
                prefix = os.path.splitext(os.path.basename(path))[0]
                files.append((path, prefix))
            else:
                print(f"  Skipping {path} (not found)")

    if not files:
        print("No data files found. Run the scrapers first.")
        return

    # Set up embedding function
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=DB_DIR)

    # Delete existing collection if resetting or re-ingesting all
    if reset or not args:
        for name in [COLLECTION_NAME, "dieworkwear"]:
            try:
                client.delete_collection(name)
                print(f"  Deleted existing '{name}' collection")
            except Exception:
                pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # Ingest each file
    total_posts = 0
    total_chunks = 0
    for path, prefix in files:
        posts, chunks = ingest_file(path, collection, prefix)
        total_posts += posts
        total_chunks += chunks

    print(f"\n{'=' * 60}")
    print(f"  ✅ DONE!")
    print(f"{'=' * 60}")
    print(f"  Total posts:   {total_posts:,}")
    print(f"  Total chunks:  {total_chunks:,}")
    print(f"  Collection:    {COLLECTION_NAME}")
    print(f"  Vector store:  {DB_DIR}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
