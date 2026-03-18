"""
Ingest scraped blog posts into Pinecone vector store.

Supports multiple sources (Die, Workwear!, Permanent Style, etc.)
in a single unified index for RAG queries.

Usage:
    python ingest.py                          # Ingest all known sources
    python ingest.py path/to/posts.json       # Ingest a single file
    python ingest.py --reset                  # Wipe index and re-ingest all
"""

import json
import sys
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

CHUNK_SIZE = 800  # words per chunk
CHUNK_OVERLAP = 100  # word overlap between chunks
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "pcsk_2PdVrk_CiVFvEbYqs8ts2GqJ69PMurqFpkeJDBjA1JdoPCzGJ6xvovgtn8jxLHyy5STAbH")
INDEX_NAME = "style-advice"

# Known data sources (relative to ~/Downloads)
KNOWN_SOURCES = [
    os.path.expanduser("~/Downloads/dieworkwear_posts.json"),
    os.path.expanduser("~/Downloads/permanentstyle_posts.json"),
    os.path.expanduser("~/Downloads/putthison_posts.json"),
    os.path.expanduser("~/Downloads/highsnobiety_posts.json"),
    os.path.expanduser("~/Downloads/hypebeast_posts.json"),
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


def ingest_file(json_path, index, model, source_prefix):
    """Ingest a single JSON file into Pinecone. Returns (posts, chunks) counts."""
    print(f"\nLoading posts from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    posts = data["posts"]
    source = data.get("source", os.path.basename(json_path))
    author = data.get("author", "Unknown")
    print(f"  Source: {source} | Author: {author}")
    print(f"  Found {len(posts)} posts, {data.get('total_words', 0):,} total words")

    all_chunks = []
    all_metadatas = []
    all_ids = []

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
                "text": chunk,  # Store the text in metadata for retrieval
            }
            all_chunks.append(chunk)
            all_metadatas.append(meta)
            all_ids.append(doc_id)

    # Embed and upsert in batches
    batch_size = 100  # Pinecone recommends ~100 vectors per upsert
    for start in range(0, len(all_chunks), batch_size):
        end = min(start + batch_size, len(all_chunks))
        batch_texts = all_chunks[start:end]
        batch_ids = all_ids[start:end]
        batch_metas = all_metadatas[start:end]

        # Generate embeddings
        embeddings = model.encode(batch_texts).tolist()

        # Upsert to Pinecone
        vectors = []
        for vid, emb, meta in zip(batch_ids, embeddings, batch_metas):
            vectors.append({"id": vid, "values": emb, "metadata": meta})
        index.upsert(vectors=vectors)

        print(f"  Indexed {end}/{len(all_chunks)} chunks...")

    post_count = sum(1 for p in posts if p.get("text"))
    print(f"  ✓ Indexed {len(all_chunks)} chunks from {post_count} posts")
    return post_count, len(all_chunks)


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

    # Set up Pinecone
    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    if reset:
        print("  Deleting all vectors...")
        index.delete(delete_all=True)
        print("  ✓ Index cleared")

    # Load embedding model
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Ingest each file
    total_posts = 0
    total_chunks = 0
    for path, prefix in files:
        posts, chunks = ingest_file(path, index, model, prefix)
        total_posts += posts
        total_chunks += chunks

    # Check final stats
    stats = index.describe_index_stats()

    print(f"\n{'=' * 60}")
    print(f"  ✅ DONE!")
    print(f"{'=' * 60}")
    print(f"  Total posts:   {total_posts:,}")
    print(f"  Total chunks:  {total_chunks:,}")
    print(f"  Pinecone vectors: {stats['total_vector_count']:,}")
    print(f"  Index:         {INDEX_NAME}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
