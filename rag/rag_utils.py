import os
import json
import numpy as np
from google import genai
from google.genai import types

EMBED_MODEL = "gemini-embedding-001"
INDEX_FILE = os.path.join(os.path.dirname(__file__), "chunk_Index.json")

def chunk_markdown(text, min_chunk_chars=200):
    """Split on markdown '##' headers, merging tiny sections together."""
    raw_sections = text.split("\n## ")
    chunks = []
    for i, section in enumerate(raw_sections):
        section = section.strip()
        if not section:
            continue
        if i > 0:
            section = "## " + section
        chunks.append(section)

    # Merge sections that are too short into the next one
    merged = []
    buffer = ""
    for c in chunks:
        buffer = (buffer + "\n\n" + c).strip() if buffer else c
        if len(buffer) >= min_chunk_chars:
            merged.append(buffer)
            buffer = ""
    if buffer:
        merged.append(buffer)
    return merged


def embed_texts(client, texts, task_type):
    """task_type: 'RETRIEVAL_DOCUMENT' for chunks, 'RETRIEVAL_QUERY' for questions."""
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=768),
    )
    return [np.array(e.values) for e in result.embeddings]


def build_index(client, kb_path="kb.md"):
    with open(kb_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_markdown(text)
    print(f"Split into {len(chunks)} chunks.")
    vectors = embed_texts(client, chunks, "RETRIEVAL_DOCUMENT")
    index = [{"chunk": c, "vector": v.tolist()} for c, v in zip(chunks, vectors)]
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f)
    print(f"Saved index with {len(index)} chunks to {INDEX_FILE}")


def load_index():
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve(client, query, top_k=3):
    index = load_index()
    query_vec = embed_texts(client, [query], "RETRIEVAL_QUERY")[0]
    scored = [(cosine_sim(query_vec, item["vector"]), item["chunk"]) for item in index]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]