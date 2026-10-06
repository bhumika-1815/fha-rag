import json
import re
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from rank_bm25 import BM25Okapi

DATA = Path("data/processed")
INDEX = Path("data/index")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

def load_chunks(name):
    """name is 'fixed' or 'section'."""
    return [json.loads(line) for line in (DATA / f"chunks_{name}.jsonl").read_text().splitlines()]


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    """Keyword search: scores a chunk by how many of the question's words it contains."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.index = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def search(self, question, k=10):
        scores = self.index.get_scores(tokenize(question))
        best = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self.chunks[i] for i in best]

class DenseRetriever:
    """Semantic search: compares the meaning of the question with the meaning of each chunk."""

    def __init__(self, chunks, name):
        self.chunks = chunks
        self.model = SentenceTransformer(MODEL_NAME)
        cache = INDEX / f"emb_{name}.npy"
        if cache.exists() and len(np.load(cache)) == len(chunks):
            self.emb = np.load(cache)
        else:
            self.emb = self.model.encode(
                [c["text"] for c in chunks],
                batch_size=32, normalize_embeddings=True, show_progress_bar=True,
            )
            INDEX.mkdir(parents=True, exist_ok=True)
            np.save(cache, self.emb)

    def search(self, question, k=10):
        q = self.model.encode(QUERY_PREFIX + question, normalize_embeddings=True)
        scores = self.emb @ q                      # similarity of the question to every chunk
        best = np.argsort(-scores)[:k]
        return [self.chunks[i] for i in best]

        
class HybridRetriever:
    """Combine keyword and semantic rankings with Reciprocal Rank Fusion."""

    def __init__(self, bm25, dense, depth=50):
        self.bm25, self.dense, self.depth = bm25, dense, depth

    def search(self, question, k=10):
        scores, by_id = {}, {}
        for retriever in (self.bm25, self.dense):
            for rank, c in enumerate(retriever.search(question, k=self.depth), start=1):
                scores[c["chunk_id"]] = scores.get(c["chunk_id"], 0) + 1 / (60 + rank)
                by_id[c["chunk_id"]] = c
        best = sorted(scores, key=lambda i: -scores[i])[:k]
        return [by_id[i] for i in best]