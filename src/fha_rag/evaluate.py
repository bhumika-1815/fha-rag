import json
from pathlib import Path

from fha_rag.retrieve import BM25Retriever, DenseRetriever, HybridRetriever, load_chunks

GOLDEN = Path("evals/golden.jsonl")
K = 10


def first_hit_rank(results, gold_ids):
    """Position (1 = top) of the first retrieved chunk containing a gold section, or None."""
    for rank, chunk in enumerate(results, start=1):
        if set(chunk["section_ids"]) & set(gold_ids):
            return rank
    return None


def evaluate(retriever, questions, verbose=False):
    ranks = []
    for q in questions:
        rank = first_hit_rank(retriever.search(q["question"], k=K), q["gold_ids"])
        ranks.append(rank)
        if verbose:
            print(f"   {q['qid']}  rank {rank or 'MISS':>4}  {q['question'][:70]}")
    n = len(ranks)
    return {
        "recall@1": sum(r is not None and r <= 1 for r in ranks) / n,
        "recall@5": sum(r is not None and r <= 5 for r in ranks) / n,
        "recall@10": sum(r is not None for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


if __name__ == "__main__":
    questions = [json.loads(line) for line in GOLDEN.read_text().splitlines()]
    answerable = [q for q in questions if q["gold_ids"]]   # unanswerable ones are scored in Step 7
    print(f"{len(answerable)} answerable questions\n")
    for chunking in ("fixed", "section"):
        chunks = load_chunks(chunking)
        bm25, dense = BM25Retriever(chunks), DenseRetriever(chunks, chunking)
        for label, retriever in (("bm25", bm25), ("dense", dense),
                                 ("hybrid", HybridRetriever(bm25, dense))):
            print(f"{label} + {chunking}")
            scores = evaluate(retriever, answerable, verbose=True)
            print("   " + "  ".join(f"{name} {value:.2f}" for name, value in scores.items()) + "\n")
