import json
from pathlib import Path

from fha_rag.retrieve import (BM25Retriever, DenseRetriever, HybridRetriever,
                              RerankRetriever, load_chunks)

GOLDEN = Path("evals/golden.jsonl")
RESULTS = Path("evals/results.md")
K = 50


def first_hit_rank(results, gold_ids):
    """Position (1 = top) of the first retrieved chunk containing a gold section, or None."""
    for rank, chunk in enumerate(results, start=1):
        if set(chunk["section_ids"]) & set(gold_ids):
            return rank
    return None


def evaluate(retriever, questions):
    ranks = [first_hit_rank(retriever.search(q["question"], k=K), q["gold_ids"]) for q in questions]
    n = len(ranks)
    scores = {f"recall@{k}": sum(r is not None and r <= k for r in ranks) / n for k in (1, 5, 10, 50)}
    scores["mrr"] = sum(1 / r for r in ranks if r) / n
    return scores, ranks


if __name__ == "__main__":
    questions = [json.loads(line) for line in GOLDEN.read_text().splitlines()]
    answerable = [q for q in questions if q["gold_ids"]]   # unanswerable ones are scored in Step 7

    table, per_question = [], {}
    for chunking in ("fixed", "section"):
        chunks = load_chunks(chunking)
        bm25, dense = BM25Retriever(chunks), DenseRetriever(chunks, chunking)
        hybrid = HybridRetriever(bm25, dense)
        for label, retriever in (("bm25", bm25), ("dense", dense), ("hybrid", hybrid),
                                 ("hybrid+rerank", RerankRetriever(hybrid))):
            scores, ranks = evaluate(retriever, answerable)
            table.append((chunking, label, scores))
            per_question[f"{label}/{chunking}"] = ranks
            print("done:", label, "+", chunking)

    names = list(table[0][2])
    lines = [f"Retrieval results on {len(answerable)} answerable questions", "",
             "| chunking | retriever | " + " | ".join(names) + " |",
             "|---|---|" + "---|" * len(names)]
    for chunking, label, scores in table:
        lines.append(f"| {chunking} | {label} | " + " | ".join(f"{scores[n]:.2f}" for n in names) + " |")
    RESULTS.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    print("\nRank of the first correct chunk, section chunking (- = not in top 50)")
    cols = [c for c in per_question if c.endswith("/section")]
    print("qid   " + "  ".join(f"{c.split('/')[0]:>13}" for c in cols))
    for i, q in enumerate(answerable):
        print(f"{q['qid']}  " + "  ".join(f"{per_question[c][i] or '-':>13}" for c in cols))