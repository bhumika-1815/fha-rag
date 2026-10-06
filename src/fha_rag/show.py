import json
import sys

from fha_rag.retrieve import BM25Retriever, DenseRetriever, load_chunks

qid = sys.argv[1]
q = next(q for q in map(json.loads, open("evals/golden.jsonl")) if q["qid"] == qid)
chunks = load_chunks("section")
print(q["question"], "| gold:", q["gold_ids"])
for label, r in (("bm25", BM25Retriever(chunks)), ("dense", DenseRetriever(chunks, "section"))):
    print("\n" + label)
    for rank, c in enumerate(r.search(q["question"], k=5), 1):
        hit = "HIT " if set(c["section_ids"]) & set(q["gold_ids"]) else "    "
        print(f" {rank} {hit}{c['citation']:28} ...{c['text'].splitlines()[0][-75:]}")