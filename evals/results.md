Retrieval results on 9 answerable questions

| chunking | retriever | recall@1 | recall@5 | recall@10 | recall@50 | mrr |
|---|---|---|---|---|---|---|
| fixed | bm25 | 0.00 | 0.22 | 0.44 | 0.67 | 0.17 |
| fixed | dense | 0.22 | 0.44 | 0.56 | 0.89 | 0.31 |
| fixed | hybrid | 0.22 | 0.44 | 0.44 | 0.78 | 0.32 |
| fixed | hybrid+rerank | 0.22 | 0.67 | 0.78 | 0.78 | 0.40 |
| section | bm25 | 0.22 | 0.44 | 0.56 | 0.67 | 0.30 |
| section | dense | 0.33 | 0.56 | 0.67 | 0.89 | 0.43 |
| section | hybrid | 0.11 | 0.56 | 0.67 | 0.78 | 0.27 |
| section | hybrid+rerank | 0.11 | 0.67 | 0.78 | 0.78 | 0.29 |
