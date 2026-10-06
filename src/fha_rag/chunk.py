import json
from pathlib import Path

SECTIONS = Path("data/processed/sections_text.jsonl")
OUT_DIR = Path("data/processed")

SIZE = 300      # target words per chunk
OVERLAP = 50    # words shared between neighbouring fixed chunks
MIN_WORDS = 80  # section-aware: don't close a chunk smaller than this
ANCHOR = 4      # section-aware: a heading at this level or above starts a new chunk


def load():
    return [json.loads(line) for line in SECTIONS.read_text().splitlines()]


def fixed(rows):
    """Baseline: ignore structure, slide a window over the whole handbook."""
    words = []   # (word, section id) for every word in reading order
    for r in rows:
        words += [(w, r["id"]) for w in r["text"].split()]
    by_id = {r["id"]: r for r in rows}
    chunks = []
    for start in range(0, len(words), SIZE - OVERLAP):
        window = words[start:start + SIZE]
        ids = sorted({i for _, i in window})
        chunks.append({
            "section_ids": ids,
            "citation": by_id[ids[0]]["citation"],
            "text": " ".join(w for w, _ in window),
        })
    return chunks


def common_path(group):
    """The part of the breadcrumb that every section in the group shares."""
    paths = [r["path"].split(" > ") for r in group]
    shared = []
    for parts in zip(*paths):
        if len(set(parts)) > 1:
            break
        shared.append(parts[0])
    return " > ".join(shared)


def section_aware(rows):
    """Merge small neighbours, split giants, and label every chunk with its path."""
    groups, group, size = [], [], 0
    for r in rows:
        new_topic = r["level"] <= ANCHOR and size >= 20   # a bare heading stays with what follows
        too_big = size + r["n_words"] > SIZE
        sibling_done = group and r["level"] <= group[0]["level"] and size >= MIN_WORDS
        if group and (new_topic or too_big or sibling_done):
            groups.append(group)
            group, size = [], 0
        group.append(r)
        size += r["n_words"]
    groups.append(group)

    chunks = []
    for g in groups:
        header = common_path(g)
        body = "\n".join(r["text"] for r in g).split()
        step = SIZE - OVERLAP if len(body) > SIZE else len(body) or 1
        for start in range(0, len(body), step):   # one pass unless the group is a giant
            chunks.append({
                "section_ids": [r["id"] for r in g],
                "citation": g[0]["citation"],
                "text": header + "\n" + " ".join(body[start:start + SIZE]),
            })
            if start + SIZE >= len(body):
                break
    return chunks


def save(name, chunks):
    for n, c in enumerate(chunks):
        c["chunk_id"] = f"{name}-{n}"
        c["n_words"] = len(c["text"].split())
    (OUT_DIR / f"chunks_{name}.jsonl").write_text("\n".join(json.dumps(c) for c in chunks))
    sizes = sorted(c["n_words"] for c in chunks)
    print(f"{name:14} chunks: {len(chunks):5} | median {sizes[len(sizes) // 2]:4} "
          f"| min {sizes[0]:3} | max {sizes[-1]:4} | under 40 words: {sum(s < 40 for s in sizes)}")


if __name__ == "__main__":
    rows = load()
    save("fixed", fixed(rows))
    save("section", section_aware(rows))