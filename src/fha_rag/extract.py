import json
from pathlib import Path
import re
import pymupdf

PDF = Path("data/raw/hud_4000_1.pdf")
SECTIONS = Path("data/processed/sections.jsonl")
OUT = Path("data/processed/sections_text.jsonl")
FOOTER = re.compile(r"Handbook 4000\.1\s*\n\d+\s*\nLast Revised: \S+\s*\n?")

BODY_TOP = 84       # <-- replace with your number
BODY_BOTTOM = 733   # <-- replace with your number


def text_between(doc, start, end):
    """All body text from one heading position to the next."""
    (p0, y0), (p1, y1) = start, end
    parts = []
    for p in range(p0, p1 + 1):
        page = doc[p]
        top = max(y0 - 2, BODY_TOP) if p == p0 else BODY_TOP
        bottom = min(y1 - 2, BODY_BOTTOM) if p == p1 else BODY_BOTTOM
        if bottom <= top:
            continue
        clip = pymupdf.Rect(0, top, page.rect.width, bottom)
        parts.append(page.get_text("text", clip=clip).strip())
    return FOOTER.sub("", "\n".join(x for x in parts if x))


def build():
    doc = pymupdf.open(PDF)
    # position of every bookmark, in the same order toc.py used for "id"
    pos = []
    for level, title, page, dest in doc.get_toc(simple=False):
        to = dest.get("to")
        pos.append((page - 1, to.y if to else BODY_TOP))

    rows = [json.loads(line) for line in SECTIONS.read_text().splitlines()]
    out = []
    for r in rows:
        i = r["id"]
        end = pos[i + 1] if i + 1 < len(pos) else (doc.page_count - 1, BODY_BOTTOM)
        r["text"] = text_between(doc, pos[i], end)
        r["n_words"] = len(r["text"].split())
        out.append(r)

    OUT.write_text("\n".join(json.dumps(r) for r in out))

    words = sorted(r["n_words"] for r in out)
    n = len(words)
    print("Sections:", n, "| total words:", sum(words))
    print("Empty (0 words):", sum(w == 0 for w in words))
    print("Under 30 words:", sum(w < 30 for w in words))
    print("Median:", words[n // 2], "| 90th pct:", words[int(n * 0.9)], "| max:", words[-1])
    print("--- sample: id 304 ---")
    print(out[304]["text"][:600])


if __name__ == "__main__":
    build()