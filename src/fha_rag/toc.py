import json
import re
from collections import Counter
from pathlib import Path

import fitz

PDF = Path("data/raw/hud_4000_1.pdf")
OUT = Path("data/processed/sections.jsonl")

DATE = re.compile(r"\s*\((\d{2}/\d{2}/\d{4})\)\s*$")    # "(03/14/2016)" at the end
LABEL = re.compile(r"^(\(?[A-Za-z0-9]+[.)])\s+(.*)$")   # "ii." or "(A)" at the start


def build():
    doc = fitz.open(PDF)
    stack = []   # the headings above the current one
    rows = []
    for i, (level, title, page) in enumerate(doc.get_toc()):
        title = " ".join(title.split())          # clean stray whitespace
        date = None
        if m := DATE.search(title):
            date, title = m.group(1), title[: m.start()]
        label, name = "", title
        if m := LABEL.match(title):
            label, name = m.group(1), m.group(2)

        stack = stack[: level - 1] + [(label, name)]
        rows.append({
            "id": i,
            "level": level,
            "title": name,
            "citation": "".join(l for l, _ in stack).rstrip(".").replace(".(", "("),
            "path": " > ".join(n for _, n in stack),
            "effective_date": date,
            "page": page,
        })

    OUT.write_text("\n".join(json.dumps(r) for r in rows))
    print("PDF pages:", doc.page_count)
    print("Sections:", len(rows))
    print("Per level:", dict(sorted(Counter(r["level"] for r in rows).items())))
    print("With a date:", sum(r["effective_date"] is not None for r in rows))
    for r in rows[300:305]:
        print(r)


if __name__ == "__main__":
    build()