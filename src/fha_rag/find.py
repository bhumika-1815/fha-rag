import json
import sys

rows = [json.loads(l) for l in open("data/processed/sections_text.jsonl")]
q = " ".join(sys.argv[1:]).lower()
hits = [r for r in rows if q in r["text"].lower()]
print(len(hits), "sections contain:", q)
for r in hits[:15]:
    i = r["text"].lower().find(q)
    print(f"\n[id {r['id']}] {r['citation']} | {' > '.join(r['path'].split(' > ')[-2:])}")
    print("   ...", r["text"][max(0, i - 80): i + 160].replace("\n", " "), "...")