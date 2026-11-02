"""Systems side by side, per set, from the scores*.csv that score.py wrote into each output directory: word errors
(and how many were deletions / insertions), how much of the other people is left, voice similarity, quality (Audiobox
PQ), tone (dB off the studio reference's spectrum, DAPS / DR-VCTK), noise left, and gate fails. The first directory's inputs give the "input" row. Each clip's word error and
insertion rates are capped at 1: Whisper sometimes loops on a clip, and one loop (10x its words) can swamp a set.

    uv run --no-project eval/compare.py eval/outputs/dfn3 eval/outputs/colored-56 ...
"""
import csv
import math
import sys
from pathlib import Path


def mean(rows, k):
    vals = [float(r[k]) for r in rows if r.get(k) not in ("", None) and not math.isnan(float(r[k]))]
    vals = [min(v, 1) for v in vals] if k[:4] in ("wer_", "ins_") else vals
    return sum(vals) / len(vals) if vals else math.nan


systems = {}
for d in map(Path, sys.argv[1:]):
    rows = [r for f in sorted(d.glob("scores*.csv")) for r in csv.DictReader(open(f))]
    systems[d.name] = rows
sets = list(dict.fromkeys(r["set"] for rows in systems.values() for r in rows))
first = next(iter(systems.values()))
cell = lambda v, f: " " * len(f % 0) if math.isnan(v) else f % v
print(f"{'set':12} {'system':16} {'WER':>6} {'del':>6} {'ins':>6} {'others':>7} {'voice':>6} {'PQ':>5} {'tone':>5} {'noise':>6} {'fails':>5}")
for s in sets:
    inp = [r for r in first if r["set"] == s]
    lines = [("input", inp, "in")] + [(name, [r for r in rows if r["set"] == s], "out") for name, rows in systems.items()]
    for name, rs, t in lines:
        if not rs:
            continue
        fails = "" if t == "in" else str(sum(r.get("gate") == "FAIL" for r in rs))
        print(f"{s:12} {name:16} {cell(mean(rs, f'wer_{t}'), '%6.3f')} {cell(mean(rs, f'del_{t}'), '%6.3f')}"
              f" {cell(mean(rs, f'ins_{t}'), '%6.3f')} {cell(math.nan if t == 'in' else mean(rs, 'other_db'), '%+7.1f')}"
              f" {cell(mean(rs, f'spk_{t}'), '%6.3f')} {cell(mean(rs, f'pq_{t}'), '%5.2f')}"
              f" {cell(mean(rs, f'tone_{t}'), '%5.2f')}"
              f" {cell(mean(rs, f'noise_{t}'), '%6.1f')} {fails:>5}")
    print()
