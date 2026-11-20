"""Generate a self-contained, loudness-matched blind A/B listening page (HTML with embedded FLAC/WAV data URIs).

Usage:
    uv run eval/make_blind_listen.py --out listen.html --seed 20261119 \
      --trial "DAPS iPad ConfRoom|raw=...|ref=...|Base=...|StudioNet=..." ...
"""
import argparse
import base64
import html
import io
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf


def active_rms(x, sr=48000):
    frame = sr // 50
    n = (len(x) // frame) * frame
    if n == 0:
        return float(np.sqrt(np.mean(x ** 2)) + 1e-8)
    e = (x[:n].reshape(-1, frame) ** 2).mean(axis=1)
    act = e >= e.max() * 1e-4
    return float(np.sqrt(e[act].mean()) + 1e-8) if act.any() else float(np.sqrt(e.mean()) + 1e-8)


def load_trim_norm(path, max_sec=14.0, target_rms=0.06):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    if max_sec and len(x) > int(max_sec * sr):
        x = x[: int(max_sec * sr)]
    g = target_rms / active_rms(x, sr)
    y = x * g
    peak = float(np.max(np.abs(y)))
    if peak > 0.891:  # -1 dBTP ceiling
        y = y * (0.891 / peak)
    buf = io.BytesIO()
    sf.write(buf, y.astype(np.float32), sr, format="FLAC", subtype="PCM_24")
    return "data:audio/flac;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_html(trials_data, key_data):
    cards = []
    for idx, t in enumerate(trials_data, 1):
        btns = []
        for label, uri in t["blind"]:
            btns.append(
                f'<div class="item"><strong>{html.escape(label)}</strong>'
                f'<audio controls preload="none" src="{uri}"></audio></div>'
            )
        refs = []
        for label, uri in t["refs"]:
            refs.append(
                f'<div class="item ref"><strong>{html.escape(label)}</strong>'
                f'<audio controls preload="none" src="{uri}"></audio></div>'
            )
        reveal = ", ".join(f"{k} = {v}" for k, v in t["reveal"].items())
        cards.append(
            f"""<section class="card">
  <h2>Trial {idx}: {html.escape(t['title'])}</h2>
  <div class="grid">{''.join(btns)}</div>
  <div class="grid refs">{''.join(refs)}</div>
  <details><summary>Reveal blind labels</summary><p class="reveal">{html.escape(reveal)}</p></details>
</section>"""
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>CleanMyMic Studio — Blind Level-Matched Listening</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; background: #0f1115; color: #eceff4; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 0.25rem; }}
  p.sub {{ color: #9aa4b2; margin-top: 0; }}
  .card {{ background: #171a21; border: 1px solid #262b36; border-radius: 12px; padding: 1.25rem; margin: 1.25rem 0; }}
  .card h2 {{ font-size: 1.05rem; margin: 0 0 0.85rem 0; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.75rem; }}
  .refs {{ margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #262b36; }}
  .item {{ background: #1e232d; padding: 0.65rem; border-radius: 8px; display: flex; flex-direction: column; gap: 0.4rem; }}
  .item.ref {{ background: #141820; }}
  audio {{ width: 100%; height: 34px; }}
  details {{ margin-top: 0.85rem; }}
  summary {{ cursor: pointer; color: #8ab4f8; font-size: 0.9rem; }}
  .reveal {{ margin: 0.5rem 0 0; font-family: ui-monospace, monospace; font-size: 0.85rem; color: #a3be8c; }}
</style>
</head>
<body>
  <h1>CleanMyMic Studio — Blind Listening</h1>
  <p class="sub">All clips are active-speech loudness matched with a −1 dBTP peak cap. Listen on headphones, judge naturalness + voice preservation + studio clarity, then open Reveal.</p>
  {''.join(cards)}
  <script>
    // Pause other players when starting one for crisp A/B comparison
    document.addEventListener('play', e => {{
      document.querySelectorAll('audio').forEach(a => {{ if (a !== e.target) a.pause(); }});
    }}, true);
  </script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-out", help="optional JSON answer key path")
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--max-sec", type=float, default=14.0)
    ap.add_argument("--trial", action="append", required=True, help="Title|raw=path|ref=path|SysA=path|SysB=path")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    trials_data, key_data = [], {}
    for idx, spec in enumerate(args.trial, 1):
        parts = spec.split("|")
        title = parts[0]
        refs, cands = [], []
        for kv in parts[1:]:
            k, v = kv.split("=", 1)
            if k.lower() in ("raw", "ref", "studio", "reference"):
                refs.append((k, v))
            else:
                cands.append((k, v))
        rng.shuffle(cands)
        letters = [chr(ord("A") + i) for i in range(len(cands))]
        blind = [(f"Clip {L}", load_trim_norm(p, max_sec=args.max_sec)) for L, (_, p) in zip(letters, cands)]
        ref_uris = [(k, load_trim_norm(p, max_sec=args.max_sec)) for k, p in refs]
        reveal = {f"Clip {L}": name for L, (name, _) in zip(letters, cands)}
        trials_data.append({"title": title, "blind": blind, "refs": ref_uris, "reveal": reveal})
        key_data[f"trial_{idx}"] = {"title": title, "mapping": reveal}

    Path(args.out).write_text(build_html(trials_data, key_data), encoding="utf-8")
    if args.key_out:
        Path(args.key_out).write_text(json.dumps(key_data, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} ({len(trials_data)} trials)")


if __name__ == "__main__":
    main()
