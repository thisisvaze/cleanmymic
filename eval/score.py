"""Score a checkpoint: does it keep the words and the voice, and does it sound better?

    uv run --group eval eval/score.py <checkpoint.pt> [--data ~/data] [--n 40] [--wet 1.0]
    uv run --group eval eval/score.py --dump <dir>     # write every input to <dir>/<set>/<name>.wav, then exit
    uv run --group eval eval/score.py <dir>            # score outputs another system made from a dump (DeepFilterNet3)

--wet is the app's strength slider: output = wet * model + (1 - wet) * raw mic. --gain DB plays the input to the
model that much louder (and turns the output back down): does a model depend on how loud the talker is?

Sets (whatever exists): eval/recordings (our takes), <data>/eval/{daps,drvctk,dns5} (eval/get_sets.py), Krisp's
voice-isolation benchmark in <data>/eval/krisp (real calls on 24 laptops / 13 headsets / 18 phones, a second person
talking over some; Krisp-AI/VoiceIsolation-Benchmark-Dataset), real conversations with others talking in
<data>/eval/chime6 (CHiME-6 dinner parties in homes, each guest's in-ear mics; openslr.org/150, CC BY-SA 4.0) and
<data>/eval/easycom (EasyCom: restaurant noise at 71 dB, close-talk headset mics and the glasses wearer's ear mics;
CC BY-NC 4.0), and no-speech clips cut from <data>/packed/noise_val. A conversation item is up to 25 s around one
wearer's talk (talks()).
--sets limits a run to some of them (e.g. to score Krisp's own outputs, which exist only for theirs).
Writes float WAV outputs + scores.csv to eval/outputs/<run>-<step>/.

Per item, by what we know about it:
  ref     paired studio audio: words vs Whisper(studio), voice similarity vs studio, in and out
  script  our reading takes: words vs sentences.txt
  text    Krisp's set and the conversations: words vs the transcript of the main talker (the wearer); other_db is how
          much of the others is left where only they talk (out vs in; a model that can't know who the user is keeps them)
  free    no reference: output words vs the input's words, voice similarity out vs in; DNS5 has a clip of each
          talker alone (enrol/), so there voice similarity is to that, in and out
  silence no speech at all: the output must not gain words
Quality goals: words and voice must not get worse than the raw input; silence must stay wordless.

Noise under speech (phase inversion): studio speech + noise 20 dB below it, processed as s+n and as s-n. Half the
difference of the outputs is what is left of the noise, half the sum is the speech, so per band we get the noise
removed while talking and in pauses, and what happened to the voice. v3 removed ~44 dB in pauses but ~3.5 dB while
talking. The split assumes s+n and s-n are processed alike; it gets unreliable where processing is strongly
signal-dependent (it read DeepFilterNet3's 2-8 kHz band as negative).
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from audiobox_aesthetics.infer import initialize_predictor
from jiwer import process_words, wer
from scipy.signal import istft, resample_poly, stft
from transformers import AutoFeatureExtractor, WavLMForXVector, logging, pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "train"))
from model import SR  # noqa: E402
from model import load as load_model  # noqa: E402
from noise import noise_db  # noqa: E402

FREE_TAKES = ("free", "expressive", "other-language", "long")  # unscripted: compare against the input transcript
DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
SECONDS = 30  # long files (DAPS scripts, DNS calls) are cut to their first 30 s
ENROL = 15  # s of enrollment audio, for enrolled-voice models (DNS5's own clips are 10 s)


def load(path, seconds=SECONDS, start=0.0, channels=None):
    info = sf.info(path)
    x, sr = sf.read(path, start=int(start * info.samplerate), frames=int(seconds * info.samplerate), dtype="float32",
                    always_2d=True)
    x = x[:, channels].mean(1) if channels else x.mean(1)
    return x if sr == SR else resample_poly(x, SR, sr).astype(np.float32)


def tone_db(x, ref):
    """How far x's tone is from the studio reference's (the Studio Voice score): mean absolute difference of their
    long-term spectra in third-octave bands from 100 Hz to 8 kHz, over each one's loud frames (within 20 dB of its
    loudest: speech, not pauses), after matching overall level. 0 = the studio's tone."""
    edges = 100 * 2 ** (np.arange(20) / 3)

    def bands(a):
        f, _, S = stft(a, SR, nperseg=2048, noverlap=1536)
        P = np.abs(S) ** 2
        e = P.sum(0)
        P = P[:, e >= e.max() / 100].mean(1)
        return 10 * np.log10([P[(f >= lo) & (f < hi)].sum() + 1e-20 for lo, hi in zip(edges[:-1], edges[1:])])

    d = bands(x) - bands(ref)
    return float(np.abs(d - d.mean()).mean())


def secs(t):
    """'0:01:23.32' -> 83.32"""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def talks(utts, wearer, seconds=25):
    """Stretches of a long recording where its wearer talks, for a 'text' item: from one of their utterances to the last
    that ends within `seconds`, cut between their utterances. utts: (speaker, start, end, words). Yields (start, end,
    their words, other): other = where only the others talk, relative to start."""
    mine = sorted((s, e, w) for spk, s, e, w in utts if spk == wearer)
    i = 0
    while i < len(mine):
        j = i
        while j + 1 < len(mine) and mine[j + 1][1] <= mine[i][0] + seconds:
            j += 1
        start = min(mine[i][0], max(mine[i][0] - 0.3, mine[i - 1][1] if i else 0))
        end = max(mine[j][1], min(mine[j][1] + 0.3, mine[j + 1][0] if j + 1 < len(mine) else mine[j][1] + 0.3))
        text = " ".join(w for _, _, w in mine[i:j + 1])
        if len(text.split()) >= 15:
            t = np.arange(start, end, 0.01)
            span = lambda spans: np.any([(t >= a) & (t < b) for a, b in spans if b > start and a < end] + [t < 0], 0)
            only = span([(s, e) for spk, s, e, _ in utts if spk != wearer]) & ~span([(s - 0.2, e + 0.2) for s, e, _ in mine])
            edges = np.flatnonzero(np.diff(np.r_[0, only.astype(int), 0]))  # run starts and ends, alternating
            other = [(a / 100, b / 100) for a, b in zip(edges[::2], edges[1::2]) if b - a >= 30]
            yield start, end, text, other
        i = j + 1


def solo(utts, wearer):
    """The wearer's utterances nobody else talks over: (start, end)."""
    others = sorted((s, e) for spk, s, e, _ in utts if spk != wearer)
    starts = np.array([s for s, _ in others])
    ends = np.maximum.accumulate([e for _, e in others]) if others else np.array([])
    out = []
    for spk, s, e, _ in utts:
        k = np.searchsorted(starts, e)  # others starting before this ends; overlap if the latest end among them is after s
        if spk == wearer and (k == 0 or ends[k - 1] <= s):
            out.append((s, e))
    return out


def enrolment(spans, wav, a, b):
    """Up to ENROL s of the wearer alone, from their longest solo utterances [(file, start, end, channels)], none within
    5 s of the item ([a, b] in wav)."""
    take, total = [], 0
    for f, s, e, ch in sorted(spans, key=lambda t: t[1] - t[2]):
        if f != wav or e < a - 5 or s > b + 5:
            take.append(load(f, e - s, s, ch))
            total += e - s
            if total >= ENROL:
                break
    return np.concatenate(take) if take else None


def items(data, n):
    """Yields (set, name, input, reference audio or None, kind)."""
    rng = random.Random(0)
    for p in sorted((HERE / "recordings").glob("*.wav")):
        take = p.stem.rsplit("_", 1)[0]
        kind = "silence" if take == "no-speech" else "free" if take.startswith(FREE_TAKES) else "script"
        yield "ours", p.stem, load(p, 600), None, kind
    daps = data / "eval" / "daps"
    devices = [p for p in sorted(daps.rglob("*.wav")) if "ipad" in p.parent.name or "iphone" in p.parent.name]
    for p in rng.sample(devices, min(n, len(devices))):
        speaker, script_id = p.stem.split("_")[:2]
        ref = next(daps.rglob(f"{speaker}_{script_id}_clean.wav"), None)
        if ref:
            yield "daps", p.stem, load(p), load(ref), "ref"
    dr_root = data / "eval" / "drvctk"
    dr = sorted(dr_root.rglob("device-recorded_testset*/*.wav"))
    for p in rng.sample(dr, min(n, len(dr))):
        ref = next(dr_root.rglob(f"clean_testset*/{p.name}"), None)
        if ref:
            yield "drvctk", p.stem, load(p), load(ref), "ref"
    for sub, set_name in (("work", "krisp-work"), ("call_center", "krisp-call"), ("phone_calls", "krisp-phone")):
        meta = data / "eval" / "krisp" / sub / "metadata.jsonl"
        rows = [json.loads(line) for line in open(meta)] if meta.exists() else []
        for r in random.Random(sub).sample(rows, min(n, len(rows))):  # own RNG: the other sets keep their samples
            other = [(g["start"], g["end"]) for g in r["segments"] if g["label"] == "secondary"]  # also: primary, mix, noise
            # enrollment: where the talker is alone in another of their calls
            same = sorted((q for q in rows if q["speaker_id"] == r["speaker_id"] and q["id"] != r["id"]), key=lambda q: q["id"])
            alone = [load(meta.parent / same[0]["file_name"], g["end"] - g["start"], g["start"])
                     for g in (same[0]["segments"] if same else []) if g["label"] == "primary"]
            ref = {"text": r["full_transcript"], "other": other, "enrol": np.concatenate(alone)[:ENROL * SR] if alone else None}
            yield set_name, r["id"], load(meta.parent / r["file_name"], 600), ref, "text"
    dns = sorted((data / "eval" / "dns5").rglob("noisy/*.wav"))
    for p in random.Random("dns5").sample(dns, min(n, len(dns))):  # each with its talker alone (enrol/, same name)
        yield "dns5", p.stem, load(p), {"enrol": load(p.parent.parent / "enrol" / p.name)}, "free"
    chime = data / "eval" / "chime6"  # dinner parties in homes, each guest's in-ear mics (dev and eval sessions)
    talk_items, alone = [], {}
    for js in sorted(chime.rglob("transcriptions/*/S*.json")):
        if js.parent.name != "train":
            utts = [(u["speaker"], secs(u["start_time"]), secs(u["end_time"]), u["words"]) for u in json.load(open(js))]
            for wav in sorted(chime.rglob(f"audio/*/{js.stem}_P*.wav")):
                talk_items += [("chime6", wav, *t) for t in talks(utts, wav.stem.split("_")[1])]
                alone[wav] = [(wav, s, e, None) for s, e in solo(utts, wav.stem.split("_")[1])]
    for set_name, wav, a, b, text, other in random.Random("chime6").sample(talk_items, min(n, len(talk_items))):
        ref = {"text": text, "other": other, "enrol": enrolment(alone[wav], wav, a, b)}
        yield set_name, f"{wav.stem}_{a:.0f}", load(wav, b - a, a), ref, "text"
    # EasyCom: per session, minute files of each close-talk mic, and the glasses (ear mics: channels 5-6) worn by the
    # one talker without a close mic; transcripts in video frames (20 fps). Enrollment from the session's other minutes.
    easy = data / "eval" / "easycom" / "Main"
    picks = {"easycom-close": [], "easycom-ear": []}
    for session in sorted(easy.glob("Speech_Transcriptions/Session_*")):
        minutes = {js.stem: [(u["Participant_ID"], u["Start_Frame"] / 20, u["End_Frame"] / 20, u["Transcription"])
                             for u in json.loads(js.read_text(errors="replace"))] for js in sorted(session.glob("*.json"))}
        mics = sorted({int(w.stem.rsplit("_", 1)[1]) for w in (easy / "Close_Microphone_Audio" / session.name).glob("*.wav")})
        said = Counter(p for utts in minutes.values() for p, *_ in utts)
        glasses = max((p for p in said if p not in mics), key=said.get, default=None)
        for pid, set_name in [(p, "easycom-close") for p in mics] + [(glasses, "easycom-ear")] * (glasses is not None):
            spans, found = [], []
            for minute, utts in minutes.items():
                wav, ch = ((easy / "Close_Microphone_Audio" / session.name / f"{minute}_Participant_ID_{pid}.wav", None)
                           if set_name == "easycom-close" else (easy / "Glasses_Microphone_Array_Audio" / session.name / f"{minute}.wav", [4, 5]))
                if wav.exists():
                    spans += [(wav, s, e, ch) for s, e in solo(utts, pid)]
                    found += [(f"{session.name}_{minute}_{pid}", wav, ch, *t) for t in talks(utts, pid)]
            picks[set_name] += [(*f, spans) for f in found]
    for set_name, found in picks.items():
        for name, wav, ch, a, b, text, other, spans in random.Random(set_name).sample(found, min(n, len(found))):
            ref = {"text": text, "other": other, "enrol": enrolment(spans, wav, a, b)}
            yield set_name, f"{name}_{a:.0f}", load(wav, b - a, a, ch), ref, "text"
    if (data / "packed" / "noise_val.npy").exists():
        noise = np.load(data / "packed" / "noise_val.npy", mmap_mode="r")
        idx = np.load(data / "packed" / "noise_val_idx.npy")
        for i in rng.sample(range(len(idx) - 1), min(n // 2, len(idx) - 1)):
            clip = noise[idx[i]:min(idx[i + 1], idx[i] + 10 * SR)].astype(np.float32) / 32768
            yield "silence", f"noise{i}", clip, None, "silence"


BANDS = (40, 150), (150, 2000), (2000, 8000), (8000, 24000)


def pi_items(data, k=6, seconds=20):
    """Yields (name, speech, noise) for the phase-inversion test: DAPS studio speech; noise from noise_val clips, or
    steady noise with the spectrum of our desk takes' pauses (what the user hears under their voice)."""
    rng = random.Random(1)
    cleans = sorted(p for p in (data / "eval" / "daps").rglob("*_clean.wav") if not p.name.startswith("._"))
    noise = np.load(data / "packed" / "noise_val.npy", mmap_mode="r")
    idx = np.load(data / "packed" / "noise_val_idx.npy")
    desks = sorted((HERE / "recordings").glob("free-desk_*.wav"))
    for j, p in enumerate(rng.sample(cleans, min(k, len(cleans)))):
        s = load(p, 10 + seconds)[10 * SR:]
        if j % 2 and desks:
            raw = load(desks[j // 2 % len(desks)], 60)
            P = np.abs(stft(raw, SR, nperseg=960, noverlap=480)[2]) ** 2
            e = P.sum(0)
            psd = P[:, e <= np.percentile(e, 10)].mean(1, keepdims=True)
            g = np.random.default_rng(j)
            S = stft(s, SR, nperseg=960, noverlap=480)[2]
            n = istft(np.sqrt(psd) * (g.standard_normal(S.shape) + 1j * g.standard_normal(S.shape)) / 2 ** 0.5,
                      SR, nperseg=960, noverlap=480)[1][: len(s)]
            kind = "desk"
        else:
            i = rng.randrange(len(idx) - 1)
            clip = noise[idx[i]:idx[i + 1]].astype(np.float32) / 32768
            n = np.resize(clip, len(s))
            kind = "noise"
        for _ in range(4):  # noise_db(s + n) depends on n; settle on 20 dB below the voice
            n = n * 10 ** ((-20 - noise_db(s + n, SR)[0]) / 20)
        yield f"{kind}{j}", s.astype(np.float32), n.astype(np.float32)


def phase_inversion(s, n, y_plus, y_minus):
    """Per band: noise removed under speech, noise removed in pauses, voice gain (dB), from the two outputs."""
    P = lambda a: np.abs(stft(a, SR, nperseg=960, noverlap=480)[2]) ** 2
    m = min(len(s), len(y_plus), len(y_minus))
    S, N = P(s[:m]), P(n[:m])
    No, So = P((y_plus[:m] - y_minus[:m]) / 2), P((y_plus[:m] + y_minus[:m]) / 2)
    f = np.fft.rfftfreq(960, 1 / SR)
    e = 10 * np.log10(S.sum(0) + 1e-12)
    talk, pause = e >= np.percentile(e, 50), e <= np.percentile(e, 15)
    out = {}
    for lo, hi in BANDS:
        b = (f >= lo) & (f < hi)
        db = lambda A, B, t: 10 * np.log10(A[b][:, t].sum() / (B[b][:, t].sum() + 1e-20) + 1e-20)
        out |= {f"talk_{lo}": db(N, No, talk), f"pause_{lo}": db(N, No, pause), f"voice_{lo}": -db(S, So, talk)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("--dump", help="write the inputs here and exit")
    ap.add_argument("--data", default="~/data")
    ap.add_argument("--n", type=int, default=40, help="items per public set")
    ap.add_argument("--wet", type=float, default=1.0)
    ap.add_argument("--gain", type=float, default=0.0)
    ap.add_argument("--no-voice", action="store_true", help="an enrolled-voice model as if nobody had enrolled")
    ap.add_argument("--sets", help="comma-separated: only these sets (and no phase-inversion test)")
    args = ap.parse_args()
    data, n = Path(args.data).expanduser(), args.n
    only = set(args.sets.split(",")) if args.sets else None
    if args.dump:
        dump = Path(args.dump).expanduser()
        for set_name, name, x, _, _ in items(data, n):
            if only and set_name not in only:
                continue
            (dump / set_name).mkdir(parents=True, exist_ok=True)
            sf.write(dump / set_name / f"{name}.wav", x, SR)
        (dump / "pi").mkdir(exist_ok=True)
        for name, s_, n_ in ([] if only else pi_items(data)):
            sf.write(dump / "pi" / f"{name}_plus.wav", s_ + n_, SR)
            sf.write(dump / "pi" / f"{name}_minus.wav", s_ - n_, SR)
        sys.exit(0)
    ckpt = Path(args.checkpoint).expanduser()
    wet_tag = (("" if args.wet == 1 else f"-wet{args.wet:g}") + (f"-gain{args.gain:g}" if args.gain else "")
               + ("-novoice" if args.no_voice else ""))
    g = 10 ** (args.gain / 20)
    if ckpt.is_dir():  # outputs another system made from --dump inputs, same names
        out_dir = HERE / "outputs" / (ckpt.name + wet_tag)

        def enhance(set_name, name, x, ref):
            y = load(ckpt / set_name / f"{name}.wav", 600)[: len(x)]
            return np.pad(y, (0, len(x) - len(y)))
    else:
        model, state = load_model(ckpt)
        model.eval()
        out_dir = HERE / "outputs" / (f"{ckpt.parent.name}-{state['step']}" + wet_tag)

        def enhance(set_name, name, x, ref):
            inputs = [torch.from_numpy(x * g)[None]]
            if getattr(model, "spk", 0):  # an enrolled-voice model: the item's enrollment, or nobody (zeros)
                enrol = ref.get("enrol") if isinstance(ref, dict) and not args.no_voice else None
                inputs.append(embed(enrol).cpu()[None] if enrol is not None else torch.zeros(1, model.spk))
            with torch.no_grad():
                return model(*inputs)[0].numpy() / g

    logging.set_verbosity_error()  # Whisper warns about chunked long-form on every call
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device=DEV,
                   torch_dtype=torch.float16 if DEV != "cpu" else torch.float32, ignore_warning=True)
    sv_fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(DEV).eval()
    aesthetics = initialize_predictor()
    norm = BasicTextNormalizer()
    script = norm(" ".join((HERE / "sentences.txt").read_text().split()))

    def words(x):
        return norm(asr(resample_poly(x, 1, 3).astype(np.float32), chunk_length_s=30, batch_size=8)["text"])

    def embed(x):
        inp = sv_fe(resample_poly(x[: 60 * SR], 1, 3), sampling_rate=16000, return_tensors="pt").to(DEV)
        with torch.no_grad():
            return torch.nn.functional.normalize(sv(**inp).embeddings, dim=-1)[0]

    def pq(x):  # in-memory audio; audiobox's file loader needs torchcodec
        return aesthetics.forward([{"path": torch.from_numpy(x)[None], "sample_rate": SR}])[0]["PQ"]

    rows = []
    for set_name, name, x, ref, kind in items(data, n):
        if only and set_name not in only:
            continue
        y = args.wet * enhance(set_name, name, x, ref) + (1 - args.wet) * x
        (out_dir / set_name).mkdir(parents=True, exist_ok=True)
        sf.write(out_dir / set_name / f"{name}.wav", y, SR, subtype="FLOAT")  # preserve quiet residuals for later analysis
        w_in, w_out = words(x), words(y)
        row = {"set": set_name, "file": name, "pq_in": pq(x), "pq_out": pq(y)}
        if kind == "silence":  # no voice to measure against: report how much quieter the noise got
            row["noise_out"] = 10 * np.log10(np.mean(y ** 2) / np.mean(x ** 2) + 1e-12)
            row |= {"words_in": len(w_in.split()), "words_out": len(w_out.split())}
            row["gate"] = "PASS" if row["words_out"] <= row["words_in"] else "FAIL"
        else:
            row |= {"noise_in": noise_db(x, SR)[0], "noise_out": noise_db(y, SR)[0]}
            if kind == "ref":
                row |= {"noise_ref": noise_db(ref, SR)[0], "tone_in": tone_db(x, ref), "tone_out": tone_db(y, ref)}
        if kind == "free":
            row["word_change"] = wer(w_in, w_out) if w_in.strip() else float("nan")
            if ref:  # DNS5: how much it sounds like the talker alone, in and out (others left in pull it down)
                e_ref = embed(ref["enrol"])
                row |= {"spk_in": float(embed(x) @ e_ref), "spk_out": float(embed(y) @ e_ref)}
            else:
                row["spk_out"] = float(embed(x) @ embed(y))
        elif kind != "silence":
            text = script if kind == "script" else norm(ref["text"]) if kind == "text" else words(ref)
            if kind == "text" and ref["other"]:
                m = np.concatenate([np.arange(int(a * SR), min(int(b * SR), len(x))) for a, b in ref["other"]])
                row["other_db"] = 10 * np.log10(np.mean(y[m] ** 2) / (np.mean(x[m] ** 2) + 1e-12) + 1e-12)
            row |= {"wer_in": wer(text, w_in), "wer_out": wer(text, w_out)}
            row |= {"text_ref": text, "text_in": w_in, "text_out": w_out, "words_ref": len(text.split())}
            for tag, hyp in (("in", w_in), ("out", w_out)):  # deletions point to over-suppression, insertions to
                o = process_words(text, hyp)                   # artifacts or Whisper writing text over silence
                row |= {f"del_{tag}": o.deletions / max(1, len(text.split())), f"ins_{tag}": o.insertions / max(1, len(text.split()))}
                row |= {f"substitutions_{tag}": o.substitutions, f"deletions_{tag}": o.deletions,
                        f"insertions_{tag}": o.insertions}
            if kind == "ref":
                e_ref = embed(ref)
                row |= {"spk_in": float(embed(x) @ e_ref), "spk_out": float(embed(y) @ e_ref)}
            ok = row["wer_out"] <= row["wer_in"] + 0.02 and row.get("spk_out", 1) >= row.get("spk_in", 0) - 0.02
            row["gate"] = "PASS" if ok else "FAIL"
        rows.append(row)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)

    pi_rows = []
    for name, s_, n_ in ([] if only else pi_items(data)):
        y1 = args.wet * enhance("pi", f"{name}_plus", s_ + n_, None) + (1 - args.wet) * (s_ + n_)
        y2 = args.wet * enhance("pi", f"{name}_minus", s_ - n_, None) + (1 - args.wet) * (s_ - n_)
        pi_rows.append({"name": name} | phase_inversion(s_, n_, y1, y2))
        print({k: round(v, 1) if isinstance(v, float) else v for k, v in pi_rows[-1].items()}, flush=True)

    cols = ["set", "file", "wer_in", "wer_out", "del_in", "del_out", "ins_in", "ins_out", "word_change", "words_in",
            "words_out", "spk_in", "spk_out", "pq_in", "pq_out", "noise_in", "noise_out", "noise_ref", "tone_in", "tone_out",
            "other_db", "gate", "text_ref", "text_in", "text_out", "words_ref",
            "substitutions_in", "deletions_in", "insertions_in", "substitutions_out", "deletions_out", "insertions_out"]
    with open(out_dir / ("scores.csv" if not only else f"scores-{'+'.join(sorted(only))}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows([dict.fromkeys(cols, "") | r for r in rows])

    def mean(rs, k):
        vals = [r[k] for r in rs if k in r]
        return np.nanmean(vals) if vals else float("nan")

    print(f"\n{'set':8} {'n':>3}  {'WER in→out':>13}  {'deleted':>11}  {'inserted':>11}  {'voice in→out':>13}"
          f"  {'quality in→out':>14}  {'noise dB in→out (studio)':>24}  {'tone dB in→out':>14}  {'gate fails':>10}")
    for s in dict.fromkeys(r["set"] for r in rows):
        rs = [r for r in rows if r["set"] == s]
        fails = sum(r.get("gate") == "FAIL" for r in rs)
        print(f"{s:8} {len(rs):3}  {mean(rs, 'wer_in'):6.3f}→{mean(rs, 'wer_out'):6.3f}"
              f"  {mean(rs, 'del_in'):5.3f}→{mean(rs, 'del_out'):5.3f}  {mean(rs, 'ins_in'):5.3f}→{mean(rs, 'ins_out'):5.3f}"
              f"  {mean(rs, 'spk_in'):6.3f}→{mean(rs, 'spk_out'):6.3f}"
              f"  {mean(rs, 'pq_in'):6.2f}→{mean(rs, 'pq_out'):6.2f}"
              f"  {mean(rs, 'noise_in'):6.1f}→{mean(rs, 'noise_out'):6.1f} ({mean(rs, 'noise_ref'):6.1f})"
              f"  {mean(rs, 'tone_in'):6.2f}→{mean(rs, 'tone_out'):6.2f}  {fails:10}")
    for s in dict.fromkeys(r["set"] for r in rows if "other_db" in r):
        print(f"{s}: the other person, out vs in where only they talk: {mean([r for r in rows if r['set'] == s], 'other_db'):+.1f} dB")
    print(f"\nnoise removed (dB) while talking / in pauses, and voice gain, per band; {len(pi_rows)} mixes, noise 20 dB down")
    print("set      " + "".join(f"{f'{lo}-{hi} Hz':>22}" for lo, hi in BANDS))
    for kind in ("desk", "noise"):
        rs = [r for r in pi_rows if r["name"].startswith(kind)]
        if rs:
            print(f"pi-{kind:5}" + "".join(f"{mean(rs, f'talk_{lo}'):7.1f} /{mean(rs, f'pause_{lo}'):5.1f} /{mean(rs, f'voice_{lo}'):+5.1f}"
                                     for lo, _ in BANDS))
    print(f"outputs: {out_dir}")
