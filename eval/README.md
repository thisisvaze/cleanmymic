# Test recordings

Record each take with `eval/record.sh <take-name>`. It captures the MacBook Pro and Studio Display XDR mics at the same time. Files go to `eval/recordings/`, which is kept out of git.

Close Zoom and Teams before recording. (Mic Mode in Control Center doesn't apply: the recorder captures the raw mic.)

Read the sentences in `sentences.txt` (Harvard sentences, lists 1–2) at a normal pace. About a minute per take is plenty. Easiest: open `takes.textream` in [Textream](https://github.com/f/textream), which shows one page per take with its record command.

| Take name | What to do |
|---|---|
| `quiet` | Normal seat, quiet room. |
| `far` | Sit or stand 1.5–2 m from the Mac. |
| `noise-fan` | Fan, AC or a loud laptop fan running nearby. |
| `noise-tv` | TV or music playing at a normal volume while you read. |
| `typing` | Type on the keyboard while you read. |
| `echo-<room>` | Bare or echoey room (kitchen, bathroom). MacBook only: `eval/record.sh echo-kitchen "MacBook Pro Microphone"`. |
| `expressive` | Laugh, sigh, breathe audibly, whisper a line, then say a line loudly. |
| `other-language` | 30 s in any other language you speak (skip if none). |
| `no-speech` | 30 s of TV or music only, with nobody talking. The model must output no words here. |
| `long` | Optional: 10+ minutes of natural talking, e.g. one side of a call. |

## Core ML processing time

With Base available locally, run the shared app processor on existing 48 kHz mono files without opening a microphone:

```sh
swiftc -O eval/latency.swift mac/Sources/CleanMyMic/Enhancer.swift -o /tmp/cleanmymic-latency
/tmp/cleanmymic-latency mac/build/CleanMyMic.app/Contents/Resources/Hold-10k.mlmodelc /path/to/input.wav
```

The JSON reports per-frame mean/percentiles, deadline overruns, and model initialization separately. File IO and device/call routing are excluded. This is an unpaced processing benchmark, not measured call latency or a quality test. See [recorded results and limits](../docs/benchmarks.md#mac-implementation-check).
