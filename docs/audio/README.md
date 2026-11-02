# Real-call comparisons

Three 10-second excerpts chosen for stronger background noise from Krisp's real-world voice-isolation benchmark: scripted speech captured in actual offices, call centers, and cars, with no synthetic mixing. Open [index.html](index.html) from a local checkout for the comparison player. GitHub displays the HTML as source.

| Recording | Original | CleanMyMic Base | DeepFilterNet3 | Krisp VI 2.5 HD |
|---|---|---|---|---|
| Office · 176 · 00:59–01:09 | [WAV](work-raw.wav) | [WAV](work-base.wav) | [WAV](work-dfn3.wav) | [WAV](work-krisp.wav) |
| Call center · 030 · 01:25–01:35 | [WAV](call_center-raw.wav) | [WAV](call_center-base.wav) | [WAV](call_center-dfn3.wav) | [WAV](call_center-krisp.wav) |
| Car · 166 · 00:01–00:11 | [WAV](phone_calls-raw.wav) | [WAV](phone_calls-base.wav) | [WAV](phone_calls-dfn3.wav) | [WAV](phone_calls-krisp.wav) |

Keep playback volume the same. Compare speech changes as well as noise suppression. These three excerpts do not establish an overall winner or the best checkpoint. In the earlier controlled mixes, DeepFilterNet3 scored higher than Base on all three intelligibility checks; see the [quality audit](../benchmark-data/demo-quality-20261031.json).

## Method

- Screen 94 original recordings for a higher background level relative to speech, using low- versus high-level 20 ms frame RMS and the speaker annotations. Choose excerpts near speech boundaries before generating model outputs. This is a background-level proxy, not an SNR measurement, a random sample, or a model-quality ranking. These are scripted recordings in real environments, not private customer conversations.
- Process each original from its beginning through the end of the excerpt, so CleanMyMic and DeepFilterNet3 have the preceding audio context. Resample 32 kHz inputs to 48 kHz, use the shared Core ML engine at full strength, append 0.2 seconds of silence to flush, remove the fixed 1,440-sample processing delay, then resample back to 32 kHz and crop. No enrollment or extra cleanup.
- CleanMyMic Base is the exact Hold-10k checkpoint in [the benchmarks](../benchmarks.md). DeepFilterNet3 uses released model-120 weights through our Core ML port, without its optional post-filter. Krisp uses the publisher's matching VI 2.5 HD file, cropped at the same times; it is not a test of the current Krisp Mac app.
- Apply one common playback gain to all four versions of each recording, targeting −20 dBFS for the original's 90th-percentile 20 ms frame RMS, with a shared peak limit of 0.95. This makes quiet recordings easier to hear without changing the relative model levels. There is no per-output normalization or added noise. All 12 files are exactly 10 seconds, 32 kHz mono PCM16, without newly introduced clipping (about 7.7 MB total).
- [sources.json](sources.json) records pinned source URLs, hashes, checkpoint identity, and excerpt locations. The prior synthetic demos remain in Git history; they are not presented as real calls.

## Credits and license

Audio © Krisp Technologies, from [VoiceIsolation-Benchmark-Dataset](https://huggingface.co/datasets/Krisp-AI/VoiceIsolation-Benchmark-Dataset) and [VoiceIsolation-Benchmark-Dataset-Processed](https://huggingface.co/datasets/Krisp-AI/VoiceIsolation-Benchmark-Dataset-Processed), under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Changes: excerpting, resampling, common playback gain, and the labeled model processing. No endorsement is implied.
