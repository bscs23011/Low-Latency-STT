# Whisperwave — low-latency streaming speech-to-text

A self-contained, real-time speech-to-text web app. Speak into any device on
your network; words stream in live with interim (partial) hypotheses that lock
into clean final text. Built for dictating notes and code with minimal latency.

- **Engine:** [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) running
  **NVIDIA Parakeet-TDT 0.6B v2 (int8)** — among the most accurate + fastest
  English ASR models available.
- **Segmentation:** Silero **VAD** decides utterance boundaries; the server
  re-decodes the in-progress utterance for live partials, then emits a clean
  final on end-of-speech.
- **Denoising:** browser-native **noise suppression + echo cancellation + AGC**
  (WebRTC audio pipeline) — zero added latency.
- **Transport:** mic → WebSocket (over TCP) → server. Binds `0.0.0.0`, so any
  device on the LAN can use it.
- **UI:** glassmorphism, animated audio visualizer, live transcript, one-tap
  copy-to-clipboard, live latency/RTF metrics.

## Quick start

```bash
scripts/setup.sh      # venv + deps + model download (self-contained, ~570 MB)
scripts/run.sh        # serves on http://<your-ip>:8000
```

Open the printed URL on this machine or any device on the same Wi-Fi, tap the
mic, and talk. (Mic access on non-localhost devices needs `https://` or a
Chrome "treat-as-secure-origin" flag — see Notes.)

## Layout

```
server/
  app.py     FastAPI: serves the UI + /ws audio websocket
  stt.py     Parakeet recognizer (thread-safe wrapper)
  vad.py     Silero VAD streaming segmenter (partials + finals)
  bench.py   latency + WER benchmark
web/
  index.html / style.css / app.js / worklet.js
models/      parakeet-tdt-0.6b-v2-int8/  +  silero_vad.onnx
scripts/     setup.sh, run.sh
```

## Wire protocol (`/ws`)

```
client -> server : binary  = mono float32 LE PCM @ 16 kHz (mic)
                   text "stop"  -> flush current segment
                   text "reset" -> clear session
server -> client : {"type":"ready"|"partial"|"final","text",
                    "seg","audio_s","proc_ms","rtf"}
```

## Benchmark

```bash
.venv/bin/python server/bench.py                                  # latency
.venv/bin/python server/bench.py --librispeech data/LibriSpeech/test-clean  # + WER
```

## Notes

- **Secure-origin mic:** browsers only grant `getUserMedia` on `localhost` or
  `https`. To use from a phone, either put the server behind HTTPS (e.g. a
  reverse proxy / `caddy`) or, in Chrome on the phone, add the server origin to
  `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.
- **Tuning** (in `server/vad.py`): `MIN_SILENCE_S` controls how fast a sentence
  finalizes; `PARTIAL_INTERVAL_S` controls how often live text updates.
- English-only model. Swap `models/` + `stt.py` for a multilingual whisper
  build if you need other languages.
