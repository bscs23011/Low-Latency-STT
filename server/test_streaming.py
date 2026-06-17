"""
test_streaming.py — offline end-to-end check of the VAD streaming pipeline.

"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stt import SAMPLE_RATE, STTEngine
from vad import StreamingSession


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: test_streaming.py <audio file>")
    audio, sr = sf.read(sys.argv[1], dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        sys.exit(f"expected 16 kHz audio, got {sr}")

    engine = STTEngine()
    engine.warmup()
    session = StreamingSession(engine=engine)

    chunk = 683  # ~43 ms @ 16 kHz, like the worklet's 2048 @ 48 kHz
    print(f"\nstreaming {audio.size / SAMPLE_RATE:.2f}s in {chunk/SAMPLE_RATE*1000:.0f}ms chunks\n")
    t0 = time.time()
    n_part = n_final = 0
    for i in range(0, audio.size, chunk):
        for ev in session.accept(audio[i:i + chunk]):
            tag = "PARTIAL" if ev.type == "partial" else "FINAL  "
            if ev.type == "partial":
                n_part += 1
            else:
                n_final += 1
            print(f"[{tag} #{ev.seg}] ({ev.proc_ms:.0f}ms, rtf={ev.rtf:.2f}) {ev.text}")
    for ev in session.flush():
        n_final += 1
        print(f"[FINAL   #{ev.seg}] ({ev.proc_ms:.0f}ms, rtf={ev.rtf:.2f}) {ev.text}")

    wall = time.time() - t0
    print(f"\n{n_part} partials, {n_final} finals in {wall:.2f}s wall "
          f"(audio was {audio.size/SAMPLE_RATE:.2f}s)")


if __name__ == "__main__":
    main()
