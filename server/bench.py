"""
bench.py — latency + accuracy benchmark for the parakeet STT engine.

"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stt import SAMPLE_RATE, STTEngine


def make_tone(seconds: float) -> np.ndarray:
    """Speech-like band-limited noise so the decoder does real work."""
    n = int(seconds * SAMPLE_RATE)
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(n).astype(np.float32)
    # crude formant-ish shaping
    for f in (0.02, 0.05, 0.11):
        t = np.arange(n)
        sig += 0.4 * np.sin(2 * np.pi * f * t).astype(np.float32)
    sig *= 0.05
    return sig.astype(np.float32)


def latency_matrix(engine: STTEngine, lengths=(1.0, 1.5, 3.0, 5.0, 10.0), runs=7):
    print("\n=== LATENCY MATRIX ===")
    print(f"{'audio':>7} | {'cold':>8} | {'warm p50':>9} | {'warm p90':>9} | "
          f"{'RTF':>6} | {'ms / 1s':>8}")
    print("-" * 64)
    rows = []
    for sec in lengths:
        audio = make_tone(sec)
        cold = engine.transcribe(audio).proc_ms  # first hit for this size
        warm = [engine.transcribe(audio).proc_ms for _ in range(runs)]
        p50 = statistics.median(warm)
        p90 = sorted(warm)[max(0, int(len(warm) * 0.9) - 1)]
        rtf = p50 / (sec * 1000.0)
        per1s = rtf * 1000.0
        rows.append((sec, cold, p50, p90, rtf, per1s))
        print(f"{sec:>6.1f}s | {cold:>7.0f}ms | {p50:>7.0f}ms | {p90:>7.0f}ms | "
              f"{rtf:>5.3f}x | {per1s:>6.0f}ms")
    avg_per1s = statistics.mean(r[5] for r in rows)
    print("-" * 64)
    print(f"≈ {avg_per1s:.0f} ms to process 1 second of audio (warm, avg)\n")
    return rows


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------
def normalize(s: str) -> list[str]:
    keep = "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in s)
    return keep.split()


def wer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = normalize(ref), normalize(hyp)
    # Levenshtein over word lists
    dp = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, len(h) + 1):
            cur = dp[j]
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[len(h)], len(r)


def accuracy_librispeech(engine: STTEngine, root: Path, limit: int | None):
    print(f"\n=== ACCURACY (WER) on {root} ===")
    refs: dict[str, str] = {}
    for trans in root.rglob("*.trans.txt"):
        for line in trans.read_text().splitlines():
            uid, _, text = line.partition(" ")
            refs[uid] = text
    flacs = sorted(root.rglob("*.flac"))
    if limit:
        flacs = flacs[:limit]
    if not flacs:
        print("No .flac files found.")
        return
    tot_err = tot_words = 0
    tot_audio = tot_proc = 0.0
    for i, fp in enumerate(flacs, 1):
        uid = fp.stem
        if uid not in refs:
            continue
        audio, sr = sf.read(str(fp), dtype="float32")
        if sr != SAMPLE_RATE:
            continue
        res = engine.transcribe(audio)
        err, words = wer(refs[uid], res.text)
        tot_err += err
        tot_words += words
        tot_audio += res.audio_seconds
        tot_proc += res.proc_ms
        if i <= 3:
            print(f"  ref: {refs[uid][:70]}")
            print(f"  hyp: {res.text[:70]}\n")
    if tot_words:
        print(f"utterances : {len(flacs)}")
        print(f"WER        : {100 * tot_err / tot_words:.2f}%  "
              f"({tot_err} edits / {tot_words} words)")
        print(f"accuracy   : {100 * (1 - tot_err / tot_words):.2f}%")
        print(f"avg RTF    : {tot_proc / (tot_audio * 1000 + 1e-9):.3f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    engine = STTEngine()
    engine.warmup()
    latency_matrix(engine)
    if args.librispeech:
        accuracy_librispeech(engine, args.librispeech, args.limit)


if __name__ == "__main__":
    main()
