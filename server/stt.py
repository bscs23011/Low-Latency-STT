"""
stt.py — Parakeet-TDT 0.6B speech-to-text engine (sherpa-onnx backend).

"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sherpa_onnx

SAMPLE_RATE = 16000
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"


@dataclass
class TranscriptResult:
    text: str
    audio_seconds: float
    proc_ms: float

    @property
    def rtf(self) -> float:
        """Real-time factor: <1.0 means faster than real time."""
        return self.proc_ms / (self.audio_seconds * 1000.0 + 1e-9)


class STTEngine:
    def __init__(self, num_threads: int | None = None):
        if num_threads is None:
            num_threads = max(1, (os.cpu_count() or 4) // 2)
        self.num_threads = num_threads
        self._lock = threading.Lock()
        self._recognizer = self._build()

    def _build(self) -> sherpa_onnx.OfflineRecognizer:
        enc = MODEL_DIR / "encoder.int8.onnx"
        dec = MODEL_DIR / "decoder.int8.onnx"
        joi = MODEL_DIR / "joiner.int8.onnx"
        tok = MODEL_DIR / "tokens.txt"
        for f in (enc, dec, joi, tok):
            if not f.exists():
                raise FileNotFoundError(
                    f"Missing model file: {f}\n"
                    "Run scripts/setup.sh (or download the parakeet model into models/)."
                )
        t0 = time.time()
        rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(enc),
            decoder=str(dec),
            joiner=str(joi),
            tokens=str(tok),
            num_threads=self.num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            model_type="nemo_transducer",
            debug=False,
        )
        print(
            f"[stt] parakeet-tdt-0.6b loaded in {time.time() - t0:.1f}s "
            f"(threads={self.num_threads})",
            flush=True,
        )
        return rec

    def transcribe(self, audio: np.ndarray) -> TranscriptResult:
        """Transcribe a mono float32 PCM buffer at 16 kHz."""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        dur = audio.size / SAMPLE_RATE
        if audio.size == 0:
            return TranscriptResult("", 0.0, 0.0)
        t0 = time.time()
        with self._lock:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, audio)
            self._recognizer.decode_stream(stream)
            text = stream.result.text.strip()
        return TranscriptResult(text, dur, (time.time() - t0) * 1000.0)

    def warmup(self) -> None:
        """Force ORT graph allocation so the first real request isn't slow."""
        self.transcribe(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))
        print("[stt] warmup complete", flush=True)
