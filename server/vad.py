"""
vad.py — streaming voice-activity segmenter with live partial hypotheses.

"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sherpa_onnx

from stt import SAMPLE_RATE, STTEngine, MODEL_DIR

SILERO_VAD = MODEL_DIR.parent / "silero_vad.onnx"

# --- tunables -------------------------------------------------------------
VAD_THRESHOLD = 0.5          # silero speech probability gate
MIN_SILENCE_S = 0.35         # silence this long ends a segment (lower = snappier)
MIN_SPEECH_S = 0.10          # ignore blips shorter than this
MAX_SPEECH_S = 14.0          # force-finalize runaway segments
WINDOW = 512                 # silero processes 512-sample (32 ms) windows
PARTIAL_INTERVAL_S = 0.45    # re-decode the in-progress utterance this often
PREROLL_S = 0.30             # audio kept before speech onset (avoids clipped starts)
# -------------------------------------------------------------------------


def build_vad() -> sherpa_onnx.VoiceActivityDetector:
    if not SILERO_VAD.exists():
        raise FileNotFoundError(
            f"Missing VAD model: {SILERO_VAD}\nRun scripts/setup.sh."
        )
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(SILERO_VAD)
    cfg.silero_vad.threshold = VAD_THRESHOLD
    cfg.silero_vad.min_silence_duration = MIN_SILENCE_S
    cfg.silero_vad.min_speech_duration = MIN_SPEECH_S
    cfg.silero_vad.max_speech_duration = MAX_SPEECH_S
    cfg.silero_vad.window_size = WINDOW
    cfg.sample_rate = SAMPLE_RATE
    cfg.num_threads = 1
    cfg.validate()
    return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)


@dataclass
class STTEvent:
    type: str           # "partial" | "final"
    text: str
    seg: int            # utterance index (stable id for partial->final replace)
    audio_s: float
    proc_ms: float
    rtf: float


@dataclass
class StreamingSession:
    engine: STTEngine
    vad: sherpa_onnx.VoiceActivityDetector = field(default_factory=build_vad)

    def __post_init__(self):
        self._tail = np.empty(0, dtype=np.float32)        # < WINDOW leftover for VAD
        self._preroll = np.empty(0, dtype=np.float32)     # rolling pre-onset buffer
        self._cur: list[np.ndarray] = []                  # current utterance (for partials)
        self._in_speech = False
        self._since_partial = 0
        self._last_partial = ""
        self._seg = 0
        self._preroll_max = int(PREROLL_S * SAMPLE_RATE)
        self._partial_n = int(PARTIAL_INTERVAL_S * SAMPLE_RATE)

    def accept(self, samples: np.ndarray) -> list[STTEvent]:
        """Feed a chunk of mono float32 16 kHz PCM. Returns 0+ events."""
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        events: list[STTEvent] = []

        # Feed VAD in exact 512-sample windows; hold the remainder.
        data = np.concatenate((self._tail, samples)) if self._tail.size else samples
        full = (data.size // WINDOW) * WINDOW
        for i in range(0, full, WINDOW):
            self.vad.accept_waveform(data[i:i + WINDOW])
        self._tail = data[full:].copy()

        speaking = self.vad.is_speech_detected()
        if speaking:
            if not self._in_speech:
                self._in_speech = True
                self._cur = [self._preroll.copy()] if self._preroll.size else []
                self._since_partial = 0
                self._last_partial = ""
            self._cur.append(samples)
            self._since_partial += samples.size
            if self._since_partial >= self._partial_n:
                self._since_partial = 0
                ev = self._decode(np.concatenate(self._cur), "partial")
                if ev and ev.text and ev.text != self._last_partial:
                    self._last_partial = ev.text
                    events.append(ev)
        else:
            self._in_speech = False

        # Drain any completed segments -> finals (clean, trimmed audio).
        while not self.vad.empty():
            seg = self.vad.front
            buf = np.asarray(seg.samples, dtype=np.float32)
            self.vad.pop()
            self._cur = []
            self._last_partial = ""
            self._since_partial = 0
            ev = self._decode(buf, "final")
            if ev:
                events.append(ev)

        # Update rolling preroll with the freshest audio.
        self._preroll = np.concatenate((self._preroll, samples))[-self._preroll_max:]
        return events

    def flush(self) -> list[STTEvent]:
        """Force-close any in-progress segment (call on stop)."""
        if self._tail.size:
            pad = np.zeros(WINDOW - self._tail.size, dtype=np.float32)
            self.vad.accept_waveform(np.concatenate((self._tail, pad)))
            self._tail = np.empty(0, dtype=np.float32)
        self.vad.flush()
        events: list[STTEvent] = []
        while not self.vad.empty():
            seg = self.vad.front
            buf = np.asarray(seg.samples, dtype=np.float32)
            self.vad.pop()
            ev = self._decode(buf, "final")
            if ev:
                events.append(ev)
        self.vad.reset()
        self._in_speech = False
        self._cur = []
        return events

    def _decode(self, buf: np.ndarray, kind: str) -> STTEvent | None:
        # A partial carries the id its final will commit under, so the UI can
        # replace the live line in place when the final arrives.
        if kind == "final":
            self._seg += 1
            seg_id = self._seg
        else:
            seg_id = self._seg + 1
        res = self.engine.transcribe(buf)
        if kind == "final" and not res.text:
            return None
        return STTEvent(
            type=kind,
            text=res.text,
            seg=seg_id,
            audio_s=round(res.audio_seconds, 3),
            proc_ms=round(res.proc_ms, 1),
            rtf=round(res.rtf, 3),
        )
