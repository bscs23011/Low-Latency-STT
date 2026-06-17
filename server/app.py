"""
app.py — FastAPI server: static web UI + low-latency streaming STT websocket.

"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stt import SAMPLE_RATE, STTEngine
from vad import StreamingSession

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="Whisperwave STT")
engine: STTEngine | None = None


@app.on_event("startup")
def _startup():
    global engine
    engine = STTEngine()
    engine.warmup()
    print("[app] ready — open http://0.0.0.0:8000", flush=True)


@app.get("/api/info")
def info():
    return JSONResponse(
        {
            "model": "parakeet-tdt-0.6b-v2-int8",
            "backend": "sherpa-onnx",
            "sample_rate": SAMPLE_RATE,
            "threads": engine.num_threads if engine else None,
        }
    )


@app.get("/healthz")
def healthz():
    return {"ok": engine is not None}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    assert engine is not None
    session = StreamingSession(engine=engine)
    loop = asyncio.get_running_loop()
    await ws.send_text(
        json.dumps({"type": "ready", "model": "parakeet-tdt-0.6b", "sample_rate": SAMPLE_RATE})
    )
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if (b := msg.get("bytes")) is not None:
                samples = np.frombuffer(b, dtype=np.float32)
                if samples.size == 0:
                    continue
                events = await loop.run_in_executor(None, session.accept, samples)
            elif (t := msg.get("text")) is not None:
                cmd = t.strip().lower()
                if cmd == "stop":
                    events = await loop.run_in_executor(None, session.flush)
                elif cmd == "reset":
                    session = StreamingSession(engine=engine)
                    events = []
                else:
                    events = []
            else:
                continue

            for ev in events:
                await ws.send_text(json.dumps(ev.__dict__))
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        print(f"[ws] error: {e}", flush=True)


# Static UI mounted last so /ws and /api/* take precedence.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
