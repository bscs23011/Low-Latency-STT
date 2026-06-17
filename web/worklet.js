// worklet.js — AudioWorklet that captures mic frames off the audio thread.
// Buffers ~85 ms of audio at the context sample rate, then ships it to the
// main thread which resamples to 16 kHz and forwards to the server.

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = [];
    this._count = 0;
    this._target = 2048; // samples per post (~43 ms @ 48 kHz)
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      const ch = input[0];
      this._buf.push(new Float32Array(ch));
      this._count += ch.length;
      if (this._count >= this._target) {
        const out = new Float32Array(this._count);
        let o = 0;
        for (const b of this._buf) {
          out.set(b, o);
          o += b.length;
        }
        this.port.postMessage(out, [out.buffer]);
        this._buf = [];
        this._count = 0;
      }
    }
    return true;
  }
}

registerProcessor("capture", CaptureProcessor);
