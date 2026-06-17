// app.js — Whisperwave client. Mic capture -> 16 kHz float32 -> WebSocket ->
// live partial/final transcript, animated visualizer, metrics, copy-to-clipboard.

const TARGET_SR = 16000;

const els = {
  status: document.getElementById("statusText"),
  dot: document.getElementById("statusDot"),
  mic: document.getElementById("micBtn"),
  hint: document.getElementById("hint"),
  out: document.getElementById("output"),
  copy: document.getElementById("copyBtn"),
  clear: document.getElementById("clearBtn"),
  viz: document.getElementById("viz"),
  mLatency: document.getElementById("mLatency"),
  mRtf: document.getElementById("mRtf"),
  mPer1s: document.getElementById("mPer1s"),
  mWords: document.getElementById("mWords"),
  toast: document.getElementById("toast"),
};

let ws = null;
let recording = false;
let audioCtx = null;
let stream = null;
let workletNode = null;
let analyser = null;
let sourceNode = null;
let muteGain = null;
let finals = [];          // committed sentences
let partialText = "";     // live, not-yet-committed text

// ----------------------------------------------------------------------------
// WebSocket
// ----------------------------------------------------------------------------
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => setStatus("idle", "Ready");
  ws.onclose = () => {
    setStatus("error", "Disconnected");
    if (recording) stopRecording();
    setTimeout(connect, 1500);
  };
  ws.onerror = () => setStatus("error", "Connection error");
  ws.onmessage = (e) => {
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }
    if (msg.type === "ready") {
      setStatus("idle", "Ready");
    } else if (msg.type === "partial") {
      partialText = msg.text;
      render();
    } else if (msg.type === "final") {
      if (msg.text) finals.push(msg.text);
      partialText = "";
      render();
      updateMetrics(msg);
    }
  };
}

function setStatus(state, text) {
  els.status.textContent = text;
  els.dot.className = "dot" + (state === "live" ? " live" : state === "error" ? " error" : "");
}

// ----------------------------------------------------------------------------
// Recording
// ----------------------------------------------------------------------------
async function startRecording() {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    toast("Not connected to server");
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,   // browser-native AEC
        noiseSuppression: true,   // browser-native denoise (RNNoise-class)
        autoGainControl: true,    // browser-native AGC
      },
    });
  } catch (err) {
    toast("Microphone permission denied");
    return;
  }

  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  await audioCtx.resume();
  const srcRate = audioCtx.sampleRate;

  await audioCtx.audioWorklet.addModule("worklet.js");
  sourceNode = audioCtx.createMediaStreamSource(stream);

  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  analyser.smoothingTimeConstant = 0.78;
  sourceNode.connect(analyser);

  workletNode = new AudioWorkletNode(audioCtx, "capture");
  sourceNode.connect(workletNode);
  // A worklet only runs when its output reaches the destination; route it
  // through a silent gain so we capture without playing the mic back.
  muteGain = audioCtx.createGain();
  muteGain.gain.value = 0;
  workletNode.connect(muteGain).connect(audioCtx.destination);
  workletNode.port.onmessage = (e) => {
    if (!recording || !ws || ws.readyState !== WebSocket.OPEN) return;
    const down = resampleTo16k(e.data, srcRate);
    if (down.length) ws.send(down.buffer);
  };

  recording = true;
  els.mic.classList.add("recording");
  els.mic.setAttribute("aria-label", "Stop transcription");
  els.hint.textContent = "Listening… tap to stop";
  setStatus("live", "Listening");
}

function stopRecording() {
  recording = false;
  els.mic.classList.remove("recording");
  els.mic.setAttribute("aria-label", "Start transcription");
  els.hint.textContent = "Tap to transcribe";
  setStatus("idle", "Ready");

  try { if (ws && ws.readyState === WebSocket.OPEN) ws.send("stop"); } catch {}
  if (workletNode) { workletNode.port.onmessage = null; workletNode.disconnect(); workletNode = null; }
  if (muteGain) { muteGain.disconnect(); muteGain = null; }
  if (sourceNode) { sourceNode.disconnect(); sourceNode = null; }
  if (analyser) { analyser.disconnect(); analyser = null; }
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
}

// Linear-interpolation resampler -> 16 kHz mono float32.
function resampleTo16k(input, srcRate) {
  if (srcRate === TARGET_SR) return Float32Array.from(input);
  const ratio = srcRate / TARGET_SR;
  const outLen = Math.floor(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const frac = pos - i0;
    out[i] = input[i0] * (1 - frac) + input[i1] * frac;
  }
  return out;
}

// ----------------------------------------------------------------------------
// Transcript rendering
// ----------------------------------------------------------------------------
function render() {
  if (!finals.length && !partialText) {
    els.out.innerHTML = '<p class="placeholder">Your words will appear here, live.</p>';
    els.mWords.textContent = "0";
    return;
  }
  const text = finals.join(" ");
  let html = "";
  if (text) html += `<span class="final">${escapeHtml(text)} </span>`;
  if (partialText) html += `<span class="partial">${escapeHtml(partialText)}</span>`;
  els.out.innerHTML = html;
  els.out.scrollTop = els.out.scrollHeight;
  const full = (text + " " + partialText).trim();
  els.mWords.textContent = full ? full.split(/\s+/).length : "0";
}

function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function updateMetrics(msg) {
  if (typeof msg.proc_ms === "number") els.mLatency.textContent = `${Math.round(msg.proc_ms)} ms`;
  if (typeof msg.rtf === "number") {
    els.mRtf.textContent = `${msg.rtf.toFixed(2)}×`;
    els.mPer1s.textContent = `${Math.round(msg.rtf * 1000)} ms`;
  }
}

// ----------------------------------------------------------------------------
// Visualizer
// ----------------------------------------------------------------------------
function setupViz() {
  const c = els.viz;
  const ctx = c.getContext("2d");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  let W, H;
  function resize() {
    W = c.clientWidth; H = c.clientHeight;
    c.width = W * dpr; c.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener("resize", resize);

  const BARS = 64;
  const freq = () => new Uint8Array(analyser ? analyser.frequencyBinCount : 0);
  let phase = 0;

  function frame() {
    ctx.clearRect(0, 0, W, H);
    const mid = H / 2;
    const gap = 3;
    const bw = (W - gap * (BARS - 1)) / BARS;
    phase += 0.04;

    let data = null;
    if (analyser && recording) { data = freq(); analyser.getByteFrequencyData(data); }

    const grad = ctx.createLinearGradient(0, 0, W, 0);
    grad.addColorStop(0, "#7c5cff");
    grad.addColorStop(0.5, "#ff5c9d");
    grad.addColorStop(1, "#2ee6c8");
    ctx.fillStyle = grad;

    for (let i = 0; i < BARS; i++) {
      let amp;
      if (data) {
        // sample lower 2/3 of spectrum where voice lives
        const idx = Math.floor((i / BARS) * (data.length * 0.66));
        amp = (data[idx] / 255) ** 1.4;
      } else {
        amp = 0.04 + 0.03 * (Math.sin(phase + i * 0.35) * 0.5 + 0.5);
      }
      const h = Math.max(3, amp * (H * 0.92));
      const x = i * (bw + gap);
      roundRect(ctx, x, mid - h / 2, bw, h, bw / 2);
    }
    requestAnimationFrame(frame);
  }
  frame();
}

function roundRect(ctx, x, y, w, h, r) {
  r = Math.min(r, h / 2, w / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
  ctx.fill();
}

// ----------------------------------------------------------------------------
// UI wiring
// ----------------------------------------------------------------------------
els.mic.addEventListener("click", () => (recording ? stopRecording() : startRecording()));

els.copy.addEventListener("click", async () => {
  const text = (finals.join(" ") + (partialText ? " " + partialText : "")).trim();
  if (!text) { toast("Nothing to copy yet"); return; }
  try {
    await navigator.clipboard.writeText(text);
    els.copy.classList.add("copied");
    els.copy.querySelector("span").textContent = "Copied";
    setTimeout(() => {
      els.copy.classList.remove("copied");
      els.copy.querySelector("span").textContent = "Copy";
    }, 1400);
  } catch {
    toast("Copy failed — select manually");
  }
});

els.clear.addEventListener("click", () => {
  finals = []; partialText = "";
  if (ws && ws.readyState === WebSocket.OPEN) ws.send("reset");
  render();
  els.mLatency.textContent = els.mRtf.textContent = els.mPer1s.textContent = "—";
});

let toastTimer = null;
function toast(msg) {
  els.toast.textContent = msg;
  els.toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => els.toast.classList.remove("show"), 2200);
}

document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && e.target === document.body) {
    e.preventDefault();
    recording ? stopRecording() : startRecording();
  }
});

connect();
setupViz();
