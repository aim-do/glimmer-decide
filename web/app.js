// Glimmer Decide - camera client. Vanilla JS, no build step.
const $ = (id) => document.getElementById(id);
const MAX_SIDE = 1024;
const API_TIMEOUT_MS = 260_000;   // server deadline is 240 s; the client waits a little longer

const state = { domains: [], frameUrl: null, lastDecision: null, stream: null,
                decideSeq: 0, saving: false, elapsedTimer: null, gpuTimer: null };

// The server requires this header on POSTs that spend money (blocks localhost CSRF).
const UI_HEADERS = { "X-Requested-With": "glimmer-decide" };

async function api(path, body) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), API_TIMEOUT_MS);
  try {
    const res = await fetch(path, body ? {
      method: "POST", headers: { "Content-Type": "application/json", ...UI_HEADERS },
      body: JSON.stringify(body), signal: ctl.signal,
    } : { headers: UI_HEADERS, signal: ctl.signal });
    if (!res.ok) {
      // Short server explanations (400/403/409) are actionable - show them as is.
      const detail = await res.json().then((j) => j.detail).catch(() => null);
      throw new Error(res.status === 429 ? "Too many requests — wait a moment and retry."
        : res.status === 504 ? "The decision timed out — try again."
        : typeof detail === "string" && detail.length < 300 ? detail
        : `Request failed (${res.status}) — try again.`);
    }
    return res.json();
  } catch (e) {
    if (e.name === "AbortError") throw new Error("The decision timed out — try again.");
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

function clearDecision() {
  // Only the model's prediction goes stale on new input. The human's gold
  // ticks are never cleared here - the labeler owns them.
  state.lastDecision = null;
  const pred = $("predicted");
  if (pred) pred.textContent = "";
}

// ---- camera / frame ---------------------------------------------------------

async function startCamera() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 } }, audio: false,
    });
    $("video").srcObject = state.stream;
    if (!state.frameUrl) showLive();
  } catch (e) {
    $("video").hidden = true;
    if (!state.frameUrl) $("nocam").hidden = false;
  }
}

function showLive() {
  state.frameUrl = null;
  clearDecision();
  $("frame").hidden = true;
  $("video").hidden = false;
  $("live").hidden = true;
}

function showFrozen() {
  $("frame").hidden = false;
  $("video").hidden = true;
  $("nocam").hidden = true;
  $("live").hidden = !state.stream;
}

// Draw a source (video or image) onto the frame canvas, downscaled, and keep a JPEG data URL.
function freeze(source, w, h) {
  const scale = Math.min(1, MAX_SIDE / Math.max(w, h));
  const c = $("frame");
  c.width = Math.round(w * scale);
  c.height = Math.round(h * scale);
  c.getContext("2d").drawImage(source, 0, 0, c.width, c.height);
  state.frameUrl = c.toDataURL("image/jpeg", 0.85);
  clearDecision();
  showFrozen();
}

function grabFrame() {
  if (state.frameUrl) return state.frameUrl;        // uploaded photo or already frozen
  const v = $("video");
  if (!state.stream || !v.videoWidth) return null;
  freeze(v, v.videoWidth, v.videoHeight);
  return state.frameUrl;
}

function redrawFrame() {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const c = $("frame");
      c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
      resolve();
    };
    img.onerror = () => resolve();   // a failed reload must not hang Decide
    img.src = state.frameUrl;
  });
}

function drawBoxes(boxes) {
  const c = $("frame");
  const ctx = c.getContext("2d");
  ctx.lineWidth = Math.max(3, c.width / 200);
  ctx.strokeStyle = "#f4c84a";
  ctx.fillStyle = "rgba(244,200,74,0.18)";
  for (const b of boxes) {
    const sx = c.width / b.w, sy = c.height / b.h;
    const x = b.x1 * sx, y = b.y1 * sy, w = (b.x2 - b.x1 + 1) * sx, h = (b.y2 - b.y1 + 1) * sy;
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
  }
}

// A photo works everywhere, camera or not: the Photo button (file picker, or the
// phone camera), dropping an image file onto the frame, or pasting one (Ctrl+V).
function loadPhoto(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) { $("saved").textContent = "That file is not an image."; return; }
  const img = new Image();
  img.onload = () => { freeze(img, img.naturalWidth, img.naturalHeight); URL.revokeObjectURL(img.src); };
  img.onerror = () => { $("saved").textContent = "That photo could not be read."; };
  img.src = URL.createObjectURL(file);
}

$("file").addEventListener("change", (e) => { loadPhoto(e.target.files[0]); e.target.value = ""; });
const stage = document.querySelector(".stage");
stage.addEventListener("dragover", (e) => { e.preventDefault(); stage.classList.add("drop"); });
stage.addEventListener("dragleave", () => stage.classList.remove("drop"));
stage.addEventListener("drop", (e) => {
  e.preventDefault();
  stage.classList.remove("drop");
  loadPhoto(e.dataTransfer.files[0]);
});
document.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (!item) return;   // plain text keeps pasting into the goal field
  e.preventDefault();
  loadPhoto(item.getAsFile());
});
$("live").addEventListener("click", showLive);

// ---- decide -----------------------------------------------------------------

async function onDecide() {
  const goal = $("goal").value.trim();
  if (!goal) { $("goal").focus(); return; }
  const blind = $("blind").checked;
  const image = grabFrame();
  if (!image && !blind) { $("saved").textContent = "No frame: allow the camera or upload a photo."; return; }

  const seq = ++state.decideSeq;
  $("decide").disabled = true;
  const t0 = Date.now();
  $("decide").textContent = "Thinking…";
  state.elapsedTimer = setInterval(() => {
    if (seq === state.decideSeq) $("decide").textContent = `Thinking… ${Math.round((Date.now() - t0) / 1000)}s`;
  }, 1000);
  try {
    if (state.frameUrl) await redrawFrame();
    const r = await api("/api/decide", {
      domain: $("domain").value, goal, image: blind ? null : image,
      decider: $("decider").value, highlight: true,
    });
    if (seq !== state.decideSeq) return;   // a newer request superseded this one
    state.lastDecision = r.decisions.find((d) => d.action) || r.decisions[0];
    renderDecisions(r.decisions, r.highlight, r.highlight_decider);
    if (r.highlight?.boxes?.length) drawBoxes(r.highlight.boxes);
  } catch (e) {
    if (seq !== state.decideSeq) return;
    renderDecisions([{ decider: "error", hint: "Request failed", error: String(e.message || e), raw: "", latency_ms: 0 }]);
  } finally {
    if (seq === state.decideSeq) {
      clearInterval(state.elapsedTimer);
      $("decide").disabled = false;
      $("decide").textContent = "What should I use?";
    }
  }
}

const DECIDER_NAMES = { encoder: "MetaEncoder · System 1", glimmer: "Glimmer · reasoning", error: "error" };

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function renderDecisions(decisions, highlight, highlightDecider) {
  const names = Object.fromEntries(state.domains.flatMap((d) => d.actions).map((a) => [a.id, a.name]));
  $("results").replaceChildren(...decisions.map((d) => {
    const card = el("div", "result");
    card.append(el("div", "who", DECIDER_NAMES[d.decider] || d.decider));
    const refusal = !d.action || d.action === "NONE" || d.action === "ASK_VIEW";
    card.append(el("div", "hint" + (refusal ? " none" : ""), d.hint));
    const parts = [d.action ? `${d.action}` : "no decision", `${d.latency_ms} ms`];
    if (d.blind) parts.push("blind");
    if (highlight && d.decider === highlightDecider) {
      parts.push(highlight.error ? "SAM highlight unavailable" : `SAM ${highlight.boxes.length} box(es), ${highlight.latency_ms} ms`);
    }
    if (d.error) parts.push(d.error);
    card.append(el("div", "meta", parts.join(" · ")));
    if (d.probs) {
      const ul = el("ul", "probs");
      Object.entries(d.probs).sort((a, b) => b[1] - a[1]).forEach(([id, p]) => {
        const li = el("li", id === d.action ? "top" : "");
        li.style.setProperty("--p", `${(p * 100).toFixed(1)}%`);
        li.append(el("span", "", names[id] || id), el("span", "", `${(p * 100).toFixed(1)}%`));
        ul.append(li);
      });
      card.append(ul);
      card.append(el("div", "meta", "Probabilities are uncalibrated — use them to compare options, not as confidence."));
    }
    const det = el("details");
    det.append(el("summary", "", "raw output"), el("pre", "", d.raw || ""));
    card.append(det);
    return card;
  }));
  // No pre-ticking: the model's own prediction must not bias the human label.
  const pred = $("predicted");
  if (pred) pred.textContent = state.lastDecision?.action
    ? `Model predicted: ${state.lastDecision.action} (not pre-ticked — label independently).` : "";
}

// ---- labeling ---------------------------------------------------------------

async function onSave() {
  if (state.saving) return;
  // Read the human labels BEFORE grabFrame(): freezing a live frame clears
  // the current decision state, which must not wipe the labeler's ticks.
  const gold = [...document.querySelectorAll("#gold input:checked")].map((cb) => cb.value);
  const goal = $("goal").value.trim();
  const image = state.frameUrl || grabFrame();
  if (!image || !goal || !gold.length) { $("saved").textContent = "Need a frame, a goal and at least one ticked action."; return; }
  state.saving = true;
  $("save").disabled = true;
  try {
    const r = await api("/api/capture", {
      domain: $("domain").value, goal, image, gold, predicted: state.lastDecision?.action ?? null,
    });
    $("saved").textContent = `saved ${r.saved}`;
  } catch (e) {
    $("saved").textContent = `Save failed: ${e.message || e}`;
  } finally {
    state.saving = false;
    $("save").disabled = false;
  }
}

// ---- voice (Web Speech API where available; Glimmer has no audio input) -----

function setupMic() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return;
  $("mic").hidden = false;
  $("mic").addEventListener("click", () => {
    const rec = new SR();
    rec.lang = navigator.language || "en-US";
    const mic = $("mic");
    mic.textContent = "…";
    rec.onresult = (e) => { $("goal").value = e.results[0][0].transcript; clearDecision(); };
    rec.onerror = () => { $("saved").textContent = "Voice input failed — type the goal instead."; };
    rec.onend = () => { mic.textContent = "🎤"; };
    try { rec.start(); } catch { mic.textContent = "🎤"; }
  });
}

// ---- MetaEncoder GPU on Modal (Start / Stop) ---------------------------------

const GPU_POLL_BUSY_MS = 3000;
const GPU_POLL_IDLE_MS = 30000;

function gpuView(s) {
  if (!s.available) return { text: s.reason, cls: "off", start: false, stop: false };
  if (s.op === "starting") return { text: "Deploying…", cls: "busy", start: false, stop: false };
  if (s.op === "warming") {
    return { text: `Starting GPU… ${s.elapsed_s}s (cold start ~2 min)`, cls: "busy", start: false, stop: true };
  }
  if (s.op === "stopping") return { text: "Stopping…", cls: "busy", start: false, stop: false };
  const err = s.error ? ` — ${s.error}` : "";
  if (s.deployed === null) return { text: `Status unknown (Modal did not answer)${err}`, cls: "off", start: true, stop: true };
  if (s.containers > 0) return { text: `Running · GPU on, ~$4/h until Stop or 5 idle min${err}`, cls: "on", start: false, stop: true };
  if (s.deployed) return { text: `Asleep · the next MetaEncoder request wakes it (~2 min)${err}`, cls: "idle", start: true, stop: true };
  return { text: `Stopped · no GPU cost${err}`, cls: "off", start: true, stop: false };
}

async function refreshGpu() {
  clearTimeout(state.gpuTimer);
  let s;
  try { s = await api("/api/encoder"); } catch (e) { s = { available: false, reason: "GPU status unavailable." }; }
  const v = gpuView(s);
  $("gpu").hidden = false;
  $("gpu-state").textContent = v.text;
  $("gpu-state").className = "gpu-state " + v.cls;
  $("gpu-start").disabled = !v.start;
  $("gpu-stop").disabled = !v.stop;
  const busy = s.available && s.op && s.op !== "idle";
  if (!document.hidden) state.gpuTimer = setTimeout(refreshGpu, busy ? GPU_POLL_BUSY_MS : GPU_POLL_IDLE_MS);
}

async function gpuAction(path) {
  $("gpu-start").disabled = true;
  $("gpu-stop").disabled = true;
  try {
    await api(path, {});
  } catch (e) {
    $("gpu-state").textContent = e.message || String(e);
    clearTimeout(state.gpuTimer);
    state.gpuTimer = setTimeout(refreshGpu, 5000);   // re-enable the buttons from real state
    return;
  }
  refreshGpu();
}

// ---- init -------------------------------------------------------------------

function renderDomain() {
  const d = state.domains.find((x) => x.id === $("domain").value);
  if (!d) return;
  clearDecision();
  $("examples").replaceChildren(...d.examples.map((g) => {
    const b = document.createElement("button");
    b.textContent = g;
    b.onclick = () => { $("goal").value = g; clearDecision(); };
    return b;
  }));
  $("gold").replaceChildren(...d.actions.map((a) => {
    const l = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = a.id;
    l.append(cb, document.createTextNode(" " + a.id.replace("TAKE_", "").toLowerCase()));
    return l;
  }));
}

async function init() {
  const [status, domains] = await Promise.all([api("/api/status"), api("/api/domains")]);
  const pill = $("status");
  const live = [status.encoder && "MetaEncoder", status.glimmer && "Glimmer", status.sam && "SAM"].filter(Boolean);
  pill.textContent = status.mock ? "mock mode" : live.join(" + ") || "nothing configured";
  pill.className = "pill " + (status.mock ? "mock" : "live");
  $("decider").value = status.default_decider;
  for (const o of $("decider").options) {
    o.disabled = (o.value === "encoder" && !status.encoder) || (o.value === "glimmer" && !status.glimmer)
      || (o.value === "both" && !(status.encoder && status.glimmer));
  }
  // No automatic GPU warm-up on page load: the GPU runs only after Start in the panel.
  $("gpu-start").addEventListener("click", () => gpuAction("/api/encoder/start"));
  $("gpu-stop").addEventListener("click", () => gpuAction("/api/encoder/stop"));
  document.addEventListener("visibilitychange", () => { if (!document.hidden) refreshGpu(); });
  refreshGpu();
  state.domains = domains;
  $("domain").replaceChildren(...domains.map((d) => new Option(d.title, d.id)));
  $("domain").addEventListener("change", renderDomain);
  renderDomain();
  $("decide").addEventListener("click", onDecide);
  $("save").addEventListener("click", onSave);
  $("goal").addEventListener("input", clearDecision);
  $("goal").addEventListener("keydown", (e) => { if (e.key === "Enter" && !$("decide").disabled) onDecide(); });
  setupMic();
  startCamera();
}

init().catch(() => {
  const pill = $("status");
  pill.textContent = "server unreachable — click to retry";
  pill.className = "pill mock";
  pill.style.cursor = "pointer";
  pill.onclick = () => window.location.reload();
  renderDecisions([{ decider: "error", hint: "Could not load the app",
                     error: "Check the server is running, then reload.", raw: "", latency_ms: 0 }]);
});
