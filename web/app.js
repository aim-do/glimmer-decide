// Glimmer Decide - camera client. Vanilla JS, no build step.
const $ = (id) => document.getElementById(id);
const MAX_SIDE = 1024;

const state = { domains: [], frameUrl: null, lastDecision: null, stream: null };

async function api(path, body) {
  const res = await fetch(path, body ? {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  } : undefined);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
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

$("file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const img = new Image();
  img.onload = () => { freeze(img, img.naturalWidth, img.naturalHeight); URL.revokeObjectURL(img.src); };
  img.src = URL.createObjectURL(f);
  e.target.value = "";
});
$("live").addEventListener("click", showLive);

// ---- decide -----------------------------------------------------------------

async function onDecide() {
  const goal = $("goal").value.trim();
  if (!goal) { $("goal").focus(); return; }
  const blind = $("blind").checked;
  const image = grabFrame();
  if (!image && !blind) { alert("No frame: allow the camera or upload a photo."); return; }

  $("decide").disabled = true;
  $("decide").textContent = "Thinking…";
  try {
    if (state.frameUrl) await redrawFrame();
    const r = await api("/api/decide", {
      domain: $("domain").value, goal, image: blind ? null : image,
      decider: $("decider").value, highlight: true,
    });
    state.lastDecision = r.decisions.find((d) => d.action) || r.decisions[0];
    renderDecisions(r.decisions, r.highlight);
    if (r.highlight?.boxes?.length) drawBoxes(r.highlight.boxes);
  } catch (e) {
    renderDecisions([{ decider: "error", hint: "Request failed", error: String(e), raw: "", latency_ms: 0 }]);
  } finally {
    $("decide").disabled = false;
    $("decide").textContent = "What should I use?";
  }
}

const DECIDER_NAMES = { encoder: "MetaEncoder · System 1", glimmer: "Glimmer · reasoning", error: "error" };

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function renderDecisions(decisions, highlight) {
  const names = Object.fromEntries(state.domains.flatMap((d) => d.actions).map((a) => [a.id, a.name]));
  $("results").replaceChildren(...decisions.map((d, i) => {
    const card = el("div", "result");
    card.append(el("div", "who", DECIDER_NAMES[d.decider] || d.decider));
    const refusal = !d.action || d.action === "NONE" || d.action === "ASK_VIEW";
    card.append(el("div", "hint" + (refusal ? " none" : ""), d.hint));
    const parts = [d.action ? `${d.action}` : "no decision", `${d.latency_ms} ms`];
    if (d.blind) parts.push("blind");
    if (i === 0 && highlight) {
      parts.push(highlight.error ? "SAM error" : `SAM ${highlight.boxes.length} box(es), ${highlight.latency_ms} ms`);
    }
    if (d.error) parts.push(d.error);
    card.append(el("div", "meta", parts.join(" · ")));
    if (d.probs) {
      const ul = el("ul", "probs");
      Object.entries(d.probs).sort((a, b) => b[1] - a[1]).slice(0, 4).forEach(([id, p], k) => {
        const li = el("li", k === 0 ? "top" : "");
        li.style.setProperty("--p", `${(p * 100).toFixed(1)}%`);
        li.append(el("span", "", names[id] || id), el("span", "", `${(p * 100).toFixed(1)}%`));
        ul.append(li);
      });
      card.append(ul);
    }
    const det = el("details");
    det.append(el("summary", "", "raw output"), el("pre", "", d.raw || ""));
    card.append(det);
    return card;
  }));
  // Pre-tick the predicted action in the labeling box; the human corrects it.
  for (const cb of document.querySelectorAll("#gold input")) cb.checked = cb.value === state.lastDecision?.action;
}

// ---- labeling ---------------------------------------------------------------

async function onSave() {
  const image = state.frameUrl || grabFrame();
  const goal = $("goal").value.trim();
  const gold = [...document.querySelectorAll("#gold input:checked")].map((cb) => cb.value);
  if (!image || !goal || !gold.length) { alert("Need a frame, a goal and at least one ticked action."); return; }
  const r = await api("/api/capture", {
    domain: $("domain").value, goal, image, gold, predicted: state.lastDecision?.action ?? null,
  });
  $("saved").textContent = `saved ${r.saved}`;
}

// ---- voice (Web Speech API where available; Glimmer has no audio input) -----

function setupMic() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return;
  $("mic").hidden = false;
  $("mic").addEventListener("click", () => {
    const rec = new SR();
    rec.lang = "en-US";
    rec.onresult = (e) => { $("goal").value = e.results[0][0].transcript; };
    rec.start();
  });
}

// ---- init -------------------------------------------------------------------

function renderDomain() {
  const d = state.domains.find((x) => x.id === $("domain").value);
  $("examples").replaceChildren(...d.examples.map((g) => {
    const b = document.createElement("button");
    b.textContent = g;
    b.onclick = () => { $("goal").value = g; };
    return b;
  }));
  $("gold").replaceChildren(...d.actions.map((a) => {
    const l = document.createElement("label");
    l.innerHTML = `<input type="checkbox" value="${a.id}"> ${a.id.replace("TAKE_", "").toLowerCase()}`;
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
  if (status.encoder && !status.mock) api("/api/warmup", {}).catch(() => {});   // start the GPU early
  state.domains = domains;
  $("domain").replaceChildren(...domains.map((d) => new Option(d.title, d.id)));
  $("domain").addEventListener("change", renderDomain);
  renderDomain();
  $("decide").addEventListener("click", onDecide);
  $("save").addEventListener("click", onSave);
  $("goal").addEventListener("keydown", (e) => { if (e.key === "Enter") onDecide(); });
  setupMic();
  startCamera();
}

init();
