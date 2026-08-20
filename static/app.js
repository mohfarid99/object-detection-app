/* YOLO12 detection UI: upload-video jobs + live webcam over websocket. */
const $ = (id) => document.getElementById(id);
const api = (path, opts) => fetch(path, opts).then(async (r) => {
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || `${r.status} ${r.statusText}`);
  return body;
});

const state = {
  meta: null,
  palette: [],
  classes: [],
  selectedClasses: new Set(),   // empty = all
  file: null,
  job: null,
  poll: null,
  ws: null,
  stream: null,
  running: false,
  inFlight: false,
  frameTimes: [],
};

/* ---------------------------------------------------------------- settings */

const settings = () => ({
  model_id: $("modelSelect").value,
  conf: parseFloat($("conf").value),
  iou: parseFloat($("iou").value),
  imgsz: parseInt($("imgsz").value, 10),
  stride: parseInt($("stride").value, 10),
  track: $("track").checked,
  classes: [...state.selectedClasses],
});

const colorFor = (cls) => state.palette[cls % state.palette.length] || "#3b9bff";

async function loadMeta() {
  const meta = await api("/api/meta");
  state.meta = meta;
  state.palette = meta.palette;
  state.classes = meta.classes;

  const sel = $("modelSelect");
  sel.innerHTML = "";
  meta.models.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `${m.label}${m.downloaded ? " ✓" : " ⤓"}`;
    opt.dataset.note = `${m.params} params · mAP ${m.map} · ${m.note}${m.downloaded ? "" : " (downloads on first use)"}`;
    sel.appendChild(opt);
  });
  sel.value = meta.default_model;
  updateModelNote();

  $("deviceBadge").textContent = `Running on ${meta.device}`;
  $("footDevice").textContent = `inference on ${meta.device} · max upload ${meta.max_upload_mb} MB`;
  buildClassList();
}

function updateModelNote() {
  const opt = $("modelSelect").selectedOptions[0];
  $("modelNote").textContent = opt ? opt.dataset.note : "";
  $("modelBadge").textContent = opt ? opt.textContent.replace(/[✓⤓]\s*$/, "").trim() : "";
}

function buildClassList() {
  const list = $("classList");
  const q = $("classSearch").value.trim().toLowerCase();
  list.innerHTML = "";
  state.classes.forEach((name, idx) => {
    if (q && !name.includes(q)) return;
    const label = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = state.selectedClasses.has(idx);
    cb.onchange = () => {
      cb.checked ? state.selectedClasses.add(idx) : state.selectedClasses.delete(idx);
      updateClassCount();
      pushLiveConfig();
    };
    const dot = document.createElement("span");
    dot.className = "swatch";
    dot.style.background = colorFor(idx);
    const txt = document.createElement("span");
    txt.textContent = name;
    label.append(cb, dot, txt);
    list.appendChild(label);
  });
  updateClassCount();
}

function updateClassCount() {
  const n = state.selectedClasses.size;
  $("classCount").textContent = n === 0 ? `all ${state.classes.length}` : `${n} selected`;
}

/* ------------------------------------------------------------------- tabs */

document.querySelectorAll(".tab").forEach((tab) => {
  tab.onclick = () => {
    location.hash = tab.dataset.tab;
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    document.querySelectorAll(".tabpane").forEach((p) => {
      p.classList.toggle("active", p.id === `tab-${tab.dataset.tab}`);
    });
    document.querySelectorAll(".video-only").forEach((el) => {
      el.style.display = tab.dataset.tab === "upload" ? "" : "none";
    });
    if (tab.dataset.tab !== "live") stopCamera();
  };
});

// #live / #upload in the URL opens that tab straight away.
const hashTab = document.querySelector(`.tab[data-tab="${location.hash.slice(1)}"]`);
if (hashTab) hashTab.click();

/* --------------------------------------------------------------- controls */

$("conf").oninput = (e) => { $("confVal").textContent = (+e.target.value).toFixed(2); pushLiveConfig(); };
$("iou").oninput = (e) => { $("iouVal").textContent = (+e.target.value).toFixed(2); pushLiveConfig(); };
$("imgsz").onchange = pushLiveConfig;
$("modelSelect").onchange = () => { updateModelNote(); pushLiveConfig(); };
$("classSearch").oninput = buildClassList;
$("clearClasses").onclick = () => { state.selectedClasses.clear(); buildClassList(); pushLiveConfig(); };

$("prepareBtn").onclick = async () => {
  const btn = $("prepareBtn");
  const id = $("modelSelect").value;
  btn.disabled = true;
  btn.textContent = "Downloading / loading…";
  try {
    const res = await api(`/api/models/${id}/prepare`, { method: "POST" });
    btn.textContent = `Ready in ${res.seconds}s ✓`;
    await loadMetaKeepingSelection(id);
  } catch (err) {
    btn.textContent = `Failed: ${err.message}`;
  } finally {
    setTimeout(() => { btn.textContent = "Download / warm up model"; btn.disabled = false; }, 2500);
  }
};

async function loadMetaKeepingSelection(id) {
  const conf = settings();
  await loadMeta();
  $("modelSelect").value = id || conf.model_id;
  updateModelNote();
}

/* ------------------------------------------------------------ upload flow */

const dropzone = $("dropzone");
const fileInput = $("fileInput");

$("browseBtn").onclick = (e) => { e.stopPropagation(); fileInput.click(); };
dropzone.onclick = () => fileInput.click();
dropzone.ondragover = (e) => { e.preventDefault(); dropzone.classList.add("drag"); };
dropzone.ondragleave = () => dropzone.classList.remove("drag");
dropzone.ondrop = (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag");
  if (e.dataTransfer.files[0]) pickFile(e.dataTransfer.files[0]);
};
fileInput.onchange = () => fileInput.files[0] && pickFile(fileInput.files[0]);

function pickFile(file) {
  state.file = file;
  $("fileName").textContent = `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`;
  $("runBtn").disabled = false;
  hide("uploadError");
}

$("runBtn").onclick = async () => {
  if (!state.file) return;
  const s = settings();
  const fd = new FormData();
  fd.append("file", state.file);
  fd.append("model_id", s.model_id);
  fd.append("conf", s.conf);
  fd.append("iou", s.iou);
  fd.append("imgsz", s.imgsz);
  fd.append("stride", s.stride);
  fd.append("track", s.track);
  if (s.classes.length) fd.append("classes", s.classes.join(","));

  hide("uploadError");
  hide("result");
  $("runBtn").disabled = true;
  $("cancelBtn").hidden = false;
  show("progressWrap");
  setProgress(0, "Uploading…", "");

  try {
    const job = await api("/api/jobs", { method: "POST", body: fd });
    state.job = job;
    pollJob(job.id);
  } catch (err) {
    failUpload(err.message);
  }
};

$("cancelBtn").onclick = async () => {
  if (state.job) await api(`/api/jobs/${state.job.id}/cancel`, { method: "POST" }).catch(() => {});
};

$("newRunBtn").onclick = () => {
  hide("result");
  hide("progressWrap");
  $("resultVideo").removeAttribute("src");
  $("fileName").textContent = "";
  state.file = null;
  $("runBtn").disabled = true;
  fileInput.value = "";
};

function pollJob(id) {
  clearInterval(state.poll);
  state.poll = setInterval(async () => {
    try {
      const job = await api(`/api/jobs/${id}`);
      state.job = job;
      renderJob(job);
      if (["done", "error", "cancelled"].includes(job.status)) {
        clearInterval(state.poll);
        $("cancelBtn").hidden = true;
        $("runBtn").disabled = false;
        if (job.status === "done") showResult(job);
        if (job.status === "error") failUpload(job.message);
        if (job.status === "cancelled") setProgress(job.progress, "Cancelled.", "");
      }
    } catch (err) {
      clearInterval(state.poll);
      failUpload(err.message);
    }
  }, 600);
}

function renderJob(job) {
  const pct = Math.round(job.progress * 100);
  const frames = job.frames_total ? `${job.frames_done} / ${job.frames_total} frames` : `${job.frames_done} frames`;
  const speed = job.process_fps ? ` · ${job.process_fps.toFixed(1)} fps` : "";
  setProgress(job.progress, `${job.message} ${frames}${speed}`, job.eta ? `~${fmtTime(job.eta)} left` : `${pct}%`);
}

function showResult(job) {
  setProgress(1, `Done · ${job.frames_done} frames in ${fmtTime(job.elapsed)}`, "100%");
  const video = $("resultVideo");
  video.src = `${job.video_url}?t=${Date.now()}`;
  $("downloadBtn").href = job.download_url;
  show("result");
  $("uploadStats").innerHTML = statTiles([
    ["Objects detected", job.detections_total.toLocaleString()],
    ["Frames", job.frames_done.toLocaleString()],
    ...(job.track ? [["Unique tracks", job.unique_tracks.toLocaleString()]] : []),
    ["Resolution", `${job.width}×${job.height}`],
    ["Processing", `${job.process_fps.toFixed(1)} fps`],
    ["Model", job.model_id],
  ]) + chips(job.class_counts);
}

function failUpload(msg) {
  const el = $("uploadError");
  el.textContent = msg;
  el.hidden = false;
  hide("progressWrap");
  $("cancelBtn").hidden = true;
  $("runBtn").disabled = false;
}

function setProgress(p, text, right) {
  $("progressBar").style.width = `${Math.round(p * 100)}%`;
  $("progressText").textContent = text;
  $("progressEta").textContent = right;
}

/* -------------------------------------------------------------- live flow */

const cam = $("cam");
const overlay = $("overlay");
const ctx = overlay.getContext("2d");
const grabber = document.createElement("canvas");
const gctx = grabber.getContext("2d");

$("mirror").onchange = () => $("stage").classList.toggle("mirrored", $("mirror").checked);
$("stage").classList.toggle("mirrored", $("mirror").checked);

$("camStart").onclick = startCamera;
$("camStop").onclick = stopCamera;
$("camSelect").onchange = () => { if (state.running) startCamera(); };

async function listCameras() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const sel = $("camSelect");
  const current = sel.value;
  sel.innerHTML = '<option value="">Default camera</option>';
  devices.filter((d) => d.kind === "videoinput").forEach((d, i) => {
    const opt = document.createElement("option");
    opt.value = d.deviceId;
    opt.textContent = d.label || `Camera ${i + 1}`;
    sel.appendChild(opt);
  });
  if (current) sel.value = current;
}

async function startCamera() {
  hide("liveError");
  if (!navigator.mediaDevices?.getUserMedia) {
    return liveError("This browser blocks camera access. Open the app over http://localhost or HTTPS.");
  }
  stopCamera(true);
  try {
    const deviceId = $("camSelect").value;
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: deviceId ? { deviceId: { exact: deviceId } } : { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    cam.srcObject = state.stream;
    await cam.play();
    await listCameras();
    $("stageEmpty").hidden = true;
    $("hud").hidden = false;
    $("camStart").disabled = true;
    $("camStop").disabled = false;
    state.running = true;
    openSocket();
  } catch (err) {
    liveError(`Could not start the camera: ${err.message}`);
  }
}

function stopCamera(silent) {
  state.running = false;
  state.inFlight = false;
  if (state.ws) { state.ws.onclose = null; state.ws.close(); state.ws = null; }
  if (state.stream) { state.stream.getTracks().forEach((t) => t.stop()); state.stream = null; }
  cam.srcObject = null;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  $("camStart").disabled = false;
  $("camStop").disabled = true;
  if (!silent) {
    $("stageEmpty").hidden = false;
    $("hud").hidden = true;
    $("liveStats").innerHTML = "";
  }
}

function openSocket() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/detect`);
  ws.binaryType = "arraybuffer";
  state.ws = ws;

  ws.onopen = () => pushLiveConfig(true);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "error") { state.inFlight = false; return liveError(msg.message); }
    if (msg.type === "ready") { hide("liveError"); return sendFrame(); }
    if (msg.type === "result") {
      state.inFlight = false;
      drawOverlay(msg);
      updateHud(msg);
      if (state.running) requestAnimationFrame(sendFrame);
    }
  };
  ws.onerror = () => liveError("Lost the connection to the detection server.");
  ws.onclose = () => { if (state.running) liveError("Detection socket closed."); };
}

function pushLiveConfig(force) {
  const ws = state.ws;
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const s = settings();
  ws.send(JSON.stringify({
    type: "config",
    model_id: s.model_id,
    conf: s.conf,
    iou: s.iou,
    imgsz: s.imgsz,
    classes: s.classes,
  }));
  if (force) $("hudFps").textContent = "loading model…";
}

function sendFrame() {
  const ws = state.ws;
  // One frame in flight at a time: the server's reply paces the capture loop.
  if (state.inFlight || !state.running || !ws || ws.readyState !== WebSocket.OPEN || !cam.videoWidth) return;
  const target = 640;
  const scale = Math.min(1, target / cam.videoWidth);
  grabber.width = Math.round(cam.videoWidth * scale);
  grabber.height = Math.round(cam.videoHeight * scale);
  gctx.drawImage(cam, 0, 0, grabber.width, grabber.height);
  state.inFlight = true;
  grabber.toBlob((blob) => {
    if (blob && state.running && ws.readyState === WebSocket.OPEN) {
      blob.arrayBuffer().then((buf) => ws.send(buf));
    } else {
      state.inFlight = false;
    }
  }, "image/jpeg", 0.6);
}

function drawOverlay(msg) {
  if (overlay.width !== cam.videoWidth || overlay.height !== cam.videoHeight) {
    overlay.width = cam.videoWidth;
    overlay.height = cam.videoHeight;
  }
  const mirrored = $("mirror").checked;
  const sx = overlay.width / msg.w;
  const sy = overlay.height / msg.h;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.lineWidth = Math.max(2, overlay.width / 400);
  ctx.font = `${Math.max(13, Math.round(overlay.width / 45))}px -apple-system, system-ui, sans-serif`;
  ctx.textBaseline = "top";

  msg.dets.forEach((d) => {
    let [x1, y1, x2, y2] = d.box;
    x1 *= sx; x2 *= sx; y1 *= sy; y2 *= sy;
    if (mirrored) { const w = overlay.width; [x1, x2] = [w - x2, w - x1]; }
    const color = colorFor(d.cls);
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const label = `${d.name} ${(d.conf * 100).toFixed(0)}%`;
    const pad = 5;
    const tw = ctx.measureText(label).width;
    const th = parseInt(ctx.font, 10) + pad;
    const ly = Math.max(0, y1 - th - 2);
    ctx.fillStyle = color;
    ctx.fillRect(x1, ly, tw + pad * 2, th);
    ctx.fillStyle = "#08121e";
    ctx.fillText(label, x1 + pad, ly + 2);
  });
}

function updateHud(msg) {
  const now = performance.now();
  state.frameTimes.push(now);
  while (state.frameTimes.length > 20) state.frameTimes.shift();
  const span = state.frameTimes[state.frameTimes.length - 1] - state.frameTimes[0];
  const fps = span > 0 ? ((state.frameTimes.length - 1) * 1000) / span : 0;

  $("hudFps").textContent = `${fps.toFixed(1)} fps`;
  $("hudMs").textContent = `${msg.ms} ms inference`;
  $("hudObjs").textContent = `${msg.dets.length} object${msg.dets.length === 1 ? "" : "s"}`;

  const counts = {};
  msg.dets.forEach((d) => { counts[d.name] = (counts[d.name] || 0) + 1; });
  $("liveStats").innerHTML = chips(counts, true);
}

function liveError(msg) {
  const el = $("liveError");
  el.textContent = msg;
  el.hidden = false;
}

/* ------------------------------------------------------------------ utils */

function statTiles(pairs) {
  return pairs.map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
}

function chips(counts, live) {
  const entries = Object.entries(counts || {});
  if (!entries.length) return "";
  const idx = (name) => state.classes.indexOf(name);
  return `<div class="chips">` + entries.map(([name, n]) =>
    `<span class="chip"><span class="dot" style="background:${colorFor(Math.max(0, idx(name)))}"></span>${name}<span class="n">${n}</span></span>`
  ).join("") + `</div>`;
}

const fmtTime = (s) => (s < 60 ? `${s.toFixed(0)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`);
const show = (id) => ($(id).hidden = false);
const hide = (id) => ($(id).hidden = true);

loadMeta().catch((err) => {
  document.body.insertAdjacentHTML("afterbegin", `<div class="error" style="margin:20px 28px">Could not reach the API: ${err.message}</div>`);
});
