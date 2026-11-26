import { fitInside, moveGeometry, validCorners, resolveApiBase, frameDetailScore } from "./overlay.js";

const DEFAULT_API = "https://backend-production-e51676e6.up.railway.app";
const STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const $ = id => document.getElementById(id);
const state = {
  running: false, starting: false, source: null, facing: "environment", timer: null,
  generation: 0, controller: null, session: crypto.randomUUID(), corners: null,
  cameraAttempt: 0, cornersSource: "auto", lockReading: false,
  manualCorners: [], locking: false, move: null, fen: null, setupFen: null,
  player: "w", orientation: "auto", resolvedOrientation: "w", depth: 10, mode: "win", busy: false,
};
let apiBase;
try {
  apiBase = resolveApiBase({ origin: location.origin, search: location.search,
    stored: localStorage.getItem("chess_api_base") || "", fallback: DEFAULT_API });
} catch { apiBase = DEFAULT_API; }

function setStatus(message, type = "") {
  $("statusMsg").textContent = message;
  $("statusBar").className = `status-bar ${type}`;
}

function clearMove(message = "Point your camera at the board to find your next move.") {
  state.move = null;
  $("frameMoveLabel").hidden = true;
  $("moveWaiting").hidden = false;
  $("moveResult").hidden = true;
  $("waitText").textContent = message;
  drawOverlay();
}

function updateControls() {
  $("btnStart").disabled = state.running || state.starting;
  $("btnStop").disabled = !state.running && !state.starting;
  $("btnSwitchCamera").disabled = state.starting;
  $("liveIndicator").hidden = !state.running;
  $("btnLockBoard").disabled = !state.source || state.source === "demo";
  $("btnUnlockBoard").disabled = !state.corners && !state.locking;
  $("btnAnalyzeFen").disabled = state.busy;
  $("btnSetPosition").disabled = state.busy;
  $("cameraFrame").classList.toggle("is-locking", state.locking);
}

function cancelRequest() {
  state.generation++;
  state.controller?.abort();
  state.controller = null;
  state.busy = false;
  clearTimeout(state.timer);
}

function resetTracking({ keepCorners = false } = {}) {
  cancelRequest();
  state.session = crypto.randomUUID();
  state.fen = null;
  state.setupFen = null;
  state.resolvedOrientation = state.orientation === "b" ? "b" : "w";
  if (!keepCorners) { state.corners = null; state.cornersSource = "auto"; }
  state.lockReading = false;
  state.manualCorners = [];
  state.locking = false;
  $("boardStatus").textContent = state.corners ? "Board locked" : "Looking for board";
  $("confBadge").textContent = "Auto detection";
  $("confBadge").hidden = !state.corners;
  clearMove();
  updateControls();
}

function stopCamera({ clear = true } = {}) {
  cancelRequest();
  state.cameraAttempt++;
  state.running = false;
  state.starting = false;
  $("cameraFeed").srcObject?.getTracks().forEach(track => track.stop());
  $("cameraFeed").srcObject = null;
  $("cameraFeed").hidden = true;
  if (clear) {
    state.source = null;
    state.corners = null;
    state.cornersSource = "auto";
    state.lockReading = false;
    state.locking = false;
    $("photoPreview").hidden = true;
    $("cameraOverlay").hidden = false;
    $("boardStatus").textContent = "Camera paused";
    $("confBadge").hidden = true;
    clearMove("Camera paused. Start it again to continue.");
    setStatus("Camera paused.");
  }
  updateControls();
}

async function startCamera() {
  if (state.running || state.starting) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("Camera needs a secure HTTPS connection. You can upload a photo instead.", "error"); return;
  }
  resetTracking();
  state.source = null;
  state.starting = true;
  const attempt = ++state.cameraAttempt;
  updateControls(); setStatus("Opening camera…");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: false,
      video: { facingMode: { ideal: state.facing }, width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 24 } } });
    if (attempt !== state.cameraAttempt) { stream.getTracks().forEach(t => t.stop()); return; }
    // Supported phones keep refocusing as the board or camera moves. Treat
    // these as best-effort hints; unsupported cameras still start normally.
    const track = stream.getVideoTracks()[0];
    const capabilities = track?.getCapabilities?.() || {};
    const continuous = {};
    if (capabilities.focusMode?.includes("continuous")) continuous.focusMode = "continuous";
    if (capabilities.exposureMode?.includes("continuous")) continuous.exposureMode = "continuous";
    if (Object.keys(continuous).length) await track.applyConstraints({ advanced: [continuous] }).catch(() => {});
    if (attempt !== state.cameraAttempt) { stream.getTracks().forEach(t => t.stop()); return; }
    $("cameraFeed").srcObject = stream;
    await $("cameraFeed").play();
    if (attempt !== state.cameraAttempt) { stream.getTracks().forEach(t => t.stop()); return; }
    cancelRequest();
    state.running = true;
    state.source = "camera";
    $("cameraFeed").hidden = false;
    $("photoPreview").hidden = true;
    $("cameraOverlay").hidden = true;
    setStatus("Scanning automatically. Keep the full board visible and your hands clear.");
    stream.getVideoTracks()[0]?.addEventListener("ended", () => {
      if ($("cameraFeed").srcObject === stream) stopCamera();
    });
    schedule(0);
  } catch (err) {
    if (attempt !== state.cameraAttempt) return;
    stopCamera({ clear: false });
    setStatus(err.name === "NotAllowedError" ? "Camera access was denied. Allow camera access in your browser or upload a photo."
      : `Could not open the camera: ${err.message}`, "error");
  } finally { if (attempt === state.cameraAttempt) state.starting = false; updateControls(); }
}

async function captureFrame(generation) {
  const source = state.source === "camera" ? $("cameraFeed") : $("photoPreview");
  const width = source.videoWidth || source.naturalWidth, height = source.videoHeight || source.naturalHeight;
  if (!width || !height) return null;
  const canvas = $("captureCanvas"), size = state.source === "camera" ? 960 : 1280;
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d"), fit = fitInside(width, height, size);
  const samples = state.source === "camera" ? 3 : 1;
  const probe = document.createElement("canvas"); probe.width = probe.height = 192;
  const probeCtx = probe.getContext("2d", { willReadFrequently: true });
  let bestImage = null, bestScore = -1;
  for (let sample = 0; sample < samples; sample++) {
    if (generation !== state.generation) return null;
    ctx.fillStyle = "#182923"; ctx.fillRect(0, 0, size, size);
    ctx.drawImage(source, fit.x, fit.y, fit.width, fit.height);
    // Ignore the square frame's letterbox when measuring camera sharpness.
    probeCtx.drawImage(source, 0, 0, 192, 192);
    const score = frameDetailScore(probeCtx.getImageData(0, 0, 192, 192).data, 192, 192);
    if (score > bestScore) { bestScore = score; bestImage = canvas.toDataURL("image/jpeg", .94).split(",")[1]; }
    if (sample + 1 < samples) await new Promise(resolve => setTimeout(resolve, 75));
  }
  return bestImage;
}

function schedule(delay = 850) {
  clearTimeout(state.timer);
  if (state.running && !state.locking && !document.hidden) state.timer = setTimeout(scanFrame, delay);
}

async function apiRequest(path, body, generation) {
  const controller = new AbortController();
  state.controller = controller;
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const res = await fetch(`${apiBase}${path}`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body), signal: controller.signal });
    const data = await res.json();
    if (generation !== state.generation) return null;
    if (!res.ok && !data.status) throw new Error(typeof data.detail === "string" ? data.detail : `Server returned ${res.status}`);
    return data;
  } catch (err) {
    if (generation !== state.generation) return null;
    if (err.name === "AbortError") throw new Error("Analysis timed out. Check your connection; the camera will retry.");
    throw err;
  } finally {
    clearTimeout(timeout);
    if (state.controller === controller) state.controller = null;
  }
}

async function scanFrame() {
  if (state.busy || state.locking || !state.source || state.source === "demo") { schedule(); return; }
  const generation = state.generation, setupFen = state.setupFen;
  state.busy = true; updateControls();
  const t0 = performance.now();
  try {
    const b64 = await captureFrame(generation);
    if (!b64 || generation !== state.generation) return;
    const data = await apiRequest("/api/analyze", {
      image_b64: b64, player_color: state.player, orientation: state.orientation,
      side_to_move: $("turnSelect").value,
      depth: state.depth, mode: state.mode, session_id: state.session,
      ...(state.corners ? { board_corners: state.corners, corners_source: state.cornersSource } : {}),
      ...(setupFen ? { setup_fen: setupFen } : {}),
    }, generation);
    if (data) {
      handleResponse(data, Math.round(performance.now() - t0));
      if (data.position_verified !== false && !data.geometry_held && data.fen
        && ["ok", "no_change", "opponent_turn", "game_over"].includes(data.status)
        && state.setupFen === setupFen) state.setupFen = null;
    }
  } catch (err) {
    if (generation === state.generation) { clearMove("Connection interrupted. Retrying…"); setStatus(err.message, "error"); }
  } finally {
    if (generation === state.generation) { state.busy = false; updateControls(); schedule(); }
  }
}

function handleResponse(data, elapsed) {
  const hasPieces = data.pieces_detected !== false;
  const goodCorners = validCorners(data.board_corners);
  const emptyBoard = data.board_detected === true && !hasPieces && data.status === "setup_required"
    && data.corners_source !== "manual" && !data.geometry_held;
  const released = data.lock_released === true || (data.board_locked === false && !goodCorners
    && ["searching", "reacquiring"].includes(data.lock_state));
  const readingPosition = ["detection_error", "position_uncertain", "tracking_error", "tracking_lost", "recognition_error", "image_blurry", "reacquiring", "recognizing", "reading_position"].includes(data.status)
    || data.position_verified === false || data.geometry_held === true;
  if (["w", "b"].includes(data.orientation)) state.resolvedOrientation = data.orientation;
  if (released || emptyBoard) {
    state.corners = null;
    state.cornersSource = "auto";
    $("boardStatus").textContent = emptyBoard ? "Board found · add pieces" : "Looking for board";
    $("confBadge").hidden = true;
  }
  if (goodCorners && !emptyBoard && !released) {
    state.corners = data.board_corners;
    if (["manual", "auto", "tracked"].includes(data.corners_source)) state.cornersSource = data.corners_source;
    $("boardStatus").textContent = data.geometry_held ? "Board held · refocusing" : readingPosition ? "Board locked · reading pieces" : "Board locked";
    $("confBadge").textContent = data.geometry_held ? "Reading a clearer frame" : `${Math.round((data.confidence || 0) * 100)}% confidence`;
    $("confBadge").hidden = false;
  } else if (state.corners && readingPosition) {
    // A noisy frame must not throw away a previously measured board boundary.
    // Retain it as a held outline; advice remains hidden until pieces are read.
    $("boardStatus").textContent = "Board held · refocusing";
    $("confBadge").textContent = "Reading a clearer frame";
    $("confBadge").hidden = false;
  }
  state.lockReading = Boolean(state.corners && readingPosition);
  if (readingPosition && ["ok", "no_change", "opponent_turn", "game_over"].includes(data.status)) {
    clearMove("Board locked. Reading the position before suggesting a move…");
    setStatus(data.message || "Waiting for a clear, verified position.");
    updateControls(); drawOverlay(); return;
  }
  if (data.fen && ["ok", "no_change", "opponent_turn", "game_over"].includes(data.status)) {
    state.fen = data.fen;
    $("customFenInput").value = data.fen;
    $("turnSelect").value = data.turn || data.fen.split(" ")[1];
  }
  if (data.engine) $("engineBadge").textContent = data.engine === "stockfish" ? "Stockfish" : "Basic engine";
  switch (data.status) {
    case "ok":
    case "no_change":
      if (data.move_uci) {
        state.move = data.move_uci;
        $("moveWaiting").hidden = true; $("moveResult").hidden = false;
        $("moveText").textContent = (data.move_human || `Play ${data.move_uci}`).replace(/^Move /, "");
        $("frameMoveLabel").textContent = $("moveText").textContent;
        $("frameMoveLabel").hidden = !state.corners;
        $("moveSan").textContent = data.move_san || data.move_uci;
        $("moveUci").textContent = `${data.move_uci.slice(0, 2)} → ${data.move_uci.slice(2, 4)}`;
        setStatus(`Your move · ${elapsed} ms${data.status === "no_change" ? " · Position unchanged" : ""}`, "ok");
        drawOverlay();
      } else if (data.status === "no_change" && state.move) setStatus("Position unchanged. Your suggested move is still shown.", "ok");
      else clearMove(data.message || "Waiting for a move suggestion.");
      break;
    case "opponent_turn":
      clearMove("Opponent’s turn. Watching for their move…");
      setStatus("Board locked. The next suggestion appears after your opponent moves."); break;
    case "setup_required":
      clearMove(hasPieces ? "Board found. Set the current position to begin tracking." : "Board found. Put the pieces on the board to begin.");
      setStatus(data.message || "Start from the normal starting position, or set the current position below.");
      if (hasPieces) $("setupPanel").open = true;
      break;
    case "game_over":
      clearMove(data.message || "Game over. No legal moves remain."); setStatus(data.message || "Game over.", "ok"); break;
    case "rate_limited": setStatus("Waiting briefly before the next scan."); break;
    case "detection_error":
    case "position_uncertain":
    case "tracking_error":
    case "tracking_lost":
    case "recognition_error":
    case "image_blurry":
    case "reacquiring":
    case "recognizing":
    case "reading_position":
      if (!state.corners) $("boardStatus").textContent = "Looking for board";
      clearMove(state.corners ? "Board held. Reading a clearer frame before showing the next move…" : "Keep the full board visible. Looking for its four corners…");
      setStatus(data.message || "Could not read this frame. Scanning again…"); break;
    case "engine_starting":
    case "model_loading":
      clearMove("The board reader is starting. Analysis will resume automatically…");
      setStatus(data.message || "Preparing the board reader. Please keep the camera steady."); break;
    default:
      clearMove(data.message || "Could not analyze this position.");
      setStatus(data.message || "Analysis failed. Check the position and try again.", "error");
  }
  updateControls(); drawOverlay();
}

function drawOverlay() {
  const svg = $("moveOverlay"); svg.replaceChildren();
  svg.setAttribute("aria-label", "Detected board alignment");
  const ns = "http://www.w3.org/2000/svg";
  const add = (name, attrs, parent = svg) => {
    const node = document.createElementNS(ns, name);
    Object.entries(attrs).forEach(([key, val]) => node.setAttribute(key, val));
    parent.appendChild(node); return node;
  };
  const points = arr => arr.map(([x, y]) => `${x * 1000},${y * 1000}`).join(" ");
  if (state.locking) {
    state.manualCorners.forEach(([x, y], i) => {
      add("circle", { cx: x * 1000, cy: y * 1000, r: 16, fill: "#c3f578", stroke: "#17392d", "stroke-width": 4 });
      const text = add("text", { x: x * 1000, y: y * 1000 + 5, "text-anchor": "middle", fill: "#17392d", "font-size": 17 });
      text.textContent = i + 1;
    }); return;
  }
  if (!state.corners) return;
  add("polygon", { points: points(state.corners), fill: "none", stroke: "#c3f578", "stroke-width": 3, "stroke-linejoin": "round", opacity: .85 });
  if (state.lockReading) svg.lastElementChild.setAttribute("stroke-dasharray", "12 8");
  state.corners.forEach(([x, y]) => add("circle", { cx: x * 1000, cy: y * 1000, r: 6, fill: "#c3f578" }));
  if (!state.move) return;
  const geometry = moveGeometry(state.move, state.corners, state.resolvedOrientation);
  if (!geometry) return;
  const defs = add("defs", {});
  const marker = add("marker", { id: "moveArrow", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 3.4, markerHeight: 3.4, orient: "auto-start-reverse" }, defs);
  add("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#c3f578" }, marker);
  add("polygon", { points: points(geometry.target), fill: "#c3f578", "fill-opacity": .28, stroke: "#c3f578", "stroke-width": 3 });
  const [fx, fy] = geometry.from.map(v => v * 1000), [tx, ty] = geometry.to.map(v => v * 1000);
  add("line", { x1: fx, y1: fy, x2: tx, y2: ty, stroke: "#17392d", "stroke-opacity": .85, "stroke-width": 19, "stroke-linecap": "round" });
  add("line", { x1: fx, y1: fy, x2: tx, y2: ty, stroke: "#c3f578", "stroke-width": 11, "stroke-linecap": "round", "marker-end": "url(#moveArrow)" });
  add("circle", { cx: fx, cy: fy, r: 13, fill: "#c3f578", stroke: "#17392d", "stroke-width": 4 });
  svg.setAttribute("aria-label", `Suggested move from ${state.move.slice(0, 2)} to ${state.move.slice(2, 4)}`);
}

async function uploadPhoto(file) {
  if (!file) return;
  if (!file.type.startsWith("image/") || file.size > 10 * 1024 * 1024) { setStatus("Choose an image smaller than 10 MB.", "error"); return; }
  stopCamera({ clear: false }); resetTracking();
  state.source = null;
  const attempt = state.cameraAttempt, url = URL.createObjectURL(file), img = $("photoPreview");
  try {
    img.src = url; await img.decode();
    if (attempt !== state.cameraAttempt) return;
    cancelRequest();
    state.source = "photo"; img.hidden = false; $("cameraOverlay").hidden = true;
    setStatus("Reading your board photo…"); updateControls(); await scanFrame();
  } catch { if (attempt === state.cameraAttempt) setStatus("Could not open this image. Try a JPEG or PNG.", "error"); }
  finally { URL.revokeObjectURL(url); $("fileInput").value = ""; }
}

function beginManualLock() {
  if (!state.source || state.source === "demo") return;
  cancelRequest(); state.locking = true; state.manualCorners = [];
  clearMove("Set the four board corners to align the move arrow.");
  setStatus("Tap the board’s top-left, top-right, bottom-right, then bottom-left corner.");
  $("boardStatus").textContent = "Tap corner 1 of 4";
  drawOverlay(); updateControls();
}

$("cameraFrame").addEventListener("click", event => {
  if (!state.locking) return;
  const rect = $("cameraFrame").getBoundingClientRect();
  state.manualCorners.push([(event.clientX - rect.left) / rect.width, (event.clientY - rect.top) / rect.height]);
  if (state.manualCorners.length === 4) {
    if (!validCorners(state.manualCorners)) { state.manualCorners = []; setStatus("Those corners crossed. Tap top-left, top-right, bottom-right, bottom-left.", "error"); }
    else {
      state.corners = state.manualCorners; state.cornersSource = "manual"; state.locking = false;
      $("boardStatus").textContent = "Board locked"; setStatus("Board corners set. Reading the position…"); scanFrame();
    }
  }
  if (state.locking) $("boardStatus").textContent = `Tap corner ${state.manualCorners.length + 1} of 4`;
  drawOverlay(); updateControls();
});

function drawDemoBoard(fen) {
  const canvas = document.createElement("canvas"); canvas.width = canvas.height = 720;
  const ctx = canvas.getContext("2d"), sq = 90;
  const symbols = { p: "♟", r: "♜", n: "♞", b: "♝", q: "♛", k: "♚", P: "♙", R: "♖", N: "♘", B: "♗", Q: "♕", K: "♔" };
  const rows = fen.split(" ")[0].split("/");
  for (let row = 0; row < 8; row++) for (let col = 0; col < 8; col++) {
    ctx.fillStyle = (row + col) % 2 ? "#779581" : "#e6edde"; ctx.fillRect(col * sq, row * sq, sq, sq);
  }
  ctx.font = "70px Georgia, serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  rows.forEach((rank, row) => {
    let col = 0;
    for (const cell of rank) {
      if (/^[1-8]$/.test(cell)) { col += Number(cell); continue; }
      if (symbols[cell] && col < 8) {
        const r = state.resolvedOrientation === "b" ? 7 - row : row, c = state.resolvedOrientation === "b" ? 7 - col : col;
        ctx.fillStyle = cell === cell.toUpperCase() ? "#fffdf2" : "#19372c"; ctx.strokeStyle = "#19372c"; ctx.lineWidth = 1.5;
        ctx.strokeText(symbols[cell], (c + .5) * sq, (r + .53) * sq); ctx.fillText(symbols[cell], (c + .5) * sq, (r + .53) * sq);
      } col++;
    }
  });
  $("photoPreview").src = canvas.toDataURL(); $("photoPreview").hidden = false; $("cameraOverlay").hidden = true;
  state.corners = [[0, 0], [1, 0], [1, 1], [0, 1]];
  $("boardStatus").textContent = "Example board"; $("confBadge").textContent = "Demo position"; $("confBadge").hidden = false;
}

async function analyzeFen(fen) {
  if (!fen) return;
  stopCamera({ clear: false }); resetTracking(); state.source = "demo"; state.busy = true;
  clearMove("Calculating your next move…"); updateControls();
  const generation = state.generation, t0 = performance.now();
  try {
    const data = await apiRequest("/api/analyze-fen", { fen, depth: state.depth, mode: state.mode }, generation);
    if (data) {
      if (["ok", "game_over"].includes(data.status)) drawDemoBoard(fen);
      handleResponse(data, Math.round(performance.now() - t0));
    }
  } catch (err) { if (generation === state.generation) setStatus(err.message, "error"); }
  finally { if (generation === state.generation) { state.busy = false; updateControls(); } }
}

function setPosition(fen) {
  const fields = fen.trim().split(/\s+/);
  if (fields.length !== 6 || !/^[wb]$/.test(fields[1])) { setStatus("Enter a complete FEN with six fields, or use the starting position.", "error"); return; }
  fields[1] = $("turnSelect").value;
  if (state.source && state.source !== "demo") {
    resetTracking({ keepCorners: true }); state.setupFen = fields.join(" ");
    clearMove("Checking the supplied position against the board…"); scanFrame();
  } else analyzeFen(fields.join(" "));
}

async function checkHealth(base = apiBase, label = $("engineStatus").querySelector(".status-label")) {
  const mainLabel = label === $("engineStatus").querySelector(".status-label");
  const displayLabel = text => { if (!mainLabel || base === apiBase) label.textContent = text; };
  displayLabel("Connecting…");
  try {
    const res = await fetch(`${base}/health`, { signal: AbortSignal.timeout(6000) });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const data = await res.json(); if (data.status !== "ok") throw new Error("Engine is not ready");
    displayLabel(data.engine === "stockfish" ? "Engine online" : "Basic engine online");
    if (base === apiBase) { $("engineStatus").className = "status-pill online"; $("engineBadge").textContent = data.engine === "stockfish" ? "Stockfish" : "Basic engine"; }
    return true;
  } catch {
    displayLabel("Engine offline"); if (base === apiBase) $("engineStatus").className = "status-pill offline"; return false;
  }
}

function openApiSettings() {
  $("apiUrlInput").value = apiBase; $("modalPingStatus").textContent = "Use your Railway backend or a local server."; $("apiModal").showModal();
}
function refreshAnalysis(message) {
  cancelRequest(); clearMove(message);
  if (state.source === "demo" && state.fen) analyzeFen(state.fen);
  else if (state.source) scanFrame();
}

$("btnStart").addEventListener("click", startCamera);
$("btnStop").addEventListener("click", () => stopCamera());
$("btnSwitchCamera").addEventListener("click", () => {
  state.facing = state.facing === "environment" ? "user" : "environment";
  if (state.running) { stopCamera({ clear: false }); startCamera(); }
  else setStatus(`The ${state.facing === "environment" ? "rear" : "front"} camera will open next.`);
});
$("btnUploadPhoto").addEventListener("click", () => $("fileInput").click());
$("fileInput").addEventListener("change", e => uploadPhoto(e.target.files[0]));
$("btnLockBoard").addEventListener("click", beginManualLock);
$("btnUnlockBoard").addEventListener("click", () => { resetTracking(); if (state.source && state.source !== "demo") scanFrame(); });
$("btnDemo").addEventListener("click", () => analyzeFen(STARTING_FEN));
$("btnAnalyzeFen").addEventListener("click", () => analyzeFen($("customFenInput").value.trim()));
$("btnSetPosition").addEventListener("click", () => setPosition($("customFenInput").value));
$("btnResetPosition").addEventListener("click", () => { $("turnSelect").value = "w"; $("customFenInput").value = STARTING_FEN; setPosition(STARTING_FEN); });
$("orientationSelect").addEventListener("change", () => {
  state.orientation = $("orientationSelect").value;
  state.resolvedOrientation = state.orientation === "b" ? "b" : "w";
  if (state.source === "demo" && state.fen) { drawDemoBoard(state.fen); drawOverlay(); }
  else { resetTracking(); if (state.source) scanFrame(); }
});
$("turnSelect").addEventListener("change", () => {
  const selected = $("turnSelect").value;
  if (state.fen) {
    const fields = state.fen.split(/\s+/); fields[1] = selected;
    // A user correcting the active side intentionally starts a fresh read.
    // Image recognition cannot recover whose turn it is from pixels alone.
    if (state.source === "demo") { analyzeFen(fields.join(" ")); return; }
    const fen = fields.join(" ");
    resetTracking({ keepCorners: true }); state.setupFen = fen;
  }
  refreshAnalysis("Updating the side to move…");
});
for (const [id, property, message] of [["colorToggle", "player", "Updating your playing color…"], ["modeToggle", "mode", "Updating analysis…"]]) {
  $(id).querySelectorAll("button").forEach(button => button.addEventListener("click", () => {
    state[property] = button.dataset.val;
    $(id).querySelectorAll("button").forEach(b => { b.classList.toggle("active", b === button); b.setAttribute("aria-pressed", String(b === button)); });
    refreshAnalysis(message);
  }));
}
$("depthSlider").addEventListener("input", () => {
  state.depth = Number($("depthSlider").value); $("depthVal").textContent = state.depth;
  $("eloHint").textContent = state.depth <= 6 ? "Quick" : state.depth <= 11 ? "Balanced" : "Deeper";
});
$("depthSlider").addEventListener("change", () => refreshAnalysis("Updating analysis depth…"));
$("engineStatus").addEventListener("click", openApiSettings);
$("btnConfigApi").addEventListener("click", openApiSettings);
$("btnCloseApi").addEventListener("click", () => $("apiModal").close());
$("btnResetApiDefault").addEventListener("click", () => { $("apiUrlInput").value = resolveApiBase({ origin: location.origin, fallback: DEFAULT_API }); });
$("btnTestPing").addEventListener("click", async () => {
  try { await checkHealth(resolveApiBase({ origin: location.origin, stored: $("apiUrlInput").value, fallback: DEFAULT_API }), $("modalPingStatus")); }
  catch (err) { $("modalPingStatus").textContent = err.message; }
});
$("btnSaveApi").addEventListener("click", () => {
  try {
    apiBase = resolveApiBase({ origin: location.origin, stored: $("apiUrlInput").value.trim(), fallback: DEFAULT_API });
    localStorage.setItem("chess_api_base", apiBase);
    state.session = crypto.randomUUID();
    if (state.source && state.source !== "demo") state.setupFen = state.fen;
    $("apiModal").close(); checkHealth(); refreshAnalysis("Connecting to the selected engine…");
  } catch (err) { $("modalPingStatus").textContent = err.message; }
});
window.addEventListener("pagehide", () => stopCamera({ clear: false }));
document.addEventListener("visibilitychange", () => {
  if (document.hidden && state.running) { cancelRequest(); clearMove("Camera analysis paused while this tab is hidden."); }
  else if (state.running) schedule(0);
});
clearMove(); updateControls(); checkHealth();
