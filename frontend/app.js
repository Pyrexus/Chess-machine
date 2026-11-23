/**
 * Chess Vision Coach — Frontend Application v2.0
 */

const DEFAULT_API_BASE = "https://backend-production-e51676e6.up.railway.app";
const urlParams        = new URLSearchParams(window.location.search);
let API_BASE           = urlParams.get("api") || localStorage.getItem("chess_api_base") || DEFAULT_API_BASE;

const POLL_INTERVAL    = 2500;
const MAX_CAPTURE_SIZE = 480;
const JPEG_QUALITY     = 0.82;
const SESSION_ID       = "session_" + Math.random().toString(36).slice(2, 9);

const DEPTH_ELO = {
  4: 800, 5: 1000, 6: 1200, 7: 1400, 8: 1600, 9: 1750,
  10: 1900, 11: 2000, 12: 2100, 13: 2200, 14: 2300, 15: 2400, 16: 2600
};

let isRunning       = false;
let pollTimer       = null;
let lastFrameHash   = null;
let playerColor     = "w";
let playMode        = "win";
let depth           = 10;
let currentInputTab = "camera";
const recentMoves   = [];

const $ = id => document.getElementById(id);

// UI Elements
const cameraFeed       = $("cameraFeed");
const captureCanvas    = $("captureCanvas");
const cameraOverlay    = $("cameraOverlay");
const liveIndicator    = $("liveIndicator");
const engineStatus     = $("engineStatus");
const engineBadge      = $("engineBadge");
const btnStart         = $("btnStart");
const btnStop          = $("btnStop");
const depthSlider      = $("depthSlider");
const depthVal         = $("depthVal");
const eloHint          = $("eloHint");

const moveWaiting      = $("moveWaiting");
const moveResult       = $("moveResult");
const moveText         = $("moveText");
const moveSan          = $("moveSan");
const moveUci          = $("moveUci");
const waitText         = $("waitText");
const statusMsg        = $("statusMsg");
const statusBar        = $("statusBar");
const boardCard        = $("boardCard");
const boardImg         = $("boardImg");
const confBadge        = $("confBadge");
const modeInfo         = $("modeInfo");
const recentContainer  = $("recentContainer");
const recentPills      = $("recentPills");

// Mode Tabs & Containers
const cameraContainer  = $("cameraContainer");
const uploadContainer  = $("uploadContainer");
const presetsContainer = $("presetsContainer");
const dropZone         = $("dropZone");
const fileInput        = $("fileInput");
const customFenInput   = $("customFenInput");
const btnAnalyzeFen    = $("btnAnalyzeFen");

// Modal Elements
const apiModal         = $("apiModal");
const apiUrlInput      = $("apiUrlInput");
const modalPingStatus  = $("modalPingStatus");
const btnTestPing      = $("btnTestPing");
const btnResetApiDefault = $("btnResetApiDefault");
const btnSaveApi       = $("btnSaveApi");
const btnConfigApi     = $("btnConfigApi");

window.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initToggleGroup("colorToggle", val => { playerColor = val; });
  initToggleGroup("modeToggle",  val => { playMode = val; updateModeInfo(); });
  initPresets();
  initUpload();
  initModal();

  depthSlider.addEventListener("input", onDepthChange);
  btnStart.addEventListener("click", startAnalysis);
  btnStop.addEventListener("click",  stopAnalysis);
  btnAnalyzeFen.addEventListener("click", () => analyzeDirectFen(customFenInput.value.trim()));

  checkBackendHealth();
  updateModeInfo();
});

// ── Tab Navigation ────────────────────────────────────────────────────────
function initTabs() {
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");

      currentInputTab = btn.dataset.tab;
      cameraContainer.style.display  = currentInputTab === "camera"  ? "block" : "none";
      uploadContainer.style.display  = currentInputTab === "upload"  ? "block" : "none";
      presetsContainer.style.display = currentInputTab === "presets" ? "block" : "none";

      if (currentInputTab !== "camera" && isRunning) {
        stopAnalysis();
      }
    });
  });
}

// ── Quick Test Presets ────────────────────────────────────────────────────
function initPresets() {
  document.querySelectorAll(".preset-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".preset-chip").forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      const fen = chip.dataset.fen;
      const col = chip.dataset.color || "w";
      customFenInput.value = fen;

      // Update color toggle
      const colorBtn = document.querySelector(`#colorToggle button[data-val="${col}"]`);
      if (colorBtn) colorBtn.click();

      analyzeDirectFen(fen);
    });
  });
}

// ── File Upload & Drag/Drop ───────────────────────────────────────────────
function initUpload() {
  fileInput.addEventListener("change", e => {
    const file = e.target.files[0];
    if (file) handleImageFile(file);
  });

  dropZone.addEventListener("dragover", e => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      handleImageFile(file);
    }
  });
}

function handleImageFile(file) {
  const reader = new FileReader();
  reader.onload = e => {
    const dataUrl = e.target.result;
    const b64 = dataUrl.split(",")[1];
    analyzeBase64Image(b64);
  };
  reader.readAsDataURL(file);
}

// ── Toggle Groups ─────────────────────────────────────────────────────────
function initToggleGroup(groupId, onChange) {
  const group = $(groupId);
  group.querySelectorAll(".toggle-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      group.querySelectorAll(".toggle-btn").forEach(b => {
        b.classList.remove("active");
        b.setAttribute("aria-pressed", "false");
      });
      btn.classList.add("active");
      btn.setAttribute("aria-pressed", "true");
      onChange(btn.dataset.val);
    });
  });
}

function onDepthChange() {
  depth = parseInt(depthSlider.value);
  depthVal.textContent = depth;
  eloHint.textContent  = `≈ ${DEPTH_ELO[depth] ?? "?"} Elo*`;
}

// ── Camera Capture & Analysis ─────────────────────────────────────────────
async function startAnalysis() {
  if (isRunning) return;

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } }
    });
    cameraFeed.srcObject = stream;
    await cameraFeed.play();
    cameraOverlay.classList.add("hidden");
    liveIndicator.style.display = "inline-block";
  } catch (err) {
    setStatus(`Camera access error: ${err.message}`, "error");
    return;
  }

  try {
    await fetch(`${API_BASE}/api/reset-session?session_id=${SESSION_ID}`, { method: "POST" });
  } catch (_) { }

  isRunning = true;
  lastFrameHash = null;
  btnStart.disabled = true;
  btnStop.disabled  = false;
  setStatus("Camera live. Analyzing overhead board…", "");
  showWaiting("Scanning camera frame for chess pieces…");

  pollTimer = setInterval(captureAndAnalyze, POLL_INTERVAL);
  captureAndAnalyze();
}

function stopAnalysis() {
  if (!isRunning) return;
  isRunning = false;
  clearInterval(pollTimer);
  pollTimer = null;

  const stream = cameraFeed.srcObject;
  if (stream) stream.getTracks().forEach(t => t.stop());
  cameraFeed.srcObject = null;
  cameraOverlay.classList.remove("hidden");
  liveIndicator.style.display = "none";

  btnStart.disabled = false;
  btnStop.disabled  = true;
  setStatus("Analysis stopped.", "");
}

async function captureAndAnalyze() {
  if (!isRunning) return;
  const vw = cameraFeed.videoWidth;
  const vh = cameraFeed.videoHeight;
  if (!vw || !vh) return;

  const scale = Math.min(MAX_CAPTURE_SIZE / vw, MAX_CAPTURE_SIZE / vh, 1);
  const cw = Math.round(vw * scale);
  const ch = Math.round(vh * scale);

  captureCanvas.width  = cw;
  captureCanvas.height = ch;
  const ctx = captureCanvas.getContext("2d");
  ctx.drawImage(cameraFeed, 0, 0, cw, ch);

  const imageB64Full = captureCanvas.toDataURL("image/jpeg", JPEG_QUALITY);
  const imageB64     = imageB64Full.split(",")[1];

  const frameHash = await quickHash(imageB64.slice(0, 2000));
  if (frameHash === lastFrameHash) return;
  lastFrameHash = frameHash;

  await analyzeBase64Image(imageB64);
}

// ── API Invocation ────────────────────────────────────────────────────────
async function analyzeBase64Image(imageB64) {
  try {
    const t0 = performance.now();
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_b64:    imageB64,
        player_color: playerColor,
        depth:        depth,
        mode:         playMode,
        session_id:   SESSION_ID,
      }),
    });

    const elapsed = Math.round(performance.now() - t0);

    if (!res.ok) {
      setStatus(`Server error ${res.status}`, "error");
      return;
    }

    const data = await res.json();
    handleResponse(data, elapsed);

  } catch (err) {
    setStatus(`Network error: ${err.message}`, "error");
  }
}

async function analyzeDirectFen(fen) {
  if (!fen) return;
  showWaiting("Evaluating position with Stockfish…");
  try {
    const t0 = performance.now();
    const res = await fetch(`${API_BASE}/api/analyze-fen`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        fen:          fen,
        player_color: playerColor,
        depth:        depth,
        mode:         playMode,
      }),
    });

    const elapsed = Math.round(performance.now() - t0);

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      setStatus(`Error: ${errData.message || res.statusText}`, "error");
      return;
    }

    const data = await res.json();
    handleResponse(data, elapsed);

  } catch (err) {
    setStatus(`Network error: ${err.message}`, "error");
  }
}

// ── Response Handler ──────────────────────────────────────────────────────
function handleResponse(data, elapsedMs) {
  switch (data.status) {

    case "ok":
      showMove(data.move_human, data.move_san, data.move_uci);
      const confStr = data.confidence < 1.0 ? ` | Conf: ${pct(data.confidence)}` : "";
      setStatus(`Optimal move computed (${elapsedMs}ms)${confStr} • ≈${data.approx_elo} Elo`, "ok");
      if (data.board_image_b64) {
        updateBoard(data.board_image_b64, data.confidence);
      }
      addRecentMove(data.move_san || data.move_uci);
      break;

    case "no_change":
      setStatus("Board unchanged — waiting for opponent's move…", "");
      break;

    case "opponent_turn":
      setStatus("Opponent's turn detected. Waiting for their move…", "");
      break;

    case "detection_error":
      setStatus(`⚠ ${data.message}`, "error");
      showWaiting("Board not fully visible. Reposition camera overhead.");
      break;

    case "engine_error":
      setStatus("⚙ Engine restarting… retrying automatically.", "");
      break;

    case "rate_limited":
      setStatus("Slowing down analysis pace…", "");
      break;

    case "game_over":
      setStatus("Checkmate, stalemate, or game over.", "ok");
      showWaiting("Game over — no legal moves remaining.");
      break;

    default:
      setStatus(data.message ?? "Unknown response.", "");
  }
}

function showMove(humanText, san, uci) {
  moveWaiting.style.display = "none";
  moveResult.style.display  = "block";
  moveText.textContent      = humanText ?? "—";
  moveSan.textContent       = san ?? uci ?? "";
  moveUci.textContent       = uci ? `${uci.slice(0, 2)} → ${uci.slice(2, 4)}` : "";

  moveResult.classList.remove("move-anim");
  void moveResult.offsetWidth;
  moveResult.classList.add("move-anim");
}

function showWaiting(msg) {
  moveWaiting.style.display = "flex";
  moveResult.style.display  = "none";
  if (msg) waitText.textContent = msg;
}

function setStatus(msg, type = "") {
  statusMsg.textContent = msg;
  statusBar.className   = "status-bar" + (type ? ` ${type}` : "");
}

function updateBoard(b64, confidence) {
  if (!b64) return;
  boardCard.style.display = "block";
  boardImg.src            = `data:image/png;base64,${b64}`;
  confBadge.textContent   = pct(confidence);
}

function addRecentMove(san) {
  if (!san) return;
  if (!recentMoves.includes(san)) {
    recentMoves.unshift(san);
    if (recentMoves.length > 5) recentMoves.pop();

    recentContainer.style.display = "flex";
    recentPills.innerHTML = recentMoves.map(m => `<span class="recent-chip">${m}</span>`).join("");
  }
}

function pct(val) {
  return val != null ? `${Math.round(val * 100)}%` : "—";
}

function updateModeInfo() {
  const modes = {
    win: {
      icon: "🏆",
      title: "Win Mode",
      desc: "Calculates sharp, decisive, high-accuracy moves tailored to win quickly.",
    },
    loss: {
      icon: "🎭",
      title: "Loss Mode",
      desc: "Provides believable, human-like inaccuracies without obvious tactical blunders.",
    },
  };
  const m = modes[playMode] || modes.win;
  modeInfo.innerHTML = `
    <span class="mode-icon">${m.icon}</span>
    <div>
      <strong>${m.title}</strong>
      <p>${m.desc}</p>
    </div>`;
}

// ── Health Check & Engine Status ──────────────────────────────────────────
async function checkBackendHealth() {
  setEngineStatus("connecting", "Connecting…");
  try {
    const t0 = performance.now();
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
    const latency = Math.round(performance.now() - t0);
    if (res.ok) {
      const data = await res.json().catch(() => ({}));
      const engineName = data.engine === "stockfish" ? "Stockfish 16" : "Python Engine";
      setEngineStatus("online", `Online (${latency}ms)`);
      if (engineBadge) engineBadge.textContent = engineName;
    } else {
      setEngineStatus("offline", `Server Error (${res.status})`);
    }
  } catch {
    setEngineStatus("offline", "Backend Offline");
  }
}

function setEngineStatus(cls, label) {
  engineStatus.className = `status-pill ${cls}`;
  engineStatus.querySelector(".status-label").textContent = label;
  engineStatus.title = `API: ${API_BASE} (Click to configure)`;
}

// ── API Configuration Modal ───────────────────────────────────────────────
function initModal() {
  engineStatus.addEventListener("click", openApiModal);
  if (btnConfigApi) btnConfigApi.addEventListener("click", openApiModal);

  btnTestPing.addEventListener("click", async () => {
    const targetUrl = apiUrlInput.value.trim().replace(/\/+$/, "");
    modalPingStatus.textContent = "Testing connection…";
    try {
      const t0 = performance.now();
      const res = await fetch(`${targetUrl}/health`, { signal: AbortSignal.timeout(5000) });
      const ms = Math.round(performance.now() - t0);
      if (res.ok) {
        modalPingStatus.textContent = `✓ Connected! Response in ${ms}ms`;
        modalPingStatus.style.color = "var(--accent)";
      } else {
        modalPingStatus.textContent = `✗ HTTP Error ${res.status}`;
        modalPingStatus.style.color = "var(--danger)";
      }
    } catch (e) {
      modalPingStatus.textContent = `✗ Unreachable: ${e.message}`;
      modalPingStatus.style.color = "var(--danger)";
    }
  });

  btnResetApiDefault.addEventListener("click", () => {
    apiUrlInput.value = DEFAULT_API_BASE;
  });

  btnSaveApi.addEventListener("click", () => {
    const val = apiUrlInput.value.trim().replace(/\/+$/, "");
    if (val) {
      API_BASE = val;
      localStorage.setItem("chess_api_base", API_BASE);
      apiModal.close();
      checkBackendHealth();
    }
  });
}

function openApiModal() {
  apiUrlInput.value = API_BASE;
  modalPingStatus.textContent = "Click 'Test Connection' to verify server reachability.";
  modalPingStatus.style.color = "var(--text-dim)";
  apiModal.showModal();
}

async function quickHash(str) {
  if (!crypto?.subtle) return str.slice(0, 64);
  const buf = await crypto.subtle.digest("SHA-1", new TextEncoder().encode(str));
  return Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2, "0")).join("");
}
