import test, { before, after } from "node:test";
import assert from "node:assert/strict";
import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = fileURLToPath(new URL("../", import.meta.url));
const repo = path.dirname(root);
const startFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const corners = [[.1, .15], [.85, .2], [.9, .9], [.08, .87]];
const advice = { status: "ok", fen: startFen, turn: "w", move_uci: "d2d4", move_san: "d4", move_human: "Pawn to d4",
  confidence: .95, board_corners: corners, board_detected: true, board_locked: true, pieces_detected: true,
  orientation: "w", engine: "stockfish" };
let browser, server, origin;
before(async () => {
  server = createServer(async (req, res) => {
    try {
      if (req.url === "/health") { res.setHeader("Content-Type", "application/json"); res.end(JSON.stringify({ status: "ok", engine: "stockfish" })); return; }
      const file = req.url === "/" ? "index.html" : req.url.split("?")[0].slice(1);
      if (!/^(index\.html|style\.css|app\.js|overlay\.js)$/.test(file)) { res.writeHead(404); res.end(); return; }
      res.setHeader("Content-Type", file.endsWith(".js") ? "text/javascript" : file.endsWith(".css") ? "text/css" : "text/html");
      res.end(await readFile(path.join(root, file)));
    } catch { res.writeHead(500); res.end(); }
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  origin = `http://127.0.0.1:${server.address().port}`;
  const chrome = process.env.CHROME_PATH || (process.platform === "win32" ? "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" : undefined);
  browser = await chromium.launch({ ...(chrome && existsSync(chrome) ? { executablePath: chrome } : {}), headless: true });
});
after(async () => { await browser?.close(); await new Promise(resolve => server?.close(resolve)); });

async function newPage(viewport = { width: 390, height: 844 }) {
  const page = await browser.newPage({ viewport });
  const errors = [];
  page.on("pageerror", err => errors.push(err.message));
  page.errors = errors;
  return page;
}
async function respond(route, data = advice) { await route.fulfill({ json: data }); }
async function photo(page) {
  const b64 = await page.evaluate(() => {
    const canvas = document.createElement("canvas"); canvas.width = 800; canvas.height = 450;
    const ctx = canvas.getContext("2d"); ctx.fillStyle = "#e2edde"; ctx.fillRect(0, 0, 800, 450);
    return canvas.toDataURL("image/png").split(",")[1];
  });
  await page.locator("#fileInput").setInputFiles({ name: "board.png", mimeType: "image/png", buffer: Buffer.from(b64, "base64") });
}
async function fakeCamera(page, delay = 0) {
  await page.addInitScript(({ delay }) => {
    navigator.mediaDevices.getUserMedia = async () => {
      await new Promise(resolve => setTimeout(resolve, delay));
      const canvas = document.createElement("canvas"); canvas.width = 640; canvas.height = 480;
      const ctx = canvas.getContext("2d"); ctx.fillStyle = "#cbdac8"; ctx.fillRect(0, 0, 640, 480);
      const stream = canvas.captureStream(5);
      window.fakeStream = stream;
      const draw = setInterval(() => ctx.fillRect(0, 0, 640, 480), 200);
      stream.getTracks()[0].addEventListener("ended", () => clearInterval(draw));
      return stream;
    };
  }, { delay });
}

test("mobile and desktop layouts have one square frame, no overflow, and usable connection dialog", async () => {
  for (const viewport of [{ width: 360, height: 800 }, { width: 390, height: 844 }, { width: 768, height: 1024 }, { width: 1440, height: 1000 }]) {
    const page = await newPage(viewport);
    await page.goto(origin);
    await page.getByText("Engine online", { exact: true }).waitFor();
    const box = await page.locator("#cameraFrame").boundingBox();
    assert.ok(Math.abs(box.width - box.height) < 1);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    assert.equal(await page.locator("#boardCard").count(), 0);
    await page.locator("#engineStatus").click();
    assert.equal(await page.locator("#apiModal").isVisible(), true);
    await page.locator("#btnCloseApi").click();
    assert.deepEqual(page.errors, []);
    await mkdir(path.join(repo, ".tmp", "screens"), { recursive: true });
    await page.screenshot({ path: path.join(repo, ".tmp", "screens", `ui-${viewport.width}.png`), fullPage: true });
    await page.close();
  }
});

test("photo overlay uses the camera frame and cached suggestions remain visible", async () => {
  const page = await newPage(); let requests = 0;
  await page.route("**/api/analyze", route => respond(route, { ...advice, status: ++requests === 1 ? "ok" : "no_change" }));
  await page.goto(origin); await photo(page);
  await page.locator("#moveResult").waitFor();
  assert.equal(await page.locator("#moveText").textContent(), "Pawn to d4");
  assert.equal(await page.locator("#moveOverlay").getAttribute("aria-label"), "Suggested move from d2 to d4");
  assert.equal(await page.locator("#frameMoveLabel").isVisible(), true);
  const frame = await page.locator("#cameraFrame").boundingBox(), overlay = await page.locator("#moveOverlay").boundingBox();
  assert.deepEqual(overlay, frame);
  await page.locator("#depthSlider").evaluate(el => { el.value = "11"; el.dispatchEvent(new Event("input")); el.dispatchEvent(new Event("change")); });
  await page.getByText(/Position unchanged/).waitFor();
  assert.equal(await page.locator("#moveOverlay line").count(), 2);
  assert.deepEqual(page.errors, []); await page.close();
});

test("camera polls every settled frame, clears opponent arrow, and resumes suggestions", async () => {
  const page = await newPage(); await fakeCamera(page); let requests = 0, active = 0, maxActive = 0;
  await page.route("**/api/analyze", async route => {
    active++; maxActive = Math.max(maxActive, active); requests++;
    assert.equal(route.request().postDataJSON().orientation, "auto");
    await new Promise(resolve => setTimeout(resolve, 150));
    const data = requests === 1 ? advice : requests === 2 ? { ...advice, status: "no_change" } : requests === 3 ?
      { ...advice, status: "opponent_turn", move_uci: null, turn: "b" } : { ...advice, move_uci: "g1f3", move_san: "Nf3", move_human: "Knight to f3" };
    await respond(route, data); active--;
  });
  await page.goto(origin); await page.locator("#btnStart").click();
  await page.locator("#moveResult").waitFor();
  await page.getByText(/Position unchanged/).waitFor();
  await page.locator("#waitText").filter({ hasText: "Opponent’s turn" }).waitFor();
  assert.equal(await page.locator("#moveOverlay line").count(), 0);
  await page.locator("#moveText").filter({ hasText: "Knight to f3" }).waitFor();
  assert.ok(requests >= 4); assert.equal(maxActive, 1);
  await page.locator("#btnStop").click();
  assert.equal(await page.locator("#moveOverlay line").count(), 0);
  assert.deepEqual(page.errors, []); await page.close();
});

test("a stopped camera cannot display an in-flight result or reopen after delayed permission", async () => {
  const page = await newPage(); await fakeCamera(page); let started;
  const requestStarted = new Promise(resolve => { started = resolve; });
  await page.route("**/api/analyze", async route => {
    started(); await new Promise(resolve => setTimeout(resolve, 1000));
    try { await respond(route); } catch { /* Request was canceled by stop. */ }
  });
  await page.goto(origin); await page.locator("#btnStart").click(); await requestStarted;
  await page.locator("#btnStop").click();
  await page.waitForTimeout(1200);
  assert.equal(await page.locator("#moveResult").isVisible(), false);
  assert.equal(await page.locator("#moveOverlay line").count(), 0);
  await page.close();
  const delayed = await newPage(); await fakeCamera(delayed, 500);
  await delayed.goto(origin); await delayed.locator("#btnStart").click(); await delayed.locator("#btnStop").click();
  await delayed.waitForTimeout(700);
  assert.equal(await delayed.locator("#cameraFeed").isVisible(), false);
  assert.equal(await delayed.evaluate(() => fakeStream.getTracks()[0].readyState), "ended");
  assert.deepEqual(delayed.errors, []); await delayed.close();
});

test("unclear position removes stale arrows while retaining the detected board lock", async () => {
  const page = await newPage(); let requests = 0;
  await page.route("**/api/analyze", route => respond(route, ++requests === 1 ? advice :
    { ...advice, status: "position_uncertain", fen: null, move_uci: null, message: "Remove your hand." }));
  await page.goto(origin); await photo(page); await page.locator("#moveResult").waitFor();
  await page.locator("#depthSlider").evaluate(el => { el.value = "12"; el.dispatchEvent(new Event("input")); el.dispatchEvent(new Event("change")); });
  await page.getByText("Remove your hand.", { exact: true }).waitFor();
  assert.equal(await page.locator("#moveOverlay line").count(), 0);
  assert.equal(await page.locator("#moveOverlay polygon").count(), 1);
  assert.equal(await page.locator("#moveResult").isVisible(), false);
  assert.deepEqual(page.errors, []); await page.close();
});

test("auto black orientation rotates the arrow and an empty board is never locked", async () => {
  const page = await newPage(); let requests = 0;
  await page.route("**/api/analyze", route => respond(route, ++requests === 1 ?
    { ...advice, board_corners: [[0, 0], [1, 0], [1, 1], [0, 1]], orientation: "b" } :
    { ...advice, status: "setup_required", fen: null, move_uci: null, pieces_detected: false, board_locked: false }));
  await page.goto(origin); await photo(page); await page.locator("#moveResult").waitFor();
  const line = page.locator("#moveOverlay line").last();
  assert.equal(Number(await line.getAttribute("x1")), 562.5);
  assert.equal(Number(await line.getAttribute("y1")), 187.5);
  await page.locator("#depthSlider").evaluate(el => { el.value = "12"; el.dispatchEvent(new Event("input")); el.dispatchEvent(new Event("change")); });
  await page.locator("#waitText").filter({ hasText: "Put the pieces" }).waitFor();
  assert.equal(await page.locator("#moveOverlay polygon").count(), 0);
  assert.equal(await page.locator("#frameMoveLabel").isVisible(), false);
  assert.deepEqual(page.errors, []); await page.close();
});

test("denied camera permission leaves controls usable and explains photo fallback", async () => {
  const page = await newPage();
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = async () => { throw new DOMException("Permission denied", "NotAllowedError"); };
  });
  await page.goto(origin); await page.locator("#btnStart").click();
  await page.locator("#statusMsg").filter({ hasText: "Camera access was denied" }).waitFor();
  assert.equal(await page.locator("#btnStart").isEnabled(), true);
  assert.equal(await page.locator("#btnStop").isEnabled(), false);
  assert.equal(await page.locator("#btnUploadPhoto").isEnabled(), true);
  assert.deepEqual(page.errors, []); await page.close();
});

test("settings during camera permission do not strand Start and stale denial cannot cancel a photo", async () => {
  const page = await newPage(); await fakeCamera(page, 500);
  await page.route("**/api/analyze", route => respond(route));
  await page.goto(origin); await page.locator("#btnStart").click();
  await page.locator('#colorToggle button[data-val="b"]').click();
  await page.locator("#cameraFeed").waitFor(); await page.locator("#moveResult").waitFor();
  assert.equal(await page.locator("#btnStop").isEnabled(), true);
  await page.locator("#btnStop").click();
  assert.equal(await page.locator("#btnStart").isEnabled(), true);
  await page.close();
  const denied = await newPage();
  await denied.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = async () => {
      await new Promise(resolve => setTimeout(resolve, 600));
      throw new DOMException("Permission denied", "NotAllowedError");
    };
  });
  await denied.route("**/api/analyze", route => respond(route));
  await denied.goto(origin); await denied.locator("#btnStart").click(); await photo(denied);
  await denied.locator("#moveResult").waitFor(); await denied.waitForTimeout(800);
  assert.equal(await denied.locator("#moveResult").isVisible(), true);
  assert.equal(await denied.locator("#photoPreview").isVisible(), true);
  assert.equal(await denied.locator("#moveText").textContent(), "Pawn to d4");
  assert.deepEqual(denied.errors, []); await denied.close();
});

test("Try demo switches an active scan and static suggestions survive tab visibility changes", async () => {
  const page = await newPage(); await fakeCamera(page); let started;
  const requestStarted = new Promise(resolve => { started = resolve; });
  await page.route("**/api/analyze", async route => { started(); await new Promise(resolve => setTimeout(resolve, 700)); try { await respond(route); } catch {} });
  await page.route("**/api/analyze-fen", route => respond(route, { ...advice, board_corners: undefined }));
  await page.goto(origin); await page.locator("#btnStart").click(); await requestStarted;
  await page.locator("#btnDemo").click(); await page.locator("#moveResult").waitFor();
  assert.equal(await page.locator("#cameraFeed").isVisible(), false);
  assert.equal(await page.locator("#photoPreview").isVisible(), true);
  await page.evaluate(() => {
    Object.defineProperty(document, "hidden", { configurable: true, get: () => true });
    document.dispatchEvent(new Event("visibilitychange"));
    Object.defineProperty(document, "hidden", { configurable: true, get: () => false });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  assert.equal(await page.locator("#moveResult").isVisible(), true);
  assert.equal(await page.locator("#moveOverlay line").count(), 2);
  assert.deepEqual(page.errors, []); await page.close();
});

test("live backend photo reads return an overlay and a legal next move", { skip: !process.env.CHESS_API_URL }, async () => {
  const page = await newPage();
  await page.goto(process.env.CHESS_API_URL);
  await page.locator("#fileInput").setInputFiles(path.join(repo, ".tmp", "camera-start.png"));
  await page.locator("#moveResult").waitFor();
  assert.equal(await page.locator("#boardStatus").textContent(), "Board locked");
  assert.equal(await page.locator("#moveOverlay line").count(), 2);
  assert.match(await page.locator("#moveUci").textContent(), /^[a-h][1-8] → [a-h][1-8]$/);
  await page.screenshot({ path: path.join(repo, ".tmp", "screens", "live-overlay-mobile.png"), fullPage: true });
  await page.locator("#btnDemo").click(); await page.locator("#moveResult").waitFor();
  assert.equal(await page.locator("#moveOverlay line").count(), 2);
  assert.deepEqual(page.errors, []); await page.close();
});
