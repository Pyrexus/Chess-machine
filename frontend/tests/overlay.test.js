import test from "node:test";
import assert from "node:assert/strict";
import { fitInside, projectPoint, moveGeometry, validCorners, resolveApiBase, frameDetailScore } from "../overlay.js";

test("landscape and portrait camera fit matches the square capture letterbox", () => {
  assert.deepEqual(fitInside(1920, 1080), { x: 0, y: 218.75, width: 1000, height: 562.5 });
  assert.deepEqual(fitInside(1080, 1920), { x: 218.75, y: 0, width: 562.5, height: 1000 });
});

test("perspective projection maps all four actual board corners exactly", () => {
  const corners = [[.12, .18], [.84, .25], [.97, .9], [.05, .95]];
  [[0, 0], [1, 0], [1, 1], [0, 1]].forEach(([x, y], i) => {
    const p = projectPoint(corners, x, y);
    assert.ok(Math.abs(p[0] - corners[i][0]) < 1e-10);
    assert.ok(Math.abs(p[1] - corners[i][1]) < 1e-10);
  });
});

test("e2 to e4 points to the correct squares for either board orientation", () => {
  const corners = [[0, 0], [1, 0], [1, 1], [0, 1]];
  assert.deepEqual(moveGeometry("e2e4", corners).from, [.5625, .8125]);
  assert.deepEqual(moveGeometry("e2e4", corners).to, [.5625, .5625]);
  assert.deepEqual(moveGeometry("e2e4", corners, "b").from, [.4375, .1875]);
  assert.deepEqual(moveGeometry("e2e4", corners, "b").to, [.4375, .4375]);
  assert.deepEqual(moveGeometry("e7e8q", corners).to, [.5625, .0625]);
  assert.equal(moveGeometry("bad", corners), null);
});

test("crossed, out-of-frame and degenerate manual locks are rejected", () => {
  assert.equal(validCorners([[.1, .1], [.9, .1], [.9, .9], [.1, .9]]), true);
  assert.equal(validCorners([[0, 0], [1, 1], [1, 0], [0, 1]]), false);
  assert.equal(validCorners([[0, 0], [1, 0], [1, 0], [0, 0]]), false);
  assert.equal(validCorners([[0, 0], [2, 0], [1, 1], [0, 1]]), false);
});

test("Railway and local apps use their own API; Pages and overrides resolve safely", () => {
  const fallback = "https://backend.up.railway.app";
  assert.equal(resolveApiBase({ origin: "https://preview.up.railway.app", fallback }), "https://preview.up.railway.app");
  assert.equal(resolveApiBase({ origin: "http://127.0.0.1:8080", fallback }), "http://127.0.0.1:8080");
  assert.equal(resolveApiBase({ origin: "https://chess.pages.dev", fallback }), fallback);
  assert.equal(resolveApiBase({ origin: "https://chess.pages.dev", search: "?api=https://other.example/", fallback }), "https://other.example");
  assert.throws(() => resolveApiBase({ origin: "https://chess.pages.dev", stored: "javascript:bad", fallback }));
});

test("burst selection favors detailed camera frames without rejecting flat frames", () => {
  const width = 16, height = 16, sharp = new Uint8ClampedArray(width * height * 4), flat = sharp.slice();
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const index = (y * width + x) * 4;
    for (let c = 0; c < 3; c++) {
      flat[index + c] = 128;
      sharp[index + c] = (x + y) % 2 ? 240 : 20;
    }
    flat[index + 3] = sharp[index + 3] = 255;
  }
  assert.ok(frameDetailScore(flat, width, height) < 1e-10);
  assert.ok(frameDetailScore(sharp, width, height) > 1000);
  assert.equal(frameDetailScore(flat, 1, 1), 0);
});
