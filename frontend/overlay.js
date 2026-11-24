/** Geometry shared by the camera overlay and its regression tests. */
export function fitInside(width, height, size = 1000) {
  const w = width >= height ? size : size * width / height;
  const h = height >= width ? size : size * height / width;
  return { x: (size - w) / 2, y: (size - h) / 2, width: w, height: h };
}

// Project unit board coordinates into the actual camera quadrilateral. A
// bilinear interpolation is insufficient for a tilted/perspective board.
export function projectPoint(corners, x, y) {
  const [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = corners;
  const dx1 = x1 - x2, dx2 = x3 - x2, dx3 = x0 - x1 + x2 - x3;
  const dy1 = y1 - y2, dy2 = y3 - y2, dy3 = y0 - y1 + y2 - y3;
  const det = dx1 * dy2 - dx2 * dy1;
  let g = 0, h = 0;
  if (Math.abs(dx3) + Math.abs(dy3) > 1e-10 && Math.abs(det) > 1e-10) {
    g = (dx3 * dy2 - dx2 * dy3) / det;
    h = (dx1 * dy3 - dx3 * dy1) / det;
  }
  const a = x1 - x0 + g * x1, b = x3 - x0 + h * x3;
  const d = y1 - y0 + g * y1, e = y3 - y0 + h * y3;
  const denominator = g * x + h * y + 1;
  return [(a * x + b * y + x0) / denominator, (d * x + e * y + y0) / denominator];
}

export function squarePosition(square, orientation = "w") {
  if (!/^[a-h][1-8]$/.test(square)) throw new Error("Invalid chess square");
  const col = square.charCodeAt(0) - 97, row = 8 - Number(square[1]);
  return orientation === "b" ? [7 - col, 7 - row] : [col, row];
}

export function moveGeometry(uci, corners, orientation = "w") {
  if (!/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(uci)) return null;
  const from = squarePosition(uci.slice(0, 2), orientation);
  const to = squarePosition(uci.slice(2, 4), orientation);
  const point = (x, y) => projectPoint(corners, x / 8, y / 8);
  return {
    from: point(from[0] + .5, from[1] + .5),
    to: point(to[0] + .5, to[1] + .5),
    target: [[0, 0], [1, 0], [1, 1], [0, 1]].map(([x, y]) => point(to[0] + x, to[1] + y)),
  };
}

export function validCorners(corners) {
  if (!Array.isArray(corners) || corners.length !== 4 || !corners.every(p =>
    Array.isArray(p) && p.length === 2 && p.every(v => Number.isFinite(v) && v >= 0 && v <= 1))) return false;
  const crosses = corners.map((p, i) => {
    const q = corners[(i + 1) % 4], r = corners[(i + 2) % 4];
    return (q[0] - p[0]) * (r[1] - q[1]) - (q[1] - p[1]) * (r[0] - q[0]);
  });
  return crosses.every(v => v > .0001);
}

export function resolveApiBase({ origin, search = "", stored = "", fallback }) {
  const override = new URLSearchParams(search).get("api") || stored;
  if (override) {
    const url = new URL(override);
    if (!["http:", "https:"].includes(url.protocol)) throw new Error("Use an HTTP or HTTPS server URL");
    return url.origin;
  }
  // Railway serves both the app and API: it must use its own deployment.
  if (/\.up\.railway\.app$/.test(new URL(origin).hostname) || /^(localhost|127\.0\.0\.1)$/.test(new URL(origin).hostname)) return origin;
  return fallback;
}
