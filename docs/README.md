# Setup and camera workflow

## How the app works

The frontend captures the complete camera image into a square canvas with letterboxing rather than cropping away board corners. The backend detects the board quadrilateral, verifies its alternating 8-by-8 pattern, and measures piece color and occupied squares. The returned corners and move coordinates are normalized to that same captured image.

The frontend uses a perspective transform to draw the move arrow over the live camera or uploaded photo. It does not open a separate board-overlay window.

Piece identity comes from a known position: either the normal opening setup or an exact FEN supplied by the user. Legal move matching then preserves kings, queens, other piece types, side to move, castling rights, and en passant information. The detector never repairs an unreadable image by adding kings, changing pieces to queens, or deleting attacking pieces.

## Start a new game

1. Arrange the pieces in their normal starting positions.
2. Select your playing color and the color at the bottom of the camera view.
3. Select **Start camera** and allow camera access in the browser.
4. Keep all 64 squares visible, preferably from overhead with even light.
5. Once the board is locked, follow the arrow and the move written below the camera view.

After you move, the app watches for your opponent's reply. An unchanged board retains the current suggestion; a detected opponent turn clears it. Keep hands clear while a new position is being read.

## Analyze an existing game

Open **Position & preferences**, enter the exact current FEN, choose the side to move, and select **Use for camera** while the camera or photo is visible. The backend checks that the observed piece colors and occupancy match that setup before starting tracking. Piece types cannot be verified from color and occupancy alone, so the FEN must describe the real board accurately.

**Analyze FEN** evaluates a position directly and displays an example board in the same square frame. **Try a demo** provides a starting-position example without camera access.

The vision pipeline does not independently identify arbitrary midgame pieces. It needs calibration when joining a game midway, after more than two moves are missed, or when several legal positions have identical observed occupancy, such as different promotion choices.

## Board alignment

Automatic alignment verifies the board pattern instead of treating any rectangle as a chessboard. When automatic alignment is difficult, select **Set corners** and tap the board's top-left, top-right, bottom-right, then bottom-left corner. Crossed or degenerate corners are rejected. **Unlock** clears the calibration and scans again.

Orientation describes the camera view, independently of whose turn it is or which color you play. Turning the camera to the opposite side requires changing the orientation and recalibrating. A partial board, glare, low contrast, or a hand over pieces can cause the app to withhold a suggestion until a readable frame returns.

## Local development

From the repository root on Windows:

```powershell
python -m venv .venv
& .venv\Scripts\python.exe -m pip install -r backend/requirements.txt
& .venv\Scripts\python.exe -m uvicorn main:app --app-dir backend --reload --host 127.0.0.1 --port 8080
```

On macOS or Linux, use `.venv/bin/python` in place of `.venv\Scripts\python.exe`. Open `http://127.0.0.1:8080` for the app and `/docs` for interactive API documentation.

Install Stockfish through your system package manager or [the official download page](https://stockfishchess.org/download/). Set `STOCKFISH_PATH` if the executable is not discoverable. Health and move responses distinguish Stockfish from the weaker Python fallback.

The frontend is plain HTML, CSS, and JavaScript modules. No frontend compilation is needed. Serve it through FastAPI to use the local API automatically. If a separate frontend server runs on another port, use its URL with `?api=http://127.0.0.1:8080` or select the API in **Connection settings**.

## Project layout

```text
frontend/
  index.html        Mobile-first camera UI
  style.css         Responsive styles
  app.js            Camera, requests, tracking UI, and overlay rendering
  overlay.js        Perspective and square-coordinate helpers
  tests/            Frontend regression tests
  _headers          Headers for separate static hosting
backend/
  main.py           FastAPI startup, health, and static frontend serving
  routers/analysis.py
  services/board_detector.py
  services/stockfish_service.py
  tests/
  requirements.txt
Dockerfile          Root deployment: backend and frontend together
railway.json        Railway build and health-check configuration
```

## Deployment and checks

See [Railway deployment](../backend/DEPLOY.md) for the recommended single-service deployment. A separate static host can publish `frontend/` with no build command and select the Railway API through connection settings.

```powershell
& .venv\Scripts\python.exe -m pytest backend/tests -q -p no:cacheprovider
npm install
npm test
npm run test:browser
```

See [API reference](API.md) for request and response details and [performance notes](PERFORMANCE.md) for timing and tracking limits.
