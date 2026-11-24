# Chess Vision Coach

A phone-friendly chess camera coach with a move arrow directly over your board and a plain-language suggestion such as **Pawn to d4**.

The camera, uploaded photo, detected board boundary, and move arrow share one square view. Start the camera, keep the whole board visible, and the app detects and locks the board automatically. It follows the position as you and your opponent move, and shows advice when it is your turn.

## Board recognition

This version uses OpenCV to find a verified checkerboard and measure piece color and occupancy. It does **not** contain a trained model for identifying arbitrary physical piece types.

- Begin with the pieces in their standard starting positions for automatic calibration.
- To join a game already in progress, enter its exact FEN once under **Position & preferences**, choose the side to move, and select **Use for camera**. The supplied piece identities are then retained through legal move tracking.
- Keep the board in view between moves. The tracker can match an unchanged position, one legal move, or two consecutive legal moves. Unclear frames and ambiguous promotions request a clear frame or position confirmation instead of inventing pieces.

Board orientation and your playing color are separate settings. Orientation is detected automatically; you can also select **White at bottom** or **Black at bottom** regardless of which color you play.

## Run locally

Use Python 3.11+ and install Stockfish for full engine strength. The app also provides a bounded Python fallback when the Stockfish binary is unavailable.

From the repository root:

```powershell
python -m venv .venv
& .venv\Scripts\python.exe -m pip install -r backend/requirements.txt
& .venv\Scripts\python.exe -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080`. FastAPI serves the frontend and API together; no frontend build is required. Phone camera access needs HTTPS when accessing a server on another device.

If Stockfish is outside the standard system paths, set `STOCKFISH_PATH` to its executable path before starting the server.

## Deploy to Railway

Deploy **the repository root**, which contains the Dockerfile that packages both `backend/` and `frontend/`:

```powershell
railway login
railway link
railway up --service backend
```

Open the service's HTTPS domain. The Railway-hosted frontend uses that deployment's own API by default. The Docker image installs Stockfish; Railway supplies `PORT`.

For a separate static frontend, publish `frontend/` and use **Connection settings** or a `?api=https://your-backend.up.railway.app` URL to select its backend.

## Tests

From the repository root:

```powershell
& .venv\Scripts\python.exe -m pip install -r backend/requirements-dev.txt
& .venv\Scripts\python.exe -m pytest backend/tests -q -p no:cacheprovider
npm ci
npm test
npm run test:browser
```

The browser suite starts its own test server and uses installed Chrome on Windows. On Linux, install its browser with `npx playwright install --with-deps chromium`. To include real backend integration, generate synthetic camera fixtures with `python backend/tests/generate_camera_fixtures.py`, start the local backend, and set `CHESS_API_URL=http://127.0.0.1:8080` before running the browser suite. CI runs all three suites and saves mobile/desktop screenshots.

## Documentation

- [Setup and camera workflow](docs/README.md)
- [API reference](docs/API.md)
- [Performance and operating limits](docs/PERFORMANCE.md)
- [Railway deployment](backend/DEPLOY.md)
