# Railway deployment

## Deploy the repository root

Use the root `Dockerfile` and root `railway.json`. The root image copies both `backend/` and `frontend/`, installs Stockfish, and runs FastAPI. The service serves the web UI at `/`, API routes at `/api/`, and health at `/health`.

Deploying only `backend/` does not package the current root frontend. The recommended deployment is the whole repository from its root directory.

## Existing project with Railway CLI

From the repository root:

```powershell
railway login
railway link
railway status
railway up --service backend
```

Use your actual service name if it differs from `backend`. The CLI can also select an environment with `--environment production`. A detached deployment returns before it is healthy; verify deployment status and `/health` afterward.

## GitHub deployment

Connect the Railway service to the repository and its intended release branch. Leave the service root directory at the repository root so the root Dockerfile packages the UI and API together. The root Railway configuration defines `/health` as the deployment health check.

The Docker image installs Stockfish and sets its executable path. Railway injects `PORT`; do not hard-code a different port in the service configuration.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `STOCKFISH_PATH` | `/usr/games/stockfish` in the Docker image | Stockfish executable. |
| `PORT` | Supplied by Railway; local Docker fallback `8080` | HTTP listening port. |
| `MIN_INTERVAL_SEC` | `0.5` | Minimum interval between analysis requests for one session. |

Camera sessions are stored in backend memory. A restart requires starting or calibrating tracking again. For multiple replicas, add session affinity or shared state before relying on continuous position tracking across replicas.

## Verify the service

Open `https://YOUR-SERVICE.up.railway.app/health`. A ready Stockfish deployment returns:

```json
{
  "status": "ok",
  "engine": "stockfish",
  "detector": "opencv",
  "recognition": "legal-occupancy-tracking",
  "requires_setup_for_unknown_positions": true,
  "camera_overlay": true
}
```

`engine: "python-chess-fallback"` means the service is available with the weaker fallback engine. Check deployment logs and `STOCKFISH_PATH` if Stockfish is expected. A readiness failure returns HTTP 503.

Then open the HTTPS root URL and use **Try a demo** to verify an engine result and an arrow in the square frame. On a phone, start the camera with a normal starting setup, verify board lock and a suggestion, then move a piece and verify the turn changes. Joining a midgame position requires its exact FEN once; this detector does not independently identify arbitrary piece types.

## Separate frontend hosting

The current frontend has no build step. A static host can publish the `frontend/` directory. Use the UI's **Connection settings** or a `?api=https://YOUR-SERVICE.up.railway.app` URL to select the backend. The Railway-hosted frontend selects its own origin automatically.

Serve phone camera access over HTTPS. The `frontend/_headers` file supplies headers for hosts that support that convention; browser camera permission still comes from the user.
