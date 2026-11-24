import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

from routers import analysis
from services.stockfish_service import StockfishService
from services.board_detector import BoardDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Chess Vision Coach backend...")
    app.state.stockfish = StockfishService()
    try:
        app.state.stockfish.start()
        logger.info(f"Chess engine started (fallback={app.state.stockfish.is_fallback}).")
    except Exception as e:
        logger.warning(f"Error during Stockfish start: {e}. Fallback active.")

    app.state.detector = BoardDetector()
    try:
        app.state.detector.load()
        logger.info("Board detector loaded.")
    except Exception as e:
        logger.error(f"Error loading BoardDetector: {e}")

    yield
    try:
        app.state.stockfish.stop()
    except Exception:
        pass
    logger.info("Chess engine stopped.")

app = FastAPI(
    title="Chess Vision Coach API",
    version="3.0.0",
    description="Lightweight chess move suggester via camera + Stockfish.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(analysis.router, prefix="/api")

@app.middleware("http")
async def web_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Permissions-Policy"] = "camera=(self)"
    if request.url.path in ("/", "/index.html", "/app.js", "/overlay.js", "/style.css"):
        # HTML, module code and styling must stay on the same deployed version.
        response.headers["Cache-Control"] = "no-cache"
    return response

@app.get("/health")
async def health(request: Request):
    sf = getattr(request.app.state, "stockfish", None)
    engine_name = "stockfish"
    if sf and sf.is_fallback:
        engine_name = "python-chess-fallback"
    detector = getattr(request.app.state, "detector", None)
    ready = bool(sf and sf.is_ready and detector and detector._ready)
    return JSONResponse({
        "status": "ok" if ready else "starting",
        "engine": engine_name,
        "detector": "opencv",
        "recognition": "legal-occupancy-tracking",
        "requires_setup_for_unknown_positions": True,
        "camera_overlay": True,
    }, status_code=200 if ready else 503)

# One source of frontend assets for local development and both Docker entry points.
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.isfile(os.path.join(frontend_dir, "index.html")):
    logger.info(f"Serving static frontend UI from: {frontend_dir}")
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
