from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
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
    version="1.0.0",
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

@app.get("/health")
async def health(request: Request):
    sf = getattr(request.app.state, "stockfish", None)
    engine_name = "stockfish"
    if sf and sf.is_fallback:
        engine_name = "python-chess-fallback"
    return {"status": "ok", "engine": engine_name, "detector": "opencv"}
