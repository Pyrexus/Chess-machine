"""Camera sessions keep a trusted chess position and reusable move advice."""

import asyncio
import base64
import binascii
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import chess
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from services.stockfish_service import DEPTH_ELO_MAP, validate_fen

logger = logging.getLogger(__name__)
router = APIRouter()
MIN_INTERVAL_SEC = max(0.0, float(os.getenv("MIN_INTERVAL_SEC", "0.5")))
SESSION_TTL_SEC = 30 * 60
MAX_SESSIONS = 512
MAX_IMAGE_BYTES = 6 * 1024 * 1024


@dataclass
class CameraSession:
    previous_fen: Optional[str] = None
    cache_key: Optional[tuple] = None
    result: Optional[dict] = None
    last_request: float = 0.0
    last_seen: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_sessions: dict[str, CameraSession] = {}


class AnalyzeRequest(BaseModel):
    image_b64: Optional[str] = Field(default=None, max_length=MAX_IMAGE_BYTES * 4 // 3 + 128)
    fen: Optional[str] = Field(default=None, max_length=128)
    setup_fen: Optional[str] = Field(default=None, max_length=128)
    player_color: Literal["w", "b"] = "w"
    orientation: Literal["w", "b", "auto"] = "auto"
    side_to_move: Optional[Literal["w", "b"]] = None
    board_corners: Optional[list[list[float]]] = None
    depth: int = Field(default=10, ge=4, le=16)
    mode: Literal["win", "loss"] = "win"
    session_id: str = Field(default="default", min_length=1, max_length=128)

    @field_validator("board_corners")
    @classmethod
    def valid_corners(cls, value):
        if value is not None:
            if len(value) != 4 or any(len(point) != 2 for point in value):
                raise ValueError("board_corners must contain four [x, y] points.")
            if any(not np.isfinite(v) or not 0 <= v <= 1 for point in value for v in point):
                raise ValueError("Board corner coordinates must be normalized between 0 and 1.")
            # Screen coordinates are clockwise: TL, TR, BR, BL. Crossed or
            # degenerate quadrilaterals cannot define a perspective overlay.
            points = np.asarray(value, dtype=np.float64)
            edges = np.roll(points, -1, axis=0) - points
            following = np.roll(edges, -1, axis=0)
            cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
            if not (cross > 1e-6).all():
                raise ValueError("Board corners must form a convex clockwise quadrilateral: top-left, top-right, bottom-right, bottom-left.")
        return value


class AnalyzeFenRequest(BaseModel):
    fen: str = Field(max_length=128)
    player_color: Optional[Literal["w", "b"]] = None
    depth: int = Field(default=10, ge=4, le=16)
    mode: Literal["win", "loss"] = "win"


def _session(session_id: str) -> CameraSession:
    now = time.monotonic()
    for key, value in list(_sessions.items()):
        if now - value.last_seen > SESSION_TTL_SEC and not value.lock.locked():
            _sessions.pop(key, None)
    if session_id not in _sessions:
        if len(_sessions) >= MAX_SESSIONS:
            idle = [(value.last_seen, key) for key, value in _sessions.items() if not value.lock.locked()]
            if not idle:
                raise HTTPException(status_code=503, detail="All analysis sessions are busy. Please retry.")
            _sessions.pop(min(idle)[1], None)
        _sessions[session_id] = CameraSession()
    session = _sessions[session_id]
    session.last_seen = now
    return session


def _decode_image(encoded: str) -> bytes:
    if encoded.startswith("data:"):
        if ";base64," not in encoded:
            raise HTTPException(status_code=400, detail="Invalid image data URL.")
        encoded = encoded.split(",", 1)[1]
    try:
        image = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise HTTPException(status_code=400, detail="Invalid base64 image.")
    if not image:
        raise HTTPException(status_code=400, detail="The image is empty.")
    if len(image) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large. Use an image under 6 MB.")
    return image


def _position_key(fen: str) -> tuple[str, ...]:
    # Turn, castling, en passant and the halfmove clock affect legal play and
    # automatic draw outcomes even when the piece placement is unchanged.
    return tuple(chess.Board(fen).fen().split()[:5])


def _fen_equal(a, b):
    if not a or not b:
        return False
    try:
        return _position_key(a) == _position_key(b)
    except ValueError:
        return False


def _overlay(fen: str, uci: Optional[str], corners, orientation: str) -> Optional[dict]:
    if not uci:
        return None
    move = chess.Move.from_uci(uci)
    board = chess.Board(fen)
    piece = board.piece_at(move.from_square)
    result = {
        "from_square": chess.square_name(move.from_square),
        "to_square": chess.square_name(move.to_square),
        "piece": chess.piece_name(piece.piece_type) if piece else None,
        "promotion": chess.piece_name(move.promotion) if move.promotion else None,
        "orientation": orientation,
    }
    if corners is None:
        return result
    transform = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), np.float32(corners)
    )

    def square_points(square):
        col, row = chess.square_file(square), 7 - chess.square_rank(square)
        if orientation == "b":
            col, row = 7 - col, 7 - row
        return np.float32([
            [(col + 0.5) / 8, (row + 0.5) / 8],
            [col / 8, row / 8], [(col + 1) / 8, row / 8],
            [(col + 1) / 8, (row + 1) / 8], [col / 8, (row + 1) / 8],
        ])

    for prefix, square in [("from", move.from_square), ("to", move.to_square)]:
        points = cv2.perspectiveTransform(square_points(square)[None, :, :], transform)[0]
        result[prefix] = points[0].round(6).tolist()
        result[prefix + "_polygon"] = points[1:].round(6).tolist()
    return result


def _geometry(detection: Optional[dict], body: AnalyzeRequest) -> dict:
    detection = detection or {}
    return {
        "board_detected": bool(detection.get("board_detected", detection.get("fen"))),
        "board_corners": detection.get("board_corners", body.board_corners),
        "orientation": detection.get("orientation", body.orientation if body.orientation != "auto" else "w"),
        "requires_setup": bool(detection.get("requires_setup", False)),
        "piece_count": detection.get("piece_count"),
        "pieces_detected": bool(detection.get("pieces_detected", detection.get("fen"))),
    }


def _attach_geometry(result: dict, geometry: dict) -> dict:
    result = {**result, **geometry}
    result["board_locked"] = bool(result["board_detected"] and result["pieces_detected"] and result["board_corners"])
    result["overlay"] = _overlay(
        result["fen"], result.get("move_uci"), result["board_corners"], result["orientation"]
    ) if result.get("fen") else None
    return result


async def _engine_result(sf, fen: str, depth: int, mode: str, confidence: float) -> dict:
    board = chess.Board(fen)
    turn = "w" if board.turn else "b"
    outcome = board.outcome()
    if outcome is not None:
        message = "Checkmate." if board.is_checkmate() else "Stalemate." if board.is_stalemate() else "Draw: " + outcome.termination.name.replace("_", " ").lower() + "."
        return {"status": "game_over", "message": message, "fen": fen, "turn": turn,
                "confidence": confidence, "move_uci": None, "move_san": None, "move_human": None,
                "board_image_b64": None, "approx_elo": 0}
    info = await run_in_threadpool(sf.get_best_move, fen, depth=depth, mode=mode)
    if not info.get("uci"):
        raise RuntimeError("Engine returned no move for a position with legal moves.")
    move = chess.Move.from_uci(info["uci"])
    if move not in board.legal_moves:
        raise RuntimeError("Engine returned an illegal move.")
    return {
        "status": "ok", "fen": fen, "turn": turn,
        "move_uci": info["uci"], "move_san": info["san"], "move_human": info["human"],
        "confidence": confidence, "approx_elo": info["elo"], "engine": info.get("engine", "stockfish"),
        "board_image_b64": None,
    }


@router.post("/analyze-fen")
async def analyze_fen(request: Request, body: AnalyzeFenRequest):
    """A FEN is authoritative about whose turn it is; player color does not rewrite it."""
    valid, error = validate_fen(body.fen)
    if not valid:
        return JSONResponse({"status": "invalid_fen", "message": error}, status_code=400)
    try:
        return await _engine_result(request.app.state.stockfish, body.fen, body.depth, body.mode, 1.0)
    except Exception:
        logger.exception("Engine failed while analyzing FEN")
        return JSONResponse({"status": "engine_error", "message": "Analysis unavailable. Please retry."}, status_code=503)


@router.post("/analyze")
async def analyze(request: Request, body: AnalyzeRequest):
    if not body.fen and not body.image_b64:
        raise HTTPException(status_code=400, detail="Either image_b64 or fen must be provided.")
    session = _session(body.session_id)
    async with session.lock:
        now = time.monotonic()
        if now - session.last_request < MIN_INTERVAL_SEC:
            return JSONResponse({"status": "rate_limited", "message": "Waiting for the next camera frame.",
                                 "retry_after": MIN_INTERVAL_SEC}, status_code=429,
                                headers={"Retry-After": str(max(1, int(MIN_INTERVAL_SEC)))})
        session.last_request = now
        detection = None
        confidence = 1.0
        if body.fen:
            fen = body.fen
        else:
            image = _decode_image(body.image_b64)
            setup_fen = body.setup_fen
            if setup_fen and body.side_to_move:
                try:
                    setup_board = chess.Board(setup_fen)
                    setup_board.turn = body.side_to_move == "w"
                    setup_fen = setup_board.fen()
                except ValueError:
                    pass  # Validation below gives the readable FEN error.
            if setup_fen:
                valid, error = validate_fen(setup_fen)
                if not valid:
                    return JSONResponse({"status": "invalid_fen", "message": error}, status_code=400)
            try:
                detection = await run_in_threadpool(
                    request.app.state.detector.detect, image, player_color=body.player_color,
                    orientation=body.orientation, previous_fen=session.previous_fen,
                    setup_fen=setup_fen, board_corners=body.board_corners,
                )
            except Exception:
                logger.exception("Board detector failed")
                return JSONResponse({"status": "detection_error", "message": "Could not read this frame. Keep the full board visible.",
                                     "confidence": 0.0, "board_detected": False, "board_locked": False, "overlay": None})
            confidence = detection.get("confidence", 0.0)
            geometry = _geometry(detection, body)
            if detection.get("error") or not detection.get("fen"):
                return _attach_geometry({
                    "status": detection.get("status", "detection_error"),
                    "message": detection.get("error") or "Set the current position to start tracking.",
                    "confidence": confidence, "fen": None, "move_uci": None, "move_human": None,
                }, geometry)
            fen = detection["fen"]
        geometry = _geometry(detection, body)
        valid, error = validate_fen(fen)
        if not valid:
            return JSONResponse(_attach_geometry({"status": "invalid_fen" if body.fen else "detection_error",
                                                 "message": error, "confidence": confidence, "fen": None}, geometry),
                                status_code=400 if body.fen else 200)
        # Trusted position tracking is independent of the engine result cache.
        session.previous_fen = fen
        board = chess.Board(fen)
        key = (_position_key(fen), body.player_color, body.depth, body.mode)
        if key == session.cache_key and session.result is not None:
            result = {**session.result, "fen": fen, "confidence": confidence, "cached": True}
            if result["status"] == "ok":
                result["status"] = "no_change"
                result["message"] = "Board unchanged. Your suggested move is still ready."
            return _attach_geometry(result, geometry)
        if ("w" if board.turn else "b") != body.player_color and not board.is_game_over():
            result = {"status": "opponent_turn", "message": "Opponent's turn. Watching for their move.",
                      "fen": fen, "turn": "w" if board.turn else "b", "confidence": confidence,
                      "move_uci": None, "move_san": None, "move_human": None, "board_image_b64": None}
        else:
            try:
                result = await _engine_result(request.app.state.stockfish, fen, body.depth, body.mode, confidence)
            except Exception:
                logger.exception("Engine failed for camera session")
                return JSONResponse(_attach_geometry({
                    "status": "engine_error", "message": "Analysis interrupted. Retrying this position automatically.",
                    "fen": fen, "confidence": confidence, "move_uci": None, "move_human": None,
                }, geometry), status_code=503)
        # Failed analyses never poison this cache. Depth/color/mode changes always reanalyze.
        session.cache_key = key
        session.result = result.copy()
        return _attach_geometry(result, geometry)


@router.get("/depth-info")
async def depth_info():
    return {"depths": [{"depth": d, "approx_elo": e} for d, e in DEPTH_ELO_MAP.items()],
            "note": "Elo values are approximate estimates, not official ratings."}


@router.post("/reset-session")
async def reset_session(session_id: str = "default"):
    session = _sessions.get(session_id)
    if session:
        async with session.lock:
            session.previous_fen = session.cache_key = session.result = None
            session.last_request = 0.0
    return {"status": "reset", "session_id": session_id}
