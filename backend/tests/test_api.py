import pytest
import chess
from fastapi.testclient import TestClient
from main import app
from routers import analysis


@pytest.fixture(autouse=True)
def fresh_sessions(monkeypatch):
    analysis._sessions.clear()
    monkeypatch.setattr(analysis, "MIN_INTERVAL_SEC", 0.0)
    yield
    analysis._sessions.clear()

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "engine" in data
    assert data["detector"] == "opencv"

def test_depth_info(client):
    res = client.get("/api/depth-info")
    assert res.status_code == 200
    data = res.json()
    assert "depths" in data
    assert len(data["depths"]) == 13

def test_reset_session(client):
    res = client.post("/api/reset-session?session_id=test_reset")
    assert res.status_code == 200
    assert res.json() == {"status": "reset", "session_id": "test_reset"}

def test_analyze_fen_valid(client):
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    res = client.post("/api/analyze-fen", json={"fen": fen, "depth": 10})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["fen"] == fen
    assert data["turn"] == "w"
    assert "move_uci" in data
    assert "move_san" in data
    assert "move_human" in data
    assert data["approx_elo"] == (None if data["engine"] == "python-chess-fallback" else 1900)

def test_analyze_fen_invalid(client):
    res = client.post("/api/analyze-fen", json={"fen": "not-valid", "depth": 10})
    assert res.status_code == 400
    assert res.json()["status"] == "invalid_fen"

def test_analyze_direct_fen(client):
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    res = client.post("/api/analyze", json={
        "fen": fen,
        "player_color": "w",
        "depth": 10,
        "mode": "win",
        "session_id": "test_direct_fen",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["turn"] == "w"
    assert data["move_uci"] is not None

def test_analyze_no_change(client):
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # First request
    first = client.post("/api/analyze", json={
        "fen": fen,
        "player_color": "w",
        "session_id": "test_no_change",
    })
    # Immediate duplicate request with same FEN
    res = client.post("/api/analyze", json={
        "fen": fen,
        "player_color": "w",
        "session_id": "test_no_change",
    })
    assert res.status_code == 200
    assert res.json()["status"] == "no_change"
    assert res.json()["move_uci"] == first.json()["move_uci"]
    assert res.json()["move_human"] == first.json()["move_human"]
    assert res.json()["overlay"]["to_square"] == first.json()["move_uci"][2:4]

def test_analyze_opponent_turn(client):
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    # Request from White's perspective when it's Black's turn
    res = client.post("/api/analyze", json={
        "fen": fen,
        "player_color": "w",
        "session_id": "test_opp_turn",
    })
    assert res.status_code == 200
    assert res.json()["status"] == "opponent_turn"

def test_analyze_invalid_base64(client):
    res = client.post("/api/analyze", json={
        "image_b64": "!!!not_valid_base64!!!",
        "session_id": "test_invalid_b64",
    })
    assert res.status_code == 400


def test_failed_engine_request_can_retry_same_position(client, monkeypatch):
    original = app.state.stockfish.get_best_move
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary engine failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(app.state.stockfish, "get_best_move", fail_once)
    body = {"fen": chess.STARTING_FEN, "session_id": "retry"}
    failed = client.post("/api/analyze", json=body)
    assert failed.status_code == 503
    assert failed.json()["status"] == "engine_error"
    assert analysis._sessions["retry"].previous_fen == chess.STARTING_FEN
    retry = client.post("/api/analyze", json=body)
    assert retry.json()["status"] == "ok"
    assert retry.json()["move_uci"]
    assert calls == 2


def test_strength_mode_and_player_changes_invalidate_advice(client, monkeypatch):
    calls = []

    def evaluate(fen, depth, mode):
        calls.append((depth, mode))
        return {"uci": "e2e4", "san": "e4", "human": "Pawn to e4", "elo": 1900}

    monkeypatch.setattr(app.state.stockfish, "get_best_move", evaluate)
    body = {"fen": chess.STARTING_FEN, "session_id": "settings"}
    assert client.post("/api/analyze", json=body).json()["status"] == "ok"
    assert client.post("/api/analyze", json={**body, "depth": 12}).json()["status"] == "ok"
    assert client.post("/api/analyze", json={**body, "mode": "loss"}).json()["status"] == "ok"
    assert client.post("/api/analyze", json={**body, "player_color": "b"}).json()["status"] == "opponent_turn"
    assert len(calls) == 3


def test_fen_comparison_includes_side_castling_and_en_passant():
    starting = chess.STARTING_FEN
    assert not analysis._fen_equal(starting, starting.replace(" w ", " b "))
    assert not analysis._fen_equal(starting, starting.replace("KQkq", "-"))
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("a7a6")
    board.push_uci("e4e5")
    board.push_uci("d7d5")
    assert not analysis._fen_equal(board.fen(), board.fen().replace(" d6 ", " - "))


def test_turn_change_same_pieces_does_not_keep_opponent_cache(client):
    body = {"fen": chess.STARTING_FEN.replace(" w ", " b "), "session_id": "turns"}
    assert client.post("/api/analyze", json=body).json()["status"] == "opponent_turn"
    result = client.post("/api/analyze", json={**body, "fen": chess.STARTING_FEN}).json()
    assert result["status"] == "ok"
    assert result["move_uci"]


def test_camera_geometry_and_previous_position_passed_to_detector(client, monkeypatch):
    board = chess.Board()
    calls = []
    corners = [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]

    def detect(image, **kwargs):
        calls.append(kwargs)
        return {"fen": board.fen(), "turn": "w" if board.turn else "b", "confidence": 0.95,
                "error": None, "status": "ok", "board_detected": True, "board_corners": corners,
                "orientation": kwargs["orientation"]}

    monkeypatch.setattr(app.state.detector, "detect", detect)
    body = {"image_b64": "ZmFrZQ==", "session_id": "camera", "orientation": "b"}
    first = client.post("/api/analyze", json=body).json()
    assert first["status"] == "ok"
    assert first["board_locked"]
    assert first["board_corners"] == corners
    assert first["overlay"]["orientation"] == "b"
    assert first["overlay"]["from"] and first["overlay"]["to"]
    assert calls[0]["previous_fen"] is None
    move = first["move_uci"]
    board.push_uci(move)
    second = client.post("/api/analyze", json=body).json()
    assert second["status"] == "opponent_turn"
    assert second["overlay"] is None
    assert calls[1]["previous_fen"] == chess.STARTING_FEN
    board.push(next(iter(board.legal_moves)))
    third = client.post("/api/analyze", json=body).json()
    assert third["status"] == "ok"
    assert chess.Move.from_uci(third["move_uci"]) in board.legal_moves


def test_unknown_camera_position_returns_setup_with_geometry(client, monkeypatch):
    corners = [[0, 0], [1, 0], [1, 1], [0, 1]]
    monkeypatch.setattr(app.state.detector, "detect", lambda *args, **kwargs: {
        "fen": None, "error": "Set the current pieces once to start tracking.", "status": "setup_required",
        "confidence": 0.8, "board_detected": True, "board_corners": corners, "requires_setup": True,
        "orientation": "w", "pieces_detected": True, "piece_count": 12,
    })
    data = client.post("/api/analyze", json={"image_b64": "ZmFrZQ==", "session_id": "setup"}).json()
    assert data["status"] == "setup_required"
    assert data["board_locked"] and data["requires_setup"]
    assert data["move_uci"] is None and data["overlay"] is None
    assert analysis._sessions["setup"].previous_fen is None


def test_perspective_overlay_centers_and_black_orientation():
    corners = [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]
    white = analysis._overlay(chess.STARTING_FEN, "e2e4", corners, "w")
    black = analysis._overlay(chess.STARTING_FEN, "e2e4", corners, "b")
    assert white["from"] == pytest.approx([0.55, 0.6875])
    assert white["to"] == pytest.approx([0.55, 0.5375])
    assert black["from"] == pytest.approx([0.45, 0.3125])
    assert black["to"] == pytest.approx([0.45, 0.4625])
    assert len(white["to_polygon"]) == 4


@pytest.mark.parametrize("body", [
    {"player_color": "blue"}, {"orientation": "sideways"}, {"depth": 99}, {"mode": "random"},
    {"board_corners": [[0, 0]]}, {"board_corners": [[-1, 0], [1, 0], [1, 1], [0, 1]]},
    {"board_corners": [[0, 0], [1, 1], [1, 0], [0, 1]]},
    {"board_corners": [[0, 0], [1, 0], [1, 0], [0, 0]]},
])
def test_invalid_camera_settings_rejected(client, body):
    response = client.post("/api/analyze", json={"fen": chess.STARTING_FEN, **body})
    assert response.status_code == 422


def test_test_prefix_does_not_bypass_rate_limiting(client, monkeypatch):
    monkeypatch.setattr(analysis, "MIN_INTERVAL_SEC", 10.0)
    body = {"fen": chess.STARTING_FEN, "session_id": "test_not_privileged"}
    assert client.post("/api/analyze", json=body).status_code == 200
    limited = client.post("/api/analyze", json=body)
    assert limited.status_code == 429
    assert limited.json()["status"] == "rate_limited"


@pytest.mark.parametrize("fen, message", [
    ("rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3", "Checkmate."),
    ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "Stalemate."),
    ("7k/8/6K1/8/8/8/8/8 w - - 0 1", "Draw: insufficient material."),
])
def test_game_over_has_specific_reason_and_no_arrow(client, fen, message):
    data = client.post("/api/analyze", json={"fen": fen, "session_id": "ended"}).json()
    assert data["status"] == "game_over"
    assert data["message"] == message
    assert data["move_uci"] is None and data["overlay"] is None


def test_illegal_engine_move_is_never_cached_or_shown(client, monkeypatch):
    monkeypatch.setattr(app.state.stockfish, "get_best_move", lambda *args, **kwargs: {
        "uci": "e2e5", "san": "e5", "human": "Pawn to e5", "elo": 1900,
    })
    response = client.post("/api/analyze", json={"fen": chess.STARTING_FEN, "session_id": "illegal"})
    assert response.status_code == 503
    assert response.json()["move_uci"] is None
    assert analysis._sessions["illegal"].result is None


def test_halfmove_clock_draw_does_not_reuse_a_cached_move(client):
    fen = "4k3/8/8/8/8/8/8/R3K3 w - - 149 76"
    body = {"fen": fen, "session_id": "draw_clock"}
    first = client.post("/api/analyze", json=body).json()
    assert first["status"] == "ok"
    assert first["move_uci"]
    second = client.post("/api/analyze", json={**body, "fen": fen.replace("149", "150")}).json()
    assert second["status"] == "game_over"
    assert second["move_uci"] is None
    assert "seventyfive moves" in second["message"]


def test_empty_detected_board_does_not_lock_or_suggest_a_move(client, monkeypatch):
    monkeypatch.setattr(app.state.detector, "detect", lambda *args, **kwargs: {
        "fen": None, "error": "Put the pieces on the board.", "status": "setup_required",
        "confidence": 0.9, "board_detected": True, "board_corners": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "orientation": "w", "pieces_detected": False, "piece_count": 0,
    })
    result = client.post("/api/analyze", json={"image_b64": "ZmFrZQ==", "session_id": "empty_board"}).json()
    assert result["board_detected"] and not result["board_locked"]
    assert result["move_uci"] is None and result["overlay"] is None
