import pytest
from fastapi.testclient import TestClient
from main import app

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
    assert data["approx_elo"] == 1900

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
    client.post("/api/analyze", json={
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
