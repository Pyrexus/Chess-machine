import cv2
import numpy as np
import chess
from services.board_detector import BoardDetector, STARTING_FEN

def test_detector_load():
    detector = BoardDetector()
    assert not detector._ready
    detector.load()
    assert detector._ready

def test_matrix_to_fen_no_backrank_pawns():
    detector = BoardDetector()
    matrix = [["empty"] * 8 for _ in range(8)]
    matrix[0][4] = "B"  # Black king on e8
    matrix[7][4] = "W"  # White king on e1
    matrix[0][0] = "W"  # White piece on row 0 (rank 8) -> should be 'Q', not 'P'
    matrix[7][0] = "B"  # Black piece on row 7 (rank 1) -> should be 'q', not 'p'

    fen, turn = detector._matrix_to_fen(matrix, 0.8, player_color="w")
    ranks = fen.split(" ")[0].split("/")
    assert "P" not in ranks[0], "White pawn should not be placed on rank 8"
    assert "p" not in ranks[7], "Black pawn should not be placed on rank 1"
    assert turn == "w"

def test_ensure_valid_fen_repairs():
    detector = BoardDetector()

    # Missing kings
    empty_fen = "8/8/8/8/8/8/8/8 w - - 0 1"
    valid, repaired = detector._ensure_valid_fen(empty_fen, "w")
    assert valid
    b = chess.Board(repaired)
    assert len(b.pieces(chess.KING, chess.WHITE)) == 1
    assert len(b.pieces(chess.KING, chess.BLACK)) == 1

    # Duplicate kings
    dup_fen = "4k3/4k3/8/8/8/8/4K3/4K3 w - - 0 1"
    valid, repaired = detector._ensure_valid_fen(dup_fen, "w")
    assert valid
    b = chess.Board(repaired)
    assert len(b.pieces(chess.KING, chess.WHITE)) == 1
    assert len(b.pieces(chess.KING, chess.BLACK)) == 1

def test_detect_blank_image():
    detector = BoardDetector()
    detector.load()
    blank = np.zeros((480, 480, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", blank)
    res = detector.detect(buf.tobytes())
    assert res["error"] is not None
    assert "Board not found" in res["error"]

def test_draw_move():
    detector = BoardDetector()
    blank = np.zeros((480, 480, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", blank)
    import base64
    b64 = base64.b64encode(buf).decode("utf-8")
    annotated = detector.draw_move(b64, "e2e4")
    assert annotated is not None
    assert len(annotated) > 0
