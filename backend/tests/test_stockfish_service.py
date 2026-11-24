import pytest
import chess
import time
from services.stockfish_service import (
    StockfishService,
    validate_fen,
    PythonFallbackEngine,
    DEPTH_ELO_MAP,
)

def test_depth_elo_map():
    assert len(DEPTH_ELO_MAP) == 13
    assert DEPTH_ELO_MAP[4] == 800
    assert DEPTH_ELO_MAP[16] == 2600

def test_validate_fen_valid_positions():
    # Starting position
    ok, err = validate_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert ok and not err

    # Endgame position (King and minor pieces - previous bug checked for pawns/rooks/queens)
    ok, err = validate_fen("8/8/4k3/8/8/4KN2/8/8 w - - 0 1")
    assert ok and not err

def test_validate_fen_invalid_positions():
    # No white king
    ok, err = validate_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1BNR w KQkq - 0 1")
    assert not ok and "white king" in err

    # No black king
    ok, err = validate_fen("rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert not ok and "black king" in err

    # White pawn on 8th rank (illegal)
    ok, err = validate_fen("rnbqkbnP/pppppppp/8/8/8/8/PPPPPPP1/RNBQKBNR w KQkq - 0 1")
    assert not ok and "pawn" in err

    # Completely invalid string
    ok, err = validate_fen("not-a-valid-fen")
    assert not ok

def test_stockfish_service_startup_resilience():
    sf = StockfishService()
    sf.start()
    assert sf.is_ready
    sf.stop()

def test_fallback_engine_eval():
    fb = PythonFallbackEngine()
    move = fb.get_best_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", depth=10, mode="win")
    assert move is not None
    board = chess.Board()
    assert chess.Move.from_uci(move) in board.legal_moves

def test_get_best_move_win_and_loss_modes():
    sf = StockfishService()
    sf.start()

    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    res_win = sf.get_best_move(start_fen, depth=10, mode="win")
    assert res_win["uci"] is not None
    assert res_win["san"] is not None
    assert res_win["human"] is not None
    assert res_win["elo"] == (None if res_win["engine"] == "python-chess-fallback" else 1900)

    board = chess.Board(start_fen)
    assert chess.Move.from_uci(res_win["uci"]) in board.legal_moves

    res_loss = sf.get_best_move(start_fen, depth=10, mode="loss")
    assert res_loss["uci"] is not None
    assert chess.Move.from_uci(res_loss["uci"]) in board.legal_moves
    sf.stop()

def test_to_human_descriptions():
    sf = StockfishService()
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    # Pawn push
    san, human = sf._to_human(start_fen, "e2e4")
    assert san == "e4"
    assert human == "Pawn to e4"

    # Knight move
    san, human = sf._to_human(start_fen, "g1f3")
    assert san == "Nf3"
    assert human == "Knight to f3"

    # Capture
    capture_fen = "r1bqkbnr/ppp1pppp/2n5/3p4/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3"
    san, human = sf._to_human(capture_fen, "e4d5")
    assert "exd5" in san
    assert "Capture on d5 with Pawn" in human

    # Check
    check_fen = "rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    assert chess.Board(check_fen).is_checkmate()
    res = sf.get_best_move(check_fen)
    assert res["uci"] is None
    assert res["human"] == "No legal moves."


@pytest.mark.parametrize("fen", [
    "8/8/8/8/8/8/4k3/4K3 w - - 0 1",  # Adjacent kings.
    "4k3/8/8/8/8/8/8/4K3 w KQ - 0 1",  # Castling without rooks.
    "4k3/8/8/8/8/8/8/4K3 w - e3 0 1",  # Impossible en passant.
    "4k3/8/8/8/8/8/PPPPPPPP/P3K3 w - - 0 1",  # Back-rank pawn.
    "4k3/8/8/8/8/8/8/4K2R w - - 0 1".replace("4k3", "7k"),  # Non-moving king in check.
])
def test_complete_fen_legality(fen):
    valid, message = validate_fen(fen)
    assert not valid
    assert message


def test_correct_side_to_move_allows_check_evasion():
    checked = "7k/8/6K1/8/8/8/8/7R b - - 0 1"
    assert validate_fen(checked)[0]
    assert not validate_fen(checked.replace(" b ", " w "))[0]
    sf = StockfishService()
    sf.start()
    try:
        result = sf.get_best_move(checked)
        assert chess.Move.from_uci(result["uci"]) in chess.Board(checked).legal_moves
    finally:
        sf.stop()


def test_fallback_is_bounded_and_reports_actual_engine():
    sf = StockfishService()
    sf._is_fallback = True
    started = time.monotonic()
    result = sf.get_best_move(chess.STARTING_FEN, depth=16)
    assert time.monotonic() - started < 2.0
    assert result["engine"] == "python-chess-fallback"
    assert result["elo"] is None
    assert chess.Move.from_uci(result["uci"]) in chess.Board().legal_moves


def test_human_text_castling_promotion_and_en_passant():
    sf = StockfishService()
    _, castle = sf._to_human("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1")
    assert castle == "Castle Kingside"
    _, promotion = sf._to_human("7k/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q")
    assert "Promote Pawn to Queen on a8" in promotion
    _, capture = sf._to_human("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6")
    assert capture == "Capture on d6 with Pawn"
