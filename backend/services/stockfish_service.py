import os
import shutil
import threading
import logging
import random
import chess
from stockfish import Stockfish

logger = logging.getLogger(__name__)

DEPTH_ELO_MAP = {
    4: 800,  5: 1000, 6: 1200, 7: 1400, 8: 1600,
    9: 1750, 10: 1900, 11: 2000, 12: 2100, 13: 2200,
    14: 2300, 15: 2400, 16: 2600,
}

_CANDIDATES = [
    os.getenv("STOCKFISH_PATH", ""),
    "/usr/games/stockfish",
    "/usr/bin/stockfish",
    "/usr/local/bin/stockfish",
    "/run/current-system/sw/bin/stockfish",
    "/nix/var/nix/profiles/default/bin/stockfish",
    "/opt/homebrew/bin/stockfish",
    r"C:\Program Files\Stockfish\stockfish.exe",
    r"C:\Program Files (x86)\Stockfish\stockfish.exe",
]

# Material weights for fallback evaluator
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# Positional piece-square tables (from White's perspective)
PAWN_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]


def _resolve_path() -> str | None:
    """Find a usable Stockfish binary on the system, or return None."""
    for p in _CANDIDATES:
        if p and os.path.isfile(p):
            try:
                if os.access(p, os.X_OK):
                    logger.info(f"Stockfish binary found at: {p}")
                    return p
            except Exception:
                pass

    for name in ["stockfish", "stockfish.exe"]:
        found = shutil.which(name)
        if found and os.path.isfile(found):
            logger.info(f"Stockfish binary resolved via PATH: {found}")
            return found

    return None


def validate_fen(fen: str) -> tuple[bool, str]:
    """Validate FEN syntax and basic chess legality."""
    try:
        board = chess.Board(fen)
        white_kings = len(board.pieces(chess.KING, chess.WHITE))
        black_kings = len(board.pieces(chess.KING, chess.BLACK))

        if white_kings != 1:
            return False, f"Expected 1 white king, found {white_kings}."
        if black_kings != 1:
            return False, f"Expected 1 black king, found {black_kings}."

        # Check for pawns on back ranks (illegal in chess)
        for sq in board.pieces(chess.PAWN, chess.WHITE):
            if chess.square_rank(sq) in (0, 7):
                return False, "White pawn detected on first or eighth rank."
        for sq in board.pieces(chess.PAWN, chess.BLACK):
            if chess.square_rank(sq) in (0, 7):
                return False, "Black pawn detected on first or eighth rank."

        # Ensure board status is valid enough for search
        status = board.status()
        if status & chess.STATUS_OPPOSITE_CHECK:
            return False, "Opposite king is in check — impossible position."

        return True, ""
    except Exception as e:
        return False, f"Invalid FEN: {e}"


def _evaluate_board(board: chess.Board) -> int:
    """Fast positional heuristic evaluation from White's perspective."""
    if board.is_checkmate():
        return -99999 if board.turn == chess.WHITE else 99999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0
    for sq, piece in board.piece_map().items():
        base_val = PIECE_VALUES.get(piece.piece_type, 0)
        pos_bonus = 0

        # Piece-square bonuses
        if piece.piece_type == chess.PAWN:
            pos_bonus = PAWN_TABLE[sq if piece.color == chess.WHITE else chess.square_mirror(sq)]
        elif piece.piece_type == chess.KNIGHT:
            pos_bonus = KNIGHT_TABLE[sq if piece.color == chess.WHITE else chess.square_mirror(sq)]
        else:
            r, c = chess.square_rank(sq), chess.square_file(sq)
            pos_bonus = int((3 - abs(3.5 - r) + (3 - abs(3.5 - c))) * 4)

        total = base_val + pos_bonus
        score += total if piece.color == chess.WHITE else -total

    return score


def _alpha_beta_search(board: chess.Board, depth: int, alpha: int, beta: int, maximizing: bool):
    """Minimax search with alpha-beta pruning for fallback engine."""
    if depth <= 0 or board.is_game_over():
        return _evaluate_board(board), None

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return _evaluate_board(board), None

    # Move ordering: captures and checks first
    legal_moves.sort(key=lambda m: (board.is_capture(m), board.gives_check(m)), reverse=True)
    best_move = legal_moves[0]

    if maximizing:
        max_eval = -999999
        for move in legal_moves:
            board.push(move)
            eval_score, _ = _alpha_beta_search(board, depth - 1, alpha, beta, False)
            board.pop()
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = 999999
        for move in legal_moves:
            board.push(move)
            eval_score, _ = _alpha_beta_search(board, depth - 1, alpha, beta, True)
            board.pop()
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move


class PythonFallbackEngine:
    """
    Pure Python chess engine fallback.
    Provides reliable, fast move recommendations when Stockfish binary is unavailable.
    """
    def __init__(self):
        pass

    def get_best_move(self, fen: str, depth: int = 10, mode: str = "win") -> str | None:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # Search depth: clamped to 2-3 for sub-100ms response time
        search_depth = 3 if depth >= 8 else 2
        maximizing = (board.turn == chess.WHITE)

        scored_moves = []
        for move in legal_moves:
            board.push(move)
            score, _ = _alpha_beta_search(board, search_depth - 1, -1000000, 1000000, not maximizing)
            board.pop()
            scored_moves.append((score, move))

        # Sort moves best to worst for current player
        scored_moves.sort(key=lambda x: x[0], reverse=maximizing)

        if mode == "loss" and len(scored_moves) > 1:
            # Pick a believable move with slight inaccuracy
            if len(scored_moves) >= 3 and random.random() < 0.5:
                chosen_move = scored_moves[min(2, len(scored_moves) - 1)][1]
            else:
                chosen_move = scored_moves[1][1]
            return chosen_move.uci()

        return scored_moves[0][1].uci()


class StockfishService:
    def __init__(self):
        self._engine: Stockfish | None = None
        self._path: str | None = None
        self._lock = threading.Lock()
        self._fallback_engine = PythonFallbackEngine()
        self._is_fallback = False

    @property
    def is_fallback(self) -> bool:
        return self._is_fallback

    @property
    def is_ready(self) -> bool:
        return self._engine is not None or self._is_fallback

    def start(self):
        self._path = _resolve_path()
        if self._path:
            try:
                self._spawn()
                self._is_fallback = False
                logger.info(f"Stockfish engine active using binary: {self._path}")
                return
            except Exception as e:
                logger.warning(f"Stockfish spawn failed ({e}). Switching to pure-Python engine fallback.")

        self._is_fallback = True
        logger.info("StockfishService initialized in Python engine fallback mode.")

    def _spawn(self):
        try:
            self._engine = Stockfish(
                path=self._path,
                depth=10,
                parameters={
                    "Threads": 1,
                    "Hash": 32,
                },
            )
            logger.info("Stockfish engine process spawned successfully.")
        except Exception as e:
            self._engine = None
            logger.error(f"Stockfish spawn failed: {e}")
            raise

    def stop(self):
        with self._lock:
            if self._engine:
                try:
                    del self._engine
                except Exception:
                    pass
                self._engine = None
            logger.info("Stockfish service stopped.")

    def get_best_move(self, fen: str, depth: int = 10, mode: str = "win") -> dict:
        with self._lock:
            ok, err = validate_fen(fen)
            if not ok:
                raise ValueError(err)

            depth = max(4, min(16, depth))
            uci = None

            # Attempt 1: Use Stockfish if available
            if not self._is_fallback and self._engine is not None:
                try:
                    effective_depth = depth if mode == "win" else max(4, depth - 3)
                    self._engine.set_depth(effective_depth)
                    self._engine.set_fen_position(fen)
                    uci = self._engine.get_best_move()

                    if mode == "loss" and uci:
                        uci = self._weaken_stockfish(uci)
                except Exception as e:
                    logger.warning(f"Stockfish engine execution failed ({e}), using fallback.")
                    self._engine = None
                    try:
                        self._spawn()
                    except Exception:
                        self._is_fallback = True

            # Attempt 2: Use fallback engine if Stockfish is unavailable or failed
            if not uci:
                uci = self._fallback_engine.get_best_move(fen, depth=depth, mode=mode)

            if not uci:
                return {"uci": None, "san": None, "human": "No legal moves.", "elo": 0}

            san, human = self._to_human(fen, uci)
            return {
                "uci": uci,
                "san": san,
                "human": human,
                "elo": DEPTH_ELO_MAP.get(depth, 1900),
            }

    def _weaken_stockfish(self, best_uci: str) -> str:
        if random.random() > 0.40:
            return best_uci
        try:
            top = self._engine.get_top_moves(3)
            if len(top) >= 2:
                cp_loss = abs((top[0].get("Centipawn") or 0) - (top[1].get("Centipawn") or 0))
                if cp_loss < 250:
                    return top[1]["Move"]
        except Exception:
            pass
        return best_uci

    def _to_human(self, fen: str, uci: str) -> tuple[str, str]:
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(uci)
            san = board.san(move)
            piece = board.piece_at(move.from_square)
            names = {
                chess.PAWN: "Pawn", chess.KNIGHT: "Knight", chess.BISHOP: "Bishop",
                chess.ROOK: "Rook", chess.QUEEN: "Queen", chess.KING: "King",
            }
            pname = names.get(piece.piece_type, "Piece") if piece else "Piece"
            to = chess.square_name(move.to_square)

            if san.startswith("O-O-O"):
                human = "Castle Queenside"
            elif san.startswith("O-O"):
                human = "Castle Kingside"
            elif "=" in san:
                promo_char = san.split("=")[1][0].upper()
                promo = {"Q": "Queen", "R": "Rook", "B": "Bishop", "N": "Knight"}.get(promo_char, "Piece")
                human = f"Promote Pawn to {promo} on {to}"
            elif board.is_capture(move):
                human = f"Capture on {to} with {pname}"
            else:
                human = f"Move {pname} to {to}"

            if "#" in san:
                human += " (Checkmate!)"
            elif "+" in san:
                human += " (Check)"

            return san, human
        except Exception:
            return uci, f"Play {uci}"
