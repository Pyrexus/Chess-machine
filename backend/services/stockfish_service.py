import os
import shutil
import threading
import logging
import random
import time
import chess
import chess.engine

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

        # Validate the complete chess state before either engine searches it.
        status = board.status()
        if status & chess.STATUS_OPPOSITE_CHECK:
            return False, "The side that just moved is still in check. Check the side to move and the piece placement."
        if status & chess.STATUS_BAD_CASTLING_RIGHTS:
            return False, "Castling rights do not match the king and rook positions."
        if status & chess.STATUS_INVALID_EP_SQUARE:
            return False, "The en passant square does not match this position."
        if not board.is_valid():
            return False, "This position is not legal. Check the kings, piece counts and side to move."

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
            pos_bonus = PAWN_TABLE[chess.square_mirror(sq) if piece.color == chess.WHITE else sq]
        elif piece.piece_type == chess.KNIGHT:
            pos_bonus = KNIGHT_TABLE[chess.square_mirror(sq) if piece.color == chess.WHITE else sq]
        else:
            r, c = chess.square_rank(sq), chess.square_file(sq)
            pos_bonus = int((3 - abs(3.5 - r) + (3 - abs(3.5 - c))) * 4)

        total = base_val + pos_bonus
        score += total if piece.color == chess.WHITE else -total

    return score


class SearchTimeout(Exception):
    pass


def _alpha_beta_search(board: chess.Board, depth: int, alpha: int, beta: int, maximizing: bool, deadline: float | None = None):
    """Minimax search with alpha-beta pruning for fallback engine."""
    if deadline is not None and time.monotonic() >= deadline:
        raise SearchTimeout()
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
            try:
                eval_score, _ = _alpha_beta_search(board, depth - 1, alpha, beta, False, deadline)
            finally:
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
            try:
                eval_score, _ = _alpha_beta_search(board, depth - 1, alpha, beta, True, deadline)
            finally:
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

        # Finish shallower iterations before deepening so every request has a
        # legal answer within a bounded CPU budget, including crowded boards.
        search_depth = 3 if depth >= 8 else 2
        deadline = time.monotonic() + 0.3
        maximizing = (board.turn == chess.WHITE)
        scored_moves = []
        for current_depth in range(1, search_depth + 1):
            iteration = []
            try:
                for move in legal_moves:
                    board.push(move)
                    try:
                        score, _ = _alpha_beta_search(board, current_depth - 1, -1000000, 1000000, not maximizing,
                                                     deadline if current_depth > 1 else None)
                    finally:
                        board.pop()
                    iteration.append((score, move))
            except SearchTimeout:
                break
            scored_moves = iteration
            legal_moves = [move for _, move in sorted(scored_moves, key=lambda item: item[0], reverse=maximizing)]

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
        self._engine: chess.engine.SimpleEngine | None = None
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
            # Cold containers can take several seconds to load Stockfish's
            # embedded neural network. Keep that startup allowance separate
            # from the timeout used for ordinary move requests.
            self._engine = chess.engine.SimpleEngine.popen_uci(self._path, timeout=15.0)
            self._engine.configure({"Threads": 1, "Hash": 32})
            self._engine.ping()
            self._engine.timeout = 3.0
            logger.info("Stockfish engine process spawned successfully.")
        except Exception as e:
            if self._engine:
                self._engine.close()
            self._engine = None
            logger.error("Stockfish spawn failed (%s): %r", type(e).__name__, e)
            raise

    def stop(self):
        with self._lock:
            if self._engine:
                try:
                    self._engine.quit()
                except Exception:
                    pass
                self._engine = None
            self._is_fallback = False
            logger.info("Stockfish service stopped.")

    def get_best_move(self, fen: str, depth: int = 10, mode: str = "win") -> dict:
        with self._lock:
            ok, err = validate_fen(fen)
            if not ok:
                raise ValueError(err)

            depth = max(4, min(16, depth))
            board = chess.Board(fen)
            if board.is_game_over():
                return {"uci": None, "san": None, "human": "No legal moves.", "elo": 0,
                        "engine": "python-chess-fallback" if self._is_fallback else "stockfish"}
            uci = None
            used_fallback = False

            # Attempt 1: Use Stockfish if available
            if not self._is_fallback and self._engine is not None:
                try:
                    effective_depth = depth if mode == "win" else max(4, depth - 3)
                    limit = chess.engine.Limit(depth=effective_depth, time=0.35)
                    result = self._engine.play(board, limit)
                    uci = result.move.uci() if result.move else None

                    if mode == "loss" and uci:
                        uci = self._weaken_stockfish(uci, board)
                except Exception as e:
                    logger.warning(f"Stockfish engine execution failed ({e}), using fallback.")
                    if self._engine:
                        self._engine.close()
                    self._engine = None
                    try:
                        self._spawn()
                    except Exception:
                        self._is_fallback = True

            # Attempt 2: Use fallback engine if Stockfish is unavailable or failed
            if not uci:
                used_fallback = True
                uci = self._fallback_engine.get_best_move(fen, depth=depth, mode=mode)

            if not uci:
                return {"uci": None, "san": None, "human": "No legal moves.", "elo": 0}

            if chess.Move.from_uci(uci) not in board.legal_moves:
                raise RuntimeError("Chess engine returned an illegal move.")
            san, human = self._to_human(fen, uci)
            return {
                "uci": uci,
                "san": san,
                "human": human,
                "elo": None if used_fallback else DEPTH_ELO_MAP.get(depth, 1900),
                "engine": "python-chess-fallback" if used_fallback else "stockfish",
            }

    def _weaken_stockfish(self, best_uci: str, board: chess.Board) -> str:
        if random.random() > 0.40:
            return best_uci
        try:
            top = self._engine.analyse(board, chess.engine.Limit(time=0.15), multipv=3)
            if len(top) >= 2:
                first = top[0]["score"].pov(board.turn).score(mate_score=100000)
                second = top[1]["score"].pov(board.turn).score(mate_score=100000)
                cp_loss = abs(first - second)
                if cp_loss < 250:
                    return top[1]["pv"][0].uci()
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
                human = f"{pname} to {to}"

            if "#" in san:
                human += " (Checkmate!)"
            elif "+" in san:
                human += " (Check)"

            return san, human
        except Exception:
            return uci, f"Play {uci}"
