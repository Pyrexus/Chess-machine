"""Board localisation and legal-move tracking without invented piece identities.

OpenCV measures the board and piece colour/occupancy, not arbitrary piece type.
Calibrate a standard new game or a supplied FEN, then match legal moves to retain
identities. An unrecognised position is never repaired into a different position.
"""

import base64
import logging

import chess
import cv2
import numpy as np

logger = logging.getLogger(__name__)
TARGET_SIZE = 960
WARP_SIZE = 480
MIN_BOARD_AREA_FRACTION = 0.08
STARTING_FEN = chess.STARTING_FEN


class BoardDetector:
    def __init__(self):
        self._ready = False

    def load(self):
        self._ready = True
        logger.info("BoardDetector ready (verified checkerboard + legal tracking).")

    def detect(self, image_bytes: bytes, player_color: str = "w",
               orientation: str = "w", previous_fen: str | None = None,
               board_corners: list | None = None, setup_fen: str | None = None,
               side_to_move: str | None = None, corners_source: str = "auto") -> dict:
        if not self._ready:
            return _err("Detector not initialised.")
        try:
            img = self._resize(self._decode(image_bytes))
        except (ValueError, cv2.error) as exc:
            return _err(f"Could not decode image: {exc}")

        h, w = img.shape[:2]
        if min(h, w) < 64:
            return _err("Image is too small to read a chessboard. Use a full-resolution photo.")
        corners, corner_confidence, geometry_source = None, 0.0, "auto"
        if board_corners is not None:
            try:
                supplied = np.asarray(board_corners, dtype=np.float32)
                if supplied.shape == (4, 2) and np.isfinite(supplied).all() and (
                        supplied >= 0).all() and (supplied <= 1).all():
                    candidate = supplied * np.array([w - 1, h - 1], np.float32)
                    if (cv2.isContourConvex(candidate) and
                            cv2.contourArea(candidate) >= h * w * MIN_BOARD_AREA_FRACTION):
                        candidate = _order_points(candidate)
                        score = self._checker_score(self._warp(img, candidate))
                        if score >= 0.60 or corners_source == "manual":
                            corners, corner_confidence = candidate, score
                            geometry_source = "manual" if corners_source == "manual" else "tracked"
            except (ValueError, cv2.error):
                pass
        if corners is None:
            corners, corner_confidence = self._find_corners(img)
        if corners is None:
            return _err("Board not found. Keep all 64 squares visible and use an overhead view.")

        warped = self._warp(img, corners)
        matrix, piece_confidence = self._classify_squares(warped)
        confidence = round(0.45 * corner_confidence + 0.55 * piece_confidence, 3)
        result = {
            "fen": None, "turn": None, "confidence": confidence,
            "error": None, "status": "ok", "board_detected": True,
            "board_corners": np.clip(corners / np.array([w - 1, h - 1]), 0, 1).round(6).tolist(),
            "orientation": "b" if orientation == "b" else "w",
            "requires_setup": False, "board_b64": _cv2_to_b64(warped),
            "corners_source": geometry_source,
            "quality": self._image_quality(warped),
            "recognition": "occupancy", "position_verified": False,
            "occupancy": matrix,
            "piece_count": sum(cell in ("W", "B") for row in matrix for cell in row),
            "pieces_detected": any(cell in ("W", "B") for row in matrix for cell in row),
        }

        known_fen = setup_fen or previous_fen
        if known_fen:
            valid, canonical = self._ensure_valid_fen(known_fen, "w")
            if not valid:
                return self._uncertain(result, "setup_required",
                                       "The supplied position is illegal. Correct the board and side to move.")
            board = chess.Board(canonical)
            possibilities = []
            choices = ("w", "b") if orientation == "auto" else (result["orientation"],)
            for direction in choices:
                observed = _orient(matrix, direction)
                if setup_fen:
                    matches = [board.fen()] if observed == _occupancy(board) else []
                else:
                    matches = self._match_legal_position(board, observed)
                possibilities.extend((fen, direction) for fen in matches)
            unique = {(fen, direction) for fen, direction in possibilities}
            if len(unique) == 1:
                fen, direction = unique.pop()
                result.update(fen=fen, turn="w" if chess.Board(fen).turn else "b", orientation=direction,
                              recognition="setup" if setup_fen else "legal_tracking", position_verified=True)
                return result
            if len(unique) > 1:
                return self._uncertain(result, "position_uncertain",
                                       "More than one legal position fits. Confirm the position, including any promotion.")
            message = ("The pieces do not match the setup. Check board orientation and the position."
                       if setup_fen else
                       "Pieces are moving or the position is unclear. Remove your hand and hold the camera steady.")
            return self._uncertain(result, "position_uncertain", message)

        # Standard new-game calibration: middle-rank pieces are never guessed.
        starting = _occupancy(chess.Board())
        choices = ("w", "b") if orientation == "auto" else (result["orientation"],)
        for direction in choices:
            if _orient(matrix, direction) == starting:
                initial = chess.Board()
                # The side to move is a user choice, never inferred from the
                # player's colour or from pixels. A known tracked FEN retains
                # its own turn and chess-rule state above.
                if side_to_move in ("w", "b"):
                    initial.turn = side_to_move == "w"
                result.update(fen=initial.fen(), turn="w" if initial.turn else "b", orientation=direction,
                              recognition="standard_opening", position_verified=True)
                return result
        return self._uncertain(result, "setup_required",
                               "Board locked. Start with the standard opening layout, or set this position once to track the game.")

    @staticmethod
    def _uncertain(result, status, message):
        result.update(status=status, error=message, requires_setup=status == "setup_required")
        return result

    @staticmethod
    def _match_legal_position(board: chess.Board, observed: list) -> list[str]:
        """Match unchanged, one ply, or two plies; retain every chess rule.

        Matching is exact. A partly moved piece/hand cannot commit a speculative
        position, and promotions with identical occupancy remain ambiguous.
        """
        if observed == _occupancy(board):
            return [board.fen()]
        one_ply = []
        first_moves = list(board.legal_moves)
        for move in first_moves:
            board.push(move)
            if observed == _occupancy(board):
                one_ply.append(board.fen())
            board.pop()
        if one_ply:
            return list(dict.fromkeys(one_ply))
        two_ply = []
        for first in first_moves:
            board.push(first)
            # One reply changes at most four squares (castling).
            if _difference(_occupancy(board), observed) <= 4:
                for second in list(board.legal_moves):
                    board.push(second)
                    if observed == _occupancy(board):
                        two_ply.append(board.fen())
                    board.pop()
            board.pop()
        return list(dict.fromkeys(two_ply))

    @staticmethod
    def _ensure_valid_fen(fen: str, turn: str) -> tuple[bool, str]:
        try:
            board = chess.Board(fen)
            return board.is_valid(), board.fen()
        except (ValueError, TypeError):
            return False, fen

    @staticmethod
    def _decode(image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            raise ValueError("Empty image.")
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Not a supported image.")
        return img

    @staticmethod
    def _resize(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if max(h, w) > TARGET_SIZE:
            scale = TARGET_SIZE / max(h, w)
            return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        return img

    def _find_corners(self, img: np.ndarray):
        """Only accept quadrilaterals containing a verified 8x8 checker pattern."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        candidates = [np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)]
        frame_score = self._checker_score(self._warp(img, candidates[0]))
        if frame_score >= 0.95:
            return candidates[0], round(frame_score, 3)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for source in (edges, threshold, 255 - threshold):
            contours, _ = cv2.findContours(source, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
                if cv2.contourArea(contour) < h * w * MIN_BOARD_AREA_FRACTION:
                    continue
                approx = cv2.approxPolyDP(contour, 0.025 * cv2.arcLength(contour, True), True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    candidates.append(_order_points(approx.reshape(4, 2).astype(np.float32)))

        best, best_score = None, 0.0
        for candidate in candidates:
            # Wooden boards often have an outer border; search projectively inward.
            for inset in (0.0, 0.02, 0.04, 0.06, 0.08):
                test = _inset_corners(candidate, inset)
                score = self._checker_score(self._warp(img, test))
                if score > best_score:
                    best, best_score = test, score
        if best_score >= 0.78:
            return best, round(best_score, 3)

        # Real pieces break contour boundaries, and a folding board may have
        # no closed outside contour at all. Intersect long, independently
        # detected grid/border lines to recover candidate quadrilaterals.
        line_candidates = self._line_candidates(gray)
        promising = []
        small_dst = np.float32([[0, 0], [159, 0], [159, 159], [0, 159]])
        for candidate in line_candidates:
            transform = cv2.getPerspectiveTransform(candidate, small_dst)
            preview = cv2.warpPerspective(img, transform, (160, 160))
            score = self._checker_score(preview)
            if score >= .58:
                promising.append((score, candidate))
        for _, candidate in sorted(promising, key=lambda item: item[0], reverse=True)[:10]:
            for inset in (0, .015, .03, .045, .06):
                test = _inset_corners(candidate, inset)
                score = self._checker_score(self._warp(img, test))
                if score > best_score:
                    best, best_score = test, score
        if best_score >= .78:
            return best, round(best_score, 3)

        # Internal grid intersections recover an obscured outline. Avoid an
        # exhaustive detector on every phone frame.
        found, points = cv2.findChessboardCorners(
            gray, (7, 7), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
        if found:
            grid = np.array([[x, y] for y in range(1, 8) for x in range(1, 8)], np.float32)
            transform, _ = cv2.findHomography(grid, points.reshape(-1, 2))
            if transform is not None:
                outer = cv2.perspectiveTransform(np.array([[[0, 0], [8, 0], [8, 8], [0, 8]]], np.float32), transform)[0]
                outer = _order_points(outer)
                if (outer[:, 0] >= -2).all() and (outer[:, 0] <= w + 1).all() and (
                        outer[:, 1] >= -2).all() and (outer[:, 1] <= h + 1).all():
                    score = self._checker_score(self._warp(img, outer))
                    if score > best_score:
                        best, best_score = outer, score
        return (best, round(best_score, 3)) if best_score >= 0.65 else (None, 0.0)

    @staticmethod
    def _line_candidates(gray: np.ndarray) -> list:
        h, w = gray.shape
        detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
        if detected is None:
            return []
        lines = detected[:, 0, :]
        lengths = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
        keep = lengths >= min(h, w) * .15
        lines, lengths = lines[keep], lengths[keep]
        if len(lines) < 4:
            return []
        angles = np.mod(np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0]), np.pi)
        hist, edges = np.histogram(angles, bins=36, range=(0, np.pi), weights=lengths)
        dominant = (edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1]) / 2
        distance = lambda a, b: abs((a - b + np.pi / 2) % np.pi - np.pi / 2)
        groups = []
        for angle in (dominant, (dominant + np.pi / 2) % np.pi):
            group = []
            for index in np.argsort(lengths)[::-1]:
                if distance(angles[index], angle) > np.deg2rad(22):
                    continue
                x1, y1, x2, y2 = lines[index]
                line = np.cross([x1, y1, 1], [x2, y2, 1])
                line /= max(1e-6, np.linalg.norm(line[:2]))
                if line[0] < 0 or abs(line[0]) < 1e-6 and line[1] < 0:
                    line *= -1
                if any(abs(np.dot(line - old, [w / 2, h / 2, 1])) < min(h, w) * .012 for old in group):
                    continue
                group.append(line)
                if len(group) >= 9:
                    break
            groups.append(group)
        if min(map(len, groups)) < 2:
            return []
        pairs = []
        for group in groups:
            pair_group = []
            for i, first in enumerate(group):
                for second in group[i + 1:]:
                    if abs(np.dot(first - second, [w / 2, h / 2, 1])) >= min(h, w) * .22:
                        pair_group.append((first, second))
            pairs.append(pair_group)
        candidates = []
        for first, second in pairs[0]:
            for third, fourth in pairs[1]:
                intersections = [np.cross(a, b) for a, b in
                                 [(first, third), (first, fourth), (second, fourth), (second, third)]]
                if any(abs(p[2]) < 1e-6 for p in intersections):
                    continue
                points = _order_points(np.float32([p[:2] / p[2] for p in intersections]))
                if (points[:, 0] < -2).any() or (points[:, 0] > w + 1).any() or (points[:, 1] < -2).any() or (points[:, 1] > h + 1).any():
                    continue
                if cv2.isContourConvex(points) and cv2.contourArea(points) >= h * w * MIN_BOARD_AREA_FRACTION:
                    candidates.append(points)
        return candidates

    @staticmethod
    def _warp(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        dst = np.array([[0, 0], [WARP_SIZE - 1, 0], [WARP_SIZE - 1, WARP_SIZE - 1], [0, WARP_SIZE - 1]], np.float32)
        transform = cv2.getPerspectiveTransform(_order_points(corners), dst)
        return cv2.warpPerspective(img, transform, (WARP_SIZE, WARP_SIZE))

    @staticmethod
    def _backgrounds(board: np.ndarray) -> np.ndarray:
        size = board.shape[0] // 8
        cells = board[:size * 8, :size * 8].reshape(8, size, 8, size, 3).transpose(0, 2, 1, 3, 4)
        patches = []
        # Sample all 64 cells together. This avoids hundreds of Python/median
        # calls for every candidate and keeps camera localisation responsive.
        # Stay clear of square edges: defocus mixes neighbouring colours for
        # several pixels. Sampling there makes an empty cell's clean centre
        # look like a foreground piece on a blurred phone frame.
        for py, px in ((0.18, 0.18), (0.18, 0.82), (0.82, 0.18), (0.82, 0.82)):
            y, x = int(py * size), int(px * size)
            patches.append(np.median(cells[:, :, y - 2:y + 3, x - 2:x + 3], axis=(2, 3)))
        return np.median(np.stack(patches), axis=0).astype(np.float32)

    def _checker_score(self, board: np.ndarray) -> float:
        colours = self._backgrounds(board)
        parity = np.indices((8, 8)).sum(axis=0) % 2
        yy, xx = np.indices((8, 8)).astype(np.float32)
        # A single global light/dark threshold fails when a phone or window
        # casts a gradient across the board. Fit illumination and checker
        # contrast together; robust weights suppress piece-covered samples.
        lighting = np.stack([np.ones(64), xx.ravel() / 7, yy.ravel() / 7], axis=1)
        # A shadow scales both the mean square colour and its checker
        # contrast. Fit both effects; a constant contrast alone shifts the
        # best-scoring boundary inward under uneven illumination.
        design = np.concatenate([lighting, lighting * (parity.ravel() * 2 - 1)[:, None]], axis=1)
        samples = colours.reshape(64, 3)
        weights = np.ones(64)
        for _ in range(4):
            root = np.sqrt(weights)[:, None]
            coeff = np.linalg.lstsq(design * root, samples * root, rcond=None)[0]
            errors = np.linalg.norm(samples - design @ coeff, axis=1)
            scale = max(4.0, float(np.median(errors)) * 1.4826)
            weights = np.minimum(1.0, 2.5 * scale / np.maximum(errors, 1e-3))
        checker = lighting @ coeff[3:]
        contrast = float(np.median(np.linalg.norm(checker, axis=1)) * 2)
        if contrast < 18:
            return 0.0
        illumination = lighting @ coeff[:3]
        distances = np.stack([np.linalg.norm(samples - illumination + checker, axis=1),
                              np.linalg.norm(samples - illumination - checker, axis=1)], axis=1)
        matching = (np.argmin(distances, axis=1) == parity.ravel()).reshape(8, 8)
        agreement = float(np.mean(matching))
        # Require checker evidence across both axes, not just one patch of
        # floor tiles, a rectangle, or screen stripes.
        if agreement < .80 or np.mean(matching.mean(axis=0) >= .625) < .75 or np.mean(matching.mean(axis=1) >= .625) < .75:
            return 0.0
        residual = float(np.median(np.min(distances, axis=1))) / contrast
        return float(np.clip(agreement * (1 - min(residual, 0.35)), 0, 1))

    @staticmethod
    def _image_quality(board: np.ndarray) -> dict:
        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5))
        return {"sharpness": round(sharpness, 1), "contrast": round(contrast, 1),
                "blurred": sharpness < 22, "low_contrast": contrast < 35}

    def _classify_squares(self, board: np.ndarray):
        size = board.shape[0] // 8
        backgrounds = self._backgrounds(board)
        lab = cv2.cvtColor(board, cv2.COLOR_BGR2LAB).astype(np.float32)
        background_lab = cv2.cvtColor(backgrounds.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        entries = []
        for row in range(8):
            for col in range(8):
                cell = lab[row * size:(row + 1) * size, col * size:(col + 1) * size]
                margin = max(3, size // 10)
                interior = cell[margin:-margin, margin:-margin]
                distance = np.linalg.norm(interior - background_lab[row, col], axis=2)
                # Phone exposure and wood grain change the background noise.
                # A fixed threshold misses pale wooden pieces on light squares.
                # Measure noise where the square background was sampled and
                # separate coherent foreground from that local variation.
                corner_noise = []
                for py, px in ((0.14, 0.14), (0.14, 0.86), (0.86, 0.14), (0.86, 0.86)):
                    y, x = int(py * size), int(px * size)
                    patch = cell[y - 2:y + 3, x - 2:x + 3]
                    corner_noise.extend(np.linalg.norm(patch - background_lab[row, col], axis=2).ravel())
                threshold = max(8.0, float(np.percentile(corner_noise, 90)) * 1.5 + 2)
                mask = (distance > threshold).astype(np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
                if count <= 1:
                    entries.append(("empty", None, 0.98))
                    continue
                # Defocus spreads alternating square edges into the cell.
                # Those edge strips must not become phantom pieces. A piece
                # footprint has substantial evidence near the cell centre.
                mh, mw = mask.shape
                central = labels[int(mh * .3):int(mh * .7), int(mw * .3):int(mw * .7)]
                candidates = [i for i in range(1, count)
                              if .18 * mw <= centroids[i, 0] <= .82 * mw
                              and .18 * mh <= centroids[i, 1] <= .82 * mh
                              and np.mean(central == i) > .06]
                if not candidates:
                    entries.append(("empty", None, 0.9))
                    continue
                index = max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])
                fraction = int(stats[index, cv2.CC_STAT_AREA]) / interior.shape[0] / interior.shape[1]
                if fraction < 0.035:
                    entries.append(("empty", None, 0.9))
                    continue
                # A hand spanning most of a square is not a chess piece.
                if fraction > 0.82:
                    entries.append(("?", None, 0.2))
                    continue
                lightness = float(np.median(interior[:, :, 0][labels == index]))
                entries.append(("piece", lightness, min(0.98, 0.65 + fraction)))

        levels = [value for kind, value, _ in entries if kind == "piece"]
        # Legal positions contain both kings. Adapt lighter/darker piece colour
        # clusters to wood tones and exposure instead of hardcoded square RGB.
        if len(levels) >= 2:
            low, high = min(levels), max(levels)
            for _ in range(6):
                midpoint = (low + high) / 2
                darker = [value for value in levels if value < midpoint]
                lighter = [value for value in levels if value >= midpoint]
                if not darker or not lighter:
                    break
                low, high = float(np.median(darker)), float(np.median(lighter))
            threshold = (low + high) / 2
            distinguishable = high - low >= 25
        else:
            threshold, distinguishable = 128, False
        cells, confidence = [], []
        for kind, value, certainty in entries:
            if kind == "piece":
                kind = ("W" if value >= threshold else "B") if distinguishable else "?"
                if kind == "?":
                    certainty = 0.2
            cells.append(kind)
            confidence.append(certainty)
        return [cells[index:index + 8] for index in range(0, 64, 8)], round(float(np.mean(confidence)), 3)

    def draw_move(self, board_b64: str, uci_move: str, orientation: str = "w") -> str:
        """Compatibility helper; the web UI draws on the live camera instead."""
        try:
            img = _b64_to_cv2(board_b64)
            size = img.shape[0] // 8
            start, end = (_sq_to_pixel(uci_move[:2], size, orientation),
                          _sq_to_pixel(uci_move[2:4], size, orientation))
            overlay = img.copy()
            cv2.circle(overlay, end, int(size * 0.4), (90, 210, 0), -1)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            cv2.arrowedLine(img, start, end, (90, 210, 0), thickness=5, tipLength=0.3)
            return _cv2_to_b64(img)
        except (ValueError, cv2.error, IndexError, TypeError) as exc:
            logger.warning("draw_move failed: %s", exc)
            return board_b64


def _err(message: str, confidence: float = 0.0) -> dict:
    return {"fen": None, "turn": None, "confidence": confidence, "error": message,
            "status": "detection_error", "board_detected": False, "board_corners": None,
            "requires_setup": False, "board_b64": None, "piece_count": 0, "pieces_detected": False,
            "quality": None, "recognition": "unreadable", "position_verified": False,
            "corners_source": "auto"}


def _occupancy(board: chess.Board) -> list:
    return [[("W" if piece.color else "B") if (piece := board.piece_at(chess.square(col, 7 - row)))
             else "empty" for col in range(8)] for row in range(8)]


def _orient(matrix: list, orientation: str) -> list:
    return [list(reversed(row)) for row in reversed(matrix)] if orientation == "b" else matrix


def _difference(first: list, second: list) -> int:
    return sum(a != b for row_a, row_b in zip(first, second) for a, b in zip(row_a, row_b))


def _order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, np.float32)
    # Sorting by sums/differences can select the same vertex twice when a
    # board is rotated 45 degrees. Order around the centre first so every
    # distinct vertex is retained, then start at the upper-left corner.
    centre = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    ordered = points[np.argsort(angles)]
    return np.roll(ordered, -int(np.argmin(ordered.sum(axis=1))), axis=0)


def _inset_corners(corners: np.ndarray, fraction: float) -> np.ndarray:
    transform = cv2.getPerspectiveTransform(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32), corners)
    points = np.array([[[fraction, fraction], [1 - fraction, fraction],
                        [1 - fraction, 1 - fraction], [fraction, 1 - fraction]]], np.float32)
    return cv2.perspectiveTransform(points, transform)[0]


def _sq_to_pixel(square: str, size: int, orientation: str = "w") -> tuple[int, int]:
    index = chess.parse_square(square)
    col, row = chess.square_file(index), 7 - chess.square_rank(index)
    if orientation == "b":
        col, row = 7 - col, 7 - row
    return round((col + 0.5) * size), round((row + 0.5) * size)


def _cv2_to_b64(img: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not success:
        raise ValueError("Image encoding failed.")
    return base64.b64encode(buffer).decode("ascii")


def _b64_to_cv2(value: str) -> np.ndarray:
    return BoardDetector._decode(base64.b64decode(value, validate=True))
