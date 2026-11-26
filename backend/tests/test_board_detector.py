import base64
import time

import chess
import cv2
import numpy as np
import pytest

from services.board_detector import BoardDetector, STARTING_FEN, _occupancy, _sq_to_pixel, _order_points


@pytest.fixture
def detector():
    value = BoardDetector()
    value.load()
    return value


def board_image(board=None, rotation=False, exposure=1.0, perspective=False, pieces=True):
    """Synthetic camera fixture: checkerboard, contrasting piece footprints,
    optional camera perspective and changed exposure. No detector mocking.
    """
    board = board or chess.Board()
    image = np.zeros((480, 480, 3), np.uint8)
    for row in range(8):
        for col in range(8):
            colour = (170, 195, 210) if (row + col) % 2 == 0 else (55, 90, 120)
            cv2.rectangle(image, (col * 60, row * 60), ((col + 1) * 60 - 1, (row + 1) * 60 - 1), colour, -1)
            piece = board.piece_at(chess.square(col, 7 - row)) if pieces else None
            if piece:
                colour = (240, 240, 240) if piece.color else (20, 20, 20)
                cv2.circle(image, (col * 60 + 30, row * 60 + 30), 15, colour, -1)
    if rotation:
        image = cv2.rotate(image, cv2.ROTATE_180)
    if perspective:
        corners = np.array([[105, 65], [575, 105], [530, 570], [60, 515]], np.float32)
        transform = cv2.getPerspectiveTransform(np.array([[0, 0], [479, 0], [479, 479], [0, 479]], np.float32), corners)
        output = np.full((640, 640, 3), (130, 140, 150), np.uint8)
        warped = cv2.warpPerspective(image, transform, (640, 640))
        mask = cv2.warpPerspective(np.full((480, 480), 255, np.uint8), transform, (640, 640))
        output[mask > 0] = warped[mask > 0]
        image = output
    return np.clip(image.astype(np.float32) * exposure, 0, 255).astype(np.uint8)


def encode(image):
    success, value = cv2.imencode(".png", image)
    assert success
    return value.tobytes()


def test_detector_requires_loading():
    value = BoardDetector()
    assert value.detect(b"")["status"] == "detection_error"
    value.load()
    assert value._ready


@pytest.mark.parametrize("image", [np.zeros((480, 480, 3), np.uint8),
                                   np.full((480, 480, 3), 220, np.uint8)])
def test_blank_image_never_returns_a_position(detector, image):
    result = detector.detect(encode(image))
    assert result["status"] == "detection_error"
    assert not result["board_detected"]
    assert result["fen"] is None


def test_rectangle_is_not_a_board(detector):
    image = np.full((480, 480, 3), 210, np.uint8)
    cv2.rectangle(image, (50, 50), (420, 420), (30, 30, 30), 10)
    assert not detector.detect(encode(image))["board_detected"]


def test_empty_checkerboard_has_no_fake_kings_or_opening(detector):
    result = detector.detect(encode(board_image(pieces=False)))
    assert result["board_detected"]
    assert not result["pieces_detected"]
    assert result["piece_count"] == 0
    assert result["fen"] is None
    assert result["status"] == "setup_required"


@pytest.mark.parametrize("exposure", [0.7, 1.0])
def test_starting_position_calibrates_with_pieces(detector, exposure):
    result = detector.detect(encode(board_image(exposure=exposure)))
    assert result["status"] == "ok", result["error"]
    assert result["fen"] == STARTING_FEN
    assert result["turn"] == "w"
    assert result["piece_count"] == 32
    assert result["confidence"] > 0.75
    assert np.allclose(result["board_corners"], [[0, 0], [1, 0], [1, 1], [0, 1]], atol=0.02)


def test_player_colour_does_not_change_starting_turn(detector):
    result = detector.detect(encode(board_image()), player_color="b")
    assert result["turn"] == "w"
    assert chess.Board(result["fen"]).turn == chess.WHITE


def test_orientation_is_independent_of_player_colour(detector):
    result = detector.detect(encode(board_image(rotation=True)), player_color="w", orientation="b")
    assert result["fen"] == STARTING_FEN
    assert result["orientation"] == "b"
    automatic = detector.detect(encode(board_image(rotation=True)), orientation="auto")
    assert automatic["fen"] == STARTING_FEN
    assert automatic["orientation"] == "b"


def test_perspective_corners_map_back_to_camera(detector):
    result = detector.detect(encode(board_image(perspective=True)))
    assert result["status"] == "ok", result["error"]
    expected = np.array([[105, 65], [575, 105], [530, 570], [60, 515]]) / 639
    assert np.allclose(result["board_corners"], expected, atol=0.018)


def test_knight_move_keeps_piece_identity_and_changes_turn(detector):
    board = chess.Board()
    board.push_uci("g1f3")
    result = detector.detect(encode(board_image(board)), previous_fen=STARTING_FEN)
    assert result["status"] == "ok", result["error"]
    assert result["fen"] == board.fen()
    assert chess.Board(result["fen"]).piece_at(chess.F3).piece_type == chess.KNIGHT
    assert result["turn"] == "b"


def test_two_moves_between_frames_are_tracked(detector):
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("d7d5")
    result = detector.detect(encode(board_image(board)), previous_fen=STARTING_FEN)
    assert result["status"] == "ok", result["error"]
    assert result["fen"] == board.fen()
    assert result["turn"] == "w"


def test_capture_is_tracked():
    board = chess.Board()
    for move in ("e2e4", "d7d5"):
        board.push_uci(move)
    before = board.copy()
    board.push_uci("e4d5")
    assert BoardDetector._match_legal_position(before, _occupancy(board)) == [board.fen()]


@pytest.mark.parametrize("fen,move", [
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
])
def test_special_moves_retain_castling_and_en_passant_rules(fen, move):
    board = chess.Board(fen)
    before = board.copy()
    board.push_uci(move)
    assert BoardDetector._match_legal_position(before, _occupancy(board)) == [board.fen()]


def test_promotion_is_not_guessed_from_occupancy():
    board = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
    before = board.copy()
    board.push_uci("a7a8q")
    matches = BoardDetector._match_legal_position(before, _occupancy(board))
    assert len(matches) == 4


def test_unknown_midgame_requires_setup_instead_of_fake_fen(detector):
    board = chess.Board("4k3/8/8/3q4/8/2N5/8/4K3 w - - 0 1")
    result = detector.detect(encode(board_image(board)))
    assert result["board_detected"]
    assert result["pieces_detected"]
    assert result["fen"] is None
    assert result["status"] == "setup_required"
    assert result["board_corners"]
    calibrated = detector.detect(encode(board_image(board)), setup_fen=board.fen())
    assert calibrated["fen"] == board.fen()


def test_partial_move_does_not_commit_a_position(detector):
    image = board_image()
    # Remove a pawn without placing it anywhere; no legal move fits.
    cv2.rectangle(image, (240, 360), (299, 419), (170, 195, 210), -1)
    result = detector.detect(encode(image), previous_fen=STARTING_FEN)
    assert result["status"] == "position_uncertain"
    assert result["fen"] is None


def test_locked_corners_are_revalidated_when_camera_moves(detector):
    initial = detector.detect(encode(board_image()))
    moved = detector.detect(encode(board_image(perspective=True)), board_corners=initial["board_corners"],
                            previous_fen=STARTING_FEN)
    assert moved["status"] == "ok", moved["error"]
    assert not np.allclose(moved["board_corners"], initial["board_corners"], atol=0.05)
    lost = detector.detect(encode(np.zeros((480, 480, 3), np.uint8)), board_corners=moved["board_corners"],
                           previous_fen=STARTING_FEN)
    assert not lost["board_detected"]


@pytest.mark.parametrize("fen", ["nonsense", "8/8/8/8/8/8/8/8 w - - 0 1",
                                "4k3/4k3/8/8/8/8/4K3/4K3 w - - 0 1"])
def test_illegal_fens_are_rejected_without_repair(fen):
    valid, value = BoardDetector._ensure_valid_fen(fen, "w")
    assert not valid
    if fen != "nonsense":
        assert chess.Board(value).board_fen() == chess.Board(fen).board_fen()


def test_draw_move_matches_orientation(detector):
    image = board_image()
    original = base64.b64encode(encode(image)).decode("ascii")
    assert detector.draw_move(original, "e2e4") != original
    assert _sq_to_pixel("e2", 60, "w") == (270, 390)
    assert _sq_to_pixel("e2", 60, "b") == (210, 90)
    assert detector.draw_move(original, "wrong") == original


@pytest.mark.parametrize("size", [(1, 1), (40, 400), (400, 40)])
def test_tiny_images_return_detection_error_instead_of_invalid_transform(detector, size):
    result = detector.detect(encode(np.zeros((*size, 3), np.uint8)))
    assert result["status"] == "detection_error"
    assert result["fen"] is None


def test_diagonal_board_corners_preserve_four_distinct_vertices():
    corners = np.array([[240, 0], [480, 240], [240, 480], [0, 240]], np.float32)
    ordered = _order_points(corners)
    assert len(np.unique(ordered, axis=0)) == 4
    assert cv2.isContourConvex(ordered)
    assert cv2.contourArea(ordered) == 115200


@pytest.mark.parametrize("corners", [
    [[0, 0], [1, 1], [1, 0], [0, 1]],
    [[0, 0], [1, 0], [1, 0], [0, 0]],
    [[0, 0], [2, 0], [1, 1], [0, 1]],
])
def test_invalid_manual_corners_fall_back_to_valid_auto_detection(detector, corners):
    result = detector.detect(encode(board_image()), board_corners=corners)
    assert result["fen"] == STARTING_FEN
    assert np.allclose(result["board_corners"], [[0, 0], [1, 0], [1, 1], [0, 1]], atol=0.02)


@pytest.mark.parametrize("fen,move", [
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1"),
    ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8"),
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
])
def test_special_moves_are_detected_from_the_camera_image(detector, fen, move):
    board = chess.Board(fen)
    board.push_uci(move)
    result = detector.detect(encode(board_image(board)), previous_fen=fen)
    assert result["status"] == "ok", result["error"]
    assert result["fen"] == board.fen()


def test_promotion_image_requires_confirmation_instead_of_guessing_queen(detector):
    fen = "7k/P7/8/8/8/8/8/7K w - - 0 1"
    board = chess.Board(fen)
    board.push_uci("a7a8q")
    result = detector.detect(encode(board_image(board)), previous_fen=fen)
    assert result["status"] == "position_uncertain"
    assert result["fen"] is None
    assert "promotion" in result["error"]


def test_letterboxed_landscape_camera_jpeg_coordinates_and_move_tracking(detector):
    camera = np.full((540, 960, 3), (80, 100, 110), np.uint8)
    frame = np.full((720, 720, 3), (35, 41, 24), np.uint8)
    camera[30:510, 240:720] = board_image()
    frame[157:562] = cv2.resize(camera, (720, 405))
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    initial = detector.detect(jpeg.tobytes())
    assert initial["fen"] == STARTING_FEN
    assert np.allclose(initial["board_corners"], [[.25, .25], [.75, .25], [.75, .75], [.25, .75]], atol=.02)
    moved = chess.Board()
    moved.push_uci("e2e4")
    camera[30:510, 240:720] = board_image(moved)
    frame[157:562] = cv2.resize(camera, (720, 405))
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    result = detector.detect(jpeg.tobytes(), previous_fen=STARTING_FEN,
                             board_corners=initial["board_corners"])
    assert result["fen"] == moved.fen()
    assert result["turn"] == "b"


def test_cropped_squares_do_not_auto_calibrate(detector):
    result = detector.detect(encode(board_image()[30:450, 30:450]))
    assert not result["board_detected"]
    assert result["fen"] is None


def test_hand_over_board_does_not_commit_a_legal_position(detector):
    image = board_image()
    cv2.rectangle(image, (190, 290), (350, 440), (100, 145, 200), -1)
    result = detector.detect(encode(image), previous_fen=STARTING_FEN)
    assert result["status"] in ("detection_error", "position_uncertain")
    assert result["fen"] is None


@pytest.mark.parametrize("pieces", [True, False])
def test_warm_wooden_piece_tones_and_camera_noise_are_not_hardcoded_black_white(detector, pieces):
    image = board_image(pieces=pieces)
    image[np.all(image == (240, 240, 240), axis=2)] = (185, 225, 240)
    image[np.all(image == (20, 20, 20), axis=2)] = (35, 60, 90)
    noise = np.random.default_rng(3).normal(0, 2.5, image.shape)
    image = np.clip(image.astype(np.float32) * .85 + noise, 0, 255).astype(np.uint8)
    result = detector.detect(encode(image))
    assert result["board_detected"]
    if pieces:
        assert result["status"] == "ok", result["error"]
        assert result["fen"] == STARTING_FEN
        assert result["piece_count"] == 32
    else:
        assert result["piece_count"] == 0
        assert result["fen"] is None


@pytest.mark.parametrize("sigma", [1.5, 3, 6])
def test_blurred_opening_retains_exact_occupancy_and_board_alignment(detector, sigma):
    image = cv2.GaussianBlur(board_image(), (0, 0), sigma)
    result = detector.detect(encode(image), corners_source="auto", side_to_move="w")
    assert result["status"] == "ok", result["error"]
    assert result["fen"] == STARTING_FEN
    assert result["piece_count"] == 32
    assert result["position_verified"]
    assert result["recognition"] == "standard_opening"
    assert result["quality"]["sharpness"] >= 0
    assert np.allclose(result["board_corners"], [[0, 0], [1, 0], [1, 1], [0, 1]], atol=.01)


def test_shadow_does_not_shift_board_boundary_or_lose_pieces(detector):
    image = np.clip(board_image().astype(float) * np.linspace(.35, 1, 480)[None, :, None], 0, 255).astype(np.uint8)
    result = detector.detect(encode(image))
    assert result["fen"] == STARTING_FEN
    assert result["piece_count"] == 32
    assert np.allclose(result["board_corners"], [[0, 0], [1, 0], [1, 1], [0, 1]], atol=.01)


def test_low_contrast_board_still_calibrates(detector):
    image = np.clip(board_image().astype(float) * .24 + 100, 0, 255).astype(np.uint8)
    result = detector.detect(encode(image))
    assert result["fen"] == STARTING_FEN
    assert result["piece_count"] == 32


def test_blurred_empty_squares_do_not_become_phantom_pieces(detector):
    image = cv2.GaussianBlur(board_image(pieces=False), (0, 0), 6)
    result = detector.detect(encode(image))
    assert result["board_detected"]
    assert result["piece_count"] == 0
    assert result["fen"] is None
    assert not result["position_verified"]


def test_manual_corner_lock_does_not_invent_a_position_on_blank_frame(detector):
    corners = [[.1, .1], [.9, .1], [.9, .9], [.1, .9]]
    result = detector.detect(encode(np.full((480, 480, 3), 120, np.uint8)),
                             board_corners=corners, corners_source="manual", previous_fen=STARTING_FEN)
    assert result["board_detected"]
    assert result["corners_source"] == "manual"
    assert result["fen"] is None
    assert not result["position_verified"]
    assert result["status"] == "position_uncertain"


def test_turn_selection_applies_only_to_new_calibration(detector):
    result = detector.detect(encode(board_image()), side_to_move="b")
    assert result["turn"] == "b"
    assert not chess.Board(result["fen"]).turn
    tracked = detector.detect(encode(board_image()), previous_fen=STARTING_FEN, side_to_move="b")
    assert tracked["fen"] == STARTING_FEN
    assert tracked["recognition"] == "legal_tracking"


def test_tracked_corners_are_distinct_from_manual_and_rejected_without_board_evidence(detector):
    corners = [[0, 0], [1, 0], [1, 1], [0, 1]]
    result = detector.detect(encode(board_image()), board_corners=corners, corners_source="tracked")
    assert result["fen"] == STARTING_FEN
    assert result["corners_source"] == "tracked"
    lost = detector.detect(encode(np.full((480, 480, 3), 120, np.uint8)),
                           board_corners=corners, corners_source="tracked", previous_fen=STARTING_FEN)
    assert not lost["board_detected"]
    assert lost["fen"] is None


def test_auto_orientation_reacquires_reversed_board_while_tracking(detector):
    initial = detector.detect(encode(board_image()), orientation="auto")
    board = chess.Board()
    board.push_uci("e2e4")
    reversed_view = detector.detect(encode(board_image(board, rotation=True)), orientation="auto",
                                    previous_fen=initial["fen"], board_corners=initial["board_corners"],
                                    corners_source="tracked")
    assert reversed_view["fen"] == board.fen()
    assert reversed_view["orientation"] == "b"
    assert reversed_view["position_verified"]
