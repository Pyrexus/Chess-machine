"""Generate reproducible synthetic camera fixtures for browser integration QA."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chess
import cv2
from test_board_detector import board_image


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[2] / ".tmp"
    output.mkdir(exist_ok=True)
    board = chess.Board()
    for name, move in [("camera-start.png", None), ("camera-e4.png", "e2e4")]:
        if move:
            board.push_uci(move)
        assert cv2.imwrite(str(output / name), board_image(board)), "Could not write camera fixture"
    print("Generated synthetic camera fixtures in .tmp")
