# API reference

The API is served by FastAPI at the same origin as the Railway-hosted frontend. Interactive documentation is available at `/docs`.

## `GET /health`

Returns HTTP 200 when the engine service and detector are ready, or HTTP 503 while they are not ready.

```json
{
  "status": "ok",
  "engine": "stockfish",
  "detector": "opencv",
  "recognition": "legal-occupancy-tracking",
  "requires_setup_for_unknown_positions": true,
  "camera_overlay": true
}
```

`engine` is `python-chess-fallback` when Stockfish is unavailable. A ready fallback service does not imply Stockfish strength.

## `POST /api/analyze`

Analyzes a camera frame or an authoritative FEN within a tracking session.

```json
{
  "image_b64": "<base64 JPEG or PNG>",
  "player_color": "w",
  "orientation": "w",
  "depth": 10,
  "mode": "win",
  "session_id": "<unique camera session>"
}
```

| Field | Default | Meaning |
| --- | --- | --- |
| `image_b64` | `null` | Encoded image; a base64 data URL is also accepted. Decoded image limit: 6 MB. |
| `fen` | `null` | Direct position input. When present, it takes precedence over the image. |
| `setup_fen` | `null` | Exact position used to calibrate an image once. Later frames omit it and use the session's trusted position. |
| `player_color` | `"w"` | The color receiving advice: `w` or `b`. |
| `orientation` | `"auto"` | Color at the bottom of the captured board: `w`, `b`, or `auto`. Automatic inference is most useful at the standard opening setup. |
| `side_to_move` | `null` | Optional `w` or `b` override when calibrating `setup_fen`; it does not rewrite an ordinary trusted FEN. |
| `board_corners` | `null` | Four normalized `[x, y]` points in image order: top-left, top-right, bottom-right, bottom-left. |
| `corners_source` | `"auto"` | `auto` or `tracked` corners are revalidated against the checkerboard; `manual` selects a user-defined region. Manual geometry does not verify a position. |
| `depth` | `10` | Integer from 4 through 16. Engine search also has a time budget. |
| `mode` | `"win"` | `win` for stronger advice; `loss` for deliberately weaker choices. |
| `session_id` | `"default"` | Unique identifier, 1–128 characters. Use a separate identifier for each camera session. |

Supply `image_b64` or `fen`. FEN strings have a maximum length of 128 characters and must describe a legal chess position. Piece types in a supplied FEN are authoritative: the detector can verify their colors and occupied squares, but cannot independently verify arbitrary physical piece identities.

For a standard new game, the detector uses the standard piece arrangement when the observed starting occupancy matches. For an unknown midgame image, it returns `setup_required`; it does not invent a position. Subsequent frames match an unchanged position, one legal move, or two consecutive legal moves from the session's trusted position.

### Move response

```json
{
  "status": "ok",
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "turn": "w",
  "move_uci": "e2e4",
  "move_san": "e4",
  "move_human": "Pawn to e4",
  "confidence": 0.9,
  "engine": "stockfish",
  "approx_elo": 1900,
  "board_image_b64": null,
  "board_detected": true,
  "board_locked": true,
  "board_corners": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
  "orientation": "w",
  "requires_setup": false,
  "piece_count": 32,
  "pieces_detected": true,
  "overlay": {
    "from_square": "e2",
    "to_square": "e4",
    "piece": "pawn",
    "promotion": null,
    "orientation": "w",
    "from": [0.55, 0.75],
    "to": [0.55, 0.55],
    "from_polygon": [[0.5, 0.7], [0.6, 0.7], [0.6, 0.8], [0.5, 0.8]],
    "to_polygon": [[0.5, 0.5], [0.6, 0.5], [0.6, 0.6], [0.5, 0.6]]
  }
}
```

The move above illustrates the response format; the engine can select another legal opening move. `fen` describes the position **before** the suggested move. `turn` comes from that position rather than the player's selected color.

All geometry uses the uploaded image coordinate system, normalized to `[0, 1]`. The frontend captures its complete camera image inside a square with letterboxing, and draws the returned geometry over that same square. `from` and `to` are projected square centers; the polygons are the projected square boundaries. When there is no image geometry, `overlay` can contain the square names without coordinate fields.

`board_image_b64` remains `null` for compatibility; the client renders the overlay directly. `confidence` describes detector measurements, not a calibrated probability of piece-type recognition. `approx_elo` is a legacy rough estimate, not a measured rating, and is `null` for the Python fallback.

### Response statuses

| Status | Meaning and client behavior |
| --- | --- |
| `ok` | Legal move found. Display its text and overlay. |
| `no_change` | Cached advice for the unchanged chess state and analysis settings. The full move remains in the response. Keep displaying it and use the current frame's geometry. |
| `opponent_turn` | Trusted position is the opponent's turn. Clear the move arrow and keep scanning. |
| `setup_required` | Board geometry exists, but piece identities need calibration. Offer standard starting setup or an exact FEN. |
| `position_uncertain` | No unique legal tracked position matches the frame. Clear the arrow and request a steady, unobstructed board or position confirmation. |
| `detection_error` | No reliable board or readable image was found. Clear the arrow and keep scanning. |
| `invalid_fen` | Illegal or malformed supplied position. Correct its pieces, side to move, and state fields. HTTP 400. |
| `rate_limited` | Requests are too close together. HTTP 429; observe `Retry-After` and the `retry_after` JSON field. |
| `game_over` | Checkmate, stalemate, or another automatic game-ending condition. No move is returned. |
| `engine_error` | Analysis failed. HTTP 503; retry the same position. Failed results do not poison the move cache. |

Detected geometry can be returned even when the position needs setup. `board_locked` normally requires a detected board and pieces; a manual region or a temporarily held boundary can also remain locked without a verified position. An empty checkerboard never produces a move.

Camera responses include `position_verified`, `recognition` (`standard_opening`, `setup`, `legal_tracking`, `occupancy`, or `unreadable`), and `quality` (sharpness, contrast, and blur/low-contrast indicators when a board region was read). These measurements are not calibrated recognition probabilities. `lock_state` is `locked`, `held`, `searching`, or `reacquiring`. A held boundary has `geometry_held: true` and `lock_age_ms`, expires after eight seconds without new board evidence, and carries no FEN, move, or overlay. `lock_released: true` tells clients to discard expired corners. Draw advice only for a verified current position, never for a held boundary.

The cache includes piece placement, turn, castling rights, en passant state, the halfmove clock, player color, depth, and mode. Changing those settings triggers fresh advice. Camera tracking advances only from a validated position, and the engine result is checked against its legal moves.

## `POST /api/analyze-fen`

Evaluates a legal FEN directly without camera detection or a tracking session:

```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "depth": 10,
  "mode": "win"
}
```

The endpoint returns `ok`, `game_over`, `invalid_fen`, or `engine_error` and the corresponding move fields. It uses the side to move from the FEN. Optional `player_color` is accepted for compatibility and does not change that side. Confidence is `1.0` for supplied-position input; it is not a camera recognition score.

## `POST /api/reset-session?session_id=...`

Clears the trusted position, cached result, board geometry, orientation, and request timer for one session:

```json
{"status": "reset", "session_id": "camera-session"}
```

The frontend can also begin a new game by creating a new session identifier. Sessions are kept in process memory, expire after 30 minutes of inactivity, and are lost when the backend restarts.

## `GET /api/depth-info`

Returns the legacy `depths` list with entries containing `depth` and `approx_elo`, plus a note that the estimates are not official ratings. The frontend presents depth as Quick, Balanced, or Deeper rather than promising an Elo rating.

## HTTP errors

| Code | Meaning |
| --- | --- |
| 400 | Missing input, invalid base64, or illegal FEN. |
| 413 | Decoded image exceeds 6 MB. |
| 422 | Request field validation failed, such as invalid color, depth, or corner coordinates. |
| 429 | Per-session request interval reached. |
| 503 | Engine unavailable or all tracking-session slots busy. |

Read `message` for structured status errors or `detail` for FastAPI input errors. Do not reuse a stale arrow after an uncertain position or failed analysis.
