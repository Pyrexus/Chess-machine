# Performance and operating limits

The application favors prompt, bounded analysis and reliable state transitions. These notes describe implementation choices, not measured guarantees for a particular phone, network, board, or Railway plan.

## Camera and network

The frontend captures a 720-by-720 JPEG at quality `0.9`. The complete native camera frame is scaled to fit and letterboxed, so a landscape or portrait source keeps all visible board corners. The square view and captured image use the same alignment.

Camera scans are scheduled after the preceding request completes, with a 1.25-second interval. This prevents overlapping requests from building a backlog. Requests time out after 15 seconds, and a generation check discards responses from an old camera, photo, or settings state.

The previous partial-image hash shortcut has been removed. Hashing a JPEG prefix can miss changed squares; every scheduled frame now reaches detection. An unchanged validated position still avoids an engine search through the server's result cache.

Scanning pauses when the tab is hidden. Camera tracks are stopped when the user stops the camera or leaves the page. Payload size and elapsed time depend on image content and network conditions; there is no fixed bandwidth-per-move guarantee.

## Vision

The detector limits incoming images to a maximum dimension of 960 pixels and warps a verified board to 480-by-480 for square measurements. It checks an 8-by-8 alternating pattern before accepting a candidate rectangle and uses adaptive foreground/color measurements for occupancy.

This pipeline has no learned piece-type model. Its valid operating path is:

1. Calibrate the standard starting arrangement or an exact supplied FEN.
2. Measure occupied squares and piece colors in the next frame.
3. Match an unchanged position, one legal move, or two consecutive legal moves.
4. Keep the position uncommitted when no unique match exists.

Clear overhead views and visible square backgrounds work best. Glare, shadows, similar piece and board colors, hands over squares, partial boards, and missed move sequences can prevent a unique result. Different promotions can have the same occupancy and require explicit position confirmation.

Automatic starting calibration assumes that a matching starting occupancy contains the normal starting piece arrangement. Same-color piece types cannot be distinguished from occupancy alone.

## Chess engine

Stockfish is kept in a persistent UCI process with one thread and a 32 MB hash table. A lock serializes commands to that process, while FastAPI runs detector and engine work in a thread pool so health and unrelated API requests can proceed.

Stockfish search uses both the selected depth and a 350 ms search budget. Weaker mode can perform an additional short candidate analysis. Process startup, UCI communication, detection, queueing, and network time are additional overhead; the search budget is not a whole-request latency guarantee.

When Stockfish cannot run, the Python evaluator performs a shallower search with a 300 ms target budget and returns legal advice. Responses label this fallback and omit an Elo estimate. Its recommendations should not be treated as Stockfish strength.

## Sessions and caching

The server caches successful advice using the chess state that affects legal moves: piece placement, side to move, castling rights, and en passant state, together with player color, depth, and mode. An unchanged position returns the complete cached suggestion and recalculated geometry for the current frame.

Failed engine calls do not store a successful-result cache entry. The validated camera position remains available so the next frame can retry it. Invalid or unclear detections cannot overwrite the trusted position.

Each session serializes its requests. Session state is in memory, with up to 512 sessions and a 30-minute inactivity expiry. A backend restart clears it. Multiple server processes or replicas do not share that state; keep a camera session on one backend instance unless shared storage is added.

The default minimum request interval is 0.5 seconds per session and can be changed with `MIN_INTERVAL_SEC`. The frontend's normal scan interval already exceeds this default.

## Validation

Run the detector, engine, and API regressions together:

```powershell
& .venv\Scripts\python.exe -m pytest backend/tests -q -p no:cacheprovider
npm test
npm run test:browser
```

Camera geometry tests and synthetic-board regressions verify expected transformations and state changes. They do not establish recognition accuracy across arbitrary physical chess sets. Evaluate real phone-camera captures from the intended board and lighting before making accuracy or latency claims.
