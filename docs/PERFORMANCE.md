# Performance and operating limits

The application favors prompt, bounded analysis and reliable state transitions. These notes describe implementation choices, not measured guarantees for a particular phone, network, board, or Railway plan.

## Camera and network

The frontend captures a 960-by-960 camera JPEG or a 1280-by-1280 uploaded photo at quality `0.94`. For a camera scan it samples three frames 75 ms apart and sends the one with the most detail. Continuous focus and exposure are requested when the camera supports them. The complete native camera frame is scaled to fit and letterboxed, so a landscape or portrait source keeps all visible board corners. The square view and captured image use the same alignment.

Camera scans are scheduled after the preceding request completes, with an 850 ms interval. Burst capture and processing add to this interval. This prevents overlapping requests from building a backlog. Requests time out after 15 seconds, and generation checks cancel unfinished captures and discard responses from an old camera, photo, or settings state.

The previous partial-image hash shortcut has been removed. Hashing a JPEG prefix can miss changed squares; every scheduled frame now reaches detection. An unchanged validated position still avoids an engine search through the server's result cache.

Scanning pauses when the tab is hidden. Camera tracks are stopped when the user stops the camera or leaves the page. Payload size and elapsed time depend on image content and network conditions; there is no fixed bandwidth-per-move guarantee.

## Vision

The detector limits incoming images to a maximum dimension of 960 pixels and warps a verified board to 480-by-480 for square measurements. It checks an 8-by-8 alternating pattern before accepting an automatic candidate, using contours, line intersections, or internal grid corners. Spatial illumination and contrast estimates accommodate uneven lighting; central foreground measurements reduce false pieces caused by blurred square edges.

A measured board boundary can be held for up to eight seconds after detection fails. Held outlines are marked as refocusing and never justify a move suggestion. The server releases expired geometry even when the client keeps submitting old corners. Manual corners describe a user-selected region and do not prove that its pieces have been read.

This pipeline has no learned piece-type model. Its valid operating path is:

1. Calibrate the standard starting arrangement or an exact supplied FEN.
2. Measure occupied squares and piece colors in the next frame.
3. Match an unchanged position, one legal move, or two consecutive legal moves.
4. Keep the position uncommitted when no unique match exists.

Clear overhead views and visible square backgrounds work best. Moderate synthetic blur, low contrast, and uneven lighting have regression coverage; this does not guarantee all real camera conditions. Severe blur, glare, similar piece and board colors, hands over squares, partial boards, and missed move sequences can prevent a unique result. Different promotions can have the same occupancy and require explicit position confirmation.

Automatic starting calibration assumes that a matching starting occupancy contains the normal starting piece arrangement. Same-color piece types cannot be distinguished from occupancy alone.

## Chess engine

Stockfish is kept in a persistent UCI process with one thread and a 32 MB hash table. A lock serializes commands to that process, while FastAPI runs detector and engine work in a thread pool so health and unrelated API requests can proceed.

Stockfish search uses both the selected depth and a 350 ms search budget. Weaker mode can perform an additional short candidate analysis. Process startup, UCI communication, detection, queueing, and network time are additional overhead; the search budget is not a whole-request latency guarantee.

Cold engine initialization allows up to 15 seconds to load the embedded neural network and confirm readiness. Once initialized, ordinary engine operations use a three-second timeout. This avoids permanently switching a slow-starting container to the fallback engine.

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
