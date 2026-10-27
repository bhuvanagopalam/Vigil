# server — teammate's half

Read, in order: `../docs/API.md` (the contract), `../docs/MATH.md` (calibration + solver),
`../PLAN.md` §3 (what is due when). Everything you need is in those three files.

## Files you own

```
app.py             FastAPI: /time /calib_image /intrinsics /lock /detections /latest /reset + static
calibrate.py       OpenCV: intrinsics from photos, extrinsics (solvePnP) from the lock photo
solver.py          MATH.md §2–§7; runnable offline: python solver.py data/<session>/
simulate.py        synthetic two-camera throws conforming to the API (MATH.md §9)
static/index.html  dashboard (MATH.md §10)
static/board.html  chessboard page, 9×6 inner corners, as large as fits the screen
config.json        { "square_size_m": 0.028, "board_cols": 9, "board_rows": 6 }
```
`static/capture.html` is Rushendra's; don't edit it, just serve it.

## Build order

1. `app.py` with all endpoints, JSONL logging, static serving. Run on Windows with plain HTTP
   for development (`uvicorn app:app --port 8000`); TLS is only needed on the Mac for phones.
2. `static/board.html`.
3. `simulate.py` → `solver.py` until `g` comes back within ±0.2 on synthetic data.
4. `calibrate.py`: test intrinsics with 8 photos of the board taken with *your* phone's Camera
   app (any photos work for development), then `solvePnP` on one more. Print baseline when two
   devices are locked. Rushendra runs GATE 1 with this at 18:00.
5. `static/index.html`.
6. Real data arrives Saturday evening (`data/` committed). Expect one sign bug — MATH.md §8.
7. Δt refinement, bounce split, restitution, drop stats.
8. Slides chart: error vs number of cameras / angle (simulator sweep).

## Tests (no phone needed)

```powershell
python test_calibrate.py                 # synthetic board photos -> intrinsics, lock, baseline
python simulate.py --check 20            # 20 drops / arcs / wall throws through the solver; PASS/FAIL
python simulate.py --out data/sim_a --mode wall --n 3 ; python solver.py data/sim_a   # offline path
uvicorn app:app --port 8000              # then, in another shell:
python test_api.py                       # every endpoint with two virtual phones
python simulate.py --post http://127.0.0.1:8000 --mode drop --n 5 --realtime        # live path
python simulate.py --sweep               # error vs camera angle / count, for the slides
```

Expect on synthetic data: `g` within ±0.2 on drops and lobs, restitution within ±0.05, and a
reported `g_sigma` per leg. A leg shorter than ~0.3 s (the first leg of a hard wall throw) is
noise-limited: ±0.5 is normal there, and the dashboard shows the ± so the number stays honest.

## Windows setup

```powershell
cd server
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Constraints

- Python 3.9 compatible (the Mac has 3.9): no `match`, use `typing.Optional`, or add
  `from __future__ import annotations` for `list[...]`.
- Dependencies: fastapi, uvicorn, numpy, scipy, opencv-python-headless, python-multipart.
- No database, no auth, no websockets. In-memory state + JSONL logs.
- `solver.py` and `calibrate.py` import no FastAPI.

## Running on the Mac at the demo

`git pull`, `source .venv/bin/activate`, `pip install -r requirements.txt`,
`uvicorn app:app --host 0.0.0.0 --port 8443 --ssl-certfile certs/cert.pem --ssl-keyfile certs/key.pem`.
