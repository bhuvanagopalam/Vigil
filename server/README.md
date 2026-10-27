# server: Vigil

FastAPI app + signal analysis + static pages. Brief and task split: `../docs/VIGIL.md`.

```
app.py              endpoints: /join /measure /me/{id} /recheck/{id} /glow/{id} /seen/{id} /patients /room /reset
vitals.py           fingertip PPG -> HR, RR, HRV, perfusion, SQI; capillary refill; red flags; NEWS-style score
static/patient.html patient phone page (join -> 20 s pulse check -> refill test -> results -> 15-min cycle)
static/board.html   nurse board (served at /)
static/room.html    room camera: contactless breathing per seat (teammate; seed = breath_test.html)
static/breath_test.html, vitals_test.html   feasibility pages, keep
dev_server.py       old stand-in; not used anymore
data/               one folder per server start: JSONL logs + raw frames per measurement
certs/              mkcert cert/key for the hotspot IP (not committed)
```

## Run (Mac, phones on the hotspot)

```bash
cd server && source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8443 --ssl-certfile certs/cert.pem --ssl-keyfile certs/key.pem
# board:    https://172.20.10.2:8443/
# patients: https://172.20.10.2:8443/static/patient.html   (QR on the board)
# room cam: https://localhost:8443/static/room.html         (Mac webcam; accept the cert warning or `mkcert -install`)
```

Windows dev: `py -3 -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt && uvicorn app:app --port 8000`
(plain HTTP is fine for the board and for pages that don't open a camera; the camera pages need HTTPS or localhost).

## Offline analysis

Every `/measure` stores `data/<session>/raw_<patient>_<t>.json`. `python -c "import json,vitals; d=json.load(open('data/<...>.json')); print(vitals.analyze_ppg(d['frames']))"`.
