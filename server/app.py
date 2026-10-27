"""Vigil server: patients' phones post fingertip-PPG measurements; the nurse board ranks everyone in the room.
Run (TLS, phones):  uvicorn app:app --host 0.0.0.0 --port 8443 --ssl-certfile certs/cert.pem --ssl-keyfile certs/key.pem
"""
from __future__ import annotations

import json
import os
import secrets
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import vitals

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", time.strftime("%Y%m%d_%H%M%S"))
os.makedirs(DATA, exist_ok=True)

app = FastAPI(title="Vigil")
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


@app.middleware("http")
async def no_cache(request: Request, call_next):
    """Phones must always get the current page, not Safari's cached copy from an hour ago."""
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-store"
    return resp

PATIENTS: Dict[str, Dict[str, Any]] = {}
ROOM: Dict[str, Dict[str, Any]] = {}      # breathing rates from the room camera, keyed by seat label


def log(name: str, obj: Dict[str, Any]) -> None:
    with open(os.path.join(DATA, name + ".jsonl"), "a") as f:
        f.write(json.dumps({**obj, "received_at": time.time()}) + "\n")


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "board.html"))


@app.get("/time")
def get_time():
    return {"server_time": time.time()}


@app.get("/qr.png")
def qr_png(request: Request):
    """QR for the patient page, pointing at whatever host the board was opened on (hotspot IP)."""
    import io
    import qrcode
    host = request.headers.get("host", "172.20.10.2:8443").split(":")[0]
    if host in ("localhost", "127.0.0.1"):            # board opened on the Mac itself: phones need the hotspot IP
        host = os.environ.get("VIGIL_HOST", "172.20.10.2")
    url = f"https://{host}:8443/go"
    img = qrcode.make(url, box_size=6, border=1)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    from fastapi.responses import Response
    return Response(content=buf.getvalue(), media_type="image/png", headers={"X-Url": url})


@app.get("/go")
def go(request: Request):
    """QR landing. Chrome on iPhone cannot run the camera page, so hand it to Safari; everyone else goes straight in."""
    from fastapi.responses import HTMLResponse, RedirectResponse
    host = request.headers.get("host", "172.20.10.2:8443")
    target = f"https://{host}/static/patient.html"
    ua = request.headers.get("user-agent", "")
    if "CriOS" in ua or ("iPhone" in ua and "Safari" not in ua):
        return HTMLResponse(f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vigil</title><body style="margin:0;background:#fff;color:#15181c;font:18px -apple-system,system-ui;display:flex;min-height:100vh;align-items:center;justify-content:center;text-align:center;padding:24px">
<div><p style="font-size:22px;font-weight:600;margin:0 0 8px">Vigil needs Safari for the camera.</p>
<p style="color:#6b7380;margin:0 0 28px">One tap, no install.</p>
<a href="x-safari-{target}" style="display:inline-block;background:#15181c;color:#fff;text-decoration:none;font-weight:600;padding:16px 28px;border-radius:10px">Open in Safari</a>
<p style="color:#6b7380;font-size:14px;margin-top:28px">or type in Safari:<br>https://{host}/static/patient.html</p></div></body>""")
    return RedirectResponse(target, status_code=302)


class Join(BaseModel):
    name: str
    age: Optional[int] = None
    complaint: str = ""
    pain: Optional[int] = None
    seat: Optional[str] = None         # chair number from the check-in form


def _norm_seat(s: Optional[str]) -> Optional[str]:
    if not s or not str(s).strip():
        return None
    n = "".join(ch for ch in str(s) if ch.isdigit())
    return f"seat {int(n)}" if n else str(s).strip()[:16]


@app.post("/join")
def join(j: Join):
    pid = secrets.token_hex(3)
    seat = _norm_seat(j.seat)
    for q in PATIENTS.values():                     # one patient per chair
        if seat and q.get("seat") == seat:
            q["seat"] = None
    PATIENTS[pid] = {"id": pid, "name": j.name[:24], "age": j.age, "complaint": j.complaint[:200], "pain": j.pain,
                     "flags": vitals.complaint_flags(j.complaint), "joined": time.time(), "checks": [],
                     "recheck_requested": False, "glow": None, "seat": seat}
    log("join", {"id": pid, **j.dict()})
    return {"id": pid, "flags": PATIENTS[pid]["flags"], "seat": seat}


class Seat(BaseModel):
    seat: Optional[str] = None         # "3" or "seat 3"; null clears


@app.post("/seat/{pid}")
def set_seat(pid: str, s: Seat):
    """Link a patient to a numbered chair so the room camera's per-seat breathing is theirs.
    Set by the patient (chair number on the phone) or by the nurse on the board."""
    p = PATIENTS.get(pid)
    if not p:
        return JSONResponse({"ok": False}, status_code=404)
    seat = _norm_seat(s.seat)
    for q in PATIENTS.values():                     # one patient per chair
        if seat and q.get("seat") == seat and q["id"] != pid:
            q["seat"] = None
    p["seat"] = seat
    log("seat", {"id": pid, "seat": seat})
    return {"ok": True, "seat": seat}


def _live(p: Dict[str, Any], now: float) -> Optional[Dict[str, Any]]:
    """The room camera's current reading for this patient's chair, if fresh."""
    s = ROOM.get(p.get("seat") or "")
    if not s or now - (s.get("t") or 0) > 10:
        return None
    return {"rr": s.get("rr"), "apnea": bool(s.get("apnea")), "snr": s.get("snr"), "t": s.get("t")}


class Measure(BaseModel):
    id: str
    frames: List[List[float]]          # [t, R, G, B]
    crt_frames: Optional[List[List[float]]] = None
    crt_release_t: Optional[float] = None


@app.post("/measure")
def measure(m: Measure):
    p = PATIENTS.get(m.id)
    if not p:
        return JSONResponse({"ok": False, "reason": "unknown patient"}, status_code=404)
    res = vitals.analyze_ppg(m.frames)
    crt = vitals.analyze_crt(m.crt_frames, m.crt_release_t) if m.crt_frames else None
    check = {"t": time.time(), **{k: v for k, v in res.items() if k != "wave"}}
    if crt and crt.get("ok"):
        check["crt_s"] = crt["crt_s"]; check["crt_tau_s"] = crt.get("tau_s")
    p["checks"].append(check)
    p["recheck_requested"] = False
    log("measure", {"id": m.id, "n_frames": len(m.frames), "result": check, "crt": crt and {k: v for k, v in crt.items() if k != "curve"}})
    # keep raw frames for offline work
    with open(os.path.join(DATA, f"raw_{m.id}_{int(time.time())}.json"), "w") as f:
        json.dump({"frames": m.frames, "crt_frames": m.crt_frames, "crt_release_t": m.crt_release_t}, f)
    return {"ok": res.get("ok", False), "result": res, "crt": crt, "score": vitals.score_patient(p, time.time())}


@app.get("/me/{pid}")
def me(pid: str):
    p = PATIENTS.get(pid)
    if not p:
        return JSONResponse({"ok": False}, status_code=404)
    last = p["checks"][-1] if p["checks"] else None
    now = time.time()
    live = _live(p, now)
    return {"ok": True, "name": p["name"], "recheck_requested": p["recheck_requested"], "glow": p["glow"],
            "seat": p.get("seat"), "live": live,
            "last_check_t": last["t"] if last else None, "last": last, "score": vitals.score_patient({**p, "live": live}, now)}


@app.post("/recheck/{pid}")
def recheck(pid: str):
    if pid in PATIENTS:
        PATIENTS[pid]["recheck_requested"] = True
    return {"ok": pid in PATIENTS}


@app.post("/glow/{pid}")
def glow(pid: str, color: Optional[str] = None):
    if pid in PATIENTS:
        PATIENTS[pid]["glow"] = None if PATIENTS[pid]["glow"] else (color or "#ff7a00")
    return {"ok": True, "glow": PATIENTS.get(pid, {}).get("glow")}


@app.post("/seen/{pid}")
def seen(pid: str):
    """Nurse took the patient in: remove from the board."""
    PATIENTS.pop(pid, None)
    return {"ok": True}


class RoomUpdate(BaseModel):
    seats: Dict[str, Dict[str, Any]]    # label -> {rr, snr, amp, t}


@app.post("/room")
def room(r: RoomUpdate):
    ROOM.update(r.seats)
    return {"ok": True}


@app.get("/patients")
def patients():
    now = time.time()
    out = []
    for p in PATIENTS.values():
        live = _live(p, now)
        sc = vitals.score_patient({**p, "live": live}, now)
        ok_checks = [c for c in p["checks"] if c.get("ok")]
        last = p["checks"][-1] if p["checks"] else None
        out.append({"id": p["id"], "name": p["name"], "age": p["age"], "complaint": p["complaint"], "pain": p["pain"], "flags": p["flags"],
                    "seat": p.get("seat"), "live": live,
                    "joined": p["joined"], "last": last, "n_checks": len(p["checks"]), "recheck_requested": p["recheck_requested"],
                    "glow": p["glow"], "hr_series": [[c["t"], c.get("hr")] for c in ok_checks], "rr_series": [[c["t"], c.get("rr")] for c in ok_checks],
                    **sc, **vitals.risk({**p, "live": live}, ok_checks)})
    out.sort(key=lambda x: (-x["score"], -(now - (x["last"]["t"] if x["last"] else x["joined"]))))
    room = {k: {**s, "patient": next((q["name"] for q in PATIENTS.values() if q.get("seat") == k), None)} for k, s in ROOM.items()}
    return {"server_time": now, "patients": out, "room": room}


@app.post("/reset")
def reset():
    PATIENTS.clear(); ROOM.clear()
    return {"ok": True}
