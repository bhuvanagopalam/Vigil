"""HawkEye server: the seven endpoints of docs/API.md, JSONL logging, static files.

    uvicorn app:app --host 0.0.0.0 --port 8000                       # Windows dev (plain HTTP)
    uvicorn app:app --host 0.0.0.0 --port 8443 --ssl-certfile certs/cert.pem --ssl-keyfile certs/key.pem

State lives in memory plus JSONL logs under data/<session>/. Calibration (intrinsics + locks) is
also mirrored to data/calibration_latest.json and reloaded on start, so a server restart does not
force the phones to re-lock. POST /reset clears throws only; delete that file to forget calibration.

Solver hook: if solver.py defines `solve_latest(buffers, cameras, up, cfg)` it is called after each
/detections batch and must return {"throws": [...], "drops": {...}} in the /latest format. Until it
exists /latest reports cameras and baseline with empty throws.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import calibrate

try:  # optional until solver.py exists
    import solver  # type: ignore
except Exception:  # noqa: BLE001
    solver = None

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
STATIC = os.path.join(HERE, "static")
SESSION = time.strftime("%Y%m%d_%H%M%S")
SESSION_DIR = os.path.join(DATA, SESSION)
CALIB_LATEST = os.path.join(DATA, "calibration_latest.json")
BUFFER_SECONDS = 60.0
UP = [0.0, -1.0, 0.0]

os.makedirs(SESSION_DIR, exist_ok=True)
CFG = calibrate.load_config()

app = FastAPI(title="HawkEye", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_lock = threading.Lock()
# device_id -> {"image_size", "K", "dist", "rms_px", "n_images", "nominal", "T_board_cam",
#               "position", "forward", "reproj_px", "lock_time", "last_seen", "fps_reported"}
devices: Dict[str, Dict[str, Any]] = {}
# device_id -> deque of {"t": server_time, "u", "v", "n", "w", "h"} (rolling 60 s)
buffers: Dict[str, Deque[Dict[str, Any]]] = {}
# throws keyed by their start time (rounded to 0.1 s) so re-solving a batch updates, not duplicates
throws: Dict[float, Dict[str, Any]] = {}
drops: Dict[str, Any] = {"count": 0, "g_mean": None, "g_std": None}


# ---------------------------------------------------------------- logging / persistence

def log_jsonl(name: str, obj: Dict[str, Any]) -> None:
    rec = dict(obj)
    rec["received_at"] = time.time()
    with open(os.path.join(SESSION_DIR, name + ".jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def save_calibration() -> None:
    out = {"saved_at": time.time(), "session": SESSION, "config": CFG, "up": UP,
           "devices": {d: {k: v for k, v in info.items() if k not in ("last_seen", "fps_reported")}
                       for d, info in devices.items()}}
    txt = json.dumps(out, indent=1)
    for path in (os.path.join(SESSION_DIR, "calibration.json"), CALIB_LATEST):
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)


def load_calibration() -> None:
    if not os.path.exists(CALIB_LATEST):
        return
    try:
        with open(CALIB_LATEST, encoding="utf-8") as f:
            saved = json.load(f)
        for d, info in saved.get("devices", {}).items():
            devices[d] = dict(info)
        print("app: restored calibration for %s from %s" % (sorted(devices) or "no devices", CALIB_LATEST))
    except Exception as e:  # noqa: BLE001
        print("app: could not load %s: %s" % (CALIB_LATEST, e))


load_calibration()


def device(device_id: str) -> Dict[str, Any]:
    return devices.setdefault(device_id, {})


def calib_dir(device_id: str) -> str:
    d = os.path.join(SESSION_DIR, "calib", safe_id(device_id))
    os.makedirs(d, exist_ok=True)
    return d


def safe_id(device_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (device_id or "unknown"))[:40] or "unknown"


def parse_size(s: Optional[str], img: np.ndarray) -> List[int]:
    if s:
        try:
            w, h = s.lower().split("x")
            return [int(w), int(h)]
        except ValueError:
            pass
    h, w = img.shape[:2]
    return [w, h]


def current_baseline() -> Optional[float]:
    locked = [d for d in devices.values() if d.get("position")]
    if len(locked) < 2:
        return None
    return round(calibrate.baseline(locked[0]["position"], locked[1]["position"]), 4)


# ---------------------------------------------------------------- endpoints

@app.get("/time")
def get_time() -> Dict[str, float]:
    return {"server_time": time.time()}


@app.post("/calib_image")
async def post_calib_image(device_id: str = Form(...), image: UploadFile = File(...),
                           image_size: Optional[str] = Form(None)) -> JSONResponse:
    data = await image.read()
    img = calibrate.decode_image(data)
    if img is None:
        return JSONResponse({"ok": False, "message": "could not decode JPEG"}, status_code=400)
    with _lock:
        d = calib_dir(device_id)
        n = len([x for x in os.listdir(d) if x.endswith(".jpg") and x[0].isdigit()]) + 1
        path = os.path.join(d, "%02d.jpg" % n)
        with open(path, "wb") as f:
            f.write(data)
        size = parse_size(image_size, img)
        corners = calibrate.find_corners(img, CFG["board_cols"], CFG["board_rows"])
        device(device_id)["image_size"] = size
        device(device_id)["last_seen"] = time.time()
    found = corners is not None
    log_jsonl("requests", {"path": "/calib_image", "device_id": device_id, "image_size": size,
                           "file": os.path.relpath(path, SESSION_DIR).replace("\\", "/"),
                           "corners_found": found, "bytes": len(data)})
    return JSONResponse({"ok": True, "count": n, "corners_found": found, "image_size": size})


class DeviceBody(BaseModel):
    device_id: str


@app.post("/intrinsics")
def post_intrinsics(body: DeviceBody) -> JSONResponse:
    device_id = body.device_id
    d = calib_dir(device_id)
    paths = sorted(os.path.join(d, x) for x in os.listdir(d) if x.endswith(".jpg") and x[0].isdigit())
    images = [im for im in (calibrate.decode_image(open(p, "rb").read()) for p in paths) if im is not None]
    res = calibrate.calibrate_intrinsics(images, CFG["board_cols"], CFG["board_rows"], CFG["square_size_m"])
    log_jsonl("requests", {"path": "/intrinsics", "device_id": device_id, "result": res})
    if res.get("ok"):
        with _lock:
            info = device(device_id)
            info.update({"image_size": res["image_size"], "K": res["K"], "dist": res["dist"],
                         "rms_px": res["rms_px"], "n_images": res["n_images"], "nominal": False})
            # a new K invalidates an old lock
            for k in ("T_board_cam", "position", "forward", "reproj_px", "lock_time"):
                info.pop(k, None)
            save_calibration()
    return JSONResponse(res)


@app.post("/lock")
async def post_lock(device_id: str = Form(...), image: UploadFile = File(...),
                    image_size: Optional[str] = Form(None)) -> JSONResponse:
    data = await image.read()
    img = calibrate.decode_image(data)
    if img is None:
        return JSONResponse({"ok": False, "message": "could not decode JPEG"}, status_code=400)
    size = parse_size(image_size, img)
    with _lock:
        info = device(device_id)
        nominal = "K" not in info
        if nominal:
            K, dist = calibrate.nominal_intrinsics(size, CFG["nominal_hfov_deg"])
        else:
            K, dist = np.array(info["K"]), np.array(info["dist"])
            if list(info.get("image_size", size)) != size:
                msg = "lock image is %dx%d but intrinsics were calibrated at %dx%d; redo /intrinsics" % (
                    size[0], size[1], info["image_size"][0], info["image_size"][1])
                log_jsonl("requests", {"path": "/lock", "device_id": device_id, "ok": False, "message": msg})
                return JSONResponse({"ok": False, "message": msg})
        res = calibrate.solve_lock(img, K, dist, CFG["board_cols"], CFG["board_rows"], CFG["square_size_m"])
        path = os.path.join(calib_dir(device_id), "lock.jpg")
        with open(path, "wb") as f:
            f.write(data)
        if res.get("ok"):
            info.update({"image_size": size, "T_board_cam": res["T_board_cam"], "position": res["camera_position"],
                         "forward": res["forward"], "reproj_px": res["reproj_px"], "lock_time": time.time(),
                         "nominal": nominal, "last_seen": time.time()})
            if nominal:
                info.update({"K": K.tolist(), "dist": dist.tolist(), "rms_px": None, "n_images": 0})
            save_calibration()
        bl = current_baseline()
    out = {"ok": bool(res.get("ok")), "nominal": nominal, "baseline_m": bl, "image_size": size}
    for k in ("camera_position", "forward", "reproj_px", "distance_to_board_m", "message"):
        if k in res:
            out[k] = res[k]
    log_jsonl("requests", {"path": "/lock", "device_id": device_id, "result": out})
    return JSONResponse(out)


class DetectionsBody(BaseModel):
    device_id: str
    clock_offset: float
    image_size: Optional[List[int]] = None
    fps_reported: Optional[float] = None
    frames: List[Dict[str, Any]] = []


@app.post("/detections")
def post_detections(body: DetectionsBody) -> Dict[str, Any]:
    now = time.time()
    rec = body.model_dump() if hasattr(body, "model_dump") else body.dict()  # pydantic v2 / v1
    log_jsonl("detections_" + safe_id(body.device_id), rec)
    with _lock:
        info = device(body.device_id)
        info["last_seen"] = now
        info["fps_reported"] = body.fps_reported
        if body.image_size:
            info["live_image_size"] = body.image_size
        buf = buffers.setdefault(body.device_id, deque())
        for fr in body.frames:
            try:
                buf.append({"t": float(fr["t"]) + body.clock_offset, "u": float(fr["u"]), "v": float(fr["v"]),
                            "n": int(fr.get("n", 0)), "w": float(fr.get("w", 0)), "h": float(fr.get("h", 0))})
            except (KeyError, TypeError, ValueError):
                continue
        cutoff = now - BUFFER_SECONDS
        while buf and buf[0]["t"] < cutoff:
            buf.popleft()
        n_buf = len(buf)
    _run_solver()
    return {"ok": True, "received": len(body.frames), "buffered": n_buf}


def _run_solver() -> None:
    """Call solver.solve_latest on the rolling buffers (MATH.md sections 3-7). Never raise into the request."""
    if solver is None or not hasattr(solver, "solve_latest"):
        return
    try:
        with _lock:
            snap_buffers = {d: list(b) for d, b in buffers.items()}
            snap_cams = {d: dict(v) for d, v in devices.items() if v.get("T_board_cam")}
        res = solver.solve_latest(snap_buffers, snap_cams, UP, CFG, now=time.time())  # type: ignore[attr-defined]
        if not res:
            return
        with _lock:
            for th in res.get("throws", []):
                key = round(th["t_start"], 1)
                # the same throw can shift by a sample when re-solved; merge onto a neighbour key
                for k in list(throws):
                    if abs(k - key) <= 0.3 and k != key:
                        del throws[k]
                throws[key] = th
            _renumber()
            drops.update(solver.drop_stats(list(throws.values())))  # type: ignore[attr-defined]
            log_jsonl("throws", {"throws": [throws[k] for k in sorted(throws)][-3:], "drops": dict(drops)})
    except Exception as e:  # noqa: BLE001
        print("solver error:", repr(e))


def _renumber() -> None:
    for i, k in enumerate(sorted(throws)):
        throws[k]["id"] = i + 1


def _throw_list() -> List[Dict[str, Any]]:
    return [throws[k] for k in sorted(throws)]


@app.get("/latest")
def get_latest() -> Dict[str, Any]:
    now = time.time()
    with _lock:
        cams = []
        for d, info in devices.items():
            cams.append({"device_id": d, "position": info.get("position"), "forward": info.get("forward"),
                         "locked": bool(info.get("T_board_cam")), "nominal": info.get("nominal"),
                         "age_s": (round(now - info["last_seen"], 2) if info.get("last_seen") else None),
                         "fps_reported": info.get("fps_reported"), "rms_px": info.get("rms_px"),
                         "reproj_px": info.get("reproj_px"), "n_calib_images": info.get("n_images", 0),
                         "buffered": len(buffers.get(d, ()))})
        return {"server_time": now, "session": SESSION, "cameras": cams, "baseline_m": current_baseline(),
                "up": UP, "throws": _throw_list(), "drops": dict(drops),
                "board": {"cols": CFG["board_cols"], "rows": CFG["board_rows"], "square_size_m": CFG["square_size_m"]}}


class SimCameras(BaseModel):
    devices: Dict[str, Dict[str, Any]]


@app.post("/sim_cameras")
def post_sim_cameras(body: SimCameras) -> Dict[str, Any]:
    """Development only: register virtual cameras from simulate.py --post (replaces real calibration)."""
    with _lock:
        devices.clear()
        buffers.clear()
        throws.clear()
        for d, info in body.devices.items():
            devices[d] = {k: v for k, v in info.items() if not k.startswith("_")}
            devices[d]["simulated"] = True
        save_calibration()
    log_jsonl("requests", {"path": "/sim_cameras", "devices": sorted(body.devices)})
    return {"ok": True, "devices": sorted(devices), "baseline_m": current_baseline()}


@app.post("/reset")
def post_reset() -> Dict[str, Any]:
    with _lock:
        throws.clear()
        drops.update({"count": 0, "g_mean": None, "g_std": None})
        for b in buffers.values():
            b.clear()
    log_jsonl("requests", {"path": "/reset"})
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index() -> Any:
    page = os.path.join(STATIC, "index.html")
    if os.path.exists(page):
        return FileResponse(page)
    return HTMLResponse("<html><body style='background:#111;color:#eee;font-family:sans-serif;padding:2em'>"
                        "<h2>HawkEye server is up</h2><p>Dashboard not built yet. "
                        "<a style='color:#9cf' href='/latest'>/latest</a> · "
                        "<a style='color:#9cf' href='/static/board.html'>board</a> · "
                        "<a style='color:#9cf' href='/static/capture.html'>capture</a></p></body></html>")


print("app: session %s  data -> %s" % (SESSION, SESSION_DIR))
