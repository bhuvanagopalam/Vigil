"""End-to-end smoke test of a running server with synthetic phones (no real phone needed).

Start the server first:  uvicorn app:app --port 8000
Then:                    python test_api.py [http://127.0.0.1:8000]

Two virtual phones each upload 8 calibration JPEGs, compute intrinsics, lock, stream a batch of
detections; then /latest, /reset, / and /static/board.html are checked. Uses only the standard
library for HTTP, so it also documents exactly what the capture page sends.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
import uuid

import cv2
import numpy as np

import test_calibrate as T

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return json.loads(r.read().decode())


def post_json(path: str, obj):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def post_multipart(path: str, fields: dict, jpeg: bytes):
    boundary = "----hawkeye" + uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += ("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n" % (boundary, k, v)).encode()
    body += ("--%s\r\nContent-Disposition: form-data; name=\"image\"; filename=\"frame.jpg\"\r\n"
             "Content-Type: image/jpeg\r\n\r\n" % boundary).encode() + jpeg + ("\r\n--%s--\r\n" % boundary).encode()
    req = urllib.request.Request(BASE + path, data=body, method="POST",
                                 headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def jpeg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    return buf.tobytes()


def main() -> int:
    fails = 0
    K = T.true_K()
    dist = np.array([-0.05, 0.01, 0.0, 0.0, 0.0])
    center = np.array([T.COLS * T.SQ / 2, T.ROWS * T.SQ / 2, 0.0])
    phones = {"simA": np.array([-1.5, -1.0, -2.0]), "simB": np.array([1.5 + T.COLS * T.SQ, -1.0, -2.2])}
    size = "%dx%d" % (T.W, T.H)

    print("GET /time ->", get("/time"))

    for dev, cam in phones.items():
        for k in range(8):
            d = T.rng.uniform(0.45, 0.9)
            ang = T.rng.uniform(-0.5, 0.5, size=2)
            pos = center + d * np.array([np.sin(ang[0]), np.sin(ang[1]), -np.cos(ang[0]) * np.cos(ang[1])])
            rvec, tvec = T.look_at_pose(pos, center + T.rng.normal(0, 0.02, 3), roll_deg=T.rng.uniform(-15, 15))
            r = post_multipart("/calib_image", {"device_id": dev, "image_size": size}, jpeg(T.render_board(K, rvec, tvec, dist)))
            if not (r.get("ok") and r.get("corners_found")):
                print("FAIL /calib_image", dev, r); fails += 1
        print("POST /calib_image x8 for", dev, "->", r)
        r = post_json("/intrinsics", {"device_id": dev})
        print("POST /intrinsics", dev, "->", {k: v for k, v in r.items() if k not in ("K", "dist")})
        if not r.get("ok") or r["rms_px"] > 0.5:
            print("FAIL intrinsics"); fails += 1
        rvec, tvec = T.look_at_pose(cam, center)
        r = post_multipart("/lock", {"device_id": dev, "image_size": size}, jpeg(T.render_board(K, rvec, tvec, dist)))
        print("POST /lock", dev, "->", r)
        err = np.linalg.norm(np.array(r.get("camera_position", [9, 9, 9])) - cam) if r.get("ok") else 9
        if not r.get("ok") or err > 0.02 or r.get("nominal"):
            print("FAIL lock position err %.3f m" % err); fails += 1

    bl_true = float(np.linalg.norm(phones["simA"] - phones["simB"]))
    if r.get("baseline_m") is None or abs(r["baseline_m"] - bl_true) > 0.05:
        print("FAIL baseline %s vs true %.3f" % (r.get("baseline_m"), bl_true)); fails += 1
    else:
        print("baseline %.3f m (true %.3f) OK" % (r["baseline_m"], bl_true))

    # detections: the exact payload the capture page sends (t = performance.now()/1000 + clock_offset)
    now = time.time()
    frames = [{"t": 100.0 + i / 60.0, "u": 800.0 + 5 * i, "v": 400.0 + 2 * i, "n": 300, "w": 20, "h": 21} for i in range(30)]
    r = post_json("/detections", {"device_id": "simA", "clock_offset": now - 100.0, "image_size": [T.W, T.H],
                                  "fps_reported": 60, "frames": frames})
    print("POST /detections ->", r)
    if not (r.get("ok") and r.get("received") == 30 and r.get("buffered") == 30):
        print("FAIL detections"); fails += 1
    r = post_json("/detections", {"device_id": "simB", "clock_offset": now - 100.0, "image_size": [T.W, T.H], "fps_reported": 30, "frames": []})
    if not r.get("ok"):
        print("FAIL empty detections"); fails += 1

    latest = get("/latest")
    print("GET /latest ->", json.dumps(latest)[:400], "...")
    cams = {c["device_id"]: c for c in latest["cameras"]}
    if set(cams) != set(phones) or not all(c["locked"] for c in cams.values()) or cams["simA"]["buffered"] != 30:
        print("FAIL /latest cameras", cams); fails += 1
    if latest["up"] != [0, -1, 0] or latest["baseline_m"] is None:
        print("FAIL /latest up/baseline"); fails += 1

    r = post_json("/reset", {})
    latest = get("/latest")
    if not r.get("ok") or latest["throws"] or latest["baseline_m"] is None or any(c["buffered"] for c in latest["cameras"]):
        print("FAIL /reset should clear throws+buffers but keep calibration", r, latest); fails += 1
    else:
        print("POST /reset OK (calibration kept, buffers cleared)")

    for path in ("/", "/static/board.html", "/static/capture.html"):
        with urllib.request.urlopen(BASE + path, timeout=10) as resp:
            body = resp.read()
        print("GET %s -> %d, %d bytes" % (path, resp.status, len(body)))
        if resp.status != 200 or b"<html" not in body.lower():
            print("FAIL", path); fails += 1

    print("RESULT:", "PASS" if fails == 0 else "%d FAILURES" % fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
