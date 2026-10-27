"""Simulator: synthetic two-camera throws in the exact API format (docs/MATH.md section 9).

    python simulate.py --check 20                 # 20 random throws of each kind through the solver; pass/fail
    python simulate.py --out data/sim_001 --mode wall   # write a session dir solver.py can read offline
    python simulate.py --post http://127.0.0.1:8000 --mode arc --realtime   # stream to a running server
    python simulate.py --sweep                    # error vs. number of cameras / angle (slides)

Scene (board frame, up = [0, -1, 0], phones at negative z in front of the screen at z = 0):
two cameras ~3 m apart, 0.3 m above the board origin, yawed ~30 deg toward the throw volume, nominal
K for 1920x1080 and no distortion. Throws start ~2 m in front of the board; the "wall" is the plane
z = 0 (the wall the laptop stands against). Each camera samples at its own fps with a random phase,
its own clock error (+-40 ms, unknown to the solver), 1.5 px noise and 5 % dropped frames.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import calibrate
import solver

W, H = 1920, 1080
G = 9.81
UP = np.array([0.0, -1.0, 0.0])
DOWN = -UP
BALL_R = 0.035
FLOOR_Y = 0.8           # floor is 0.8 m below the board origin (laptop on a table)


# ---------------------------------------------------------------- cameras

def look_at(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """T_board_cam (camera -> board) for a camera at cam_pos looking at target, image y pointing down."""
    z = target - cam_pos
    z = z / np.linalg.norm(z)
    x = np.cross(DOWN, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    T[:3, 3] = cam_pos
    return T


def make_cameras(n: int = 2, separation: float = 3.0, angle_deg: Optional[float] = None,
                 fps: Optional[List[float]] = None, rng: Optional[np.random.Generator] = None) -> Dict[str, Dict[str, Any]]:
    """n cameras on an arc around the throw volume center. For n = 2 the baseline is `separation`
    meters (or set by angle_deg, the angle subtended at the volume center)."""
    rng = rng or np.random.default_rng()
    K, dist = calibrate.nominal_intrinsics((W, H), 69.0)
    center = np.array([0.3, -1.1, -1.3])          # throw volume center (1.1 m above the board origin)
    radius = 3.2
    if angle_deg is None:
        angle_deg = math.degrees(2 * math.asin(min(1.0, separation / (2 * radius))))
    cams = {}
    for i in range(n):
        if n == 1:
            a = 0.0
        else:
            a = math.radians(-angle_deg / 2 + angle_deg * i / (n - 1))
        pos = center + radius * np.array([math.sin(a), 0.0, -math.cos(a)])
        pos[1] = -0.3                              # 0.3 m above the board origin
        T = look_at(pos, center)
        f = (fps[i] if fps and i < len(fps) else float(rng.choice([30, 60])))
        cams["sim%d" % (i + 1)] = {"K": K.tolist(), "dist": dist.tolist(), "T_board_cam": T.tolist(),
                                    "image_size": [W, H], "position": pos.tolist(),
                                    "forward": T[:3, 2].tolist(), "fps": f, "nominal": True, "rms_px": None}
    return cams


# ---------------------------------------------------------------- throws

def make_throw(mode: str, rng: np.random.Generator, restitution: float = 0.7, dt: float = 1 / 1000.0) -> Dict[str, Any]:
    """Ground-truth trajectory sampled at 1 kHz. Returns dict with T, P, and truth metrics."""
    if mode == "drop":
        p0 = np.array([rng.uniform(-0.3, 0.5), rng.uniform(-1.3, -0.9), rng.uniform(-1.8, -1.0)])  # hand height
        v0 = rng.normal(0, 0.05, 3)
    else:
        p0 = np.array([rng.uniform(-1.2, -0.6), rng.uniform(-0.6, -0.2), rng.uniform(-2.0, -1.6)])
        speed = rng.uniform(4.0, 8.0)                     # an underhand lob, peak ~1-2 m
        elev = math.radians(rng.uniform(35, 65))
        if mode == "wall":
            az = math.radians(rng.uniform(-20, 20))       # toward +z (the wall), slight x spread
            hdir = np.array([math.sin(az), 0.0, math.cos(az)])
        else:
            az = math.radians(rng.uniform(-25, 25))       # across the volume along +x
            hdir = np.array([math.cos(az), 0.0, math.sin(az)])
        v0 = speed * (math.cos(elev) * hdir + math.sin(elev) * UP)
    a = G * DOWN
    T, P = [0.0], [p0.copy()]
    pos, vel, t = p0.copy(), v0.copy(), 0.0
    bounced = None
    while t < 3.0:
        vel = vel + a * dt
        pos = pos + vel * dt
        t += dt
        if mode == "wall" and bounced is None and pos[2] >= -BALL_R and vel[2] > 0:
            v_in = vel.copy()
            vel[2] = -restitution * vel[2]
            bounced = {"t": t, "point": pos.copy(), "v_in": v_in, "v_out": vel.copy()}
        if pos[1] >= FLOOR_Y - BALL_R:          # hit the floor (y grows downward)
            break
        T.append(t)
        P.append(pos.copy())
    T, P = np.array(T), np.array(P)
    hspeed = np.linalg.norm(v0 - np.dot(v0, UP) * UP)
    return {"mode": mode, "T": T, "P": P, "p0": p0, "v0": v0, "g": G, "restitution": restitution if bounced else None,
            "bounce": bounced, "speed": float(np.linalg.norm(v0)),
            "launch_angle_deg": math.degrees(math.asin(np.dot(v0, UP) / max(np.linalg.norm(v0), 1e-9))),
            "apex_height_m": float(max(0.0, np.dot(v0, UP) ** 2 / (2 * G))), "hspeed": float(hspeed)}


# ---------------------------------------------------------------- cameras observe the throw

def observe(throw: Dict[str, Any], cams: Dict[str, Dict[str, Any]], rng: np.random.Generator, t_server0: float,
            noise_px: float = 1.5, drop_frac: float = 0.05, clock_err_s: float = 0.040,
            batch_s: float = 0.5) -> Dict[str, List[Dict[str, Any]]]:
    """Returns device -> list of /detections payloads (with a 'received_at' field like app.py logs)."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for dev, cam in cams.items():
        fps = cam["fps"]
        true_offset = rng.uniform(1e5, 2e5)                    # phone clock = server clock - true_offset
        reported_offset = true_offset + rng.uniform(-clock_err_s, clock_err_s)
        cam["_clock_err_s"] = reported_offset - true_offset   # truth, for --check only
        phase = rng.uniform(0, 1 / fps)
        t_frames = np.arange(phase, throw["T"][-1], 1 / fps)
        P = np.stack([np.interp(t_frames, throw["T"], throw["P"][:, k]) for k in range(3)], axis=1)
        uv, depth = calibrate.project_points(P, cam["K"], cam["dist"], cam["T_board_cam"])
        uv = uv + rng.normal(0, noise_px, uv.shape)
        f = cam["K"][0][0]
        frames = []
        for k in range(len(t_frames)):
            if depth[k] < 0.3 or not (0 <= uv[k, 0] < W and 0 <= uv[k, 1] < H):
                continue
            if rng.uniform() < drop_frac:
                continue
            r_px = BALL_R * f / depth[k]
            t_server = t_server0 + t_frames[k]
            frames.append({"t": round(t_server - true_offset, 4), "u": round(float(uv[k, 0]), 1), "v": round(float(uv[k, 1]), 1),
                           "n": int(math.pi * r_px * r_px), "w": int(2 * r_px), "h": int(2 * r_px)})
        # batch every 0.5 s of phone time, like the capture page
        batches = []
        if frames:
            t_first = frames[0]["t"]
            n_batches = int((frames[-1]["t"] - t_first) / batch_s) + 1
            for bi in range(n_batches):
                lo, hi = t_first + bi * batch_s, t_first + (bi + 1) * batch_s
                fr = [x for x in frames if lo <= x["t"] < hi]
                batches.append({"device_id": dev, "clock_offset": round(reported_offset, 4), "image_size": [W, H],
                                "fps_reported": fps, "frames": fr, "received_at": round(hi + reported_offset, 4)})
        out[dev] = batches
    return out


def to_buffers(payloads: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    buffers: Dict[str, List[Dict[str, Any]]] = {}
    for dev, batches in payloads.items():
        for b in batches:
            for fr in b["frames"]:
                fr = dict(fr)
                fr["t"] = fr["t"] + b["clock_offset"]
                buffers.setdefault(dev, []).append(fr)
    return buffers


# ---------------------------------------------------------------- modes

def run_check(n_each: int, seed: int, verbose: bool) -> int:
    rng = np.random.default_rng(seed)
    fails = 0
    marginal = 0
    results: Dict[str, List[Dict[str, Any]]] = {}
    for mode in ("drop", "arc", "wall"):
        for k in range(n_each):
            cams = make_cameras(rng=rng)
            throw = make_throw(mode, rng)
            payloads = observe(throw, cams, rng, t_server0=1e9 + 100.0 * k)
            res = solver.solve_latest(to_buffers(payloads), cams, UP)
            errs = [c["_clock_err_s"] for c in cams.values()]
            nframes = [sum(len(b["frames"]) for b in payloads[d]) for d in cams]
            row: Dict[str, Any] = {"mode": mode, "k": k, "fps": [c["fps"] for c in cams.values()],
                                   "frames": nframes, "dt_true_ms": round(1000 * (errs[1] - errs[0]), 1)}
            # frames each camera saw after the bounce (server time); both need >= 10 for a bounce to be solvable
            after = []
            if throw["bounce"]:
                for d, c in cams.items():
                    tb = throw["bounce"]["t"] + 1e9 + 100.0 * k + c["_clock_err_s"]
                    after.append(sum(1 for b in payloads[d] for f in b["frames"] if f["t"] + b["clock_offset"] > tb))
            bounce_visible = bool(after) and min(after) >= 10
            if min(nframes) < solver.PARAMS["min_points"]:
                row["skipped"] = "coverage: a camera saw only %d frames" % min(nframes)
            elif not res["throws"]:
                row["error"] = "no throw found"
                fails += 1
            else:
                th = res["throws"][0]
                f0 = th["segments"][0]["fit"]
                row.update({"g": f0["g_fit"], "g_err": round(f0["g_fit"] - G, 3), "dir_err": f0["g_dir_err_deg"],
                            "rms_cm": f0["rms_residual_cm"], "dt_ms": th["dt_refined_ms"], "n": th["n_points"],
                            "speed_err": round(f0["speed_mps"] - throw["speed"], 3),
                            "angle_err": round(f0["launch_angle_deg"] - throw["launch_angle_deg"], 2),
                            "apex_err": round(f0["apex_height_m"] - throw["apex_height_m"], 3),
                            "is_drop": f0["is_drop"], "n_seg": len(th["segments"]), "n_throws": len(res["throws"])})
                row["g_sigma"] = f0["g_sigma"]
                # pass: within 0.2, or consistent with the fit's own uncertainty on a short leg
                tol = max(0.2, 2.5 * f0["g_sigma"]) if f0["flight_time_s"] < 0.5 else 0.2
                bad = abs(row["g_err"]) > tol or row["dir_err"] > 5
                if mode == "wall" and not bounce_visible:
                    row["note"] = "bounce not visible to both cameras (%s frames after)" % after
                    bad = bad or len(res["throws"]) != 1
                elif mode == "wall":
                    if th["bounce"] is None or len(th["segments"]) != 2:
                        row["error"] = "bounce not found"; bad = True
                    else:
                        row["e"] = th["bounce"]["restitution"]
                        row["e_err"] = round(row["e"] - throw["restitution"], 3)
                        row["g2"] = th["segments"][1]["fit"]["g_fit"]
                        f1 = th["segments"][1]["fit"]
                        tol2 = max(0.2, 2.5 * f1["g_sigma"]) if f1["flight_time_s"] < 0.5 else 0.2
                        bad = bad or abs(row["e_err"]) > 0.05 or abs(row["g2"] - G) > tol2
                elif len(th["segments"]) != 1 or len(res["throws"]) != 1:
                    row["error"] = "expected 1 segment / 1 throw"; bad = True
                if mode == "drop" and not f0["is_drop"]:
                    row["error"] = "drop not classified"; bad = True
                # gross = a wrong answer; marginal = outside the demo tolerance but within noise
                sig1 = f0["g_sigma"]
                sig2 = th["segments"][1]["fit"]["g_sigma"] if len(th["segments"]) > 1 else 0.0
                gross = bool(row.get("error")) or abs(row.get("g_err", 0)) > max(0.5, 3 * sig1) or row.get("dir_err", 0) > 15                     or abs(row.get("e_err", 0)) > 0.15 or abs(row.get("g2", G) - G) > max(0.5, 3 * sig2)
                if gross:
                    fails += 1
                    row["FAIL"] = "gross"
                elif bad:
                    marginal += 1
                    row["FAIL"] = "marginal"
            results.setdefault(mode, []).append(row)
            if verbose or row.get("FAIL") or row.get("error") or row.get("skipped"):
                print(json.dumps(row))
    for mode, rows in results.items():
        ok = [r for r in rows if "g_err" in r]
        if ok:
            ge = np.array([r["g_err"] for r in ok])
            line = "%-5s n=%d  g err: mean %+.3f  std %.3f  max|.| %.3f   dir err max %.1f deg" % (
                mode, len(ok), ge.mean(), ge.std(), np.abs(ge).max(), max(r["dir_err"] for r in ok))
            if mode == "wall" and any("e_err" in r for r in ok):
                ee = np.array([r["e_err"] for r in ok if "e_err" in r])
                line += "   restitution err: mean %+.3f max|.| %.3f" % (ee.mean(), np.abs(ee).max())
            print(line)
    skipped = sum(1 for rows in results.values() for r in rows if r.get("skipped"))
    solved = 3 * n_each - skipped
    ok = fails == 0 and marginal <= 0.1 * solved
    print("RESULT: %s  %d gross failures, %d marginal (outside +-0.2 g / +-0.05 e but within noise) of %d solved%s" % (
        "PASS" if ok else "FAIL", fails, marginal, solved, ", %d skipped for camera coverage" % skipped if skipped else ""))
    return 0 if ok else 1


def write_session(out_dir: str, mode: str, n_throws: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    cams = make_cameras(rng=rng)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump({"saved_at": time.time(), "session": os.path.basename(out_dir), "up": UP.tolist(),
                   "simulated": True, "devices": cams}, f, indent=1)
    truths = []
    t0 = time.time()
    for k in range(n_throws):
        throw = make_throw(mode, rng)
        payloads = observe(throw, cams, rng, t_server0=t0 + 5.0 * k)
        for dev, batches in payloads.items():
            with open(os.path.join(out_dir, "detections_%s.jsonl" % dev), "a", encoding="utf-8") as f:
                for b in batches:
                    f.write(json.dumps(b) + "\n")
        truths.append({"k": k, "mode": mode, "t_start": t0 + 5.0 * k, "speed": throw["speed"],
                       "launch_angle_deg": throw["launch_angle_deg"], "apex_height_m": throw["apex_height_m"],
                       "g": G, "restitution": throw["restitution"]})
    with open(os.path.join(out_dir, "truth.json"), "w", encoding="utf-8") as f:
        json.dump(truths, f, indent=1)
    print("wrote %s: %d %s throws, cameras %s" % (out_dir, n_throws, mode, list(cams)))


def post_session(base: str, mode: str, n_throws: int, seed: int, realtime: bool) -> None:
    """Stream to a running server. Real phones register through /lock with photos; the simulator
    instead registers its virtual cameras through the development-only POST /sim_cameras."""
    rng = np.random.default_rng(seed)
    cams = make_cameras(rng=rng)

    def post(path: str, obj: Any) -> Any:
        req = urllib.request.Request(base + path, data=json.dumps(obj).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())

    try:
        print("/sim_cameras ->", post("/sim_cameras", {"devices": cams}))
    except Exception as e:  # noqa: BLE001
        print("server has no /sim_cameras endpoint (%s); run app.py from this repo" % e)
        return
    with urllib.request.urlopen(base + "/time", timeout=10) as r:
        t_server = json.loads(r.read().decode())["server_time"]
    for k in range(n_throws):
        throw = make_throw(mode, rng)
        # not realtime: stamp the throws in the recent past so the server sees them as finished
        t0 = time.time() + 1.0 if realtime else t_server - 5.0 * (n_throws + 1) + 5.0 * k
        payloads = observe(throw, cams, rng, t_server0=t0)
        events = sorted(((b["received_at"], dev, b) for dev, bs in payloads.items() for b in bs), key=lambda x: x[0])
        for recv, dev, b in events:
            if realtime:
                delay = recv - time.time()
                if delay > 0:
                    time.sleep(delay)
            body = {k2: v for k2, v in b.items() if k2 != "received_at"}
            post("/detections", body)
        # a final empty batch after the quiet period so the server solves the throw
        if realtime:
            time.sleep(0.8)
        for dev in cams:
            post("/detections", {"device_id": dev, "clock_offset": payloads[dev][0]["clock_offset"] if payloads[dev] else 0,
                                 "image_size": [W, H], "fps_reported": cams[dev]["fps"], "frames": []})
        t_server += 5.0
        print("throw %d (%s) sent" % (k + 1, mode))
    with urllib.request.urlopen(base + "/latest", timeout=10) as r:
        latest = json.loads(r.read().decode())
    print("/latest: %d throws, drops %s" % (len(latest["throws"]), latest["drops"]))
    for th in latest["throws"]:
        f0 = th["segments"][0]["fit"]
        print("  throw %s: g %.2f (dir err %.1f deg) speed %.2f n=%d%s" % (
            th["id"], f0["g_fit"], f0["g_dir_err_deg"], f0["speed_mps"], th["n_points"],
            "  bounce e=%.2f" % th["bounce"]["restitution"] if th.get("bounce") else ""))


def run_sweep(seed: int, n_rep: int = 8) -> None:
    """Median 3D point error and g error vs number of cameras and vs the angle between two cameras.
    Needs an N-camera triangulation; for N > 2 this uses the stretch formula of MATH 5 on the first
    throw only, so it is a slides chart, not a demo path."""
    rng = np.random.default_rng(seed)
    print("angle sweep (2 cameras, 60 fps, g error / 3D residual):")
    for ang in (10, 20, 30, 45, 60, 90, 120):
        gerr, res = [], []
        for _ in range(n_rep):
            cams = make_cameras(2, angle_deg=ang, fps=[60, 60], rng=rng)
            throw = make_throw("arc", rng)
            out = solver.solve_latest(to_buffers(observe(throw, cams, rng, 1e9)), cams, UP)
            if out["throws"]:
                gerr.append(out["throws"][0]["segments"][0]["fit"]["g_fit"] - G)
                res.append(out["throws"][0]["segments"][0]["fit"]["rms_residual_cm"])
        if gerr:
            print("  angle %3d deg: |g err| median %.3f  rms residual %.1f cm  (%d/%d solved)" % (
                ang, float(np.median(np.abs(gerr))), float(np.median(res)), len(gerr), n_rep))
    print("camera-count sweep (N cameras on a 90 deg arc, 60 fps, N-ray least squares):")
    for n in (2, 3, 4, 6, 8):
        errs = []
        for _ in range(n_rep):
            cams = make_cameras(n, angle_deg=90, fps=[60] * n, rng=rng)
            throw = make_throw("arc", rng)
            errs.append(n_camera_error(throw, cams, rng))
        print("  N=%d: median 3D error %.2f cm" % (n, 100 * float(np.median(errs))))


def n_camera_error(throw: Dict[str, Any], cams: Dict[str, Dict[str, Any]], rng: np.random.Generator) -> float:
    """Triangulate with all cameras at common 60 fps sample times (perfect sync) and report the median
    distance to the true position: X = (sum(I - d d^T))^-1 sum((I - d d^T) C)."""
    t = np.arange(0.0, throw["T"][-1], 1 / 60.0)
    P = np.stack([np.interp(t, throw["T"], throw["P"][:, k]) for k in range(3)], axis=1)
    rays = []
    for cam in cams.values():
        uv, depth = calibrate.project_points(P, cam["K"], cam["dist"], cam["T_board_cam"])
        uv = uv + rng.normal(0, 1.5, uv.shape)
        C, D = calibrate.pixels_to_rays(uv, cam["K"], cam["dist"], cam["T_board_cam"])
        rays.append((C, D, depth > 0.3))
    errs = []
    for i in range(len(t)):
        A = np.zeros((3, 3)); b = np.zeros(3)
        for C, D, ok in rays:
            if not ok[i]:
                continue
            M = np.eye(3) - np.outer(D[i], D[i])
            A += M; b += M @ C
        if np.linalg.matrix_rank(A) == 3:
            errs.append(np.linalg.norm(np.linalg.solve(A, b) - P[i]))
    return float(np.median(errs)) if errs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", type=int, metavar="N", help="run N throws of each mode through the solver")
    ap.add_argument("--out", help="write a session directory")
    ap.add_argument("--post", help="stream to a running server, e.g. http://127.0.0.1:8000")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--mode", choices=["drop", "arc", "wall"], default="arc")
    ap.add_argument("--n", type=int, default=3, help="throws for --out / --post")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--realtime", action="store_true", help="pace --post like real phones")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    if a.check:
        return run_check(a.check, a.seed, a.v)
    if a.out:
        write_session(a.out, a.mode, a.n, a.seed)
        return 0
    if a.post:
        post_session(a.post.rstrip("/"), a.mode, a.n, a.seed, a.realtime)
        return 0
    if a.sweep:
        run_sweep(a.seed)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
