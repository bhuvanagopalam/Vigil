"""Solver: 2D detections from two locked cameras -> 3D flight, parabola fits, g, bounce metrics.

Implements docs/MATH.md sections 2-7. numpy + OpenCV only; no FastAPI. Python 3.9 compatible.

Offline:   python solver.py data/<session>/          # prints the /latest JSON
In-server: app.py calls solve_latest(buffers, cameras, up, cfg, now=time.time()) after each batch.

Conventions: times are server seconds (clock_offset already applied). Board frame; up = [0, -1, 0].
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import calibrate

PARAMS: Dict[str, float] = {
    "min_n": 20,                 # drop detections with fewer blob pixels          (MATH 3)
    "max_bbox_frac": 0.15,       # ... or a bbox wider/taller than this fraction of the image
    "gap_s": 0.25,               # a run ends after this silence
    "min_points": 12,            # per device per throw
    "min_disp_px": 80,           # 2D displacement over the run
    "quiet_s": 0.5,              # solve only once the run has been quiet this long (live mode)
    "dt_search_s": 0.080,        # time-offset refinement window (+-)                (MATH 4)
    "dt_step_s": 0.001,
    "max_dt_cost_m": 0.05,       # calibration is off if the best ray-ray distance exceeds this
    "max_resid_m": 0.10,         # triangulation residual                          (MATH 5)
    "min_depth_m": 0.3,
    "bounce_angle_deg": 60.0,    # velocity turn that counts as a bounce            (MATH 6)
    "bounce_min_speed": 1.0,
    "min_seg_points": 8,
    "ransac_k": 3.0,             # drop residuals > k * median, refit once           (MATH 7)
    "drop_max_hspeed": 0.5,      # drop classification
    "drop_min_vup": -3.0,            # the track may start a few frames after release
    "max_bounce_gap_m": 0.15,    # legs whose fits do not meet are separate throws, not a bounce
    "short_leg_sigma": 0.5,      # for bounce speeds, legs with g_sigma above this use a g-fixed fit            # the track may start a few frames after release
}


# ---------------------------------------------------------------- cameras

class Camera:
    def __init__(self, device_id: str, info: Dict[str, Any]):
        self.device_id = device_id
        self.K = np.asarray(info["K"], np.float64)
        self.dist = np.asarray(info.get("dist") or [0, 0, 0, 0, 0], np.float64).ravel()
        self.T = np.asarray(info["T_board_cam"], np.float64)
        self.C = self.T[:3, 3].copy()
        size = info.get("image_size") or info.get("live_image_size") or [1920, 1080]
        self.image_size = (int(size[0]), int(size[1]))

    def rays(self, uv: np.ndarray) -> np.ndarray:
        return calibrate.pixels_to_rays(uv, self.K, self.dist, self.T)[1]


def cameras_from(devices: Dict[str, Dict[str, Any]]) -> Dict[str, Camera]:
    return {d: Camera(d, info) for d, info in devices.items() if info.get("T_board_cam") and info.get("K")}


# ---------------------------------------------------------------- detections -> runs (MATH 3)

def clean(frames: Sequence[Dict[str, Any]], cam: Camera, p: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Filter and sort one device's detections. Returns (t (N,), uv (N, 2))."""
    W, H = cam.image_size
    rows = []
    for f in frames:
        try:
            if f.get("n", 0) < p["min_n"]:
                continue
            if f.get("w", 0) > p["max_bbox_frac"] * W or f.get("h", 0) > p["max_bbox_frac"] * H:
                continue
            rows.append((float(f["t"]), float(f["u"]), float(f["v"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not rows:
        return np.zeros(0), np.zeros((0, 2))
    arr = np.array(sorted(rows))
    # drop exact duplicate timestamps (keepalive re-sends)
    keep = np.concatenate([[True], np.diff(arr[:, 0]) > 1e-6])
    arr = arr[keep]
    return arr[:, 0], arr[:, 1:3]


def split_runs(t: np.ndarray, gap: float) -> List[np.ndarray]:
    if len(t) == 0:
        return []
    cuts = np.where(np.diff(t) > gap)[0] + 1
    return [idx for idx in np.split(np.arange(len(t)), cuts)]


def good_runs(t: np.ndarray, uv: np.ndarray, p: Dict[str, float]) -> List[np.ndarray]:
    out = []
    for idx in split_runs(t, p["gap_s"]):
        if len(idx) < p["min_points"]:
            continue
        disp = np.ptp(uv[idx], axis=0)
        if max(disp) < p["min_disp_px"]:
            continue
        out.append(idx)
    return out


def find_events(data: Dict[str, Tuple[np.ndarray, np.ndarray]], now: Optional[float],
                p: Dict[str, float]) -> List[Dict[str, np.ndarray]]:
    """Pair overlapping runs of the two busiest devices. Returns [{dev: idx, dev2: idx}, ...]."""
    devs = sorted(data, key=lambda d: -len(data[d][0]))[:2]
    if len(devs) < 2:
        return []
    a, b = devs
    runs_a = good_runs(data[a][0], data[a][1], p)
    runs_b = good_runs(data[b][0], data[b][1], p)
    events = []
    for ra in runs_a:
        ta0, ta1 = data[a][0][ra[0]], data[a][0][ra[-1]]
        for rb in runs_b:
            tb0, tb1 = data[b][0][rb[0]], data[b][0][rb[-1]]
            overlap = min(ta1, tb1) - max(ta0, tb0)
            if overlap < 0.2:
                continue
            if now is not None and max(ta1, tb1) > now - p["quiet_s"]:
                continue  # still in flight; solve on a later batch
            events.append({a: ra, b: rb})
    return events


# ---------------------------------------------------------------- geometry (MATH 4, 5)

def line_line_distance(C1: np.ndarray, D1: np.ndarray, C2: np.ndarray, D2: np.ndarray) -> np.ndarray:
    n = np.cross(D1, D2)
    nn = np.linalg.norm(n, axis=1)
    w0 = C1 - C2
    out = np.abs(np.sum(w0 * n, axis=1)) / np.maximum(nn, 1e-9)
    # parallel rays: distance of C2 from line 1
    par = nn < 1e-6
    if par.any():
        cross = np.cross(w0 * np.ones_like(D1), D1)
        out[par] = np.linalg.norm(cross[par], axis=1)
    return out


def interp_uv(t_query: np.ndarray, t: np.ndarray, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of pixel tracks; returns (uv_q, valid mask inside the track's time span)."""
    valid = (t_query >= t[0]) & (t_query <= t[-1])
    u = np.interp(t_query, t, uv[:, 0])
    v = np.interp(t_query, t, uv[:, 1])
    return np.stack([u, v], axis=1), valid


def refine_dt(tA: np.ndarray, uvA: np.ndarray, camA: Camera, tB: np.ndarray, uvB: np.ndarray, camB: Camera,
              p: Dict[str, float]) -> Tuple[float, float]:
    """MATH 4: dt such that B(t + dt) matches A(t). Returns (dt, cost_m at dt)."""
    DA = camA.rays(uvA)
    best = (0.0, float("inf"))
    for dt in np.arange(-p["dt_search_s"], p["dt_search_s"] + 1e-9, p["dt_step_s"]):
        uvq, ok = interp_uv(tA + dt, tB, uvB)
        if ok.sum() < 6:
            continue
        DB = camB.rays(uvq[ok])
        cost = float(np.median(line_line_distance(camA.C[None, :], DA[ok], camB.C[None, :], DB)))
        if cost < best[1]:
            best = (float(dt), cost)
    return best


def triangulate(C1: np.ndarray, D1: np.ndarray, C2: np.ndarray, D2: np.ndarray, p: Dict[str, float]):
    """MATH 5 closed form, vectorized. Returns (X (N,3), resid (N,), ok mask)."""
    w0 = (np.asarray(C1) - np.asarray(C2)).reshape(3)
    b = np.sum(D1 * D2, axis=1)
    d = D1 @ w0
    e = D2 @ w0
    den = 1.0 - b * b
    ok = den > 1e-4
    den = np.where(ok, den, 1.0)
    s = (b * e - d) / den
    t = (e - b * d) / den
    P1 = C1 + s[:, None] * D1
    P2 = C2 + t[:, None] * D2
    X = 0.5 * (P1 + P2)
    resid = np.linalg.norm(P1 - P2, axis=1)
    ok &= (resid <= p["max_resid_m"]) & (s >= p["min_depth_m"]) & (t >= p["min_depth_m"])
    return X, resid, ok


# ---------------------------------------------------------------- segmentation (MATH 6)

def fit_fixed_g(T: np.ndarray, X: np.ndarray, up: np.ndarray, g: float = 9.81) -> Optional[Fit]:
    """Velocity-only fit with the acceleration fixed to g downward; used for bounce speeds on legs
    too short to determine their own acceleration (never for the reported g)."""
    if len(T) < 3:
        return None
    t0 = float(T[0])
    tau = T - t0
    a = -g * up
    Y = X - 0.5 * np.outer(tau ** 2, a)
    A = np.stack([np.ones_like(tau), tau], axis=1)
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    r = np.linalg.norm(Y - A @ coef, axis=1)
    return Fit(t0, coef[0], coef[1], a, float(T[-1]), float(np.sqrt(np.mean(r ** 2))), len(T), len(T), 0.0)


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def median3(P: np.ndarray) -> np.ndarray:
    if len(P) < 3:
        return P.copy()
    Q = P.copy()
    Q[1:-1] = np.median(np.stack([P[:-2], P[1:-1], P[2:]]), axis=0)
    return Q


def split_at_bounces(T: np.ndarray, X: np.ndarray, p: Dict[str, float], corner_s: float = 0.0) -> List[np.ndarray]:
    """Index arrays of flight segments separated at sharp velocity turns.

    Samples within a camera frame of the impact are smeared, because the other camera's pixel track
    is interpolated straight across the corner. So the turn at sample i is measured between the mean
    velocity over samples [i-2k, i-k] and over [i+k, i+2k] (k ~ 50 ms of samples), skipping the
    smeared neighbours, and samples within `corner_s` of the impact are left out of both legs."""
    n = len(T)
    if n < 2 * p["min_seg_points"]:
        return [np.arange(n)] if n >= p["min_seg_points"] else []
    Pm = median3(X)
    frame = float(np.median(np.diff(T)))
    k = max(2, int(round(0.05 / max(frame, 1e-3))))
    ang = np.zeros(n)
    ok = np.zeros(n, bool)
    for i in range(k + 1, n - k - 1):
        b0, b1 = max(0, i - 2 * k), i - k
        a0, a1 = i + k, min(n - 1, i + 2 * k)
        v_before = (Pm[b1] - Pm[b0]) / (T[b1] - T[b0])
        v_after = (Pm[a1] - Pm[a0]) / (T[a1] - T[a0])
        if np.linalg.norm(v_before) > p["bounce_min_speed"] and np.linalg.norm(v_after) > p["bounce_min_speed"]:
            ang[i] = angle_deg(v_before, v_after)
            ok[i] = ang[i] > p["bounce_angle_deg"]
    cuts: List[int] = []
    i = 0
    while i < n:
        if ok[i]:
            j = i
            while j + 1 < n and ok[j + 1]:
                j += 1
            cuts.append(i + int(np.argmax(ang[i:j + 1])))   # the impact sample itself
            i = j + 1
        else:
            i += 1
    segs = []
    start = 0
    for c in cuts + [n]:
        if c < n:
            t_imp = T[c]
            idx = np.arange(start, c)
            idx = idx[np.abs(T[idx] - t_imp) >= corner_s]
            start = c + 1
            while start < n and T[start] - t_imp < corner_s:
                start += 1
        else:
            idx = np.arange(start, n)
        if len(idx) >= p["min_seg_points"]:
            segs.append(idx)
    return segs


# ---------------------------------------------------------------- parabola fit (MATH 7)

class Fit:
    def __init__(self, t0: float, p0: np.ndarray, v0: np.ndarray, a: np.ndarray, t_end: float,
                 rms_m: float, n_used: int, n_total: int, a_sigma: float = 0.0):
        self.t0, self.p0, self.v0, self.a, self.t_end = t0, p0, v0, a, t_end
        self.rms_m, self.n_used, self.n_total, self.a_sigma = rms_m, n_used, n_total, a_sigma

    def pos(self, t: Any) -> np.ndarray:
        tau = np.asarray(t, np.float64) - self.t0
        tau = tau[..., None]
        return self.p0 + self.v0 * tau + 0.5 * self.a * tau ** 2

    def vel(self, t: float) -> np.ndarray:
        return self.v0 + self.a * (t - self.t0)


def fit_parabola(T: np.ndarray, X: np.ndarray, p: Dict[str, float]) -> Optional[Fit]:
    if len(T) < 4:
        return None
    t0 = float(T[0])
    tau = T - t0
    A = np.stack([np.ones_like(tau), tau, 0.5 * tau ** 2], axis=1)
    keep = np.ones(len(T), bool)
    coef = None
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(A[keep], X[keep], rcond=None)
        r = np.linalg.norm(X - A @ coef, axis=1)
        med = float(np.median(r[keep]))
        new_keep = r <= max(p["ransac_k"] * med, 0.01)
        if new_keep.sum() < 4 or new_keep.sum() == keep.sum():
            keep = new_keep if new_keep.sum() >= 4 else keep
            break
        keep = new_keep
    r = np.linalg.norm(X - A @ coef, axis=1)
    rms = float(np.sqrt(np.mean(r[keep] ** 2)))
    # standard error of the acceleration along the fitted direction (per-axis noise = rms / sqrt(3))
    try:
        cov22 = float(np.linalg.inv(A[keep].T @ A[keep])[2, 2])
        dof = max(int(keep.sum()) - 3, 1)
        a_sigma = float(np.sqrt(cov22 * np.sum(r[keep] ** 2) / 3.0 / dof))
    except np.linalg.LinAlgError:
        a_sigma = float("nan")
    return Fit(t0, coef[0], coef[1], coef[2], float(T[-1]), rms, int(keep.sum()), len(T), a_sigma)


def segment_metrics(fit: Fit, up: np.ndarray, p: Dict[str, float]) -> Dict[str, Any]:
    v0, a = fit.v0, fit.a
    taus = np.linspace(0.0, fit.t_end - fit.t0, 60)
    traj = fit.pos(fit.t0 + taus)
    heights = (traj - fit.p0) @ up
    horiz = lambda v: v - np.dot(v, up) * up  # noqa: E731
    speed = float(np.linalg.norm(v0))
    hspeed = float(np.linalg.norm(horiz(v0)))
    vup = float(np.dot(v0, up))
    return {
        "p0": [round(float(x), 4) for x in fit.p0], "v0": [round(float(x), 4) for x in v0],
        "a": [round(float(x), 4) for x in a],
        "g_fit": round(float(np.linalg.norm(a)), 3),
        "g_sigma": round(fit.a_sigma, 3),
        "g_dir_err_deg": round(angle_deg(a, -up), 2),
        "rms_residual_cm": round(fit.rms_m * 100, 2),
        "speed_mps": round(speed, 3),
        "launch_angle_deg": round(math.degrees(math.asin(max(-1.0, min(1.0, vup / max(speed, 1e-9))))), 2),
        "apex_height_m": round(float(heights.max()), 3),
        "range_m": round(float(np.linalg.norm(horiz(traj[-1] - traj[0]))), 3),
        "flight_time_s": round(fit.t_end - fit.t0, 3),
        "n_points": fit.n_total, "n_used": fit.n_used,
        "is_drop": bool(hspeed < p["drop_max_hspeed"] and vup > p["drop_min_vup"]),
    }


def closest_approach(f1: Fit, f2: Fit, t_lo: float, t_hi: float) -> float:
    ts = np.linspace(t_lo, t_hi, 201)
    gap = np.linalg.norm(f1.pos(ts) - f2.pos(ts), axis=1)
    return float(ts[int(np.argmin(gap))])


def bounce_metrics(f1: Fit, f2: Fit, t_lo: float, t_hi: float) -> Dict[str, Any]:
    """Impact = where the two fits are closest between the last point of seg 1 and first of seg 2."""
    t_imp = closest_approach(f1, f2, t_lo, t_hi)
    gap = np.linalg.norm(f1.pos(t_imp) - f2.pos(t_imp))
    point = 0.5 * (f1.pos(t_imp) + f2.pos(t_imp))
    v_in, v_out = f1.vel(t_imp), f2.vel(t_imp)
    n = v_out - v_in
    n = n / max(np.linalg.norm(n), 1e-9)
    e = abs(float(np.dot(v_out, n))) / max(abs(float(np.dot(v_in, n))), 1e-9)
    return {"point": [round(float(x), 4) for x in point], "t": round(t_imp, 4),
            "speed_in_mps": round(float(np.linalg.norm(v_in)), 3), "speed_out_mps": round(float(np.linalg.norm(v_out)), 3),
            "normal": [round(float(x), 4) for x in n], "restitution": round(e, 3),
            "gap_cm": round(float(gap) * 100, 2)}


# ---------------------------------------------------------------- one throw end to end

def solve_event(tA: np.ndarray, uvA: np.ndarray, camA: Camera, tB: np.ndarray, uvB: np.ndarray, camB: Camera,
                up: np.ndarray, p: Dict[str, float]) -> List[Dict[str, Any]]:
    """One paired event -> a list of throws (usually one; two if the legs do not meet at a bounce)."""
    # the device with more samples defines the sample times; dt is always reported as the clock of
    # the alphabetically second device minus the first, whichever is the reference
    sign = 1.0 if camA.device_id < camB.device_id else -1.0
    if len(tB) > len(tA):
        tA, uvA, camA, tB, uvB, camB = tB, uvB, camB, tA, uvA, camA
        sign = -sign
    dt, cost = refine_dt(tA, uvA, camA, tB, uvB, camB, p)
    # B's corrected clock: tB - dt lines up with tA. Triangulate at A's sample times (B interpolated)
    # and at B's sample times (A interpolated), then merge; more samples, same geometry.
    Ts, Xs, Rs = [], [], []
    for (t1, uv1, c1, t2, uv2, c2) in ((tA, uvA, camA, tB - dt, uvB, camB), (tB - dt, uvB, camB, tA, uvA, camA)):
        uvq, ok = interp_uv(t1, t2, uv2)
        if ok.sum() < 4:
            continue
        D1 = c1.rays(uv1[ok])
        D2 = c2.rays(uvq[ok])
        X, resid, good = triangulate(c1.C[None, :], D1, c2.C[None, :], D2, p)
        Ts.append(t1[ok][good]); Xs.append(X[good]); Rs.append(resid[good])
    if not Ts:
        return []
    T = np.concatenate(Ts); X = np.concatenate(Xs); resid = np.concatenate(Rs)
    order = np.argsort(T, kind="stable")
    T, X, resid = T[order], X[order], resid[order]
    keep = np.concatenate([[True], np.diff(T) > 1e-4])     # coincident samples from both cameras
    T, X, resid = T[keep], X[keep], resid[keep]
    if len(T) < p["min_seg_points"]:
        return []
    # Smearing from interpolating across the impact lasts up to one frame of the slower camera.
    corner_s = 1.2 * max(float(np.median(np.diff(tA))) if len(tA) > 1 else 0.0,
                         float(np.median(np.diff(tB))) if len(tB) > 1 else 0.0)
    # Pass 1: coarse split with a generous exclusion; pass 2: locate each impact as the closest
    # approach of the neighbouring leg fits, then re-assign points with the tight exclusion and refit.
    coarse = split_at_bounces(T, X, p, 3.0 * corner_s)
    fits0 = [fit_parabola(T[idx], X[idx], p) for idx in coarse]
    bounds: List[Tuple[float, float]] = []
    for i, idx in enumerate(coarse):
        lo = float(T[idx[0]]) if i == 0 else bounds[-1][1]
        if i + 1 < len(coarse) and fits0[i] is not None and fits0[i + 1] is not None:
            t_imp = closest_approach(fits0[i], fits0[i + 1], float(T[idx[-1]]), float(T[coarse[i + 1][0]]))
            hi = t_imp
        else:
            hi = float(T[idx[-1]]) + 1e-6 if i + 1 == len(coarse) else float(T[coarse[i + 1][0]])
        bounds.append((lo, hi))
    segments = []
    fits: List[Tuple[Fit, np.ndarray]] = []
    for i, (lo, hi) in enumerate(bounds):
        lo_ex = lo + (corner_s if i > 0 else 0.0)
        hi_ex = hi - (corner_s if i + 1 < len(bounds) else 0.0)
        idx = np.where((T >= lo_ex) & (T <= hi_ex))[0]
        if len(idx) < p["min_seg_points"]:
            continue
        fit = fit_parabola(T[idx], X[idx], p)
        if fit is None:
            continue
        fits.append((fit, idx))
        segments.append({"points": [[round(float(x), 4), round(float(y), 4), round(float(z), 4), round(float(t), 4)]
                                    for (x, y, z), t in zip(X[idx], T[idx])],
                         "fit": segment_metrics(fit, up, p)})
    if not segments:
        return []
    common = {"dt_refined_ms": round(sign * dt * 1000, 1), "dt_cost_cm": round(cost * 100, 2),
              "ref_device": camA.device_id, "tri_resid_cm": round(float(np.median(resid)) * 100, 2),
              "calibration_suspect": bool(cost > p["max_dt_cost_m"])}
    # group consecutive legs into throws: a bounce joins two legs whose fits meet at the impact
    throws: List[Dict[str, Any]] = []
    groups: List[List[int]] = [[0]]
    for i in range(1, len(fits)):
        f1, i1 = fits[i - 1]
        f2, i2 = fits[i]
        t_imp = closest_approach(f1, f2, float(T[i1[-1]]), float(T[i2[0]]))
        gap = float(np.linalg.norm(f1.pos(t_imp) - f2.pos(t_imp)))
        if gap <= p["max_bounce_gap_m"] and len(groups[-1]) == 1:
            groups[-1].append(i)
        else:
            groups.append([i])
    for grp in groups:
        segs = [segments[i] for i in grp]
        idx_all = np.concatenate([fits[i][1] for i in grp])
        bounce = None
        if len(grp) == 2:
            (f1, i1), (f2, i2) = fits[grp[0]], fits[grp[1]]
            # legs too short to carry their own g use a g-fixed fit for the impact velocities
            g1 = f1 if f1.a_sigma <= p["short_leg_sigma"] else (fit_fixed_g(T[i1], X[i1], up) or f1)
            g2 = f2 if f2.a_sigma <= p["short_leg_sigma"] else (fit_fixed_g(T[i2], X[i2], up) or f2)
            bounce = bounce_metrics(g1, g2, float(T[i1[-1]]), float(T[i2[0]]))
        th = dict(common)
        th.update({"t_start": round(float(T[idx_all[0]]), 4), "t_end": round(float(T[idx_all[-1]]), 4),
                   "n_points": int(len(idx_all)), "segments": segs, "bounce": bounce})
        throws.append(th)
    return throws


def drop_stats(throws: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    gs = [t["segments"][0]["fit"]["g_fit"] for t in throws
          if len(t.get("segments", [])) == 1 and t["segments"][0]["fit"].get("is_drop")]
    if not gs:
        return {"count": 0, "g_mean": None, "g_std": None}
    return {"count": len(gs), "g_mean": round(float(np.mean(gs)), 3),
            "g_std": round(float(np.std(gs)), 3) if len(gs) > 1 else 0.0}


# ---------------------------------------------------------------- entry points

def solve_latest(buffers: Dict[str, Sequence[Dict[str, Any]]], devices: Dict[str, Dict[str, Any]],
                 up: Sequence[float], cfg: Optional[Dict[str, Any]] = None,
                 now: Optional[float] = None) -> Dict[str, Any]:
    """buffers: device -> detections with server-time t. devices: calibration dicts (app.py format).
    Returns {"throws": [...], "drops": {...}} in the /latest format; throw ids are assigned by the caller."""
    p = dict(PARAMS)
    p.update((cfg or {}).get("solver", {}))
    upv = np.asarray(up, np.float64)
    cams = cameras_from(devices)
    data = {d: clean(buffers.get(d, []), cams[d], p) for d in cams}
    data = {d: v for d, v in data.items() if len(v[0])}
    throws = []
    for ev in find_events(data, now, p):
        (a, ia), (b, ib) = list(ev.items())
        throws.extend(solve_event(data[a][0][ia], data[a][1][ia], cams[a], data[b][0][ib], data[b][1][ib], cams[b], upv, p))
    throws.sort(key=lambda t: t["t_start"])
    for i, t in enumerate(throws):
        t["id"] = i + 1
    return {"throws": throws, "drops": drop_stats(throws)}


def load_session(session_dir: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]], List[float]]:
    with open(os.path.join(session_dir, "calibration.json"), encoding="utf-8") as f:
        calib = json.load(f)
    devices = calib["devices"]
    up = calib.get("up", [0, -1, 0])
    buffers: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(glob.glob(os.path.join(session_dir, "detections_*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                dev = rec["device_id"]
                off = float(rec.get("clock_offset", 0.0))
                buf = buffers.setdefault(dev, [])
                for fr in rec.get("frames", []):
                    fr = dict(fr)
                    fr["t"] = float(fr["t"]) + off
                    buf.append(fr)
    return devices, buffers, up


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    devices, buffers, up = load_session(sys.argv[1])
    res = solve_latest(buffers, devices, up)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
