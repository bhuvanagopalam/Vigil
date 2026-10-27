"""OpenCV calibration: intrinsics from checkerboard photos, extrinsics (solvePnP) from the lock photo.

Implements docs/MATH.md section 1. Pure numpy + OpenCV; no FastAPI. Python 3.9 compatible.

Conventions (docs/API.md): the world frame is the board frame. Origin at the board's first inner
corner, +x along columns (right on the screen), +y along rows (down the screen), +z out of the
screen toward the phones, so "up" is [0, -1, 0] with the laptop lid vertical. Camera space is
OpenCV's: +x right, +y down, +z forward.

Command line (development):
    python calibrate.py intrinsics  IMG1.jpg IMG2.jpg ...   [--cols 9 --rows 6 --square 0.028]
    python calibrate.py lock        LOCK.jpg --K-json calib.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UP = np.array([0.0, -1.0, 0.0])


# ---------------------------------------------------------------- config / helpers

def load_config(path: Optional[str] = None) -> dict:
    path = path or os.path.join(HERE, "config.json")
    with open(path) as f:
        cfg = json.load(f)
    cfg.setdefault("board_cols", 9)
    cfg.setdefault("board_rows", 6)
    cfg.setdefault("square_size_m", 0.028)
    cfg.setdefault("nominal_hfov_deg", 69.0)
    return cfg


def object_points(cols: int, rows: int, square: float) -> np.ndarray:
    """(cols*rows, 3) board-frame coordinates of the inner corners: (i*s, j*s, 0), column-major
    within each row, matching the order findChessboardCorners returns (row by row)."""
    pts = np.zeros((rows * cols, 3), np.float32)
    k = 0
    for j in range(rows):
        for i in range(cols):
            pts[k] = (i * square, j * square, 0.0)
            k += 1
    return pts


def decode_image(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def find_corners(img: np.ndarray, cols: int, rows: int) -> Optional[np.ndarray]:
    """Return (cols*rows, 1, 2) float32 corner pixels, or None. Tries the robust SB detector first,
    then the classic detector with sub-pixel refinement."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = (cols, rows)
    found, corners = False, None
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray, size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if not found:
        found, corners = cv2.findChessboardCorners(
            gray, size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    if not found or corners is None:
        return None
    corners = np.asarray(corners, np.float32).reshape(-1, 1, 2)
    # Make the ordering deterministic: first corner top-left on the screen (smallest u+v), rows going
    # down. OpenCV may return the board rotated by 180 degrees depending on the view.
    first, last = corners[0, 0], corners[-1, 0]
    if first[0] + first[1] > last[0] + last[1]:
        corners = corners[::-1].copy()
    return corners


def nominal_intrinsics(image_size: Sequence[int], hfov_deg: float = 69.0) -> Tuple[np.ndarray, np.ndarray]:
    """MATH.md section 1b: fx = fy = W / (2 tan(hfov/2)), principal point at the center, no distortion."""
    w, h = int(image_size[0]), int(image_size[1])
    f = w / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], np.float64)
    dist = np.zeros(5, np.float64)
    return K, dist


# ---------------------------------------------------------------- intrinsics

def calibrate_intrinsics(images: Sequence[np.ndarray], cols: int, rows: int, square: float,
                         min_images: int = 5, max_rms_px: float = 1.5) -> dict:
    """cv2.calibrateCamera over the images where the board is found. Returns the API.md response
    dict for POST /intrinsics (ok, image_size, K, dist, rms_px, n_images, plus n_found/message)."""
    if not images:
        return {"ok": False, "message": "no calibration images", "n_images": 0}
    h, w = images[0].shape[:2]
    objp = object_points(cols, rows, square)
    obj_pts: List[np.ndarray] = []
    img_pts: List[np.ndarray] = []
    for img in images:
        if img.shape[:2] != (h, w):
            continue  # wrong resolution; must match the live stream
        c = find_corners(img, cols, rows)
        if c is not None:
            obj_pts.append(objp)
            img_pts.append(c)
    n = len(img_pts)
    if n < min_images:
        return {"ok": False, "n_images": n, "image_size": [w, h],
                "message": "board found in only %d of %d images; need %d" % (n, len(images), min_images)}
    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(obj_pts, img_pts, (w, h), None, None)
    dist = np.asarray(dist).ravel()[:5]
    out = {"ok": bool(rms <= max_rms_px), "image_size": [w, h], "K": K.tolist(),
           "dist": dist.tolist(), "rms_px": round(float(rms), 3), "n_images": n}
    if rms > max_rms_px:
        out["message"] = "rms %.2f px > %.1f px; retake photos with more tilt variety" % (rms, max_rms_px)
    return out


# ---------------------------------------------------------------- extrinsics (lock)

def solve_lock(img: np.ndarray, K: np.ndarray, dist: np.ndarray, cols: int, rows: int, square: float) -> dict:
    """solvePnP of the board in the lock photo. Returns T_board_cam (camera -> board), the camera
    center and forward axis in the board frame, and the corner reprojection error in px."""
    corners = find_corners(img, cols, rows)
    if corners is None:
        return {"ok": False, "message": "board not found in the lock photo"}
    objp = object_points(cols, rows, square)
    K = np.asarray(K, np.float64)
    dist = np.asarray(dist, np.float64).ravel()
    flags = getattr(cv2, "SOLVEPNP_IPPE", cv2.SOLVEPNP_ITERATIVE)
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=flags)
    if not ok:
        return {"ok": False, "message": "solvePnP failed"}
    # refine
    rvec, tvec = cv2.solvePnPRefineLM(objp, corners, K, dist, rvec, tvec)
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    reproj = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2, axis=1))))
    R_cb, _ = cv2.Rodrigues(rvec)                 # board -> camera
    T_cam_board = np.eye(4)
    T_cam_board[:3, :3] = R_cb
    T_cam_board[:3, 3] = tvec.ravel()
    T_board_cam = np.linalg.inv(T_cam_board)      # camera -> board
    C = T_board_cam[:3, 3]
    R = T_board_cam[:3, :3]
    forward = R @ np.array([0.0, 0.0, 1.0])
    return {"ok": True, "T_board_cam": T_board_cam.tolist(), "camera_position": [round(float(x), 4) for x in C],
            "forward": [round(float(x), 4) for x in forward], "reproj_px": round(reproj, 3),
            "rvec": rvec.ravel().tolist(), "tvec": tvec.ravel().tolist(),
            "distance_to_board_m": round(float(np.linalg.norm(C)), 3)}


def baseline(C1: Sequence[float], C2: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(C1, float) - np.asarray(C2, float)))


# ---------------------------------------------------------------- pixel -> ray (MATH.md section 2)

def pixel_to_ray(u: float, v: float, K: np.ndarray, dist: np.ndarray, T_board_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (C, d_world): camera center and unit direction in the board frame."""
    K = np.asarray(K, np.float64)
    dist = np.asarray(dist, np.float64).ravel()
    T = np.asarray(T_board_cam, np.float64)
    pts = np.array([[[u, v]]], np.float64)
    und = cv2.undistortPoints(pts, K, dist, P=K).reshape(2)
    d_cam = np.array([(und[0] - K[0, 2]) / K[0, 0], (und[1] - K[1, 2]) / K[1, 1], 1.0])
    d_cam /= np.linalg.norm(d_cam)
    d_world = T[:3, :3] @ d_cam
    return T[:3, 3].copy(), d_world


def pixels_to_rays(uv: np.ndarray, K: np.ndarray, dist: np.ndarray, T_board_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized pixel_to_ray. uv (N, 2) -> (C (3,), D (N, 3) unit directions in the board frame)."""
    K = np.asarray(K, np.float64)
    dist = np.asarray(dist, np.float64).ravel()
    T = np.asarray(T_board_cam, np.float64)
    uv = np.asarray(uv, np.float64).reshape(-1, 1, 2)
    if len(uv) == 0:
        return T[:3, 3].copy(), np.zeros((0, 3))
    und = cv2.undistortPoints(uv, K, dist, P=K).reshape(-1, 2)
    d = np.stack([(und[:, 0] - K[0, 2]) / K[0, 0], (und[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(und))], axis=1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return T[:3, 3].copy(), d @ T[:3, :3].T


def project_points(P: np.ndarray, K: np.ndarray, dist: np.ndarray, T_board_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of pixels_to_rays for the simulator: board-frame points (N, 3) -> (uv (N, 2), depth (N,))."""
    T = np.asarray(T_board_cam, np.float64)
    T_cam_board = np.linalg.inv(T)
    R, t = T_cam_board[:3, :3], T_cam_board[:3, 3]
    Pc = np.asarray(P, np.float64).reshape(-1, 3) @ R.T + t
    rvec, _ = cv2.Rodrigues(R)
    uv, _ = cv2.projectPoints(np.asarray(P, np.float64).reshape(-1, 3), rvec, t, np.asarray(K, np.float64),
                              np.asarray(dist, np.float64).ravel())
    return uv.reshape(-1, 2), Pc[:, 2]


# ---------------------------------------------------------------- CLI for development

def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["intrinsics", "lock", "corners"])
    ap.add_argument("images", nargs="+")
    ap.add_argument("--cols", type=int)
    ap.add_argument("--rows", type=int)
    ap.add_argument("--square", type=float)
    ap.add_argument("--K-json", help="json with K and dist (output of intrinsics mode) for lock mode")
    a = ap.parse_args()
    cfg = load_config()
    cols = a.cols or cfg["board_cols"]
    rows = a.rows or cfg["board_rows"]
    square = a.square or cfg["square_size_m"]
    imgs = [cv2.imread(p) for p in a.images]
    if a.mode == "corners":
        for p, im in zip(a.images, imgs):
            c = find_corners(im, cols, rows)
            print(p, "found" if c is not None else "NOT found")
        return
    if a.mode == "intrinsics":
        print(json.dumps(calibrate_intrinsics(imgs, cols, rows, square), indent=1))
        return
    if a.K_json:
        with open(a.K_json) as f:
            kd = json.load(f)
        K, dist = np.array(kd["K"]), np.array(kd["dist"])
    else:
        h, w = imgs[0].shape[:2]
        K, dist = nominal_intrinsics((w, h), cfg["nominal_hfov_deg"])
        print("(nominal intrinsics)")
    print(json.dumps(solve_lock(imgs[0], K, dist, cols, rows, square), indent=1))


if __name__ == "__main__":
    _main()
