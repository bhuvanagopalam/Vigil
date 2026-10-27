"""Self-test for calibrate.py with synthetic photos: no phone needed.

Renders the 9x6 board through virtual 1920x1080 cameras with known K and known poses, then checks
that calibrate_intrinsics recovers K, that solve_lock recovers each camera's position, and that the
baseline between two locked cameras matches the truth within 1 cm. Also exercises pixel_to_ray.

    python test_calibrate.py
"""
from __future__ import annotations

import math
import sys

import cv2
import numpy as np

import calibrate as C

W, H = 1920, 1080
COLS, ROWS, SQ = 9, 6, 0.028
rng = np.random.default_rng(1)


def true_K() -> np.ndarray:
    return np.array([[1450.0, 0, 955.0], [0, 1450.0, 545.0], [0, 0, 1]])


def render_board(K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Draw the 10x7-square board (white margin around it) as seen by the camera, by projecting each
    square's four corners and filling the polygon. Simple, and exact enough for sub-pixel corners."""
    img = np.full((H, W), 128, np.uint8)
    # white background rectangle one square wider on each side
    bg = np.array([[-2.0 * SQ, -2.0 * SQ, 0], [(COLS + 1.0) * SQ, -2.0 * SQ, 0],
                   [(COLS + 1.0) * SQ, (ROWS + 1.0) * SQ, 0], [-2.0 * SQ, (ROWS + 1.0) * SQ, 0]], np.float64)
    p, _ = cv2.projectPoints(bg, rvec, tvec, K, dist)
    cv2.fillConvexPoly(img, np.round(p.reshape(-1, 2) * 16).astype(np.int32), 255, lineType=cv2.LINE_AA, shift=4)
    # 10x7 squares: square i spans [i*SQ, (i+1)*SQ], i = -1..COLS-1, so inner corners are 0..COLS-1
    for j in range(-1, ROWS):
        for i in range(-1, COLS):
            if (i + j) % 2 == 0:
                continue
            sq = np.array([[i * SQ, j * SQ, 0], [(i + 1) * SQ, j * SQ, 0],
                           [(i + 1) * SQ, (j + 1) * SQ, 0], [i * SQ, (j + 1) * SQ, 0]], np.float64)
            p, _ = cv2.projectPoints(sq, rvec, tvec, K, dist)
            cv2.fillConvexPoly(img, np.round(p.reshape(-1, 2) * 16).astype(np.int32), 0, lineType=cv2.LINE_AA, shift=4)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def look_at_pose(cam_pos: np.ndarray, target: np.ndarray, roll_deg: float = 0.0):
    """Camera at cam_pos looking at target (board frame). Returns rvec, tvec (board -> camera)."""
    z = target - cam_pos
    z /= np.linalg.norm(z)
    down = np.array([0.0, 1.0, 0.0])               # board +y is down the screen = camera +y
    x = np.cross(down, z)                          # right-handed: x = y cross z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_wc = np.stack([x, y, z], axis=1)             # camera -> world
    if roll_deg:
        c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
        R_wc = R_wc @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R_cw = R_wc.T
    t = -R_cw @ cam_pos
    rvec, _ = cv2.Rodrigues(R_cw)
    return rvec, t.reshape(3, 1)


def main() -> int:
    K = true_K()
    dist_true = np.array([-0.05, 0.01, 0.0, 0.0, 0.0])
    center = np.array([COLS * SQ / 2, ROWS * SQ / 2, 0.0])
    fails = 0

    # --- intrinsics from 8 varied close-up views
    imgs = []
    for k in range(8):
        d = rng.uniform(0.45, 0.9)
        ang = rng.uniform(-0.5, 0.5, size=2)
        pos = center + d * np.array([math.sin(ang[0]), math.sin(ang[1]), math.cos(ang[0]) * math.cos(ang[1])])
        rvec, tvec = look_at_pose(pos, center + rng.normal(0, 0.02, 3), roll_deg=rng.uniform(-15, 15))
        imgs.append(render_board(K, rvec, tvec, dist_true))
    res = C.calibrate_intrinsics(imgs, COLS, ROWS, SQ)
    print("intrinsics:", {k: v for k, v in res.items() if k not in ("K", "dist")})
    if not res.get("ok"):
        print("FAIL intrinsics not ok"); fails += 1
    else:
        Ke = np.array(res["K"])
        ferr = abs(Ke[0, 0] - K[0, 0]) / K[0, 0]
        print("  fx est %.1f true %.1f (%.2f%%), cx %.1f cy %.1f" % (Ke[0, 0], K[0, 0], 100 * ferr, Ke[0, 2], Ke[1, 2]))
        if ferr > 0.01:
            print("FAIL fx off by more than 1%"); fails += 1

    # --- lock two cameras ~3 m apart, 2 m from the board, like the demo
    # Board frame: +x right, +y down the screen, so +z points INTO the screen (right-handed) and the
    # phones, in front of the screen, sit at negative z. Up is [0, -1, 0].
    camA = np.array([-1.5, -1.0, -2.0])    # left, 1 m above board origin, 2 m in front
    camB = np.array([1.5 + COLS * SQ, -1.0, -2.2])
    Ke, de = (np.array(res["K"]), np.array(res["dist"])) if res.get("ok") else (K, dist_true)
    positions = []
    for name, cam in (("A", camA), ("B", camB)):
        rvec, tvec = look_at_pose(cam, center)
        img = render_board(K, rvec, tvec, dist_true)
        lock = C.solve_lock(img, Ke, de, COLS, ROWS, SQ)
        if not lock.get("ok"):
            print("FAIL lock", name, lock); fails += 1; continue
        err = np.linalg.norm(np.array(lock["camera_position"]) - cam)
        print("lock %s: pos %s true %s err %.3f m reproj %.2f px fwd %s" % (
            name, lock["camera_position"], cam.round(3).tolist(), err, lock["reproj_px"], lock["forward"]))
        if err > 0.02:
            print("FAIL camera position off by > 2 cm"); fails += 1
        positions.append((lock, cam))

        # pixel -> ray must pass through the board center
        p, _ = cv2.projectPoints(center.reshape(1, 3), rvec, tvec, K, dist_true)
        u, v = p.reshape(2)
        Cc, dw = C.pixel_to_ray(u, v, Ke, de, np.array(lock["T_board_cam"]))
        s = -Cc[2] / dw[2]                          # hit z = 0 plane
        hit = Cc + s * dw
        if np.linalg.norm(hit - center) > 0.01:
            print("FAIL ray through board center misses by %.3f m" % np.linalg.norm(hit - center)); fails += 1

    if len(positions) == 2:
        bl = C.baseline(positions[0][0]["camera_position"], positions[1][0]["camera_position"])
        bl_true = float(np.linalg.norm(camA - camB))
        print("baseline: %.3f m, true %.3f m (GATE 1 wants within 0.05)" % (bl, bl_true))
        if abs(bl - bl_true) > 0.01:
            print("FAIL baseline"); fails += 1

    # --- nominal intrinsics sanity
    Kn, _ = C.nominal_intrinsics((W, H), 69.0)
    print("nominal fx for 1920x1080 @69deg: %.0f" % Kn[0, 0])

    print("RESULT:", "PASS" if fails == 0 else "%d FAILURES" % fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
