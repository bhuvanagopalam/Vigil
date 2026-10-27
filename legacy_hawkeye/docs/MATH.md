# MATH — calibration and the solver, step by step

numpy + scipy + opencv-python. Must run offline on a session directory and inside the server.
Keep functions small; the simulator (§9) is the test.

## 1. Calibration (OpenCV, textbook)

**Board.** `server/static/board.html` draws a chessboard with **9×6 inner corners** (10×7 squares)
as large as fits the Mac screen. Square size is measured with a ruler (expect ≈ 0.028 m on a 14"
display) and entered in `config.json`. Lid vertical. Max brightness.

**1a. Intrinsics per phone.** 8 photos of the board, varied distance (0.4–1.2 m) and tilt (±30°),
board filling 1/3–1/2 of the frame. `cv2.findChessboardCornersSB` (or `findChessboardCorners` +
`cornerSubPix`), then `cv2.calibrateCamera` with a 5-parameter distortion model. Want RMS < 1 px.
Save `K`, `dist`, `image_size` per device. Must be the same stream resolution as live.

**1b. Nominal intrinsics (fallback only).** iPhone main camera, 16:9 stream: horizontal FOV ≈ 69°
→ `fx = fy ≈ W / (2·tan(34.5°)) ≈ 0.727·W`, `cx = W/2`, `cy = H/2`, `dist = 0`. ~3% scale error.

**1c. Extrinsics (Lock).** One photo per phone of the board at a fixed spot (1.5–2.5 m from each
phone, roughly facing both). `cv2.solvePnP(objectPoints, corners, K, dist, flags=SOLVEPNP_IPPE)`
→ `rvec, tvec` = `T_cam_board`. Then `T_board_cam = inv(T_cam_board)`; camera center
`C = T_board_cam[:3, 3]`, rotation `R = T_board_cam[:3, :3]` (camera → board).
Object points: `(i·s, j·s, 0)` for column `i`, row `j`, square size `s`. `up = [0, -1, 0]`.
Cameras come out at **negative z** (OpenCV's board z points into the screen); that is expected.
Use `flags=cv2.CALIB_FIX_K3` in `calibrateCamera` — with 8 views k3 overfits (seen: k3 = 4.1).
Reference numbers from Saturday (real data): intrinsics RMS 0.22–0.34 px; lock reprojection
0.26–0.43 px with the board 250–320 px wide in a 1920-px frame.
Reproject the corners and report the error (want < 1 px); if the baseline is off at GATE 1, this
number and the board detection at distance are the first suspects.

`baseline_m = |C_A − C_B|`.

## 2. Pixel → ray

```
(u', v') = cv2.undistortPoints([[u, v]], K, dist, P=K)     # undistorted pixel
d_cam    = normalize([(u' - cx)/fx, (v' - cy)/fy, 1.0])
d_world  = R @ d_cam
ray      = (C, d_world)
```

## 3. Throw-event detection

Per device: `t_server = t + clock_offset`. Rolling buffer. A **throw** is a window where both
devices have ≥ 12 detections within 1.5 s and 2D displacement > 80 px. Extend while detections
keep arriving (gap < 0.25 s). Solve once quiet for 0.5 s. Drop detections with `n` < 20 or
bbox > 15% of the image.

## 4. Time-offset refinement (Δt)

```
for dt in arange(-0.080, 0.080, 0.001):
    cost(dt) = median over A's frame times t of  ray-ray distance( rayA(A(t)), rayB(B_interp(t+dt)) )
dt* = argmin cost
```
B's (u, v) linearly interpolated. Apply `dt*` to B; report `dt_refined_ms`. If `cost(dt*)` > 5 cm
the calibration is off.

## 5. Triangulation (two rays, closed form)

```
w0 = C1 - C2;  b = d1·d2;  d = d1·w0;  e = d2·w0;  den = 1 - b*b   (reject if den < 1e-4)
s = (b*e - d)/den;  t = (e - b*d)/den
P1 = C1 + s*d1;  P2 = C2 + t*d2;  X = (P1+P2)/2;  resid = |P1-P2|
```
Reject `resid > 0.10 m` or `s < 0.3` or `t < 0.3`. Output `(t_server, X, resid)`.

N cameras (stretch / slides): `X = (Σ(I − d_i d_iᵀ))⁻¹ Σ(I − d_i d_iᵀ) C_i`.

## 6. Bounce segmentation

3-point median filter on positions; central-difference velocity. Bounce = angle between
consecutive velocity vectors > 60° **and** speed > 1 m/s. Split there. Discard segments < 8 points.

## 7. Parabola fit and metrics (per segment)

`p(t) = p0 + v0 τ + ½ a τ²`, `τ = t − t0`; linear least squares per axis; RANSAC-lite (drop
residuals > 3× median, refit once). **Do not fix g in the fit.**

- `g_fit = |a|`, `g_dir_err_deg = angle(a, up)`  (a should point along up·(−1)... i.e. down)
- `rms_residual_cm`, `speed_mps = |v0|`, `launch_angle_deg = asin((v0·up)/|v0|)`
- `apex_height_m = max(p·up) − (p_first·up)`, `range_m` (horizontal), `flight_time_s`

**Bounce:** impact point = average of both fits at the split time; `v_in`, `v_out` from the fits
at that time; `n = normalize(v_out − v_in)`; `restitution = |v_out·n| / |v_in·n|`.

**Drop:** horizontal speed < 0.5 m/s and `v0·up` > −0.5 → counts toward `g_mean/g_std/count`.

## 8. When g comes out wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| `g` ≈ 9.8 but points up | `up` sign | `up = [0,-1,0]` in board frame (board y goes down the screen) |
| `g` ≈ 9.8 but sideways | board lid not vertical, or object-point order wrong | check corner ordering vs object points; lid at 90° |
| `g` off by 2× or ½ | time units / wrong device offset | `t` in seconds; apply each device's `clock_offset` |
| `g` right, residuals > 5 cm | extrinsics off | redo Lock; check square size; board detection at distance |
| `g` 3–5% off consistently | nominal intrinsics or wrong square size | run `/intrinsics`; re-measure squares |
| noisy run to run | sync | widen Δt search; check RTT during sync |
| `g` low with a paper ball | drag | use a dense ball for the number |

## 9. Simulator (`server/simulate.py`) — build this first

Two virtual cameras ~3 m apart, 1.0 m high, yawed ~30° toward the center, `K` nominal for
1920×1080, zero distortion. Random throw (`v0` 5–9 m/s, 30–50° elevation, gravity 9.81 along
`[0,+1,0]` i.e. "down" in the board frame); optional wall at z = −2.5 m with restitution 0.7.
Sample at 30 or 60 fps per camera with independent time offsets (±40 ms), project with the §2
conventions inverted, σ = 1.5 px noise, random frame drops. Emit API payloads (POST or JSONL).
Pass criterion: solver recovers `g` within ±0.2 and restitution within ±0.05. Later: sweep
N = 2..8 cameras and the angle between them; plot median 3D error (slides).

## 10. Dashboard (`server/static/index.html`)

One page, Plotly from CDN, polls `/latest` every second. Dark background, big fonts.
3D plot with cameras + raw points + fitted curves + impact marker, "up" vertical. Big numbers:
speed, launch angle, apex, flight time, `g = 9.78 m/s² (dir err 1.9°)`. Demo B: speed in/out,
restitution. Status: cameras alive, baseline, Δt, drops `g_mean ± g_std (n)`. Reset button.
