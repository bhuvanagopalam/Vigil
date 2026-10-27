# API — the contract between the capture page and the server

JSON over HTTPS. Server listens on `0.0.0.0:8443` (TLS) on the Mac. Units: **meters, seconds,
pixels**; degrees only in dashboard text. Matrices are row-major nested lists. Phones send JPEGs
only during calibration, never during throws.

## Conventions (read twice)

- **Time.** Phones timestamp with `performance.now()/1000`. Every payload carries `clock_offset`
  so the server converts: `server_time = t + clock_offset`. The phone computes `clock_offset`
  from `GET /time` (min-RTT). The solver refines a residual Δt between the two phones.
- **Pixels.** `(u, v)` are pixel coordinates in the **full-resolution video frame** as delivered
  by `getUserMedia` (`image_size = [width, height]`, e.g. `[1920, 1080]`), origin top-left, `u`
  right, `v` down — the same frame the calibration JPEGs are taken from.
- **Camera space (OpenCV):** `+x` right, `+y` down, `+z` forward.
  **Pixel → ray:** `d = normalize([(u - cx)/fx, (v - cy)/fy, 1])` (after undistorting `(u, v)`
  with the calibrated distortion, `cv2.undistortPoints`).
- **World frame = the board frame** from the Lock photo: origin at the board's first inner corner,
  `+x` along the board's columns (to the right on the screen), `+y` along rows (down the screen),
  `+z = x × y`, i.e. **into** the screen — so cameras in front of the screen have **negative z**
  (verified on real data: both phones at z ≈ −0.6 m). With the laptop lid vertical,
  **up = `[0, -1, 0]`**.
  `solvePnP` gives `T_cam_board` (board → camera); the server stores `T_board_cam = inv(...)`.
- All rays, points, and fits are expressed in the board frame.

## Endpoints

### `GET /time` → `{ "server_time": 1761350400.123 }`
Wall clock on the server. Phone: 30 requests, keep offset from the smallest-RTT sample.

### `POST /calib_image` (multipart: `device_id`, `image` JPEG, `image_size` "WxH")
Store under `data/<session>/calib/<device_id>/NN.jpg`. Response:
`{ "ok": true, "count": 5, "corners_found": true }` — run `findChessboardCorners` immediately so
the phone can tell the user whether the shot was usable.

### `POST /intrinsics` (`{ "device_id": "iphone16" }`)
Runs `cv2.calibrateCamera` over that device's stored images. Store and return:
```json
{ "ok": true, "image_size": [1920,1080], "K": [[fx,0,cx],[0,fy,cy],[0,0,1]],
  "dist": [k1,k2,p1,p2,k3], "rms_px": 0.41, "n_images": 8 }
```
If `n_images < 5` or `rms_px > 1.5`, return `ok: false` with a message.

### `POST /lock` (multipart: `device_id`, `image` JPEG)
Finds the board, `solvePnP` with that device's intrinsics (requires `/intrinsics` done; otherwise
use nominal intrinsics and flag `nominal: true`). Store `T_board_cam`. Response:
```json
{ "ok": true, "camera_position": [x,y,z], "reproj_px": 0.6, "baseline_m": 2.97, "nominal": false }
```
`baseline_m` is present once two devices are locked — **this is what GATE 1 reads.**

### `POST /detections`
Every ~0.5 s from each phone in Live mode, including when empty.
```json
{ "device_id": "iphone16", "clock_offset": 1761350000.42, "image_size": [1920,1080],
  "fps_reported": 60,
  "frames": [ { "t": 12345.678, "u": 812.3, "v": 455.1, "n": 420, "w": 24, "h": 26 } ] }
```
Append to `data/<session>/detections_<device_id>.jsonl`; keep a rolling 60 s buffer; run event
detection + solve on each batch. Respond `{ "ok": true }`.

### `GET /latest`
Polled by the dashboard every second.
```json
{
  "server_time": 1761350400.1,
  "cameras": [ { "device_id": "iphone16", "position": [x,y,z], "forward": [dx,dy,dz],
                 "age_s": 0.4, "fps_reported": 60, "rms_px": 0.41 } ],
  "baseline_m": 2.97, "up": [0,-1,0],
  "throws": [ {
      "id": 7, "t_start": 1761350380.2, "dt_refined_ms": -3.0,
      "segments": [ { "points": [[x,y,z,t], ...],
          "fit": { "p0": [..], "v0": [..], "a": [..], "g_fit": 9.74, "g_dir_err_deg": 2.1,
                   "rms_residual_cm": 1.8, "speed_mps": 6.9, "launch_angle_deg": 38.0,
                   "apex_height_m": 1.92, "range_m": 3.4, "flight_time_s": 0.81 } } ],
      "bounce": { "point": [x,y,z], "t": 1761350380.9, "speed_in_mps": 6.1,
                  "speed_out_mps": 4.4, "normal": [nx,ny,nz], "restitution": 0.72 } } ],
  "drops": { "count": 10, "g_mean": 9.79, "g_std": 0.15 }
}
```
`bounce` is null for Demo A. `drops` = throws classified as free drops (GATE 3 statistic).

### `POST /reset` → clears in-memory throws (keeps calibration and logs).
### `GET /static/capture.html`, `GET /static/board.html`, `GET /` (dashboard).

## Logging

One directory per server start: `data/YYYYMMDD_HHMMSS/`. Every POST body (minus JPEG bytes) is
appended as one JSON line with `received_at`. Calibration results saved as `calibration.json`.
The solver must run offline: `python solver.py data/<session>/` prints the `/latest` JSON.
