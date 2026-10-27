> **Superseded Sat 20:30 — the project is now Vigil. Read `docs/VIGIL.md`.** This file is the original HawkEye plan, kept for the record.

# HawkEye — the plan

Two people. One Mac, one Windows laptop, two iPhones, a browser. Two demos. Nothing that does
not show up on stage.

**Clock:** Sat 14:30 start → Sat ~23:30 stop → Sun 07:00 resume → Sun 10:00 done.
Submission form by Sun 12:00; demos 12:00–15:00.

## 0. Principles

1. **Hard part first.** The thing most likely to kill the project is getting both cameras into one
   accurate coordinate frame (calibration). It is tested by 18:00 Saturday, before the solver is
   anything more than a simulator. If it fails the gate, we fall back (§5) the same evening.
2. **Rushendra owns what can fail in a non-obvious way**: the phone capture page in Safari
   (camera, frame rate, detection speed, HTTPS trust, wake lock), the hotspot, integration, and
   every gate test. **Teammate owns deterministic math + plumbing**: server, OpenCV calibration,
   solver, dashboard — all developable against synthetic data from minute zero.
3. **One place for data: the Mac.** Phones push JSON over HTTPS to the Mac. The Mac serves the
   capture page, runs calibration + solver, and serves the dashboard. No native app, no Xcode,
   no peer-to-peer.
4. **Phones never move after calibration.**
5. **No feature after Sat 22:30.** Sunday morning is rehearsal, slides, backup video, submit.

## 1. The two demos (what "done" means)

**Demo A — arc throw.** Phones ~3 m apart, both seeing a ~4 m wide volume. Underhand lob, peak
~2 m. Dashboard shows: the 3D arc, release speed, launch angle, apex height, flight time, and the
fitted gravity `g` (target 9.8 ± 0.3 m/s²) with its direction error vs. "up" (< 5°).

**Demo B — wall bounce.** Same setup with a wall at the far side. Throw at the wall, catch the
return. Dashboard shows: two arcs split at the impact, speed in / out, impact point, restitution,
and `g` fitted independently on both legs.

**Proof-of-reality beat (both demos):** drop the ball from rest → `g = 9.8` on screen.

## 2. Roles

### Rushendra — capture page + integration (`server/static/capture.html`, spec `docs/WEB.md`)
- HTTPS on the Mac (mkcert), root CA trusted on both phones. Hotspot.
- `capture.html`: camera at the highest frame rate Safari gives, on-page chroma detection,
  "Sample ball", calibration photo capture, lock, continuous POST of detections, wake lock.
- Runs every gate test (§4) with the teammate's code; owns the demo rehearsals.

### Teammate — server, calibration, solver, dashboard (`server/`, specs `docs/API.md`, `docs/MATH.md`)
- FastAPI server: 7 endpoints, every request logged to JSONL.
- OpenCV calibration: intrinsics from ~8 photos of the on-screen checkerboard per phone;
  extrinsics from one photo per phone of the board at a fixed spot (`solvePnP`).
- Simulator → solver (rays, triangulation, Δt refinement, bounce split, parabola fits, metrics).
- Dashboard (one HTML page, Plotly, auto-refresh).

Both halves are fully specified in `docs/`. An agent can build either half from the docs alone.

## 3. Timeline

| When | Rushendra | Teammate | Gate / decision |
|---|---|---|---|
| Sat 14:30–15:30 | mkcert + HTTPS server skeleton serving a static page; CA trusted on both phones; hotspot; Mac IP noted; `capture.html` shows the camera | Repo, venv (Windows), FastAPI with all endpoints + JSONL logging; checkerboard page | Both phones open the page over HTTPS and see their camera |
| Sat 15:30–18:00 | Calibration capture UI (take photo → POST), lock flow; measure board square size; take intrinsics photos for both phones | `calibrate.py`: intrinsics + extrinsics with OpenCV; `/latest` reports baseline | **GATE 1 (18:00): baseline check** — distance between phones from calibration vs. tape measure, within 5 cm. Fail → §5 |
| Sat 18:00–18:30 | Dinner (Bruininks 432). Grab an orange or two. | Dinner | |
| Sat 18:30–20:30 | Detection in JS (chroma centroid at ≥ 30 fps), "Sample ball", continuous POST, clock sync | Simulator → solver v0 recovers `g` on synthetic data; dashboard v0 | Detections visible in server log from both phones |
| Sat 20:30–21:30 | First real throws, drop tests | Solver on real data; fix signs (MATH.md §8) | **GATE 2 (21:30): drop test `g` within 9.8 ± 1.0** |
| Sat 21:30–23:30 | Wall bounce runs; tolerance tuning; re-lock flow; 10 drops | Δt refinement, bounce split, restitution, drop stats, dashboard readable from the back | **GATE 3 (23:30): `g` std < 0.3 over 10 drops.** Feature freeze |
| Sun 07:00–08:30 | Set up in the demo spot; 5 full rehearsals of both demos; record backup video | N-phones chart; slides | |
| Sun 08:30–10:00 | README, repo tidy, submit | Slides final | Done by 10:00 |

## 4. Gates in detail

**GATE 1 — baseline check.** After both phones lock on the board, `/latest` shows each camera's
position in the board frame and their distance. Tape-measure the real distance between the two
phones' lenses. Within 5 cm → pass. 5–15 cm → pass, note it. > 15 cm or `solvePnP` unstable → §5.

**GATE 2 — first physics.** Drop the ball from rest between the phones. The solver fits
`p(t) = p0 + v0 t + ½ a t²`. Pass if `|a|` within 9.8 ± 1.0 and `a` within 10° of "up". Wrong
signs show up here as `g` sideways or upward; fix table in MATH.md §8.

**GATE 3 — repeatable.** 10 drops: mean `g` within ±0.3 of 9.8, std < 0.3. Goes on a slide.

## 5. Kill-switches

- **GATE 1 fails because the on-screen board is not detected at distance** → print the board
  (any printer on campus; A4 is fine, measure the squares) or show it larger on a TV. 20 minutes.
- **GATE 1 fails because intrinsics are bad** (reprojection error > 1 px) → retake the 8 photos
  with more tilt variety; as a last resort use nominal intrinsics (MATH.md §1b) and accept ~3%
  scale error — the demo still works, `g` reads ~9.5–10.1.
- **Safari gives < 25 fps or detection cannot keep up** → downscale the analysis canvas to 320×180;
  if still bad, record short clips with the Camera app and post-process on the Mac with OpenCV
  (slower loop, same math, same demo).
- **Camera page blocked (HTTPS trust fails on a phone)** → `cloudflared tunnel --url http://localhost:8000`
  gives a public HTTPS URL in one command (needs cellular data on the hotspot). Fallback, not plan.
- **Nothing triangulates by 21:30** → stop adding; spend the remaining evening making the 2D
  tracks + calibration + simulator chart into an honest "here is what works and why" demo. Do not
  switch ideas at 21:30 — there is no time to build another one.

## 6. Hardware / logistics

- Phones: iPhone 16 (iOS 27, Rushendra), iPhone 16 Pro Max (iOS 26.6.2, teammate). Both run the
  page in Safari. Native app ruled out: Xcode 27 needs macOS 26.6 (Mac is on 15.7), Xcode 26 cannot
  target iOS 27, and the Mac has 22 GB free.
- Mac (M4 Pro, macOS 15.7.9, Python 3.9): server + calibration + solver + dashboard. Keep free
  space in mind; nothing here needs more than ~1 GB.
- Teammate laptop (Windows): develops `server/`; runs the server locally against the simulator.
- Network: one phone's **Personal Hotspot**; Mac + both phones join it. Note the Mac's IP; the
  HTTPS cert includes it and the Mac's `.local` name.
- Ball: for development, anything bright, matte, single-colored, roughly spherical (an orange
  from dinner is ideal; a paper ball wrapped in a colored sticky note works). For the `g` demo use
  something dense — tennis ball or orange — a light paper ball has visible drag and will read low.
- Phone stands: water bottles or laptop edges. Phones landscape, resting, not held, plugged in.
- Board: `server/static/board.html` fullscreen on the Mac, lid vertical, square size measured.

## 7. Decisions log

- **Web page in Safari, not a native app.** Reason: see §6. Cost: no ARKit; calibration via
  OpenCV checkerboard instead (deterministic, textbook, teammate-owned).
- **HTTPS via mkcert**, because `getUserMedia` only works in a secure context. Fallback cloudflared.
- **Intrinsics from calibration photos, not nominal FOV.** Reason: Safari's stream crop is unknown;
  8 photos take 3 minutes and remove a 3% scale error.
- **Frame rate: whatever Safari gives (30–60).** The solver does not care; a lob at 30 fps is
  ~20 cm per frame and fits fine.
- **HTTP POST batches every 0.5 s; server detects throw events.** No buttons during a throw.
- **Clock sync = `/time` min-RTT on the phone + Δt refinement in the solver.**
- **Public repo.**
