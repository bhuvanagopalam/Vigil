# Capture page spec (Rushendra) — `server/static/capture.html`

One static HTML file served by the FastAPI server over HTTPS. Plain JS, no framework, no build
step. Opened in Safari on each phone at `https://<mac-ip-or-name>:8443/static/capture.html`.
Phone in **landscape**, orientation lock off, screen brightness up, plugged in.

## 0. HTTPS (do this first; the camera API refuses to run over plain http)

On the Mac:

```bash
brew install mkcert
mkcert -install                               # creates a local CA, trusts it on the Mac
mkcert "$(scutil --get LocalHostName).local" 172.20.10.2 localhost   # use the Mac's hotspot IP
# -> two files: <name>+2.pem (cert) and <name>+2-key.pem (key); move them to server/certs/
mkcert -CAROOT                                # folder containing rootCA.pem
```

Trust the CA on each phone (once): AirDrop `rootCA.pem` to the phone → open it → Settings →
"Profile Downloaded" → Install → then Settings → General → About → Certificate Trust Settings →
enable full trust for the mkcert root. After that Safari accepts `https://<mac>:8443` silently.

Run the server with TLS: `uvicorn app:app --host 0.0.0.0 --port 8443 --ssl-certfile certs/cert.pem --ssl-keyfile certs/key.pem`.
If the hotspot gives the Mac a different IP later, regenerate the cert (10 seconds).

Fallback if a phone refuses the CA: `brew install cloudflared && cloudflared tunnel --url http://localhost:8000`
prints a public `https://*.trycloudflare.com` URL; open that instead. Needs internet on the hotspot.

## 1. Page layout (one screen, big buttons, landscape)

```
[ video preview with crosshair ]   device: [iphone16]   server: ok  rtt 12ms  fps 58  n=412 u=812 v=455
[ Sample ball ]  [ Calib photo (0/8) ]  [ Lock ]  [ Live: on/off ]   tol [====|---]  minpx [==|---]
status line: "intrinsics ok (rms 0.4px)"  /  "LOCKED baseline 2.97 m"  /  "LIVE"
```

Persist `deviceId`, sampled color, tolerance, minpx in `localStorage`.

## 2. Camera

```js
const stream = await navigator.mediaDevices.getUserMedia({
  video: { facingMode: { exact: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 },
           frameRate: { ideal: 60 } }, audio: false });
video.srcObject = stream; await video.play();
// log what we actually got:
const s = stream.getVideoTracks()[0].getSettings();  // width, height, frameRate
```

- Report `s.width, s.height, s.frameRate` on screen and in every payload (`image_size`,
  `fps_reported`). Calibration photos and live detections must come from the **same** settings
  (same resolution, same orientation). Never rotate the phone after calibration.
- Keep the screen awake: `navigator.wakeLock.request('screen')` (re-request on `visibilitychange`).
- Safari may choose 30 fps; that is fine. Do not fight it.
- The ultra-wide/tele cannot be selected reliably; `environment` gives the main camera. Good.

## 3. Per-frame detection

Use `video.requestVideoFrameCallback((now, meta) => ...)` (Safari 15.4+). Each callback:

1. `ctx.drawImage(video, 0, 0, AW, AH)` into an analysis canvas, `AW×AH = 480×270` (scale factor
   `sx = s.width/AW`, `sy = s.height/AH`).
2. `const px = ctx.getImageData(0,0,AW,AH).data` and loop over pixels (RGBA). Chroma test on
   normalized color: `r' = r/(r+g+b+1)`, `g' = g/(r+g+b+1)`; in range if
   `|r'-r0| < tol && |g'-g0| < tol && (r+g+b) > 120` (reject dark). `(r0,g0)` from "Sample ball".
   Accumulate `count, sumX, sumY, minX, maxX, minY, maxY`.
3. Accept if `count >= minpx` and bbox `< 15%` of AW/AH. Centroid in full-res pixels:
   `u = (sumX/count + 0.5) * sx`, `v = (sumY/count + 0.5) * sy`; bbox scaled the same.
4. Timestamp: `t = performance.now()/1000` at callback (use `meta.captureTime` if present —
   Safari exposes it on some versions; fall back to `now`). Store `{t, u, v, n: count, w, h}`.
5. Every 500 ms, POST the buffered detections (possibly `[]`) to `/detections` with
   `clock_offset` (see §5). `fetch(..., {keepalive: true})`, ignore failures, never await inside
   the frame callback.

Budget: 480×270 = 130k pixels; a plain `for` loop over `Uint8ClampedArray` is ~1–2 ms on an
iPhone 16. Draw a small square on the preview at the centroid (optional) and print `n, u, v, fps`.

**Sample ball:** crosshair at the preview center; hold the ball there; tap. Average a 9×9 patch of
the analysis canvas at its center → `(r0, g0)`. Show the color swatch. Tune `tol` until `n` is
stable with the ball at 4 m and `0` without it. Start with `tol = 0.06`, `minpx = 25`.

## 4. Calibration capture

- **Calib photo:** draws the **full-resolution** video frame to a hidden canvas of size
  `s.width×s.height`, `canvas.toBlob('image/jpeg', 0.92)`, POST to `/calib_image` as multipart
  with `device_id`. Take 8 photos of the on-screen checkerboard from different distances/angles
  (fill ~1/3–1/2 of the frame, tilt ±30°). Then tap "Compute intrinsics" → `POST /intrinsics` →
  show the RMS reprojection error (want < 1 px).
- **Lock:** with the phone in its final resting position and the board at its fixed spot, tap
  Lock → one full-res JPEG → `POST /lock`. Server runs `solvePnP` and returns this camera's
  position in the board frame and, once both are locked, the baseline. Show both. **After Lock,
  do not touch the phone.** Show a red banner if the user taps Live before Lock.

## 5. Clock sync

At page load and every 60 s: 30× `GET /time`; `t0 = performance.now()/1000` before,
`t1` after; keep `offset = server_time - (t0 + rtt/2)` from the smallest-RTT sample. Show the RTT.
Over a hotspot expect 5–30 ms RTT; the solver refines the residual.

## 6. Order of work (hard part first)

1. HTTPS + page shows the camera on **both** phones (14:30–15:30). Stop here until it works.
2. Calib photo + Lock flow against the teammate's endpoints. Run **GATE 1** by 18:00.
3. Detection numbers on screen; ball at 1 m, 4 m, thrown; fps check.
4. POST + clock sync; see detections in the server log.
5. Drop test with the solver. **GATE 2** by 21:30.
6. Polish: tolerance sliders, swatch, banners, localStorage, re-lock.

## 7. Gotchas

- Safari pauses the camera when the tab is backgrounded; keep the page in front; wake lock on.
- Auto-exposure/auto-focus change the ball's apparent color; sample the ball at the demo spot
  under demo lighting, after the camera has settled (~2 s).
- `requestVideoFrameCallback` fires per *delivered* frame; if the preview drops to 30 fps,
  detections are 30 Hz. Fine.
- iOS asks camera permission per origin; the HTTPS origin must be exactly the one in the cert.
- Both phones must run the same page version; hard-reload after changes.
