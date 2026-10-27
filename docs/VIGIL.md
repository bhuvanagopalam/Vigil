# Vigil: the waiting room gets a monitor

Safari web pages on phones → FastAPI on the Mac over HTTPS → a browser dashboard.

## One paragraph

Triage is a snapshot at arrival. Waiting is where people get worse unseen. Re-checks of waiting
patients are local policy, hours apart, and understaffed ERs miss them. Vigil turns every phone in the
waiting room into the monitor that notices: scan a QR, put a fingertip on the camera for 20 s (flash on) →
heart rate, breathing rate, HRV, perfusion, and a capillary-refill test (press 4 s, relax 6 s) → the nurse's
board ranks everyone by a transparent NEWS-style score plus *trend since arrival*, asks for re-checks
every 15 minutes, and can make a patient's phone glow to find them. A room camera counts everyone's
breathing between checks. Honest claims only: catch deterioration sooner, not prevent it; a reassessment
aid, not a diagnostic; no SpO2, no BP number.

## What exists (`server/`)

- `app.py`: FastAPI. `POST /join`, `POST /measure` (raw [t,R,G,B] frames + CRT frames), `GET /me/{id}`,
  `POST /recheck/{id}`, `POST /glow/{id}`, `POST /seen/{id}`, `GET /patients`, `POST /room`, `POST /reset`,
  `GET /qr.png`.
  Run: `uvicorn app:app --host 0.0.0.0 --port 8443 --ssl-certfile certs/cert.pem --ssl-keyfile certs/key.pem`
- `vitals.py`: PPG analysis (HR, RR from respiratory modulation, RMSSD/SDNN, perfusion index, SQI),
  capillary refill (exponential recovery after release), complaint red flags, NEWS-style score with reasons.
  Verified on real data: HR 72 (contact finger, 120 fps, SNR 155); 113 and 130 bpm on synthetic fast rates.
- `model/train.py` + `model/deterioration.json`: logistic model trained on PhysioNet/CinC 2019 (3,964 ICU patients,
  151k hourly rows): HR, RR, age, change over 3 h and since arrival → sepsis within 6 h. Test AUROC 0.67 vs 0.64 for
  the NEWS bands on HR+RR. Shown on the board as a "Risk" percentile next to the score; the score still sorts.
  Retrain: `python model/train.py <dir of p*.psv>` (needs scikit-learn; the server only reads the JSON).
- `static/patient.html`: join → 20 s pulse check with live waveform → press/relax refill test → results →
  15-min countdown; polls `/me` for nurse re-check requests and "find me" glow.
- `static/board.html`: nurse board: ranked table, sparklines, reasons, Re-check / Find / Seen, QR to join,
  room-breathing panel.
- `static/room.html`: room camera. Tap up to 6 chests; per-seat breathing rate from chest motion, apnea flag,
  `POST /room` every 2 s. Runs on the Mac webcam (`https://localhost:8443/static/room.html`) or a phone on a stand.
  Seats are linked to patients by chair number: the patient types it on the result screen (`POST /seat/{id}`) or
  the nurse picks a name under the seat on the board. A linked seat's live breathing shows in the patient's row
  and feeds the score (camera apnea +3, camera breathing over 20/min by NEWS bands).
- `docs/slides.html`: the deck, 5 slides. Arrow keys, F fullscreen, cmd-P for PDF.
- `test_e2e.py`: end-to-end with real finger frames; creates three patients, exercises re-check, glow, room, board.
  Run `POST /reset` after it so the board is empty.

## Tasks

Teammate:
1. Open `https://172.20.10.2:8443/static/patient.html` on your phone (hotspot). Check in, finger on camera.
   Report the four numbers and whether your row shows on the board.
2. Open `room.html` on your phone on a stand, tap a chest, confirm a rate appears and APNEA on a breath hold.
3. Read `docs/DEMO.md`. You run the squats re-check and the Find glow in the demo.

Rushendra:
1. Same check-in on your phone. Tune `vitals.py` thresholds if CRT or RR look wrong on a real finger.
2. Record a backup video of the full flow.
3. Export slides to PDF.

## Sunday

07:00 room.html test in the theatre lighting · 08:00 rehearsals (judge scans QR, finger check, squats,
room breathing, Find) · 09:00 README + slides final · 10:00 submit (form deadline 12:00).

## Demo script (5 min)

1. "Triage is a snapshot. Waiting is where people get worse unseen." (20 s)
2. Judge scans QR, types a complaint, finger on camera, 20 s + 10 s. Their row appears with HR, breathing,
   HRV, refill. "That's a pulse-oximeter-grade waveform from the phone in your pocket." (90 s)
3. Board: ranking, reasons, trend; teammate (checked in earlier) re-checks after squats → jumps to top,
   flagged. Nurse taps Find → her phone glows. (90 s)
4. Room camera: breathing rates over the audience; someone holds their breath → apnea flag. (45 s)
5. Honest limits + what's next. (30 s)
