# Vigil: the waiting room gets a monitor

Triage is a snapshot at arrival. Waiting is where people get worse unseen. Vigil turns every phone in
an ER waiting room into the monitor that notices.

Scan a QR → a web page (no app) → fingertip on the camera with the flash on for 20 s → a
pulse-oximeter-grade waveform → **heart rate, breathing rate, HRV, perfusion**, then a 10-second
press-and-relax **capillary refill** test. The nurse's board ranks everyone in the room with a
transparent NEWS-style score plus *trend since arrival*, asks for re-checks every 15 minutes, and can
make a patient's phone glow to find them. Next to the score, a **risk percentile** from a logistic model
trained on 3,964 PhysioNet ICU patients (CinC 2019: heart rate, breathing, age, trend → sepsis within 6 h;
test AUROC 0.67 vs 0.64 for the NEWS bands). A room camera counts everyone's breathing between checks,
contactless, from chest motion.

Built in 24 h by Bhuvana Gopalam and Rushendra Sidibomma at Social Hack 2026 (UMN), theme *Connect & Intersect*. Honest claims only: it catches
deterioration sooner, it doesn't prevent it; a reassessment aid, not a diagnostic; no SpO2, no BP.

## Run

```bash
cd server
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8443 --ssl-certfile certs/cert.pem --ssl-keyfile certs/key.pem
python redirect.py   # port 80: typing just the Mac's IP in any browser lands on the check-in page
# nurse board: https://<mac-ip>:8443/      patients: scan the QR on the board, or type the Mac's IP
# room camera: https://<mac-ip>:8443/static/room.html on a phone on a stand (or the Mac webcam)
```

Phones need HTTPS for the camera: `mkcert`, 2 min, see `docs/HTTPS.md`. Everything joins one phone's hotspot.

## Repo

```
server/                 FastAPI app, signal analysis (vitals.py), patient / board / room pages
docs/slides.html        the deck (arrow keys; cmd-P for PDF)
docs/VIGIL.md           the brief: flow, tasks, demo script
docs/DEMO.md            setup checklist, script, judge Q&A
docs/HTTPS.md           local certificates for the phones
legacy_hawkeye/         an earlier prototype, not part of Vigil
```
