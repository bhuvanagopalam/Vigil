# Demo: Vigil

## Setup (10 min, in this order)

1. Hotspot on (Rushendra's phone). Mac + teammate's phone joined. Mac IP = `172.20.10.2` (check `ipconfig getifaddr en0`; if different, `mkcert` a new cert, 10 s, and set `VIGIL_HOST`).
2. Mac: `cd server && source .venv/bin/activate && uvicorn app:app --host 0.0.0.0 --port 8443 --ssl-certfile certs/cert.pem --ssl-keyfile certs/key.pem` and, in a second terminal, `python redirect.py` (port 80, so typing `172.20.10.2` works). A typed address must otherwise start with `https://`, or Safari says "connection lost".
3. Projector: `https://localhost:8443/` (accept the cert warning once, or `mkcert -install`). Press **Reset**.
4. Room camera: teammate's phone on a stand facing the audience, `https://172.20.10.2:8443/static/room.html`, Start (rear), tap 3–4 chests. (Or the Mac webcam at `https://localhost:8443/static/room.html`.) Boxes are numbered seat 1, 2, 3 in the order tapped. Put paper labels "1", "2", "3" on the chairs in the same order. A patient types their chair number on the result screen (or the nurse picks a name under the seat on the board), and from then on that seat's breathing is theirs: it shows under their breathing number as "camera 16/min", and apnea or fast breathing adds to their score with a "camera:" reason.
5. Phones must trust the local certificate. On any phone, open `http://172.20.10.2/ca` in Safari → Allow → Settings → Profile Downloaded → Install → then Settings → General → About → Certificate Trust Settings → turn on full trust for the mkcert root. Both team phones are done; a judge's phone would need this too, so the judge does the finger test on one of ours.
6. Pre-seed the board: teammate checks in on her phone 10 min before (so she has a baseline to "deteriorate" from). Two friends check in too, so the board isn't empty.
7. Backup video of a full run on the Mac desktop.

## Script (5 min)

- **0:00** Slides 1 and 2. "Triage is a snapshot. Waiting is where people get worse unseen. We don't prevent it, we notice it sooner."
- **0:40** Slide 3, the three numbers, 20 s. Slide 4, the flow, 20 s. Then: "Let's do it on you." Judge scans the QR, types *chest pain*, finger on camera, flash on. 20 s + 10 s. Their row appears: HR, breathing, HRV, refill, score 2 with "complaint: chest pain" in the reasons. *"That waveform is the same signal as the clip on your finger in a hospital."*
- **2:10** Board walk-through: ranking, reasons, trend lines. Teammate: "I checked in 10 minutes ago, resting." She does 30 squats (or runs the stairs beforehand), re-checks on stage: her row jumps to the top, red, *"HR up 38% since arrival (+1), RR 24 (+2)."* Nurse taps **Find** → her phone glows orange in the audience.
- **3:30** Room camera: breathing rates over the audience's chests. "Hold your breath" → APNEA on one seat in ~8 s.
- **4:15** Slide 5: the closing line, honest limits spoken (reassessment aid, not diagnostic; no SpO2 or BP), what's next. Done.

## If something breaks

- Finger reading "unreadable": press the finger flatter, keep still, re-check. Don't debug; move on to the next beat.
- Board doesn't update: refresh the tab. Server log: `server/data/uvicorn.log`.
- Room camera finds nothing: the chest box must be on a textured shirt, person still; skip to slide 5 if it fights you.
- Total failure: play the backup video, narrate.

## Judge Q&A

- **Why not a smartwatch?** Most patients don't wear one; this works on any phone, no app, no account, in ten seconds.
- **Accuracy?** Heart rate from contact PPG at 120 fps is pulse-oximeter class (we match a counted pulse). Breathing from pulse modulation is ±2–3/min when still. Capillary refill is a 10-ms measurement of something nurses eyeball at 1-s resolution.
- **How does the camera know who is who?** Chairs are numbered, like a gate at an airport. The patient types the chair number on their phone after the first check, or the nurse picks the name under the seat on the board. One tap, then the seat's breathing is tied to that patient's row and score. (Next step: the "Find" glow on the phone screen lets the camera tag the chair automatically.)
- **Does this exist?** Phone pulse apps exist for one person. Remote monitoring exists with wearables. Nobody runs the waiting room off patients' own phones with trend-based re-triage on a nurse board.
- **Liability / regulation?** It's a reassessment prompt for a nurse, not a diagnostic. Pilot as decision support; FDA path later.
- **Privacy?** First name, age, complaint; raw signal never leaves the room's Mac; patient removed on "Seen".
- **Why the web and not an app?** A scared person in an ER won't install anything. A QR and 10 seconds, they will.
- **What about people without phones?** The ER keeps a few loaner phones at the desk; the page works on any of them.
