"""End-to-end test against a running server using REAL fingertip data + a synthetic capillary-refill curve.
Usage: .venv/bin/python -I test_e2e.py [https://localhost:8443]
Creates 3 patients (one with chest pain, one who 'did squats' = HR scaled up), prints the board ranking."""
import json
import ssl
import sys
import urllib.request

import numpy as np

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://localhost:8443"
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE


def call(path, body=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method or ("POST" if data is not None else "GET"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, context=CTX, timeout=30) as r:
        return json.loads(r.read().decode())


def real_finger_frames():
    rows = []
    for src in ("data/dev/vitals.jsonl",):
        try:
            for line in open(src):
                try: o = json.loads(line)
                except Exception: continue
                if o.get("mode") == "finger": rows += [list(f) for f in o.get("frames", [])]
        except FileNotFoundError:
            pass
    rows.sort()
    t = np.array([r[0] for r in rows]); gaps = np.where(np.diff(t) > 0.5)[0]; b = [0] + list(gaps + 1) + [len(t)]
    a, z = max([(b[i], b[i + 1]) for i in range(len(b) - 1)], key=lambda r: r[1] - r[0])
    return rows[a:z]


def scaled(frames, hr_factor):
    """Speed up time to simulate a faster heart (and breathing) after exercise."""
    t0 = frames[0][0]
    return [[t0 + (f[0] - t0) / hr_factor, f[1], f[2], f[3]] for f in frames]


def synthetic_crt(t0, tau=0.45, fps=120):
    """Red channel: baseline 150, pressed -> 205 for 5 s, release -> exponential back to 150."""
    out = []
    for i in range(int(13 * fps)):
        t = i / fps
        if t < 0.4: r = 150 + 55 * t / 0.4
        elif t < 5.0: r = 205 + np.random.randn() * 0.3
        else: r = 150 + 55 * np.exp(-(t - 5.0) / tau) + np.random.randn() * 0.3
        out.append([round(t0 + t, 4), round(float(r), 2), 6.0, 24.0])
    return out


frames = real_finger_frames()
print(f"real finger frames: {len(frames)}")
call("/reset", method="POST")
pts = [
    ("Ana", 34, "sore throat and cough", 3, frames, None),
    ("Ben", 58, "chest pain since this morning, short of breath", 6, frames, synthetic_crt(frames[-1][0] + 1, tau=0.5)),
    ("Cal", 27, "twisted ankle", 5, scaled(frames, 1.45), synthetic_crt(frames[-1][0] + 1, tau=1.2)),
]
ids = {}
for name, age, complaint, pain, fr, crt in pts:
    j = call("/join", {"name": name, "age": age, "complaint": complaint, "pain": pain}); ids[name] = j["id"]
    m = call("/measure", {"id": j["id"], "frames": fr, "crt_frames": crt, "crt_release_t": (crt[0][0] + 5.0) if crt else None})
    R = m["result"]; C = m.get("crt") or {}
    print(f"{name:4s} ok={R['ok']} hr={R['hr']} rr={R['rr']} rmssd={R['rmssd_ms']} pi={R['pi']} sqi={R['sqi']} beats={R['beats']} crt={C.get('crt_s')} ({C.get('reason') or 'ok'}) score={m['score']['score']} {m['score']['reasons']}")
# Cal 'did squats': a second, faster reading -> trend flag
m = call("/measure", {"id": ids["Cal"], "frames": scaled(frames, 1.8)})
print("Cal after squats:", m["result"]["hr"], m["score"]["reasons"])
call(f"/recheck/{ids['Ana']}", method="POST"); call(f"/glow/{ids['Ben']}", method="POST")
me = call(f"/me/{ids['Ana']}"); print("Ana /me recheck_requested:", me["recheck_requested"])
me = call(f"/me/{ids['Ben']}"); print("Ben /me glow:", me["glow"])
call("/room", {"seats": {"seat 1": {"rr": 14.2, "snr": 6.1, "amp": 1.3, "apnea": False, "t": __import__('time').time()}}})
board = call("/patients")
print("\nBOARD ORDER:")
for p in board["patients"]:
    L = p["last"] or {}
    print(f"  [{p['score']}] {p['name']:4s} hr={L.get('hr')} rr={L.get('rr')} crt={L.get('crt_s')} flags={p['flags']} reasons={p['reasons']}")
print("room:", board["room"])
