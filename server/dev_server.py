#!/usr/bin/env python3
"""Stand-in HTTPS server for developing capture.html before app.py exists. Rushendra's.

Serves ./static over HTTPS, answers /time, and logs every POST to ./data/dev/*.jsonl
(JPEG uploads saved as files). Responses are dummies — enough for the page to run.
Replace with the teammate's app.py as soon as it exists.

Usage:  python3 dev_server.py            # https://0.0.0.0:8443
"""
import cgi
import json
import os
import ssl
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "dev")
os.makedirs(DATA, exist_ok=True)


def log_line(name, obj):
    obj = dict(obj)
    obj["received_at"] = time.time()
    with open(os.path.join(DATA, name + ".jsonl"), "a") as f:
        f.write(json.dumps(obj) + "\n")


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=HERE, **k)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/time"):
            return self._json(200, {"server_time": time.time()})
        if self.path == "/" or self.path.startswith("/latest"):
            return self._json(200, {"server_time": time.time(), "cameras": [], "throws": [], "note": "dev_server stub"})
        return super().do_GET()

    def do_POST(self):
        ctype = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length") or 0)
        if ctype.startswith("multipart/form-data"):
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": ctype})
            dev = form.getvalue("device_id", "unknown")
            img = form["image"] if "image" in form else None
            n = 0
            if img is not None and img.file:
                d = os.path.join(DATA, "calib", dev)
                os.makedirs(d, exist_ok=True)
                n = len([x for x in os.listdir(d) if x.endswith(".jpg")]) + 1
                tag = "lock" if self.path.startswith("/lock") else "%02d" % n
                with open(os.path.join(d, tag + ".jpg"), "wb") as f:
                    f.write(img.file.read())
            log_line("uploads", {"path": self.path, "device_id": dev, "image_size": form.getvalue("image_size", "")})
            if self.path.startswith("/lock"):
                return self._json(200, {"ok": True, "camera_position": [0, 0, 0], "reproj_px": None, "baseline_m": None, "stub": True})
            return self._json(200, {"ok": True, "count": n, "corners_found": None, "stub": True})
        raw = self.rfile.read(length) if length else b"{}"
        try:
            obj = json.loads(raw.decode() or "{}")
        except Exception:
            obj = {"raw": raw[:200].decode(errors="replace")}
        name = self.path.strip("/").split("?")[0].replace("/", "_") or "root"
        log_line(name, obj)
        if self.path.startswith("/intrinsics"):
            return self._json(200, {"ok": False, "message": "dev_server stub: intrinsics not computed", "stub": True})
        return self._json(200, {"ok": True, "stub": True})

    def log_message(self, fmt, *args):  # quieter
        if "/detections" in (args[0] if args else ""):
            return
        super().log_message(fmt, *args)


if __name__ == "__main__":
    import sys
    if "--http" in sys.argv:   # plain HTTP on 8000: for the Mac's own browser (http://localhost is a secure context)
        httpd = ThreadingHTTPServer(("0.0.0.0", 8000), H)
        print("dev_server: http://localhost:8000/static/  (logs -> data/dev/)")
    else:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(os.path.join(HERE, "certs", "cert.pem"), os.path.join(HERE, "certs", "key.pem"))
        httpd = ThreadingHTTPServer(("0.0.0.0", 8443), H)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        print("dev_server: https://0.0.0.0:8443/static/capture.html  (logs -> data/dev/)")
    httpd.serve_forever()
