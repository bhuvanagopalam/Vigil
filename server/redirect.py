"""Plain-HTTP front door on port 80: anything typed without https:// is sent to the TLS server.

    python redirect.py            # listens on 0.0.0.0:80 (macOS lets a normal user bind 80)

So a patient can type just 172.20.10.2 in any browser and land on the check-in page.
"""
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 80


CA_FILE = os.environ.get("VIGIL_CA", os.path.expanduser("~/Desktop/HawkEye-rootCA.pem"))


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        host = self.headers.get("Host", os.environ.get("VIGIL_HOST", "172.20.10.2")).split(":")[0]
        if self.path.rstrip("/") == "/ca" and os.path.exists(CA_FILE):
            # the local root certificate, served over plain http so a phone that does not trust us yet can fetch it;
            # iOS offers to install it as a profile
            data = open(CA_FILE, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-x509-ca-cert")
            self.send_header("Content-Disposition", 'attachment; filename="Vigil-rootCA.crt"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        p = self.path.rstrip("/")
        path = "/" if p == "/board" else "/static/room.html" if p == "/room" else "/go"   # patients land on check-in
        self.send_response(302)
        self.send_header("Location", f"https://{host}:8443{path}")
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_HEAD = do_GET

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.client_address[0], fmt % args))


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", PORT), H).serve_forever()
