# HTTPS for the phones

Safari refuses the camera API over plain http, so the server runs TLS with a local certificate.

On the Mac:

```bash
brew install mkcert
mkcert -install        # creates a local CA and trusts it on the Mac (needs sudo; see below if it fails)
mkcert "$(scutil --get LocalHostName).local" 172.20.10.2 localhost 127.0.0.1   # the Mac's hotspot IP
# -> <name>+3.pem (cert) and <name>+3-key.pem (key): copy to server/certs/cert.pem and server/certs/key.pem
mkcert -CAROOT         # folder containing rootCA.pem
```

If `mkcert -install` cannot get sudo, trust the CA for your own user instead:

```bash
security add-trusted-cert -r trustRoot -k ~/Library/Keychains/login.keychain-db "$(mkcert -CAROOT)/rootCA.pem"
```

Trust the CA on each phone (once): AirDrop `rootCA.pem` to the phone → open it → Settings →
"Profile Downloaded" → Install → then Settings → General → About → Certificate Trust Settings →
enable full trust for the mkcert root. After that Safari accepts `https://172.20.10.2:8443` silently.

Run the server:

```bash
cd server && source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8443 --ssl-certfile certs/cert.pem --ssl-keyfile certs/key.pem
```

If the hotspot gives the Mac a different IP later, regenerate the cert (10 seconds) and set `VIGIL_HOST=<ip>`
so the QR on the board points at the right address.
