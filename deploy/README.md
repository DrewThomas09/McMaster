# Using McMaster-Vision on your phone

The interface is a web page, so any phone browser works. Camera capture via the
"Take a photo" button works over plain HTTP; the **live camera preview** and
**Add to Home Screen** (PWA install) need HTTPS or `localhost`.

## 1. Same Wi-Fi as your laptop (fastest to set up)

```bash
mcv up                    # demo catalog or your built one, QR code printed
mcv serve --host 0.0.0.0 --qr    # same, without demo mode
```

On the laptop you can also open `http://localhost:8000/connect` to show the QR
code full-screen.

The command prints the LAN URLs and a QR code; scan it with the phone camera.
Photos go straight to your laptop; nothing leaves the network.

## 2. Same Wi-Fi with HTTPS (PWA install + live camera)

```bash
mcv serve --host 0.0.0.0 --https --qr        # self-signed certificate in data/certs/
```

The phone will warn about the certificate once; accept it (or install
`data/certs/mcv.crt` as a trusted profile on iOS / Android for a clean lock icon).

## 3. From anywhere, no port forwarding

Run `mcv serve` at home and expose it with a tunnel that gives you HTTPS:

* **Tailscale**: `tailscale serve --bg 8000` -> `https://<machine>.<tailnet>.ts.net`
* **Cloudflare Tunnel**: `cloudflared tunnel --url http://localhost:8000`

## 4. A real server with a domain (team use)

```bash
DOMAIN=parts.example.com docker compose -f deploy/docker-compose.prod.yml up -d
```

Caddy obtains a Let's Encrypt certificate automatically. Set `MCV_API_TOKEN`
in `.env` to protect `/admin/*`, and `MCV_RATE_LIMIT_PER_MINUTE` if the URL is
public. Mount the same `data/` directory used by `mcv bootstrap`.

Everything the deployment learns (confirmed photos, logs, index, calibration)
lives in that `data/` directory. Back it up nightly and keep the archive off
the box:

```bash
0 2 * * *  cd /srv/mcmaster-vision && docker compose run --rm backup && rsync -a data/backups/ nas:/backups/mcv/
```

`mcv restore data/backups/mcv-<stamp>.tar.gz` puts it back; the running API
picks the restored index up within 15 s.

## Tips on the phone

* Add to Home Screen: Safari share sheet -> "Add to Home Screen"; Chrome menu ->
  "Install app". It opens full-screen with the camera button on top.
* Hold the part 20-40 cm away on a plain background; use "Add another angle"
  for screws (head and side) and "This is it" when you confirm, which makes the
  system learn your parts.
* Put a quarter or a card next to the part and tap **Measure**, then the two
  ends of the coin: the measured size separates look-alikes (1" vs 1-1/4").
* No signal? Confirmations wait in an outbox on the phone and sync on their own.
