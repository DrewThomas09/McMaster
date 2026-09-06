# Demo script (5 minutes, one laptop + one phone)

## Before

```bash
pip install -e ".[dev,phone,ml]"     # once; ml = torch for the learned model
mcv up                                # builds the demo catalog on first run (~1 min), then serves
```

Leave the terminal visible: it prints the phone URL and a QR code. (Or open
`http://localhost:8000/connect` on the laptop for a full-screen QR code.)
Optional: `mcv up --https` for the installable app and live camera preview;
accept the certificate warning once on the phone.

## On the phone

1. Scan the QR code. The app opens with a big **Take a photo** button.
2. **No part at hand?** Tap a sample in the strip: a photo-style render of a
   catalog part is identified live and badged *correct / ranked #n / missed*;
   the true part is outlined in the list. Tap **shuffle** for another set.
3. **Point and see** (HTTPS or localhost): open **Live camera**, tap **Live
   ID**, and pan across a few parts; the overlay updates every second with the
   best guess, tier colour, confidence, and server time. Press the shutter for
   the full result.
4. **Real photo:** tap **Take a photo**, shoot any part on a plain background.
   Show the verdict card (tier, part number, specs, confidence), tap a
   candidate image to compare it side by side with the photo.
5. **Look-alikes:** when the answer is a family ("Looks like Socket Head
   Screw across 3 look-alike SKUs"), tap a length/thread chip to resolve it in
   one step.
6. **Learning loop:** tap **This is it** on the right candidate. Then open
   `/metrics` on the laptop: the confirmed top-1 rate is the live scorecard,
   and `mcv retrain` folds those photos into the next model.
7. **Paper demo:** tap **print a sheet**, print it, and photograph the paper
   with the phone: the same pipeline, real camera, real lighting.

## What to say

* It is retrieval, not a 700k-way classifier: every catalog image becomes a
  vector once; a photo is embedded with the same model and matched by nearest
  neighbour, so new SKUs are one index row and look-alike SKUs are handled
  honestly as a family with the attribute that tells them apart.
* Everything shown runs on the laptop's CPU; the learned model was trained
  from scratch on synthetic renders in two hours. Real McMaster images plus the
  confirmations from this very screen are what make it accurate on real parts.
* `mcv bootstrap <folder of images>` is the whole path from a drop of images to
  this app; `RUNBOOK.md` has the rest.
