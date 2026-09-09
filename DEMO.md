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
5b. **Measure it:** put a quarter (or a card) next to the part, tap **Measure**,
   tap the two ends of the coin, pick "US quarter". The verdict now shows the
   measured size and every candidate says whether its catalog length / OD fits.
5c. **Threads:** with the scale set, a threaded part also shows its pitch
   ("thread pitch ≈ 1.27 mm, 20 tpi"); a 1/4"-28 look-alike drops in the list.
6. **Buy it:** tap **Add to cart** on the right candidate (or **Buy now** on an
   exact answer), open the cart from the header badge, **Check out**. The order
   confirmation says how many photos now teach the model. Nothing is charged.
7. **Learning loop:** open `/dashboard` on the laptop: the **Learning loop**
   panel shows the funnel (photo -> cart -> checkout), what was predicted vs
   what was bought, and a plain-language issues list. Press **Learn now** (or
   run `mcv learn`): the bought photos join the gallery in seconds and the
   phone finds the same part again from that angle. `mcv simulate --customers
   40 --learn` shows the whole loop, before and after, with no phone at all.
8. **Marketplace:** after a couple of orders the home screen shows **For you**
   (parts due again, staples, recent buys, and one slot for something new: a
   complement, a favourite of shops like this one, or an unbought part in its
   usual aisle and material) and search results re-sort among the variants of a
   name towards what this shop buys; the family answer stars the size they
   usually take. On the laptop, `/dashboard` lists the customer
   segments and how well each is served, and `mcv simulate-market --shops 200`
   replays a thousand-order marketplace to show the lift with and without the
   shop id.
9. **Paper demo:** tap **print a sheet**, print it, and photograph the paper
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
