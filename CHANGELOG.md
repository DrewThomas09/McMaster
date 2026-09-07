# Changelog

## 0.4.0

- Shipped model: `assets/tinycnn_synthetic.pt` retrained on the fixed renderer (same recipe,
  8,000 parts, 24 epochs). Held-out 800-part catalog, 400 photo-style queries: Recall@1
  0.30 -> 0.47, Recall@5 0.70 -> 0.86, family Recall@1 0.49 -> 0.65, MRR 0.48 -> 0.63.

- Purchase loop: `POST /cart`, `DELETE /cart/{pn}`, `POST /checkout`, `GET /orders`; the phone
  UI gets Add-to-cart on the verdict and candidates, a cart drawer with quantities and the
  confidence each line came with, and one-tap checkout with an order confirmation. Every
  purchased item that came from a photo is filed as a `checkout` confirmation (weight 3,
  tap 2, cart 1: `FEEDBACK_WEIGHTS`), which the usage prior and training honour.
- Backend tracking: `data/logs/events.jsonl` records identify / cart / checkout / feedback /
  error events; `GET /analytics` and the dashboard's "Learning loop" panel turn them into the
  funnel, predicted-vs-bought confusion pairs, tier precision when bought, confidence when
  right vs wrong, latency and errors, and a plain-language issues list with the fix for each.
- Recursive learning: `mcv learn` adds new confirmed photos to the live index incrementally
  (`add_photos`, seconds, `meta.learned_paths` keeps each photo once) and runs a full
  purchase-weighted `mcv retrain` once `MCV_LEARN_RETRAIN_AFTER` new confirmations arrived;
  `POST /admin/learn` and a dashboard button do the fast step; the manifest records
  `learned_at` / `retrained_at`.
- `mcv simulate --customers N [--learn] [--json]`: synthetic customers walk the whole
  journey in-process and the analytics report what is wrong, then before/after learning.
- Found by the simulation and fixed: a learned photo (similarity ~1.0 to its own gallery
  row) could still lose to a look-alike with a category prior and a purchase history, so
  the reranker adds a near-identical bonus (`w_exact`) that no prior can outweigh; the
  tiers were wrong on real photos, so `mcv learn` scores every new confirmed photo before
  it joins the gallery and refits temperature and thresholds once 30 samples exist
  (`models/calibration_samples.jsonl`, `calibration_from_purchases` in the manifest);
  "none of these" is counted in the funnel with its own issue.
- Phone UI: **Buy now** on exact / likely answers (one tap from photo to placed order);
  quantity changes replace the line (`set_quantity`) instead of remove-and-re-add; the order
  confirmation reports what the server really learned.
- Review fixes: `GET /orders` shows a phone only its own orders (the full list needs the
  token); an unknown request id ties a cart line to no photo; carts live on disk so every
  worker sees them and identify events are looked up in the shared log; commerce routes
  are rate-limited; an add-to-cart files weight-1 evidence that a checkout upgrades;
  `mcv simulate` runs on a scratch copy unless `--live` and reloads the learned index
  before the after-run; `mcv learn` takes a lock, stamps `learned_at` before the work,
  rebuilds when a corrected label deleted a learned photo, keeps a retrain's held-out
  photos out of the gallery and refits calibration only with enough wrong samples;
  `mcv retrain` trains on purchase-weighted photos but indexes each once; the usage prior
  is back on its tuned scale; event rows without a kind, half-written order lines and
  naive timestamps no longer break anything.
- Second review pass: `mcv retrain` takes the learn lock; calibration samples carry the
  model version and only the served model's samples refit; two different parts from one
  photo (a customer comparing) file no label at cart or checkout; event-log compaction and
  cart writes are locked per file with unique temp names; unknown parts are not re-scored;
  a switched retrain withholds nothing from the live gallery; the simulation baseline is
  scored without the usage prior and the learned index is reloaded explicitly; cart calls
  have their own rate budget; Buy now checks out only when the cart holds just that part.
- Phone UI: the header wraps under the cart badge on narrow phones instead of clipping the
  navigation; candidates that other customers bought, or whose photo was learned, carry a
  visible "bought before" / "seen before" chip; the part page has Add to cart.
- Third review pass: the part page's Add-to-cart script was HTML-escaped inside `<script>`
  (a syntax error; the button did nothing) and is now a JSON literal with a browser test
  that clicks it; a switched retrain records its results under `sibling_retrain` instead of
  describing the served index; the calibration fit raises `exact` rather than lowering a
  just-raised `likely` and needs a 5-point gain to move a threshold; confusion pairs compare
  name and category too and ignore price/url keys; the orders tail read keeps a complete
  first line; the chip says "confirmed before".
- Catalog fittings pages (13-22): the importer reads reducing bushings and couplings with two
  pipe sizes per row, max-psi cells per column, butt-weld wall thickness and the (C)
  dimension, flange OD / bolt columns, thread adapters (NPT x metric, BSPP x NPT ...), pipe by
  the foot, and "Connections:" lines. Pipe wall schedules 10 / 40 / 80 give a fitting's bore
  (`pipe_id_mm`), the Measure tool reads the bore of an end-on part, size matching compares
  it with the candidate's schedule or wall, and the part page lists the bore.
- Importer review: gender words are no longer read as thread standards; a max-psi cell
  without its comma no longer starts a new size row; header-named row fields are
  type-checked (a missing cell skips the field, a package quantity is not a pressure);
  outlets keep their own thread as the pipe size and the range as `fits_pipe_size`; the
  bore is the largest hole (a flange's centre bore, not its bolt holes) and the bore rule
  applies only to plain pipe, unthreaded fittings and parts with a stated wall; schedule
  10S walls for 1/8 - 3/8; the UI shows the bore in inches too.
- Phone UI review: Buy now never adds a second unit of a part already in the cart; "This is
  it" no longer disables Add to cart / Buy now; a slow startup cart fetch cannot overwrite a
  fresh add (cart sequence guard); a new query started while a result body was downloading
  is not rendered over it; "Identify another part" really starts over; failed removes and
  422 details are reported; quantity taps are optimistic; Order again counts real adds;
  the cart badge and sample strip work from the keyboard; Live ID survives a camera flip
  and says when it is rate limited; live-preview frames have their own rate budget so a
  minute of Live ID cannot 429 the next real photo; a confirmation re-uploads the photo
  only when the server says it no longer has it.
- The catalog's female measuring rule: an end-on photo of a threaded female fitting is
  matched by its bore against the nominal size's thread minor diameter (tap-drill size,
  `FEMALE_THREAD_ID_IN`); the part page lists that bore next to the OD and the pipe ID.
- The coin no longer votes on looks: when a reference segment is given, the disc it spans
  is painted over with the surrounding bench (colour and noise matched) before the photo
  is embedded, so the coin only sets the scale. Measured on 60 demo parts with the shipped
  model: photos with a coin were top-1 51/60 against 44/60 for the same photos without one
  (before the fix, the coin in frame had cost 30 points).
- `mcv simulate --coin-rate R`: that share of customers photograph the part next to a
  quarter (`POST /demo/try?coin=true` composes it at the part's true scale and carries the
  coin through the photo-style warp via an augmenter mask); the report and `/analytics`
  show bought-top-1 with and without a measurement.
- Phone UI: the coin the server found is drawn on the photo as a dashed ring ("coin?") so
  the user can see what would set the scale before choosing the coin.
- The coin finder prefers the flatter of two round blobs (even brightness), so a screw head
  with a socket, a knob or a pulley next to a real coin is no longer offered as the coin.
- `mcv report`: the purchase-loop analytics (funnel, bought-top-1, measured vs not, tiers,
  weakest categories, latency, learning state, daily trend, issues) from the event log on
  disk, for operators without the dashboard; `--json` for scripts.
- Purchases by category: `/analytics` and the dashboard show which categories customers get
  wrong most (top-1 precision when bought) with an issue for the weakest; identify events
  carry the category. A `learner` service in both compose files runs `mcv learn` hourly.
- Phone UI: "Your orders" in the cart drawer with one-tap "Order again".
- Analytics: a per-day trend (identifications, parts bought, bought-top-1 rate, learn and
  retrain events, which `mcv learn` / `mcv retrain` now write to the event log) on
  `/analytics` and the dashboard, so the loop's effect is visible over time.
- `mcv selfcheck` gains a purchase-loop step: cart, checkout, the photo becomes a purchase
  confirmation, `mcv learn` adds it to the index incrementally.
- Calibration: when no threshold reaches a tier's precision target, the fit raises the
  threshold to the most precise supported one instead of keeping a default known to be
  wrong (never lowered); the tier-precision issue says when a stronger backbone is needed.
- Synthetic catalog: pipe nipples, couplings, flanges, caps and hex reducing bushings with
  pipe size, thread type (NPT / NPTF / BSPT), gender, length and reduced-to attributes, so
  the pipe-sizing and thread-compatibility features have parts to act on (44 kinds).

## 0.3.6

- Training review: the trainer no longer warm-starts from `MCV_BACKBONE_CHECKPOINT` by
  accident (`warm_start` recipe key); small catalogs no longer train zero steps; web imports
  get a family key and splits fall back to the category (no look-alike leakage); unreadable
  confirmation photos are dropped from the view cache instead of becoming black positives;
  the mild first cache only when a refresh follows; ties keep the later epoch and tiny
  validation splits are ignored; validation and mining see serving preprocessing; one
  confirmed photo per part is held out (2+); the cached trainer is TinyCNN-only; the dataset
  pickles under spawn; transparent PNGs composite on white; ONNX export pins the dynamo
  exporter (torch >= 2.5).
- Second-pass server review: IPv4-mapped rate-limit buckets, multi-worker serve honours the
  resolved settings, retrain switches on the full model version and keeps calibration beside
  the sibling index, `mcv up` rebuilds on a version mismatch, IPv6 port probe, cross-worker
  request ids, a failed index load is not retried every 15 s.
- `mcv selfcheck`; evaluation A/B: query expansion stays off.

- Phone UI review: the `hidden` attribute now wins over class display rules (the live
  overlay, install and connect buttons and the coin chooser were always visible); yellow
  badges readable in dark mode; one reset path for every new query so measure points and
  stale results never carry over; request sequencing for samples, search and "Start over";
  the shutter always starts a new query; camera and Live ID generation guards; the offline
  outbox stores a small image and reports a real failure; confirming a sample files
  feedback; keyboard-focusable capture buttons; Escape closes the lightbox.
- Thread pitch review: the reference coin is excluded from the profile; short profiles and
  hole edges no longer yield a pitch; mixed-number and `M6-1.0` threads, `threads_per_inch`
  attributes; bare fastener sizes are not pipe threads; NPT/BSP band tightened to ±1.5%.
- `identify-dir --coin` (the coin in each photo sets the scale) and `import-pages --dry-run`.

## 0.3.5

- Thread pitch from the photo: with a scale set, the crest period along the part's axis
  gives the pitch (threads per inch); candidates whose catalog thread (`1/4"-20`, `M6 x 1`,
  `#8-32`, or pipe size + NPT/BSP) disagrees lose score. Shown in the verdict.
- Evaluation reports Recall by category and the hardest queries; the dashboard shows the
  last measured accuracy.
- Page importer review fixes: interleaved two-column rows, wrapped rows whose continuation
  starts with a price, placeholder cells, "(cont.)" tables, dimensions mistaken for prices,
  material-named page titles, headers not starting with "Pipe Size", page numbering and
  de-duplication across files.
- Size matching for pipe fittings compares the silhouette with the pipe OD for both
  genders (a female body wraps the pipe, 1.1x to 1.6x); "female" no longer reads as male.
- Pipe size parsing: zero denominators, `NPTF`, and `1/2 in.`; a round part alone in the
  frame is no longer offered as the coin; measurement notes get a "clear scale" action.
- `mcv serve` / `mcv up` refuse a busy port with a clear message.

## 0.3.4

Built from the catalog's own "Selecting and Measuring Pipe & Fittings" pages.

- Pipe sizing knowledge (`pipeline/pipe.py`): nominal pipe size -> real OD (male threads)
  and schedule 40 ID (female threads), threads per inch for NPT vs BSP, and the thread
  compatibility table. Size matching now compares a fitting by its pipe OD/ID, not the
  nominal number (a "3/8" fitting is 0.675" across, not 3/8").
- `mcv import-pages`: parts from the OCR text of printed catalog pages (part numbers, pipe
  sizes, materials, fitting types, lengths, prices, page), with look-alike families per
  fitting type and material. Images come later via `fetch-images` / `import-web`.
- Part pages show the real dimensions of a fitting's pipe size, its thread pitch, and which
  thread types mate with it.

## 0.3.3

- Coin hint: a round blob next to the part is reported (`coin_hint`) and the app offers it
  as the scale reference in one pick; no tapping needed.

Bug-hunt release: four independent reviews plus API fuzzing and concurrency probes; every
finding below has a regression test.

- Security: quotes in catalog strings could break out of inline handlers (stored XSS);
  `/feedback` accepted unlimited uploads for any id; 500s leaked paths; thumbnails wrote
  unbounded cache files; part numbers are now URL-encoded everywhere.
- Serving: the CPU gate no longer exhausts the thread pool (health checks and pages stayed
  responsive under load); logging and photo storage run off the event loop; batch requests
  are charged per photo by the rate limiter; IPv6 clients are bucketed per /64; concurrent
  backups serialise; a rebuilt or restored index with a different backbone is refused
  instead of failing on every photo; `mcv retrain` with a different backbone writes a
  sibling index and prints the switch, never clobbering the live one; `mcv up` reuses a
  demo index with the backbone it was built with.
- Catalog: FTS maintenance was quadratic (20k re-ingest took 77 s; now 2.7 s, index-backed); a
  quote-only search crashed; Excel "CSV UTF-8" (BOM) and ragged rows crashed ingest;
  `__MACOSX` / `.cache` clutter turned a photo folder into a folder-per-part source;
  `import-web` wiped existing images (now merges); robots.txt wildcards and stacked
  user-agents; JSON-LD `ImageObject` and breadcrumb shapes; unsafe page-supplied part
  numbers could write outside the image directory; redirected pages resolve links
  against their final URL.
- Index and training: FAISS asserted on an empty index; ids/vector counts are checked on
  load; ensemble members no longer all load one checkpoint; calibration keeps LIKELY
  below EXACT; the cached trainer crashed on the sampler's tail batch; grayscale
  confirmation photos and lower-case feedback folders are handled; the Claude reranker's
  schema was being rejected by structured outputs (now closed, no bounds, no refs).

## 0.3.2

- Size matching: a "Measure" tool in the phone UI (tap the two ends of a coin, a card edge
  or 1 inch on a ruler) gives the photo's scale; the object's extent is measured along its
  principal axes and compared with each candidate's catalog dimensions (length, OD, thread
  size), which separates look-alike SKUs that share one catalog image. The marked
  reference is excluded from the measurement. API: `mm_per_px` and `ref`; CLI `mcv identify --mm-per-px`.
- Service worker is network-first: phones no longer keep an old UI or a stale dashboard.
- `mcv validate` / `bootstrap` report how many parts carry a parseable dimension.

## 0.3.1

- Nothing gets lost: query photos wait on disk so a confirmation lands after a restart or
  on another worker; the request log reloads at boot; feedback re-confirmations replace
  instead of double-counting; `mcv backup` / `mcv restore` (+ `POST /admin/backup`,
  `GET /admin/backups`) bundle catalog, index, calibration, photos, logs and manifest.
- `mcv build-index --with-feedback` and `mcv retrain` embed confirmed photos as gallery
  entries, so a part photographed once is found again from that angle without training.
- The API picks up a rebuilt or restored index automatically (`MCV_AUTO_RELOAD`).
- Phone UI: confirmations made offline wait in an outbox and sync when the network is back.
- Reranker: a small usage prior from confirmation counts breaks ties toward parts people
  actually confirm (log-scaled; never overrides visual evidence).
- Dashboard: storage panel with sizes, ages, last backup and a "Back up now" button;
  `mcv doctor` flags a missing or stale backup.
- Uploads: decompression-bomb guard and reduced-scale JPEG decoding (a 12 MP JPEG decodes in
  116 ms instead of 165 ms on one CPU core);
  `/health` reports uptime, requests served and index/backbone mismatch.

## 0.3.0

- McMaster-inspired theme (dark-green header, yellow selection, dense spec tables) across the app and all pages; clearly labelled unofficial.
- New pages: `/browse` (taxonomy + part grid), `/part/{pn}` (gallery, specs, look-alike family), `/dashboard` (stats, tiers, recent identifications, build info).
- `/search` category filter and offset; Live ID; concurrency gate; `/categories`.

## 0.2.2

- Live ID: continuous identification overlay on the live camera; preview frames skip the request log.
- Concurrency gate (MCV_MAX_CONCURRENCY) and `/categories`.
- `mcv up`: one-command demo/serve with QR code; demo mode (sample parts identified live, printable sheet, `/connect`); install button; CORS setting; DEMO.md.
- Query-embedding cache; `mcv up` defaults to TinyCNN for a sub-minute first build.

## 0.2.1

- Shipped checkpoint retrained on 8,000 synthetic parts (held-out Recall@1 0.33 vs 0.29).
- Phone interface v3: live camera, compare lightbox, image strips, spec table, history, Fast/Accurate.
- Thumbnails, gzip, TTA modes, query-embedding cache; `mcv serve --qr/--https`; Caddy deployment.

## 0.2.0

Groundwork complete: everything except real McMaster-Carr imagery.

- Intake: `mcv validate`, normalisation and de-duplication, `mcv fetch-images`
  (URL lists), `mcv import-web` (product pages, polite), screenshot folders,
  `mcv enrich` (metadata for image-only drops), `mcv bootstrap` one-command
  pipeline with `data/manifest.json`.
- Retrieval: TTA multi-query, gallery augmentation, alpha query expansion,
  category prior, FAISS auto-selection above 50k vectors, parallel and
  incremental index builds with count-weighted centroids, atomic index swaps.
- Models: TinyCNN trained from scratch (shipped `assets/tinycnn_synthetic.pt`
  + model card), cached-view trainer (SupCon + classification, hard negatives,
  curriculum), ensemble backbone, CLIP / DINOv2 adapters, ONNX / TorchScript.
- Answers: calibrated tiers with precision-targeted thresholds, family answers
  with distinguishing attributes, attribute constraints (loose matching),
  category guesses, honest `notes`.
- Interface: camera-first PWA, several angles per query, one-tap confirmation
  feedback, text search, batch endpoint and `mcv identify-dir`, HEIC support.
- Operations: `doctor`, `status` (incl. `index_stale`), `metrics`, request log,
  `retrain` (cron) with held-out evaluation, hot reload, rate limit, API token,
  multi-worker serving, `export-dataset`, `review-unknowns`, RUNBOOK.md.
- Quality: 79 tests including a real-browser UI run; two code-review passes
  with all findings fixed.

## 0.1.0

Initial skeleton: catalog store, synthetic renderer, hash / CLIP / DINOv2
backbones, numpy / FAISS index, identification pipeline with Claude vision
reranker, FastAPI service and upload UI, training loop, CLI, tests, CI.
