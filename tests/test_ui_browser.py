"""Browser-level test of the one-photo interface: upload a photo through the page
and check that a verdict card with the right part number appears. Skipped when
Playwright or Chromium is unavailable."""

from __future__ import annotations

import glob
import os
import socket
import threading
import time

import pytest
from PIL import Image

pw = pytest.importorskip("playwright.sync_api")


def _chromium_path() -> str | None:
    """A Playwright Chromium build: PLAYWRIGHT_BROWSERS_PATH, or the default cache that
    `playwright install chromium` fills (CI)."""
    roots = [
        os.environ.get("PLAYWRIGHT_BROWSERS_PATH", ""),
        os.path.expanduser("~/.cache/ms-playwright"),
    ]
    for root in roots:
        if not root:
            continue
        candidates = sorted(glob.glob(os.path.join(root, "chromium-*", "chrome-linux*", "chrome")))
        if candidates:
            return candidates[-1]
    return None


@pytest.fixture(scope="module")
def server(identifier, tmp_path_factory):
    import uvicorn

    from mcmaster_vision.api import create_app
    from mcmaster_vision.config import Settings

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    config = uvicorn.Config(
        create_app(
            Settings(queries_dir=tmp_path_factory.mktemp("queries"), demo_mode=True),
            identifier=identifier,
        ),
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(timeout=5)


def test_take_photo_flow(server, store, tmp_path):
    exe = _chromium_path()
    if exe is None:
        pytest.skip("no Playwright Chromium build available")
    part = next(store.iter_parts(with_images_only=True))
    photo = tmp_path / "photo.jpg"
    Image.open(part.image_paths[0]).convert("RGB").save(photo, format="JPEG", quality=90)

    with pw.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 390, "height": 800})
        page.goto(server + "/")
        assert page.locator("label.btn.primary").is_visible()
        page.set_input_files("#camera", str(photo))
        page.wait_for_selector(".verdict", timeout=30000)
        verdict = page.locator(".verdict").inner_text()
        assert part.part_number in verdict
        assert page.locator(".cand").count() >= 1
        page.screenshot(path=str(tmp_path / "result.png"), full_page=True)
        # "This is it" -> /feedback files the photo under the confirmed part number
        page.locator(".cand .confirm button.yes").first.click()
        page.wait_for_function(
            "document.querySelector('.cand .confirm button.yes').textContent.startsWith('Saved as')",
            timeout=30000,
        )
        # offline confirmation: /feedback unreachable -> outbox, synced once the network is back
        page.evaluate(
            "() => { lastResult = Object.assign({}, lastResult, {request_id: 'offline0001'}); "
            "document.querySelectorAll('.cand .confirm button').forEach(b => { b.disabled = false; "
            "b.classList.remove('done'); }); }"
        )
        page.route("**/feedback", lambda route: route.abort())
        page.locator(".cand .confirm button.yes").first.click()
        page.wait_for_function(
            "document.querySelector('.cand .confirm button.yes').textContent.includes('will sync')",
            timeout=30000,
        )
        assert page.evaluate("JSON.parse(localStorage.getItem('mcv.outbox')).length") == 1
        assert not page.locator("#outbox").is_hidden()
        page.unroute("**/feedback")
        page.evaluate("window.dispatchEvent(new Event('online'))")
        page.wait_for_function(
            "JSON.parse(localStorage.getItem('mcv.outbox') || '[]').length === 0", timeout=30000
        )
        assert page.locator("#outbox").is_hidden()
        # text search fallback
        page.fill("#q", part.part_number)
        page.click("#searchform button")
        page.wait_for_function(
            "document.querySelectorAll('#results .cand').length >= 1 && "
            f"document.body.innerText.includes('{part.part_number}')",
            timeout=30000,
        )
        # paste path: clipboard image -> identify
        page.evaluate(
            "async () => {"
            f"const r = await fetch('/parts/{part.part_number}/image'); const blob = await r.blob();"
            "const dt = new DataTransfer(); dt.items.add(new File([blob], 'shot.png', {type: 'image/png'}));"
            "window.dispatchEvent(new ClipboardEvent('paste', {clipboardData: dt}));"
            "}"
        )
        page.wait_for_function("document.querySelectorAll('.cand').length >= 1", timeout=30000)
        browser.close()
    assert (tmp_path / "result.png").stat().st_size > 1000
    (tmp_path / "result.png").replace(os.environ.get("MCV_UI_SHOT", str(tmp_path / "result.png")))


def test_demo_sample_flow(server, tmp_path):
    """Demo mode: tap a sample part -> a photo-style render is identified and badged."""
    exe = _chromium_path()
    if exe is None:
        pytest.skip("no Playwright Chromium build available")
    with pw.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.goto(server + "/")
        page.wait_for_selector("#samplestrip img", timeout=30_000)
        page.locator("#samplestrip img").first.click()
        page.wait_for_selector(".verdict .badge", timeout=60_000)
        badge = page.locator(".verdict .badge").inner_text()
        assert badge == "correct" or "ranked" in badge or badge == "missed"
        # the hidden attribute must win over class display rules
        assert page.locator("#mchoose").is_hidden() and page.locator("#overlay").is_hidden()
        assert page.locator("#install").is_hidden()
        # a sample is a real identification: confirming it files feedback
        page.locator(".cand .confirm button.yes").first.click()
        page.wait_for_function(
            "document.querySelector('.cand .confirm button.yes').textContent.startsWith('Saved as')",
            timeout=30000,
        )
        assert page.locator("#preview img").get_attribute("src").startswith("/demo/query/")
        page.screenshot(path=str(tmp_path / "demo.png"), full_page=True)
        (tmp_path / "demo.png").replace(
            os.environ.get("MCV_UI_SHOT_DEMO", str(tmp_path / "demo.png"))
        )
        page.goto(server + "/demo/sheet?n=6")
        assert page.locator("figure").count() == 6
        browser.close()


def test_live_id_overlay_with_fake_camera(server, tmp_path):
    """Live ID: Chromium's fake camera feeds frames; the overlay must show a running guess."""
    exe = _chromium_path()
    if exe is None:
        pytest.skip("no Playwright Chromium build available")
    with pw.sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=exe,
            args=[
                "--no-sandbox",
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
            ],
        )
        ctx = browser.new_context(viewport={"width": 390, "height": 844}, permissions=["camera"])
        page = ctx.new_page()
        page.goto(server + "/")  # 127.0.0.1 is a secure context, so the live camera button appears
        page.wait_for_selector("#livebtn:not([hidden])", timeout=15_000)
        page.click("#livebtn")
        page.wait_for_function("document.getElementById('video').videoWidth > 0", timeout=30_000)
        page.click("#livetoggle")
        page.wait_for_function(
            "(() => { const o = document.getElementById('overlay'); return !o.hidden && /ms|No match/.test(o.innerText); })()",
            timeout=60_000,
        )
        page.screenshot(path=str(tmp_path / "live.png"))
        (tmp_path / "live.png").replace(
            os.environ.get("MCV_UI_SHOT_LIVE", str(tmp_path / "live.png"))
        )
        page.click("#shutter")
        page.wait_for_selector(".verdict", timeout=60_000)
        assert page.locator("#livetoggle").get_attribute("class") in (
            None,
            "ghost",
            "ghost ",
        )  # live stops on capture
        browser.close()


def test_measure_tool_sets_scale_and_matches_sizes(server, store, tmp_path):
    """Measure: two taps on the photo + a reference length -> mm_per_px is sent and the
    verdict shows the measured size."""
    exe = _chromium_path()
    if exe is None:
        pytest.skip("no Playwright Chromium build available")
    from PIL import ImageDraw

    part = next(store.iter_parts(with_images_only=True))
    photo = tmp_path / "photo.jpg"
    # a "coin" on the left, the part on the right (512 x 256): the user marks the coin
    render = Image.open(part.image_paths[0]).convert("RGB").resize((256, 256))
    canvas = Image.new("RGB", (512, 256), (255, 255, 255))
    ImageDraw.Draw(canvas).ellipse((40, 48, 200, 208), fill=(184, 172, 120))
    canvas.paste(render, (256, 0))
    canvas.save(photo, format="JPEG", quality=90)
    with pw.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 390, "height": 800})
        page.goto(server + "/")
        page.set_input_files("#camera", str(photo))
        page.wait_for_selector(".verdict", timeout=30000)
        # the server spotted the coin: one pick sets the scale without any tapping
        page.wait_for_selector("#coinpick", timeout=10000)
        page.select_option("#coinpick", "24.26")
        page.wait_for_function(
            "document.body.innerText.includes('Measured from your photo')", timeout=30000
        )
        assert page.locator("#coinpick").count() == 0  # the hint goes away once a scale is set
        page.evaluate("(async () => { scale = null; refSeg = null; await send(); })()")
        page.wait_for_selector("#coinpick", timeout=30000)
        assert page.locator("#msvg").is_hidden()  # the overlay must not block the lightbox
        page.click("#measurebtn")
        assert page.locator("#msvg").is_visible() and page.locator("#measurebox").is_visible()
        box = page.locator("#img").bounding_box()
        # the 2:1 photo is letterboxed in the square box; tap the coin's left and right
        # edges (x 40..200 of 512) on the photo's middle row
        y = box["y"] + box["height"] / 2
        page.mouse.click(box["x"] + box["width"] * (48 / 512), y)
        page.mouse.click(box["x"] + box["width"] * (192 / 512), y)
        page.wait_for_selector("#mchoose:not([hidden])")
        assert "px" in page.locator("#mhint").inner_text()
        page.select_option("#mref", "25.4")
        page.click("#muse")
        page.wait_for_function(
            "document.body.innerText.includes('Measured from your photo')", timeout=30000
        )
        measured = page.evaluate("lastResult.measured")
        # the coin (144 px marked as 25.4 mm) is excluded; the 256 px render measures
        # well under the 45 mm the coin would give, and not as a 25 mm disc
        assert measured and 20 < measured["long_mm"] < 60
        scale = page.evaluate("scale")
        assert scale and scale > 0
        ref = page.evaluate("refSeg")
        assert ref and len(ref) == 4 and abs(ref[2] - ref[0]) > 50  # the drawn line, uploaded px
        # every candidate now carries a size verdict when it has a dimension to compare
        assert page.evaluate(
            "lastResult.candidates.some(c => c.reasons.some(r => r.includes('measured')))"
        )
        page.screenshot(path=str(tmp_path / "measure.png"), full_page=True)
        (tmp_path / "measure.png").replace(
            os.environ.get("MCV_UI_SHOT_MEASURE", str(tmp_path / "measure.png"))
        )
        page.click("#mclear")
        page.wait_for_function("scale === null")
        assert page.locator("#msvg").is_hidden()
        browser.close()


def test_clear_scale_note_action(server, store, tmp_path):
    """When the marked reference is the only blob, the note offers 'clear scale' and using
    it removes the scale and re-queries."""
    exe = _chromium_path()
    if exe is None:
        pytest.skip("no Playwright Chromium build available")
    part = next(store.iter_parts(with_images_only=True))
    photo = tmp_path / "photo.jpg"
    Image.open(part.image_paths[0]).convert("RGB").save(photo, format="JPEG", quality=90)
    with pw.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 390, "height": 800})
        page.goto(server + "/")
        page.set_input_files("#camera", str(photo))
        page.wait_for_selector(".verdict", timeout=30000)
        # mark the part itself as the reference: nothing is left to measure
        page.evaluate(
            "(async () => { scale = 0.1; refSeg = [10, 128, 246, 128]; await send(); })()"
        )
        page.wait_for_function("document.body.innerText.includes('size not used')", timeout=30000)
        assert page.locator("button:has-text('clear scale')").count() == 1
        page.click("button:has-text('clear scale')")
        page.wait_for_function(
            "scale === null && !document.body.innerText.includes('size not used')",
            timeout=30000,
        )
        browser.close()


def test_cart_to_checkout_flow(server, tmp_path):
    """Identify a sample, add it to the cart, check out: the order confirmation appears
    and the purchase is filed as a checkout confirmation."""
    exe = _chromium_path()
    if exe is None:
        pytest.skip("no Playwright Chromium build available")
    import httpx

    with pw.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.goto(server + "/")
        page.wait_for_selector("#samplestrip img", timeout=30_000)
        page.locator("#samplestrip img").first.click()
        page.wait_for_selector(".cand button.buy", timeout=60_000)
        page.locator(".cand button.buy").first.click()
        page.wait_for_function("document.getElementById('cartn').textContent === '1'")
        assert page.locator("#cartpill").is_visible()
        page.locator("#cartpill").click()
        page.wait_for_selector("#cart.on .citem")
        page.locator(".citem .qty button").nth(1).click()  # +
        page.wait_for_function("document.getElementById('cartn').textContent === '2'")
        page.locator("#checkoutbtn").click()
        page.wait_for_selector("#cartorder .order", timeout=30_000)
        text = page.locator("#cartorder").inner_text()
        assert "Order" in text and "placed" in text and "teaches the model" in text
        assert page.locator("#cartn").inner_text() == "0"
        page.screenshot(path=str(tmp_path / "checkout.png"), full_page=True)
        # the order is listed (drawer still open) and can be re-ordered in one tap
        page.wait_for_selector("#pastorders:not([hidden])")
        page.locator("#pastorders summary").click()
        page.locator("#pastlist button").first.click()
        page.wait_for_function("document.getElementById('cartn').textContent === '2'")
        browser.close()
    orders = httpx.get(server + "/orders").json()
    assert orders and orders[0]["items"][0]["quantity"] == 2
    a = httpx.get(server + "/analytics").json()
    assert a["window"]["checkout"] >= 1 and a["learning"]["new_purchases"] >= 1


def test_part_page_add_to_cart_button_works(server, store, tmp_path):
    exe = _chromium_path()
    if exe is None:
        pytest.skip("no Playwright Chromium build available")
    import httpx

    part = next(store.iter_parts(with_images_only=True))
    with pw.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 390, "height": 844})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{server}/part/{part.part_number}")
        page.locator("#addcart").click()
        page.wait_for_function("document.getElementById('addcart').textContent.includes('In cart')")
        cid = page.evaluate("localStorage.getItem('mcv.client')")
        assert not errors, errors
        browser.close()
    cart = httpx.get(f"{server}/cart?client_id={cid}").json()
    assert cart and cart[0]["part_number"] == part.part_number
