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
    part = next(store.iter_parts(with_images_only=True))
    photo = tmp_path / "photo.jpg"
    Image.open(part.image_paths[0]).convert("RGB").save(photo, format="JPEG", quality=90)
    with pw.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 390, "height": 800})
        page.goto(server + "/")
        page.set_input_files("#camera", str(photo))
        page.wait_for_selector(".verdict", timeout=30000)
        assert page.locator("#msvg").is_hidden()  # the overlay must not block the lightbox
        page.click("#measurebtn")
        assert page.locator("#msvg").is_visible() and page.locator("#measurebox").is_visible()
        box = page.locator("#img").bounding_box()
        # the photo is square, so it fills the box: tap at 20% and 80% of the width
        y = box["y"] + box["height"] / 2
        page.mouse.click(box["x"] + box["width"] * 0.2, y)
        page.mouse.click(box["x"] + box["width"] * 0.8, y)
        page.wait_for_selector("#mchoose:not([hidden])")
        assert "px" in page.locator("#mhint").inner_text()
        page.select_option("#mref", "25.4")
        page.click("#muse")
        page.wait_for_function(
            "document.body.innerText.includes('Measured from your photo')", timeout=30000
        )
        measured = page.evaluate("lastResult.measured")
        assert measured and measured["long_mm"] > 0
        scale = page.evaluate("scale")
        assert scale and scale > 0
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
