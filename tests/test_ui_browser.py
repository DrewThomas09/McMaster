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
    root = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")
    candidates = (
        glob.glob(os.path.join(root, "chromium-*", "chrome-linux*", "chrome")) if root else []
    )
    return candidates[0] if candidates else None


@pytest.fixture(scope="module")
def server(identifier):
    import uvicorn

    from mcmaster_vision.api import create_app
    from mcmaster_vision.config import Settings

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    config = uvicorn.Config(
        create_app(Settings(), identifier=identifier),
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
        # paste path: clipboard image -> identify
        page.evaluate(
            """async () => {
                const r = await fetch('/parts/%s/image'); const blob = await r.blob();
                const dt = new DataTransfer(); dt.items.add(new File([blob], 'shot.png', {type: 'image/png'}));
                window.dispatchEvent(new ClipboardEvent('paste', {clipboardData: dt}));
            }"""
            % part.part_number
        )
        page.wait_for_function("document.querySelectorAll('.cand').length >= 1", timeout=30000)
        browser.close()
    assert (tmp_path / "result.png").stat().st_size > 1000
    (tmp_path / "result.png").replace(os.environ.get("MCV_UI_SHOT", str(tmp_path / "result.png")))
