"""Live browser view for Persian transcripts (terminals can't render RTL)."""

from __future__ import annotations

import json
import urllib.request

from robot.live_view import LiveView


def _get(url: str) -> tuple[str, str]:
    with urllib.request.urlopen(url, timeout=2) as resp:
        return resp.headers.get("Content-Type", ""), resp.read().decode("utf-8")


def test_state_endpoint_returns_current_persian_text() -> None:
    view = LiveView(port=0)
    view.start()
    try:
        view.update(committed="سلام خوبی", partial="حرف می‌زنم")
        ctype, body = _get(view.url + "state")
        data = json.loads(body)
        assert "application/json" in ctype
        assert data["committed"] == "سلام خوبی"
        assert data["partial"] == "حرف می‌زنم"
        assert data["done"] is False

        view.finish("سلام خوبی حرف می‌زنم")
        data = json.loads(_get(view.url + "state")[1])
        assert data["committed"] == "سلام خوبی حرف می‌زنم"
        assert data["partial"] == ""
        assert data["done"] is True
    finally:
        view.stop()


def test_page_is_right_to_left_utf8() -> None:
    view = LiveView(port=0)
    view.start()
    try:
        ctype, body = _get(view.url)
        assert "charset=utf-8" in ctype.lower()
        assert 'dir="rtl"' in body
    finally:
        view.stop()


def test_listen_web_flag_defaults_off() -> None:
    import importlib

    cli = importlib.import_module("robot.main")
    assert cli.build_parser().parse_args(["listen"]).web is False
    assert cli.build_parser().parse_args(["listen", "--web"]).web is True
