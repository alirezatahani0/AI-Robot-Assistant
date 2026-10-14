"""Tiny local web page that shows the live transcript with proper RTL Persian."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_PAGE = """<!doctype html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8">
<title>Robot — متن زنده</title>
<style>
  body { margin: 0; padding: 32px; background: #111; color: #eee;
         font-family: "Vazirmatn", "Geeza Pro", "Noto Naskh Arabic", Tahoma, sans-serif;
         font-size: 30px; line-height: 1.9; }
  #status { font-size: 16px; color: #888; margin-bottom: 24px; }
  #committed { white-space: pre-wrap; }
  #partial { color: #7fb3ff; }
  #done { display: none; margin-top: 32px; padding: 16px; border: 1px solid #3a6;
          border-radius: 8px; font-size: 18px; color: #8d8; }
</style>
</head>
<body>
<div id="status">گوش می‌دهم… فارسی حرف بزن. برای پایان در ترمینال Ctrl+C را بزن.</div>
<div><span id="committed"></span> <span id="partial"></span></div>
<div id="done">متن نهایی — لطفاً بررسی و تایید کن.</div>
<script>
async function poll() {
  try {
    const r = await fetch("/state", {cache: "no-store"});
    const s = await r.json();
    document.getElementById("committed").textContent = s.committed;
    document.getElementById("partial").textContent = s.partial;
    if (s.done) {
      document.getElementById("status").textContent = "تمام شد.";
      document.getElementById("done").style.display = "block";
      return;
    }
  } catch (e) {
    document.getElementById("status").textContent = "اتصال قطع شد.";
    return;
  }
  setTimeout(poll, 300);
}
poll();
</script>
</body>
</html>
"""


class LiveView:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self._lock = threading.Lock()
        self._state = {"committed": "", "partial": "", "done": False}
        view = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path.startswith("/state"):
                    with view._lock:
                        body = json.dumps(view._state, ensure_ascii=False)
                    self._send(body, "application/json; charset=utf-8")
                elif self.path in {"/", "/index.html"}:
                    self._send(_PAGE, "text/html; charset=utf-8")
                else:
                    self.send_error(404)

            def _send(self, body: str, ctype: str) -> None:
                data = body.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format, *args) -> None:  # noqa: A002
                return None

        self._server = ThreadingHTTPServer((host, port), _Handler)
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/"

    def start(self) -> None:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def update(self, committed: str, partial: str) -> None:
        with self._lock:
            self._state = {"committed": committed, "partial": partial, "done": False}

    def finish(self, final_text: str) -> None:
        with self._lock:
            self._state = {"committed": final_text, "partial": "", "done": True}

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
