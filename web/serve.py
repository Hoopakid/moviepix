"""Serve the demo page on this computer only:  python3 web/serve.py  ->  http://127.0.0.1:8000

Adds the two headers that let ONNX Runtime Web use several CPU threads (cross-origin isolation),
and the right types for .mjs and .wasm. Listens on 127.0.0.1, so nothing outside this Mac can reach it.
"""
import argparse
import functools
import http.server
from pathlib import Path


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm",
                      ".json": "application/json", ".onnx": "application/octet-stream"}

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    root = Path(__file__).resolve().parent
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), functools.partial(Handler, directory=str(root)))
    print(f"moviepix demo: http://127.0.0.1:{a.port}  (Ctrl+C stops it)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
