"""Fetch the pinned ONNX Runtime Web files the demo page needs into web/vendor/onnxruntime-web/.

    uv run python -I scripts/fetch_ort_web.py

The page loads nothing from a CDN at run time, but the runtime's WebAssembly files are 15-30 MB each,
so they are fetched once (here) instead of living in git. The npm tarball is checked against its
published sha512 before anything is read from it, and only the six files below are written.
onnxruntime-web is MIT-licensed (Microsoft); see web/vendor/onnxruntime-web/SOURCE.txt.
"""
import base64
import hashlib
import io
import tarfile
import urllib.request
from pathlib import Path

VERSION = "1.31.1"
URL = f"https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{VERSION}.tgz"
SHA512 = "5GeH+bWQ+HxM10z7erGlsWqryrwMbSvm/1tkIsKTT48g0fI19VXVVC435t4t7VsPdSQeS7NmuZPXrQ73k82URw=="
FILES = (  # WASM build (default) and the WebGPU build (loaded only when chosen)
    "ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm",
    "ort.webgpu.min.mjs", "ort-wasm-simd-threaded.asyncify.mjs", "ort-wasm-simd-threaded.asyncify.wasm",
)
OUT = Path(__file__).resolve().parent.parent / "web/vendor/onnxruntime-web"


def main():
    req = urllib.request.Request(URL, headers={"User-Agent": "moviepix-fetch"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read(200_000_000)
    got = base64.b64encode(hashlib.sha512(data).digest()).decode()
    if got != SHA512:
        raise SystemExit(f"sha512 mismatch for {URL}: got {got}")
    OUT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = {m.name: m for m in tar.getmembers()}
        for name in FILES:
            m = members.get(f"package/dist/{name}")
            if m is None or not m.isfile():
                raise SystemExit(f"{name} is missing from the package or is not a regular file")
            (OUT / name).write_bytes(tar.extractfile(m).read())
            print(f"{name:40s} {m.size / 1e6:6.1f} MB")
    (OUT / "SOURCE.txt").write_text(f"onnxruntime-web {VERSION}\n{URL}\nsha512-{SHA512}\nLicence: MIT (Microsoft "
                                    "Corporation), https://github.com/microsoft/onnxruntime/blob/main/LICENSE\n")
    print(OUT)


if __name__ == "__main__":
    main()
