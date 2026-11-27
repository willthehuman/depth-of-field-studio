#!/usr/bin/env python3
"""Headless probe for the DOF renderer.

Starts the synthetic server and a headless Chromium (SwiftShader), loads the
real client, sets the renderer state through the ?debug hook, paints one frame
and saves the canvas as a PNG.

    python3 tests/render_probe.py --out /tmp/base.png --focus 8 --fstop 3 --lens 135

The point is to test the WebGL pipeline without Depth Pro 2, torch or a GPU.
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parent.parent
PORT = 9341
URL = "http://127.0.0.1:7860/?debug=1"
CHROME = "/usr/sbin/chromium"
PROFILE = Path("/home/crow/.hermes/cache/scratch/dof-consult/chrome-profile-probe")


class CDP:
    def __init__(self, ws_url: str):
        self.ws = connect(ws_url, max_size=128 * 1024 * 1024)
        self.n = 0
        self.events: list[dict] = []

    def send(self, method: str, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
            self.events.append(msg)

    def eval(self, expr: str, await_promise: bool = False, timeout_s: int = 240):
        res = self.send("Runtime.evaluate", expression=expr, awaitPromise=await_promise,
                        returnByValue=True, timeout=timeout_s * 1000)
        if res.get("exceptionDetails"):
            raise RuntimeError(json.dumps(res["exceptionDetails"])[:3000])
        return res["result"].get("value")

    def wait_for(self, expr: str, timeout_s: float = 180.0):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                if self.eval(expr):
                    return True
            except Exception:
                pass
            time.sleep(0.25)
        return False


def wait_http(url: str, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except Exception:
            time.sleep(0.25)
    raise RuntimeError(f"{url} did not come up")


def launch_browser() -> subprocess.Popen:
    PROFILE.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([
        CHROME, "--headless=new", f"--remote-debugging-port={PORT}",
        f"--user-data-dir={PROFILE}", "--no-first-run", "--no-default-browser-check",
        "--no-sandbox", "--disable-dev-shm-usage", "--hide-scrollbars",
        "--window-size=1400,900", "--enable-unsafe-swiftshader",
        "--use-gl=angle", "--use-angle=swiftshader", URL,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait_http(f"http://127.0.0.1:{PORT}/json/version")
    return proc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--focus", type=float, default=8.0)
    ap.add_argument("--fstop", type=int, default=3, help="index into FSTOPS (3 = f/1.4)")
    ap.add_argument("--lens", type=float, default=135.0, help="0 = native, else mm")
    ap.add_argument("--blur", type=float, default=1.0)
    ap.add_argument("--highlight", type=float, default=0.4, help="0..1 -> 0..5 stops")
    ap.add_argument("--grain", type=float, default=0.3)
    ap.add_argument("--taps", type=int, default=96)
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--cx", type=float, default=0.5)
    ap.add_argument("--cy", type=float, default=0.5)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--mode", type=int, default=0)
    args = ap.parse_args()

    server = subprocess.Popen([sys.executable, str(ROOT / "tests" / "mock_depth_server.py"),
                               "--port", "7860"], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    browser = None
    try:
        wait_http("http://127.0.0.1:7860/api/status")
        browser = launch_browser()
        targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list").read())
        page = next((t for t in targets if t["type"] == "page" and "127.0.0.1" in t.get("url", "")), None)
        if page is None:
            page = next(t for t in targets if t["type"] == "page")
        cdp = CDP(page["webSocketDebuggerUrl"])
        cdp.send("Runtime.enable")
        cdp.send("Log.enable")
        cdp.send("Page.enable")
        if "127.0.0.1" not in page.get("url", ""):
            cdp.send("Page.navigate", url=URL)
        if not cdp.wait_for("document.readyState === 'complete' && !!document.getElementById('sampleBtn')", 60):
            raise RuntimeError("page did not load")

        # Load the synthetic sample through the real UI path.
        cdp.eval("document.getElementById('sampleBtn').click()")
        if not cdp.wait_for("window.__dof && !document.getElementById('exportBtn').disabled", 180):
            errs = [e for e in cdp.events if e.get("method") in
                    ("Runtime.exceptionThrown", "Log.entryAdded")]
            raise RuntimeError("sample never rendered; events: " + json.dumps(errs)[:4000])
        cdp.eval("document.getElementById('sampleBtn').blur()")

        js = f"""(() => {{
          const d = window.__dof;
          const c = document.getElementById('view');
          d.S.focus = {args.focus};
          d.S.fstopIdx = {args.fstop};
          d.S.lensMM = {args.lens};
          d.S.blur = {args.blur};
          d.S.highlight = {args.highlight};
          d.S.grain = {args.grain};
          d.S.mode = {args.mode};
          d.S.zoom = {args.zoom};
          d.S.cx = {args.cx};
          d.S.cy = {args.cy};
          d.S.holdOriginal = false;
          d.updateFocusUI();
          d.updateLensUI();
          const prevW = c.width, prevH = c.height;
          if ({str(args.export).lower()}) {{
            c.width = d.S.meta.width;
            c.height = d.S.meta.height;
          }}
          d.render({args.taps}, {str(args.export).lower()});
          const data = c.toDataURL('image/png');
          if ({str(args.export).lower()}) {{
            c.width = prevW;
            c.height = prevH;
            d.layout();
          }}
          return {{
            data,
            w: c.width, h: c.height,
            focus: d.S.focus, fstop: d.S.fstopIdx, lens: d.S.lensMM,
            meta: {{width: d.S.meta.width, height: d.S.meta.height, f_px: d.S.meta.f_px}},
          }};
        }})()"""
        result = cdp.eval(js, timeout_s=420)
        data = result.pop("data")
        png = base64.b64decode(data.split(",", 1)[1])
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(png)

        errs = [e for e in cdp.events if e.get("method") in
                ("Runtime.exceptionThrown", "Log.entryAdded")]
        print(json.dumps({"out": str(out), "bytes": len(png), "result": result,
                          "errors": errs[:5]}, indent=2))
    finally:
        if browser is not None:
            browser.terminate()
            try:
                browser.wait(timeout=10)
            except Exception:
                browser.kill()
        server.terminate()
        try:
            server.wait(timeout=10)
        except Exception:
            server.kill()


if __name__ == "__main__":
    main()
