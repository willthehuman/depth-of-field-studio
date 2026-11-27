"""Drive headless Brave over CDP to prove the WebGL DOF pipeline actually works.

Checks, in order:
  1. the WebGL canvas shows the same photo as the served file (orientation/colour),
  2. the depth map puts the sky far and the foreground near (texture orientation),
  3. opening the aperture blurs depth bands that are away from the focus plane and
     leaves the in-focus band alone (circle-of-confusion follows the depth).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parent
PORT = 9346
URL = "http://127.0.0.1:7860/"
BROWSER = os.environ.get("DOF_BROWSER") or "/usr/sbin/chromium"
PROFILE = Path(os.environ.get("DOF_PROFILE", str(Path.home() / ".cache" / "dof-browser-test2")))


def launch() -> subprocess.Popen:
    PROFILE.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [
            BROWSER,
            "--headless=new",
            f"--remote-debugging-port={PORT}",
            f"--user-data-dir={PROFILE}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            "--window-size=1600,1000",
            "--hide-scrollbars",
            "--no-sandbox", "--disable-dev-shm-usage",
            "--enable-unsafe-swiftshader", "--use-gl=angle", "--use-angle=swiftshader",
            URL,
        ],
        stdout=subprocess.DEVNULL,
        stderr=open("/tmp/cdp-browser.log", "wb"),
    )
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version", timeout=1)
            return proc
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("browser did not expose CDP")


class CDP:
    def __init__(self, ws_url: str):
        self.ws = connect(ws_url, max_size=64 * 1024 * 1024)
        self.n = 0

    def send(self, method: str, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def eval(self, expr: str, await_promise: bool = False, timeout_s: int = 240):
        res = self.send(
            "Runtime.evaluate",
            expression=expr,
            awaitPromise=await_promise,
            returnByValue=True,
            timeout=timeout_s * 1000,
        )
        if res.get("exceptionDetails"):
            raise RuntimeError(json.dumps(res["exceptionDetails"])[:2000])
        return res["result"].get("value")


def main() -> None:
    proc = launch()
    try:
        targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list").read())
        page = next(t for t in targets if t["type"] == "page")
        cdp = CDP(page["webSocketDebuggerUrl"])
        cdp.send("Runtime.enable")

        for _ in range(60):
            if cdp.eval("document.readyState") == "complete":
                break
            time.sleep(0.5)

        script = (ROOT / "browser_test.js").read_text()
        out = cdp.eval(script, await_promise=True, timeout_s=420)
        print(out if isinstance(out, str) else json.dumps(out, indent=2))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
