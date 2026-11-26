"""Screenshot the whole app (empty state and a loaded photo) for a layout check."""
from __future__ import annotations

import base64
import json
import urllib.request
from pathlib import Path

from cdp_test import PORT, CDP, launch

ROOT = Path(__file__).resolve().parent

LOAD = r"""
(async () => {
  const $ = (id) => document.getElementById(id);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const frame = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  $("sampleBtn").click();
  for (let i = 0; i < 900; i++) { if (!$("exportBtn").disabled) break; await sleep(200); }
  await frame();
  const rect = $("view").getBoundingClientRect();
  $("view").dispatchEvent(new MouseEvent("click", {
    bubbles: true,
    clientX: rect.left + 0.17 * rect.width,
    clientY: rect.top + 0.72 * rect.height,
  }));
  document.querySelector('#lens button[data-mm="85"]').click();
  const el = $("fstop"); el.value = "9"; el.dispatchEvent(new Event("input", {bubbles: true}));
  await frame(); await sleep(400);
  return "ok";
})()
"""


def shot(cdp: CDP, name: str) -> None:
    res = cdp.send("Page.captureScreenshot", format="png")
    path = ROOT / f"docs_shot_{name}.png"
    path.write_bytes(base64.b64decode(res["data"]))
    print(f"{name}: {path} ({path.stat().st_size/1024:.0f} KB)")


def main() -> None:
    proc = launch()
    try:
        targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list").read())
        page = next(t for t in targets if t["type"] == "page")
        cdp = CDP(page["webSocketDebuggerUrl"])
        cdp.send("Runtime.enable")
        cdp.send("Page.enable")
        for _ in range(60):
            if cdp.eval("document.readyState") == "complete":
                break
        import time

        time.sleep(2)
        shot(cdp, "empty")
        print(cdp.eval(LOAD, await_promise=True, timeout_s=420))
        shot(cdp, "loaded")
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
