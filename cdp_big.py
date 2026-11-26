"""Load a 27 MP photo through the real drag-and-drop path and check the render."""
from __future__ import annotations

import json
import shutil
import time
import urllib.request
from pathlib import Path

from cdp_test import PORT, CDP, launch

ROOT = Path(__file__).resolve().parent
shutil.copy(ROOT / "test_big.jpg", ROOT / "static/big.jpg")

SCRIPT = r"""
(async () => {
  const $ = (id) => document.getElementById(id);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const frame = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  const out = {};
  const t0 = performance.now();

  const blob = await (await fetch("/big.jpg")).blob();
  const file = new File([blob], "big.jpg", { type: "image/jpeg" });
  const dt = new DataTransfer();
  dt.items.add(file);
  $("stage").dispatchEvent(new DragEvent("drop", { dataTransfer: dt, bubbles: true, cancelable: true }));

  for (let i = 0; i < 3000; i++) { if (!$("exportBtn").disabled) break; await sleep(100); }
  out.loaded_ms = Math.round(performance.now() - t0);
  out.enabled = !$("exportBtn").disabled;
  await frame();

  const c = $("view");
  out.canvas = [c.width, c.height];
  out.hud = $("hudStats").textContent;
  out.file = $("hudFile").textContent;

  const probe = document.createElement("canvas");
  probe.width = 320; probe.height = 240;
  const ctx = probe.getContext("2d");
  ctx.drawImage(c, 0, 0, 320, 240);
  const d = ctx.getImageData(0, 0, 320, 240).data;
  let sum = 0, min = 255, max = 0;
  for (let i = 0; i < d.length; i += 4) {
    const l = (d[i] + d[i + 1] + d[i + 2]) / 3;
    sum += l; if (l < min) min = l; if (l > max) max = l;
  }
  out.meanLuma = +(sum / (d.length / 4)).toFixed(1);
  out.lumaRange = [min, max];
  out.vramHint = performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1e6) + " MB js heap" : "n/a";

  // depth view on the big photo
  document.querySelector('#viewTabs .chip[data-mode="2"]').click();
  await frame();
  document.querySelector('#viewTabs .chip[data-mode="0"]').click();
  await frame();
  out.modeSwitchOk = true;
  return JSON.stringify(out, null, 2);
})()
"""


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
        print(cdp.eval(SCRIPT, await_promise=True, timeout_s=900))
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
