"""Grab a couple of full-canvas frames from the running app for a visual check."""
from __future__ import annotations

import base64
import json
import urllib.request
from pathlib import Path

from cdp_test import PORT, CDP, launch

ROOT = Path(__file__).resolve().parent

SCRIPT = r"""
(async () => {
  const $ = (id) => document.getElementById(id);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const frame = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  const out = {};

  $("sampleBtn").click();
  for (let i = 0; i < 900; i++) { if (!$("exportBtn").disabled) break; await sleep(200); }
  await frame();

  const src = $("view");
  const W = src.width, H = src.height;

  // focus on the nearest thing in the lower half, like a person would
  const depth = (() => {
    const c = document.createElement("canvas");
    c.width = W; c.height = H;
    c.getContext("2d").drawImage(src, 0, 0);
    return c.getContext("2d").getImageData(0, 0, W, H).data;
  })();

  // click a point 18% in from the left, 72% down: the fountain
  const rect = src.getBoundingClientRect();
  const cx = 0.17, cy = 0.72;
  src.dispatchEvent(new MouseEvent("click", {
    bubbles: true,
    clientX: rect.left + cx * rect.width,
    clientY: rect.top + cy * rect.height,
  }));
  await frame();

  const setF = (i) => { const el = $("fstop"); el.value = String(i); el.dispatchEvent(new Event("input", {bubbles:true})); };
  const setLens = (mm) => { document.querySelector(`#lens button[data-mm="${mm}"]`).click(); };

  setF(9);  // f/2.8
  await frame();
  out.focus = $("focusOut").textContent;
  out.aperture = $("fstopOut").textContent;
  out.native = src.toDataURL("image/jpeg", 0.93);

  setLens(85);
  await frame();
  out.lens85 = $("lensOut").textContent;
  out.shallow = src.toDataURL("image/jpeg", 0.93);

  setF(24); // f/16
  await frame();
  out.deep = src.toDataURL("image/jpeg", 0.93);

  document.querySelector('#viewTabs .chip[data-mode="1"]').click();
  await frame();
  out.depthmap = src.toDataURL("image/png");

  setF(9);
  document.querySelector('#viewTabs .chip[data-mode="0"]').click();
  await frame();
  return JSON.stringify(out);
})()
"""


def main() -> None:
    proc = launch()
    try:
        targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list").read())
        page = next(t for t in targets if t["type"] == "page")
        cdp = CDP(page["webSocketDebuggerUrl"])
        cdp.send("Runtime.enable")
        payload = json.loads(cdp.eval(SCRIPT, await_promise=True, timeout_s=420))
        for key in ("shallow", "deep", "depthmap", "native"):
            data = payload.pop(key)
            head, _, b64 = data.partition(",")
            ext = "png" if "png" in head else "jpg"
            path = ROOT / f"docs_preview_{key}.{ext}"
            path.write_bytes(base64.b64decode(b64))
            payload[f"{key}_file"] = str(path)
            print(f"{key}: {path} ({path.stat().st_size/1024:.0f} KB)")
        print(json.dumps(payload, indent=2))
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
