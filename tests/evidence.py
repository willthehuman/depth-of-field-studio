#!/usr/bin/env python3
"""Regenerate the synthetic evidence table from the current working tree.

    python3 tests/evidence.py

Runs the probe suite on the synthetic scene and writes docs/evidence.md. The
table records SHA-256 hashes of the code that produced it, not a commit hash:
the evidence file is committed in a docs-only commit after the code, and a
docs-only commit leaves these hashes unchanged.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PORT = 7861
TMP = Path.home() / ".cache" / "dof-evidence"
TMP.mkdir(parents=True, exist_ok=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def run(name: str, args: list[str]) -> tuple[Path, dict]:
    out = TMP / f"{name}.png"
    cmd = [sys.executable, str(ROOT / "tests" / "render_probe.py"),
           "--out", str(out), "--port", str(PORT), *args]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise SystemExit(f"probe {name} failed")
    i = p.stdout.find('{\n  "out"')
    if i < 0:
        i = p.stdout.find('{"out"')
    if i < 0:
        raise SystemExit(f"probe {name}: no JSON in stdout: {p.stdout[-1000:]}")
    return out, json.loads(p.stdout[i:])


def L(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"), dtype=float)


def center(a: np.ndarray, x: int, y: int, r: int) -> np.ndarray:
    return a[y - r:y + r, x - r:x + r].mean(axis=(0, 1))


def main() -> None:
    runs: dict[str, tuple[Path, dict]] = {}

    def r(name, args):
        runs[name] = run(name, args)

    r("coin", ["--focus", "8", "--fstop", "3", "--lens", "135", "--taps", "96",
               "--highlight", "0.4", "--grain", "0", "--coin-debug"])
    r("flat", ["--focus", "1.5", "--mode", "0", "--taps", "96",
               "--highlight", "0", "--grain", "0"])
    r("hl_preview", ["--focus", "8", "--mode", "0", "--taps", "96",
                     "--highlight", "1", "--grain", "0"])
    r("hl_export", ["--focus", "8", "--mode", "0", "--taps", "192",
                    "--highlight", "1", "--grain", "0", "--export"])
    r("poly", ["--focus", "8", "--mode", "0", "--taps", "96", "--fstop", "18",
               "--highlight", "0.4", "--grain", "0"])
    r("preview", ["--focus", "8", "--mode", "0", "--taps", "96",
                  "--highlight", "0.4", "--grain", "0.3"])
    r("export", ["--focus", "8", "--mode", "0", "--taps", "192",
                 "--highlight", "0.4", "--grain", "0.3", "--export"])
    r("export_notile", ["--focus", "8", "--mode", "0", "--taps", "192",
                        "--highlight", "0.4", "--grain", "0.3",
                        "--export", "--no-tile"])
    r("zoom", ["--focus", "8", "--mode", "0", "--taps", "96",
               "--highlight", "0.4", "--grain", "0", "--zoom", "4",
               "--cx", "0.5", "--cy", "0.5"])
    r("blur0", ["--focus", "8", "--mode", "0", "--taps", "96", "--blur", "0",
                "--highlight", "0", "--grain", "0", "--export"])
    r("orig", ["--focus", "8", "--mode", "2", "--taps", "96",
               "--highlight", "0", "--grain", "0", "--export"])
    r("scale04", ["--focus", "8", "--mode", "0", "--taps", "96",
                  "--highlight", "0", "--grain", "0", "--preview-scale", "0.4"])
    r("export_ui", ["--focus", "8", "--mode", "0", "--taps", "192",
                    "--highlight", "0.4", "--grain", "0", "--export",
                    "--export-ui", "--download-dir", str(TMP / "dl")])

    coin_png, coin_d = runs["coin"]
    flat = L(runs["flat"][0])
    hlp = L(runs["hl_preview"][0])
    hle = L(runs["hl_export"][0])
    prev = L(runs["preview"][0])
    exp = L(runs["export"][0])
    notile = L(runs["export_notile"][0])
    b0 = L(runs["blur0"][0])
    orig = L(runs["orig"][0])
    scale = L(runs["scale04"][0])
    zoom = L(runs["zoom"][0])
    poly = L(runs["poly"][0])
    c = coin_d["result"]["coin"]

    edge = np.array(c["edgeAlpha"])
    xs = np.arange(150, 261)

    def cross(level: float) -> int:
        idx = np.where(edge >= level)[0]
        return int(xs[idx[0]]) if len(idx) else -1

    x10, x90 = cross(0.10), cross(0.90)

    lum = poly @ np.array([0.2126, 0.7152, 0.0722])
    reg = lum[263:383, 243:363]
    thr = 0.55 * reg.max()
    vals = []
    for k in range(72):
        ang = 2 * np.pi * k / 72
        rs = np.arange(1, 60, 0.5)
        xx = np.clip((303 + rs * np.cos(ang)).astype(int), 0, poly.shape[1] - 1)
        yy = np.clip((323 + rs * np.sin(ang)).astype(int), 0, poly.shape[0] - 1)
        hit = rs[lum[yy, xx] > thr]
        vals.append(hit.max() if len(hit) else 0)
    vals = np.array(vals)
    th = np.arange(72) * 2 * np.pi / 72
    a9 = abs(np.sum((vals - vals.mean()) * np.exp(-1j * 9 * th))) * 2 / 72

    prev960 = np.asarray(Image.fromarray(prev.astype(np.uint8)).resize((960, 640), Image.LANCZOS), dtype=float)
    d_prev = np.abs(prev960 - exp)
    d_notile = np.abs(exp - notile)
    d_b0 = np.abs(b0 - orig)
    z_img = Image.open(runs["zoom"][0])
    zw, zh = z_img.size
    crop = exp[exp.shape[0] * 3 // 8:exp.shape[0] * 5 // 8,
               exp.shape[1] * 3 // 8:exp.shape[1] * 5 // 8]
    zoom_crop = np.asarray(Image.fromarray(crop.astype(np.uint8)).resize((zw, zh), Image.LANCZOS), dtype=float)
    d_zoom = np.abs(zoom - zoom_crop)
    scale_exp = np.asarray(Image.fromarray(exp.astype(np.uint8)).resize(
        (scale.shape[1], scale.shape[0]), Image.LANCZOS), dtype=float)
    d_scale = np.abs(scale - scale_exp)
    grad = np.abs(np.diff(scale.mean(axis=2), axis=0))
    grad = np.vstack([grad, np.zeros((1, scale.shape[1]))])
    mask = grad >= np.percentile(grad, 90)
    d_scale_edge = d_scale.mean(axis=2)[mask].mean()

    near = c["nearAlpha"]
    near_rows = [a for x, y, a in near if y <= 209]
    near_rect = [a for x, y, a in near if y == 300 and x >= 620]
    gl_errors = max(int(d["result"].get("glErrors") or 0) for _, d in runs.values())

    md = f"""# Evidence (synthetic)

Generated by `python3 tests/evidence.py` from the working tree. Code hashes
(SHA-256, first 16): `static/app.js` `{sha(ROOT / 'static/app.js')}`,
`tests/mock_depth_server.py` `{sha(ROOT / 'tests/mock_depth_server.py')}`,
`tests/render_probe.py` `{sha(ROOT / 'tests/render_probe.py')}`,
`tests/evidence.py` `{sha(ROOT / 'tests/evidence.py')}`. Docs-only commits do
not change these hashes; re-run the script after any code change.

Synthetic scene, 135 mm f/1.4, focus 8 m unless stated, headless
Chromium/SwiftShader through the real client at `?debug=1`. The probe starts its
own mock server on port {PORT}.

| Test | Result |
|---|---|
| Focused near patch (1.5 m) | {center(flat, 330, 360, 20).round(1).tolist()} |
| Original flat patch (960 export, mode 2) | {center(orig, 330, 360, 20).round(1).tolist()} |
| `blur = 0`, 960x640 export vs original | mean {d_b0.mean():.3f}, max {d_b0.max():.1f} |
| Near alpha edge, 10-90% | {x90 - x10} px (x {x10}->{x90}) |
| Near alpha rows 0-209 | {min(near_rows):.3f} to {max(near_rows):.3f} |
| Near alpha 4 m rectangle | {min(near_rect):.3f} to {max(near_rect):.3f} |
| Highlight core | {hlp[293:353, 273:333].max(axis=(0, 1)).astype(int).tolist()} preview / {hle[290:350, 270:330].max(axis=(0, 1)).astype(int).tolist()} export |
| Preview vs export (960) | {d_prev.mean():.3f} / 255 mean, p99 {np.percentile(d_prev, 99):.2f} |
| 8 px coin, 16x16 centre | {center(L(coin_png), 640, 500, 8).round(1).tolist()} |
| Coin at (640, 500): near / far | near {[round(v, 4) for v in c['coinPixel']]}, far {[round(v, 4) for v in c['coinFar']]} |
| Coin snap depth | {c['depth500'][0]:.4f} |
| Textured 0.7-2.2 m slope | raw == snap at {len(c['slope'])} points; snap != raw {c['slopeDiff'][0]}/{c['slopeDiff'][1]} |
| 9 px 2-4 m ramp | snap != raw {c['rampDiff'][0]}/{c['rampDiff'][1]}; samples {c['ramp']} |
| Polygon edge | {vals.min():.2f}-{vals.max():.2f} px, std {vals.std():.2f}, 9th harmonic {a9:.2f} |
| Tiled vs non-tiled export | {d_notile.mean():.4f} mean, {d_notile.max():.1f} max |
| 4x zoom vs export quarter | {d_zoom.mean():.3f} mean, {d_zoom.max():.1f} max |
| 0.4x canvas preview vs 960 export | {d_scale.mean():.3f} mean, p99 {np.percentile(d_scale, 99):.2f}, max {d_scale.max():.1f}, edge band {d_scale_edge:.3f} |
| Snap/depth size at 0.4 canvas | {runs['scale04'][1]['result'].get('snap')} (depth {runs['scale04'][1]['result'].get('depth')}, canvas {runs['scale04'][1]['result'].get('w')}x{runs['scale04'][1]['result'].get('h')}) |
| Export restore state | {runs['export_ui'][1]['result']} |
| GL errors (max over suite) | {gl_errors}; final getError {runs['scale04'][1]['result'].get('glErr')} |

## Commands

```bash
python3 tests/evidence.py
```

The probe starts its own mock on port {PORT}; no app server is needed. Individual
runs:

```bash
python3 tests/render_probe.py --out /tmp/c.png --focus 8 --fstop 3 --lens 135 --taps 96 --highlight 0.4 --grain 0 --coin-debug
python3 tests/render_probe.py --out /tmp/f.png --focus 1.5 --taps 96 --highlight 0 --grain 0
python3 tests/render_probe.py --out /tmp/e.png --focus 8 --taps 192 --highlight 0.4 --grain 0.3 --export
python3 tests/render_probe.py --out /tmp/x.png --focus 8 --taps 192 --highlight 0.4 --grain 0.3 --export --no-tile
python3 tests/render_probe.py --out /tmp/z.png --focus 8 --taps 96 --highlight 0.4 --grain 0 --zoom 4 --cx 0.5 --cy 0.5
python3 tests/render_probe.py --out /tmp/b.png --focus 8 --taps 96 --blur 0 --highlight 0 --grain 0 --export
python3 tests/render_probe.py --out /tmp/o.png --focus 8 --mode 2 --taps 96 --highlight 0 --grain 0 --export
python3 tests/render_probe.py --out /tmp/s.png --focus 8 --taps 96 --highlight 0 --grain 0 --preview-scale 0.4
```

## Manual real-model run (not regenerated by the script)

Run 2026-11-27 on an RTX 4070 SUPER (12 GB), real Depth Pro 2 H+ weights, real
5656x4242 image (`static/big.jpg`). These are hand-recorded because they need the
app server and weights; they are not part of the table above.

```bash
venv/bin/python app.py
venv/bin/python test_backend.py static/big.jpg
python3 tests/render_probe.py --out /tmp/r.png --real-image /big.jpg --gpu --focus 8 --fstop 3 --lens 135 --taps 192 --highlight 0.4 --grain 0.3 --export-ui --download-dir /tmp/dof-dl
python3 tests/render_probe.py --out /tmp/r04.png --real-image /big.jpg --gpu --focus 8 --fstop 3 --lens 135 --taps 96 --highlight 0.4 --grain 0.3 --preview-scale 0.4
python3 cdp_test.py
```

Results: cold `test_backend.py` analysis 3.11 s (the app reports 1.3 s for the
cached job); inference 0.76 s, refine 0.14 s; reported peak 3.96 GB before
`guided_upsample`; export 5656x4242 PNG 21.9 MB, 4.0 s wall with a cached
analysis; tiled vs non-tiled 0.000/0.0; `nvidia-smi` sampled 2.8 GiB total
during the export, while texture accounting plus the resident model is ~3.7 GB;
0.4x preview vs export mean 0.458/255, p99 4, max 39; browser suite `ok: true`
on SwiftShader WebGL; zoom drift 3.7 code sum on a 5x5 patch.
"""
    out = ROOT / "docs" / "evidence.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
