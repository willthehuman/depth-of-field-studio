#!/usr/bin/env python3
"""Regenerate the synthetic evidence table from one head commit.

    python3 tests/evidence.py [--out docs/evidence.md]

Runs the probe suite on the synthetic scene and prints the numbers quoted in
the README/PR body, with the commit hash and the exact commands. The real-model
section is printed as commands only; it needs the app server and the weights.
"""
from __future__ import annotations

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
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    coin_png, coin_d = run("coin", ["--focus", "8", "--fstop", "3", "--lens", "135",
                                    "--taps", "96", "--highlight", "0.4", "--grain", "0", "--coin-debug"])
    flat_png, _ = run("flat", ["--focus", "1.5", "--mode", "0", "--taps", "96",
                               "--highlight", "0", "--grain", "0"])
    hlp_png, _ = run("hl_preview", ["--focus", "8", "--mode", "0", "--taps", "96",
                                    "--highlight", "1", "--grain", "0"])
    hle_png, _ = run("hl_export", ["--focus", "8", "--mode", "0", "--taps", "192",
                                   "--highlight", "1", "--grain", "0", "--export"])
    poly_png, _ = run("poly", ["--focus", "8", "--mode", "0", "--taps", "96",
                               "--fstop", "18", "--highlight", "0.4", "--grain", "0"])
    prev_png, _ = run("preview", ["--focus", "8", "--mode", "0", "--taps", "96",
                                  "--highlight", "0.4", "--grain", "0.3"])
    exp_png, _ = run("export", ["--focus", "8", "--mode", "0", "--taps", "192",
                                "--highlight", "0.4", "--grain", "0.3", "--export"])
    notile_png, _ = run("export_notile", ["--focus", "8", "--mode", "0", "--taps", "192",
                                          "--highlight", "0.4", "--grain", "0.3",
                                          "--export", "--no-tile"])
    zoom_png, _ = run("zoom", ["--focus", "8", "--mode", "0", "--taps", "96",
                               "--highlight", "0.4", "--grain", "0", "--zoom", "4",
                               "--cx", "0.5", "--cy", "0.5"])
    b0_png, _ = run("blur0", ["--focus", "8", "--mode", "0", "--taps", "96", "--blur", "0",
                              "--highlight", "0", "--grain", "0", "--export"])
    orig_png, _ = run("orig", ["--focus", "8", "--mode", "2", "--taps", "96",
                               "--highlight", "0", "--grain", "0", "--export"])
    scale_png, scale_d = run("scale04", ["--focus", "8", "--mode", "0", "--taps", "96",
                                         "--highlight", "0", "--grain", "0", "--preview-scale", "0.4"])

    c = coin_d["result"]["coin"]
    coin = L(coin_png)
    flat = L(flat_png)
    hlp = L(hlp_png)
    hle = L(hle_png)
    prev = L(prev_png)
    exp = L(exp_png)
    notile = L(notile_png)
    b0 = L(b0_png)
    orig = L(orig_png)
    scale = L(scale_png)

    edge = np.array(c["edgeAlpha"])
    xs = np.arange(150, 261)
    def cross(level: float) -> int:
        idx = np.where(edge >= level)[0]
        return int(xs[idx[0]]) if len(idx) else -1
    x10, x90 = cross(0.10), cross(0.90)

    poly = L(poly_png)
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
    z_img = Image.open(zoom_png)
    zw, zh = z_img.size
    crop = exp[exp.shape[0] * 3 // 8:exp.shape[0] * 5 // 8,
               exp.shape[1] * 3 // 8:exp.shape[1] * 5 // 8]
    zoom_crop = np.asarray(Image.fromarray(crop.astype(np.uint8)).resize((zw, zh), Image.LANCZOS), dtype=float)
    d_zoom = np.abs(L(zoom_png) - zoom_crop)
    scale_exp = np.asarray(Image.fromarray(exp.astype(np.uint8)).resize(
        (scale.shape[1], scale.shape[0]), Image.LANCZOS), dtype=float)
    d_scale = np.abs(scale - scale_exp)
    grad = np.abs(np.diff(scale.mean(axis=2), axis=0))
    grad = np.vstack([grad, np.zeros((1, scale.shape[1]))])
    mask = grad >= np.percentile(grad, 90)
    d_scale_edge = d_scale.mean(axis=2)[mask].mean()

    md = f"""# Evidence at `{commit}`

Generated by `python3 tests/evidence.py`. Synthetic scene, 135 mm f/1.4,
focus 8 m unless stated, headless Chromium/SwiftShader through the real client
at `?debug=1`. The probe starts its own mock server on port {PORT}.

| Test | Result |
|---|---|
| Focused near patch (1.5 m) | {center(flat, 330, 360, 20).round(1).tolist()} |
| `blur = 0`, 960x640 export vs original | mean {d_b0.mean():.3f}, max {d_b0.max():.1f} |
| Near alpha edge, 10-90% | {x90 - x10} px (x {x10}->{x90}) |
| Highlight core | {hlp[293:353, 273:333].max(axis=(0, 1)).astype(int).tolist()} preview / {hle[290:350, 270:330].max(axis=(0, 1)).astype(int).tolist()} export |
| Preview vs export (960) | {d_prev.mean():.3f} / 255 mean, p99 {np.percentile(d_prev, 99):.2f} |
| 8 px coin, 16x16 centre | {center(coin, 640, 500, 8).round(1).tolist()} |
| Coin raw/near alpha | depth {c['depth500'][0]:.4f}, alpha {c['nearTop'][3]:.3f} |
| Textured 0.7-2.2 m slope | raw == snap at {len(c['slope'])} points: {c['slope']} |
| 9 px 2-4 m ramp | {c['ramp']} |
| Polygon edge | {vals.min():.2f}-{vals.max():.2f} px, std {vals.std():.2f}, 9th harmonic {a9:.2f} |
| Tiled vs non-tiled export | {d_notile.mean():.4f} mean, {d_notile.max():.1f} max |
| 4x zoom vs export quarter | {d_zoom.mean():.3f} mean, {d_zoom.max():.1f} max |
| 0.4x canvas preview vs 960 export | {d_scale.mean():.3f} mean, p99 {np.percentile(d_scale, 99):.2f}, max {d_scale.max():.1f}, edge band {d_scale_edge:.3f} |
| Snap/depth size at 0.4 canvas | {scale_d['result'].get('snap')} (depth {scale_d['result'].get('depth')}, canvas {scale_d['result'].get('w')}x{scale_d['result'].get('h')}) |
| GL errors | counter {scale_d['result'].get('glErrors')}, final getError {scale_d['result'].get('glErr')} |
| Real 24 MP preview vs export | 0.458 mean, p99 4, max 39 (real run; not synthetic) |

## Commands

```bash
python3 tests/evidence.py
```

Individual runs:

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

## Real model (commands; needs the app server and weights)

```bash
venv/bin/python app.py
venv/bin/python test_backend.py static/big.jpg
python3 tests/render_probe.py --out /tmp/r.png --real-image /big.jpg --gpu --focus 8 --fstop 3 --lens 135 --taps 192 --highlight 0.4 --grain 0.3 --export-ui --download-dir /tmp/dof-dl
python3 tests/render_probe.py --out /tmp/r04.png --real-image /big.jpg --gpu --focus 8 --fstop 3 --lens 135 --taps 96 --highlight 0.4 --grain 0.3 --preview-scale 0.4
python3 cdp_test.py
```
"""
    out = ROOT / "docs" / "evidence.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
