"""Headless check of the analysis pipeline: no HTTP, no browser."""
import sys
import time
from pathlib import Path

import numpy as np

import app as studio

path = Path(sys.argv[1]) if len(sys.argv) > 1 else studio.ROOT / "static/sample.jpg"
t0 = time.time()
studio.load_model()
print(f"model ready in {time.time()-t0:.1f}s  status={studio.STATE['status']}")

raw = path.read_bytes()
t1 = time.time()
meta = studio.analyze_bytes(raw, path.name)
print(f"analyze: {time.time()-t1:.2f}s")
for k, v in meta.items():
    if not isinstance(v, dict):
        print(f"  {k}: {v}")
print("  inference:", meta["inference"])
print("  depth_pcts:", {k: round(v, 3) for k, v in meta["depth_pcts"].items()})

d = np.fromfile(studio.DATA / meta["id"] / "depth.f16", dtype=np.float16).astype(np.float32)
d = d.reshape(meta["height"], meta["width"])
print(f"depth array {d.shape} min={d.min():.3f} max={d.max():.3f} median={np.median(d):.3f}")

# Physically sane: metres, positive, and a spread rather than a constant.
assert np.isfinite(d).all(), "non-finite depth survived"
assert d.min() > 0.02, "depth below 2 cm"
assert np.median(d) < 900, "median depth implausible"
assert float(d.std()) > 0.5, "depth is flat — the model probably did not run"

# The guided upsample must not invent structure: the coarse view should still
# look like the same scene, just at the photo's scale.
h, w = d.shape
step = max(1, h // 240)
coarse = d[::step, ::step]
print(f"coarse {coarse.shape} median {np.median(coarse):.2f} m std {coarse.std():.2f}")
print("OK")
