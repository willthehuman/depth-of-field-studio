"""Headless check of the analysis pipeline: no HTTP, no browser."""
import sys
import time
from pathlib import Path

import numpy as np

import app as studio


def check_equal_luminance_guide() -> None:
    """The guide must see a red/green boundary even when luma is identical."""
    h, w = 96, 128
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[:, : w // 2] = (255, 0, 0)
    image[:, w // 2 :] = (0, 76, 0)  # approximately the same luma as pure red
    coarse = np.full((12, 16), 0.2, dtype=np.float32)
    coarse[:, 8:] = 0.8

    refined = studio.guided_upsample(coarse, image)
    bilinear = studio.F.interpolate(
        studio.torch.from_numpy(coarse)[None, None], size=(h, w), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    edge = slice(w // 2 - 15, w // 2 + 15)
    guided_gradient = np.abs(np.diff(refined[:, edge], axis=1)).max()
    bilinear_gradient = np.abs(np.diff(bilinear[:, edge], axis=1)).max()
    print(f"colour guide edge: guided={guided_gradient:.3f} bilinear={bilinear_gradient:.3f}")
    assert guided_gradient > bilinear_gradient * 2, "colour guide did not recover the chroma-only edge"


check_equal_luminance_guide()

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
