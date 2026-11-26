"""Find the largest inference size that fits comfortably beside a desktop."""
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_HOME", str(ROOT / "torch_cache"))

import numpy as np
import torch
from PIL import Image
from depth_pro_2.model import create_model_and_transforms, to_normalized_tensor
from depth_pro_2.utils import load_rgb

t0 = time.time()
model, _ = create_model_and_transforms("depthpro2-h", device="cpu", precision=torch.bfloat16)
model.encoder = model.encoder.to(torch.bfloat16)
model.decoder = model.decoder.to(torch.bfloat16)
model = model.to("cuda").eval()
print(f"model on gpu in {time.time()-t0:.1f}s, weights {torch.cuda.memory_allocated()/1e9:.2f} GB")

transform = lambda a: to_normalized_tensor(a, torch.device("cuda"))
img, f_px = load_rgb(ROOT / "static/sample.jpg")

for side in (1024, 1280, 1536, 1728):
    h, w = img.shape[:2]
    s = side / max(h, w)
    iw = max(32, int(round(w * s / 32)) * 32)
    ih = max(32, int(round(h * s / 32)) * 32)
    small = np.ascontiguousarray(np.asarray(Image.fromarray(img).resize((iw, ih), Image.LANCZOS)))
    x = transform(small)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    try:
        with torch.inference_mode():
            pred = model.infer(x, f_px=None if f_px is None else f_px * iw / w)
        torch.cuda.synchronize()
        dt = time.time() - t
        print(
            f"  {side:5d}px -> {iw}x{ih}  {dt:5.2f}s  peak {torch.cuda.max_memory_allocated()/1e9:5.2f} GB"
        )
    except torch.cuda.OutOfMemoryError as exc:
        print(f"  {side:5d}px -> {iw}x{ih}  OOM: {str(exc)[:80]}")
        torch.cuda.empty_cache()
    del x, pred
    torch.cuda.empty_cache()
