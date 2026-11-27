#!/usr/bin/env python3
"""Synthetic depth server for renderer QA.

Serves the real `static/` client, but answers /api/* with a generated scene so
the WebGL pipeline can be exercised without Depth Pro 2, torch, or a GPU.
The scene is intentionally simple and known:

  - background plane: textured, depth 6 m at the top to 12 m at the bottom
  - midground rectangle: flat 4 m
  - near disc: flat 1.5 m, with a fine pattern and one clipped specular dot

Launch:  python3 tests/mock_depth_server.py --port 7860
"""
from __future__ import annotations

import argparse
import io
import struct
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
W, H = 960, 640


def build_scene() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:H, 0:W]
    z_bg = 6.0 + 6.0 * (yy / (H - 1))

    near = (xx - 330) ** 2 + (yy - 360) ** 2 < 130 ** 2
    mid = (xx > 600) & (xx < 860) & (yy > 180) & (yy < 460)

    base = 0.32 + 0.10 * np.sin(xx / 23.0) + 0.07 * np.cos(yy / 31.0)
    img = np.stack([base * 0.82, base * 0.92, base * 1.10], axis=-1)
    img[mid] = np.array([0.22, 0.48, 0.30])
    tex = 0.14 * np.sin(xx / 7.0) + 0.14 * np.cos(yy / 9.0)
    img[near] = np.stack([0.82 + tex, 0.24 + 0.5 * tex, 0.16 + 0.25 * tex], axis=-1)[near]
    spec = (xx - 300) ** 2 + (yy - 320) ** 2 < 5 ** 2
    img[spec] = [1.0, 1.0, 1.0]
    img += rng.normal(0.0, 0.018, img.shape)
    img = np.clip(img, 0.0, 1.0)

    z = z_bg.copy()
    z[mid] = 4.0
    z[near] = 1.5
    inv = (1.0 / z).astype(np.float32)
    return (img * 255.0 + 0.5).astype(np.uint8), inv


PHOTO, INV = build_scene()
PHOTO_PNG = io.BytesIO()
Image.fromarray(PHOTO).save(PHOTO_PNG, format="PNG")
PHOTO_PNG = PHOTO_PNG.getvalue()
PHOTO_JPG = io.BytesIO()
Image.fromarray(PHOTO).save(PHOTO_JPG, format="JPEG", quality=94)
PHOTO_JPG = PHOTO_JPG.getvalue()
DEPTH_F16 = np.minimum(INV, 1e4).astype(np.float16).tobytes()

# UI statistics are in metres, as app.py produces.
DEPTH_M = 1.0 / np.maximum(INV, 1e-6)
sample = DEPTH_M[:: max(1, H // 400), :: max(1, W // 400)]
pct = np.percentile(sample, [0.5, 5, 50, 95, 99.5])
F_PX = 50.0 * W / 36.0

META = {
    "id": "mock",
    "filename": "synthetic.png",
    "width": W,
    "height": H,
    "f_px": F_PX,
    "f_px_source": "exif",
    "focal35": 50.0,
    "depth_encoding": "disparity_m",
    "depth_pcts": {
        "p0_5": float(pct[0]),
        "p5": float(pct[1]),
        "p50": float(pct[2]),
        "p95": float(pct[3]),
        "p99_5": float(pct[4]),
    },
    "inference": {"width": W, "height": H, "side": W, "seconds": 0.12,
                  "refine_seconds": 0.01, "peak_vram_gb": 0.0},
    "model": "synthetic",
    "image_url": "/api/jobs/mock/image",
    "depth_url": "/api/jobs/mock/depth",
    "total_seconds": 0.13,
}

app = FastAPI()


@app.post("/api/warmup")
def warmup():
    return {"status": "ready"}


@app.get("/api/status")
def status():
    return {"status": "ready", "device": "cuda", "gpu": "synthetic", "model": "synthetic",
            "load_seconds": 0.0, "side": W}


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...)):
    file.file.read()
    return META


@app.get("/api/jobs/mock/image")
def image():
    return Response(PHOTO_PNG, media_type="image/png")


@app.get("/api/jobs/mock/depth")
def depth():
    return Response(DEPTH_F16, media_type="application/octet-stream")


@app.get("/sample.jpg")
def sample_jpg():
    return Response(PHOTO_JPG, media_type="image/jpeg")


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
