#!/usr/bin/env python3
"""Depth of Field Studio.

A small local web app that relights the geometry of a photo: it estimates metric
depth with Apple's Depth Pro 2 (H+ preset) and applies a physically-based
circle-of-confusion so you can pick the focus plane and the aperture afterwards.

    py -3.13 app.py            ->  http://127.0.0.1:7860

The depth estimate happens once per photo on the GPU; everything that follows
(focus point, aperture, export) is a WebGL shader in the browser, so the sliders
stay interactive even on a 24 MP frame.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
import traceback
import uuid
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_HOME", str(ROOT / "torch_cache"))

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from depth_pro_2.model import create_model_and_transforms, to_normalized_tensor
from depth_pro_2.utils import load_rgb

#: Longest side fed to the network. H+ peaks at 4.0 GB of the 6 GB available when a
#: desktop is also using the card; 1536 buys a little edge detail for 1 GB more.
DEFAULT_SIDE = int(os.environ.get("DEPTH_SIDE", "1280"))
FALLBACK_SIDES = (1024, 768)
PRESET = os.environ.get("DEPTH_MODEL", "depthpro2-h")
MODEL_LABEL = "Depth Pro 2 · H+"

DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

STATE = {
    "status": "idle",
    "message": "Model not loaded yet",
    "device": None,
    "model": None,
    "transform": None,
    "load_seconds": None,
    "error": None,
}
MODEL_LOCK = threading.RLock()
GLOBAL_LOCK = threading.Lock()
CACHE: dict[str, dict] = {}


# --------------------------------------------------------------------------- model


def load_model() -> None:
    """Build H+ on the CPU, cast the backbones to bf16, then move it to the GPU.

    The released weights are float32 (3.5 GB); building straight on a 6 GB card
    leaves no room for activations. The encoder/decoder only ever run under
    bf16 autocast anyway, so holding them as bf16 is the same arithmetic with
    half the footprint. The depth head and the focal estimator stay float32,
    which is how the package runs them.
    """
    with MODEL_LOCK:
        if STATE["model"] is not None:
            return
        STATE.update(status="loading", message="Downloading / loading Depth Pro 2 H+ …")
        t0 = time.time()
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _ = create_model_and_transforms(PRESET, device="cpu", precision=torch.bfloat16)
            if device == "cuda":
                model.encoder = model.encoder.to(torch.bfloat16)
                model.decoder = model.decoder.to(torch.bfloat16)
                model = model.to("cuda")
                torch.cuda.empty_cache()
            else:
                model.precision = torch.float32
            model.eval()
            transform = partial(to_normalized_tensor, device=torch.device(device))

            # Touch every path once so the first real photo does not pay for it.
            with torch.inference_mode():
                probe = np.zeros((256, 256, 3), dtype=np.uint8)
                model.infer(transform(probe), f_px=None)
            if device == "cuda":
                torch.cuda.empty_cache()

            STATE.update(
                status="ready",
                message=f"{MODEL_LABEL} ready on {device}",
                device=device,
                model=model,
                transform=transform,
                load_seconds=round(time.time() - t0, 1),
                error=None,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            STATE.update(status="idle", message=str(exc), error=traceback.format_exc())
            print(traceback.format_exc())


# --------------------------------------------------------------------- depth post


def _box(x: torch.Tensor, r: int) -> torch.Tensor:
    """Mean over a (2r+1)^2 window, separable, replicate padding."""
    if r <= 0:
        return x
    k = 2 * r + 1
    channels = x.shape[1]
    x = F.pad(x, (r, r, 0, 0), mode="replicate")
    kernel_x = torch.full((channels, 1, 1, k), 1.0 / k, dtype=x.dtype, device=x.device)
    x = F.conv2d(x, kernel_x, groups=channels)
    x = F.pad(x, (0, 0, r, r), mode="replicate")
    kernel_y = torch.full((channels, 1, k, 1), 1.0 / k, dtype=x.dtype, device=x.device)
    x = F.conv2d(x, kernel_y, groups=channels)
    return x


def guided_upsample(inverse_depth: np.ndarray, image: np.ndarray,
                    radius: int = 8, device: str = "cpu") -> np.ndarray:
    """Upsample inverse depth to the photo's resolution, snapped to its edges.

    A plain bilinear upsample of a 1536-wide depth map melts every silhouette
    over ~4 px of the full frame, which shows up as a soft in-focus rim next to a
    heavily blurred background. This is the guided filter from He et al., applied
    to each colour channel and fused by local contrast. A luminance-only guide
    misses edges between colours of equal luminance (for example red against
    green); the colour channels keep those boundaries available to the depth map.
    """
    h, w = image.shape[:2]
    dev = torch.device(device)
    p = torch.from_numpy(np.ascontiguousarray(inverse_depth, dtype=np.float32))[None, None].to(dev)
    if p.shape[-2:] != (h, w):
        p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)
    guide = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
    guide = guide.to(dev).to(torch.float32).div_(255.0)

    eps = 2e-4
    mean_p = _box(p, radius)
    merged = torch.zeros_like(p)
    total_weight = torch.zeros_like(p)

    # Fuse the channels by local structural contrast. A channel with a real edge
    # gets more say than one that is flat across the same window.
    for channel in range(3):
        guide_c = guide[:, channel : channel + 1]
        mean_i = _box(guide_c, radius)
        corr_i = _box(guide_c * guide_c, radius)
        corr_ip = _box(guide_c * p, radius)
        var_i = corr_i - mean_i * mean_i
        cov_ip = corr_ip - mean_i * mean_p
        a = cov_ip / (var_i + eps)
        b = mean_p - a * mean_i
        q = _box(a, radius) * guide_c + _box(b, radius)
        channel_weight = _box(var_i, radius) + eps
        merged.add_(q * channel_weight)
        total_weight.add_(channel_weight)

    return (merged / total_weight)[0, 0].float().cpu().numpy()


def upsample_radius(h: int, w: int, ih: int, iw: int) -> int:
    """Guided-filter radius, in full-res pixels, from the upsample factor.

    The network's own depth edges are already sharp (a plain bilinear upsample
    keeps them at ~2 px); the guided filter is only there to joint the depth to
    the photo's edges. Every extra pixel of radius widens each depth edge by
    ~2 px, which is what read as a halo in the depth map. Keep the window just
    above one coarse pixel: factor 1.5 -> r1, a 27 MP upsample (factor ~4.7)
    -> r4.
    """
    factor = max(h / max(ih, 1), w / max(iw, 1))
    return int(min(8, max(1, round(0.75 * factor))))


#: Full-frame sensor diagonal in mm (36 x 24). Using the diagonal makes the
#: 35mm-equivalent invariant to portrait/landscape rotation and aspect ratio.
FF_DIAGONAL_MM = float(np.hypot(36.0, 24.0))  # 43.2666...


def focal35_from_fpx(f_px: float, width: int, height: int) -> float:
    """35mm-equivalent focal length from focal-in-pixels, using the diagonal."""
    return float(f_px * FF_DIAGONAL_MM / np.hypot(width, height))


def disparity_to_metres(inv: np.ndarray, cap: float = 2000.0) -> np.ndarray:
    """Convert inverse depth (1/m) to metres for UI/stats; sky (0) -> cap."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = 1.0 / np.maximum(inv, 1e-6)
    return np.minimum(z, cap).astype(np.float32)


def true_disparity(normalized: np.ndarray, width: int, f_px: float) -> np.ndarray:
    """Convert the network's focal/width-normalized inverse depth to true 1/z (1/m).

    The network predicts inverse depth scaled by f_px/width; the shaders (and
    the physics) need metres^-1, where CoC is exactly linear in |1/z - 1/s|.
    """
    return np.maximum(normalized, 0.0).astype(np.float32) * (width / max(f_px, 1e-6))


# ------------------------------------------------------------------------ inference


def _infer_at(model, transform, image: np.ndarray, f_px: float | None, side: int):
    h, w = image.shape[:2]
    scale = min(1.0, side / max(h, w))
    iw = max(32, int(round(w * scale / 32)) * 32)
    ih = max(32, int(round(h * scale / 32)) * 32)
    small = np.asarray(Image.fromarray(image).resize((iw, ih), Image.LANCZOS))
    x = transform(small)
    with torch.inference_mode():
        pred = model.infer(x, f_px=None if f_px is None else f_px * iw / w)
    depth = pred["depth"].float().cpu().numpy()
    f_px_small = float(pred["focallength_px"])
    return depth, f_px_small, (iw, ih)


def estimate(image_path: Path, side: int) -> dict:
    """Run depth estimation on a photo, returning depth + metadata at full res."""
    model, transform = STATE["model"], STATE["transform"]
    image, f_px = load_rgb(image_path)  # upright, EXIF focal length in px if present
    h, w = image.shape[:2]

    tries = [side] + [s for s in FALLBACK_SIDES if s < side]
    last_error = None
    for attempt in tries:
        try:
            t0 = time.time()
            depth_small, f_px_small, (iw, ih) = _infer_at(model, transform, image, f_px, attempt)
            seconds = time.time() - t0
            break
        except torch.cuda.OutOfMemoryError as exc:
            last_error = exc
            torch.cuda.empty_cache()
            print(f"[depth] {attempt}px did not fit, retrying smaller: {exc}")
    else:
        raise RuntimeError(f"Depth inference failed at every resolution: {last_error}")

    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    f_px_full = f_px_small * w / iw
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_small = np.nan_to_num(f_px_small / (iw * depth_small), nan=0.0, posinf=0.0)

    t1 = time.time()
    radius = upsample_radius(h, w, ih, iw)
    try:
        inv_full = guided_upsample(inv_small, image, radius=radius,
                                   device=STATE["device"] or "cpu")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("[depth] guided upsample did not fit on GPU, falling back to CPU")
        inv_full = guided_upsample(inv_small, image, radius=radius, device="cpu")
    refine_seconds = time.time() - t1
    # The network's inverse depth is normalized by focal length and width
    # (inv = f_px/width * 1/z). Divide that scale out so the texture and the
    # shaders hold true inverse depth in 1/m, where CoC is exactly linear.
    inv_full = true_disparity(inv_full, iw, f_px_small)

    # Metres are only for the UI range/percentiles; the stored texture is 1/z.
    depth_full = disparity_to_metres(inv_full)

    sample = depth_full[:: max(1, h // 400), :: max(1, w // 400)]
    pct = np.percentile(sample, [0.5, 5, 50, 95, 99.5])
    focal35 = focal35_from_fpx(f_px_full, w, h)

    return {
        "image": image,
        "depth": inv_full,
        "depth_encoding": "disparity_m",
        "width": w,
        "height": h,
        "f_px": float(f_px_full),
        "f_px_source": "exif" if f_px else "estimated",
        "focal35": float(focal35),
        "depth_pcts": {
            "p0_5": float(pct[0]),
            "p5": float(pct[1]),
            "p50": float(pct[2]),
            "p95": float(pct[3]),
            "p99_5": float(pct[4]),
        },
        "inference": {
            "width": iw,
            "height": ih,
            "side": attempt,
            "seconds": round(seconds, 2),
            "refine_seconds": round(refine_seconds, 2),
            "peak_vram_gb": round(peak, 2),
        },
        "model": MODEL_LABEL,
    }


# --------------------------------------------------------------------------- store


def job_paths(job_id: str) -> tuple[Path, Path, Path]:
    d = DATA / job_id
    return d / "photo.png", d / "depth.f16", d / "meta.json"


JOB_RE = re.compile(r"[0-9a-f]{16}")


def job_dir(job_id: str) -> Path | None:
    """Return the job directory only for a well-formed id (blocks traversal)."""
    if not JOB_RE.fullmatch(job_id):
        return None
    return DATA / job_id


def prune_cache(keep: int = 8) -> None:
    jobs = sorted(
        (p for p in DATA.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for old in jobs[keep:]:
        shutil.rmtree(old, ignore_errors=True)


def analyze_bytes(raw: bytes, filename: str) -> dict:
    job_id = hashlib.sha1(raw + b"|v5-guided").hexdigest()[:16]
    photo, depth_file, meta_file = job_paths(job_id)
    with GLOBAL_LOCK:
        if meta_file.exists() and photo.exists() and depth_file.exists():
            if job_id not in CACHE:
                CACHE[job_id] = json.loads(meta_file.read_text())
            os.utime(photo.parent, None)  # refresh mtime so prune keeps hot jobs
            return CACHE[job_id]

        load_model()
        if STATE["status"] != "ready":
            raise RuntimeError(STATE["message"])

        photo.parent.mkdir(parents=True, exist_ok=True)
        tmp = photo.parent / f"upload_{job_id}_{uuid.uuid4().hex}"
        tmp.write_bytes(raw)
        try:
            t0 = time.time()
            result = estimate(tmp, DEFAULT_SIDE)
            total = time.time() - t0
            meta = {k: v for k, v in result.items() if k not in ("image", "depth")}
            meta.update(
                id=job_id,
                filename=filename,
                total_seconds=round(total, 2),
                image_url=f"/api/jobs/{job_id}/image",
                depth_url=f"/api/jobs/{job_id}/depth",
            )
            Image.fromarray(result["image"]).save(photo, format="PNG", compress_level=1)
            result["depth"].astype(np.float16).tofile(depth_file)
            meta_file.write_text(json.dumps(meta))
            CACHE[job_id] = meta
            prune_cache()
            return meta
        finally:
            tmp.unlink(missing_ok=True)


# ----------------------------------------------------------------------------- app

app = FastAPI(title="Depth of Field Studio")


@app.get("/api/status")
def status():
    return {
        "status": STATE["status"],
        "message": STATE["message"],
        "device": STATE["device"],
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model": MODEL_LABEL,
        "load_seconds": STATE["load_seconds"],
        "side": DEFAULT_SIDE,
    }


@app.post("/api/warmup")
def warmup():
    threading.Thread(target=load_model, daemon=True).start()
    return {"status": STATE["status"]}


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...)):
    raw = file.file.read()
    if not raw:
        return JSONResponse({"error": "Empty upload"}, status_code=400)
    try:
        return analyze_bytes(raw, file.filename or "photo")
    except Exception as exc:  # noqa: BLE001
        print(traceback.format_exc())
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/jobs/{job_id}/image")
def job_image(job_id: str):
    d = job_dir(job_id)
    path = d / "photo.png" if d else None
    if not path or not path.exists():
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/depth")
def job_depth(job_id: str):
    d = job_dir(job_id)
    path = d / "depth.f16" if d else None
    if not path or not path.exists():
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return Response(
        path.read_bytes(),
        media_type="application/octet-stream",
        headers={"Cache-Control": "max-age=3600"},
    )


@app.get("/api/jobs/{job_id}/meta")
def job_meta(job_id: str):
    d = job_dir(job_id)
    path = d / "meta.json" if d else None
    if not path or not path.exists():
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return json.loads(path.read_text())


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    print(f"Depth of Field Studio — {MODEL_LABEL}")
    threading.Thread(target=load_model, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "7860")), log_level="info")
