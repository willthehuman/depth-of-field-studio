# Depth of Field Studio

Add a real depth of field to any photo, after the fact. Apple's **Depth Pro 2 (H+)**
estimates metric depth — every pixel in metres — and the browser turns that into a
physical lens: pick a focus distance, pick an aperture, and the circle of confusion
follows the thin-lens formula. Nothing is baked in, so you can re-focus the shot
forever.

![The app with a 85 mm f/2.8 simulation](docs/screenshot.png)

## Install

Needs **Python ≥ 3.11** and an NVIDIA GPU with ~4 GB free (it runs on CPU too, slowly).

```bat
git clone https://github.com/apple-aiml-research/ml-depth-pro-2.git repo
py -3.13 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m pip install -e ./repo
```

`requirements.txt` pins `torch==2.11.0+cu128` / `torchvision==0.26.0+cu128` from the
CUDA index — plain PyPI wheels are CPU-only, and mismatched torch/torchvision pairs
fail at import with `operator torchvision::nms does not exist`.

The weights download themselves on first run (3.5 GB into `torch_cache/`).

## Run it

```bat
run.bat
```

or

```bat
venv\Scripts\python.exe app.py
```

then open <http://127.0.0.1:7860>.

The first launch downloads the H+ checkpoint (3.5 GB) into `torch_cache/`. Later
launches take ~12 s to build the model on the CPU and hand it to the GPU.

## What it does

| | |
|---|---|
| **Depth** | `apple/DepthPro2-H` (872 M params, bf16) at 1280 px on the GPU, refined to full resolution with a colour-aware guided filter so silhouettes stay aligned with the photo, including chroma edges of equal luminance. ~1.5–2.5 s for a photo, ~4 s for a 27 MP one. The DOF gathers fetch depth with nearest sampling and a min/max nearest-colour snap, so the filter's 2 px mixed-depth ramp cannot paint a sharp outline when it crosses the focus plane. |
| **Focus** | Slider, *Auto* (median of the centre), *Centre*, or click anywhere in the photo. Distances are in metres, straight from the model. |
| **Aperture** | f/1.0 → f/22 in third stops. |
| **Lens** | *Native* uses the focal length the model estimates; 50/85/135 mm simulate a longer lens, which is how you get a portrait look out of a wide-angle frame. |
| **Zoom** | Wheel zoom about the cursor up to 12×, drag to pan, double-click to fit. The DOF is recomputed at the zoomed scale and the blur cap scales with the zoom, so the inspected blur matches the export. Export always renders the whole frame. |
| **Render** | Two-field WebGL2 bokeh in **linear light**: a far/focus field with a per-tap CoC reach test and a 16-tap background fill, then a near field with a **premultiplied coverage alpha**, so a defocused foreground spreads over a sharp background instead of being cut out. The near gather radius comes from a **tile-max + neighbour-max CoC field**, not a colour-mip average, so a small very defocused object keeps its full disc. The aperture is a nine-blade polygon whose roundness follows the f-stop, with equal-radius cat's-eye clipping toward the frame edge and an energy-neutral spherical-aberration rim; the gather covers the polygon's circumradius, so the 9-gon is actually visible at f/4+. A tap beyond the pixel's own blur radius must be nearer than the destination, so a blurrier background cannot paint over the midground. 96 taps preview, 192 export, prefiltered to the tap spacing. |
| **Highlights** | A 0–5 stop slider expands only clipped, isolated speculars in a full-resolution **radiance texture**, before the layer mips prefilter the colour. Bokeh balls carry real energy, the sharp base stays unexpanded, and preview/export agree at every scale. Large clipped areas (sky, windows) are left alone. |
| **Grain** | Matching grain in blurred areas, weighted by the local defocus and by signal (shot-noise-like), so a wide-aperture render does not look unnaturally clean at 1:1. |
| **Export** | Full-resolution JPEG/PNG rendered in the browser, plus the depth map as a PNG. Export draws are split into ≤ 2 MP scissored bands with a flush between them to stay under the Windows TDR limit. |

The depth-of-field band under the focus slider is the honest lens model
(35 mm-equivalent focal length, 30 µm CoC, hyperfocal distance), so it tells you
what a real lens of that focal length and aperture would do. The blur,
highlight and grain sliders are explicitly artistic: they change the render,
not the optics.

## Hardware

Started on a 6 GB card and verified on an RTX 4070 SUPER (12 GB). The released
checkpoint is a 3.5 GB float32 download; the app builds the model on the CPU,
casts the encoder and decoder to bf16 (they only ever run under bf16 autocast
anyway) and keeps the depth head and focal estimator in float32. The resident
encoder/decoder is about 1.8 GB. Inference peaks at 3.96 GB at 1280 px on the
12 GB card; 4.9 GB at 1536 px (`set DEPTH_SIDE=1536` to try it). A 5656×4242
photo analysed in 3.1 s (standalone backend) and exported in 4.0 s wall; tiled
and non-tiled exports are pixel-identical. `nvidia-smi` sampled 2.8 GiB total
during the export, but the texture accounting plus the resident model is about
3.7 GB; the 6 GB fit is not measured.

The two-field renderer keeps four canvas-sized float layer textures plus their
mips, two blur targets, the full-resolution linear photo and the full-resolution
radiance texture. At 24 MP the `nvidia-smi` sample was 2.8 GiB, while the texture accounting plus
the resident model is about 3.7 GB. It fits on the 12 GB card; the 6 GB target
remains an estimate, not a measurement, and an 8192 px export can exceed either. The export is drawn in ≤ 2 MP bands with a flush between draws to stay
under the Windows TDR limit.

## Layout

```
app.py              FastAPI server: /api/analyze, depth IO, static hosting
static/index.html   UI shell
static/style.css    dark photo-editor styling
static/app.js       two-field WebGL2 DOF pipeline, controls, zoom/pan, export
probe_sizes.py      measures VRAM and latency per inference size
test_backend.py     headless pipeline check (no browser)
tests/mock_depth_server.py  synthetic scene + depth, no torch/GPU needed
tests/render_probe.py       headless Chromium probe: focus/aperture/zoom/export
cdp_test.py         drives headless Brave over CDP: orientation, depth ordering,
                    aperture/refocus behaviour, zoom/pan, export path
                    -> browser_test.js
cdp_screen.py       screenshots the empty and loaded UI
data/<id>/          cached photo.png + depth.f16 + meta.json (last 8 photos)
```

## Notes

* Depth is metric *for the camera that took the photo* — the model predicts depth
  in metres from a monocular cue, and the absolute scale is approximate on
  unusual scenes (very wide lenses, heavy crops, macro). Relative depth is solid.
* Sky is infinite depth (1/z = 0); it gets the blur for the chosen focus, which
  is the largest blur for distant content but not always the largest in frame —
  a close foreground can be defocused more.
* The source photo's own depth of field is not deconvolved. The renderer adds
  the target CoC; a photo already stopped down cannot recover detail the original
  lens removed. On DSLR sources the right target is closer to
  √(c² − c_native²) when EXIF gives the source aperture, but that is a
  heuristic, not a deconvolution.
* The blur radius is capped at 10% of the short side of the frame. On a 24 MP
  export that is 400 px, above the physical CoC for the built-in lens presets at
  their widest and longest settings.

## Licence

The **code** here is MIT — see [LICENSE](LICENSE). Apple's material (the model
weights, the `ml-depth-pro-2` package and the sample photo) is **not** MIT and
stays research-only — see [NOTICE.md](NOTICE.md).

This repository is not a redistribution of Depth Pro 2, and the MIT licence does
not extend to Apple's material:

* **Model weights** — `apple/DepthPro2-H` and `apple/DepthPro2-focal` are
  downloaded at runtime under the
  [Apple Machine Learning Research Model Licence](https://github.com/apple-aiml-research/ml-depth-pro-2/blob/main/LICENSE_MODEL):
  **non-commercial scientific research and academic development only**. Nothing
  in this repo grants rights to those weights or to the `ml-depth-pro-2`
  package.
* **Sample image** — `static/sample.jpg` and the renders in `docs/` come from
  Apple's example photo in that repository and are included only so the "Try a
  sample" button works. Treat them as research-use material, and replace them if
  you redistribute this project.

