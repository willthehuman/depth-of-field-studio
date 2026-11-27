# Realism v2 — what changed

Implementation of the renderer corrections in
[`opus-realism-review.md`](opus-realism-review.md), plus the PR-review fixes.
The only server change is one `torch.cuda.empty_cache()` after the estimate;
everything else is client-side.

## Renderer

* **Two fields, not one gather.** The photo is split into a premultiplied
  near/far pair at the focus plane. The far/focus field is gathered with the
  per-tap CoC reach test (`srad >= r`) and a **16-tap background fill** behind
  the subject. The near field is gathered at a dilated radius and written with a
  **premultiplied coverage alpha** rather than a normalised average. A defocused
  foreground spreads over a sharp background instead of being cut out.
* **One alpha convention.** The early-outs and the gathers all write
  premultiplied colour; the final pass is Porter-Duff `over`
  (`n.rgb + f.rgb * (1 - n.a)`). The old mix of premultiplied early-outs and
  straight gather colours darkened an in-focus subject to 0.75× linear; the
  flat-disc test now matches the original exactly at 172.5/52.9/36.6 in the 960×640 export (40×40 patch).
* **Coverage alpha from mip 0.** The colour is prefiltered to the tap spacing,
  but coverage and the source's own CoC are read from mip 0. The coarse mip
  would smear the layer's coverage outward and turn the alpha into an opaque
  average.
* **Near small-CoC coverage.** One 0.5 px floor (`sradE`) is used in both the
  shape test and the weight. With the real `srad` in the shape and a 0.5 px
  weight floor, a near pixel whose CoC is below 0.5 px had an expected alpha of
  `4·srad²` — near content just in front of the focus plane went 20–80 %
  transparent to the far fill (rows 165–191 of the mock read 0.21–0.98). The
  0.5 px floor and the 24-tap inner set cover that case; the same rows read
  six sampled rows read 0.96–1.0 and the 4 m rectangle reads 1.0. An always-on self tap was
  removed after it proved unreachable (`selfRadE >= firstR` in every branch).
* **Near-over-near ordering.** Taps whose source is nearer than the destination
  (`sd > d`) go to a separate `fg` bucket. The occlusion is normalised,
  `occ = fg.a / max(shapeArea, 0.2)`, and the near value is
  `fg + acc·(1 − occ)`, so a nearer foreground occludes the destination's own
  pixel instead of being cut behind it and an unnormalised `fg.a` cannot
  subtract colour.
* **Near radius from a true max.** A 32 px tile-max of the near CoC plus a
  reach-limited neighbour-max (a tile contributes only if its own CoC reaches
  the destination tile) replaces the coverage-weighted colour-mip probe. A
  small very defocused object keeps its full disc; a destination far from a
  foreground does not pay for a full tap set.
* **Depth test.** A tap beyond the destination's own blur radius only counts if
  it is nearer. A blurrier background cannot paint over the midground.
* **Depth edges.** A 17 px, stride-2 9×9 min/max snap runs once per photo at
  the **depth resolution** into an R16F texture; the layer, blur and final
  `texelFetch` it instead of repeating the search seven times per frame. The
  snap takes the min/max depth and chooses the extreme whose cluster-mean
  colour is closest, but only when the centre sits inside the mixed band, the
  near and far 10 % clusters each hold at least 20 % of the samples, the middle
  holds at most 55 %, and the two cluster means differ in colour
  (`|mn − mf|² > 0.01`). A real step has plateaus on both sides; a planar slope
  (even textured) spreads through the middle and is left smooth. The mock now
  has a textured 0.7–2.2 m ground plane (snap equals raw depth at three points)
  and a 9 px 2–4 m ramp with plateaus. On the ramp, six of the nine samples
  snap to a plateau at y = 590, and 146 of 270 ramp pixels change overall: one
  edge pixel per row remains mixed and the row ends do not snap. The snap
  reduces a guided outline but does not erase every mixed pixel. It is built once per photo, not
  per focus tick. It is
  stored in **photo orientation** (the FBO row is flipped once in `SNAP_FRAG`);
  every reader indexes `texelFetch(u_depthSnap, p, 0)` top-down, so a
  bottom-of-frame near object is not classified with the top plane's depth. A
  0.4-scale preview (384×256) still builds the snap at the full depth size
  (960×640), which is the size the readers index; the earlier canvas-size
  rebuild was the bug.
* **Prefiltered taps.** The gather reads the layer mips at
  `log2(radius * sqrt(pi / taps))`, so 96/192 taps cover a 40–280 px disc
  without the mip-0 stipple.
* **Radius-stratified taps.** When the destination's effective CoC
  `ownE = max(own, 0.5)` is below a quarter of the dilated gather radius, 24
  taps form an inner disc at `1.07·ownE` with their own area weight and their
  own (sharper) mip; the rest cover the outer annulus. Without it a small-CoC
  source beside a large near blur got only 1–5 taps and its alpha speckled;
  the `0.5` floor also catches destinations just in front of focus.
* **Preview LOD.** The base photo reads `log2(1/u_scale)` instead of mip 0, so
  a 24 MP frame on a 1200 px canvas no longer aliases in the original view.
* **Aperture.** A nine-blade polygon whose roundness follows the f-stop, with a
  **hard** cat's-eye clip by a circle of radius `1.07 − 0.07·cat` offset
  `0.60·cat` toward the frame edge: at `cat = 0` it sits outside the 1.042
  polygon, at `cat = 1` it is the equal-radius almond. The old
  `mix(1, clip2, cat)` left a translucent `1−cat` crescent instead of a
  geometric clip. An energy-neutral spherical-aberration rim is added after.
  The gather covers the polygon's circumradius (1.07×, also centring the
  source-disc ramp), so the 9-gon is visible at f/4+. The alpha is divided by
  the exact two-circle intersection area `catArea2(1, rc, 0.60·cat)/π`, not a
  linear fit.
* **Highlights.** A 0–5 stop expansion in a **full-resolution radiance texture**
  at mip 0, before the layer mips average the colour. The isolation gate is the
  mip-5 clip fraction; the layer mixes the radiance in by the pixel's own blur
  radius, so the sharp base stays unexpanded and preview/export agree at every
  scale. The old version read the clip mask at `baseLod`, so preview highlights
  were up to an order of magnitude dimmer than the export.
* **Grain.** Signal-weighted (shot-noise-like) grain in blurred areas, scaled by
  the **effective** defocus: the far radius mixed with the near max by the near
  alpha, so the grain does not switch on the geometric silhouette inside a soft
  foreground edge. Exposed as a slider.
* **Blur cap.** The cap is now a literal 10% of the short side
  (`min(0.5 * coc, u_maxRadius)`); the old code still applied it to the
  diameter and the effective radius was 5%.
* **Export safety.** `exportRender` restores the canvas, mode and zoom in a
  `finally`; a WebGL context-loss handler shows an error instead of a dead
  canvas; the drawing buffer is checked against the requested export size; and
  export draws are split into ≤ 2 MP scissored bands with a flush between them.

## Zoom / pan

Wheel zoom about the cursor up to 12×, drag to pan, double-click to fit. The
layer passes read the zoomed window and the CoC includes `u_scale * zoom`, so
the inspected blur matches the export. Export resets zoom and pan and renders
the whole frame. The WIP branch's `w = 1.0` near-side rule is **not** merged: it
traded a cut-out for an eroded, smeared rim. The zoom math and tests are.

## Tests

`tests/mock_depth_server.py` generates a known scene (textured background
6–12 m, a 4 m rectangle, a 1.5 m disc with a clipped specular, and a small 8 px
1.5 m coin) and serves it without torch or a GPU. `tests/render_probe.py` drives
headless Chromium (SwiftShader) through the real client at `?debug=1` and
captures the canvas.

Measured on the synthetic scene, 135 mm f/1.4, focus 8 m (near disc 1.5 m),
unless stated:

* **Focus-band brightness (the blocker).** `blur = 0` at focus 8 m, compared at
  the same 960×640 export size: **0.000 / 255 mean and 0.0 max** — exactly the
  original. The earlier 971×647 preview comparison showed 0.108 mean and up to
  57 codes at the clipped specular because result mode resamples in linear light
  while original mode resamples in sRGB; that was a harness artefact, not the
  renderer. Focus exactly on the near disc at 1.5 m: the 40×40 near patch is
  172.5/52.9/36.6 (R/G/B), identical to the original 172.5/52.9/36.6 in the 960×640 export.
* **Near edge.** The linear `nearBlur.a` 10–90% transition along y = 360 runs
  x 181→229: **48 px**, within 2 % of the 49 px hard-disc (pillbox) prediction
  (1.37 R). The earlier 36 px figure was the colour transition, not the alpha;
  there is no truncation. The blur cap is not active in this frame (uncapped
  radius 35.8 px). The old colour-mip version measured ~56 px because it
  overestimated the near radius.
* **Highlights.** A full-slider specular bokeh at focus 8 m: the 8 px core is
  **188/102/94** in the preview (971×647) and **186/99/89** in the export
  (960×640). Preview vs export downsampled to 960 is 0.712 / 255 mean absolute.
* **Small object.** The 8 px near coin's 16×16 centre is now **90.6/81.1/83.1**
  (R/G/B) against the ~80 analytic; the near layer's alpha is 0.045, the depth
  snap reads 0.6665 (1.5 m) and the near-coverage mips read 0.595 / 0.302 /
  0.088 at levels 3/4/5. The earlier 94/187 flip-flop was the depth snap being
  written bottom-up: the coin was classified with the top plane's depth. The
  alpha is now the plain aperture coverage, and the coin's PSF sees the near
  source for only 4.5 % of the aperture. Near/far decomposition at the coin
  centre: near R 0.031 (premultiplied, alpha 0.045) + far R 0.072 × (1 − 0.045)
  = 0.0998 linear ≈ 89 sRGB, with the 16×16 mean 90.6 because the patch includes
  the brighter coin edge. The +10 codes against 80 are the near layer's own red
  (0.676 straight × 0.045), not a far-fill error. The flat near-disc centre still
  matches the original at 172.5/172.5.
* **Polygon.** At f/8 the per-angle specular bokeh edge radius varies 7.5–11 px
  (std 1.02). The 9th angular harmonic
  is 0.18 px, so this 5 px source cannot isolate the 9-gon; the polygon is
  visible but this metric does not demonstrate it. The theoretical
  vertex/inradius difference is ~0.6 px at that edge.
* **Preview vs export.** 0.712 / 255 mean absolute between the 971×647 preview
  and the 960×640 export downsampled to 960, 96/192 taps, with grain on.
* **Zoom.** A 4× centre crop vs the export's central 240×160 quarter upscaled:
  1.41 / 255 mean absolute, max 35; the earlier 2.37 was not a like-for-like
  crop and is superseded.

## Real-photo validation

On an RTX 4070 SUPER (12 GB), with the real Depth Pro 2 H+ weights and the real
client (not the mock):

- Model load 9.2 s warm; 1920×1440 analysis 0.70 s; 5656×4242 standalone
  `test_backend.py` analysis 3.11 s (the app's 1.3 s is a cached meta value, not a measurement);
  inference 0.76 s, refine 0.14 s. The reported 3.96 GB peak is read before
  `guided_upsample`; the refine pass is not separately sampled.
- 5656×4242 export: 21.9 MB PNG, 4.0 s wall with a **cached** analysis;
  the cold standalone analysis is 3.11 s; tiled vs non-tiled **0.000 / 0.0**. `nvidia-smi` sampled 2.8 GiB
  total during the export, but the texture accounting plus the resident model
  is about 3.7 GB; the 6 GB fit is not measured.
- `cdp_test.py` browser suite against the real model server: **ok: true**, no
  failures; export 1920×1440; click-to-focus while zoomed works. WebGL ran on
  SwiftShader in that suite. Zoom anchor drift is a 3.7 code **sum** over three
  channels on a 5×5 patch, not a 0–255 mean.
- 0.4× preview vs the 24 MP export downsampled: mean 0.458/255, p99 4.0,
  max 39; top-decile gradient band mean 1.25/255.

The 6 GB card was the original target; the 12 GB run shows the actual
inference and texture costs, but the 6 GB re-run is still open.

## Deliberately not done

These are the review's remaining items, in its order:

* A subject matte / segmentation network (BiRefNet/ViTMatte class) and
  foreground colour estimation for hair and mixed pixels.
* Push-pull filling across all mip levels; the 16-tap fill is a local
  approximation. So is connected-component isolation for the highlight gate;
  the mip-5 local clip still glows faintly at the edge of a large blown region.
* Ultra HDR / gain-map decoding for real highlight headroom; the current
  expansion reconstructs from the 8-bit clip mask.
* K-slice compositing for near-over-near ordering.
* Longitudinal chromatic aberration (rated optional and often read as a
  defect).
* Grain in photo-pixel space rather than output-pixel space; the export grain
  is finer than the preview at the same viewing size.
* Deconvolution of the source photo's existing depth of field; the README notes
  the heuristic.
