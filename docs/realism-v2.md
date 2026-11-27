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
  flat-disc test now matches the original to 172.7/53.6/37.1 code values.
* **Coverage alpha from mip 0.** The colour is prefiltered to the tap spacing,
  but coverage and the source's own CoC are read from mip 0. The coarse mip
  would smear the layer's coverage outward and turn the alpha into an opaque
  average.
* **Near radius from a true max.** A 32 px tile-max of the near CoC plus a
  reach-limited neighbour-max (a tile contributes only if its own CoC reaches
  the destination tile) replaces the coverage-weighted colour-mip probe. A
  small very defocused object keeps its full disc; a destination far from a
  foreground does not pay for a full tap set.
* **Depth test.** A tap beyond the destination's own blur radius only counts if
  it is nearer. A blurrier background cannot paint over the midground.
* **Depth edges.** The 5×5 min/max snap runs once per photo into an R16F
  texture; the layer, blur and final `texelFetch` it instead of repeating the
  search seven times per frame. The snap takes the min/max depth and chooses
  the extreme whose colour is closest, only when the centre sits inside the
  mixed band. It is stored in **photo orientation** (the FBO row is flipped
  once in `SNAP_FRAG`); every reader indexes `texelFetch(u_depthSnap, p, 0)`
  top-down, so a bottom-of-frame near object is not classified with the top
  plane's depth.
* **Prefiltered taps.** The gather reads the layer mips at
  `log2(radius * sqrt(pi / taps))`, so 96/192 taps cover a 40–280 px disc
  without the mip-0 stipple.
* **Preview LOD.** The base photo reads `log2(1/u_scale)` instead of mip 0, so
  a 24 MP frame on a 1200 px canvas no longer aliases in the original view.
* **Aperture.** A nine-blade polygon whose roundness follows the f-stop, with
  equal-radius cat's-eye clipping toward the frame edge and an energy-neutral
  spherical-aberration rim. The gather covers the polygon's circumradius
  (1.07×, also centring the source-disc ramp), so the 9-gon is actually visible
  at f/4+; the old circular source test clipped the vertices. The alpha is
  divided by the exact two-circle intersection area, not a linear fit.
* **Highlights.** A 0–5 stop expansion in a **full-resolution radiance texture**
  at mip 0, before the layer mips average the colour. The isolation gate is the
  mip-5 clip fraction; the layer mixes the radiance in by the pixel's own blur
  radius, so the sharp base stays unexpanded and preview/export agree at every
  scale. The old version read the clip mask at `baseLod`, so preview highlights
  were up to an order of magnitude dimmer than the export.
* **Grain.** Signal-weighted (shot-noise-like) grain in blurred areas, scaled by
  the local defocus. Exposed as a slider.
* **Blur cap.** The cap is now a literal 10% of the short side
  (`min(0.5 * coc, u_maxRadius)`); the old code still applied it to the
  diameter and the effective radius was 5%.
* **Export safety.** `exportRender` restores the canvas, mode and zoom in a
  `finally`; a WebGL context-loss handler shows an error instead of a dead
  canvas; the drawing buffer is checked against the requested export size; and
  export draws are split into 512-row scissored bands with a flush between them.

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

* **Focus-band brightness (the blocker).** `blur = 0` at focus 8 m against the
  original: mean absolute 0.108 / 255, p99 1.0 / 255; a few pixels of the 5 px
  clipped specular reach 57 / 255. Focus exactly on the near disc at 1.5 m: the
  120 px near patch is 172.7/53.6/37.1 (R/G/B) versus 172.6/53.5/37.0 for the
  original.
* **Near edge.** The near disc's colour transition is ~36 px (10–90%), centred
  on the geometric edges (200/460); the old colour-mip version measured ~56 px
  because it overestimated the near radius. The blur cap is no longer active in
  this frame (uncapped radius 35.8 px). A Gaussian-step prediction would be
  ~49 px (1.37 R); the near layer is a **coverage** alpha from a hard polygon
  aperture rather than a Gaussian, so a steeper 10–90% edge is expected, and the
  36 px is not truncation.
* **Highlights.** A full-slider specular bokeh at focus 8 m: the 8 px core is
  **188/102/94** in the preview (971×647) and **186/99/89** in the export
  (960×640); the core means agree to ~1 code value. Preview versus export at
  highlight 1.0 is 0.578 / 255 mean absolute; the 480 px preview against the
  export downsampled to 480 is 0.418 / 255.
* **Small object.** The 8 px near coin's 16×16 centre is now **97.6/84.6/82.1**
  (R/G/B) against the ~80 analytic; the near layer's alpha is 0.046, the depth
  snap reads 0.6665 (1.5 m) and the near-coverage mips read 0.595 / 0.302 /
  0.088 at levels 3/4/5. The earlier 94/187 flip-flop was the 5×5 snap being
  written bottom-up: the coin was classified with the top plane's depth. The
  alpha is now the plain aperture coverage, and the coin's PSF sees the near
  source for only 4.6 % of the aperture. Near/far decomposition at the coin
  centre: near R 0.031 (premultiplied, alpha 0.046) + far R 0.072 × (1 − 0.046)
  = 0.0997 linear ≈ 89 sRGB, with the 16×16 mean 97.6 because the patch includes
  the brighter coin edge. The +18 codes against 80 are the near layer's own red
  (0.716 straight × 0.046), not a far-fill error. The flat near-disc centre still
  matches the original at 172.7/172.6.
* **Polygon.** At f/8 the specular bokeh edge radius varies 7.5–11 px around the
  disc; the 1.04 px std includes tap and 5 px source noise. The theoretical
  circumradius/inradius spread is ±0.35 px at that edge; measure the 9th angular
  harmonic if you need to separate the polygon from tap noise. At f/1.4 it is
  round.
* **Preview vs export (no highlight test).** 0.90 / 255 mean absolute between
  the 971×647 preview and 960×640 export at 96/192 taps with grain on.
* **Zoom.** The previous 4× crop vs export crop measurement was 2.37 / 255
  mean absolute; not re-run after the review fixes.

SwiftShader timings are not representative of the RTX 4070; the probe takes
~16 s per preview frame there, mostly software rasterisation. The 24 MP memory
figure is an accounting estimate, not a measurement on the 6 GB card.

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
