# Realism v2 — what changed

Implementation of the renderer corrections in
[`opus-realism-review.md`](opus-realism-review.md). The server and model code are
unchanged; this is a client-side renderer rewrite.

## Renderer

* **Two fields, not one gather.** The photo is split into a premultiplied
  near/far pair at the focus plane. The far/focus field is gathered with the
  per-tap CoC reach test (`srad >= r`) and a mip-filled background behind the
  subject. The near field is gathered at a dilated radius and written with a
  **coverage alpha** rather than a normalised average. A defocused foreground
  now spreads over a sharp background instead of being cut out of it.
* **Coverage alpha from mip 0.** The colour is prefiltered to the tap spacing,
  but coverage and the source's own CoC are read from mip 0. The coarse mip
  would smear the layer's coverage outward and turn the alpha into an opaque
  average.
* **Depth test.** A tap beyond the destination's own blur radius only counts if
  it is nearer. A blurrier background cannot paint over the midground.
* **Depth edges.** The DOF fetches use nearest depth plus a 3×3 nearest-colour
  snap, so the guided filter's 2 px mixed-depth ramp cannot paint a sharp
  outline when it crosses the focus plane.
* **Prefiltered taps.** The gather reads the layer mips at
  `log2(radius * sqrt(pi / taps))`, so 96/192 taps cover a 40–280 px disc
  without the mip-0 stipple. Preview and export now match: the synthetic probe
  measured a mean absolute difference of **0.89 code values** between the
  export (960×640, 192 taps) and the preview (971×647, 96 taps) of the same
  frame.
* **Preview LOD.** The base photo reads `log2(1/u_scale)` instead of mip 0, so
  a 24 MP frame on a 1200 px canvas no longer aliases in the original view.
* **Aperture.** A nine-blade polygon whose roundness follows the f-stop, with
  mechanical cat's-eye clipping toward the frame edge and a spherical-aberration
  rim. The shape is evaluated in the **source** frame (the destination relative
  to the source, flipped for the near field), with an area normalisation so the
  cat's eye does not make an opaque foreground transparent at the corners.
* **Highlights.** A 0–5 stop clipped-highlight expansion, gated on the clipping
  mask and on isolation (the mip-5 clip fraction) and faded in with the blur
  radius. The sharp base stays unexpanded. Replaces the old weight bias.
* **Grain.** Shadow-weighted grain in blurred areas, scaled by the local
  defocus. Exposed as a slider.
* **Blur cap.** The cap is now a literal 10% of the short side (the old code
  applied it to the diameter, halving it), and it scales with zoom.

## Zoom / pan

Wheel zoom about the cursor up to 12×, drag to pan, double-click to fit. The
layer passes read the zoomed window and the CoC includes `u_scale * zoom`, so
the inspected blur matches the export. Export resets zoom and pan and renders
the whole frame. The WIP branch's `w = 1.0` near-side rule is **not** merged: it
traded a cut-out for an eroded, smeared rim. The zoom math and tests are.

## Tests

`tests/mock_depth_server.py` generates a known scene (textured background
6–12 m, a 4 m rectangle, a 1.5 m disc with a clipped specular) and serves it
without torch or a GPU. `tests/render_probe.py` drives headless Chromium
(SwiftShader) through the real client at `?debug=1` and captures the canvas.

Measured on the synthetic scene, 135 mm f/1.4, focus 8 m (near disc 1.5 m):

* Old single-pass: the near disc's edge changes from background to foreground
  in ~5 px; the blur is cut off at the geometric silhouette.
* New two-field: the near alpha's 10/50/90 crossings are 180/202/223 px on the
  left and 436/459/483 px on the right, centred on the geometric edges
  (200/460) with the expected ~35 px CoC radius.
* Near subject in focus (focus 1.5 m): the subject edge is sharp, the
  background is blurred, no halo.
* Export vs preview mean absolute difference: 0.89 / 255.
* Zoom 4× crop vs the same export crop: 2.37 / 255 mean absolute.

SwiftShader timings are not representative of the 4070; the probe takes ~16 s
per preview frame there, mostly software rasterisation.

## Deliberately not done

These are the review's remaining items, in its order:

* A subject matte / segmentation network (BiRefNet/ViTMatte class) and
  foreground colour estimation for hair and mixed pixels.
* Ultra HDR / gain-map decoding for real highlight headroom; the current
  expansion reconstructs from the 8-bit clip mask.
* K-slice compositing for near-over-near ordering.
* Longitudinal chromatic aberration (rated optional and often read as a
  defect).
* Deconvolution of the source photo's existing depth of field; the README notes
  the heuristic.
