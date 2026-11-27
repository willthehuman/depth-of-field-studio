# Depth of Field Studio: a realism review of the synthetic depth of field

Tags: **[Public]** means documented or established practice. **[Inferred]** means it comes from reasoning about the code shown here. Nothing was run.

## Bottom line

Not yet: "stop adding renderer features" would be the wrong call today. The current renderer has three structural defects that matter more than any lens feature:

1. **Edges.** The occlusion structure causes the cut-out silhouette. The gather radius is capped at each pixel's own blur radius, and the weights are normalised so they always add up to full opacity.
2. **Sampling.** Exports are undersampled by about 12× compared with the preview. An export does not look like the preview, and small highlights hardly form bokeh discs at all.
3. **Highlights.** The 8-bit source caps highlight energy, so bokeh from clipped highlights comes out dim and grey.

Once those are fixed, stop adding bokeh-character features. From then on, the remaining gap is the subject's silhouette (matte and depth edges) and the source image (8-bit clipping, noise, any blur already in the photo).

---

## §6: verdict on each suspicion

**1. Missing features and polygon rotation.** Partly true.
- True: the active path has no polygon, no cat's eye and no HDR. The README is wrong.
- False, as written: the polygon does not rotate per pixel. `polygonLimit(angle, …)` is evaluated at the tap's actual direction in image space, and that direction includes `spin`. The 0.17 rad offset is absolute. So the 9-gon keeps a fixed orientation. `spin` only rotates which directions get sampled. [Inferred]

  Re-enabling it as-is would still be wrong, for other reasons:
  - **Wrong frame.** In a gather, the shape test belongs in the *source's* frame. The question is whether the destination lies inside the source's kernel. That means evaluating at `angle + π` with `srad`, not at `angle` with the destination's `radius`. With 9 blades the shape is not symmetric under a 180° turn, so the error is visible. Defocus in front of the focus plane should also show the aperture flipped 180° compared with defocus behind it. [Public, geometric optics]
  - **Uneven density.** If `limit` was used to stretch `r` radially, sample density goes as 1/L(θ)², so the corners get fewer samples.
  - **Cat's-eye `sgn` does nothing.** Two circles with equal radii at ±offset are symmetric, so `max(d1, d2)` is unchanged when the sign flips. Real cat's eye comes from two clipping circles of *different* radii (front and rear element). [Inferred]
  - **Undefined `smoothstep`.** `smoothstep(1.0, 0.92, x)` has edge0 > edge1. GLSL ES 3.00 leaves that undefined. [Public, GLSL ES 3.00 spec]

**2. The gather rule.** Half true.
- The per-tap test `srad ≥ r` is the correct scatter-as-gather acceptance test. [Public]
- But two other things in the active shader cause the cut-out (both detailed under Q2):
  - The gather radius is the destination pixel's own blur radius, and there is an early return when it is below 0.55 px. A sharp background pixel next to a blurred foreground object never reaches that object, so the foreground can't spread outward.
  - Dividing by `wsum` forces opacity to 1. The inner rim of a defocused foreground object stays fully opaque, when it should be partly see-through.
- The WIP's `w = 1.0` treats the inside half of the edge with the wrong physics, and does nothing for the outside half.
- The depth-edge ramp (see 6) adds a little to the cut-out. The weight rule plus the radius cap are the main cause.

**3. Undersampling.** True, and worse than described:

| Case | Disc area | Taps | Share of the disc sampled |
|---|---|---|---|
| Preview, R = 28 px | 2,463 px | 96 | 3.9 % |
| Export, R = 140 px | 61,575 px | 192 | 0.31 % |

- A one-pixel highlight lands under a bilinear tap with probability ≈ 4N/(πR²). That is about 16 % of the pixels in its disc on the preview and about 1.2 % on export. So exported point highlights come out as scattered speckles instead of discs. The preview and the export do not match. [Inferred]
- Prefiltering is the right fix, as long as it is depth-aware. Accumulation alone cannot fix export (see Q3).

**4. Hard 0.55 px cutoff.** True, but it matters little as a visible line: the step goes from sharp to a blur of about 0.5 px. Two things are visible:
- the point where the dithering noise starts;
- the early return, which also stops the pixel from *receiving* blur from its neighbours. That second effect is the important one.

Fix: always run the gather and blend in by radius.

**5. Weight bias on bright taps.** Half wrong.
- It is not energy-conserving, and it tilts the average towards bright colours. Its threshold (max channel above 0.55 in linear light, about 0.77 in sRGB) also catches skin, sky and white shirts.
- But the output is still a convex mix of inputs that are at most 1.0. So it cannot make anything brighter than the brightest source.
- The real symptom is the opposite of "flat white discs". A small clipped highlight spread over a large disc averages down to a **dim grey disc**. Flat white discs would only appear once HDR expansion is added and the output is clipped, and that is actually what real camera JPEGs show for strong speculars. [Inferred]

**6. Phantom depths at silhouettes.** True, assuming `u_depth` uses LINEAR filtering. R16F can be filtered in WebGL2. [Public]
- Bilinear filtering is the smaller source. The guided-filter upsample already writes a 2–4 px ramp of mixed depths into the full-resolution texture.
- When the edge spans the focus plane (foreground at 1 m, background at 10 m, focus at 2 m), the ramp passes through 1/s and paints a **sharp outline** around the blurred object.
- Next to an in-focus subject, the ramp gives the classic "less-blurred background halo".
- Interpolating metres instead of 1/z only moves the midpoint. It is not a reference. [Inferred]

**7. CoC derivation.** Correct.
- Thin lens: c_mm = f²/(N(s−f)) · |z−s|/z = (f²/N) · |1/s − 1/z| · s/(s−f). [Public]
- At fixed framing, the simulated sensor is scaled by f_sim/f_native. Its pixels per mm are then f_px_native/f_sim, so c_px = f_px_native · f_sim/N · |Δ(1/z)|[1/m] · 0.001 · s/(s−f_sim). ✓
- In practice, f_sim/N is a pupil-diameter control: perspective and field of view do not change. [Inferred]

Checks:
- All your table values reproduce with f_px = 6000/36 × f, **except 50 mm f/1.8 at infinity**. |0 − 0.5| = |1 − 0.5|, so infinity must equal the 1 m value: **119 px, not 116**.
- f_px is linear in the CoC. A 10 % error in the network's focal estimate means about 0.14 stop of aperture error.
- For 4:3 phone images, derive f_px from focal35 using the diagonal (43.27 mm), not the width. [Inferred]

**New bug: the radius cap is halved.** `u_maxblur` is documented as a radius, but the code does `0.5 * min(cocDiameter, u_maxblur)`. That caps the *diameter*, so the real radius cap is 5 % of the short side:

| Target | Documented radius cap | Actual radius cap |
|---|---|---|
| 1200×800 preview | 80 px | 40 px |
| 4000 px-tall export | 400 px | 200 px |

- The 135 mm f/2.8 cases at 1 m (radius 291) and 10 m (radius 232) are clipped at 200 on export. That squashes the background's depth gradation. [Inferred]

**New bug: the preview aliases.** At `u_scale ≈ 0.2–0.4`, `textureLod(..., 0.0)` reads mip 0 of a minified photo. Fine in-focus texture shows moiré, and the preview looks crunchier than the export. Use lod = log2(1/u_scale) for the base and the taps. [Inferred]

---

## Q1: ranking, in brief

The full ranking is the table at the end. In order of value:
1. Fix the near-field occlusion (coverage-alpha near field with a dilated gather).
2. Depth-aware multi-resolution sampling.
3. Snap depth edges and fetch depth with nearest sampling.
4. Expand clipped highlights, replacing the weight bias.
5. Add matched grain in blurred areas.
6. Fix the radius cap and the preview LOD.
7. Add a subject matte.

Lens character (cat's eye, aperture shape, aberrations) comes after all of these. "Do nothing" is not the honest answer yet.

---

## Q2: occlusion and edges

**What actually causes the cut-out** [Inferred]. Take a blurred foreground object (radius R) over a sharp background.
- **Outside the edge:** background pixels have radius ≈ 0 and return early. The foreground never spreads outward. The edge is hard on the outside.
- **Inside the edge:** foreground taps all pass the test, while sharp background taps fail it. After dividing by `wsum`, the pixel is 100 % foreground right up to the silhouette. The edge is hard on the inside.
- **Physically:** the result is opacity α(p) = (foreground mask ⊛ disc of radius R)(p). That is 0.5 at the silhouette, with partial transparency R px on *both* sides. [Public, thin-lens partial occlusion]

**What the WIP does.** `w = 1` turns the foreground pixel into a plain box blur over radius R. That gives the right mixing ratio on the inside. But:
- the background seen through the rim gets smeared at the foreground's radius instead of staying sharp;
- nothing changes outside, so the blur sits on one side only and the object looks **shrunk by about R/2**;
- on surfaces that cross the focus plane continuously (ground in front of feet, a forearm reaching towards the camera), the near pixels pull in sharp in-focus content. Physically, sharp points cannot reach them, so this is a real halo generator.
- In short: it trades a cut-out for an eroded, smeared rim.

**Recommended architecture: two fields, continuous blur radius, a coverage-alpha near field.** This is the Jimenez structure ("Next Generation Post Processing in Call of Duty: Advanced Warfare", SIGGRAPH 2014 Advances course [Public]), adapted to a still-image budget.

1. **Signed blur radius per pixel.** Read depth with `texelFetch` (nearest sampling). Near = radius > 0.5 px in front of focus. Far/focus = everything else.
2. **Far/focus field.** Keep the current gather and test, with three changes:
   - skip near taps entirely;
   - weight each tap by `inside / max(srad², 0.25)`, which matches the scatter energy;
   - at pixels covered by near content, gather far taps only (this inpaints the hidden background), with a pull-push fallback. Pull-push is the hole fill from Kraus & Strengert 2007. [Public]
3. **Near field.** Work at half resolution once R exceeds about 8 px.
   - Take the tile maximum of the near radius over 16×16 tiles, dilated by the largest near radius.
   - Gather over that radius, near taps only, **without normalising the opacity**:

```glsl
float A = PI*R*R/float(N);                 // area represented by one tap
vec4 acc = vec4(0.0);
for (...) {
  float sd = texelFetch(u_depth, ivec2(suv*depthSize), 0).r;
  if (signedCoc(sd) <= 0.5) continue;      // near sources only
  float srad = cocRadius(sd);
  float inside = clamp(srad - r + 0.5, 0.0, 1.0);
  float w = inside * A / (PI*max(srad*srad, 0.25));
  acc += vec4(sc*w, w);
}
vec3 nearRGB = acc.rgb / max(acc.a, 1e-4);
float alpha  = clamp(acc.a, 0.0, 1.0);     // =1 deep inside, ~0.5 at silhouette, 0 at R outside
out = mix(farFilled, nearRGB, alpha);
```

4. **Depth test for any gather radius larger than the pixel's own.** A tap beyond the pixel's own radius only counts if it is nearer than the destination. Without this rule, a blurrier background paints over the midground. (The WIP's far-side cross is missing exactly this.)

**Failure modes:**
- no ordering between two overlapping near objects at different depths;
- the background behind near rims is a guess (inpainting);
- thin near objects such as branches lose coverage when taps are sparse;
- tile dilation gets expensive when R is large, so keep the near field at reduced resolution.

How it differs from the dead two-layer version:
- the per-tap test stays in both fields;
- near opacity comes from coverage, not from a split mask;
- the fill comes from a far-only premultiplied source, not from raw mips that contain the subject's colours. Raw mips are the likely source of the old "haloed subject edges". [Inferred]

**Escalation path.** K-slice back-to-front compositing:
- slices in signed blur radius, K ≈ 8–12, with tent weights between neighbouring slices;
- variable per-tap radius inside each slice;
- premultiplied, pull-push-filled slices.

This handles near-over-near ordering. It costs K passes. With constant-kernel slices it causes banding or double rims, so keep per-tap radii. Nuke's ZDefocus works by depth layers [Public]. Only go there if overlapping foreground objects are common in your photos.

**Production tools** (as far as public documentation goes):
- **Google Pixel Portrait** (Wadhwa et al., SIGGRAPH 2018): dual-pixel depth plus a person-segmentation network, then depth-ordered rendering. [Public] Recalled, to verify: the paper deliberately widens the in-focus range beyond physical so the whole subject stays sharp.
- **Apple Portrait / Depth Control:** the rendering is not documented. AVFoundation exposes a high-resolution `AVPortraitEffectsMatte` and semantic mattes (hair, skin, teeth, glasses), which shows edges are handled by mattes, not by depth alone. [Public]
- **Photoshop, classic Lens Blur:** depth map, iris (blade count, curvature, rotation), specular highlight Brightness/Threshold, and **Noise**, added to put grain back. [Public]
- **Lightroom / ACR Lens Blur (2023+):** AI depth, bokeh styles including cat's eye, highlight "Boost", focal range, and a refine brush for the depth. [Public]
- The common pattern: depth-ordered compositing plus high-resolution subject mattes plus grain. [Inferred]

---

## Q3: sampling

**Brute force is out.** Spacing of 2 px at R = 140 needs about 15,400 taps. At 24 MP that is about 7×10¹¹ fetches (colour + depth). That means minutes in WebGL, and the GPU watchdog will kill it. [Inferred]

**Recommendation: dense taps at a level matched to the blur radius.**
1. Build a **far-only premultiplied pyramid** of (rgb·m, m, coc·m), where m is the far mask. This keeps subject colours out of the background bokeh.
2. For a destination with radius R, sample at level L = max(0, ceil(log2(R/24))), so the radius at that level is about 24 texels.
3. Use **N = 384** golden-spiral taps. Spacing at L = 24·√(π/384) ≈ 2.2 texels. That is properly sampled, with bilinear filtering doing the rest.
4. Compute destinations with large R at reduced resolution as well (the same L). Merge with the full-resolution small-R result using a blend guided by the blur radius, not plain bilinear filtering.
5. Cost on export: about 375k destinations × 384 taps per coarse level. That is a few times 10⁸ fetches. Tile the export into 1024² blocks with an apron of R_max to avoid losing the GPU context. [Public, browser/OS GPU watchdogs]

**Does prefiltering flatten discs?** Not if it is done this way.
- A mean-preserving prefilter keeps a highlight's energy, so disc brightness is correct. That is more correct than sparse mip-0 taps, which miss about 99 % of hits.
- Edges soften by about one texel at level L, about 4 % of R, which is below what most scenes show. [Inferred]
- What softened and haloed the old discs:
  - sparse taps on a prefilter at the tap spacing (edge softening ≈ spacing ≈ 13 % of R);
  - unmasked mips mixing depths;
  - an 8-bit clipped source with no expansion. [Inferred]

**Progressive accumulation:**
- **Preview:** yes. 8 × 96 taps reaches 31 % coverage at R = 28. Show frame 1 immediately, then refine while the user is idle.
- **Export:** not on its own (768 taps only reaches 1.2 % coverage at R = 140). Use the multi-resolution scheme.
- **Determinism:**
  - frame k uses rotation `fract(ign(p) + k·0.6180339)` and a sub-pixel offset from the R2 sequence;
  - fixed K;
  - no time-based seed;
  - accumulate in RGBA32F with ping-pong buffers. Float blending requires `EXT_float_blend`. [Public]
  - The output is bit-identical on the same GPU and driver, not across GPUs.

---

## Q4: bokeh appearance

Ranked by what an eye reads as "fast lens":

1. **A clean, energy-correct disc.** Everything in Q2 and Q3. This is most of the perceived quality.
2. **Cat's eye** (mechanical vignetting). The disc becomes the intersection of the pupil with an offset circle of a different radius, elongated tangentially. It is strongest wide open and fades as the lens stops down. [Public] This is the most recognisable feature after the disc itself.
3. **Aperture shape tied to the f-stop.** Circular wide open (many lenses with rounded blades stay round for the first 1–2 stops), then a rounded polygon. Flip it 180° for the near field. [Public, rounded-blade designs]
4. **Spherical-aberration profile.** A bright rim on one side of focus and a soft centre on the other. Cheap: a radial weight that depends on the sign of defocus. [Public]
5. **Longitudinal chromatic aberration** (magenta/green fringes in front of and behind focus). Optional, off by default. It needs a blur radius per channel and is often read as a defect.
6. **Skip:** onion rings, focus breathing (meaningless for a still image), swirl.

How to stay energy-correct and avoid the frame bug. Evaluate everything in the **source** frame and normalise by the shape's area:

```glsl
vec2 q = p_px - s_px;                      // destination relative to source
if (sourceIsNear) q = -q;                  // near-field PSF is point-reflected
float lim = shapeRadius(atan(q.y,q.x), srad, fieldPos(suv), u_fstop);
float inside = clamp(lim - length(q) + 0.5, 0.0, 1.0);
float w = inside / shapeArea(srad, fieldPos(suv), u_fstop);  // LUT for cat's-eye area
```

- Polygon area: (n/2)·R²·sin(2π/n), with R the circumradius.
- Cat's-eye area: a small precomputed LUT indexed by (field radius, f-stop).
- Normalising this way keeps synthetic vignetting from darkening the blur. The source photo already contains its own vignetting. [Inferred]

---

## Q5: depth quality

**Where the artifacts will be worst** [Inferred]:
1. **Hair, fur, foliage, glasses frames, spokes, fences.** Anything thinner than about 5 px at full resolution: 1280 → 6000 px is a 4.7× upsample.
2. **Silhouette ramps** from the guided filter. These are mixed depths by construction (see §6, point 6).
3. **Glass, mirrors, water.** One depth per pixel is wrong for these.
4. **Floating objects or bent planes** around the subject.

**Where errors barely matter.**
- **Far depth.** With focus at 2 m, the blur is proportional to |1/z − 0.5|: 0.4 at 10 m and 0.5 at infinity. Far errors and sky handling hardly show unless the focus is far.
- **Uniform metric scale error.** It acts like an aperture error: 15 % of scale error ≈ 0.2 stop, provided focus is set by clicking. [Inferred]
- **Inside the subject, depth does matter at long focal lengths.** At 85 mm f/1.8 and 2 m, the 10 cm from nose to ear gives an 18 px blur diameter at full resolution.

**Most realism per unit of effort:**
1. **Snap depth edges.** Replace the guided-filter output at depth discontinuities with a joint weighted median, or "snap to the local min/max of 1/z according to colour affinity". Fetch depth with `texelFetch` for blur lookups. [Public, weighted-median upsampling] Effort S–M, large gain.
2. **Lean towards the foreground** at uncertain edges, by about 1 full-resolution pixel. A thin sharp strip of background next to a sharp subject reads as natural. Subject colour blurred into the bokeh reads as a glow halo. [Inferred]
3. **Subject matte plus foreground colour estimation**, with a high-resolution matting or segmentation network (BiRefNet or ViTMatte class [Public, both exist]). Render α·F_sharp + (1−α)·B_blurred, with B inpainted behind the subject. This is the only real fix for mixed pixels in hair, and it is what Apple's mattes point to. Effort L, the largest gain for portraits.
4. **Check the network's native input size.** The original Depth Pro was designed for 1536 px input. [Public] Running at 1280 may cost edge detail. "Depth Pro 2 H+" is unknown here: check its native resolution.
5. **Low value here:** confidence estimation, multi-scale fusion (BoostingMonocularDepth-style [Public]), sky refinement.

**Is depth the biggest gap?** After the Q2/Q3 renderer fixes, yes for portraits and close subjects: the subject silhouette dominates. Before those fixes, no: the renderer is the gap. [Inferred]

**The source photo, two cheap things:**
- **Grain.** Blurred areas lose the sensor noise and look synthetic. Estimate the noise level (MAD of a high-pass on flat areas, as a function of luminance) and put matching noise back, weighted by blur radius. Photoshop's Lens Blur has offered this for years. [Public] Effort S, very visible at 1:1.
- **Blur already in the photo** (DSLR sources). Use a target blur radius of about √(c² − c_native²) when EXIF gives the aperture and the focus distance. [Inferred]

---

## Q6: highlights and HDR

**Principle.** An unclipped pixel's linear value is the truth (up to the camera's tone curve). Only clipped pixels hide energy. Expansion must therefore be gated on **clipping**, not on contrast.

**Is the old `hdrExpand()` sound?**
- **The right part:** "isolated against its surroundings" is a good proxy for a point source.
- **The wrong parts:**
  - it gates on local contrast, so it boosts unclipped white text, shirt edges and catchlights;
  - a slope of ×12 is arbitrary;
  - if applied before the gather, it changes in-focus pixels that pass through the gather.

**Recommended replacement** (applied only to the copy the blur reads from):

```glsl
float m = max(c.r, max(c.g, c.b));
float clipped = smoothstep(0.90, 0.995, m);
float iso = 1.0 - smoothstep(0.15, 0.6, clipFrac(uv, 5));  // mip-5 of clip mask: small blobs only
c *= exp2(u_hlStops * clipped * iso);                       // skies/windows (clipFrac≈1) untouched
```

- Keep the sharp base unexpanded.
- On the blurred output, clip, or apply a gentle shoulder above about 0.9. Camera JPEGs clip strong bokeh to white too, so a white core with a coloured fringe is realistic.
- Take the hue of the clipped blob from its unclipped ring.

**The slider.** Keep it, but redefine it as **"Clipped highlight strength, 0–5 stops"**. Delete the weight bias: its threshold (max channel above 0.55 linear, about 0.77 sRGB) catches skin, sky and white shirts, and it is not energy-conserving.

**Real data instead of a guess.** Ultra HDR JPEGs (JPEG_R) and gain-map images (ISO 21496-1, Apple HDR photos) carry real highlight headroom. Decode them on the server, for example with libultrahdr. [Public] RAW/DNG input would be the real fix.

---

## Q7: the WIP branch

**What is correct:**
- `u_scale *= zoom`: a blur radius in screen pixels = photo pixels × screen pixels per photo pixel. ✓
- Tap offsets `/ (u_res * u_zoom)`. ✓ This only holds if `u_res` is the photo's displayed size, not a letterboxed canvas.

**Concrete bugs:**
1. `float suv = clamp(vec2…)`: type error, it won't compile (unless this is a transcription slip). `x`/`y` are undefined in the excerpt.
2. **`u_maxblur` is not scaled by zoom.** At zoom 4 the cap drops to 2.5 % of the photo, so the zoomed inspection shows less blur than the export. Define the cap in photo pixels: `0.10 · photoShort · u_scale · zoom`, and fix the ½ diameter/radius bug at the same time.
3. **A ±1.5-texel cross is not a dilation.** The foreground spills into a band 1–2 px wide and then stops, which gives a thin ring instead of an R-wide transition.
4. **The far-side max has no depth test.** A farther, blurrier neighbour also enlarges the radius, so background bleeds onto midground edges.
5. **Near-side `w = 1`:**
   - sharp content halos into near blur on continuous surfaces;
   - the background gets smeared into the foreground rim;
   - the foreground looks eroded.
6. Fixed 96 taps under zoom: still sparser sampling.
7. **Mip 0 at zoom 1** (pre-existing preview aliasing).
8. Check that export resets the **pan** as well as the zoom, otherwise it renders a shifted crop. Not verifiable from the excerpt.

**Before merging:**
1. Split the branch. Merge zoom/pan with fixes 1, 2 and 7.
2. Add a browser test: zoomed-preview crop vs the same export crop, PSNR above a threshold.
3. Drop the gather-rule change and replace it with the Q2 near field.

---

## Ranked table

| # | Item | Realism gain (1–5) | Risk (1–5) | Effort | Confidence |
|---|---|---|---|---|---|
| 1 | Near field with coverage alpha + tile-max dilation + depth test; drop WIP `w=1` | 5 | 3 | M | High |
| 2 | Depth-aware multi-resolution sampling (far-only premultiplied pyramid, L = log2(R/24), 384 taps, low-res compute for large R) | 5 (export) | 2 | M | High |
| 3 | Snapped depth edges + `texelFetch` depth for the gather | 4 | 2 | S–M | Med–High |
| 4 | Clipped-highlight expansion (stops slider), delete weight bias; gain maps when present | 3–4 | 2 | S–M | Medium |
| 5 | Matched grain in blurred areas | 3–4 (1:1) | 1 | S | High |
| 6 | Fix radius cap (×½ bug) + scale it with zoom | 2–3 | 1 | S | High |
| 7 | Subject matte + foreground colour estimation (hair) | 5 (portraits) | 2 | L | Medium |
| 8 | Preview LOD = log2(1/u_scale) (aliasing) | 2 (preview) | 1 | S | High |
| 9 | Progressive 8×96 preview accumulation, R2 seeds | 2 (preview) | 1 | S | High |
| 10 | Continuous blur onset (remove 0.55 early return) | 1 | 1 | S | High |
| 11 | Cat's eye in source frame, area-normalised, unequal circles | 2 | 2 | S–M | Medium |
| 12 | Aperture shape by f-stop (round → rounded polygon), near flipped 180° | 1–2 | 2 | S | Medium |
| 13 | Subtract native blur (DSLR sources) | 2 (those sources) | 2 | S | Medium |
| 14 | Check the depth model's native input size | 2–3 | 1 | S | Low |
| 15 | Spherical-aberration rim / soft profile by defocus sign | 1–2 | 2 | S | Medium |
| 16 | K-slice compositing (if near-over-near overlaps are common) | 4 | 3 | L | Medium |
| 17 | Longitudinal chromatic aberration | 1 | 3 | M | Medium |
| 18 | Onion rings, focus breathing, swirl | 0–1 | 2 | M | High (skip) |

**Direct answer:** fix items 1–6 and 8–10, which are all renderer corrections, not features. After that, stop adding lens features: the remaining gap is the subject's silhouette (item 7, depth edges) and the source image (8-bit clipping, noise, blur already in the photo).
