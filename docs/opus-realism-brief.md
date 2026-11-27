# Depth of Field Studio — how to make the synthetic DOF as realistic as possible

Answer from the text below only. Do not use tools, do not read or clone anything, do not run commands. Answer in English. Tag claims you know from public documentation/established practice as [Public], and reasoning or guesses as [Inferred]. All the code you need is inline.

## 1. What the app is

A local web app: a photo is uploaded once, a monocular metric-depth network (Depth Pro 2 H+, 872 M params) runs on a CUDA GPU at 1280 px on the long side, and the browser renders a synthetic depth of field in WebGL2. The user picks a focus distance in metres and an aperture from f/1.0 to f/22, plus an optional simulated focal length (native / 50 / 85 / 135 mm). The render is a single-pass, 96-tap (preview) or 192-tap (export) golden-angle spiral gather in linear light; export is full resolution up to 8192 px. It is a still-image app: a render may take hundreds of milliseconds, and there is no animation requirement.

Server side, depth is:
- inferred at 1280 px (retries 1024, 768 on OOM);
- converted to true inverse depth in 1/m (`inv = f_px/width * 1/z`, with the network's focal/width normalisation divided out);
- upsampled to full resolution with a colour-guided filter (He et al., radius = 0.75 × the upsample factor, so ~2 px edges at 24 MP);
- stored as float16 1/z, sky exactly 0 (which the shader treats as infinite distance).

Camera intrinsics: `f_px` is the focal length in pixels of the full-resolution photo (from EXIF when present, otherwise estimated by the network), and `focal35` is the 35 mm-equivalent focal length using the sensor diagonal.

## 2. Active fragment shader (this is the whole renderer)

```glsl
#version 300 es
precision highp float;
precision highp sampler2D;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_photo;     // linear-light RGBA16F photo, mips generated
uniform sampler2D u_depth;     // R16F inverse depth (1/m), no mips
uniform float u_fpxNative;     // native focal length in pixels (full-res photo)
uniform float u_fmm;           // simulated focal length, mm
uniform float u_fstop;
uniform float u_focus;         // metres
uniform float u_scale;         // canvas.width / photo.width (CoC px scaling)
uniform float u_blur;          // artistic multiplier, default 1
uniform float u_maxblur;       // 0.10 * min(canvas.width, canvas.height), px radius
uniform float u_highlight;     // 0..1, default 0.15
uniform int u_mode;            // 0 = DOF, 1 = depth map, 2 = original
uniform int u_linear;
uniform int u_taps;            // 96 preview, 192 export
uniform vec2 u_res;            // canvas pixels

const int MAXTAPS = 192;
const float GOLDEN = 2.399963229728653;

vec3 linearToSrgb(vec3 c) { ... }   // standard, not clamped
vec3 encode(vec3 c) { return u_linear == 1 ? linearToSrgb(c) : c; }

float ign(vec2 p) { return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }

// d = inverse depth (1/m). Returns CoC DIAMETER in target pixels.
float cocDiameter(float d) {
  float s = clamp(u_focus, 0.05, 4000.0);
  float smm = s * 1000.0;
  float mag = smm / max(smm - u_fmm, 1.0);          // s / (s - f), exact thin lens
  float c = (u_fpxNative * u_fmm / max(u_fstop, 0.7))
          * abs(d - 1.0 / s) * 0.001 * mag;
  return c * u_blur * u_scale;
}

void main() {
  vec3 base = textureLod(u_photo, v_uv, 0.0).rgb;
  if (u_mode == 2) { outColor = vec4(encode(base), 1.0); return; }

  float d = textureLod(u_depth, v_uv, 0.0).r;
  if (u_mode == 1) { /* depth-map view, not relevant here */ }

  float radius = 0.5 * min(cocDiameter(d), u_maxblur);
  if (radius < 0.55) { outColor = vec4(encode(base), 1.0); return; }   // hard sharp/blur cutoff

  // Single-pass scatter-as-gather. A tap only contributes when the tap's own
  // blur disc reaches this pixel, so a sharp foreground cannot bleed outward.
  float spin = ign(gl_FragCoord.xy) * 6.2831853;
  vec3 sum = base;
  float wsum = 1.0;

  for (int i = 0; i < MAXTAPS; i++) {
    if (i >= u_taps) break;
    float fi = float(i) + 0.5;
    float r = sqrt(fi / float(u_taps)) * radius;      // uniform in the disc
    float a = fi * GOLDEN + spin;
    vec2 suv = clamp(v_uv + vec2(cos(a), sin(a)) * r / u_res, 0.0, 1.0);

    vec3 sc = textureLod(u_photo, suv, 0.0).rgb;
    float sd = textureLod(u_depth, suv, 0.0).r;
    float srad = 0.5 * min(cocDiameter(sd), u_maxblur);
    float w = smoothstep(r - 1.5, r + 1.5, srad);

    // Specular emphasis: bright taps get more weight.
    w *= 1.0 + u_highlight * smoothstep(0.55, 0.95, max(sc.r, max(sc.g, sc.b)));

    sum += sc * w;
    wsum += w;
  }
  outColor = vec4(encode(sum / wsum), 1.0);
}
```

JS-side facts that matter:
- Preview canvas is typically ~1200×800 CSS px at DPR up to 2, so a 24 MP photo is displayed at `u_scale ≈ 0.2`. Export sets the canvas to the photo size (capped 8192) and `u_scale ≈ 1`.
- `u_maxblur` is 10% of the *short side of the render target*, in pixels: ~80 px radius on a 1200×800 preview, ~400 px on a 4000 px-tall export.
- `u_blur` is an artistic multiplier; the default is 1.0.
- The photo texture has a full mip pyramid; the depth texture does not.
- There is no temporal accumulation; each render is one frame.

## 3. Dead code that used to exist (why the current renderer looks the way it does)

A previous version rendered in two layers (near/far split at the focus plane) with a mip-filled far layer and a 9-blade polygon aperture plus mechanical cat's-eye vignetting. It was removed because it "flattened bokeh discs and haloed subject edges". The dead shaders are still in the file but never called. Two details from them are relevant:

```glsl
float polygonLimit(float angle, float radius) {
  const float blades = 9.0;
  float sector = TAU / blades;
  float local = mod(angle + 0.17 + sector * 0.5, sector) - sector * 0.5;
  return radius * cos(PI / blades) / max(cos(local), 0.20);
}
// In the dead blur loop:
float angle = fi * GOLDEN + spin;               // spin is per-pixel random
float limit = polygonLimit(angle, radius);      // so the polygon ROTATES per pixel
...
// Mechanical vignetting: two offset circles -> almond (cat's eye)
vec2 field = (uv - 0.5) * 2.0;  field.x *= u_res.x / max(u_res.y, 1.0);
float wide = 1.0 - smoothstep(1.6, 8.0, u_fstop);
vec2 ap = vec2(cos(angle), sin(angle)) * (rpx / max(radius, 1e-3));
float vig = 0.55 * wide * smoothstep(0.2, 1.2, length(field));
vec2 radial = normalize(field + vec2(1e-4));
float sgn = u_layer == 0 ? 1.0 : -1.0;          // near/far clip opposite rims
float d1 = length(ap - sgn * vig * radial);
float d2 = length(ap + sgn * vig * radial);
float catW = smoothstep(1.0, 0.92, max(d1, d2));
```

The dead layer shader also had an HDR highlight expansion: `hdrExpand()` compared each pixel's luminance with its own mip-5 blur, and boosted `1 + u_highlight * 12 * smoothstep(0.02, 0.30, l - bg) * spread` in linear light. That is gone from the active path; the active path only has the weight bias shown above.

## 4. What I measured / computed (so you can check the scale)

CoC diameter in full-resolution pixels, exact thin-lens formula, focus 2.0 m, 24 MP (6000×4000):

- 50 mm f/1.8: subject at 1.0 m → 119 px; 1.5 m → 40 px; 5 m → 71 px; 10 m → 95 px; infinity → 116 px.
- 85 mm f/1.8: 1.0 m → 349 px; 1.5 m → 116 px; 5 m → 210 px; 10 m → 279 px.
- 135 mm f/2.8: 1.0 m → 582 px; 5 m → 349 px; 10 m → 465 px.
- 35 mm f/1.4: 1.0 m → 74 px; 10 m → 59 px.

Tap spacing of a Vogel spiral is about `radius * sqrt(pi / taps)`:
- 96 taps at 28 px radius (the 85 mm/10 m case on a 1200 px preview) → ~5 px between taps.
- 192 taps at 140 px radius (same case on a full-res export) → ~18 px between taps.
- To get 2 px spacing at 140 px radius you need ~15,000 taps, i.e. far more than one pass.

The active gather always samples mip 0, so each tap is one source pixel, not a prefiltered average. The dead blur shader used `lod = clamp(log2(max(1.0, radius * sqrt(PI / taps))) - 0.5, 0, maxLod)`, i.e. it deliberately prefiltered to the tap spacing.

## 5. A WIP branch that may or may not help (review this too)

Branch `wip/zoom-pan-and-dof-edges` (no PR; one commit, plus browser tests) adds wheel zoom / drag pan and changes the gather rule. The changed lines:

```glsl
// Max-CoC over a small cross, but only on the far side of focus.
vec2 dtx = 1.5 / vec2(textureSize(u_depth, 0));
float rNear = max(max(coc(puv + x), coc(puv - x)), max(coc(puv + y), coc(puv - y)));
float ownCoc = cocDiameter(d);
float dFocus = 1.0 / clamp(u_focus, 0.05, 4000.0);
float cocPx = d < dFocus ? max(ownCoc, rNear) : ownCoc;   // d < dFocus = FARTHER than focus
float radius = 0.5 * min(cocPx, u_maxblur);
...
float suv = clamp(puv + vec2(cos(a), sin(a)) * r / (u_res * u_zoom), 0.0, 1.0);
...
float w = d > dFocus ? 1.0 : smoothstep(r - 1.5, r + 1.5, srad);
```

The branch comment says the strict rule left defocused foreground objects with a "hard, cut-out silhouette", so on the near side of focus (d > dFocus, i.e. closer than the focus plane) every tap gets weight 1.0. It also scales CoC by `u_zoom` for zoomed inspection, and resets zoom to 1 on export. Is this physically the right trade, or does it trade a cut-out for background bleeding into foreground edges? What would you do instead?

## 6. What I suspect, and want you to confirm or refute

These are my own static reads; mark each true/false and say whether it matters for realism:

1. The active renderer has no polygon aperture, no cat's-eye vignetting, and no HDR highlight reconstruction; the README still advertises a "crisp polygon aperture". The polygon code that exists rotates the polygon per pixel (because `polygonLimit` is evaluated on an angle that includes the per-pixel `spin`), so even if re-enabled as-is it would smear the 9-gon into a noisy disc.
2. The single-pass rule ("a tap counts only if its own CoC reaches this pixel") is actually the correct scatter-as-gather condition, and the WIP's near-side `w = 1.0` is the physically wrong direction: it lets background samples bleed over a near object's edge. If true, the cut-out look must come from something else (depth-edge interpolation, the 0.55 px cutoff, or the depth map's 2 px silhouette), not from the weight rule.
3. The biggest realism loss at large apertures is undersampling: 96/192 mip-0 taps over a 40–280 px CoC disc is a few percent of the disc area, so textured backgrounds will show spiral/stipple noise rather than a smooth optical blur. Prefiltering (mip LOD matched to tap spacing) or progressive multi-frame accumulation would fix more than adding taps.
4. The hard `radius < 0.55 → sharp` branch creates a visible line where blur switches on; a real lens transitions continuously through the depth of field.
5. The weight-bias "highlight bokeh" is not energy-conserving and makes bright taps brighter than the physical average; clipping at 1.0 in linear light means bright speculars become flat white discs instead of HDR discs that roll off in the tonemap.
6. Bilinear interpolation of inverse depth at a silhouette produces a one-pixel ramp of intermediate CoC at every depth edge (the depth-map *view* avoids this by interpolating metres, but the DOF gather does not).
7. The CoC formula is exact thin-lens and the simulated-lens scaling is intentional: holding framing fixed while increasing focal length is equivalent to scaling the sensor, so CoC is linear in the simulated focal length, with the native focal in pixels as the px/mm conversion. I derived `c_px = f_px_native * f_mm/N * |1/z - 1/s| * 0.001 * s/(s-f_mm)`; check the derivation.

## 7. Questions — answer each, then give one ranked table

1. **Rank the realism improvements** by (realism gain, risk of new artifacts, implementation effort) for this specific app. Concrete shader changes, not principles. Include "do nothing" if the honest answer is that the renderer is near its practical ceiling.
2. **Occlusion and edges.** Is the current scatter-as-gather rule correct? What actually causes the "cut-out silhouette" the WIP author saw, and is the WIP's near-side uniform weighting an improvement or a halo generator? For a still app with a few hundred ms budget, what is the right architecture: layered alpha, N-layer gather, scatter/splat, hybrid, or something else? Name the specific algorithm and its failure modes, and say what production still-photo tools (iPhone Portrait, Google Camera, Photoshop Lens Blur, etc.) actually do, as far as public documentation goes.
3. **Sampling.** Quantify the undersampling; what tap count / pattern / LOD / accumulation scheme would you use? Is progressive accumulation over several idle frames (e.g. 8 × jittered 96 taps into a float buffer) worth it for preview and export, and how would you keep it deterministic for export? Would mip prefiltering flatten bokeh discs the way the old layered pipeline did, and how do you avoid that?
4. **Bokeh appearance.** What should a physically plausible fast-lens PSF add beyond a uniform disc: polygon blades, rounded blade tips, cat's eye, onion ring, spherical-aberration bright rim, longitudinal chromatic aberration, focus breathing? Which are worth implementing here, in what order, and how do you keep energy conservation and avoid the per-pixel polygon-rotation bug?
5. **Depth quality.** Given a 1280 px monocular depth map upsampled with a guided filter, where will the remaining artifacts be worst, and what depth post-processing gives the most realism per effort (edge matting, foreground dilation, confidence, multi-scale fusion, sky handling, mixed pixels)? Is the app's biggest remaining gap actually the depth map rather than the renderer?
6. **Highlights / HDR.** With an 8-bit sRGB source, what is the physically sensible way to render bokeh from clipped speculars? Is the dead `hdrExpand()` approach sound? Should the highlight slider exist at all, and if so what should it control?
7. **The WIP branch.** Review the zoom/pan + near-side rule. Is the zoom math correct (`u_scale *= zoom`, tap offsets divided by `u_zoom`, `u_maxblur` still capped at 10% of the viewport)? Any concrete bugs? Is the near-side rule right? What would you change before merging?

Finish with a ranked table: item | realism gain (1–5) | risk (1–5) | effort (S/M/L) | confidence. And answer explicitly: if the honest answer is "stop adding renderer features; the remaining gap is depth capture / source-image quality", say that instead of padding the list.
