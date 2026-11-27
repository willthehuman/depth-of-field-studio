/* Depth of Field Studio — client.
   Depth is estimated once on the server; focus and aperture are a layered
   WebGL bokeh pipeline, so the sliders are instant and the export is full-res.
   The gather runs in linear light: the photo is decoded from sRGB into a
   half-float texture first (see buildLinearTexture), near-clipped highlights
   are expanded back above 1.0 so bokeh discs carry real energy, and the
   result is encoded back to sRGB on the way out. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const FSTOPS = [1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 2.8, 3.2, 3.5, 4.0, 4.5, 5.0,
    5.6, 6.3, 7.1, 8.0, 9.0, 10, 11, 13, 14, 16, 18, 20, 22];
  const PREVIEW_TAPS = 96;
  const EXPORT_TAPS = 192;
  const COC_LIMIT = 0.03; // 30 µm, the full-frame circle of confusion

  const S = {
    meta: null,
    bitmap: null,
    mode: 0,
    holdOriginal: false,
    fstopIdx: 6,
    focus: 4,
    dLo: 0.3,
    dHi: 40,
    blur: 1,
    highlight: 0.4,
    grain: 0.30,
    lensMM: 0,
    zoom: 1,
    cx: 0.5,
    cy: 0.5,
    drag: null,
    suppressClick: false,
    format: "jpeg",
    quality: 95,
    depthLow: null,
    lowW: 0,
    lowH: 0,
    depthW: 0,
    depthH: 0,
    gl: null,
    progPre: null,
    progFinal: null,
    progLayer: null,
    progBlur: null,
    locPre: {},
    locFinal: {},
    locLayer: {},
    locBlur: {},
    fbo: null,
    linearOK: false,
    colorFloat: false,
    maxExport: 8192,
    tex: { photo: null, depth: null, linear: null },
    targets: null,
    dirty: false,
    busy: false,
  };

  /* --------------------------------------------------------------- shaders */

  const VERT = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = vec2(a_pos.x * 0.5 + 0.5, 0.5 - a_pos.y * 0.5);
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

  // Pre-pass: decode the sRGB photo into a linear-light texture, once per
  // photo, and build the source mip pyramid in linear space. Alpha carries the
  // clipping mask: 1 where the source is at or above display white. The
  // highlight expansion reads it later, and its mip pyramid gives the local
  // clip fraction that separates a small specular from a white sky.
  const PRE_FRAG = `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_src;

vec3 srgbToLinear(vec3 c) {
  vec3 lo = c / 12.92;
  vec3 hi = pow((c + 0.055) / 1.055, vec3(2.4));
  return mix(lo, hi, step(vec3(0.04045), c));
}

void main() {
  // Framebuffer rows are bottom-up while v_uv is top-down for the on-screen
  // pass. Flip here so this texture has the same orientation as the photo.
  vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
  vec3 c = srgbToLinear(texture(u_src, uv).rgb);
  float clipped = smoothstep(0.90, 0.995, max(c.r, max(c.g, c.b)));
  outColor = vec4(c, clipped);
}`;

  // The final pass doubles as the original/depth preview. In DOF mode the
  // near field carries coverage alpha and composites over the far field, which
  // is filled behind the near objects. Grain is added in linear light where the
  // defocus is strong enough to have erased the source noise.
  const FINAL_FRAG = `#version 300 es
precision highp float;
precision highp sampler2D;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_photo;
uniform sampler2D u_depth;
uniform sampler2D u_near;
uniform sampler2D u_far;
uniform vec2 u_depthSize;
uniform float u_fpxNative;
uniform float u_fmm;
uniform float u_fstop;
uniform float u_focus;
uniform float u_scale;
uniform float u_blur;
uniform float u_maxRadius;
uniform float u_grain;
uniform float u_dnear;
uniform float u_dfar;
uniform int u_mode;
uniform int u_linear;
uniform vec2 u_res;
uniform float u_zoom;
uniform vec2 u_center;

const float PI = 3.14159265359;

vec3 linearToSrgb(vec3 c) {
  c = max(c, vec3(0.0));
  vec3 lo = c * 12.92;
  vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
  return mix(lo, hi, step(vec3(0.0031308), c));
}

vec3 encode(vec3 c) {
  return u_linear == 1 ? linearToSrgb(c) : c;
}

float cocDiameter(float d) {
  float s = clamp(u_focus, 0.05, 4000.0);
  float smm = s * 1000.0;
  float mag = smm / max(smm - u_fmm, 1.0);
  float c = (u_fpxNative * u_fmm / max(u_fstop, 0.7))
          * abs(d - 1.0 / s) * 0.001 * mag;
  return c * u_blur * u_scale;
}

vec2 zoomUV(vec2 uv) {
  return clamp(u_center + (uv - 0.5) / u_zoom, vec2(0.0), vec2(1.0));
}

float baseLod() {
  return max(0.0, log2(1.0 / max(u_scale, 1e-4)));
}

// Depth lookup with a 3x3 nearest-colour snap. The guided filter leaves a
// 2-4 px ramp of mixed 1/z at a silhouette; bilinear reads of that ramp paint
// a sharp outline when it crosses the focus plane. Take the neighbour whose
// colour is closest to the centre instead, and break ties towards the foreground.
float depthAt(vec2 puv) {
  ivec2 p = ivec2(clamp(puv, 0.0, 1.0) * u_depthSize);
  p = clamp(p, ivec2(0), ivec2(u_depthSize) - 1);
  vec3 c0 = textureLod(u_photo, (vec2(p) + 0.5) / u_depthSize, 0.0).rgb;
  float bestD = texelFetch(u_depth, p, 0).r;
  float bestDist = 1e9;
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      ivec2 q = clamp(p + ivec2(dx, dy), ivec2(0), ivec2(u_depthSize) - 1);
      vec3 c = textureLod(u_photo, (vec2(q) + 0.5) / u_depthSize, 0.0).rgb;
      float dist = dot(c0 - c, c0 - c);
      float d = texelFetch(u_depth, q, 0).r;
      if (dist < bestDist - 1e-7 || (abs(dist - bestDist) <= 1e-7 && d > bestD)) {
        bestDist = dist;
        bestD = d;
      }
    }
  }
  return bestD;
}

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

vec3 depthRamp(float t) {
  const vec3 c0 = vec3(1.00, 0.30, 0.16);
  const vec3 c1 = vec3(1.00, 0.70, 0.20);
  const vec3 c2 = vec3(0.98, 0.95, 0.62);
  const vec3 c3 = vec3(0.36, 0.86, 0.86);
  const vec3 c4 = vec3(0.24, 0.45, 0.95);
  t = clamp(t, 0.0, 1.0);
  float s = t * 4.0;
  if (s < 1.0) return mix(c0, c1, s);
  if (s < 2.0) return mix(c1, c2, s - 1.0);
  if (s < 3.0) return mix(c2, c3, s - 2.0);
  return mix(c3, c4, s - 3.0);
}

void main() {
  vec2 puv = zoomUV(v_uv);
  vec3 base = textureLod(u_photo, puv, baseLod()).rgb;
  if (u_mode == 2) {
    outColor = vec4(encode(base), 1.0);
    return;
  }

  if (u_mode == 1) {
    // The depth map is a data view: interpolate METRES, not disparity.
    vec2 dsz = u_depthSize;
    vec2 tuv = puv * dsz - 0.5;
    ivec2 i0 = ivec2(floor(tuv));
    vec2 f = clamp(tuv - vec2(i0), 0.0, 1.0);
    ivec2 i1 = min(i0 + 1, ivec2(dsz) - 1);
    i0 = max(i0, ivec2(0));
    float z00 = 1.0 / max(texelFetch(u_depth, i0, 0).r, 1e-6);
    float z10 = 1.0 / max(texelFetch(u_depth, ivec2(i1.x, i0.y), 0).r, 1e-6);
    float z01 = 1.0 / max(texelFetch(u_depth, ivec2(i0.x, i1.y), 0).r, 1e-6);
    float z11 = 1.0 / max(texelFetch(u_depth, i1, 0).r, 1e-6);
    float z = mix(mix(z00, z10, f.x), mix(z01, z11, f.x), f.y);
    float lz = log(max(z, 1e-3));
    float t = (lz - log(u_dnear)) / max(log(u_dfar) - log(u_dnear), 1e-4);
    outColor = vec4(depthRamp(clamp(t, 0.0, 1.0)), 1.0);
    return;
  }

  float d = depthAt(puv);
  vec4 n = texture(u_near, v_uv);
  vec4 f = texture(u_far, v_uv);
  float a = clamp(n.a, 0.0, 1.0);
  vec3 col = n.rgb * a + f.rgb * (1.0 - a);

  // Grain: sensor noise is added after the lens, so it stays sharp in the
  // source. The synthetic blur averages it out; put a matching amount back
  // only where the blur is strong, with a shadow-weighted amplitude.
  float radius = 0.5 * min(cocDiameter(d), u_maxRadius);
  float grain = smoothstep(0.75, 4.0, radius) * u_grain;
  float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
  float n1 = hash12(gl_FragCoord.xy);
  float n2 = hash12(gl_FragCoord.xy + vec2(17.0, 31.0));
  col += vec3((n1 - 0.5), (n1 - 0.5) * 0.7, (n2 - 0.5)) * grain
       * (0.35 + 0.65 * (1.0 - clamp(luma, 0.0, 1.0)));

  outColor = vec4(encode(col), 1.0);
}`;

  // Split the photo into premultiplied near/far layers. RGB is filtered as
  // coverage-weighted colour and A is coverage. The second output is
  // radius*coverage, which becomes the layer's CoC pyramid. The layer is the
  // blur source, so clipped highlights are expanded here, gated on clipping and
  // isolation and faded in with the blur radius; the sharp base stays unexpanded.
  const LAYER_FRAG = `#version 300 es
precision highp float;
precision highp sampler2D;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_photo;
uniform sampler2D u_depth;
uniform vec2 u_depthSize;
uniform float u_fpxNative;
uniform float u_fmm;
uniform float u_fstop;
uniform float u_focus;
uniform float u_scale;
uniform float u_blur;
uniform float u_maxRadius;
uniform float u_hlStops;
uniform float u_infoScale;
uniform int u_layer;
uniform int u_info;
uniform int u_linear;
uniform float u_zoom;
uniform vec2 u_center;

vec2 zoomUV(vec2 uv) {
  return clamp(u_center + (uv - 0.5) / u_zoom, vec2(0.0), vec2(1.0));
}

// Minified previews must read a prefiltered level of the photo, not mip 0.
float baseLod() {
  return max(0.0, log2(1.0 / max(u_scale, 1e-4)));
}

float cocDiameter(float d) {
  float s = clamp(u_focus, 0.05, 4000.0);
  float smm = s * 1000.0;
  float mag = smm / max(smm - u_fmm, 1.0);
  float c = (u_fpxNative * u_fmm / max(u_fstop, 0.7))
          * abs(d - 1.0 / s) * 0.001 * mag;
  return c * u_blur * u_scale;
}

float depthAt(vec2 puv) {
  ivec2 p = ivec2(clamp(puv, 0.0, 1.0) * u_depthSize);
  p = clamp(p, ivec2(0), ivec2(u_depthSize) - 1);
  vec3 c0 = textureLod(u_photo, (vec2(p) + 0.5) / u_depthSize, 0.0).rgb;
  float bestD = texelFetch(u_depth, p, 0).r;
  float bestDist = 1e9;
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      ivec2 q = clamp(p + ivec2(dx, dy), ivec2(0), ivec2(u_depthSize) - 1);
      vec3 c = textureLod(u_photo, (vec2(q) + 0.5) / u_depthSize, 0.0).rgb;
      float dist = dot(c0 - c, c0 - c);
      float d = texelFetch(u_depth, q, 0).r;
      if (dist < bestDist - 1e-7 || (abs(dist - bestDist) <= 1e-7 && d > bestD)) {
        bestDist = dist;
        bestD = d;
      }
    }
  }
  return bestD;
}

void main() {
  vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
  vec2 puv = zoomUV(uv);
  float d = depthAt(puv);
  float dFocus = 1.0 / clamp(u_focus, 0.05, 4000.0);
  float band = max(dFocus * 0.02, 1e-4);
  float nearPart = smoothstep(dFocus - band, dFocus + band, d);
  float belongs = u_layer == 0 ? nearPart : 1.0 - nearPart;
  float radius = 0.5 * min(cocDiameter(d), u_maxRadius);

  if (u_info == 1) {
    outColor = vec4(radius * belongs / u_infoScale, 0.0, 0.0, 1.0);
    return;
  }

  vec3 c = textureLod(u_photo, puv, baseLod()).rgb;
  float clipped = textureLod(u_photo, puv, baseLod()).a;
  float localClip = textureLod(u_photo, puv, 5.0 + baseLod()).a;
  float spread = smoothstep(0.5, 4.0, radius);
  float iso = 1.0 - smoothstep(0.15, 0.6, localClip);
  c *= exp2(u_hlStops * clipped * iso * spread);
  outColor = vec4(c * belongs, belongs);
}`;

  // Blur one layer using its own CoC field. The mip probe pulls a filled
  // radius and colour from coarser levels where the layer is occluded. The
  // aperture is a nine-blade polygon whose roundness depends on the f-stop,
  // with mechanical cat's-eye clipping toward the frame edge. The shape test is
  // evaluated in the SOURCE frame (the destination relative to the source, with
  // the near-field PSF point-reflected), not in the destination frame, and the
  // weights are normalised by the aperture area so the near-field alpha is real
  // coverage rather than an opaque average.
  const LAYER_BLUR_FRAG = `#version 300 es
precision highp float;
precision highp sampler2D;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_layerTex;
uniform sampler2D u_infoTex;
uniform sampler2D u_photo;
uniform sampler2D u_depth;
uniform vec2 u_depthSize;
uniform vec2 u_res;
uniform float u_fpxNative;
uniform float u_fmm;
uniform float u_fstop;
uniform float u_focus;
uniform float u_scale;
uniform float u_blur;
uniform float u_maxRadius;
uniform float u_maxLod;
uniform float u_infoScale;
uniform int u_layer;
uniform int u_taps;
uniform float u_zoom;
uniform vec2 u_center;

const int MAXTAPS = 192;
const float GOLDEN = 2.399963229728653;
const float PI = 3.14159265359;
const float TAU = 6.28318530718;

vec2 zoomUV(vec2 uv) {
  return clamp(u_center + (uv - 0.5) / u_zoom, vec2(0.0), vec2(1.0));
}

// Minified previews must read a prefiltered level of the photo, not mip 0.
float baseLod() {
  return max(0.0, log2(1.0 / max(u_scale, 1e-4)));
}

float cocDiameter(float d) {
  float s = clamp(u_focus, 0.05, 4000.0);
  float smm = s * 1000.0;
  float mag = smm / max(smm - u_fmm, 1.0);
  float c = (u_fpxNative * u_fmm / max(u_fstop, 0.7))
          * abs(d - 1.0 / s) * 0.001 * mag;
  return c * u_blur * u_scale;
}

float depthAt(vec2 puv) {
  ivec2 p = ivec2(clamp(puv, 0.0, 1.0) * u_depthSize);
  p = clamp(p, ivec2(0), ivec2(u_depthSize) - 1);
  vec3 c0 = textureLod(u_photo, (vec2(p) + 0.5) / u_depthSize, 0.0).rgb;
  float bestD = texelFetch(u_depth, p, 0).r;
  float bestDist = 1e9;
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      ivec2 q = clamp(p + ivec2(dx, dy), ivec2(0), ivec2(u_depthSize) - 1);
      vec3 c = textureLod(u_photo, (vec2(q) + 0.5) / u_depthSize, 0.0).rgb;
      float dist = dot(c0 - c, c0 - c);
      float d = texelFetch(u_depth, q, 0).r;
      if (dist < bestDist - 1e-7 || (abs(dist - bestDist) <= 1e-7 && d > bestD)) {
        bestDist = dist;
        bestD = d;
      }
    }
  }
  return bestD;
}

float ign(vec2 p) {
  return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

// Fixed 9-gon in image space. roundness=1 -> circle, 0 -> polygon. The polygon
// is scaled so its area matches the circle's, keeping the coverage normalisation.
float apertureLimit(float angle, float roundness) {
  const float blades = 9.0;
  const float areaScale = 1.042; // 1 / sqrt(9-gon area / circle area)
  float sector = TAU / blades;
  float local = mod(angle + 0.17 + sector * 0.5, sector) - sector * 0.5;
  float poly = cos(PI / blades) / max(cos(local), 0.20);
  return mix(poly * areaScale, 1.0, roundness);
}

float apertureShape(vec2 q, float srad, vec2 field, float fstop, float nearSign) {
  float r = length(q);
  float roundness = 1.0 - smoothstep(2.0, 4.0, fstop);
  float limit = apertureLimit(atan(q.y, q.x), roundness);
  float shape = smoothstep(1.0, 0.965, r / max(limit, 1e-3));

  // Mechanical cat's eye: the pupil is clipped by an offset circle of a
  // slightly smaller radius. The two radii differ, unlike the old dead code
  // where the sign flip did nothing.
  float fieldR = length(field);
  float cat = smoothstep(0.55, 1.65, fieldR) * (1.0 - smoothstep(2.8, 5.6, fstop));
  vec2 dir = field / max(fieldR, 1e-4);
  float r2 = 1.0 - 0.22 * cat;
  float clip2 = smoothstep(r2, r2 - 0.05, length(q - dir * (0.30 * cat)));
  shape *= mix(1.0, clip2, cat);

  // Mild spherical-aberration rim: bright rim on one side of focus, soft centre
  // on the other. The coverage normalisation below makes it energy-neutral.
  float rim = 1.0 + 0.10 * nearSign * (2.0 * pow(min(r, 1.0), 2.0) - 0.55);
  return max(shape * rim, 0.0);
}

void main() {
  vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
  vec2 puv = zoomUV(uv);
  float d = depthAt(puv);
  float dFocus = 1.0 / clamp(u_focus, 0.05, 4000.0);
  float band = max(dFocus * 0.02, 1e-4);
  float nearPart = smoothstep(dFocus - band, dFocus + band, d);
  float belongs = u_layer == 0 ? nearPart : 1.0 - nearPart;
  float own = belongs > 0.01 ? 0.5 * min(cocDiameter(d), u_maxRadius) : 0.0;

  // Mip fill. The near layer takes the max radius across covered levels so a
  // defocused foreground spreads outward. The far layer takes the first covered
  // level: that is the background colour hidden behind the subject.
  float fill = own;
  vec4 fillCol = vec4(0.0);
  for (int probe = 0; probe < 12; probe++) {
    float lod = min(float(probe), u_maxLod);
    vec4 probeCol = textureLod(u_layerTex, uv, lod);
    float cov = probeCol.a;
    if (cov > 0.02) {
      float pr = textureLod(u_infoTex, uv, lod).r * u_infoScale / max(cov, 1e-4);
      fill = max(fill, pr);
      if (probeCol.a > fillCol.a) {
        fillCol = probeCol;
      }
      if (u_layer == 1) break;
    }
    if (lod >= u_maxLod) break;
  }
  float radius = min(fill, u_maxRadius);

  if (radius < 0.55) {
    if (u_layer == 1) {
      // The background is a full plane. If this pixel is covered by a near
      // object, the probe found the surrounding background; use it so the
      // composite never shows black behind a near edge.
      float cov = smoothstep(0.0, 0.08, fillCol.a);
      vec3 col = fillCol.rgb / max(fillCol.a, 1e-4);
      outColor = vec4(col * cov, cov);
    } else {
      outColor = textureLod(u_layerTex, uv, 0.0);
    }
    return;
  }

  vec2 field = (puv - 0.5) * 2.0;
  field.x *= u_res.x / max(u_res.y, 1.0);
  float fieldR = length(field);
  float catDest = smoothstep(0.55, 1.65, fieldR) * (1.0 - smoothstep(2.8, 5.6, u_fstop));
  // Full cat's-eye clipping (two unit circles offset by 0.30, r2 = 0.78) leaves
  // 0.585 of the aperture. Divide the coverage alpha by the shape area so a
  // vignetted pupil does not make an opaque foreground transparent at the edge.
  float shapeArea = 1.0 - 0.415 * catDest;
  float spin = ign(gl_FragCoord.xy + vec2(float(u_layer) * 19.0, 0.0)) * TAU;
  float lod = clamp(log2(max(1.0, radius * sqrt(PI / float(u_taps)))) - 0.5, 0.0, u_maxLod);
  float A = PI * radius * radius / float(u_taps);
  vec4 acc = vec4(0.0);

  for (int i = 0; i < MAXTAPS; i++) {
    if (i >= u_taps) break;
    float fi = float(i) + 0.5;
    float r = sqrt(fi / float(u_taps)) * radius;
    float a = fi * GOLDEN + spin;
    vec2 sUV = clamp(uv + vec2(cos(a), sin(a)) * r / u_res, vec2(0.0), vec2(1.0));
    vec4 s = textureLod(u_layerTex, sUV, lod);
    vec3 sc = s.rgb / max(s.a, 1e-4);
    // Coverage and the source's own CoC come from mip 0, not from the
    // prefiltered level: the coarse mip would smear the layer's coverage
    // outward and defeat the coverage alpha. Only the colour is prefiltered.
    vec4 s0 = textureLod(u_layerTex, sUV, 0.0);
    float cov = max(s0.a, 1e-4);
    float srad = textureLod(u_infoTex, sUV, 0.0).r * u_infoScale / cov;
    float inside = smoothstep(r - 1.0, r + 1.0, srad);

    // A source beyond the destination's own blur radius is only allowed to
    // contribute if it is NEARER than the destination; otherwise a blurrier
    // background would paint over the midground.
    if (r > own + 1.0) {
      vec2 puvS = zoomUV(sUV);
      ivec2 sp = ivec2(clamp(puvS, 0.0, 1.0) * u_depthSize);
      sp = clamp(sp, ivec2(0), ivec2(u_depthSize) - 1);
      float sd = texelFetch(u_depth, sp, 0).r;
      if (sd < d) inside = 0.0;
    }

    vec2 q = (uv - sUV) * u_res;
    if (u_layer == 0) q = -q;             // near-field PSF is point-reflected
    vec2 puvS = zoomUV(sUV);
    vec2 fieldS = (puvS - 0.5) * 2.0;
    fieldS.x *= u_res.x / max(u_res.y, 1.0);
    float shape = apertureShape(q / max(srad, 1e-3), srad, fieldS,
                                u_fstop, u_layer == 0 ? 1.0 : -1.0);
    // The weight of a fully covered source must integrate to 1. wFull is the
    // aperture-area share of this tap; inside is the fraction of the source's
    // disc that reaches the destination. The near-field alpha is the covered
    // share divided by the full aperture share, i.e. real coverage.
    float wFull = shape * A / (PI * max(srad * srad, 0.25));
    float w = inside * wFull;

    acc += vec4(sc * cov * w, cov * w);
  }

  // A/(PI*srad^2) makes a fully covered source integrate to 1, so acc.a is
  // the covered fraction of the aperture. The cat's-eye mask reduces the
  // aperture area at the frame edge; divide by that area so an opaque subject
  // stays opaque while the blur still carries the vignetting shape.
  float alpha = clamp(acc.a / max(shapeArea, 0.20), 0.0, 1.0);
  vec3 col = acc.rgb / max(acc.a, 1e-4);
  if (u_layer == 1) {
    // Far field: a full background plane, filled from the probe where needed.
    if (acc.a < 1e-4) {
      col = fillCol.rgb / max(fillCol.a, 1e-4);
      alpha = 1.0;
    }
    outColor = vec4(col, clamp(alpha, 0.0, 1.0));
  } else {
    outColor = vec4(col, clamp(alpha, 0.0, 1.0));
  }
}`;

  /* -------------------------------------------------------------- webgl init */

  function compile(gl, type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(sh) || "shader compile failed");
    }
    return sh;
  }

  function makeProgram(gl, fragment, label) {
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, fragment));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(prog) || (label + " link failed"));
    }
    return prog;
  }

  function uniformLocations(gl, prog, names) {
    const out = {};
    for (const name of names) out[name] = gl.getUniformLocation(prog, name);
    return out;
  }

  function glError(gl, label, fail = true) {
    const code = gl.getError();
    if (code === gl.NO_ERROR) return;
    const msg = label + ": WebGL error 0x" + code.toString(16);
    if (fail) throw new Error(msg);
    console.error(msg);
  }

  function initGL() {
    const canvas = $("view");
    const gl = canvas.getContext("webgl2", {
      antialias: false,
      alpha: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: true,
      powerPreference: "high-performance",
    });
    if (!gl) throw new Error("WebGL2 is not available in this browser.");
    S.gl = gl;

    S.progFinal = makeProgram(gl, FINAL_FRAG, `final`);
    S.progLayer = makeProgram(gl, LAYER_FRAG, `layer`);
    S.progBlur = makeProgram(gl, LAYER_BLUR_FRAG, `layer blur`);
    S.progPre = makeProgram(gl, PRE_FRAG, `pre-pass`);
    S.locFinal = uniformLocations(gl, S.progFinal, [
      `u_photo`, `u_depth`, `u_near`, `u_far`, `u_depthSize`,
      `u_fpxNative`, `u_fmm`, `u_fstop`, `u_focus`, `u_scale`, `u_blur`,
      `u_maxRadius`, `u_grain`, `u_dnear`, `u_dfar`, `u_mode`, `u_linear`,
      `u_res`, `u_zoom`, `u_center`,
    ]);
    S.locLayer = uniformLocations(gl, S.progLayer, [
      `u_photo`, `u_depth`, `u_depthSize`, `u_fpxNative`, `u_fmm`,
      `u_fstop`, `u_focus`, `u_scale`, `u_blur`, `u_maxRadius`, `u_hlStops`,
      `u_infoScale`, `u_layer`, `u_info`, `u_linear`, `u_zoom`, `u_center`,
    ]);
    S.locBlur = uniformLocations(gl, S.progBlur, [
      `u_layerTex`, `u_infoTex`, `u_photo`, `u_depth`, `u_depthSize`,
      `u_res`, `u_fpxNative`, `u_fmm`, `u_fstop`, `u_focus`, `u_scale`,
      `u_blur`, `u_maxRadius`, `u_maxLod`, `u_infoScale`, `u_layer`,
      `u_taps`, `u_zoom`, `u_center`,
    ]);
    S.locPre.u_src = gl.getUniformLocation(S.progPre, "u_src");
    S.fbo = gl.createFramebuffer();

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    for (const key of ["photo", "depth", "linear"]) {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      S.tex[key] = t;
    }

    const dims = gl.getParameter(gl.MAX_VIEWPORT_DIMS);
    S.maxExport = Math.max(1, Math.min(
      gl.getParameter(gl.MAX_TEXTURE_SIZE),
      gl.getParameter(gl.MAX_RENDERBUFFER_SIZE),
      dims[0],
      dims[1],
    ));
    S.colorFloat = !!(gl.getExtension("EXT_color_buffer_float") ||
                      gl.getExtension("EXT_color_buffer_half_float"));
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    glError(gl, "WebGL initialisation");
  }

  // Attribute locations are per-program state, and the pre-pass program is a
  // different program — re-point the shared quad buffer at whichever program
  // is about to draw.
  function useQuad(prog) {
    const gl = S.gl;
    if (prog._aPos === undefined) prog._aPos = gl.getAttribLocation(prog, "a_pos");
    gl.enableVertexAttribArray(prog._aPos);
    gl.vertexAttribPointer(prog._aPos, 2, gl.FLOAT, false, 0, 0);
  }

  // Render the sRGB photo texture into the linear-light target, once per
  // photo. Half float where the driver can render to it; an 8-bit linear
  // target as fallback (the pipeline stays linear, it just bands sooner in
  // the shadows). If even that fails, S.linearOK stays false and render()
  // falls back to the raw photo texture with the legacy gamma behaviour.
  function buildLinearTexture(w, h) {
    const gl = S.gl;
    const canFloat = S.colorFloat;
    gl.bindTexture(gl.TEXTURE_2D, S.tex.linear);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    if (canFloat) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
    } else {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, S.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, S.tex.linear, 0);
    const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    if (ok) {
      gl.viewport(0, 0, w, h);
      gl.useProgram(S.progPre);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, S.tex.photo);
      gl.uniform1i(S.locPre.u_src, 0);
      useQuad(S.progPre);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (ok) {
      gl.bindTexture(gl.TEXTURE_2D, S.tex.linear);
      gl.generateMipmap(gl.TEXTURE_2D);
      glError(gl, "linear photo mip pyramid");
    }
    S.linearOK = ok;
    return ok;
  }

  const halfFloatScratch = new Float32Array(1);
  const halfBitsScratch = new Uint32Array(halfFloatScratch.buffer);

  function floatToHalf(value) {
    halfFloatScratch[0] = value;
    const bits = halfBitsScratch[0];
    const sign = (bits >>> 16) & 0x8000;
    const exponent = (bits >>> 23) & 0xff;
    let mantissa = bits & 0x7fffff;

    if (exponent === 0xff) return sign | 0x7c00 | (mantissa ? 0x0200 : 0);
    let halfExp = exponent - 127 + 15;
    if (halfExp >= 0x1f) return sign | 0x7c00;
    if (halfExp <= 0) {
      if (halfExp < -10) return sign;
      mantissa |= 0x800000;
      const shift = 14 - halfExp;
      let halfMant = mantissa >>> shift;
      const remainder = mantissa & ((1 << shift) - 1);
      const halfway = 1 << (shift - 1);
      if (remainder > halfway || (remainder === halfway && (halfMant & 1))) halfMant++;
      return sign | halfMant;
    }

    let halfMant = mantissa >>> 13;
    const remainder = mantissa & 0x1fff;
    if (remainder > 0x1000 || (remainder === 0x1000 && (halfMant & 1))) {
      halfMant++;
      if (halfMant === 0x400) {
        halfMant = 0;
        halfExp++;
        if (halfExp >= 0x1f) return sign | 0x7c00;
      }
    }
    return sign | (halfExp << 10) | halfMant;
  }

  function resizeDepthHalf(depthRaw, sourceW, sourceH, targetW, targetH) {
    const expected = sourceW * sourceH * 2;
    if (depthRaw.byteLength < expected) {
      throw new Error("depth buffer is shorter than its metadata dimensions");
    }
    const source = new Uint16Array(depthRaw);
    if (sourceW === targetW && sourceH === targetH) return source;

    const out = new Uint16Array(targetW * targetH);
    for (let y = 0; y < targetH; y++) {
      const sy = Math.max(0, Math.min(sourceH - 1, (y + 0.5) * sourceH / targetH - 0.5));
      const y0 = Math.floor(sy), y1 = Math.min(sourceH - 1, y0 + 1);
      const fy = sy - y0;
      for (let x = 0; x < targetW; x++) {
        const sx = Math.max(0, Math.min(sourceW - 1, (x + 0.5) * sourceW / targetW - 0.5));
        const x0 = Math.floor(sx), x1 = Math.min(sourceW - 1, x0 + 1);
        const fx = sx - x0;
        const d00 = halfToFloat(source[y0 * sourceW + x0]);
        const d10 = halfToFloat(source[y0 * sourceW + x1]);
        const d01 = halfToFloat(source[y1 * sourceW + x0]);
        const d11 = halfToFloat(source[y1 * sourceW + x1]);
        const top = d00 + (d10 - d00) * fx;
        const bottom = d01 + (d11 - d01) * fx;
        out[y * targetW + x] = floatToHalf(top + (bottom - top) * fy);
      }
    }
    return out;
  }

  function uploadTextures(bitmap, depthRaw, meta) {
    const gl = S.gl;
    const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    const k = Math.min(1, maxTex / Math.max(meta.width, meta.height));
    const targetW = Math.max(1, Math.round(meta.width * k));
    const targetH = Math.max(1, Math.round(meta.height * k));

    let bmp = bitmap;
    if (bitmap.width !== targetW || bitmap.height !== targetH) {
      const c = document.createElement("canvas");
      c.width = targetW;
      c.height = targetH;
      c.getContext("2d").drawImage(bitmap, 0, 0, targetW, targetH);
      bmp = c;
    }
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.bindTexture(gl.TEXTURE_2D, S.tex.photo);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, bmp);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    glError(gl, "photo upload");

    gl.bindTexture(gl.TEXTURE_2D, S.tex.depth);
    const half = resizeDepthHalf(depthRaw, meta.width, meta.height, targetW, targetH);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, targetW, targetH, 0, gl.RED, gl.HALF_FLOAT, half);
    S.depthW = targetW;
    S.depthH = targetH;
    glError(gl, "depth upload");

    buildLinearTexture(targetW, targetH);
  }

  /* --------------------------------------------------------------- rendering */

  function releaseTargets() {
    if (!S.targets) return;
    const gl = S.gl;
    for (const key of ["nearColour", "farColour", "nearInfo", "farInfo", "nearBlur", "farBlur"]) {
      if (S.targets[key]) gl.deleteTexture(S.targets[key]);
    }
    S.targets = null;
  }

  function makeTargetTexture(internal, w, h, levels, mipmapped) {
    const gl = S.gl;
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER,
      mipmapped ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texStorage2D(gl.TEXTURE_2D, levels, internal, w, h);
    return t;
  }

  function attachTarget(texture) {
    const gl = S.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, S.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error("WebGL framebuffer is incomplete for the bokeh layer");
    }
  }

  function ensureTargets(outputW, outputH) {
    // The layer textures match the render target exactly: the final composite
    // reads them at canvas UV, and the near/far fields are built at the same
    // resolution the canvas is drawn at. A preview is small; an export is the
    // photo size (capped by maxExport).
    const w = Math.max(1, Math.round(outputW));
    const h = Math.max(1, Math.round(outputH));
    if (S.targets && S.targets.w === w && S.targets.h === h) return S.targets;

    releaseTargets();
    const gl = S.gl;
    const levels = 1 + Math.floor(Math.log2(Math.max(w, h)));
    const colourFormat = S.colorFloat ? gl.RGBA16F : gl.RGBA8;
    const infoFormat = S.colorFloat ? gl.R16F : gl.R8;
    const targets = {
      w, h, levels,
      maxLod: levels - 1,
      infoScale: S.colorFloat ? 1 : Math.max(1, Math.min(w, h) * 0.20),
      nearColour: makeTargetTexture(colourFormat, w, h, levels, true),
      farColour: makeTargetTexture(colourFormat, w, h, levels, true),
      nearInfo: makeTargetTexture(infoFormat, w, h, levels, true),
      farInfo: makeTargetTexture(infoFormat, w, h, levels, true),
      nearBlur: makeTargetTexture(colourFormat, w, h, 1, false),
      farBlur: makeTargetTexture(colourFormat, w, h, 1, false),
    };
    S.targets = targets;
    for (const key of ["nearColour", "farColour", "nearInfo", "farInfo", "nearBlur", "farBlur"]) {
      attachTarget(targets[key]);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    glError(gl, "bokeh target allocation");
    return targets;
  }

  function setLayerUniforms(loc, w, h, linear) {
    const gl = S.gl, m = S.meta;
    const fmm = effFocal();
    gl.uniform1f(loc.u_fpxNative, m.f_px);   // native projection, fixed
    gl.uniform1f(loc.u_fmm, fmm);            // simulated (or native) focal, mm
    gl.uniform1f(loc.u_fstop, FSTOPS[S.fstopIdx]);
    gl.uniform1f(loc.u_focus, S.focus);
    gl.uniform1f(loc.u_scale, w / m.width * S.zoom);
    gl.uniform1f(loc.u_blur, S.blur);
    gl.uniform1f(loc.u_maxRadius, Math.min(w, h) * 0.10 * S.zoom);
    gl.uniform1i(loc.u_linear, linear ? 1 : 0);
    gl.uniform2f(loc.u_depthSize, S.depthW || m.width, S.depthH || m.height);
    gl.uniform1f(loc.u_zoom, S.zoom);
    gl.uniform2f(loc.u_center, S.cx, S.cy);
  }

  function prepareLayers(linear) {
    const gl = S.gl, t = S.targets;
    const source = linear ? S.tex.linear : S.tex.photo;
    gl.useProgram(S.progLayer);
    useQuad(S.progLayer);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, source);
    gl.uniform1i(S.locLayer.u_photo, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, S.tex.depth);
    gl.uniform1i(S.locLayer.u_depth, 1);
    setLayerUniforms(S.locLayer, t.w, t.h, linear);
    // The clip mask only exists on the linear texture; the raw fallback has no
    // highlight headroom to recover.
    gl.uniform1f(S.locLayer.u_hlStops, linear ? S.highlight * 5.0 : 0.0);
    gl.uniform1f(S.locLayer.u_infoScale, t.infoScale);

    for (let layer = 0; layer < 2; layer++) {
      const colour = layer === 0 ? t.nearColour : t.farColour;
      const info = layer === 0 ? t.nearInfo : t.farInfo;
      gl.uniform1i(S.locLayer.u_layer, layer);
      gl.viewport(0, 0, t.w, t.h);

      attachTarget(colour);
      gl.uniform1i(S.locLayer.u_info, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      attachTarget(info);
      gl.uniform1i(S.locLayer.u_info, 1);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    for (const texture of [t.nearColour, t.farColour, t.nearInfo, t.farInfo]) {
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.generateMipmap(gl.TEXTURE_2D);
    }
    glError(gl, "layer pyramid");
  }

  function blurLayers(taps, linear) {
    const gl = S.gl, t = S.targets;
    gl.useProgram(S.progBlur);
    useQuad(S.progBlur);
    gl.uniform2f(S.locBlur.u_res, t.w, t.h);
    gl.uniform1f(S.locBlur.u_maxLod, t.maxLod);
    gl.uniform1f(S.locBlur.u_infoScale, t.infoScale);
    gl.uniform1i(S.locBlur.u_taps, Math.max(8, Math.min(192, taps || PREVIEW_TAPS)));
    setLayerUniforms(S.locBlur, t.w, t.h, linear);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, linear ? S.tex.linear : S.tex.photo);
    gl.uniform1i(S.locBlur.u_photo, 2);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, S.tex.depth);
    gl.uniform1i(S.locBlur.u_depth, 3);
    gl.viewport(0, 0, t.w, t.h);

    for (let layer = 0; layer < 2; layer++) {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, layer === 0 ? t.nearColour : t.farColour);
      gl.uniform1i(S.locBlur.u_layerTex, 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, layer === 0 ? t.nearInfo : t.farInfo);
      gl.uniform1i(S.locBlur.u_infoTex, 1);
      gl.uniform1i(S.locBlur.u_layer, layer);
      attachTarget(layer === 0 ? t.nearBlur : t.farBlur);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (S.busy) glError(gl, "layered bokeh", false);
  }

  function renderFinal(mode, useLinear, taps) {
    const gl = S.gl, canvas = $("view"), m = S.meta;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(S.progFinal);
    useQuad(S.progFinal);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, useLinear ? S.tex.linear : S.tex.photo);
    gl.uniform1i(S.locFinal.u_photo, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, S.tex.depth);
    gl.uniform1i(S.locFinal.u_depth, 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, S.targets ? S.targets.nearBlur : null);
    gl.uniform1i(S.locFinal.u_near, 2);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, S.targets ? S.targets.farBlur : null);
    gl.uniform1i(S.locFinal.u_far, 3);

    const fmm = effFocal();
    gl.uniform1f(S.locFinal.u_fpxNative, m.f_px);
    gl.uniform1f(S.locFinal.u_fmm, fmm);
    gl.uniform1f(S.locFinal.u_fstop, FSTOPS[S.fstopIdx]);
    gl.uniform1f(S.locFinal.u_focus, S.focus);
    gl.uniform1f(S.locFinal.u_scale, canvas.width / m.width * S.zoom);
    gl.uniform1f(S.locFinal.u_blur, S.blur);
    gl.uniform1f(S.locFinal.u_maxRadius, Math.min(canvas.width, canvas.height) * 0.10 * S.zoom);
    gl.uniform1f(S.locFinal.u_grain, S.grain * 0.02);
    gl.uniform1f(S.locFinal.u_dnear, S.dLo);
    gl.uniform1f(S.locFinal.u_dfar, S.dHi);
    gl.uniform1i(S.locFinal.u_mode, mode);
    gl.uniform1i(S.locFinal.u_linear, useLinear ? 1 : 0);
    gl.uniform2f(S.locFinal.u_res, canvas.width, canvas.height);
    gl.uniform2f(S.locFinal.u_depthSize, S.depthW || m.width, S.depthH || m.height);
    gl.uniform1f(S.locFinal.u_zoom, S.zoom);
    gl.uniform2f(S.locFinal.u_center, S.cx, S.cy);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (S.busy) glError(gl, "final render", false);
  }

  function render(taps, forExport = false) {
    const gl = S.gl, canvas = $("view");
    if (!S.meta || !S.progFinal) return;
    if (!canvas.width || !canvas.height || gl.isContextLost()) return;

    const effMode = S.holdOriginal ? 2 : S.mode;
    const useLinear = S.linearOK && effMode !== 2;
    if (effMode === 0) {
      ensureTargets(canvas.width, canvas.height);
      prepareLayers(useLinear);
      blurLayers(taps, useLinear);
    }
    renderFinal(effMode, useLinear, taps);
  }

  function requestRender() {
    if (S.dirty) return;
    S.dirty = true;
    requestAnimationFrame(() => {
      S.dirty = false;
      render(PREVIEW_TAPS);
    });
  }

  function layout() {
    if (!S.meta) return;
    const stage = $("stage").getBoundingClientRect();
    const pad = 26;
    const availW = Math.max(80, stage.width - pad * 2);
    const availH = Math.max(80, stage.height - pad * 2);
    const ar = S.meta.width / S.meta.height;
    let w = availW, h = w / ar;
    if (h > availH) { h = availH; w = h * ar; }
    const canvas = $("view");
    canvas.style.width = `${Math.round(w)}px`;
    canvas.style.height = `${Math.round(h)}px`;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.round(w * dpr));
    canvas.height = Math.max(1, Math.round(h * dpr));

    // Keep the HUD glued to the photo, not to the empty stage around it.
    const sr = $("stage").getBoundingClientRect();
    const cr = canvas.getBoundingClientRect();
    const st = $("stage").style;
    st.setProperty("--hud-left", `${Math.round(cr.left - sr.left)}px`);
    st.setProperty("--hud-right", `${Math.round(sr.right - cr.right)}px`);
    st.setProperty("--hud-top", `${Math.round(cr.top - sr.top)}px`);
    st.setProperty("--hud-bottom", `${Math.round(sr.bottom - cr.bottom)}px`);

    requestRender();
  }

  /* ------------------------------------------------------------------- ui glue */

  function fmtDist(m) {
    if (!isFinite(m)) return "∞";
    if (m < 1) return `${(m * 100).toFixed(0)} cm`;
    if (m < 10) return `${m.toFixed(2)} m`;
    if (m < 100) return `${m.toFixed(1)} m`;
    return `${Math.round(m)} m`;
  }

  const logPos = (d) =>
    (Math.log(Math.max(d, S.dLo)) - Math.log(S.dLo)) / (Math.log(S.dHi) - Math.log(S.dLo));

  // The model's own focal length is the honest default; a longer "simulated lens"
  // is how you get a portrait look out of a wide-angle frame.
  const effFocal = () =>
    S.meta ? (S.lensMM > 0 ? S.lensMM : clamp(S.meta.focal35, 12, 200)) : 50;

  const fromSlider = (v) => S.dLo * Math.pow(S.dHi / S.dLo, v / 1000);
  const toSlider = (d) => Math.round(1000 * Math.log(Math.max(d, S.dLo) / S.dLo) / Math.log(S.dHi / S.dLo));

  function dofLimits() {
    const f = effFocal();
    const N = FSTOPS[S.fstopIdx];
    const s = S.focus * 1000;
    const H = (f * f) / (N * COC_LIMIT) + f;
    const near = (s * (H - f)) / (H + s - 2 * f);
    const far = H > s ? (s * (H - f)) / (H - s) : Infinity;
    return { near: near / 1000, far: far / 1000, hyper: H / 1000 };
  }

  function updateFocusUI() {
    $("focusOut").textContent = fmtDist(S.focus);
    const p = Math.max(0, Math.min(1, logPos(S.focus)));
    $("dofMarker").style.left = `${p * 100}%`;

    const { near, far, hyper } = dofLimits();
    const a = Math.max(0, Math.min(1, logPos(near)));
    const b = Math.max(0, Math.min(1, isFinite(far) ? logPos(far) : 1));
    const band = $("dofBand");
    band.style.left = `${a * 100}%`;
    band.style.width = `${Math.max(0.6, (b - a) * 100)}%`;

    $("dofNear").textContent = fmtDist(near);
    $("dofFar").textContent = isFinite(far) ? fmtDist(far) : "∞";
    $("dofMid").textContent = `hyperfocal ${fmtDist(hyper)}`;
    $("dofInfo").textContent = far >= S.dHi
      ? "Depth of field covers the whole frame — nothing left to blur."
      : `In focus ${fmtDist(near)} → ${fmtDist(far)} · f/${FSTOPS[S.fstopIdx].toFixed(1)} at ${effFocal().toFixed(0)} mm eq.`;
    if ($("lensOut")) $("lensOut").textContent = `${effFocal().toFixed(0)} mm`;
  }

  function updateLensUI() {
    $("fstopOut").textContent = `f/${FSTOPS[S.fstopIdx].toFixed(1)}`;
    document.querySelectorAll("#stops button").forEach((b) => {
      b.classList.toggle("active", Math.abs(parseFloat(b.dataset.stop) - FSTOPS[S.fstopIdx]) < 0.01);
    });
    $("blurOut").textContent = `${S.blur.toFixed(2)}×`;
    $("highlightOut").textContent = `${(S.highlight * 5).toFixed(1)} stops`;
    if ($("grainOut")) $("grainOut").textContent = S.grain.toFixed(2);
  }

  /* ------------------------------------------------------------------- zoom/pan */

  function clampView() {
    S.zoom = clamp(S.zoom, 1, 12);
    const half = 0.5 / S.zoom;
    S.cx = clamp(S.cx, half, 1 - half);
    S.cy = clamp(S.cy, half, 1 - half);
  }

  function updateZoomUI() {
    const view = $("view");
    view.style.cursor = S.zoom > 1 ? "grab" : "crosshair";
    $("hudHint").innerHTML = S.zoom > 1
      ? `Zoom ${Math.round(S.zoom * 100)}% · drag to pan · double-click to fit`
      : `Click photo to focus · wheel to zoom · <kbd>Space</kbd> original`;
  }

  function setMode(mode) {
    S.mode = mode;
    document.querySelectorAll("#viewTabs .chip").forEach((c) => {
      c.classList.toggle("active", +c.dataset.mode === (S.holdOriginal ? 2 : mode));
    });
    requestRender();
  }

  /* ------------------------------------------------------------------ loading */

  function overlay(show, title, sub, bar) {
    const el = $("overlay");
    el.hidden = !show;
    if (title) $("overlayTitle").textContent = title;
    if (sub) $("overlaySub").textContent = sub;
    if (bar !== undefined) $("overlayBar").style.width = `${bar}%`;
  }

  function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

  async function openFile(file) {
    if (S.busy) return;
    S.busy = true;
    overlay(true, "Estimating depth", "Depth Pro 2 · H+ · metric depth at the photo's scale", 12);

    let bar = 12;
    const tick = setInterval(() => {
      bar = Math.min(92, bar + (92 - bar) * 0.12);
      $("overlayBar").style.width = `${bar}%`;
    }, 220);

    try {
      // Warm the model first: the first run may still be downloading 3.5 GB.
      await warmup();
      $("overlaySub").textContent = "Running H+ on the GPU";

      const fd = new FormData();
      fd.append("file", file, file.name || "photo.jpg");
      const res = await fetch("/api/analyze", { method: "POST", body: fd });
      const meta = await res.json();
      if (!res.ok || meta.error) throw new Error(meta.error || `HTTP ${res.status}`);
      S.meta = meta;
      S.zoom = 1;
      S.cx = 0.5;
      S.cy = 0.5;

      $("overlayTitle").textContent = "Loading depth";
      $("overlaySub").textContent = `${meta.width} × ${meta.height} px`;
      const [bmp, depthBuf] = await Promise.all([
        fetch(meta.image_url).then((r) => r.blob()).then((b) => createImageBitmap(b)),
        fetch(meta.depth_url).then((r) => r.arrayBuffer()),
      ]);
      if (S.bitmap && S.bitmap.close) S.bitmap.close();
      uploadTextures(bmp, depthBuf, meta);
      if (bmp.close) bmp.close();   // texture owns the pixels now
      S.bitmap = null;
      prepareDepthLow(new Uint16Array(depthBuf), meta);

      S.dLo = Math.max(0.15, meta.depth_pcts.p5 * 0.55);
      S.dHi = Math.min(400, Math.max(meta.depth_pcts.p99_5 * 1.6, S.dLo * 8));
      S.focus = clamp(autoFocusValue(), S.dLo * 1.05, S.dHi * 0.95);
      $("focus").value = toSlider(S.focus);
      $("fstop").value = S.fstopIdx;

      const p = meta.depth_pcts;
      $("kvRange").textContent = `${fmtDist(p.p0_5)} – ${fmtDist(p.p99_5)}`;
      $("kvMedian").textContent = fmtDist(p.p50);
      $("kvInfer").textContent = `${meta.inference.width} × ${meta.inference.height}`;
      $("kvTime").textContent = `${meta.inference.seconds.toFixed(2)} s`;
      $("depthFocal").textContent = `Native ${meta.focal35.toFixed(0)} mm`;
      $("hudFile").textContent = `${meta.filename} · ${meta.width} × ${meta.height}`;
      $("hudStats").textContent =
        `H+ ${meta.inference.width}px · ${meta.inference.seconds.toFixed(2)}s · ` +
        `peak ${meta.inference.peak_vram_gb.toFixed(1)} GB · focus ${meta.f_px_source}`;

      $("empty").classList.add("hidden");
      $("view").classList.remove("hidden");
      $("panel").classList.remove("empty-state");
      $("exportBtn").disabled = false;
      $("saveDepth").disabled = false;

      updateFocusUI();
      updateLensUI();
      updateZoomUI();
      layout();
      overlay(false);
    } catch (err) {
      console.error(err);
      overlay(true, "Could not process that photo", String(err.message || err), 100);
      setTimeout(() => overlay(false), 5200);
    } finally {
      clearInterval(tick);
      S.busy = false;
    }
  }

  function clamp(v, a, b) { return Math.min(b, Math.max(a, v)); }

  function prepareDepthLow(u16, meta) {
    const lowW = Math.min(640, meta.width);
    const lowH = Math.max(1, Math.round((meta.height / meta.width) * lowW));
    const out = new Float32Array(lowW * lowH);
    for (let y = 0; y < lowH; y++) {
      const sy = Math.min(meta.height - 1, Math.floor((y + 0.5) * meta.height / lowH));
      for (let x = 0; x < lowW; x++) {
        const sx = Math.min(meta.width - 1, Math.floor((x + 0.5) * meta.width / lowW));
        const dv = halfToFloat(u16[sy * meta.width + sx]);
        out[y * lowW + x] = dv > 1e-6 ? 1.0 / dv : 1e6;  // sky -> huge, filtered out
      }
    }
    S.depthLow = out;
    S.lowW = lowW;
    S.lowH = lowH;
  }

  function halfToFloat(h) {
    const s = (h & 0x8000) >> 15;
    const e = (h & 0x7c00) >> 10;
    const f = h & 0x03ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }

  function depthAt(u, v) {
    if (!S.depthLow) return null;
    const x = clamp(Math.floor(u * S.lowW), 0, S.lowW - 1);
    const y = clamp(Math.floor(v * S.lowH), 0, S.lowH - 1);
    return S.depthLow[y * S.lowW + x];
  }

  function autoFocusValue() {
    const d = dofValuesRegion(0.3, 0.7, 0.3, 0.7);
    if (!d.length) return 3;
    d.sort((a, b) => a - b);
    return d[Math.floor(d.length * 0.45)];
  }

  function dofValuesRegion(u0, u1, v0, v1) {
    const out = [];
    if (!S.depthLow) return out;
    const x0 = Math.floor(u0 * S.lowW), x1 = Math.ceil(u1 * S.lowW);
    const y0 = Math.floor(v0 * S.lowH), y1 = Math.ceil(v1 * S.lowH);
    for (let y = y0; y < y1; y += 2) {
      for (let x = x0; x < x1; x += 2) {
        const z = S.depthLow[y * S.lowW + x];
        if (isFinite(z) && z > 0.05 && z < 500) out.push(z);
      }
    }
    return out;
  }

  /* ------------------------------------------------------------------- export */

  function download(blob, name) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 8000);
  }

  function baseName() {
    const n = (S.meta && S.meta.filename) || "photo";
    return n.replace(/\.[^.]+$/, "").replace(/[^\w\-. ]+/g, "_") || "photo";
  }

  async function exportRender({ mode, mime, ext, taps, tag = "dof" }) {
    const canvas = $("view");
    const meta = S.meta;
    const k = Math.min(1, S.maxExport / Math.max(meta.width, meta.height));
    const w = Math.round(meta.width * k);
    const h = Math.round(meta.height * k);

    overlay(true, "Rendering", `${w} × ${h} px · two-field bokeh · ${taps} aperture taps`, 35);
    await sleep(60);

    const prevW = canvas.width, prevH = canvas.height;
    const prevMode = S.mode;
    const prevZoom = S.zoom, prevCx = S.cx, prevCy = S.cy;
    canvas.width = w;
    canvas.height = h;
    S.mode = mode;
    S.zoom = 1; S.cx = 0.5; S.cy = 0.5;   // export is always the whole frame
    render(taps, true);

    const blob = await new Promise((res) => canvas.toBlob(res, mime, S.quality / 100));

    S.mode = prevMode;
    S.zoom = prevZoom;
    S.cx = prevCx;
    S.cy = prevCy;
    canvas.width = prevW;
    canvas.height = prevH;
    layout();
    updateZoomUI();
    overlay(false);

    if (!blob) throw new Error("The browser could not encode the image.");
    download(blob, `${baseName()}_${tag}.${ext}`);
  }

  async function exportPhoto() {
    if (!S.meta || S.busy) return;
    S.busy = true;
    $("exportBtn").disabled = true;
    try {
      await exportRender({
        mode: 0,
        mime: S.format === "png" ? "image/png" : "image/jpeg",
        ext: `${S.format === "png" ? "png" : "jpg"}`,
        taps: EXPORT_TAPS,
      });
    } catch (e) {
      console.error(e);
      overlay(true, "Export failed", String(e.message || e), 100);
      setTimeout(() => overlay(false), 4000);
    } finally {
      S.busy = false;
      $("exportBtn").disabled = false;
    }
  }

  async function exportDepth() {
    if (!S.meta || S.busy) return;
    S.busy = true;
    try {
      await exportRender({ mode: 1, mime: "image/png", ext: "png", taps: 1, tag: "depth" });
    } catch (e) {
      console.error(e);
    } finally {
      S.busy = false;
    }
  }

  /* ------------------------------------------------------------------- events */

  function wire() {
    $("openBtn").onclick = () => $("fileInput").click();
    $("pickBtn").onclick = () => $("fileInput").click();
    $("fileInput").onchange = (e) => {
      const f = e.target.files && e.target.files[0];
      if (f) openFile(f);
      e.target.value = "";
    };

    $("sampleBtn").onclick = async () => {
      const r = await fetch("/sample.jpg");
      const b = await r.blob();
      openFile(new File([b], "sample.jpg", { type: b.type || "image/jpeg" }));
    };

    const stage = $("stage");
    ["dragenter", "dragover"].forEach((ev) =>
      stage.addEventListener(ev, (e) => { e.preventDefault(); stage.classList.add("dragging"); }));
    ["dragleave", "drop"].forEach((ev) =>
      stage.addEventListener(ev, (e) => { e.preventDefault(); stage.classList.remove("dragging"); }));
    stage.addEventListener("drop", (e) => {
      const f = e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) openFile(f);
    });

    window.addEventListener("paste", (e) => {
      const items = e.clipboardData && e.clipboardData.files;
      if (items && items[0]) openFile(items[0]);
    });

    $("focus").addEventListener("input", (e) => {
      S.focus = fromSlider(+e.target.value);
      updateFocusUI();
      requestRender();
    });

    $("fstop").addEventListener("input", (e) => {
      S.fstopIdx = +e.target.value;
      updateLensUI();
      updateFocusUI();
      requestRender();
    });

    document.querySelectorAll("#stops button").forEach((b) => {
      b.onclick = () => {
        const v = parseFloat(b.dataset.stop);
        let best = 0;
        FSTOPS.forEach((f, i) => { if (Math.abs(f - v) < Math.abs(FSTOPS[best] - v)) best = i; });
        S.fstopIdx = best;
        $("fstop").value = best;
        updateLensUI();
        updateFocusUI();
        requestRender();
      };
    });

    document.querySelectorAll("#lens button").forEach((b) => {
      b.onclick = () => {
        S.lensMM = parseFloat(b.dataset.mm) || 0;
        document.querySelectorAll("#lens button").forEach((x) => x.classList.toggle("active", x === b));
        updateFocusUI();
        requestRender();
      };
    });

    $("blur").addEventListener("input", (e) => {
      S.blur = +e.target.value / 100;
      updateLensUI();
      requestRender();
    });

    $("highlight").addEventListener("input", (e) => {
      S.highlight = +e.target.value / 100;
      updateLensUI();
      requestRender();
    });

    $("grain").addEventListener("input", (e) => {
      S.grain = +e.target.value / 100;
      updateLensUI();
      requestRender();
    });

    $("autoFocus").onclick = () => {
      if (!S.meta) return;
      S.focus = clamp(autoFocusValue(), S.dLo, S.dHi);
      $("focus").value = toSlider(S.focus);
      updateFocusUI();
      requestRender();
    };

    $("centerFocus").onclick = () => {
      if (!S.meta) return;
      S.focus = clamp(depthAt(0.5, 0.5) || S.focus, S.dLo, S.dHi);
      $("focus").value = toSlider(S.focus);
      updateFocusUI();
      requestRender();
    };

    document.querySelectorAll("#viewTabs .chip").forEach((c) => {
      c.onclick = () => setMode(+c.dataset.mode);
    });

    document.querySelectorAll("#formatSeg button").forEach((b) => {
      b.onclick = () => {
        S.format = b.dataset.format;
        document.querySelectorAll("#formatSeg button").forEach((x) => x.classList.toggle("active", x === b));
        $("quality").disabled = S.format === "png"; // Q is meaningless for PNG
      };
    });

    $("quality").addEventListener("input", (e) => {
      S.quality = +e.target.value;
      $("qualityOut").textContent = String(S.quality);
    });

    $("exportBtn").onclick = exportPhoto;
    $("saveDepth").onclick = exportDepth;

    $("view").addEventListener("click", (e) => {
      if (S.suppressClick) { S.suppressClick = false; return; } // that was a pan
      if (e.detail > 1) return; // second click of a double-click: don't refocus
      if (!S.meta || S.mode === 2) return; // focus picking works in Result and Depth Map
      const rect = $("view").getBoundingClientRect();
      const u = S.cx + ((e.clientX - rect.left) / rect.width - 0.5) / S.zoom;
      const v = S.cy + ((e.clientY - rect.top) / rect.height - 0.5) / S.zoom;
      const z = depthAt(u, v);
      if (!z || !isFinite(z)) return;
      S.focus = clamp(z, S.dLo, S.dHi);
      $("focus").value = toSlider(S.focus);
      updateFocusUI();
      requestRender();

      const ring = $("focusRing");
      const pt = stagePoint(e.clientX, e.clientY);
      ring.style.left = `${pt.x}px`;
      ring.style.top = `${pt.y}px`;
      ring.hidden = true;
      void ring.offsetWidth;
      ring.hidden = false;
      clearTimeout(ring._t);
      ring._t = setTimeout(() => { ring.hidden = true; }, 600);
    });

    const viewEl = $("view");
    viewEl.addEventListener("wheel", (e) => {
      if (!S.meta) return;
      e.preventDefault();
      const rect = viewEl.getBoundingClientRect();
      const sx = (e.clientX - rect.left) / rect.width;
      const sy = (e.clientY - rect.top) / rect.height;
      // keep the photo point under the cursor fixed while zooming
      const px = S.cx + (sx - 0.5) / S.zoom;
      const py = S.cy + (sy - 0.5) / S.zoom;
      S.zoom = clamp(S.zoom * Math.exp(-e.deltaY * 0.0015), 1, 12);
      S.cx = px - (sx - 0.5) / S.zoom;
      S.cy = py - (sy - 0.5) / S.zoom;
      clampView();
      updateZoomUI();
      requestRender();
    }, { passive: false });

    viewEl.addEventListener("mousedown", (e) => {
      if (!S.meta || e.button !== 0) return;
      S.suppressClick = false;
      S.drag = { x: e.clientX, y: e.clientY, cx: S.cx, cy: S.cy, moved: false,
                 rect: viewEl.getBoundingClientRect() };
      if (S.zoom > 1) viewEl.style.cursor = "grabbing";
    });

    window.addEventListener("mousemove", (e) => {
      if (!S.drag) return;
      const d = S.drag;
      if (!d.moved && Math.abs(e.clientX - d.x) + Math.abs(e.clientY - d.y) < 3) return;
      d.moved = true;
      S.cx = d.cx - (e.clientX - d.x) / d.rect.width / S.zoom;
      S.cy = d.cy - (e.clientY - d.y) / d.rect.height / S.zoom;
      clampView();
      requestRender();
    });

    window.addEventListener("mouseup", () => {
      if (!S.drag) return;
      S.suppressClick = S.drag.moved; // a pan must not also set focus
      S.drag = null;
      updateZoomUI();
    });

    viewEl.addEventListener("dblclick", () => {
      if (!S.meta) return;
      S.zoom = 1;
      S.cx = 0.5;
      S.cy = 0.5;
      updateZoomUI();
      requestRender();
    });

    window.addEventListener("keydown", (e) => {
      const tag = (e.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea") return;
      if (e.code === "Space" && !e.repeat) {
        e.preventDefault();
        S.holdOriginal = true;
        requestRender();
      } else if (e.key === "d" || e.key === "D") {
        setMode(S.mode === 1 ? 0 : 1);
      } else if (e.key === "e" || e.key === "E") {
        exportPhoto();
      }
    });
    window.addEventListener("keyup", (e) => {
      if (e.code === "Space") { S.holdOriginal = false; requestRender(); }
    });
    // Alt-tabbing while Space is held never delivers keyup: release on blur.
    window.addEventListener("blur", () => {
      if (S.holdOriginal) { S.holdOriginal = false; requestRender(); }
    });

    window.addEventListener("resize", layout);

    const ro = new ResizeObserver(() => layout());
    ro.observe($("stage"));
  }

  function stagePoint(clientX, clientY) {
    const rect = $("stage").getBoundingClientRect();
    return { x: clientX - rect.left, y: clientY - rect.top };
  }

  /* ----------------------------------------------------------------- warmup/status */

  let warmupPromise = null;
  function warmup() {
    if (warmupPromise) return warmupPromise;
    warmupPromise = (async () => {
      try {
        await fetch("/api/warmup", { method: "POST" });
        for (;;) {
          const s = await fetch("/api/status").then((r) => r.json());
          const badge = $("gpuBadge");
          if (s.status === "ready") {
            badge.textContent = `${s.device === "cuda" ? "GPU" : "CPU"} · H+ ready`;
            badge.className = "badge ok";
            return true;
          }
          if (s.status === "error") {
            badge.textContent = "model error";
            badge.className = "badge warn";
            throw new Error(s.message || "model failed to load");
          }
          badge.textContent = s.status === "loading" ? "loading H+ …" : "warming up …";
          badge.className = "badge ghost";
          if (!$("overlay").hidden) {
            $("overlaySub").textContent =
              s.status === "loading" ? "First run downloads 3.5 GB of H+ weights …" : s.message || "starting";
          }
          await sleep(1200);
        }
      } catch (e) {
        console.error(e);
        warmupPromise = null; // allow a retry on the next openFile
        throw e;
      }
    })();
    return warmupPromise;
  }

  async function statusProbe() {
    try {
      const s = await fetch("/api/status").then((r) => r.json());
      const badge = $("gpuBadge");
      if (s.status === "ready") {
        badge.textContent = `${s.device === "cuda" ? "GPU" : "CPU"} · H+ ready`;
        badge.className = "badge ok";
      } else if (s.status === "error") {
        badge.textContent = "model error";
        badge.className = "badge warn";
      } else {
        badge.textContent = "model idle";
        badge.className = "badge ghost";
      }
    } catch {
      $("gpuBadge").textContent = "server offline";
      $("gpuBadge").className = "badge warn";
    }
  }

  /* --------------------------------------------------------------------- boot */

  initGL();
  wire();
  $("panel").classList.add("empty-state");
  $("view").classList.add("hidden");
  $("fstop").value = S.fstopIdx;
  S.fstopIdx = +$("fstop").value;
  updateLensUI();
  updateZoomUI();
  statusProbe();

  // Headless-renderer test hook. Only with ?debug: a probe can set the state
  // and paint a frame without going through the sliders.
  if (new URLSearchParams(location.search).has("debug")) {
    window.__dof = { S, render, requestRender, layout, updateFocusUI, updateLensUI };
  }
})();
