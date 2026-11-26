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
  const PREVIEW_TAPS = 20;
  const EXPORT_TAPS = 40;
  const MAX_WORK_PIXELS = 6_000_000;
  const MAX_WORK_SIDE = 4096;
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
    highlight: 0.15,
    lensMM: 0,
    format: "jpeg",
    quality: 95,
    depthLow: null,
    lowW: 0,
    lowH: 0,
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
  // photo, and build the source mip pyramid in linear space.
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
  outColor = vec4(srgbToLinear(texture(u_src, uv).rgb), 1.0);
}`;

  // The final pass doubles as the original/depth preview. In DOF mode the
  // near layer is premultiplied and the far layer has been filled by its mip
  // pyramid, so the physical order is simply near over far.
  const FINAL_FRAG = `#version 300 es
precision highp float;
precision highp sampler2D;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_photo;
uniform sampler2D u_depth;
uniform sampler2D u_near;
uniform sampler2D u_far;
uniform float u_fpx;
uniform float u_fmm;
uniform float u_fstop;
uniform float u_focus;
uniform float u_scale;
uniform float u_blur;
uniform float u_maxblur;
uniform float u_dnear;
uniform float u_dfar;
uniform int u_mode;
uniform int u_linear;

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

float cocDiameter(float z) {
  float zmm = clamp(z, 0.05, 4000.0) * 1000.0;
  float smm = clamp(u_focus, 0.05, 4000.0) * 1000.0;
  float c = (u_fpx / max(u_fstop, 0.7)) * u_fmm * abs(1.0 / zmm - 1.0 / smm);
  return c * u_blur * u_scale;
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
  vec3 base = textureLod(u_photo, v_uv, 0.0).rgb;
  if (u_mode == 2) {
    outColor = vec4(encode(base), 1.0);
    return;
  }

  float z = textureLod(u_depth, v_uv, 0.0).r;
  if (u_mode == 1) {
    float lz = log(max(z, 1e-3));
    float t = (lz - log(u_dnear)) / max(log(u_dfar) - log(u_dnear), 1e-4);
    outColor = vec4(depthRamp(clamp(t, 0.0, 1.0)), 1.0);
    return;
  }

  vec4 nearLayer = texture(u_near, v_uv);
  vec4 farLayer = texture(u_far, v_uv);
  float nearA = clamp(nearLayer.a, 0.0, 1.0);
  float farA = clamp(farLayer.a, 0.0, 1.0);

  vec3 under = farLayer.rgb + base * (1.0 - farA);
  vec3 layered = nearLayer.rgb + under * (1.0 - nearA);

  float radius = 0.5 * min(cocDiameter(z), u_maxblur);
  float edgeCoverage = sin(PI * nearA);
  float blend = max(smoothstep(0.55, 1.75, radius), 0.70 * edgeCoverage);
  vec3 col = mix(base, layered, clamp(blend, 0.0, 1.0));
  outColor = vec4(encode(col), 1.0);
}`;

  // Split the photo into premultiplied near/far layers. RGB is filtered as
  // coverage-weighted colour and A is coverage. The second output is
  // radius*coverage, which becomes the layer's CoC pyramid.
  const LAYER_FRAG = `#version 300 es
precision highp float;
precision highp sampler2D;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_photo;
uniform sampler2D u_depth;
uniform float u_fpx;
uniform float u_fmm;
uniform float u_fstop;
uniform float u_focus;
uniform float u_scale;
uniform float u_blur;
uniform float u_highlight;
uniform float u_maxblur;
uniform float u_infoScale;
uniform int u_layer;
uniform int u_info;
uniform int u_linear;

float cocDiameter(float z) {
  float zmm = clamp(z, 0.05, 4000.0) * 1000.0;
  float smm = clamp(u_focus, 0.05, 4000.0) * 1000.0;
  float c = (u_fpx / max(u_fstop, 0.7)) * u_fmm * abs(1.0 / zmm - 1.0 / smm);
  return c * u_blur * u_scale;
}

vec3 hdrExpand(vec3 c, float spread) {
  float m = max(c.r, max(c.g, c.b));
  float t = smoothstep(0.6, 1.0, m);
  float gain = 1.0 + float(u_linear) * u_highlight * 24.0 * t * t * spread;
  return c * gain;
}

void main() {
  vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
  float z = textureLod(u_depth, uv, 0.0).r;
  float nearPart = 1.0 - step(u_focus, z);
  float belongs = u_layer == 0 ? nearPart : 1.0 - nearPart;
  float radius = 0.5 * min(cocDiameter(z), u_maxblur);

  if (u_info == 1) {
    outColor = vec4(radius * belongs / u_infoScale, 0.0, 0.0, 1.0);
    return;
  }

  vec3 c = textureLod(u_photo, uv, 0.0).rgb;
  c = hdrExpand(c, smoothstep(0.5, 6.0, radius));
  outColor = vec4(c * belongs, belongs);
}`;

  // Blur one layer using its own CoC field. The mip probe pulls a filled
  // radius and colour from coarser levels where the layer is occluded. The
  // aperture is a fixed nine-blade polygon across the frame; the per-pixel
  // phase changes sample placement, not the polygon's orientation.
  const LAYER_BLUR_FRAG = `#version 300 es
precision highp float;
precision highp sampler2D;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_layerTex;
uniform sampler2D u_infoTex;
uniform sampler2D u_depth;
uniform vec2 u_res;
uniform float u_fpx;
uniform float u_fmm;
uniform float u_fstop;
uniform float u_focus;
uniform float u_scale;
uniform float u_blur;
uniform float u_maxblur;
uniform float u_maxLod;
uniform float u_infoScale;
uniform int u_layer;
uniform int u_taps;

const int MAXTAPS = 40;
const float GOLDEN = 2.399963229728653;
const float PI = 3.14159265359;
const float TAU = 6.28318530718;

float cocDiameter(float z) {
  float zmm = clamp(z, 0.05, 4000.0) * 1000.0;
  float smm = clamp(u_focus, 0.05, 4000.0) * 1000.0;
  float c = (u_fpx / max(u_fstop, 0.7)) * u_fmm * abs(1.0 / zmm - 1.0 / smm);
  return c * u_blur * u_scale;
}

float ign(vec2 p) {
  return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

float polygonLimit(float angle, float radius) {
  const float blades = 9.0;
  float sector = TAU / blades;
  float local = mod(angle + 0.17 + sector * 0.5, sector) - sector * 0.5;
  return radius * cos(PI / blades) / max(cos(local), 0.20);
}

void main() {
  vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
  float z = textureLod(u_depth, uv, 0.0).r;
  float nearPart = 1.0 - step(u_focus, z);
  float belongs = u_layer == 0 ? nearPart : 1.0 - nearPart;
  float radius = belongs > 0.5 ? 0.5 * min(cocDiameter(z), u_maxblur) : 0.0;

  // The mip chain is the background fill. Walk coarser until this layer owns
  // some coverage, then recover its coverage-weighted source CoC.
  for (int probe = 0; probe < 12; probe++) {
    float lod = min(float(probe), u_maxLod);
    float coverage = textureLod(u_layerTex, uv, lod).a;
    if (coverage > 0.02) {
      float packedRadius = textureLod(u_infoTex, uv, lod).r * u_infoScale;
      radius = max(radius, packedRadius / max(coverage, 1e-4));
      break;
    }
    if (lod >= u_maxLod) break;
  }

  if (radius < 0.55) {
    vec4 direct = textureLod(u_layerTex, uv, 0.0);
    if (u_layer == 1) {
      float coverage = smoothstep(0.0, 0.08, direct.a);
      vec3 colour = direct.rgb / max(direct.a, 1e-4);
      outColor = vec4(colour * coverage, coverage);
    } else {
      outColor = direct;
    }
    return;
  }

  // Cat's-eye clipping is strongest toward the corners and at wide apertures.
  vec2 field = (uv - 0.5) * 2.0;
  field.x *= u_res.x / max(u_res.y, 1.0);
  float wide = 1.0 - smoothstep(1.6, 8.0, u_fstop);
  float catEye = 1.0 - 0.16 * wide * smoothstep(0.2, 1.2, length(field));
  radius *= catEye;

  float spin = ign(gl_FragCoord.xy + vec2(float(u_layer) * 19.0, 0.0)) * TAU;
  float lod = clamp(log2(max(1.0, radius * 0.45)), 0.0, u_maxLod);
  vec4 sum = vec4(0.0);

  for (int i = 0; i < MAXTAPS; i++) {
    if (i >= u_taps) break;
    float fi = float(i) + 0.5;
    float angle = fi * GOLDEN + spin;
    float limit = polygonLimit(angle, radius);
    float r = sqrt(fi / float(u_taps)) * limit;
    vec2 sampleUV = clamp(uv + vec2(cos(angle), sin(angle)) * r / u_res, vec2(0.0), vec2(1.0));
    sum += textureLod(u_layerTex, sampleUV, lod);
  }
  sum /= float(u_taps);

  if (u_layer == 1) {
    float coverage = smoothstep(0.0, 0.08, sum.a);
    vec3 colour = sum.rgb / max(sum.a, 1e-4);
    outColor = vec4(colour * coverage, coverage);
  } else {
    outColor = sum;
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
      `u_photo`, `u_depth`, `u_near`, `u_far`, `u_fpx`, `u_fmm`,
      `u_fstop`, `u_focus`, `u_scale`, `u_blur`, `u_maxblur`,
      `u_dnear`, `u_dfar`, `u_mode`, `u_linear`,
    ]);
    S.locLayer = uniformLocations(gl, S.progLayer, [
      `u_photo`, `u_depth`, `u_fpx`, `u_fmm`, `u_fstop`, `u_focus`,
      `u_scale`, `u_blur`, `u_highlight`, `u_maxblur`, `u_infoScale`,
      `u_layer`, `u_info`, `u_linear`,
    ]);
    S.locBlur = uniformLocations(gl, S.progBlur, [
      `u_layerTex`, `u_infoTex`, `u_depth`, `u_res`, `u_fpx`, `u_fmm`,
      `u_fstop`, `u_focus`, `u_scale`, `u_blur`, `u_maxblur`,
      `u_maxLod`, `u_infoScale`, `u_layer`, `u_taps`,
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
    const areaScale = Math.sqrt(MAX_WORK_PIXELS / (outputW * outputH));
    const sideScale = MAX_WORK_SIDE / Math.max(outputW, outputH);
    const scale = Math.min(1, areaScale, sideScale);
    const w = Math.max(1, Math.round(outputW * scale));
    const h = Math.max(1, Math.round(outputH * scale));
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
    gl.uniform1f(loc.u_fpx, (fmm * m.width) / 36);
    gl.uniform1f(loc.u_fmm, fmm);
    gl.uniform1f(loc.u_fstop, FSTOPS[S.fstopIdx]);
    gl.uniform1f(loc.u_focus, S.focus);
    gl.uniform1f(loc.u_scale, w / m.width);
    gl.uniform1f(loc.u_blur, S.blur);
    gl.uniform1f(loc.u_maxblur, Math.min(w, h) * 0.10);
    gl.uniform1i(loc.u_linear, linear ? 1 : 0);
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
    gl.uniform1f(S.locLayer.u_highlight, S.highlight);
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
    gl.uniform1i(S.locBlur.u_taps, Math.max(8, Math.min(40, taps || PREVIEW_TAPS)));
    setLayerUniforms(S.locBlur, t.w, t.h, linear);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, S.tex.depth);
    gl.uniform1i(S.locBlur.u_depth, 2);
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

  function renderFinal(mode, useLinear) {
    const gl = S.gl, canvas = $("view"), m = S.meta;
    const t = S.targets;
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
    gl.bindTexture(gl.TEXTURE_2D, mode === 0 && t ? t.nearBlur : S.tex.depth);
    gl.uniform1i(S.locFinal.u_near, 2);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, mode === 0 && t ? t.farBlur : S.tex.depth);
    gl.uniform1i(S.locFinal.u_far, 3);

    const fmm = effFocal();
    gl.uniform1f(S.locFinal.u_fpx, (fmm * m.width) / 36);
    gl.uniform1f(S.locFinal.u_fmm, fmm);
    gl.uniform1f(S.locFinal.u_fstop, FSTOPS[S.fstopIdx]);
    gl.uniform1f(S.locFinal.u_focus, S.focus);
    gl.uniform1f(S.locFinal.u_scale, canvas.width / m.width);
    gl.uniform1f(S.locFinal.u_blur, S.blur);
    gl.uniform1f(S.locFinal.u_maxblur, Math.min(canvas.width, canvas.height) * 0.10);
    gl.uniform1f(S.locFinal.u_dnear, S.dLo);
    gl.uniform1f(S.locFinal.u_dfar, S.dHi);
    gl.uniform1i(S.locFinal.u_mode, mode);
    gl.uniform1i(S.locFinal.u_linear, useLinear ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (S.busy) glError(gl, "final render", false);
  }

  function render(taps) {
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
    renderFinal(effMode, useLinear);
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
    $("highlightOut").textContent = S.highlight.toFixed(2);
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

      $("overlayTitle").textContent = "Loading depth";
      $("overlaySub").textContent = `${meta.width} × ${meta.height} px`;
      const [bmp, depthBuf] = await Promise.all([
        fetch(meta.image_url).then((r) => r.blob()).then((b) => createImageBitmap(b)),
        fetch(meta.depth_url).then((r) => r.arrayBuffer()),
      ]);
      S.bitmap = bmp;

      uploadTextures(bmp, depthBuf, meta);
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
        out[y * lowW + x] = halfToFloat(u16[sy * meta.width + sx]);
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

  async function exportRender({ mode, mime, ext, taps }) {
    const canvas = $("view");
    const meta = S.meta;
    const k = Math.min(1, S.maxExport / Math.max(meta.width, meta.height));
    const w = Math.round(meta.width * k);
    const h = Math.round(meta.height * k);

    overlay(true, "Rendering", `${w} × ${h} px · layered mip bokeh · ${taps} aperture taps`, 35);
    await sleep(60);

    const prevW = canvas.width, prevH = canvas.height;
    const prevMode = S.mode;
    canvas.width = w;
    canvas.height = h;
    S.mode = mode;
    render(taps);

    const blob = await new Promise((res) => canvas.toBlob(res, mime, S.quality / 100));

    S.mode = prevMode;
    canvas.width = prevW;
    canvas.height = prevH;
    layout();
    overlay(false);

    if (!blob) throw new Error("The browser could not encode the image.");
    download(blob, `${baseName()}_dof_${ext}`);
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
      await exportRender({ mode: 1, mime: "image/png", ext: "depth.png", taps: 1 });
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
      };
    });

    $("quality").addEventListener("input", (e) => {
      S.quality = +e.target.value;
      $("qualityOut").textContent = String(S.quality);
    });

    $("exportBtn").onclick = exportPhoto;
    $("saveDepth").onclick = exportDepth;

    $("view").addEventListener("click", (e) => {
      if (!S.meta || S.mode !== 0) return;
      const rect = $("view").getBoundingClientRect();
      const u = (e.clientX - rect.left) / rect.width;
      const v = (e.clientY - rect.top) / rect.height;
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
  statusProbe();
})();
