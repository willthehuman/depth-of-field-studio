/* Depth of Field Studio — client.
   Depth is estimated once on the server; focus and aperture are a WebGL
   gather-bokeh shader, so the sliders are instant and the export is full-res. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const FSTOPS = [1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 2.8, 3.2, 3.5, 4.0, 4.5, 5.0,
    5.6, 6.3, 7.1, 8.0, 9.0, 10, 11, 13, 14, 16, 18, 20, 22];
  const MAX_EXPORT = 8192;
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
    prog: null,
    tex: { photo: null, depth: null },
    loc: {},
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

  const FRAG = `#version 300 es
precision highp float;
precision highp sampler2D;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_photo;
uniform sampler2D u_depth;
uniform vec2  u_res;        // drawing buffer, px
uniform vec2  u_tex;        // photo, px
uniform float u_fpx;        // effective focal length, photo px
uniform float u_fmm;        // effective focal length, mm on a 36 mm frame
uniform float u_fstop;
uniform float u_focus;      // metres
uniform float u_scale;      // drawing-buffer px per photo px
uniform float u_blur;       // artistic multiplier
uniform float u_highlight;
uniform float u_maxblur;
uniform float u_dnear;
uniform float u_dfar;
uniform int   u_mode;       // 0 dof, 1 depth, 2 original
uniform int   u_taps;

const int MAXTAPS = 192;
const float GOLDEN = 2.399963229728653;

float cocDiameter(float z) {
  // Thin-lens circle of confusion, expressed in photo pixels.
  float zmm = clamp(z, 0.05, 4000.0) * 1000.0;
  float smm = clamp(u_focus, 0.05, 4000.0) * 1000.0;
  float c = (u_fpx / max(u_fstop, 0.7)) * u_fmm * abs(1.0 / zmm - 1.0 / smm);
  return c * u_blur * u_scale;
}

vec3 depthRamp(float t) {
  // near (0) -> far (1); bright at both ends so the map reads as a photograph
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

float ign(vec2 p) {
  return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

void main() {
  vec3 base = texture(u_photo, v_uv).rgb;

  if (u_mode == 2) { outColor = vec4(base, 1.0); return; }

  float z = texture(u_depth, v_uv).r;

  if (u_mode == 1) {
    float lz = log(max(z, 1e-3));
    float t = (lz - log(u_dnear)) / max(log(u_dfar) - log(u_dnear), 1e-4);
    outColor = vec4(depthRamp(clamp(t, 0.0, 1.0)), 1.0);
    return;
  }

  float coc = min(cocDiameter(z), u_maxblur);
  float radius = 0.5 * coc;
  if (radius < 0.55) { outColor = vec4(base, 1.0); return; }

  float spin = ign(gl_FragCoord.xy) * 6.2831853;
  vec3 sum = base;
  float wsum = 1.0;

  for (int i = 0; i < MAXTAPS; i++) {
    if (i >= u_taps) break;
    float fi = float(i) + 0.5;
    float r = sqrt(fi / float(u_taps)) * radius;
    float a = fi * GOLDEN + spin;
    vec2 off = vec2(cos(a), sin(a)) * r / u_res;
    vec2 suv = clamp(v_uv + off, vec2(0.0), vec2(1.0));

    vec3 sc = textureLod(u_photo, suv, 0.0).rgb;
    float sz = textureLod(u_depth, suv, 0.0).r;

    // Scatter-as-gather: a neighbour only reaches this pixel if its own blur
    // disc is at least as wide as the distance to it. Sharp foreground
    // therefore stays clean instead of bleeding into the background.
    float srad = 0.5 * min(cocDiameter(sz), u_maxblur);
    float w = smoothstep(r - 1.5, r + 1.5, srad);

    // Optional emphasis on specular highlights, the way a fast lens renders them.
    w *= 1.0 + u_highlight * smoothstep(0.70, 1.0, max(sc.r, max(sc.g, sc.b)));

    sum += sc * w;
    wsum += w;
  }

  vec3 col = sum / wsum;
  outColor = vec4(col, 1.0);
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

    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(prog) || "program link failed");
    }
    S.prog = prog;

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, "a_pos");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    for (const n of ["u_photo", "u_depth", "u_res", "u_tex", "u_fpx", "u_fmm", "u_fstop", "u_focus",
      "u_scale", "u_blur", "u_highlight", "u_maxblur", "u_dnear", "u_dfar", "u_mode", "u_taps"]) {
      S.loc[n] = gl.getUniformLocation(prog, n);
    }

    for (const key of ["photo", "depth"]) {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      S.tex[key] = t;
    }
  }

  function uploadTextures(bitmap, depthRaw, meta) {
    const gl = S.gl;
    const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);

    let bmp = bitmap;
    if (meta.width > maxTex || meta.height > maxTex) {
      const k = maxTex / Math.max(meta.width, meta.height);
      const c = document.createElement("canvas");
      c.width = Math.round(meta.width * k);
      c.height = Math.round(meta.height * k);
      c.getContext("2d").drawImage(bitmap, 0, 0, c.width, c.height);
      bmp = c;
    }
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.bindTexture(gl.TEXTURE_2D, S.tex.photo);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, bmp);

    gl.bindTexture(gl.TEXTURE_2D, S.tex.depth);
    const half = new Uint16Array(depthRaw);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, meta.width, meta.height, 0, gl.RED, gl.HALF_FLOAT, half);
  }

  /* --------------------------------------------------------------- rendering */

  function render(taps) {
    const gl = S.gl, canvas = $("view");
    if (!S.meta || !S.prog) return;
    if (!canvas.width || !canvas.height) return;

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(S.prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, S.tex.photo);
    gl.uniform1i(S.loc.u_photo, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, S.tex.depth);
    gl.uniform1i(S.loc.u_depth, 1);

    const m = S.meta;
    const fmm = effFocal();
    gl.uniform2f(S.loc.u_res, canvas.width, canvas.height);
    gl.uniform2f(S.loc.u_tex, m.width, m.height);
    gl.uniform1f(S.loc.u_fpx, (fmm * m.width) / 36);
    gl.uniform1f(S.loc.u_fmm, fmm);
    gl.uniform1f(S.loc.u_fstop, FSTOPS[S.fstopIdx]);
    gl.uniform1f(S.loc.u_focus, S.focus);
    gl.uniform1f(S.loc.u_scale, canvas.width / m.width);
    gl.uniform1f(S.loc.u_blur, S.blur);
    gl.uniform1f(S.loc.u_highlight, S.highlight);
    gl.uniform1f(S.loc.u_maxblur, Math.min(canvas.width, canvas.height) * 0.10);
    gl.uniform1f(S.loc.u_dnear, S.dLo);
    gl.uniform1f(S.loc.u_dfar, S.dHi);
    gl.uniform1i(S.loc.u_mode, S.holdOriginal ? 2 : S.mode);
    gl.uniform1i(S.loc.u_taps, taps || 96);

    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  function requestRender() {
    if (S.dirty) return;
    S.dirty = true;
    requestAnimationFrame(() => {
      S.dirty = false;
      render(96);
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
    const k = Math.min(1, MAX_EXPORT / Math.max(meta.width, meta.height));
    const w = Math.round(meta.width * k);
    const h = Math.round(meta.height * k);

    overlay(true, "Rendering", `${w} × ${h} px · ${taps} taps per pixel`, 35);
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
        taps: 192,
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
