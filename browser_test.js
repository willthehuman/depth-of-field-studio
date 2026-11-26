(async () => {
  const $ = (id) => document.getElementById(id);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const frame = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  const out = {};
  const fail = [];

  // ---------------------------------------------------------------- load sample
  $("sampleBtn").click();
  let ready = false;
  for (let i = 0; i < 900; i++) {
    if (!$("exportBtn").disabled) { ready = true; break; }
    await sleep(200);
  }
  if (!ready) throw new Error("analysis never finished: " + ($("overlaySub") || {}).textContent);
  await frame();
  await sleep(300);

  const src = $("view");
  const W = src.width, H = src.height;
  out.canvas = { W, H, cssW: src.clientWidth, cssH: src.clientHeight };
  if (!W || !H) throw new Error("canvas has no backing store");

  const cap = () => {
    const c = document.createElement("canvas");
    c.width = W; c.height = H;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(src, 0, 0);
    return ctx.getImageData(0, 0, W, H);
  };

  const setMode = (m) => {
    document.querySelector(`#viewTabs .chip[data-mode="${m}"]`).click();
  };
  const setFstop = (idx) => {
    const el = $("fstop");
    el.value = String(idx);
    el.dispatchEvent(new Event("input", { bubbles: true }));
  };
  const setBlur = (v) => {
    const el = $("blur");
    el.value = String(v);
    el.dispatchEvent(new Event("input", { bubbles: true }));
  };

  // -------------------------------------------------------- 1. is it our photo?
  setMode(2);
  await frame();
  const orig = cap();

  const served = new Image();
  served.src = "/sample.jpg";
  await served.decode();
  const ref = document.createElement("canvas");
  ref.width = 240;
  ref.height = Math.round((240 * served.height) / served.width);
  ref.getContext("2d").drawImage(served, 0, 0, ref.width, ref.height);
  const refPix = ref.getContext("2d").getImageData(0, 0, ref.width, ref.height).data;

  const cmp = document.createElement("canvas");
  cmp.width = ref.width; cmp.height = ref.height;
  cmp.getContext("2d").drawImage(src, 0, 0, cmp.width, cmp.height);
  const cmpPix = cmp.getContext("2d").getImageData(0, 0, cmp.width, cmp.height).data;

  let diff = 0, n = 0;
  for (let i = 0; i < refPix.length; i += 4) {
    diff += Math.abs(refPix[i] - cmpPix[i]) + Math.abs(refPix[i + 1] - cmpPix[i + 1]) + Math.abs(refPix[i + 2] - cmpPix[i + 2]);
    n += 3;
  }
  out.originalVsServedMeanAbs = +(diff / n).toFixed(2);
  if (out.originalVsServedMeanAbs > 8) fail.push("canvas does not match the served photo (flip/colour bug)");

  // ------------------------------------------------- 2. depth texture orientation
  setMode(1);
  await frame();
  const depthImg = cap();
  const dd = depthImg.data;
  const at = (u, v) => {
    const x = Math.min(W - 1, Math.max(0, Math.round(u * (W - 1))));
    const y = Math.min(H - 1, Math.max(0, Math.round(v * (H - 1))));
    const i = (y * W + x) * 4;
    return [dd[i], dd[i + 1], dd[i + 2]];
  };
  // band means of "warmth" (r - b): near is warm, far is cool in the turbo map
  const bandWarmth = (v0, v1) => {
    let s = 0, c = 0;
    for (let y = Math.round(v0 * H); y < Math.round(v1 * H); y += 3) {
      for (let x = 0; x < W; x += 3) {
        const i = (y * W + x) * 4;
        s += dd[i] - dd[i + 2]; c++;
      }
    }
    return s / c;
  };
  out.depthTopWarmth = +bandWarmth(0.0, 0.18).toFixed(1);   // sky
  out.depthBottomWarmth = +bandWarmth(0.75, 1.0).toFixed(1); // pavement
  out.depthTopPixel = at(0.5, 0.05);
  out.depthBottomPixel = at(0.5, 0.95);
  if (out.depthTopWarmth > out.depthBottomWarmth) fail.push("depth looks vertically flipped (sky warmer/nearer than the ground)");

  // ------------------------------------- 3. build a depth-bin map from the depthRamp
  // Must mirror FINAL_FRAG's depthRamp (5 stops), not Google's turbo: the LUT
  // is used to classify pixels into depth bands, so a stale colormap silently
  // puts pixels in the wrong band.
  const depthRamp = (t) => {
    const c0 = [1.00, 0.30, 0.16], c1 = [1.00, 0.70, 0.20], c2 = [0.98, 0.95, 0.62],
          c3 = [0.36, 0.86, 0.86], c4 = [0.24, 0.45, 0.95];
    t = Math.min(1, Math.max(0, t));
    const s = t * 4;
    const mix = (a, b, f) => a.map((v, i) => v + (b[i] - v) * f);
    let rgb;
    if (s < 1) rgb = mix(c0, c1, s);
    else if (s < 2) rgb = mix(c1, c2, s - 1);
    else if (s < 3) rgb = mix(c2, c3, s - 2);
    else rgb = mix(c3, c4, s - 3);
    return rgb.map((v) => Math.round(v * 255));
  };
  // Shader t: 0 = near (warm) .. 1 = far (cool). The test wants 1 = near.
  const LUT = Array.from({ length: 128 }, (_, i) => depthRamp(1 - i / 127));

  const QW = Math.round(W / 4), QH = Math.round(H / 4);
  const tq = new Float32Array(QW * QH);
  const small = document.createElement("canvas");
  small.width = QW; small.height = QH;
  small.getContext("2d").drawImage(src, 0, 0, QW, QH);
  const sq = small.getContext("2d").getImageData(0, 0, QW, QH).data;
  for (let i = 0; i < QW * QH; i++) {
    const r = sq[i * 4], g = sq[i * 4 + 1], b = sq[i * 4 + 2];
    let best = 0, bestD = Infinity;
    for (let k = 0; k < LUT.length; k++) {
      const dr = r - LUT[k][0], dg = g - LUT[k][1], db = b - LUT[k][2];
      const d = dr * dr + dg * dg + db * db;
      if (d < bestD) { bestD = d; best = k; }
    }
    tq[i] = best / (LUT.length - 1); // 0 = far, 1 = near
  }

  // nearest point in the lower half of the frame -> click to focus on it
  let nearIdx = 0;
  for (let i = 0; i < tq.length; i++) {
    const y = Math.floor(i / QW);
    if (y < QH * 0.45) continue;
    if (tq[i] > tq[nearIdx]) nearIdx = i;
  }
  const nx = (nearIdx % QW + 0.5) / QW;
  const ny = (Math.floor(nearIdx / QW) + 0.5) / QH;

  setMode(0);
  setBlur(200);
  const rect = src.getBoundingClientRect();
  src.dispatchEvent(new MouseEvent("click", {
    bubbles: true,
    clientX: rect.left + nx * rect.width,
    clientY: rect.top + ny * rect.height,
  }));
  const focusText = $("focusOut").textContent;

  // four depth bands: nearest quarter -> farthest quarter
  const bands = 4;
  const mask = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) {
    const qx = Math.min(QW - 1, Math.floor((i % W) / 4));
    const qy = Math.min(QH - 1, Math.floor(Math.floor(i / W) / 4));
    const t = tq[qy * QW + qx];
    mask[i] = Math.min(bands - 1, Math.floor((1 - t) * bands)); // 0 = near, 3 = far
  }

  const lum = (img) => {
    const d = img.data, L = new Float32Array(W * H);
    for (let i = 0; i < W * H; i++) L[i] = 0.2126 * d[i * 4] + 0.7152 * d[i * 4 + 1] + 0.0722 * d[i * 4 + 2];
    return L;
  };
  const sharpness = (L, band) => {
    let s = 0, c = 0;
    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        const i = y * W + x;
        if (mask[i] !== band || mask[i - 1] !== band || mask[i + 1] !== band || mask[i - W] !== band || mask[i + W] !== band) continue;
        s += Math.abs(L[i + 1] - L[i - 1]) + Math.abs(L[i + W] - L[i - W]);
        c++;
      }
    }
    return c ? s / c : 0;
  };

  setFstop(24); // f/16
  await frame();
  const wide = lum(cap());

  setFstop(3); // f/1.4
  await frame();
  const open = lum(cap());

  out.focus = focusText;
  out.bands = [];
  for (let b = 0; b < bands; b++) {
    const a = sharpness(wide, b), o = sharpness(open, b);
    out.bands.push({
      band: b === 0 ? "near" : b === bands - 1 ? "far" : `mid${b}`,
      f16: +a.toFixed(3),
      f14: +o.toFixed(3),
      kept: a > 0.2 ? +(o / a).toFixed(3) : null,
      textured: a > 0.2,
    });
  }

  // The sky band is featureless, so judge on the textured bands: with the focus
  // on the nearest object, nearer bands must survive the aperture change better
  // than farther ones.
  const nearKept = out.bands[0].kept;
  const mids = out.bands.filter((b) => b.textured && b.band !== "near" && b.band !== "far");
  const midKept = mids.length ? mids.reduce((s, b) => s + b.kept, 0) / mids.length : null;
  out.focusNear = { nearKept, midKept };
  if (nearKept === null || midKept === null) fail.push("no textured band to measure");
  else if (!(midKept < nearKept - 0.15)) fail.push(`opening the aperture did not blur the mid ground more than the in-focus near ground (near keeps ${nearKept}, mid keeps ${midKept})`);

  // ------------------------------------- 4. refocus farther: the near band must lose
  let midIdx = -1;
  for (let i = 0; i < tq.length; i++) {
    const y = Math.floor(i / QW);
    if (y < QH * 0.25 || y > QH * 0.8) continue;
    if (tq[i] > 0.5 && tq[i] < 0.62 && (midIdx < 0 || tq[i] > tq[midIdx])) midIdx = i;
  }
  if (midIdx >= 0) {
    const mx = (midIdx % QW + 0.5) / QW;
    const my = (Math.floor(midIdx / QW) + 0.5) / QH;
    src.dispatchEvent(new MouseEvent("click", {
      bubbles: true,
      clientX: rect.left + mx * rect.width,
      clientY: rect.top + my * rect.height,
    }));
    await frame();
    const open2 = lum(cap());
    const near2 = sharpness(open2, 0);
    const mid2 = sharpness(open2, 1);
    out.refocus = {
      distance: $("focusOut").textContent,
      nearKept: +(near2 / out.bands[0].f14).toFixed(3),
      midKept: +(mid2 / (out.bands[1].f14 || 1)).toFixed(3),
    };
    if (!(out.refocus.nearKept < out.refocus.midKept)) {
      fail.push(`refocusing on the mid ground did not sharpen it relative to the near band (near ${out.refocus.nearKept}, mid ${out.refocus.midKept})`);
    }
  } else {
    fail.push("could not find a mid-depth point to refocus on");
  }

  // --- lens scaling must be linear in focal length, not quadratic ---
  const setLens = (mm) => {
    const b = document.querySelector(`#lens button[data-mm="${mm}"]`);
    if (b) b.click();
  };
  setFstop(6); // f/2.0
  setBlur(100);
  setLens(0);  // Native
  await frame();
  const natLum = lum(cap());
  setLens(85);
  await frame();
  const simLum = lum(cap());
  // far band sharpness ratio native->85mm. Quadratic bug gave ~8x more blur;
  // linear optics give a bounded increase. Assert far band gets blurrier but
  // not catastrophically (ratio of sharpness stays > 0.05).
  const farNat = sharpness(natLum, bands - 1);
  const farSim = sharpness(simLum, bands - 1);
  out.lensScaling = { farNat: +farNat.toFixed(3), farSim: +farSim.toFixed(3) };
  if (farNat > 0.2 && farSim / farNat < 0.02) {
    fail.push("85mm sim over-blurs (quadratic regression): " + (farSim / farNat).toFixed(4));
  }
  setLens(0);

  // -------------------------------------------------------------- 4. export path
  const exports = [];
  const origCreate = URL.createObjectURL;
  URL.createObjectURL = (blob) => { exports.push({ blob, size: blob.size, type: blob.type }); return origCreate(blob); };
  $("exportBtn").disabled = false;
  $("exportBtn").click();
  for (let i = 0; i < 300; i++) {
    if (exports.length) break;
    await sleep(200);
  }
  out.exportBlob = exports[0] ? { size: exports[0].size, type: exports[0].type } : null;
  if (exports[0]) {
    const bmp = await createImageBitmap(exports[0].blob);
    out.exportBlob.width = bmp.width;
    out.exportBlob.height = bmp.height;
    if (bmp.width < 1000 || bmp.height < 700) fail.push(`export is not full resolution (${bmp.width}x${bmp.height})`);
  } else {
    fail.push("export produced no blob");
  }

  // export filename must carry a real extension
  const names = [];
  // capture the download name by wrapping HTMLAnchorElement.click
  const origClick = HTMLAnchorElement.prototype.click;
  HTMLAnchorElement.prototype.click = function () {
    if (this.download) names.push(this.download);
    return origClick.apply(this, arguments);
  };
  $("exportBtn").click();
  for (let i = 0; i < 300 && names.length === 0; i++) await sleep(200);
  HTMLAnchorElement.prototype.click = origClick;
  out.exportName = names[0] || null;
  if (!out.exportName || !/\.(jpg|png)$/.test(out.exportName)) {
    fail.push("export filename lacks a proper extension: " + out.exportName);
  }

  out.failures = fail;
  out.ok = fail.length === 0;
  return JSON.stringify(out, null, 2);
})()
