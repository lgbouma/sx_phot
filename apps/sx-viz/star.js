/* sx-viz star page: spectrum, SED model comparison, cutouts, TARS vetting, verdicts. */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const on = (id, ev, fn) => { const el = $(id); if (el) el.addEventListener(ev, fn); };
  const params = new URLSearchParams(location.search);
  const ticid = params.get("ticid");
  const state = {
    star: null, sed: null, nav: null, token: localStorage.getItem("sxviz.token") || "",
    selectedKey: null, meta: null, tarsvet: null, tarsvetIdx: 0,
  };
  const MASK = { HARD: 1, SOFT: 2, EDGE: 4, OUTLIER: 8 };
  const PLOT_LAYOUT = {
    paper_bgcolor: "#0b0b0b", plot_bgcolor: "#0b0b0b",
    font: { color: "#ddd", size: 11, family: "Arial" },
    margin: { l: 55, r: 10, t: 24, b: 40 },
    xaxis: { gridcolor: "#222", zerolinecolor: "#333" },
    yaxis: { gridcolor: "#222", zerolinecolor: "#333" },
    legend: { orientation: "h", x: 0, y: 1.08, font: { size: 10 } },
    hovermode: "closest",
  };
  const PLOT_CONFIG = { displaylogo: false, responsive: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d"] };
  const MODEL_STYLES = {
    full: ["single star", "#ff9f43", "solid", true],
    blue: ["blue-anchored", "#6cc04a", "dash", true],
    binary: ["two stars", "#c084fc", "dashdot", true],
    disk: ["star + dust", "#4fc3f7", "longdash", true],
    bin1: ["primary", "#c084fc", "dot", false],
    bin2: ["secondary", "#f06292", "dot", false],
    disk_bb: ["dust blackbody", "#4fc3f7", "dot", false],
  };

  const TARS_ROWS = [
    ["Tmag", "Tmag", (v) => v.toFixed(2)],
    ["phot_g_mean_mag", "G", (v) => v.toFixed(2)],
    ["BpmRp0", "(Bp-Rp)0", (v) => v.toFixed(3)],
    ["teff", "Teff (TARS)", (v) => v.toFixed(0) + " K"],
    ["parallax", "parallax", (v) => v.toFixed(3) + " mas"],
    ["dist_pc", "distance", (v) => v.toFixed(1) + " pc"],
    ["ruwe", "RUWE", (v) => v.toFixed(2)],
    ["non_single_star", "Gaia non_single_star", (v) => String(v)],
    ["extinction_a0", "A0", (v) => v.toFixed(3)],
    ["adopted_period", "P rot", (v) => v.toFixed(4) + " d"],
    ["adopted_period_unc", "P unc", (v) => v.toFixed(4) + " d"],
    ["median_amplitude", "amplitude", (v) => v.toFixed(2) + " %"],
    ["n_secs", "N sectors", (v) => String(v)],
    ["n_sec_ratio", "sector ratio", (v) => v.toFixed(2)],
    ["sectors", "sectors", (v) => String(v)],
    ["sector_periods", "sector periods", (v) => String(v)],
    ["sector_amplitudes", "sector amps", (v) => String(v)],
    ["final_n_contams", "N contams", (v) => String(v)],
    ["flag_possible_binary", "flag possible binary", (v) => v ? "yes" : "no"],
    ["flag_multiple_periods", "flag multiple periods", (v) => v ? "yes" : "no"],
    ["flag_doubled_period", "flag doubled period", (v) => v ? "yes" : "no"],
    ["dr3_source_id", "Gaia DR3", (v) => String(v)],
  ];
  const SUMMARY_ROWS = [
    ["n_rows", "N images", (v) => String(v)],
    ["n_used", "N used", (v) => String(v)],
    ["n_soft_used", "  of which soft-flagged", (v) => String(v)],
    ["n_masked", "N masked", (v) => String(v)],
    ["n_hard", "  hard flag bits", (v) => String(v)],
    ["n_edge", "  detector edge", (v) => String(v)],
    ["n_outlier", "  gross outliers", (v) => String(v)],
    ["n_clipped", "N spline-clipped", (v) => String(v)],
    ["n_bkg_fallback", "N bkg fallback", (v) => String(v)],
    ["n_epochs", "N epochs (days)", (v) => String(v)],
    ["med_snr", "median S/N", (v) => v.toFixed(0)],
    ["resid_mad", "median |res|/err", (v) => v.toFixed(2)],
    ["mjd_min", "MJD range", (v) => v.toFixed(0)],
    ["c_1p6_0p9", "log F1.6/F0.9", (v) => v.toFixed(3)],
    ["c_3p6_1p6", "log F3.6/F1.6", (v) => v.toFixed(3)],
    ["c_4p7_3p6", "log F4.7/F3.6", (v) => v.toFixed(3)],
  ];

  function kv(table, rows, src, highlight) {
    table.innerHTML = "";
    rows.forEach(([key, label, f]) => {
      const v = src[key];
      if (v === null || v === undefined) return;
      const tr = document.createElement("tr");
      let text;
      try { text = f(v); } catch (e) { text = String(v); }
      if (key === "mjd_min" && src.mjd_max) text = `${v.toFixed(0)}–${src.mjd_max.toFixed(0)}`;
      tr.innerHTML = `<td>${label}</td><td>${text}</td>`;
      if (highlight && highlight(key, v)) tr.classList.add("hi");
      table.appendChild(tr);
    });
  }

  async function fetchJson(url, opts) {
    const r = await fetch(url, opts);
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
    return body;
  }
  const has = (v) => v !== null && v !== undefined && Number.isFinite(v);

  // ---------------------------------------------------------------- nav
  function loadNav() {
    try {
      const nav = JSON.parse(sessionStorage.getItem("sxviz.nav") || "null");
      if (nav && Array.isArray(nav.ids)) {
        state.nav = nav;
        state.nav.ids = nav.ids.map(String);
      }
    } catch (e) { /* ignore */ }
    const idx = state.nav ? state.nav.ids.indexOf(String(ticid)) : -1;
    $("nav-pos").textContent = idx >= 0 ? `${idx + 1} / ${state.nav.ids.length}` : "";
    $("prev-star").disabled = idx <= 0;
    $("next-star").disabled = !(idx >= 0 && idx + 1 < state.nav.ids.length);
    if (state.nav && state.nav.qs) $("back-link").href = "index.html?" + state.nav.qs;
  }
  function go(delta) {
    if (!state.nav) return;
    const idx = state.nav.ids.indexOf(String(ticid));
    const next = idx + delta;
    if (idx < 0 || next < 0 || next >= state.nav.ids.length) return;
    saveNotesIfDirty().finally(() => {
      location.href = `star.html?ticid=${state.nav.ids[next]}`;
    });
  }
  function goRandom() {
    if (!state.nav || !state.nav.ids.length) return;
    const k = Math.floor(Math.random() * state.nav.ids.length);
    location.href = `star.html?ticid=${state.nav.ids[k]}`;
  }

  // ------------------------------------------------------------ verdicts
  let notesDirty = false;
  function renderVerdict(v) {
    document.querySelectorAll("[data-verdict]").forEach((b) => {
      b.classList.toggle("active", !!v && b.dataset.verdict === v.verdict);
    });
    $("notes").value = v ? v.notes || "" : "";
    notesDirty = false;
    $("verdict-status").textContent = v && v.updated_utc ? `saved ${v.updated_utc}` : "";
  }
  async function postVerdict(verdict, notes) {
    const headers = { "Content-Type": "application/json" };
    if (state.token) headers["X-SX-Token"] = state.token;
    try {
      const saved = await fetchJson("/api/verdicts", {
        method: "POST", headers,
        body: JSON.stringify({ ticid: Number(ticid), verdict, notes }),
      });
      renderVerdict(saved.verdict || saved.notes ? saved : null);
      $("verdict-status").textContent = `saved ${saved.updated_utc}`;
      return true;
    } catch (e) {
      if (String(e.message).includes("token")) {
        const t = prompt("Write token (from ~/.config/sx-viz/token on the server):", state.token);
        if (t !== null) {
          state.token = t.trim();
          localStorage.setItem("sxviz.token", state.token);
          return postVerdict(verdict, notes);
        }
      }
      $("verdict-status").textContent = "error: " + e.message;
      return false;
    }
  }
  function currentVerdict() {
    const b = document.querySelector("[data-verdict].active");
    return b ? b.dataset.verdict : "";
  }
  async function setVerdict(v) {
    const ok = await postVerdict(v, $("notes").value);
    if (ok && v && $("auto-advance").checked) go(1);
  }
  async function saveNotesIfDirty() {
    if (!notesDirty) return;
    await postVerdict(currentVerdict(), $("notes").value);
  }

  // --------------------------------------------------------------- plots
  function maskReason(bits, masked, fit) {
    const r = [];
    if (bits & MASK.HARD) r.push("hard flag");
    if (bits & MASK.EDGE) r.push("detector edge");
    if (bits & MASK.OUTLIER) r.push("gross outlier");
    if (bits & MASK.SOFT) r.push(masked ? "soft flag, rejected" : "soft flag");
    if (!masked && !fit) r.push("spline-clipped");
    return r.length ? r.join(", ") : "ok";
  }

  function spectrumTraces() {
    const sp = state.star.spectrum;
    const n = sp.wavelength_um.length;
    const bits = sp.mask_bits || new Array(n).fill(0);
    const usedClean = [], usedSoft = [], clipped = [], masked = [];
    for (let i = 0; i < n; i++) {
      if (sp.masked[i]) masked.push(i);
      else if (sp.fit_mask[i]) ((bits[i] & MASK.SOFT) ? usedSoft : usedClean).push(i);
      else clipped.push(i);
    }
    const pick = (idx, key, scale) => idx.map((i) => (sp[key][i] === null ? null : sp[key][i] * (scale || 1)));
    const hover = (idx) => idx.map((i) =>
      `${sp.wavelength_um[i].toFixed(4)} um<br>${(sp.flux_jy[i] * 1e3).toFixed(4)} ± ${(sp.flux_err_jy[i] * 1e3).toFixed(4)} mJy` +
      `<br>TJD ${sp.tjd_avg[i]}<br>${maskReason(bits[i], sp.masked[i], sp.fit_mask[i])} · flag ${sp.flag_or[i]} · bad px ${sp.n_bad_pix[i]} · bkg mode ${sp.bkg_mode[i]}` +
      `<br>bkg ${(sp.bkg_jy[i] * 1e3).toFixed(4)} mJy · x,y ${sp.x[i]},${sp.y[i]}`);
    const colorTime = $("color-time").checked;
    const ptype = n > 4000 ? "scattergl" : "scatter";
    const base = { type: ptype, mode: "markers", hoverinfo: "text" };
    const traces = [];
    const timeColor = (idx) => ({ size: 4, color: pick(idx, "tjd_avg"), colorscale: "Viridis", showscale: true,
      colorbar: { title: "TJD", thickness: 8, len: 0.5 } });
    traces.push(Object.assign({}, base, {
      name: "used", x: pick(usedClean, "wavelength_um"), y: pick(usedClean, "flux_jy", 1e3),
      error_y: { type: "data", array: pick(usedClean, "flux_err_jy", 1e3), visible: true, color: "#777", thickness: 0.8, width: 0 },
      text: hover(usedClean), customdata: usedClean,
      marker: colorTime ? timeColor(usedClean) : { size: 4, color: "#f2f2f2" },
    }));
    if (usedSoft.length) {
      traces.push(Object.assign({}, base, {
        name: "used (soft flag)", x: pick(usedSoft, "wavelength_um"), y: pick(usedSoft, "flux_jy", 1e3),
        error_y: { type: "data", array: pick(usedSoft, "flux_err_jy", 1e3), visible: true, color: "#555", thickness: 0.8, width: 0 },
        text: hover(usedSoft), customdata: usedSoft,
        marker: colorTime ? Object.assign(timeColor(usedSoft), { symbol: "circle-open", showscale: false }) : { size: 5, color: "#f2f2f2", symbol: "circle-open" },
      }));
    }
    if ($("show-clipped").checked && clipped.length) {
      traces.push(Object.assign({}, base, {
        name: "clipped", x: pick(clipped, "wavelength_um"), y: pick(clipped, "flux_jy", 1e3),
        text: hover(clipped), customdata: clipped,
        marker: { size: 6, color: "#ef7b6d", symbol: "x" },
      }));
    }
    if ($("show-masked").checked && masked.length) {
      traces.push(Object.assign({}, base, {
        name: "masked", x: pick(masked, "wavelength_um"), y: pick(masked, "flux_jy", 1e3),
        text: hover(masked), customdata: masked,
        marker: { size: 4, color: "#666", symbol: "diamond-open" },
      }));
    }
    if ($("show-spline").checked) {
      const idx = [];
      for (let i = 0; i < n; i++) if (sp.model_flux[i] !== null) idx.push(i);
      traces.push({
        type: ptype, mode: "lines", name: "spline", hoverinfo: "skip",
        x: pick(idx, "wavelength_um"), y: pick(idx, "model_flux", 1e3),
        line: { color: "#7ad7c7", width: 1.2 },
      });
    }
    if ($("show-binned").checked) {
      const b = state.star.binned;
      traces.push({
        type: "scatter", mode: "markers", name: "binned R=40", hoverinfo: "text",
        x: b.lam, y: b.flux.map((v) => (v === null ? null : v * 1e3)),
        error_y: { type: "data", array: b.err.map((v) => (v === null ? null : v * 1e3)), visible: true, color: "#f2c94c", thickness: 1 },
        text: b.lam.map((l, i) => `${l} um  n=${b.n[i]}`),
        marker: { size: 6, color: "#f2c94c", symbol: "square-open" },
      });
    }
    if ($("show-sed").checked && state.sed && state.sed.curves) {
      Object.entries(MODEL_STYLES).forEach(([key, [name, color, dash, visible]]) => {
        const c = state.sed.curves[key];
        if (!c) return;
        traces.push({
          type: "scatter", mode: "lines", name, hoverinfo: "skip",
          x: c.lam, y: c.flux.map((v) => (v === null ? null : v * 1e3)),
          line: { color, width: visible ? 1.3 : 1, dash },
          visible: visible ? true : "legendonly",
        });
      });
    }
    return traces;
  }

  function drawSpectrum() {
    const log = $("log-axes").checked;
    const sp = state.star.spectrum;
    const pos = sp.flux_jy.filter((v, i) => v !== null && v > 0 && !sp.masked[i]).map((v) => v * 1e3).sort((a, b) => a - b);
    const lo = pos.length ? pos[Math.floor(pos.length * 0.01)] : 0.001;
    const hi = pos.length ? pos[Math.floor(pos.length * 0.99)] : 1;
    const layout = Object.assign({}, PLOT_LAYOUT, {
      xaxis: Object.assign({}, PLOT_LAYOUT.xaxis, { title: "wavelength [um]", type: log ? "log" : "linear", range: log ? [Math.log10(0.72), Math.log10(5.2)] : [0.72, 5.2] }),
      yaxis: Object.assign({}, PLOT_LAYOUT.yaxis, { title: "flux [mJy]", type: log ? "log" : "linear",
        range: log ? [Math.log10(lo * 0.5), Math.log10(hi * 2)] : [0, hi * 1.3] }),
    });
    Plotly.react("spectrum-plot", spectrumTraces(), layout, PLOT_CONFIG);
  }

  function drawRatio() {
    const b = state.star.binned;
    let ref = null, label = "binned / spline";
    const bm = state.sed && state.sed.binned_models ? state.sed.binned_models : {};
    if (bm.blue) { ref = bm.blue; label = "binned / blue-anchored model"; }
    else if (b.model) { ref = b.model; }
    const traces = [];
    if (ref) {
      const y = b.flux.map((v, i) => (v === null || ref[i] === null || !(ref[i] > 0) ? null : v / ref[i]));
      const e = b.err.map((v, i) => (v === null || ref[i] === null || !(ref[i] > 0) ? null : v / ref[i]));
      traces.push({ type: "scatter", mode: "markers", name: label, x: b.lam, y,
        error_y: { type: "data", array: e, visible: true, color: "#888", thickness: 1 },
        marker: { size: 5, color: "#f2f2f2" } });
      const line = (name, arr, color, dash) => traces.push({ type: "scatter", mode: "lines", name, x: b.lam,
        y: arr.map((v, i) => (v === null || !(ref[i] > 0) ? null : v / ref[i])), line: { color, width: 1, dash } });
      if (bm.full) line("single / blue model", bm.full, MODEL_STYLES.full[1], "solid");
      if (bm.bin1 && bm.bin2) line("two stars / blue model", bm.bin1.map((v, i) => (v === null || bm.bin2[i] === null ? null : v + bm.bin2[i])), MODEL_STYLES.binary[1], "dashdot");
      if (bm.disk_star && bm.disk_bb) line("star + dust / blue model", bm.disk_star.map((v, i) => (v === null || bm.disk_bb[i] === null ? null : v + bm.disk_bb[i])), MODEL_STYLES.disk[1], "longdash");
    }
    let yr = [0.6, 1.6];
    if (traces.length) {
      const vals = traces[0].y.filter((v) => v !== null && isFinite(v)).sort((a, b) => a - b);
      if (vals.length) {
        const lo = vals[Math.floor(vals.length * 0.02)], hi = vals[Math.floor(vals.length * 0.98)];
        yr = [Math.min(0.6, 0.9 * lo), Math.max(1.6, 1.1 * hi)];
      }
    }
    const layout = Object.assign({}, PLOT_LAYOUT, {
      margin: { l: 55, r: 10, t: 10, b: 36 },
      xaxis: Object.assign({}, PLOT_LAYOUT.xaxis, { title: "wavelength [um]", type: "log", range: [Math.log10(0.72), Math.log10(5.2)] }),
      yaxis: Object.assign({}, PLOT_LAYOUT.yaxis, { title: "ratio", range: yr }),
      shapes: [{ type: "line", x0: 0.7, x1: 5.2, y0: 1, y1: 1, line: { color: "#6cc04a", width: 0.8 } }],
      legend: { orientation: "h", x: 0, y: 1.15, font: { size: 10 } },
    });
    Plotly.react("ratio-plot", traces, layout, PLOT_CONFIG);
  }

  function drawAll() { drawSpectrum(); drawRatio(); }

  // ------------------------------------------------------------- cutouts
  function overlaySvg() {
    const scale = 300 / 180.0; // px per arcsec at 0.6"/px
    const c = 150;
    const r1 = 2 * 6.1 * scale, r2 = 6 * 6.1 * scale, r3 = 8 * 6.1 * scale;
    return `<svg viewBox="0 0 300 300">
      <circle cx="${c}" cy="${c}" r="${r1.toFixed(1)}" fill="none" stroke="#7ad7c7" stroke-width="1.2"/>
      <circle cx="${c}" cy="${c}" r="${r2.toFixed(1)}" fill="none" stroke="#f2c94c" stroke-width="0.8" stroke-dasharray="4 3"/>
      <circle cx="${c}" cy="${c}" r="${r3.toFixed(1)}" fill="none" stroke="#f2c94c" stroke-width="0.8" stroke-dasharray="4 3"/>
      <line x1="10" y1="285" x2="${(10 + 60 * scale).toFixed(1)}" y2="285" stroke="#fff" stroke-width="2"/>
      <text x="${(10 + 30 * scale).toFixed(1)}" y="280" fill="#fff" font-size="11" text-anchor="middle">1'</text>
    </svg>`;
  }
  function loadSky(slot, survey) {
    const host = $(slot);
    const img = host.querySelector("img");
    host.querySelectorAll("svg").forEach((s) => s.remove());
    host.querySelector(".label").textContent = (state.meta.surveys[survey] || survey) + " …";
    img.src = `/api/cutout/${ticid}?survey=${survey}`;
    img.onload = () => {
      host.querySelector(".label").textContent = state.meta.surveys[survey] || survey;
      host.insertAdjacentHTML("beforeend", overlaySvg());
    };
    img.onerror = () => { host.querySelector(".label").textContent = `${survey}: unavailable`; };
    localStorage.setItem("sxviz." + slot, survey);
  }
  function setupSky() {
    ["a", "b"].forEach((k, j) => {
      const sel = $("survey-" + k);
      sel.innerHTML = "";
      Object.entries(state.meta.surveys).forEach(([key, name]) => {
        const o = document.createElement("option");
        o.value = key; o.textContent = name; sel.appendChild(o);
      });
      const def = localStorage.getItem("sxviz.sky-" + k) || (j === 0 ? "dss2r" : "dss2ir");
      sel.value = def;
      sel.addEventListener("change", () => loadSky("sky-" + k, sel.value));
      loadSky("sky-" + k, sel.value);
    });
  }
  function cutoutCard(c) {
    const div = document.createElement("div");
    div.className = "cutout";
    div.dataset.key = c.image_key;
    div.innerHTML = `<img alt="" src="/api/spxcut/${ticid}/${c.image_key}.png" /><span class="label">${c.wavelength_um} um</span>`;
    const img = div.querySelector("img");
    img.onload = () => { div.querySelector(".label").textContent = ""; };
    img.onerror = () => { div.querySelector(".label").textContent = `${c.wavelength_um} um: no image`; };
    div.addEventListener("click", () => selectImage(c.image_key));
    return div;
  }
  function setupSpx() {
    const grid = $("spx-grid");
    grid.innerHTML = "";
    state.star.cutouts.forEach((c) => grid.appendChild(cutoutCard(c)));
    if (state.star.cutouts.length) selectImage(state.star.cutouts[0].image_key);
  }
  function selectImage(key) {
    const sp = state.star.spectrum;
    const i = sp.image_key.indexOf(String(key));
    state.selectedKey = String(key);
    const host = $("spx-selected");
    host.querySelector("img").src = `/api/spxcut/${ticid}/${key}.png`;
    host.querySelector(".label").textContent = i >= 0 ? `SPHEREx ${sp.wavelength_um[i].toFixed(3)} um` : String(key);
    $("spx-selected-info").textContent = i >= 0
      ? `selected: image ${key} · TJD ${sp.tjd_avg[i]} · ${(sp.flux_jy[i] * 1e3).toFixed(3)} mJy · ${maskReason((sp.mask_bits || [])[i] || 0, sp.masked[i], sp.fit_mask[i])} · flag ${sp.flag_or[i]}`
      : "";
    document.querySelectorAll("#spx-grid .cutout").forEach((d) => d.classList.toggle("selected", d.dataset.key === String(key)));
  }
  function onPointClick(ev) {
    if (!ev.points || !ev.points.length) return;
    const p = ev.points[0];
    if (p.customdata === undefined) return;
    selectImage(state.star.spectrum.image_key[p.customdata]);
  }

  // -------------------------------------------------------- TARS vetting
  async function loadTarsVet() {
    const host = $("tarsvet");
    const row = state.star.row;
    $("period-text").textContent = row.adopted_period ? `P = ${row.adopted_period.toFixed(4)} d, amplitude ${row.median_amplitude ?? "?"} %` : "no TARS period";
    host.innerHTML = '<span class="muted small">loading…</span>';
    let d;
    try { d = await fetchJson(`/api/tarsvet/${ticid}`); } catch (e) {
      host.innerHTML = `<span class="muted small">unavailable: ${e.message}</span>`;
      return;
    }
    $("lcviz-link").href = d.star_url;
    state.tarsvet = d;
    state.tarsvetIdx = 0;
    if (d.error) {
      host.innerHTML = `<p class="muted small">${d.error}</p>`;
    } else if (!d.images.length) {
      host.innerHTML = '<p class="muted small">no vetting images</p>';
    } else {
      host.innerHTML = '<a id="tarsvet-a" target="_blank"><img id="tarsvet-img" alt="" /></a>';
    }
    renderTarsVet();
  }

  function renderTarsVet() {
    const d = state.tarsvet;
    const n = d && d.images ? d.images.length : 0;
    $("tarsvet-prev").disabled = n < 2;
    $("tarsvet-next").disabled = n < 2;
    if (!n) { $("tarsvet-pos").textContent = ""; return; }
    const i = ((state.tarsvetIdx % n) + n) % n;
    state.tarsvetIdx = i;
    const im = d.images[i];
    const label = im.sector !== null ? `sector ${im.sector}` : im.name;
    $("tarsvet-pos").textContent = `${label} (${i + 1}/${n})`;
    $("tarsvet-a").href = im.url;
    const img = $("tarsvet-img");
    img.src = im.url; img.alt = im.name;
    // Preload the neighbours so paging is instant.
    [i + 1, i - 1].forEach((j) => { if (n > 1) new Image().src = d.images[((j % n) + n) % n].url; });
  }

  function stepTarsVet(delta) {
    if (!state.tarsvet || !state.tarsvet.images || state.tarsvet.images.length < 2) return;
    state.tarsvetIdx += delta;
    renderTarsVet();
  }

  // ------------------------------------------------------------------ SED
  function renderModelTable() {
    const t = $("model-table");
    t.innerHTML = "";
    $("model-verdict").textContent = "";
    if (!state.sed || !state.sed.fit) return;
    const f = state.sed.fit;
    const row = state.star.row;
    const fmtR = (v) => (has(v) ? v.toFixed(2) : "–");
    const rows = [
      ["single star", `${f.teff_full} K, R ${fmtR(f.radius_full)}`, f.chi2_full, f.dof_full, f.bic_single, 0],
      ["two stars", has(f.teff_bin1) ? `${f.teff_bin1} + ${f.teff_bin2} K, R ${fmtR(f.radius_bin1)} + ${fmtR(f.radius_bin2)}` : "–",
        f.chi2_binary, f.dof_binary, f.bic_binary, f.dbic_binary],
      ["star + dust", has(f.teff_disk) ? `${f.teff_disk_star} K, R ${fmtR(f.radius_disk_star)}; dust ${f.teff_disk.toFixed(0)} K, L_d/L_* ${f.f_disk.toExponential(1)}` : "–",
        f.chi2_disk, f.dof_disk, f.bic_disk, f.dbic_disk],
    ];
    const head = document.createElement("tr");
    head.innerHTML = "<th>model</th><th>chi2 / dof</th><th>BIC</th><th>ΔBIC vs single</th>";
    t.appendChild(head);
    let best = null;
    rows.forEach((r) => { if (has(r[4]) && (best === null || r[4] < best[4])) best = r; });
    rows.forEach((r) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${r[0]}</td><td>${has(r[2]) ? `${r[2].toFixed(0)} / ${r[3]}` : "–"}</td>` +
        `<td>${has(r[4]) ? r[4].toFixed(0) : "–"}</td><td>${has(r[5]) ? (r[5] > 0 ? "+" : "") + r[5].toFixed(0) : "–"}</td>`;
      if (best && r === best) tr.classList.add("best");
      t.appendChild(tr);
      const sub = document.createElement("tr");
      sub.className = "params";
      sub.innerHTML = `<td colspan="4" style="text-align:left; padding-left:14px">${r[1]}</td>`;
      t.appendChild(sub);
    });
    const parts = [];
    if (best) parts.push(`lowest BIC: ${best[0]}`);
    if (has(f.dbic_disk_vs_binary)) parts.push(`ΔBIC (two stars − star+dust) = ${f.dbic_disk_vs_binary > 0 ? "+" : ""}${f.dbic_disk_vs_binary.toFixed(0)}`);
    if (has(row.red_excess_rel_sig)) parts.push(`Teff-relative red excess ${(100 * row.red_excess_rel).toFixed(1)}% (${row.red_excess_rel_sig.toFixed(1)}σ)`);
    if (has(row.radius_ratio_full_rel)) parts.push(`radius ratio (Teff-rel) ${row.radius_ratio_full_rel.toFixed(2)}`);
    $("model-verdict").textContent = parts.join(" · ");
  }

  function renderSed() {
    renderModelTable();
    const t = $("sed-table");
    t.innerHTML = "";
    if (!state.sed || !state.sed.fit) return;
    const f = state.sed.fit;
    const row = state.star.row;
    const pct = (v) => (has(v) ? `${(100 * v).toFixed(1)} %` : "");
    const sig = (v, e) => (has(v) && e ? ` (${(v / e).toFixed(1)}σ)` : "");
    const rows = [
      ["bins used", f.n_bins_used],
      ["single: Teff", has(f.teff_full) ? `${f.teff_full} K [${f.teff_full_lo}–${f.teff_full_hi}] (${f.grid_full}, log g ${f.logg_full})` : ""],
      ["single: radius", has(f.radius_full) ? `${f.radius_full.toFixed(3)} Rsun = ${f.radius_ratio_full.toFixed(2)} × dwarf` + (has(row.radius_ratio_full_rel) ? `, ${row.radius_ratio_full_rel.toFixed(2)} × population` : "") : "(no parallax)"],
      ["blue-anchored: Teff", has(f.teff_blue) ? `${f.teff_blue} K (${f.grid_blue}; chi2 ${f.chi2_blue.toFixed(0)} / ${f.dof_blue})` : ""],
      ["blue-anchored: radius", has(f.radius_blue) ? `${f.radius_blue.toFixed(3)} Rsun (${f.radius_ratio_blue.toFixed(2)} × dwarf)` : ""],
      ["excess 1.64–2.42 um (b3)", pct(f.excess_b3) + sig(f.excess_b3, f.excess_b3_err)],
      ["excess 2.42–3.82 um (b4)", pct(f.excess_b4) + sig(f.excess_b4, f.excess_b4_err)],
      ["excess 3.82–4.42 um (b5)", pct(f.excess_b5) + sig(f.excess_b5, f.excess_b5_err)],
      ["excess 4.42–5.00 um (b6)", pct(f.excess_b6) + sig(f.excess_b6, f.excess_b6_err)],
      ["red excess (> 2.4 um)", pct(f.red_excess) + sig(f.red_excess, f.red_excess_err)],
      ["red excess, Teff-relative", has(row.red_excess_rel) ? pct(row.red_excess_rel) + (has(row.red_excess_rel_sig) ? ` (${row.red_excess_rel_sig.toFixed(1)}σ)` : "") : ""],
      ["two stars: Teff", has(f.teff_bin1) ? `${f.teff_bin1} + ${f.teff_bin2} K` : ""],
      ["two stars: radii", has(f.radius_bin1) ? `${f.radius_bin1.toFixed(3)} + ${f.radius_bin2.toFixed(3)} Rsun` : ""],
      ["two stars: Δchi2 vs single", has(f.dchi2_binary) ? f.dchi2_binary.toFixed(0) : ""],
      ["star + dust: dust T", has(f.teff_disk) ? `${f.teff_disk.toFixed(0)} K` : ""],
      ["star + dust: L_dust / L_star", has(f.f_disk) ? f.f_disk.toExponential(2) : ""],
      ["star + dust: dust radius", has(f.radius_disk) ? `${f.radius_disk.toFixed(2)} Rsun (sqrt of area)` : ""],
      ["star + dust: Δchi2 vs single", has(f.dchi2_disk) ? f.dchi2_disk.toFixed(0) : ""],
    ];
    rows.forEach(([k, v]) => {
      if (v === "" || v === undefined || v === null) return;
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${k}</td><td>${v}</td>`;
      if ((k.startsWith("red excess, Teff") && row.red_excess_rel_sig > 5)) tr.classList.add("hi");
      t.appendChild(tr);
    });
  }

  // ----------------------------------------------------------------- init
  function links(row) {
    const ra = row.ra, dec = row.dec;
    const items = [
      [`https://exofop.ipac.caltech.edu/tess/target.php?id=${ticid}`, "ExoFOP"],
      [`https://simbad.cds.unistra.fr/simbad/sim-coo?Coord=${ra}+${dec}&Radius=20&Radius.unit=arcsec`, "SIMBAD"],
      [`https://vizier.cds.unistra.fr/viz-bin/VizieR-5?-source=I/355/gaiadr3&-c=${ra}+${dec}&-c.rs=10`, "Gaia DR3"],
      [`https://aladin.cds.unistra.fr/AladinLite/?target=${ra}%20${dec}&fov=0.05`, "Aladin"],
      [`https://irsa.ipac.caltech.edu/applications/finderchart/?searchRa=${ra}&searchDec=${dec}`, "IRSA finder"],
      [`${state.meta.lcviz_url || "http://lcviz.lgbouma.com:8056"}/star.html?ticid=${ticid}`, "lcviz"],
      [`/api/star/${ticid}`, "JSON"],
    ];
    $("links").innerHTML = items.map(([u, n]) => `<a href="${u}" target="_blank">${n}</a>`).join(" · ");
  }

  async function init() {
    if (!ticid) { $("star-title").textContent = "no ticid"; return; }
    loadNav();
    on("prev-star", "click", () => go(-1));
    on("next-star", "click", () => go(1));
    on("random-star", "click", goRandom);
    document.querySelectorAll("[data-verdict]").forEach((b) => b.addEventListener("click", () => setVerdict(b.dataset.verdict)));
    on("clear-verdict", "click", () => postVerdict("", ""));
    on("save-notes", "click", () => postVerdict(currentVerdict(), $("notes").value));
    on("notes", "input", () => { notesDirty = true; });
    ["show-masked", "show-clipped", "show-spline", "show-binned", "show-sed", "log-axes", "color-time"].forEach((id) => {
      on(id, "change", drawSpectrum);
    });
    on("tarsvet-prev", "click", () => stepTarsVet(-1));
    on("tarsvet-next", "click", () => stepTarsVet(1));
    document.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") {
        if (e.key === "Escape") e.target.blur();
        return;
      }
      if (e.key === "j" || e.key === "ArrowRight") go(1);
      else if (e.key === "k" || e.key === "ArrowLeft") go(-1);
      else if (e.key === "r") goRandom();
      else if (e.key === "1") setVerdict("binary");
      else if (e.key === "2") setVerdict("single");
      else if (e.key === "3") setVerdict("unsure");
      else if (e.key === "4") setVerdict("junk");
      else if (e.key === "0") postVerdict("", "");
      else if (e.key === "m") { $("show-masked").checked = !$("show-masked").checked; drawSpectrum(); }
      else if (e.key === "n") { e.preventDefault(); $("notes").focus(); }
      else if (e.key === "[") stepTarsVet(-1);
      else if (e.key === "]") stepTarsVet(1);
    });

    const t0 = performance.now();
    try {
      [state.meta, state.star] = await Promise.all([fetchJson("/api/meta"), fetchJson(`/api/star/${ticid}`)]);
    } catch (e) {
      $("star-title").textContent = `TIC ${ticid}: ${e.message}`;
      return;
    }
    const row = state.star.row;
    if (row.parallax > 0) row.dist_pc = 1000 / row.parallax;
    state.sed = state.star.sed || null;
    document.title = `sx-viz | TIC ${ticid}`;
    $("star-title").textContent = `TIC ${ticid}`;
    $("star-sub").textContent = ` RA ${row.ra.toFixed(5)}  Dec ${row.dec.toFixed(5)}  ·  T=${row.Tmag?.toFixed(2)}  Teff=${row.teff?.toFixed(0)} K  P=${row.adopted_period?.toFixed(3)} d  ·  loaded in ${(performance.now() - t0).toFixed(0)} ms`;
    kv($("tars-table"), TARS_ROWS, row, (k, v) => (k === "ruwe" && v > 1.4) || (k === "non_single_star" && v > 0) || (k === "flag_possible_binary" && v));
    kv($("summary-table"), SUMMARY_ROWS, row);
    links(row);
    renderVerdict(state.star.verdict);
    $("sed-json").href = `/api/sed/${ticid}`;
    if (!state.sed) {
      try { state.sed = await fetchJson(`/api/sed/${ticid}`); } catch (e) { /* no fit */ }
    }
    renderSed();
    drawAll();
    $("spectrum-plot").on("plotly_click", onPointClick);
    setupSky();
    setupSpx();
    $("report-link").href = `/api/report/${ticid}.png`;
    $("report-img").src = `/api/report/${ticid}.png`;
    loadTarsVet();
  }

  window.addEventListener("error", (e) => {
    const el = $("star-sub");
    if (el) el.textContent = "JS error: " + e.message;
  });
  window.addEventListener("unhandledrejection", (e) => {
    const el = $("star-sub");
    if (el) el.textContent = "error: " + (e.reason && e.reason.message ? e.reason.message : e.reason);
  });

  init();
})();
