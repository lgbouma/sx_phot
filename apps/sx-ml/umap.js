/* sx-ml UMAP explorer: WebGL scatter of ~1e6 spectra, colour by any column,
 * star panel (binned spectrum + empirical template fits), lasso selections,
 * a Gaia colour-magnitude panel linked to the map, and a switch between the
 * datasets the server holds (e.g. TARS rotators vs all TARS targets). */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const API = "api/";
  const TARVIZ = "http://lcviz.lgbouma.com:8056";
  const showError = (msg) => {
    $("status").textContent = "error: " + msg;
    $("hud").textContent = "error: " + msg;
  };
  window.addEventListener("error", (e) => showError(e.message));
  window.addEventListener("unhandledrejection", (e) => showError(e.reason && e.reason.message || e.reason));

  // ---- colormaps (anchor colours, interpolated to 256 entries) ----------
  const ANCHORS = {
    viridis: [[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],[31,158,137],[53,183,121],[109,205,89],[180,222,44],[253,231,37]],
    magma: [[0,0,4],[28,16,68],[79,18,123],[129,37,129],[181,54,122],[229,80,100],[251,135,97],[254,194,135],[252,253,191]],
    turbo: [[48,18,59],[70,107,227],[40,180,240],[31,228,181],[110,251,98],[197,239,51],[253,186,43],[246,112,22],[205,42,6],[122,4,3]],
    coolwarm: [[59,76,192],[98,130,234],[141,176,254],[184,208,249],[221,221,221],[245,196,173],[244,154,123],[222,96,77],[180,4,38]],
    grey: [[30,30,30],[235,235,235]],
    density: [[62,64,84],[118,122,150],[190,195,215],[250,250,252]],
  };
  const LUTS = {};
  function lut(name, flip) {
    const key = name + (flip ? "_r" : "");
    if (LUTS[key]) return LUTS[key];
    const a = ANCHORS[name] || ANCHORS.viridis;
    const out = new Uint8Array(256 * 3);
    for (let i = 0; i < 256; i++) {
      let f = i / 255;
      if (flip) f = 1 - f;
      const x = f * (a.length - 1);
      const k = Math.min(a.length - 2, Math.floor(x));
      const w = x - k;
      for (let c = 0; c < 3; c++) out[3 * i + c] = Math.round(a[k][c] * (1 - w) + a[k + 1][c] * w);
    }
    LUTS[key] = out;
    return out;
  }

  const state = {
    meta: null, ds: null, datasets: [], emb: null, col: null, values: null, ticids: null,
    vmin: null, vmax: null, cmap: "viridis", flip: false, logc: false,
    selection: null, flags: null, filter: null, star: null, colCache: {},
    hoverCols: ["teff_emp", "teff_tars", "bprp0", "mg0", "tmag", "dbic_binary", "red_excess"],
  };
  try { state.specLegend = localStorage.getItem("sxml.specLegend") !== "0"; } catch (e) { state.specLegend = true; }
  try { state.hrShown = localStorage.getItem("sxml.hr") !== "0"; } catch (e) { state.hrShown = true; }
  function setSpecLegend(on) {
    state.specLegend = on;
    $("speclegend").textContent = on ? "hide legend" : "show legend";
    try { localStorage.setItem("sxml.specLegend", on ? "1" : "0"); } catch (e) { /* private mode */ }
    if ($("spec").data) Plotly.relayout($("spec"), { showlegend: on });
  }
  function setHRShown(on) {
    state.hrShown = on;
    $("hrhide").textContent = on ? "hide" : "show";
    $("hrwrap").classList.toggle("hidden", !on);
    try { localStorage.setItem("sxml.hr", on ? "1" : "0"); } catch (e) { /* private mode */ }
    if (on) hr.resize();
  }

  let scatter, hr;
  try {
    scatter = new Scatter($("glcanvas"), $("overlay"), {
      onHover: (i, e) => showTip(i, e, $("tooltip"), $("overlay")),
      onClick: (i) => loadStar(i), onLasso: onLasso,
    });
    hr = new HRPanel($("hrwrap"), {
      onHover: (i, e) => showTip(i, e, $("hrtip"), $("hrwrap")),
      onClick: (i) => loadStar(i), onLasso: onLasso,
      onDraw: (n) => { $("hrcount").textContent = n.toLocaleString() + " in view"; },
      lut: lut("density"),
    });
  } catch (err) {
    showError(err.message + " (the UMAP page needs WebGL2)");
    throw err;
  }
  window.sxScatter = scatter;  // for debugging from the console
  window.sxHR = hr;

  // ---- helpers ------------------------------------------------------------
  function url(path, params) {
    const p = new URLSearchParams(params || {});
    if (state.ds) p.set("ds", state.ds);
    const q = p.toString();
    return API + path + (q ? "?" + q : "");
  }
  async function getJSON(u) {
    const r = await fetch(u);
    if (!r.ok) throw new Error(`${u}: ${r.status}`);
    return r.json();
  }
  async function getF32(u) {
    const r = await fetch(u);
    if (!r.ok) throw new Error(`${u}: ${r.status}`);
    return new Float32Array(await r.arrayBuffer());
  }
  async function getColumn(name) {
    if (!state.colCache[name]) state.colCache[name] = await getF32(url("col/" + name));
    return state.colCache[name];
  }
  function hasColumn(name) {
    return !!(state.meta && state.meta.columns.find((c) => c.name === name));
  }
  function fmt(v, col) {
    if (v === null || v === undefined || Number.isNaN(v)) return "";
    if (typeof v === "boolean") return v ? "yes" : "no";
    if (typeof v !== "number") return String(v);
    if (Number.isInteger(v)) return String(v);
    const a = Math.abs(v);
    if (col && /^(teff|chi2|bic|dbic|dchi2)/.test(col)) return v.toFixed(0);
    if (a >= 1000) return v.toFixed(0);
    if (a >= 10) return v.toFixed(2);
    if (a >= 0.01) return v.toFixed(3);
    return v.toExponential(2);
  }
  function setStatus(s) { $("status").textContent = s; }

  // ---- colours ----------------------------------------------------------------
  function percentile(arr, p) {
    const vals = [];
    const step = Math.max(1, Math.floor(arr.length / 200000));
    for (let i = 0; i < arr.length; i += step) if (Number.isFinite(arr[i])) vals.push(arr[i]);
    if (!vals.length) return NaN;
    vals.sort((a, b) => a - b);
    return vals[Math.min(vals.length - 1, Math.floor(p * (vals.length - 1)))];
  }
  function autoRange() {
    const v = state.values;
    let lo = percentile(v, 0.01), hi = percentile(v, 0.99);
    if (!(hi > lo)) { lo = percentile(v, 0); hi = percentile(v, 1); }
    if (!(hi > lo)) { hi = lo + 1; }
    state.vmin = lo; state.vmax = hi;
    $("vmin").value = fmt(lo); $("vmax").value = fmt(hi);
  }
  function applyColors() {
    const v = state.values;
    if (!v) return;
    const n = v.length;
    const rgb = new Uint8Array(3 * n);
    const L = lut(state.cmap, state.flip);
    let lo = state.vmin, hi = state.vmax;
    const log = state.logc && lo > 0;
    if (log) { lo = Math.log10(lo); hi = Math.log10(hi); }
    const inv = 1 / (hi - lo);
    for (let i = 0; i < n; i++) {
      let x = v[i];
      if (!Number.isFinite(x) || (log && x <= 0)) { rgb[3 * i] = 90; rgb[3 * i + 1] = 90; rgb[3 * i + 2] = 95; continue; }
      if (log) x = Math.log10(x);
      let f = (x - lo) * inv;
      f = f < 0 ? 0 : f > 1 ? 1 : f;
      const k = 3 * Math.round(f * 255);
      rgb[3 * i] = L[k]; rgb[3 * i + 1] = L[k + 1]; rgb[3 * i + 2] = L[k + 2];
    }
    scatter.setColors(rgb);
    hr.setColor(v, state.vmin, state.vmax, L, state.logc);
    drawColorbar();
    updateHash();
  }
  function drawColorbar() {
    const c = $("cbcanvas"), ctx = c.getContext("2d");
    const L = lut(state.cmap, state.flip);
    const img = ctx.createImageData(256, 1);
    for (let i = 0; i < 256; i++) { img.data[4 * i] = L[3 * i]; img.data[4 * i + 1] = L[3 * i + 1]; img.data[4 * i + 2] = L[3 * i + 2]; img.data[4 * i + 3] = 255; }
    const tmp = document.createElement("canvas"); tmp.width = 256; tmp.height = 1;
    tmp.getContext("2d").putImageData(img, 0, 0);
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.drawImage(tmp, 0, 0, c.width, c.height);
    $("cbmin").textContent = fmt(state.vmin, state.col);
    $("cbmax").textContent = fmt(state.vmax, state.col);
    $("cbname").textContent = state.col + (state.logc ? " (log)" : "");
  }

  async function setColumn(name) {
    if (!name) return;
    state.col = name;
    $("col").value = name;
    $("colsearch").value = name;
    const m = state.meta.columns.find((c) => c.name === name);
    $("colnote").textContent = m ? (m.note || "") + ` · n=${m.n.toLocaleString()}` : "";
    setStatus(`loading ${name}…`);
    state.values = await getColumn(name);
    autoRange();
    applyColors();
    setStatus(`${state.meta.n.toLocaleString()} stars · ${state.emb}`);
  }

  // ---- embedding -----------------------------------------------------------------
  async function setEmbedding(name) {
    state.emb = name;
    $("emb").value = name;
    setStatus(`loading embedding ${name}…`);
    const xy = await getF32(url("points", { emb: name }));
    scatter.setPoints(xy);
    if (state.flags) await applyFilter(false);
    if (state.selection) scatter.setSelection(state.selection);
    setStatus(`${state.meta.n.toLocaleString()} stars · ${name}`);
    updateHash();
  }

  // ---- filter ---------------------------------------------------------------------
  async function applyFilter(fromUI) {
    if (fromUI) {
      const col = $("fcol").value;
      if (!col) { state.filter = null; }
      else {
        state.filter = { col, min: parseFloat($("fmin").value), max: parseFloat($("fmax").value) };
      }
    }
    const n = state.meta.n;
    const flags = new Uint8Array(n).fill(1);
    if (state.filter) {
      const v = await getColumn(state.filter.col);
      const lo = Number.isFinite(state.filter.min) ? state.filter.min : -Infinity;
      const hi = Number.isFinite(state.filter.max) ? state.filter.max : Infinity;
      let kept = 0;
      for (let i = 0; i < n; i++) {
        const x = v[i];
        if (Number.isFinite(x) && x >= lo && x <= hi) kept++; else flags[i] = 0;
      }
      $("fstatus").textContent = `${kept.toLocaleString()} shown`;
    } else {
      $("fstatus").textContent = "";
    }
    state.flags = flags;
    scatter.setFlags(flags);   // zeroes the stars without embedding coordinates
    hr.setFlags(flags);
    updateHash();
  }

  // ---- hover / star ----------------------------------------------------------------
  let hoverVals = {};
  async function prepHoverCols() {
    hoverVals = {};
    for (const c of state.hoverCols) {
      if (hasColumn(c)) hoverVals[c] = await getColumn(c);
    }
  }
  function showTip(i, e, tip, anchor) {
    if (i < 0) { tip.style.display = "none"; return; }
    const parts = [`TIC ${state.ticids ? state.ticids[i] : i}`];
    if (state.values) parts.push(`${state.col}=${fmt(state.values[i], state.col)}`);
    for (const c of Object.keys(hoverVals)) if (c !== state.col) parts.push(`${c}=${fmt(hoverVals[c][i], c)}`);
    tip.innerHTML = parts.join("<br>");
    tip.style.display = "block";
    const rect = anchor.getBoundingClientRect();
    let left = e.clientX - rect.left + 14, top = e.clientY - rect.top + 10;
    if (left + 150 > rect.width) left = Math.max(0, e.clientX - rect.left - 160);
    if (top + 20 * parts.length > rect.height) top = Math.max(0, e.clientY - rect.top - 16 * parts.length - 10);
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  }

  const SHOW_INFO = ["ticid", "in_table2", "teff_emp", "teff_tars", "bprp0", "mg0", "dms", "dmag_ms",
    "tmag", "plx", "a0", "ruwe", "nss", "n_contams", "period", "amplitude", "med_snr",
    "n_bins_used", "chi2_red", "red_excess", "red_excess_sig", "blue_excess",
    "dbic_binary", "dbic_binary_pct", "teff_bin1", "teff_bin2", "flux_ratio_binary",
    "dbic_binary_ms", "dbic_disk", "teff_disk", "disk_frac_b6", "dbic_disk_vs_binary",
    "resid_a1", "resid_a2", "max_abs_z", "lam_max_abs_z", "sequence_train",
    "sed_teff_full", "sed_dbic_binary_pct", "sed_red_excess_rel", "flag_possible_binary"];

  async function loadStar(i) {
    if (i < 0) return;
    showTab("star");
    scatter.setMarker(i);
    hr.setMarker(i);
    const s = await getJSON(url("star/" + i));
    state.star = s;
    const info = s.info;
    const links = s.links || {};
    // Older servers send no tar-viz link: build it for the TARS rotators.
    if (!links.tarviz && links.sxviz) links.tarviz = `${TARVIZ}/star.html?ticid=${s.ticid}`;
    $("startitle").innerHTML = `<b>TIC ${s.ticid}</b>` +
      (links.sxviz ? ` · <a href="${links.sxviz}" target="_blank">sx-viz page</a>` : "") +
      (links.tarviz ? ` · <a href="${links.tarviz}" target="_blank">tar-viz page</a>` : "") +
      (info.teff_emp ? ` · Teff_emp ${fmt(info.teff_emp, "teff")} K` : "");
    plotStar(s);
    const tbl = $("starinfo");
    tbl.innerHTML = "";
    for (const k of SHOW_INFO) {
      if (!(k in info) || info[k] === null) continue;
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${k}</td><td>${fmt(info[k], k)}</td>`;
      tbl.appendChild(tr);
    }
    updateHash();
  }

  function clearStar() {
    state.star = null;
    scatter.setMarker(-1);
    hr.setMarker(-1);
    $("startitle").textContent = "click a point";
    $("starinfo").innerHTML = "";
    for (const id of ["spec", "resid"]) if ($(id).data) Plotly.purge($(id));
  }

  function plotStar(s) {
    const lam = s.lam;
    const toMJy = (a) => a.map((v) => (v === null ? null : 1e3 * v));
    const traces = [{
      x: lam, y: toMJy(s.flux_dered), error_y: { type: "data", array: toMJy(s.err_dered), visible: true, thickness: 0.8, width: 0 },
      mode: "markers", type: "scatter", name: "binned (dereddened)", marker: { color: "#e6e6e6", size: 5 },
    }];
    const c = s.curves || {};
    const style = {
      single: ["#ffd34e", "solid", "single template"],
      binary: ["#6fb3ff", "dashdot", "two templates"],
      bin1: ["#6fb3ff", "dot", "primary"], bin2: ["#6fb3ff", "dot", "secondary"],
      disk: ["#ff8c69", "longdash", "template + blackbody"],
      disk_bb: ["#ff8c69", "dot", "blackbody"],
    };
    for (const k of ["single", "binary", "bin1", "bin2", "disk", "disk_bb"]) {
      if (!c[k]) continue;
      const [col, dash, name] = style[k];
      traces.push({ x: lam, y: toMJy(c[k]), mode: "lines", type: "scatter", name,
        line: { color: col, dash, width: k === "single" ? 2 : 1.2 }, visible: (k === "bin1" || k === "bin2" || k === "disk_bb") ? "legendonly" : true });
    }
    const layout = baseLayout("wavelength [um]", "F_nu [mJy]", true, true);
    layout.legend = { x: 1, y: 1, xanchor: "right", font: { size: 10 } };
    layout.showlegend = state.specLegend;
    Plotly.react($("spec"), traces, layout, { displayModeBar: false, responsive: true });
    // Residuals.
    const rtr = [];
    if (c.single) {
      const ratio = s.flux_dered.map((v, i) => (v === null || !c.single[i] ? null : 100 * (v / c.single[i] - 1)));
      const sig = s.sigma_eff ? s.sigma_eff.map((v, i) => (v === null || !c.single[i] ? null : 100 * v / c.single[i])) : null;
      if (sig) {
        rtr.push({ x: lam, y: sig, mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" });
        rtr.push({ x: lam, y: sig.map((v) => (v === null ? null : -v)), mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: "rgba(255,211,78,0.15)", name: "±1σ (population)", hoverinfo: "skip" });
      }
      rtr.push({ x: lam, y: ratio, mode: "markers", type: "scatter", name: "data / single − 1", marker: { color: "#e6e6e6", size: 5 } });
      if (c.binary) rtr.push({ x: lam, y: c.binary.map((v, i) => 100 * (v / c.single[i] - 1)), mode: "lines", name: "binary / single − 1", line: { color: "#6fb3ff", dash: "dashdot", width: 1.2 } });
      if (c.disk) rtr.push({ x: lam, y: c.disk.map((v, i) => 100 * (v / c.single[i] - 1)), mode: "lines", name: "disk / single − 1", line: { color: "#ff8c69", dash: "longdash", width: 1.2 } });
    }
    const rl = baseLayout("wavelength [um]", "residual [%]", true, false);
    rl.showlegend = false;
    rl.shapes = [{ type: "line", x0: 0.7, x1: 5.2, y0: 0, y1: 0, line: { color: "#777", width: 0.8 } }];
    Plotly.react($("resid"), rtr, rl, { displayModeBar: false, responsive: true });
  }

  function baseLayout(xt, yt, logx, logy) {
    return {
      margin: { l: 55, r: 10, t: 10, b: 40 }, paper_bgcolor: "#1b1c22", plot_bgcolor: "#121216",
      font: { color: "#e6e6e6", size: 11 },
      xaxis: { title: xt, type: logx ? "log" : "linear", gridcolor: "#2c2d36", zeroline: false, range: logx ? [Math.log10(0.72), Math.log10(5.2)] : undefined },
      yaxis: { title: yt, type: logy ? "log" : "linear", gridcolor: "#2c2d36", zeroline: false },
      showlegend: true, legend: { font: { size: 10 } },
    };
  }

  // ---- selection ------------------------------------------------------------------
  async function onLasso(indices) {
    if (!indices.length) return;
    state.selection = indices;
    scatter.setSelection(indices);
    hr.setSelection(indices);
    $("selinfo").textContent = `${indices.length.toLocaleString()} stars`;
    showTab("sel");
    $("seltitle").textContent = `computing statistics for ${indices.length.toLocaleString()} stars…`;
    const r = await fetch(url("selection"), { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rows: Array.from(indices.length > 200000 ? indices.subarray(0, 200000) : indices) }) });
    const s = await r.json();
    state.selStats = s;
    renderSelection(s);
  }

  function renderSelection(s) {
    $("seltitle").innerHTML = `<b>${s.n.toLocaleString()} stars</b>` + (s.teff_med ? ` · median t=${s.t_med.toFixed(3)} (Teff ${s.teff_med.toFixed(0)} K)` : "");
    const lam = s.lam;
    const pct = (a) => a.map((v) => (v === null ? null : 100 * (Math.pow(10, v) - 1)));
    const tr = [];
    if (s.resid_med) {
      tr.push({ x: lam, y: pct(s.resid_p84), mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" });
      tr.push({ x: lam, y: pct(s.resid_p16), mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: "rgba(111,179,255,0.2)", name: "16-84%", hoverinfo: "skip" });
      tr.push({ x: lam, y: pct(s.resid_med), mode: "lines+markers", name: "median data / own single template − 1", line: { color: "#6fb3ff", width: 2 }, marker: { size: 4 } });
    }
    const l1 = baseLayout("wavelength [um]", "residual from single template [%]", true, false);
    l1.shapes = [{ type: "line", x0: 0.7, x1: 5.2, y0: 0, y1: 0, line: { color: "#777", width: 0.8 } }];
    l1.legend = { x: 0, y: 1, font: { size: 10 } };
    Plotly.react($("selresid"), tr, l1, { displayModeBar: false, responsive: true });
    const tr2 = [];
    if (s.norm_med) {
      tr2.push({ x: lam, y: s.norm_med, mode: "lines", name: "median normalized spectrum", line: { color: "#e6e6e6", width: 2 } });
      tr2.push({ x: lam, y: s.template_med, mode: "lines", name: `template at t=${s.t_med.toFixed(2)}`, line: { color: "#ffd34e", width: 1.5, dash: "dash" } });
    }
    const l2 = baseLayout("wavelength [um]", "log10 F / scale", true, false);
    l2.legend = { x: 1, y: 0, xanchor: "right", font: { size: 10 } };
    Plotly.react($("selnorm"), tr2, l2, { displayModeBar: false, responsive: true });
    const tr3 = [];
    if (s.t_hist) {
      const e = s.t_hist.edges;
      tr3.push({ x: e.slice(0, -1).map((v, i) => 0.5 * (v + e[i + 1])), y: s.t_hist.counts, type: "bar", marker: { color: "#6fb3ff" }, name: "t_single" });
    }
    const l3 = baseLayout("t_single (0 = hot)", "count", false, false);
    l3.showlegend = false;
    l3.xaxis.range = [0, 1];
    Plotly.react($("selhist"), tr3, l3, { displayModeBar: false, responsive: true });
    const tbl = $("selstats");
    tbl.innerHTML = "<tr><td>column</td><td>median [16-84%]</td></tr>";
    for (const [k, v] of Object.entries(s.columns || {})) {
      const trr = document.createElement("tr");
      trr.innerHTML = `<td>${k}</td><td>${fmt(v.med, k)} [${fmt(v.p16, k)}, ${fmt(v.p84, k)}]</td>`;
      tbl.appendChild(trr);
    }
  }

  function clearSelection() {
    state.selection = null;
    scatter.setSelection(null);
    hr.setSelection(null);
    $("selinfo").textContent = "none";
    $("seltitle").textContent = "shift+drag to lasso a group";
  }

  function selectionCSV() {
    if (!state.selection) return;
    const rows = ["row,ticid" + (state.col ? "," + state.col : "")];
    for (const i of state.selection) {
      rows.push(`${i},${state.ticids ? state.ticids[i] : ""}` + (state.col ? "," + state.values[i] : ""));
    }
    const blob = new Blob([rows.join("\n")], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `sxml_selection_${state.ds}_${state.emb}_${state.selection.length}.csv`;
    a.click();
  }

  function zoomSelection() {
    if (!state.selection || !scatter.xy) return;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const i of state.selection) {
      const x = scatter.xy[2 * i], y = scatter.xy[2 * i + 1];
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    scatter.view.cx = 0.5 * (x0 + x1); scatter.view.cy = 0.5 * (y0 + y1);
    scatter.view.half = 0.6 * Math.max(x1 - x0, y1 - y0, 1e-3);
    scatter.redraw();
  }

  function showTab(which) {
    $("tab-star").classList.toggle("active", which === "star");
    $("tab-sel").classList.toggle("active", which === "sel");
    $("pane-star").classList.toggle("hidden", which !== "star");
    $("pane-sel").classList.toggle("hidden", which !== "sel");
    if (which === "star") { Plotly.Plots.resize($("spec")); Plotly.Plots.resize($("resid")); }
  }

  // ---- URL state ---------------------------------------------------------------------
  function updateHash() {
    const p = new URLSearchParams();
    if (state.ds) p.set("ds", state.ds);
    if (state.emb) p.set("emb", state.emb);
    if (state.col) p.set("col", state.col);
    if (state.vmin !== null) { p.set("vmin", fmt(state.vmin)); p.set("vmax", fmt(state.vmax)); }
    p.set("cmap", state.cmap + (state.flip ? "_r" : ""));
    if (state.logc) p.set("log", "1");
    if (state.filter) { p.set("fcol", state.filter.col); if (Number.isFinite(state.filter.min)) p.set("fmin", state.filter.min); if (Number.isFinite(state.filter.max)) p.set("fmax", state.filter.max); }
    if (state.star) { p.set("row", state.star.row); p.set("tic", state.star.ticid); }
    history.replaceState(null, "", "#" + p.toString());
    $("permalink").href = "#" + p.toString();
  }
  function readHash() {
    return new URLSearchParams(location.hash.slice(1));
  }

  // ---- datasets ------------------------------------------------------------------------
  function buildDatasetButtons() {
    // The Sample section appears only when there is a choice to make:
    // datasets still being built are not shown.
    const box = $("dsbtns");
    box.innerHTML = "";
    const ready = state.datasets.filter((d) => d.ready !== false);
    $("dssection").classList.toggle("hidden", ready.length < 2);
    for (const d of ready) {
      const b = document.createElement("button");
      b.className = "ds";
      b.dataset.name = d.name;
      b.textContent = d.label;
      b.title = `${d.n.toLocaleString()} stars · ${d.run_dir || ""}`;
      b.addEventListener("click", () => {
        if (d.name !== state.ds) loadDataset(d.name).catch((err) => showError(err.message));
      });
      box.appendChild(b);
    }
    markDatasetButtons();
  }
  function markDatasetButtons() {
    for (const b of $("dsbtns").children) b.classList.toggle("active", b.dataset.name === state.ds);
  }

  function fillSelect(sel, names, first) {
    sel.innerHTML = "";
    if (first) { const o = document.createElement("option"); o.value = ""; o.textContent = first; sel.appendChild(o); }
    for (const n of names) { const o = document.createElement("option"); o.value = n; o.textContent = n; sel.appendChild(o); }
  }

  /* Load (or switch to) a dataset.  `h` are the URL-hash parameters,
   * honoured on the first load only; on a switch the current embedding,
   * colour column, filter and star are kept where they exist. */
  async function loadDataset(name, h, meta) {
    const prevTic = state.star ? state.star.ticid : (h && h.get("tic") ? parseInt(h.get("tic"), 10) : null);
    state.ds = name;
    if (!meta) meta = await getJSON(url("meta"));
    state.meta = meta;
    state.colCache = {};
    state.values = null;
    state.flags = null;
    state.ticids = null;
    hoverVals = {};
    clearSelection();
    clearStar();
    markDatasetButtons();
    $("dsnote").textContent = `${meta.n.toLocaleString()} stars` + (meta.run_dir ? ` · ${meta.run_dir.split("/").slice(-1)[0]}` : "");
    const embSel = $("emb");
    embSel.innerHTML = "";
    for (const e of meta.embeddings) {
      const o = document.createElement("option");
      o.value = e.name; o.textContent = `${e.name} (${e.n.toLocaleString()})`;
      embSel.appendChild(o);
    }
    const names = meta.columns.map((c) => c.name).sort();
    fillSelect($("col"), names);
    fillSelect($("collist"), names);
    fillSelect($("fcol"), names, "(none)");
    if (!meta.embeddings.length) {
      setStatus("no embeddings yet: run drivers/run_umap.py");
      $("hud").textContent = "no umap_*.npy found in the ml directory";
      return;
    }
    state.ticids = new Uint32Array(await (await fetch(url("ticids"))).arrayBuffer());
    if (h) {
      const cm = h.get("cmap") || "viridis";
      state.flip = cm.endsWith("_r"); state.cmap = cm.replace(/_r$/, ""); $("cmap").value = state.cmap;
      state.logc = h.get("log") === "1"; $("logc").checked = state.logc;
    }
    const embNames = meta.embeddings.map((e) => e.name);
    const wantEmb = (h && h.get("emb")) || state.emb;
    await setEmbedding(embNames.includes(wantEmb) ? wantEmb
      : embNames.includes("shape") ? "shape" : embNames[0]);
    const wantCol = (h && h.get("col")) || state.col;
    const defCol = names.includes(wantCol) ? wantCol
      : names.includes("teff_emp") ? "teff_emp" : names.includes("teff_tars") ? "teff_tars" : names[0];
    await setColumn(defCol);
    if (h && h.get("vmin") !== null && h.get("vmax") !== null) {
      state.vmin = parseFloat(h.get("vmin")); state.vmax = parseFloat(h.get("vmax"));
      $("vmin").value = state.vmin; $("vmax").value = state.vmax; applyColors();
    }
    let fromUI = false;
    if (h && h.get("fcol")) {
      $("fcol").value = h.get("fcol"); $("fmin").value = h.get("fmin") || ""; $("fmax").value = h.get("fmax") || "";
      fromUI = true;
    } else if (state.filter) {
      if (names.includes(state.filter.col)) { $("fcol").value = state.filter.col; }
      else { state.filter = null; $("fcol").value = ""; $("fmin").value = ""; $("fmax").value = ""; }
    }
    await applyFilter(fromUI);
    // Gaia CMD panel.
    if (hasColumn("bprp0") && hasColumn("mg0")) {
      hr.setData(await getColumn("bprp0"), await getColumn("mg0"));
      hr.setFlags(state.flags);
      hr.setMainSequence(meta.mainseq || null);
      if (state.values) hr.setColor(state.values, state.vmin, state.vmax, lut(state.cmap, state.flip), state.logc);
    } else {
      hr.setData(new Float32Array(0), new Float32Array(0));
      $("hrcount").textContent = "no Gaia columns";
    }
    prepHoverCols();
    $("hud").textContent = `UMAP (${meta.umap.n_pc || "?"} PCs, n_neighbors=${meta.umap.n_neighbors || "?"}, min_dist=${meta.umap.min_dist || "?"})` + (scatter.mode === "canvas2d" ? " · 2D canvas renderer (no WebGL2)" : "");
    if (h && h.get("row") && !prevTic) loadStar(parseInt(h.get("row"), 10));
    else if (prevTic) findTic(prevTic, false);
    updateHash();
  }

  // ---- init ---------------------------------------------------------------------------------
  async function init() {
    const h = readHash();
    let meta;
    try {
      meta = await getJSON(API + "meta" + (h.get("ds") ? "?ds=" + encodeURIComponent(h.get("ds")) : ""));
    } catch (err) {
      meta = await getJSON(API + "meta");
    }
    state.datasets = meta.datasets || [{ name: meta.name || "", label: "all stars", n: meta.n }];
    buildDatasetButtons();
    bindControls();
    setSpecLegend(state.specLegend);
    setHRShown(state.hrShown);
    await loadDataset(meta.name || state.datasets[0].name, h, meta);
  }

  function bindControls() {
    $("emb").addEventListener("change", () => setEmbedding($("emb").value));
    $("col").addEventListener("change", () => setColumn($("col").value));
    $("colsearch").addEventListener("change", () => { if (hasColumn($("colsearch").value)) setColumn($("colsearch").value); });
    $("vmin").addEventListener("change", () => { state.vmin = parseFloat($("vmin").value); applyColors(); });
    $("vmax").addEventListener("change", () => { state.vmax = parseFloat($("vmax").value); applyColors(); });
    $("autorange").addEventListener("click", () => { autoRange(); applyColors(); });
    $("flipcmap").addEventListener("click", () => { state.flip = !state.flip; applyColors(); });
    $("cmap").addEventListener("change", () => { state.cmap = $("cmap").value; applyColors(); });
    $("logc").addEventListener("change", () => { state.logc = $("logc").checked; applyColors(); });
    $("size").addEventListener("input", () => { $("sizeval").textContent = $("size").value; scatter.setPointSize(parseFloat($("size").value)); });
    $("alpha").addEventListener("input", () => { $("alphaval").textContent = $("alpha").value; scatter.setAlpha(parseFloat($("alpha").value)); });
    $("fapply").addEventListener("click", () => applyFilter(true));
    $("fclear").addEventListener("click", () => { $("fcol").value = ""; $("fmin").value = ""; $("fmax").value = ""; applyFilter(true); });
    $("go").addEventListener("click", () => findTic(parseInt($("tic").value, 10), true));
    $("tic").addEventListener("keydown", (e) => { if (e.key === "Enter") findTic(parseInt($("tic").value, 10), true); });
    $("selclear").addEventListener("click", clearSelection);
    $("selcsv").addEventListener("click", selectionCSV);
    $("selzoom").addEventListener("click", zoomSelection);
    $("tab-star").addEventListener("click", () => showTab("star"));
    $("tab-sel").addEventListener("click", () => showTab("sel"));
    $("speclegend").addEventListener("click", () => setSpecLegend(!state.specLegend));
    $("hrhide").addEventListener("click", () => setHRShown(!state.hrShown));
    $("hrfit").addEventListener("click", () => hr.fitView());
    $("hrmode").addEventListener("change", () => hr.setMode($("hrmode").value));
    document.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      if (e.key === "f") scatter.fitView();
      if (e.key === "r") randomStar();
      if (e.key === "Escape") clearSelection();
    });
  }

  async function findTic(tic, center) {
    if (!tic) return;
    try {
      const s = await getJSON(url("star/" + tic, { by: "ticid" }));
      const xy = s.embedding[state.emb];
      if (center && xy && xy[0] !== null) scatter.centerOn(s.row, Math.min(scatter.view.half, 2));
      loadStar(s.row);
    } catch (err) { setStatus(`TIC ${tic} not in this sample`); }
  }

  function randomStar() {
    const n = state.meta.n;
    for (let k = 0; k < 1000; k++) {
      const i = Math.floor(Math.random() * n);
      if (scatter.flags[i]) { loadStar(i); return; }
    }
  }

  init().catch((err) => { setStatus("error: " + err.message); console.error(err); });
})();
