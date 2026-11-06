/* sx-ml spectral sequence: slide along the empirical template library. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const API = "api/";
  const state = { lib: null, meta: null, t: 0.5, a1: 0, at: null, seed: 1, playing: null, pending: null };

  async function getJSON(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    return r.json();
  }
  const fmt = (v, d) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(d);

  // ---- interpolation along the nodes (mirrors specml.interpolate_nodes) ----
  function interpRow(nodes, rows, t) {
    const n = nodes.length;
    if (t <= nodes[0]) return rows[0].slice();
    if (t >= nodes[n - 1]) return rows[n - 1].slice();
    let k = 0;
    while (k < n - 2 && nodes[k + 1] < t) k++;
    const f = (t - nodes[k]) / (nodes[k + 1] - nodes[k]);
    const a = rows[k], b = rows[k + 1];
    return a.map((v, i) => (v === null || b[i] === null) ? null : (1 - f) * v + f * b[i]);
  }
  function interp1(nodes, vals, t) {
    return interpRow(nodes, vals.map((v) => [v]), t)[0];
  }
  function tFromArc(s) { const L = state.lib; return interpXY(L.arc, L.nodes, s); }
  function arcFromT(t) { const L = state.lib; return interpXY(L.nodes, L.arc, t); }
  function tFromTeff(te) { const L = state.lib; const x = L.teff_mono.map((v) => -v); return interpXY(x, L.nodes, -te); }
  function teffFromT(t) { const L = state.lib; return interpXY(L.nodes, L.teff_mono, t); }
  function interpXY(xs, ys, x) {
    // xs increasing (may have ties); linear interpolation with clamping.
    const n = xs.length;
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
    const d = xs[hi] - xs[lo];
    return d > 0 ? ys[lo] + (ys[hi] - ys[lo]) * (x - xs[lo]) / d : ys[lo];
  }

  // ---- slider <-> t ---------------------------------------------------------------
  function sliderToT(v) {
    const mode = $("mode").value;
    if (mode === "arc") return tFromArc(v);
    if (mode === "teff") { const L = state.lib; const lo = L.teff_mono[L.teff_mono.length - 1], hi = L.teff_mono[0]; return tFromTeff(hi - v * (hi - lo)); }
    return v;
  }
  function tToSlider(t) {
    const mode = $("mode").value;
    if (mode === "arc") return arcFromT(t);
    if (mode === "teff") { const L = state.lib; const lo = L.teff_mono[L.teff_mono.length - 1], hi = L.teff_mono[0]; return (hi - teffFromT(t)) / (hi - lo); }
    return t;
  }

  // ---- rendering ------------------------------------------------------------------------
  function currentTemplate(t, a1) {
    const L = state.lib;
    let y = interpRow(L.nodes, L.templates, t);
    if (a1) {
      const e = interpRow(L.nodes, L.resid_comps.map((c) => c[0]), t);
      const amp = interp1(L.nodes, L.resid_amp.map((a) => a[0]), t);
      y = y.map((v, i) => v + a1 * amp * e[i]);
    }
    return y;
  }

  function render() {
    const L = state.lib, t = state.t;
    const abs = $("showabs").checked;
    const lin = $("linear").checked;
    const off = abs ? interp1(L.nodes, L.abs_offset, t) : 0;
    const y = currentTemplate(t, state.a1).map((v) => v + off);
    const sc = interpRow(L.nodes, L.scatter_int, t);
    const toY = (arr) => lin ? arr.map((v) => (v === null ? null : Math.pow(10, v))) : arr;
    const traces = [];
    if ($("showband").checked) {
      traces.push({ x: L.lam, y: toY(y.map((v, i) => v + sc[i])), mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" });
      traces.push({ x: L.lam, y: toY(y.map((v, i) => v - sc[i])), mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: "rgba(255,211,78,0.18)", name: "±1σ population", hoverinfo: "skip" });
    }
    if ($("showghost").checked) {
      for (const dt of [-0.04, -0.02, 0.02, 0.04]) {
        const tt = Math.min(1, Math.max(0, t + dt));
        const yy = currentTemplate(tt, 0).map((v) => v + (abs ? interp1(L.nodes, L.abs_offset, tt) : 0));
        traces.push({ x: L.lam, y: toY(yy), mode: "lines", line: { color: "rgba(230,230,230,0.25)", width: 1 }, name: `t=${tt.toFixed(2)}`, showlegend: false, hoverinfo: "skip" });
      }
    }
    if ($("showex").checked && state.at && state.at.examples) {
      state.at.examples.forEach((ex, k) => {
        const yy = ex.y.map((v) => (v === null ? null : v + off));
        traces.push({ x: L.lam, y: toY(yy), mode: "lines", line: { color: "rgba(111,179,255,0.45)", width: 1 }, name: `TIC ${ex.ticid}`, showlegend: k === 0, legendgroup: "ex", hovertemplate: `TIC ${ex.ticid}<extra></extra>` });
      });
    }
    if ($("showmodel").checked && state.at && state.at.model) {
      const m = state.at.model;
      traces.push({ x: L.lam, y: toY(m.log_norm.map((v) => v + off)), mode: "lines", line: { color: "#ff8c69", dash: "dash", width: 1.5 }, name: `${m.grid} ${m.teff.toFixed(0)} K` });
    }
    traces.push({ x: L.lam, y: toY(y), mode: "lines+markers", line: { color: "#ffd34e", width: 2.5 }, marker: { size: 4 }, name: `template t=${t.toFixed(3)}` + (state.a1 ? ` + ${state.a1}×e1` : "") });
    const layout = base("wavelength [um]", abs ? (lin ? "F_nu at 10 pc [Jy]" : "log10 F_nu at 10 pc [Jy]") : (lin ? "F_nu / geometric mean" : "normalized log10 F_nu"), true);
    layout.legend = { x: 1, y: 1, xanchor: "right", font: { size: 10 } };
    layout.yaxis.type = lin ? "log" : "linear";
    if (!abs && !lin) layout.yaxis.range = [-0.75, 0.55];
    layout.uirevision = "main";
    Plotly.react($("main"), traces, layout, { displayModeBar: false, responsive: true });
    renderRatio();
    renderImageMarker();
    renderMapping();
    renderScatterPlot();
    renderInfo();
  }

  function renderRatio() {
    const L = state.lib, t = state.t;
    const traces = [];
    const y = currentTemplate(t, 0);
    if (state.at && state.at.model) {
      const m = state.at.model.log_norm;
      traces.push({ x: L.lam, y: y.map((v, i) => 100 * (Math.pow(10, v - m[i]) - 1)), mode: "lines+markers", name: `template / ${state.at.model.grid} − 1`, line: { color: "#ff8c69", width: 1.5 }, marker: { size: 3 } });
    }
    const e = interpRow(L.nodes, L.resid_comps.map((c) => c[0]), t);
    const amp = interp1(L.nodes, L.resid_amp.map((a) => a[0]), t);
    traces.push({ x: L.lam, y: e.map((v) => 100 * (Math.pow(10, amp * v) - 1)), mode: "lines", name: "1σ of residual eigenvector 1", line: { color: "#6fb3ff", width: 1.5, dash: "dot" } });
    const e2 = interpRow(L.nodes, L.resid_comps.map((c) => c[1]), t);
    const amp2 = interp1(L.nodes, L.resid_amp.map((a) => a[1]), t);
    traces.push({ x: L.lam, y: e2.map((v) => 100 * (Math.pow(10, amp2 * v) - 1)), mode: "lines", name: "eigenvector 2", line: { color: "#9b8cff", width: 1, dash: "dot" } });
    const sc = interpRow(L.nodes, L.scatter_int, t);
    traces.push({ x: L.lam, y: sc.map((v) => 100 * (Math.pow(10, v) - 1)), mode: "lines", name: "population scatter", line: { color: "#aaa", width: 1 } });
    const layout = base("wavelength [um]", "percent", true);
    layout.legend = { x: 0, y: 1, font: { size: 9 } };
    layout.shapes = [{ type: "line", x0: 0.7, x1: 5.2, y0: 0, y1: 0, line: { color: "#777", width: 0.8 } }];
    layout.yaxis.range = [-20, 20];
    layout.uirevision = "ratio";
    Plotly.react($("ratio"), traces, layout, { displayModeBar: false, responsive: true });
  }

  function imageAxis() {
    const L = state.lib, mode = $("mode").value;
    if (mode === "teff") return { y: L.teff_mono, label: "Teff [K]", cur: teffFromT(state.t), rev: true };
    if (mode === "arc") return { y: L.arc, label: "arc length", cur: arcFromT(state.t), rev: false };
    return { y: L.nodes, label: "t", cur: state.t, rev: false };
  }

  let imageDrawn = false;
  function renderImage() {
    const L = state.lib;
    const ax = imageAxis();
    const z = L.templates;
    const trace = { x: L.lam, y: ax.y, z, type: "heatmap", colorscale: "RdBu", reversescale: true, zmin: -0.6, zmax: 0.6, colorbar: { title: "log F", thickness: 10, len: 0.9, tickfont: { size: 9 } } };
    const layout = base("wavelength [um]", ax.label, true);
    layout.margin.r = 50;
    if (ax.rev) layout.yaxis.autorange = "reversed";
    layout.uirevision = "image" + $("mode").value;
    Plotly.react($("image"), [trace], layout, { displayModeBar: false, responsive: true }).then(() => {
      if (!imageDrawn) {
        imageDrawn = true;
        $("image").on("plotly_click", (ev) => {
          const yv = ev.points[0].y;
          const mode = $("mode").value;
          let t = mode === "teff" ? tFromTeff(yv) : mode === "arc" ? tFromArc(yv) : yv;
          setT(t);
        });
      }
    });
    renderImageMarker();
  }
  function renderImageMarker() {
    const ax = imageAxis();
    Plotly.relayout($("image"), { shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: ax.cur, y1: ax.cur, line: { color: "#ffd34e", width: 2 } }] });
  }

  function renderMapping() {
    const L = state.lib, s = L.summary;
    const ax = imageAxis();
    const x = ax.y;
    const traces = [
      { x, y: s.teff_tars_med, mode: "lines", name: "TARS Teff median", line: { color: "#e6e6e6", width: 1.5 } },
      { x, y: s.teff_tars_p84, mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
      { x, y: s.teff_tars_p16, mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: "rgba(230,230,230,0.15)", name: "16-84%", hoverinfo: "skip" },
      { x, y: L.teff_mono, mode: "lines", name: "monotone mapping", line: { color: "#ffd34e", width: 1, dash: "dash" } },
      { x, y: s.mg0_med, mode: "lines", name: "M_G0 median", yaxis: "y2", line: { color: "#6fb3ff", width: 1.5 } },
    ];
    const layout = base(ax.label, "Teff [K]", false);
    layout.yaxis2 = { title: "M_G0", overlaying: "y", side: "right", autorange: "reversed", gridcolor: "#2c2d36", titlefont: { color: "#6fb3ff" }, tickfont: { color: "#6fb3ff" } };
    layout.margin.r = 50;
    layout.legend = { x: 0.5, y: 1, font: { size: 9 }, xanchor: "center", orientation: "h" };
    if (ax.rev) layout.xaxis.autorange = "reversed";
    layout.shapes = [{ type: "line", yref: "paper", y0: 0, y1: 1, x0: ax.cur, x1: ax.cur, line: { color: "#ffd34e", width: 1.5 } }];
    layout.uirevision = "map" + $("mode").value;
    Plotly.react($("mapping"), traces, layout, { displayModeBar: false, responsive: true });
  }

  function renderScatterPlot() {
    const L = state.lib;
    const ax = imageAxis();
    const picks = [0.9, 1.3, 1.9, 2.7, 3.5, 4.7];
    const traces = picks.map((lam) => {
      let i = 0; for (let k = 0; k < L.lam.length; k++) if (Math.abs(L.lam[k] - lam) < Math.abs(L.lam[i] - lam)) i = k;
      return { x: ax.y, y: L.scatter_int.map((r) => 100 * Math.LN10 * r[i]), mode: "lines", name: `${L.lam[i].toFixed(2)} um`, line: { width: 1.2 } };
    });
    traces.push({ x: ax.y, y: L.summary.counts, mode: "lines", name: "stars per window", yaxis: "y2", line: { color: "#888", width: 1, dash: "dot" } });
    const layout = base(ax.label, "intrinsic scatter [%]", false);
    layout.yaxis.type = "log";
    layout.yaxis2 = { title: "N", overlaying: "y", side: "right", type: "log", gridcolor: "#2c2d36", tickfont: { color: "#888" } };
    layout.margin.r = 50;
    layout.legend = { x: 0.5, y: 1, font: { size: 9 }, xanchor: "center", orientation: "h" };
    if (ax.rev) layout.xaxis.autorange = "reversed";
    layout.shapes = [{ type: "line", yref: "paper", y0: 0, y1: 1, x0: ax.cur, x1: ax.cur, line: { color: "#ffd34e", width: 1.5 } }];
    layout.uirevision = "sc" + $("mode").value;
    Plotly.react($("scatterplot"), traces, layout, { displayModeBar: false, responsive: true });
  }

  function renderInfo() {
    const L = state.lib, t = state.t, at = state.at || {};
    const teff = teffFromT(t);
    $("teffval").textContent = teff.toFixed(0);
    $("tval").textContent = t.toFixed(3);
    $("arcval").textContent = arcFromT(t).toFixed(3);
    const rows = [
      ["Teff (monotone TARS map)", `${teff.toFixed(0)} K`],
      ["TARS Teff median [16-84%]", at.teff_tars_med ? `${at.teff_tars_med.toFixed(0)} [${at.teff_tars_p16.toFixed(0)}, ${at.teff_tars_p84.toFixed(0)}]` : "—"],
      ["BP-RP0 median", fmt(at.bprp0_med, 3)],
      ["M_G0 median [16-84%]", at.mg0_med ? `${at.mg0_med.toFixed(2)} [${at.mg0_p16.toFixed(2)}, ${at.mg0_p84.toFixed(2)}]` : "—"],
      ["dwarf mass (Mamajek)", at.mass_ms ? `${at.mass_ms.toFixed(2)} Msun` : fmt(interp1(L.nodes, L.mass_ms, t), 2) + " Msun"],
      ["dwarf radius (Mamajek)", at.radius_ms ? `${at.radius_ms.toFixed(2)} Rsun` : fmt(interp1(L.nodes, L.radius_ms, t), 2) + " Rsun"],
      ["A0 median", fmt(at.a0_med, 3)],
      ["Tmag median", fmt(at.tmag_med, 2)],
      ["rotation period median", at.period_med ? `${at.period_med.toFixed(2)} d` : "—"],
      ["stars in node window", fmt(interp1(L.nodes, L.summary.counts, t), 0)],
      ["model atmosphere", at.model ? `${at.model.grid} ${at.model.teff.toFixed(0)} K` : "—"],
      ["resid. eigenvector 1 rms", `${(100 * Math.LN10 * interp1(L.nodes, L.resid_amp.map((a) => a[0]), t)).toFixed(2)} %`],
      ["resid. variance fraction 1", fmt(interp1(L.nodes, L.summary.resid_var_frac_1 || L.nodes.map(() => null), t), 2)],
    ];
    $("nodeinfo").innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
    const ex = (state.at && state.at.examples) || [];
    $("exlist").innerHTML = ex.map((e) =>
      `<div>TIC ${e.ticid} · ${e.teff_tars ? e.teff_tars.toFixed(0) + " K" : ""} · T=${e.tmag ? e.tmag.toFixed(1) : ""} · ` +
      `<a href="${state.meta.sxviz_url}/star.html?ticid=${e.ticid}" target="_blank">sx-viz</a> · <a href="umap.html#row=${e.row}" target="_blank">umap</a></div>`).join("") || "<span class='muted'>none</span>";
  }

  function base(xt, yt, logx) {
    return {
      margin: { l: 55, r: 10, t: 10, b: 40 }, paper_bgcolor: "#1b1c22", plot_bgcolor: "#121216",
      font: { color: "#e6e6e6", size: 11 },
      xaxis: { title: xt, type: logx ? "log" : "linear", gridcolor: "#2c2d36", zeroline: false, range: logx ? [Math.log10(0.72), Math.log10(5.2)] : undefined },
      yaxis: { title: yt, gridcolor: "#2c2d36", zeroline: false },
      showlegend: true,
    };
  }

  // ---- server round trip (debounced) ------------------------------------------------
  function fetchAt() {
    if (state.pending) clearTimeout(state.pending);
    state.pending = setTimeout(async () => {
      const n = $("showex").checked ? parseInt($("nex").value, 10) || 0 : 0;
      try {
        const at = await getJSON(`${API}sequence/at?t=${state.t.toFixed(4)}&n=${n}&seed=${state.seed}`);
        if (Math.abs(at.t - state.t) < 1e-3) { state.at = at; render(); }
      } catch (err) { $("status").textContent = "error: " + err.message; }
    }, 120);
  }

  function setT(t, fromSlider) {
    state.t = Math.min(1, Math.max(0, t));
    if (!fromSlider) $("pos").value = tToSlider(state.t);
    render();
    fetchAt();
    history.replaceState(null, "", `#t=${state.t.toFixed(4)}&mode=${$("mode").value}&a1=${state.a1}`);
  }

  function updateSliderLabels() {
    const L = state.lib, mode = $("mode").value;
    const lo = L.teff_mono[0], hi = L.teff_mono[L.teff_mono.length - 1];
    $("poslo").textContent = mode === "teff" ? `${lo.toFixed(0)} K` : `hot (${lo.toFixed(0)} K)`;
    $("poshi").textContent = mode === "teff" ? `${hi.toFixed(0)} K` : `cool (${hi.toFixed(0)} K)`;
  }

  async function init() {
    const [meta, lib] = await Promise.all([getJSON(API + "meta"), getJSON(API + "sequence")]);
    state.meta = meta; state.lib = lib;
    $("status").textContent = `${lib.nodes.length} nodes · ${lib.meta.n_inlier ? lib.meta.n_inlier.toLocaleString() : "?"} training stars · built ${lib.meta.created || ""}`;
    $("a1note").textContent = "adds the first residual eigenvector at this t (sign: positive at 5 um)";
    const h = new URLSearchParams(location.hash.slice(1));
    if (h.get("mode")) $("mode").value = h.get("mode");
    if (h.get("a1")) { state.a1 = parseFloat(h.get("a1")); $("a1").value = state.a1; $("a1val").textContent = state.a1; }
    updateSliderLabels();
    renderImage();
    setT(h.get("t") ? parseFloat(h.get("t")) : 0.5);
    $("pos").addEventListener("input", () => setT(sliderToT(parseFloat($("pos").value)), true));
    $("mode").addEventListener("change", () => { updateSliderLabels(); $("pos").value = tToSlider(state.t); renderImage(); render(); });
    $("a1").addEventListener("input", () => { state.a1 = parseFloat($("a1").value); $("a1val").textContent = state.a1.toFixed(1); render(); });
    for (const id of ["showabs", "showband", "showmodel", "showghost", "linear"]) $(id).addEventListener("change", render);
    $("showex").addEventListener("change", fetchAt);
    $("nex").addEventListener("change", fetchAt);
    $("reshuffle").addEventListener("click", () => { state.seed = Math.floor(Math.random() * 1e6); fetchAt(); });
    document.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      const step = 0.005;
      if (e.key === "ArrowLeft") setT(sliderToT(Math.max(0, parseFloat($("pos").value) - step)));
      if (e.key === "ArrowRight") setT(sliderToT(Math.min(1, parseFloat($("pos").value) + step)));
      if (e.key === "p") togglePlay();
    });
  }

  function togglePlay() {
    if (state.playing) { clearInterval(state.playing); state.playing = null; return; }
    state.playing = setInterval(() => {
      let v = parseFloat($("pos").value) + 0.004;
      if (v > 1) v = 0;
      $("pos").value = v;
      setT(sliderToT(v), true);
    }, 150);
  }

  init().catch((err) => { $("status").textContent = "error: " + err.message; console.error(err); });
})();
