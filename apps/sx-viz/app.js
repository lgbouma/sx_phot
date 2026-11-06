/* sx-viz landing page: server-side filtered table of TARS stars. */
(function () {
  "use strict";

  const DEFAULT_COLUMNS = [
    "ticid", "Tmag", "teff", "BpmRp0", "parallax", "adopted_period",
    "median_amplitude", "ruwe", "n_used", "med_snr", "teff_full",
    "radius_full", "radius_ratio_full_rel", "red_excess_rel",
    "red_excess_rel_sig", "dbic_binary", "dbic_binary_pct", "dbic_disk",
    "dbic_disk_vs_binary", "teff_bin2", "teff_disk", "f_disk",
  ];
  const FILTERS = [
    ["Tmag", "Tmag"], ["teff", "Teff (TARS)"], ["BpmRp0", "(Bp-Rp)0"],
    ["parallax", "parallax mas"], ["adopted_period", "P rot d"],
    ["median_amplitude", "amp %"], ["ruwe", "RUWE"],
    ["n_used", "N spectra pts"], ["med_snr", "median S/N"],
    ["n_epochs", "N epochs"],
    ["teff_full", "Teff fit"], ["radius_ratio_full", "R / R_dwarf"],
    ["red_excess", "red excess"], ["red_excess_sig", "red excess sigma"],
    ["red_excess_rel", "red excess (Teff-rel)"],
    ["red_excess_rel_sig", "red excess rel sigma"],
    ["radius_ratio_full_rel", "R ratio (Teff-rel)"],
    ["final_n_contams", "TESS contams"],
    ["dchi2_binary", "binary dchi2"], ["teff_bin2", "Teff secondary"],
    ["dbic_binary", "ΔBIC two stars"], ["dbic_disk", "ΔBIC star+dust"],
    ["dbic_disk_vs_binary", "ΔBIC dust − binary"],
    ["dbic_binary_pct", "ΔBIC binary pct (Teff)"],
    ["dbic_disk_pct", "ΔBIC dust pct (Teff)"],
    ["teff_disk", "dust T (K)"], ["f_disk", "L_dust / L_star"],
    ["chi2_full", "chi2 single"], ["resid_mad", "spline |res|/err"],
    ["c_3p6_1p6", "log F3.6/F1.6"],
  ];
  const FLAGS = [
    ["flag_possible_binary", "TARS possible binary"],
    ["flag_multiple_periods", "multiple periods"],
    ["flag_doubled_period", "doubled period"],
    ["spline_ok", "spline ok"],
  ];
  const PRESETS = {
    "Red excess 5-100%, >5σ (Teff-rel)": { min_red_excess_rel: 0.05, max_red_excess_rel: 1.0, min_red_excess_rel_sig: 5, min_n_used: 100, sort: "red_excess_rel_sig", desc: 1 },
    "Red excess 5-30%, >5σ, no TESS contams": { min_red_excess_rel: 0.05, max_red_excess_rel: 0.3, min_red_excess_rel_sig: 5, max_final_n_contams: 0, min_n_used: 100, sort: "red_excess_rel_sig", desc: 1 },
    "Extreme red excess (> 100%, blends?)": { min_red_excess_rel: 1.0, min_n_used: 100, sort: "red_excess_rel", desc: 1 },
    "Overluminous (R/Rdwarf rel 1.3-3)": { min_radius_ratio_full_rel: 1.3, max_radius_ratio_full_rel: 3, min_parallax: 2, min_n_used: 100, sort: "radius_ratio_full_rel", desc: 1 },
    "Overluminous + red excess": { min_radius_ratio_full_rel: 1.25, max_radius_ratio_full_rel: 3, min_red_excess_rel: 0.05, max_red_excess_rel: 1.0, min_red_excess_rel_sig: 3, min_n_used: 100, sort: "red_excess_rel_sig", desc: 1 },
    "Two stars: top 2% ΔBIC at its Teff": { min_dbic_binary_pct: 0.98, max_dbic_disk_vs_binary: 0, min_n_used: 100, sort: "dbic_binary", desc: 1 },
    "Two stars preferred (ΔBIC > 30, beats dust)": { min_dbic_binary: 30, max_dbic_disk_vs_binary: 0, min_n_used: 100, sort: "dbic_binary", desc: 1 },
    "Dust preferred (ΔBIC > 30, beats two stars by 10)": { min_dbic_disk: 30, min_dbic_disk_vs_binary: 10, min_n_used: 100, sort: "dbic_disk_vs_binary", desc: 1 },
    "M dwarfs, fast (P < 1 d)": { max_teff: 4000, max_adopted_period: 1, sort: "Tmag", desc: 0 },
    "Bright, many points": { max_Tmag: 10, min_n_used: 300, sort: "n_used", desc: 1 },
    "Binary dchi2 > 200": { min_dchi2_binary: 200, min_n_used: 100, sort: "dchi2_binary", desc: 1 },
  };
  const FORMATS = {
    ticid: (v) => String(v),
    ra: (v) => v.toFixed(5), dec: (v) => v.toFixed(5),
    adopted_period: (v) => v.toFixed(4), parallax: (v) => v.toFixed(3),
    teff: (v) => v.toFixed(0), teff_full: (v) => v.toFixed(0),
    teff_blue: (v) => v.toFixed(0), teff_bin1: (v) => v.toFixed(0),
    teff_bin2: (v) => v.toFixed(0),
    red_excess: (v) => (100 * v).toFixed(1) + "%",
    red_excess_rel: (v) => (100 * v).toFixed(1) + "%",
    excess_b3_rel: (v) => (100 * v).toFixed(1) + "%",
    excess_b4_rel: (v) => (100 * v).toFixed(1) + "%",
    excess_b5_rel: (v) => (100 * v).toFixed(1) + "%",
    excess_b6_rel: (v) => (100 * v).toFixed(1) + "%",
    excess_b3: (v) => (100 * v).toFixed(1) + "%",
    excess_b4: (v) => (100 * v).toFixed(1) + "%",
    excess_b5: (v) => (100 * v).toFixed(1) + "%",
    excess_b6: (v) => (100 * v).toFixed(1) + "%",
    dchi2_binary: (v) => v.toFixed(0), chi2_full: (v) => v.toFixed(0),
    dchi2_disk: (v) => v.toFixed(0), chi2_disk: (v) => v.toFixed(0),
    bic_single: (v) => v.toFixed(0), bic_binary: (v) => v.toFixed(0),
    bic_disk: (v) => v.toFixed(0), dbic_binary: (v) => v.toFixed(0),
    dbic_disk: (v) => v.toFixed(0), dbic_disk_vs_binary: (v) => v.toFixed(0),
    teff_disk: (v) => v.toFixed(0), teff_disk_star: (v) => v.toFixed(0),
    f_disk: (v) => v.toExponential(1),
    dbic_binary_pct: (v) => v.toFixed(3), dbic_disk_pct: (v) => v.toFixed(3),
    dbic_disk_vs_binary_pct: (v) => v.toFixed(3),
    chi2_blue: (v) => v.toFixed(0), chi2_binary: (v) => v.toFixed(0),
    med_snr: (v) => v.toFixed(0), mjd_min: (v) => v.toFixed(1),
    mjd_max: (v) => v.toFixed(1),
  };

  const state = {
    meta: null, columns: null, offset: 0, limit: 100, sort: "ticid",
    desc: false, filters: {}, q: "", verdicts: {}, lastTotal: 0,
  };

  const $ = (id) => document.getElementById(id);

  function fmt(col, v) {
    if (v === null || v === undefined) return "";
    if (typeof v === "boolean") return v ? "yes" : "";
    if (typeof v === "number") {
      if (FORMATS[col]) return FORMATS[col](v);
      if (Number.isInteger(v)) return String(v);
      const a = Math.abs(v);
      if (a >= 1000) return v.toFixed(0);
      if (a >= 10) return v.toFixed(2);
      if (a >= 0.01) return v.toFixed(3);
      return v.toExponential(2);
    }
    return String(v);
  }

  function loadColumns() {
    try {
      const saved = JSON.parse(localStorage.getItem("sxviz.columns") || "null");
      if (Array.isArray(saved) && saved.length) return saved;
    } catch (e) { /* ignore */ }
    return DEFAULT_COLUMNS.slice();
  }
  function saveColumns() {
    localStorage.setItem("sxviz.columns", JSON.stringify(state.columns));
  }

  function queryString(extra) {
    const p = new URLSearchParams();
    Object.entries(state.filters).forEach(([k, v]) => {
      if (v !== "" && v !== null && v !== undefined) p.set(k, v);
    });
    if (state.q) p.set("q", state.q);
    p.set("sort", state.sort);
    p.set("desc", state.desc ? "1" : "0");
    Object.entries(extra || {}).forEach(([k, v]) => p.set(k, v));
    return p.toString();
  }

  function stateToUrl() {
    const qs = queryString({ offset: state.offset, limit: state.limit });
    history.replaceState(null, "", "?" + qs);
  }
  function stateFromUrl() {
    const p = new URLSearchParams(location.search);
    p.forEach((v, k) => {
      if (k === "sort") state.sort = v;
      else if (k === "desc") state.desc = v === "1";
      else if (k === "offset") state.offset = parseInt(v, 10) || 0;
      else if (k === "limit") state.limit = parseInt(v, 10) || 100;
      else if (k === "q") state.q = v;
      else state.filters[k] = v;
    });
  }

  function buildFilters() {
    const host = $("filters");
    host.innerHTML = "";
    FILTERS.forEach(([col, label]) => {
      if (state.meta && !state.meta.numeric.includes(col)) return;
      const row = document.createElement("div");
      row.className = "filter-row";
      row.innerHTML = `<label title="${col}">${label}</label>
        <input type="number" step="any" placeholder="min" data-key="min_${col}" />
        <input type="number" step="any" placeholder="max" data-key="max_${col}" />`;
      host.appendChild(row);
    });
    host.querySelectorAll("input").forEach((inp) => {
      inp.value = state.filters[inp.dataset.key] ?? "";
      inp.addEventListener("change", () => {
        state.filters[inp.dataset.key] = inp.value;
      });
      inp.addEventListener("keydown", (e) => { if (e.key === "Enter") apply(); });
    });
    const chips = $("flag-chips");
    chips.innerHTML = "";
    FLAGS.forEach(([col, label]) => {
      if (state.meta && !state.meta.boolean.includes(col)) return;
      const key = "flag_" + col;
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = label;
      const sync = () => {
        const v = state.filters[key];
        b.className = v === "1" ? "active" : (v === "0" ? "primary" : "");
        b.title = v === "1" ? "only flagged" : (v === "0" ? "only unflagged" : "any");
      };
      b.addEventListener("click", () => {
        const v = state.filters[key];
        state.filters[key] = v === "1" ? "0" : (v === "0" ? "" : "1");
        sync();
        apply();
      });
      sync();
      chips.appendChild(b);
    });
    const presets = $("presets");
    presets.innerHTML = "";
    Object.entries(PRESETS).forEach(([name, cfg]) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = name;
      b.addEventListener("click", () => {
        state.filters = {};
        state.q = "";
        Object.entries(cfg).forEach(([k, v]) => {
          if (k === "sort") state.sort = v;
          else if (k === "desc") state.desc = !!v;
          else state.filters[k] = String(v);
        });
        buildFilters();
        $("search-input").value = "";
        apply();
      });
      presets.appendChild(b);
    });
  }

  function buildColumnToggles() {
    const host = $("column-toggles");
    host.innerHTML = "";
    state.meta.columns.forEach((col) => {
      const lab = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = state.columns.includes(col);
      cb.addEventListener("change", () => {
        if (cb.checked && !state.columns.includes(col)) state.columns.push(col);
        if (!cb.checked) state.columns = state.columns.filter((c) => c !== col);
        state.columns = state.meta.columns.filter((c) => state.columns.includes(c));
        if (!state.columns.includes("ticid")) state.columns.unshift("ticid");
        saveColumns();
        fetchTable();
      });
      lab.appendChild(cb);
      lab.appendChild(document.createTextNode(" " + col));
      host.appendChild(lab);
    });
  }

  async function fetchJson(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
    return r.json();
  }

  async function fetchTable() {
    $("status").textContent = "loading…";
    const t0 = performance.now();
    const qs = queryString({
      offset: state.offset, limit: state.limit, cols: state.columns.join(","),
    });
    let data;
    try {
      data = await fetchJson("/api/table?" + qs);
    } catch (e) {
      $("status").textContent = String(e);
      return;
    }
    state.lastTotal = data.total;
    renderTable(data);
    stateToUrl();
    $("status").textContent = `${(performance.now() - t0).toFixed(0)} ms`;
  }

  function renderTable(data) {
    const head = $("table-headers");
    head.innerHTML = "";
    data.columns.forEach((col) => {
      const th = document.createElement("th");
      th.textContent = col + (col === data.sort ? (data.desc ? " ▼" : " ▲") : "");
      if (col === data.sort) th.classList.add("sorted");
      th.addEventListener("click", () => {
        if (state.sort === col) state.desc = !state.desc;
        else { state.sort = col; state.desc = true; }
        state.offset = 0;
        fetchTable();
      });
      head.appendChild(th);
    });
    const vth = document.createElement("th");
    vth.textContent = "verdict";
    head.appendChild(vth);

    const body = $("table-body");
    body.innerHTML = "";
    const frag = document.createDocumentFragment();
    data.rows.forEach((row) => {
      const tr = document.createElement("tr");
      data.columns.forEach((col) => {
        const td = document.createElement("td");
        td.textContent = fmt(col, row[col]);
        tr.appendChild(td);
      });
      const v = state.verdicts[String(row.ticid)];
      const td = document.createElement("td");
      td.textContent = v ? v.verdict : "";
      if (v) td.className = "verdict-" + v.verdict;
      tr.appendChild(td);
      tr.addEventListener("click", (e) => openStar(row.ticid, e.metaKey || e.ctrlKey));
      frag.appendChild(tr);
    });
    body.appendChild(frag);
    $("row-count").textContent = `${data.total.toLocaleString()} rows`;
    $("sort-pill").textContent = `sort: ${data.sort} ${data.desc ? "desc" : "asc"}`;
    const page = Math.floor(data.offset / data.limit) + 1;
    const pages = Math.max(1, Math.ceil(data.total / data.limit));
    $("page-text").textContent = `page ${page} / ${pages}`;
    $("prev-btn").disabled = data.offset <= 0;
    $("next-btn").disabled = data.offset + data.limit >= data.total;
    $("csv-link").href = "data:text/csv," + encodeURIComponent(toCsv(data));
    $("csv-link").download = "sxviz_page.csv";
  }

  function toCsv(data) {
    const lines = [data.columns.join(",")];
    data.rows.forEach((r) => lines.push(data.columns.map((c) => r[c] ?? "").join(",")));
    return lines.join("\n");
  }

  async function navIds() {
    // Up to 2000 ids of the current filtered + sorted list, for j/k nav.
    const qs = queryString({ offset: 0, limit: 2000, ids_only: 1 });
    try {
      const d = await fetchJson("/api/table?" + qs);
      sessionStorage.setItem("sxviz.nav", JSON.stringify({ ids: d.ids, qs: queryString() }));
    } catch (e) { /* ignore */ }
  }

  async function openStar(ticid, newTab) {
    await navIds();
    const url = `star.html?ticid=${ticid}`;
    if (newTab) window.open(url, "_blank");
    else location.href = url;
  }

  async function openRandom() {
    if (!state.lastTotal) return;
    const k = Math.floor(Math.random() * state.lastTotal);
    const qs = queryString({ offset: k, limit: 1, ids_only: 1 });
    const d = await fetchJson("/api/table?" + qs);
    if (d.ids && d.ids.length) openStar(d.ids[0], false);
  }

  function apply() {
    state.q = $("search-input").value.trim();
    state.offset = 0;
    fetchTable();
  }

  async function init() {
    stateFromUrl();
    state.columns = loadColumns();
    state.meta = await fetchJson("/api/meta");
    state.columns = state.meta.columns.filter((c) => state.columns.includes(c));
    if (!state.columns.includes("ticid")) state.columns.unshift("ticid");
    const v = state.meta.viz || {};
    $("meta-text").textContent =
      ` · ${state.meta.n_stars.toLocaleString()} stars` +
      (v.created ? ` · products built ${v.created.slice(0, 10)}` : "") +
      (state.meta.has_sed ? "" : " · (no SED fits yet)");
    try { state.verdicts = await fetchJson("/api/verdicts"); } catch (e) { /* ignore */ }
    buildFilters();
    buildColumnToggles();
    $("search-input").value = state.q;
    $("page-size").value = String(state.limit);
    $("apply-btn").addEventListener("click", apply);
    $("reset-btn").addEventListener("click", () => {
      state.filters = {}; state.q = ""; state.sort = "ticid"; state.desc = false;
      $("search-input").value = "";
      buildFilters();
      apply();
    });
    $("search-input").addEventListener("keydown", (e) => { if (e.key === "Enter") apply(); });
    $("prev-btn").addEventListener("click", () => {
      state.offset = Math.max(0, state.offset - state.limit); fetchTable();
    });
    $("next-btn").addEventListener("click", () => {
      state.offset += state.limit; fetchTable();
    });
    $("page-size").addEventListener("change", () => {
      state.limit = parseInt($("page-size").value, 10); state.offset = 0; fetchTable();
    });
    $("random-btn").addEventListener("click", openRandom);
    document.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      if (e.key === "r") openRandom();
      if (e.key === "n" || e.key === "ArrowRight") $("next-btn").click();
      if (e.key === "p" || e.key === "ArrowLeft") $("prev-btn").click();
    });
    fetchTable();
  }

  window.addEventListener("error", (e) => {
    const el = $("status");
    if (el) el.textContent = "JS error: " + e.message;
  });
  window.addEventListener("unhandledrejection", (e) => {
    const el = $("status");
    if (el) el.textContent = "error: " + (e.reason && e.reason.message ? e.reason.message : e.reason);
  });

  init();
})();
