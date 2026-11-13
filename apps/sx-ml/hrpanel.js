/* hrpanel.js: Gaia colour-magnitude diagram panel for the UMAP explorer.
 *
 *   const hr = new HRPanel(container, {onHover, onClick, onLasso, onDraw, lut});
 *   hr.setData(x, y);             // Float32Array each (BP-RP0, M_G0); NaN ok
 *   hr.setFlags(flags);           // Uint8Array(N): 0 hidden, 1 shown
 *   hr.setSelection(indices);     // drawn as points on top
 *   hr.setMarker(i);              // the loaded star (same crosshair as the map)
 *   hr.setMainSequence({x, y});   // running-median main sequence
 *   hr.setColor(values, lo, hi, lut, log); hr.setMode("density" | "mean")
 *
 * The shown stars are binned on a pixel grid at every redraw: log density,
 * or the mean of the colour column per cell with the density as brightness.
 * Drag pans, wheel zooms about the cursor, shift+drag lassos, click picks
 * the nearest shown star, double click resets the view.  M_G0 increases
 * downwards.  Needs GridIndex and Scatter.drawMarker from scatter.js.
 */
(function () {
  "use strict";

  const EQUAL_MASS_MAG = 0.753;   // 2.5 log10(2)

  function niceStep(range, target) {
    const raw = Math.abs(range) / target;
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const f = raw / p;
    return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p;
  }

  function greyLut() {
    const out = new Uint8Array(256 * 3);
    for (let i = 0; i < 256; i++) {
      const v = Math.round(70 + (235 - 70) * i / 255);
      out[3 * i] = v; out[3 * i + 1] = v; out[3 * i + 2] = v;
    }
    return out;
  }

  class HRPanel {
    constructor(container, opts) {
      this.el = container;
      this.opts = opts || {};
      const mk = () => {
        const c = document.createElement("canvas");
        c.style.position = "absolute";
        c.style.left = "0";
        c.style.top = "0";
        c.style.width = "100%";
        c.style.height = "100%";
        container.appendChild(c);
        return c;
      };
      this.base = mk();
      this.over = mk();
      this.over.style.cursor = "crosshair";
      this.ctx = this.base.getContext("2d");
      this.octx = this.over.getContext("2d");
      this.margin = { l: 40, r: 8, t: 6, b: 28 };
      this.cell = 2;                      // CSS px per density cell
      this.lut = this.opts.lut || greyLut();
      this.background = [18, 18, 22];
      this.panel = "#1b1c22";
      this.n = 0;
      this.xy = null;
      this.flags = null;
      this.index = null;
      this.selection = null;
      this.marker = -1;
      this.hoverIdx = -1;
      this.view = null;
      this.home = null;
      this.ms = null;
      this.mode = "density";
      this.values = null;
      this.color = null;
      this.lasso = null;
      this.drag = null;
      this.nShown = 0;
      this.dpr = window.devicePixelRatio || 1;
      this._bindEvents();
      this.resize();
      if (window.ResizeObserver) {
        new ResizeObserver(() => this.resize()).observe(container);
      }
    }

    // -- data -----------------------------------------------------------------
    setData(x, y) {
      const n = x.length;
      const xy = new Float32Array(2 * n);
      for (let i = 0; i < n; i++) { xy[2 * i] = x[i]; xy[2 * i + 1] = y[i]; }
      this.xy = xy;
      this.n = n;
      this.index = new GridIndex(xy, n);
      // Home view: 0.1-99.9 percentiles of a subsample, padded.
      const xs = [], ys = [];
      const step = Math.max(1, Math.floor(n / 200000));
      for (let i = 0; i < n; i += step) {
        if (Number.isFinite(x[i]) && Number.isFinite(y[i])) { xs.push(x[i]); ys.push(y[i]); }
      }
      xs.sort((a, b) => a - b);
      ys.sort((a, b) => a - b);
      const q = (a, p) => (a.length ? a[Math.min(a.length - 1, Math.floor(p * (a.length - 1)))] : 0);
      const x0 = q(xs, 0.001), x1 = q(xs, 0.999), y0 = q(ys, 0.001), y1 = q(ys, 0.999);
      const px = 0.06 * ((x1 - x0) || 1), py = 0.06 * ((y1 - y0) || 1);
      this.home = { x0: x0 - px, x1: x1 + px, y0: y0 - py, y1: y1 + py };
      this.marker = -1;
      this.hoverIdx = -1;
      this.selection = null;
      this.fitView();
    }
    setFlags(flags) { this.flags = flags; this.redraw(); }
    setSelection(indices) {
      this.selection = indices && indices.length ? indices : null;
      this.redraw();
    }
    setMarker(i) { this.marker = i; this.redraw(); }
    setMainSequence(curve) {
      this.ms = curve && curve.x && curve.x.length ? curve : null;
      this.redraw();
    }
    setColor(values, lo, hi, lut, log) {
      this.values = values;
      this.color = { lo, hi, lut, log };
      if (this.mode === "mean") this.redraw();
    }
    setMode(mode) { this.mode = mode; this.redraw(); }
    fitView() {
      if (!this.home) return;
      this.view = Object.assign({}, this.home);
      this.redraw();
    }

    // -- geometry (CSS px) ------------------------------------------------------
    area() {
      const W = this.base.width / this.dpr, H = this.base.height / this.dpr, m = this.margin;
      return { x: m.l, y: m.t, w: Math.max(1, W - m.l - m.r), h: Math.max(1, H - m.t - m.b) };
    }
    dataToScreen(x, y) {
      const a = this.area(), v = this.view;
      return [a.x + (x - v.x0) / (v.x1 - v.x0) * a.w,
              a.y + (y - v.y0) / (v.y1 - v.y0) * a.h];
    }
    screenToData(px, py) {
      const a = this.area(), v = this.view;
      return [v.x0 + (px - a.x) / a.w * (v.x1 - v.x0),
              v.y0 + (py - a.y) / a.h * (v.y1 - v.y0)];
    }
    _pick(px, py, rpx) {
      if (!this.index || !this.view) return -1;
      const a = this.area(), v = this.view;
      if (px < a.x || px > a.x + a.w || py < a.y || py > a.y + a.h) return -1;
      const [dx, dy] = this.screenToData(px, py);
      return this.index.nearest(dx, dy, rpx * (v.x1 - v.x0) / a.w,
                                rpx * (v.y1 - v.y0) / a.h, this.flags);
    }

    resize() {
      const dpr = window.devicePixelRatio || 1;
      const w = this.el.clientWidth, h = this.el.clientHeight;
      if (!w || !h) return;
      for (const c of [this.base, this.over]) {
        c.width = Math.round(w * dpr);
        c.height = Math.round(h * dpr);
      }
      this.dpr = dpr;
      this.redraw();
    }

    // -- drawing ----------------------------------------------------------------
    redraw() {
      if (this._pending) return;
      this._pending = true;
      const run = () => {
        if (!this._pending) return;
        this._pending = false;
        cancelAnimationFrame(this._raf);
        clearTimeout(this._tmo);
        this._draw();
      };
      this._raf = requestAnimationFrame(run);
      this._tmo = setTimeout(run, 100);
    }

    _draw() {
      const dpr = this.dpr;
      if (this.base.width !== Math.round(this.el.clientWidth * dpr)
          || this.base.height !== Math.round(this.el.clientHeight * dpr)) {
        this.resize();
      }
      const ctx = this.ctx;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = this.panel;
      ctx.fillRect(0, 0, this.base.width, this.base.height);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const a = this.area();
      const bg = this.background;
      ctx.fillStyle = `rgb(${bg[0]},${bg[1]},${bg[2]})`;
      ctx.fillRect(a.x, a.y, a.w, a.h);
      this.nShown = 0;
      if (this.xy && this.view) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(a.x, a.y, a.w, a.h);
        ctx.clip();
        this._drawDensity(ctx, a);
        this._drawMainSequence(ctx, a);
        ctx.restore();
      }
      if (this.view) this._drawAxes(ctx, a);
      this._drawOverlay();
      if (this.opts.onDraw) this.opts.onDraw(this.nShown);
    }

    _drawDensity(ctx, a) {
      const cell = this.cell;
      const nx = Math.max(1, Math.ceil(a.w / cell)), ny = Math.max(1, Math.ceil(a.h / cell));
      const counts = new Uint32Array(nx * ny);
      const mean = this.mode === "mean" && this.values && this.color;
      const vsum = mean ? new Float64Array(nx * ny) : null;
      const vcnt = mean ? new Uint32Array(nx * ny) : null;
      const v = this.view, xy = this.xy, flags = this.flags, vals = this.values;
      const sx = nx / (v.x1 - v.x0), sy = ny / (v.y1 - v.y0);
      const log = mean && this.color.log && this.color.lo > 0;
      let shown = 0;
      for (let i = 0; i < this.n; i++) {
        if (flags && !flags[i]) continue;
        const x = xy[2 * i], y = xy[2 * i + 1];
        if (!(x >= v.x0 && x < v.x1 && y >= v.y0 && y < v.y1)) continue;
        shown++;
        const k = Math.floor((y - v.y0) * sy) * nx + Math.floor((x - v.x0) * sx);
        counts[k]++;
        if (mean) {
          let z = vals[i];
          if (Number.isFinite(z) && !(log && z <= 0)) {
            if (log) z = Math.log10(z);
            vsum[k] += z;
            vcnt[k]++;
          }
        }
      }
      this.nShown = shown;
      let max = 0;
      for (let k = 0; k < counts.length; k++) if (counts[k] > max) max = counts[k];
      const img = ctx.createImageData(nx, ny);
      const d = img.data, bg = this.background, lut = this.lut;
      const lmax = Math.log1p(max) || 1;
      let lo = 0, hi = 1, inv = 1, clut = null;
      if (mean) {
        lo = this.color.lo; hi = this.color.hi;
        if (log) { lo = Math.log10(lo); hi = Math.log10(hi); }
        inv = 1 / ((hi - lo) || 1);
        clut = this.color.lut;
      }
      for (let k = 0; k < counts.length; k++) {
        const c = counts[k], o = 4 * k;
        d[o + 3] = 255;
        if (!c) { d[o] = bg[0]; d[o + 1] = bg[1]; d[o + 2] = bg[2]; continue; }
        const f = Math.log1p(c) / lmax;
        if (mean && vcnt[k]) {
          let g = (vsum[k] / vcnt[k] - lo) * inv;
          g = g < 0 ? 0 : g > 1 ? 1 : g;
          const j = 3 * Math.round(g * 255);
          const w = 0.45 + 0.55 * f;
          d[o] = bg[0] + (clut[j] - bg[0]) * w;
          d[o + 1] = bg[1] + (clut[j + 1] - bg[1]) * w;
          d[o + 2] = bg[2] + (clut[j + 2] - bg[2]) * w;
        } else {
          const j = 3 * Math.round(f * 255);
          d[o] = lut[j]; d[o + 1] = lut[j + 1]; d[o + 2] = lut[j + 2];
        }
      }
      const tmp = this._tmp || (this._tmp = document.createElement("canvas"));
      if (tmp.width !== nx || tmp.height !== ny) { tmp.width = nx; tmp.height = ny; }
      tmp.getContext("2d").putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(tmp, a.x, a.y, nx * cell, ny * cell);
    }

    _drawMainSequence(ctx, a) {
      if (!this.ms) return;
      const trace = (dy, dash, color, width) => {
        ctx.beginPath();
        for (let k = 0; k < this.ms.x.length; k++) {
          const s = this.dataToScreen(this.ms.x[k], this.ms.y[k] + dy);
          if (k === 0) ctx.moveTo(s[0], s[1]); else ctx.lineTo(s[0], s[1]);
        }
        ctx.setLineDash(dash);
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.stroke();
      };
      trace(0, [5, 3], "rgba(255, 211, 78, 0.9)", 1.2);
      trace(-EQUAL_MASS_MAG, [2, 3], "rgba(255, 211, 78, 0.5)", 1);
      ctx.setLineDash([]);
      void a;
    }

    _drawAxes(ctx, a) {
      const v = this.view;
      ctx.strokeStyle = "#33343d";
      ctx.lineWidth = 1;
      ctx.strokeRect(a.x + 0.5, a.y + 0.5, a.w - 1, a.h - 1);
      ctx.fillStyle = "#9a9aa5";
      ctx.font = "10px -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif";
      const fmtTick = (t, step) => {
        const dec = Math.max(0, -Math.floor(Math.log10(step)));
        return (Math.abs(t) < 1e-9 ? 0 : t).toFixed(dec);
      };
      // x ticks (BP-RP0, increasing to the right).
      const xs = niceStep(v.x1 - v.x0, 5);
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let t = Math.ceil(v.x0 / xs) * xs; t <= v.x1 + 1e-9; t += xs) {
        const px = this.dataToScreen(t, v.y0)[0];
        ctx.beginPath();
        ctx.moveTo(px, a.y + a.h);
        ctx.lineTo(px, a.y + a.h + 4);
        ctx.stroke();
        ctx.fillText(fmtTick(t, xs), px, a.y + a.h + 5);
      }
      // y ticks (M_G0, increasing downwards).
      const ys = niceStep(v.y1 - v.y0, 5);
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let t = Math.ceil(v.y0 / ys) * ys; t <= v.y1 + 1e-9; t += ys) {
        const py = this.dataToScreen(v.x0, t)[1];
        ctx.beginPath();
        ctx.moveTo(a.x - 4, py);
        ctx.lineTo(a.x, py);
        ctx.stroke();
        ctx.fillText(fmtTick(t, ys), a.x - 6, py);
      }
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText("BP-RP0", a.x + a.w / 2, a.y + a.h + this.margin.b - 1);
      ctx.save();
      ctx.translate(10, a.y + a.h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textBaseline = "middle";
      ctx.fillText("M_G0", 0, 0);
      ctx.restore();
    }

    _drawOverlay() {
      const ctx = this.octx, dpr = this.dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, this.over.width, this.over.height);
      if (!this.xy || !this.view) return;
      const a = this.area();
      ctx.save();
      ctx.beginPath();
      ctx.rect(a.x, a.y, a.w, a.h);
      ctx.clip();
      if (this.selection) {
        const flags = this.flags, xy = this.xy, sel = this.selection;
        ctx.fillStyle = "rgba(255, 211, 78, 0.85)";
        for (let q = 0; q < sel.length; q++) {
          const i = sel[q];
          if (flags && !flags[i]) continue;
          const x = xy[2 * i], y = xy[2 * i + 1];
          if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
          const s = this.dataToScreen(x, y);
          ctx.fillRect(s[0] - 1, s[1] - 1, 2, 2);
        }
      }
      if (this.lasso && this.lasso.length > 1) {
        ctx.beginPath();
        this.lasso.forEach((p, k) => {
          const s = this.dataToScreen(p[0], p[1]);
          if (k === 0) ctx.moveTo(s[0], s[1]); else ctx.lineTo(s[0], s[1]);
        });
        ctx.closePath();
        ctx.strokeStyle = "rgba(255, 220, 80, 0.9)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.fillStyle = "rgba(255, 220, 80, 0.12)";
        ctx.fill();
      }
      const at = (i) => {
        const x = this.xy[2 * i], y = this.xy[2 * i + 1];
        return Number.isFinite(x) && Number.isFinite(y) ? this.dataToScreen(x, y) : null;
      };
      if (this.hoverIdx >= 0) {
        const s = at(this.hoverIdx);
        if (s) Scatter.drawHover(ctx, s[0], s[1]);
      }
      if (this.marker >= 0) {
        const s = at(this.marker);
        if (s) Scatter.drawMarker(ctx, s[0], s[1]);
      }
      ctx.restore();
    }

    // -- events -------------------------------------------------------------
    _bindEvents() {
      const el = this.over;
      el.style.touchAction = "none";
      el.addEventListener("pointerdown", (e) => {
        if (!this.view) return;
        try { el.setPointerCapture(e.pointerId); } catch (err) { /* synthetic events */ }
        const [dx, dy] = this.screenToData(e.offsetX, e.offsetY);
        if (e.shiftKey) {
          this.lasso = [[dx, dy]];
          this.drag = { kind: "lasso" };
        } else {
          this.drag = { kind: "pan", x: e.offsetX, y: e.offsetY,
                        view: Object.assign({}, this.view), moved: false };
        }
      });
      el.addEventListener("pointermove", (e) => {
        if (!this.view) return;
        if (this.drag && this.drag.kind === "lasso") {
          const [dx, dy] = this.screenToData(e.offsetX, e.offsetY);
          this.lasso.push([dx, dy]);
          this.redraw();
          return;
        }
        if (this.drag && this.drag.kind === "pan") {
          const a = this.area(), v0 = this.drag.view;
          const kx = (v0.x1 - v0.x0) / a.w, ky = (v0.y1 - v0.y0) / a.h;
          const mx = e.offsetX - this.drag.x, my = e.offsetY - this.drag.y;
          if (Math.abs(mx) + Math.abs(my) > 2) this.drag.moved = true;
          this.view = { x0: v0.x0 - mx * kx, x1: v0.x1 - mx * kx,
                        y0: v0.y0 - my * ky, y1: v0.y1 - my * ky };
          this.redraw();
          return;
        }
        const i = this._pick(e.offsetX, e.offsetY, 8);
        if (i !== this.hoverIdx) {
          this.hoverIdx = i;
          this.redraw();
          if (this.opts.onHover) this.opts.onHover(i, e);
        }
      });
      const finish = (e) => {
        if (!this.drag) return;
        const d = this.drag;
        this.drag = null;
        if (d.kind === "lasso") {
          const poly = this.lasso;
          this.lasso = null;
          this.redraw();
          if (poly && poly.length > 2 && this.opts.onLasso && this.index) {
            this.opts.onLasso(this.index.pointsInPolygon(poly, this.flags));
          }
        } else if (d.kind === "pan" && !d.moved) {
          const i = this._pick(e.offsetX, e.offsetY, 10);
          if (i >= 0 && this.opts.onClick) this.opts.onClick(i, e);
        }
      };
      el.addEventListener("pointerup", finish);
      el.addEventListener("pointercancel", () => { this.drag = null; this.lasso = null; this.redraw(); });
      el.addEventListener("pointerleave", () => {
        if (this.hoverIdx !== -1) {
          this.hoverIdx = -1;
          this.redraw();
          if (this.opts.onHover) this.opts.onHover(-1);
        }
      });
      el.addEventListener("wheel", (e) => {
        if (!this.view) return;
        e.preventDefault();
        const f = Math.exp(0.0015 * e.deltaY);
        const [dx, dy] = this.screenToData(e.offsetX, e.offsetY);
        const v = this.view;
        this.view = { x0: dx + (v.x0 - dx) * f, x1: dx + (v.x1 - dx) * f,
                      y0: dy + (v.y0 - dy) * f, y1: dy + (v.y1 - dy) * f };
        this.redraw();
      }, { passive: false });
      el.addEventListener("dblclick", () => this.fitView());
      window.addEventListener("resize", () => this.resize());
    }
  }

  window.HRPanel = HRPanel;
})();
