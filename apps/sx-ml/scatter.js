/* scatter.js: minimal WebGL2 scatter plot for ~1e6 points.
 *
 *   const sc = new Scatter(canvas, overlay, {onHover, onClick, onLasso});
 *   sc.setPoints(xy);            // Float32Array(2 N): x0 y0 x1 y1 ...
 *   sc.setColors(rgb);           // Uint8Array(3 N)
 *   sc.setFlags(flags);          // Uint8Array(N): 0 hidden, 1 shown
 *   sc.setSelection(indices);    // drawn on top, others dimmed
 *   sc.setPointSize(px); sc.setAlpha(a); sc.fitView(); sc.centerOn(i)
 *
 * Drag pans, wheel zooms about the cursor, shift+drag draws a lasso,
 * click picks the nearest visible point.  A uniform grid over the data
 * gives nearest-point queries in ~1 ms.
 */
(function () {
  "use strict";

  const VS = `#version 300 es
    precision highp float;
    precision highp int;
    in vec2 a_pos;
    in vec3 a_col;
    in float a_flag;
    uniform vec2 u_scale;
    uniform vec2 u_offset;
    uniform float u_size;
    uniform float u_dim;
    uniform int u_mode;   // 0 normal pass, 1 selection pass
    out vec4 v_col;
    void main() {
      vec2 p = a_pos * u_scale + u_offset;
      gl_Position = vec4(p, 0.0, 1.0);
      float alpha = 1.0;
      if (a_flag < 0.5) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); }
      if (u_mode == 0) { alpha = u_dim; gl_PointSize = u_size; }
      else { gl_PointSize = u_size * 1.8 + 2.0; }
      v_col = vec4(a_col / 255.0, alpha);
    }`;
  const FS = `#version 300 es
    precision highp float;
    precision highp int;
    in vec4 v_col;
    uniform float u_alpha;
    uniform int u_mode;
    out vec4 outColor;
    void main() {
      vec2 c = gl_PointCoord - vec2(0.5);
      float r2 = dot(c, c);
      if (r2 > 0.25) discard;
      if (u_mode == 1 && r2 < 0.12) { outColor = vec4(v_col.rgb, 1.0); return; }
      if (u_mode == 1) { outColor = vec4(0.0, 0.0, 0.0, 1.0); return; }
      outColor = vec4(v_col.rgb, u_alpha * v_col.a);
    }`;

  function compile(gl, type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(sh));
    }
    return sh;
  }

  /* Uniform grid over interleaved (x, y) points: nearest visible point
   * and point-in-polygon queries.  Shared by the UMAP scatter and the
   * HR-diagram panel. */
  class GridIndex {
    constructor(xy, n) {
      this.xy = xy;
      this.n = n;
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (let i = 0; i < n; i++) {
        const x = xy[2 * i], y = xy[2 * i + 1];
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        if (x < x0) x0 = x; if (x > x1) x1 = x;
        if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
      if (!Number.isFinite(x0)) { x0 = -1; x1 = 1; y0 = -1; y1 = 1; }
      const pad = 1e-6 * Math.max(x1 - x0, y1 - y0, 1e-9);
      this.bbox = { x0: x0 - pad, x1: x1 + pad, y0: y0 - pad, y1: y1 + pad };
      const G = Math.max(16, Math.min(1024, Math.ceil(Math.sqrt(n / 4))));
      this.G = G;
      const counts = new Uint32Array(G * G + 1);
      const cell = new Int32Array(n).fill(-1);
      const sx = G / (this.bbox.x1 - this.bbox.x0);
      const sy = G / (this.bbox.y1 - this.bbox.y0);
      for (let i = 0; i < n; i++) {
        const x = xy[2 * i], y = xy[2 * i + 1];
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        const cx = Math.min(G - 1, Math.floor((x - this.bbox.x0) * sx));
        const cy = Math.min(G - 1, Math.floor((y - this.bbox.y0) * sy));
        const c = cy * G + cx;
        cell[i] = c;
        counts[c + 1]++;
      }
      for (let c = 0; c < G * G; c++) counts[c + 1] += counts[c];
      const fill = counts.slice(0, G * G);
      const order = new Uint32Array(counts[G * G]);
      for (let i = 0; i < n; i++) {
        const c = cell[i];
        if (c < 0) continue;
        order[fill[c]++] = i;
      }
      this.offsets = counts;
      this.order = order;
      this.scale = [sx, sy];
    }

    nearest(dx, dy, rx, ry, flags) {
      // Nearest point (flags[i] != 0) to (dx, dy) within the ellipse of
      // semi-axes rx, ry (data units); -1 if none.
      const G = this.G, [sx, sy] = this.scale, b = this.bbox;
      const cx0 = Math.max(0, Math.floor((dx - rx - b.x0) * sx));
      const cx1 = Math.min(G - 1, Math.floor((dx + rx - b.x0) * sx));
      const cy0 = Math.max(0, Math.floor((dy - ry - b.y0) * sy));
      const cy1 = Math.min(G - 1, Math.floor((dy + ry - b.y0) * sy));
      let best = -1, bestD = 1;
      const xy = this.xy, ix = 1 / rx, iy = 1 / ry;
      for (let cy = cy0; cy <= cy1; cy++) {
        for (let cx = cx0; cx <= cx1; cx++) {
          const c = cy * G + cx;
          const lo = this.offsets[c], hi = this.offsets[c + 1];
          for (let k = lo; k < hi; k++) {
            const i = this.order[k];
            if (flags && !flags[i]) continue;
            const ddx = (xy[2 * i] - dx) * ix, ddy = (xy[2 * i + 1] - dy) * iy;
            const d = ddx * ddx + ddy * ddy;
            if (d < bestD) { bestD = d; best = i; }
          }
        }
      }
      return best;
    }

    pointsInPolygon(poly, flags) {
      // Ray casting over all points (flags[i] != 0) inside the polygon's
      // bounding box; poly is [[x, y], ...] in data units.
      const n = this.n, xy = this.xy;
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const p of poly) {
        if (p[0] < x0) x0 = p[0]; if (p[0] > x1) x1 = p[0];
        if (p[1] < y0) y0 = p[1]; if (p[1] > y1) y1 = p[1];
      }
      const m = poly.length;
      const px = new Float64Array(m), py = new Float64Array(m);
      for (let k = 0; k < m; k++) { px[k] = poly[k][0]; py[k] = poly[k][1]; }
      const out = [];
      for (let i = 0; i < n; i++) {
        if (flags && !flags[i]) continue;
        const x = xy[2 * i], y = xy[2 * i + 1];
        if (x < x0 || x > x1 || y < y0 || y > y1) continue;
        let inside = false;
        for (let a = 0, b = m - 1; a < m; b = a++) {
          if (((py[a] > y) !== (py[b] > y)) &&
              (x < (px[b] - px[a]) * (y - py[a]) / (py[b] - py[a]) + px[a])) {
            inside = !inside;
          }
        }
        if (inside) out.push(i);
      }
      return Uint32Array.from(out);
    }
  }

  /* The red crosshair that marks the loaded star, and the hover ring;
   * (x, y) in CSS pixels on a context already scaled by the DPR. */
  function drawMarker(ctx, x, y) {
    ctx.beginPath();
    ctx.arc(x, y, 10, 0, 2 * Math.PI);
    ctx.strokeStyle = "#ff5050";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x - 16, y); ctx.lineTo(x - 10, y);
    ctx.moveTo(x + 10, y); ctx.lineTo(x + 16, y);
    ctx.moveTo(x, y - 16); ctx.lineTo(x, y - 10);
    ctx.moveTo(x, y + 10); ctx.lineTo(x, y + 16);
    ctx.stroke();
  }
  function drawHover(ctx, x, y) {
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, 2 * Math.PI);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  class Scatter {
    constructor(canvas, overlay, opts) {
      this.canvas = canvas;
      this.overlay = overlay;
      this.ctx = overlay.getContext("2d");
      this.opts = opts || {};
      const gl = canvas.getContext("webgl2", { antialias: false, alpha: false,
                                               premultipliedAlpha: false });
      this.gl = gl;
      this.mode = gl ? "webgl2" : "canvas2d";
      if (gl) {
        this._initGL(gl);
      } else {
        // Fallback: splat the points into an ImageData (slower, works
        // everywhere).
        this.ctx2 = canvas.getContext("2d");
        if (!this.ctx2) throw new Error("no WebGL2 and no 2D canvas");
        this.img = null;
      }
      this.n = 0;
      this.xy = null;
      this.flags = null;
      this.selection = null;
      this.nSel = 0;
      this.pointSize = 2.0;
      this.alpha = 0.6;
      this.view = { cx: 0, cy: 0, half: 1 };   // data units; square aspect
      this.hoverIdx = -1;
      this.lasso = null;
      this.drag = null;
      this.background = [0.07, 0.07, 0.09];
      this.dpr = window.devicePixelRatio || 1;
      this._bindEvents();
      this.resize();
      if (window.ResizeObserver) {
        new ResizeObserver(() => this.resize()).observe(this.canvas);
      }
    }

    _initGL(gl) {
      const prog = gl.createProgram();
      gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
      gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(prog));
      }
      this.prog = prog;
      gl.useProgram(prog);
      this.loc = {
        pos: gl.getAttribLocation(prog, "a_pos"),
        col: gl.getAttribLocation(prog, "a_col"),
        flag: gl.getAttribLocation(prog, "a_flag"),
        scale: gl.getUniformLocation(prog, "u_scale"),
        offset: gl.getUniformLocation(prog, "u_offset"),
        size: gl.getUniformLocation(prog, "u_size"),
        alpha: gl.getUniformLocation(prog, "u_alpha"),
        dim: gl.getUniformLocation(prog, "u_dim"),
        mode: gl.getUniformLocation(prog, "u_mode"),
      };
      this.bufPos = gl.createBuffer();
      this.bufCol = gl.createBuffer();
      this.bufFlag = gl.createBuffer();
      this.bufSel = gl.createBuffer();
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    }

    // -- data ---------------------------------------------------------------
    setPoints(xy) {
      const gl = this.gl;
      this.xy = xy;
      this.n = xy.length / 2;
      if (gl) {
        gl.bindBuffer(gl.ARRAY_BUFFER, this.bufPos);
        gl.bufferData(gl.ARRAY_BUFFER, xy, gl.STATIC_DRAW);
      }
      if (!this.flags || this.flags.length !== this.n) {
        this.flags = new Uint8Array(this.n).fill(1);
        for (let i = 0; i < this.n; i++) {
          if (!Number.isFinite(xy[2 * i])) this.flags[i] = 0;
        }
        this._uploadFlags();
      }
      if (!this.colors || this.colors.length !== 3 * this.n) {
        this.setColors(new Uint8Array(3 * this.n).fill(160));
      }
      this._buildIndex();
      this.fitView();
    }

    setColors(rgb) {
      const gl = this.gl;
      this.colors = rgb;
      if (gl) {
        gl.bindBuffer(gl.ARRAY_BUFFER, this.bufCol);
        gl.bufferData(gl.ARRAY_BUFFER, rgb, gl.STATIC_DRAW);
      }
      this.redraw();
    }

    setFlags(flags) {
      this.flags = flags;
      if (this.xy) {
        for (let i = 0; i < this.n; i++) {
          if (!Number.isFinite(this.xy[2 * i])) this.flags[i] = 0;
        }
      }
      this._uploadFlags();
      this.redraw();
    }

    _uploadFlags() {
      const gl = this.gl;
      if (!gl) return;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.bufFlag);
      gl.bufferData(gl.ARRAY_BUFFER, this.flags, gl.STATIC_DRAW);
    }

    setSelection(indices) {
      const gl = this.gl;
      if (!indices || indices.length === 0) {
        this.selection = null;
        this.nSel = 0;
      } else {
        this.selection = indices instanceof Uint32Array ? indices
          : Uint32Array.from(indices);
        this.nSel = this.selection.length;
        if (gl) {
          gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.bufSel);
          gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, this.selection, gl.STATIC_DRAW);
        }
      }
      this.redraw();
    }

    setPointSize(px) { this.pointSize = px; this.redraw(); }
    setAlpha(a) { this.alpha = a; this.redraw(); }

    // -- spatial index --------------------------------------------------------
    _buildIndex() {
      this.index = new GridIndex(this.xy, this.n);
      this.bbox = this.index.bbox;
    }

    nearest(dx, dy, maxDataDist) {
      // Nearest visible point to data coordinates (dx, dy).
      if (!this.index) return -1;
      return this.index.nearest(dx, dy, maxDataDist, maxDataDist, this.flags);
    }

    // -- view -----------------------------------------------------------------
    resize() {
      const dpr = window.devicePixelRatio || 1;
      const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
      if (w === 0 || h === 0) return;
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
      this.overlay.width = Math.round(w * dpr);
      this.overlay.height = Math.round(h * dpr);
      this.dpr = dpr;
      if (this.gl) this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      this.img = null;
      this.redraw();
    }

    fitView() {
      if (!this.bbox) return;
      const b = this.bbox;
      this.view.cx = 0.5 * (b.x0 + b.x1);
      this.view.cy = 0.5 * (b.y0 + b.y1);
      this.view.half = 0.53 * Math.max(b.x1 - b.x0, b.y1 - b.y0);
      this.redraw();
    }

    centerOn(i, zoomHalf) {
      if (!this.xy || i < 0) return;
      this.view.cx = this.xy[2 * i];
      this.view.cy = this.xy[2 * i + 1];
      if (zoomHalf) this.view.half = zoomHalf;
      this.redraw();
    }

    _transform() {
      // Square data aspect: the shorter canvas side spans 2 * half.
      const w = this.canvas.width, h = this.canvas.height;
      const aspect = w / h;
      const hx = aspect >= 1 ? this.view.half * aspect : this.view.half;
      const hy = aspect >= 1 ? this.view.half : this.view.half / aspect;
      return { sx: 1 / hx, sy: 1 / hy, hx, hy };
    }

    dataToScreen(x, y) {
      const t = this._transform();
      const cx = (x - this.view.cx) * t.sx;   // clip
      const cy = (y - this.view.cy) * t.sy;
      return [(cx + 1) * 0.5 * this.canvas.width / this.dpr,
              (1 - cy) * 0.5 * this.canvas.height / this.dpr];
    }

    screenToData(px, py) {
      const t = this._transform();
      const cx = 2 * px * this.dpr / this.canvas.width - 1;
      const cy = 1 - 2 * py * this.dpr / this.canvas.height;
      return [this.view.cx + cx / t.sx, this.view.cy + cy / t.sy];
    }

    pixelToData() {
      // Data units per CSS pixel.
      const t = this._transform();
      return 2 * t.hx / (this.canvas.width / this.dpr);
    }

    // -- drawing ----------------------------------------------------------------
    redraw() {
      // Coalesce redraws into one frame.  A timeout backs up the animation
      // frame so a stalled frame (hidden tab, headless browser) cannot
      // leave the latch set and the canvas blank.
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
      const gl = this.gl;
      if (this.canvas.width !== Math.round(this.canvas.clientWidth * this.dpr)
          || this.canvas.height !== Math.round(this.canvas.clientHeight * this.dpr)) {
        this.resize();
      }
      if (!gl) { this._draw2d(); return; }
      const bg = this.background;
      gl.clearColor(bg[0], bg[1], bg[2], 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      if (!this.n) { this._drawOverlay(); return; }
      gl.useProgram(this.prog);
      const t = this._transform();
      gl.uniform2f(this.loc.scale, t.sx, t.sy);
      gl.uniform2f(this.loc.offset, -this.view.cx * t.sx, -this.view.cy * t.sy);
      gl.uniform1f(this.loc.size, this.pointSize * this.dpr);
      gl.uniform1f(this.loc.alpha, this.alpha);
      gl.uniform1f(this.loc.dim, this.nSel ? 0.25 : 1.0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.bufPos);
      gl.enableVertexAttribArray(this.loc.pos);
      gl.vertexAttribPointer(this.loc.pos, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.bufCol);
      gl.enableVertexAttribArray(this.loc.col);
      gl.vertexAttribPointer(this.loc.col, 3, gl.UNSIGNED_BYTE, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.bufFlag);
      gl.enableVertexAttribArray(this.loc.flag);
      gl.vertexAttribPointer(this.loc.flag, 1, gl.UNSIGNED_BYTE, false, 0, 0);
      gl.uniform1i(this.loc.mode, 0);
      gl.drawArrays(gl.POINTS, 0, this.n);
      if (this.nSel) {
        gl.uniform1i(this.loc.mode, 1);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.bufSel);
        gl.drawElements(gl.POINTS, this.nSel, gl.UNSIGNED_INT, 0);
      }
      this._drawOverlay();
    }

    _disc(size) {
      // Pixel offsets of a disc of the given diameter (cached).
      if (this._discSize === size) return this._disc_;
      const r = size / 2, out = [];
      const lo = -Math.ceil(r), hi = Math.ceil(r);
      for (let dy = lo; dy <= hi; dy++) {
        for (let dx = lo; dx <= hi; dx++) {
          if (size <= 2 ? (dx >= 0 && dx < size && dy >= 0 && dy < size)
              : (dx * dx + dy * dy <= r * r)) out.push(dx, dy);
        }
      }
      this._discSize = size;
      this._disc_ = Int16Array.from(out);
      return this._disc_;
    }

    _draw2d() {
      const ctx = this.ctx2;
      const W = this.canvas.width, H = this.canvas.height;
      if (!W || !H) return;
      if (!this.img || this.img.width !== W || this.img.height !== H) {
        this.img = ctx.createImageData(W, H);
      }
      const d = this.img.data;
      const bg = this.background.map((v) => Math.round(255 * v));
      for (let k = 0; k < d.length; k += 4) {
        d[k] = bg[0]; d[k + 1] = bg[1]; d[k + 2] = bg[2]; d[k + 3] = 255;
      }
      if (this.n) {
        const t = this._transform();
        const ax = 0.5 * W * t.sx, bx = 0.5 * W * (1 - this.view.cx * t.sx);
        const ay = 0.5 * H * t.sy, by = 0.5 * H * (1 + this.view.cy * t.sy);
        const xy = this.xy, flags = this.flags, col = this.colors;
        const size = Math.max(1, Math.round(this.pointSize * this.dpr));
        const disc = this._disc(size), m = disc.length;
        const a = Math.round(255 * this.alpha * (this.nSel ? 0.25 : 1));
        const pad = size + 1;
        for (let i = 0; i < this.n; i++) {
          if (!flags[i]) continue;
          const px = xy[2 * i] * ax + bx, py = by - xy[2 * i + 1] * ay;
          if (!(px > -pad && px < W + pad && py > -pad && py < H + pad)) continue;
          const x0 = Math.floor(px), y0 = Math.floor(py);
          const r = col[3 * i], g = col[3 * i + 1], b = col[3 * i + 2];
          for (let j = 0; j < m; j += 2) {
            const x = x0 + disc[j], y = y0 + disc[j + 1];
            if (x < 0 || x >= W || y < 0 || y >= H) continue;
            const k = 4 * (y * W + x);
            d[k] += ((r - d[k]) * a) >> 8;
            d[k + 1] += ((g - d[k + 1]) * a) >> 8;
            d[k + 2] += ((b - d[k + 2]) * a) >> 8;
          }
        }
        if (this.nSel) {
          const big = Math.round(size * 1.8 + 2);
          const ring = this._disc(big), mr = ring.length;
          const core = Math.max(1, Math.round(0.7 * big));
          const inner = this._disc(core), mi = inner.length;
          const sel = this.selection;
          const off = (big - core) / 2;
          for (let q = 0; q < this.nSel; q++) {
            const i = sel[q];
            if (!flags[i]) continue;
            const px = xy[2 * i] * ax + bx, py = by - xy[2 * i + 1] * ay;
            if (!(px > -big && px < W + big && py > -big && py < H + big)) continue;
            const x0 = Math.floor(px - big / 2), y0 = Math.floor(py - big / 2);
            for (let j = 0; j < mr; j += 2) {
              const x = x0 + ring[j], y = y0 + ring[j + 1];
              if (x < 0 || x >= W || y < 0 || y >= H) continue;
              const k = 4 * (y * W + x);
              d[k] = 0; d[k + 1] = 0; d[k + 2] = 0;
            }
            const x1 = Math.floor(px - big / 2 + off), y1 = Math.floor(py - big / 2 + off);
            const r = col[3 * i], g = col[3 * i + 1], b = col[3 * i + 2];
            for (let j = 0; j < mi; j += 2) {
              const x = x1 + inner[j], y = y1 + inner[j + 1];
              if (x < 0 || x >= W || y < 0 || y >= H) continue;
              const k = 4 * (y * W + x);
              d[k] = r; d[k + 1] = g; d[k + 2] = b;
            }
          }
        }
      }
      ctx.putImageData(this.img, 0, 0);
      this._drawOverlay();
    }

    _drawOverlay() {
      const ctx = this.ctx, dpr = this.dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
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
      if (this.hoverIdx >= 0 && this.xy) {
        const s = this.dataToScreen(this.xy[2 * this.hoverIdx],
                                    this.xy[2 * this.hoverIdx + 1]);
        drawHover(ctx, s[0], s[1]);
      }
      if (this.marker >= 0 && this.xy) {
        const s = this.dataToScreen(this.xy[2 * this.marker],
                                    this.xy[2 * this.marker + 1]);
        drawMarker(ctx, s[0], s[1]);
      }
    }

    setMarker(i) { this.marker = i; this.redraw(); }

    // -- events -------------------------------------------------------------
    _bindEvents() {
      const el = this.overlay;
      el.style.touchAction = "none";
      el.addEventListener("pointerdown", (e) => {
        try { el.setPointerCapture(e.pointerId); } catch (err) { /* synthetic events */ }
        const [dx, dy] = this.screenToData(e.offsetX, e.offsetY);
        if (e.shiftKey) {
          this.lasso = [[dx, dy]];
          this.drag = { kind: "lasso" };
        } else {
          this.drag = { kind: "pan", x: e.offsetX, y: e.offsetY,
                        cx: this.view.cx, cy: this.view.cy, moved: false };
        }
      });
      el.addEventListener("pointermove", (e) => {
        if (this.drag && this.drag.kind === "lasso") {
          const [dx, dy] = this.screenToData(e.offsetX, e.offsetY);
          this.lasso.push([dx, dy]);
          this.redraw();
          return;
        }
        if (this.drag && this.drag.kind === "pan") {
          const k = this.pixelToData();
          const mx = e.offsetX - this.drag.x, my = e.offsetY - this.drag.y;
          if (Math.abs(mx) + Math.abs(my) > 2) this.drag.moved = true;
          this.view.cx = this.drag.cx - mx * k;
          this.view.cy = this.drag.cy + my * k;
          this.redraw();
          return;
        }
        const [dx, dy] = this.screenToData(e.offsetX, e.offsetY);
        const i = this.nearest(dx, dy, 8 * this.pixelToData());
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
          if (poly && poly.length > 2 && this.opts.onLasso) {
            this.opts.onLasso(this.pointsInPolygon(poly));
          }
        } else if (d.kind === "pan" && !d.moved) {
          const [dx, dy] = this.screenToData(e.offsetX, e.offsetY);
          const i = this.nearest(dx, dy, 10 * this.pixelToData());
          if (i >= 0 && this.opts.onClick) this.opts.onClick(i, e);
        }
      };
      el.addEventListener("pointerup", finish);
      el.addEventListener("pointercancel", () => { this.drag = null; this.lasso = null; this.redraw(); });
      el.addEventListener("pointerleave", () => {
        if (this.hoverIdx !== -1) { this.hoverIdx = -1; this.redraw(); if (this.opts.onHover) this.opts.onHover(-1); }
      });
      el.addEventListener("wheel", (e) => {
        e.preventDefault();
        const f = Math.exp(0.0015 * e.deltaY);
        const [dx, dy] = this.screenToData(e.offsetX, e.offsetY);
        this.view.cx = dx + (this.view.cx - dx) * f;
        this.view.cy = dy + (this.view.cy - dy) * f;
        this.view.half *= f;
        this.redraw();
      }, { passive: false });
      window.addEventListener("resize", () => this.resize());
    }

    pointsInPolygon(poly) {
      // Visible points inside the lasso polygon (data units).
      if (!this.index) return new Uint32Array(0);
      return this.index.pointsInPolygon(poly, this.flags);
    }
  }

  Scatter.GridIndex = GridIndex;
  Scatter.drawMarker = drawMarker;
  Scatter.drawHover = drawHover;
  window.Scatter = Scatter;
  window.GridIndex = GridIndex;
})();
