// Canvas timeline for comparing diarizations: ruler, waveform, then groups of speaker lanes
// (reference, system) and an error lane, plus the overview strip. Read-only: click to seek,
// double-click a turn to play it, drag the ruler to scrub, wheel to scroll, Ctrl + wheel to zoom.
// Drawing is split into a cached static layer and a per-frame overlay (playhead, hover).
import { speakerColor, speakerHatched } from "./palette.js";
import { clamp, fmtTime } from "./util.js";

const GUTTER = 150;      // lane labels
const RULER_H = 22;
const WAVE_H = 56;
const GROUP_H = 22;      // group title row
const LANE_MIN = 12, LANE_MAX = 24, LANES_MAX_H = 260;
const PEAK_HOP = 0.01;   // seconds per peak pair
const INK = "#191014", MUTED2 = "#948b8f", BORDER = "#e7e1e0", PRIMARY = "#e75f1b";
const FONT = '"Kantumruy Pro", system-ui, sans-serif';

export const ERROR_COLORS = { confusion: "#e34948", miss: "#eda100", fa: "#4a3aa7" };
export const ERROR_NAMES = { confusion: "Confusion", miss: "Missed", fa: "False alarm" };

export class Timeline {
  // view: { duration, peaks, time, groups: [{title, lanes: [{label, color | fill, turns}]}],
  //         errors: [[start, end, kind]] | null (no error lane), errorsLabel,
  //         shade: [[start, end]] hatched across the lanes (e.g. not scored),
  //         onHover(t), onViewChange(), seek(t), playSpan(s, e) }
  constructor(view, canvas, overview) {
    this.v = view;
    this.cv = canvas;
    this.ov = overview;
    this.ctx = canvas.getContext("2d");
    this.octx = overview.getContext("2d");
    this.start = 0;
    this.pps = 50;
    this.minPps = 0.001;  // set by resize(), which may run after the first zoom
    this.static = document.createElement("canvas");
    this.dirty = true;
    this.hoverX = null;
    this.drag = null;
    this.frame = null;
    this._bind();
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
  }

  // ---------------------------------------------------------------- geometry
  get width() { return this.cv.clientWidth; }
  get plotW() { return Math.max(10, this.width - GUTTER); }
  get viewDur() { return this.plotW / this.pps; }
  get end() { return this.start + this.viewDur; }
  x(t) { return GUTTER + (t - this.start) * this.pps; }
  t(x) { return this.start + (x - GUTTER) / this.pps; }

  // rows from top to bottom below the waveform: group titles, lanes, the error lane
  rows() {
    const rows = [];
    for (const g of this.v.groups) {
      rows.push({ kind: "title", title: g.title, h: GROUP_H });
      for (const lane of g.lanes) rows.push({ kind: "lane", lane, h: this.laneH() });
    }
    if (this.v.errors) {
      rows.push({ kind: "title", title: "Disagreement", h: GROUP_H });
      rows.push({ kind: "errors", h: this.laneH() + 4 });
    }
    let y = RULER_H + WAVE_H;
    for (const r of rows) { r.y = y; y += r.h; }
    return rows;
  }
  laneH() {
    const n = this.v.groups.reduce((a, g) => a + g.lanes.length, 0) + (this.v.errors ? 1 : 0);
    return clamp(Math.floor(LANES_MAX_H / n), LANE_MIN, LANE_MAX);
  }
  heightNeeded() {
    const rows = this.rows(), last = rows[rows.length - 1];
    return last.y + last.h + 6;
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = this.cv.parentElement.clientWidth, h = this.heightNeeded();
    for (const c of [this.cv, this.static]) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    this.cv.style.height = h + "px";
    const ow = this.ov.clientWidth;
    this.ov.width = Math.round(ow * dpr);
    this.ov.height = Math.round(30 * dpr);
    this.minPps = this.plotW / Math.max(1, this.v.duration);
    this.pps = clamp(this.pps, this.minPps, 600);
    this.setStart(this.start);
    this.invalidate();
  }

  setStart(s) {
    this.start = clamp(s, 0, Math.max(0, this.v.duration - this.viewDur));
  }

  zoomAt(factor, t = null) {
    t ??= this.v.time;
    const fx = (this.x(t) - GUTTER) / this.plotW;
    this.pps = clamp(this.pps * factor, this.minPps, 600);
    this.setStart(t - fx * this.viewDur);
    this.invalidate();
  }

  setZoomSeconds(seconds, center) {
    this.pps = clamp(this.plotW / seconds, this.minPps, 600);
    this.setStart(center - this.viewDur / 2);
    this.invalidate();
  }

  reveal(t0, t1 = t0, margin = 0.1) {
    const m = this.viewDur * margin;
    if (t1 - t0 > this.viewDur * 0.8) this.setStart(t0 - m);
    else if (t0 < this.start + m) this.setStart(t0 - m);
    else if (t1 > this.end - m) this.setStart(t1 - this.viewDur + m);
    else return;
    this.invalidate();
  }

  invalidate() { this.dirty = true; this.requestDraw(); this.v.onViewChange?.(); }
  requestDraw() {
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => { this.frame = null; this.draw(); });
  }

  // ---------------------------------------------------------------- drawing
  draw() {
    if (this.cv.height !== Math.round(this.heightNeeded() * (window.devicePixelRatio || 1))) this.resize();
    const dpr = window.devicePixelRatio || 1;
    if (this.dirty) {
      const c = this.static.getContext("2d");
      c.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.drawStatic(c);
      this.dirty = false;
    }
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.cv.width, this.cv.height);
    ctx.drawImage(this.static, 0, 0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.drawOverlay(ctx);
    this.drawOverview();
  }

  drawStatic(c) {
    const W = this.width, H = this.heightNeeded();
    c.clearRect(0, 0, W, H);
    c.fillStyle = "#ffffff";
    c.fillRect(0, 0, W, H);
    c.textBaseline = "middle";
    this.drawRuler(c, W);
    this.drawWave(c);

    const rows = this.rows();
    let stripe = 0;
    for (const r of rows) {
      if (r.kind === "title") {
        c.fillStyle = "rgba(79,39,36,0.045)";
        c.fillRect(0, r.y, W, r.h);
        c.fillStyle = MUTED2;
        c.font = `600 10px ${FONT}`;
        c.fillText(r.title.toUpperCase(), 10, r.y + r.h / 2 + 0.5);
        stripe = 0;
      } else {
        if (stripe++ % 2 === 1) { c.fillStyle = "rgba(79,39,36,0.025)"; c.fillRect(GUTTER, r.y, W - GUTTER, r.h); }
        if (r.kind === "lane") {
          this.swatch(c, r.lane.fill ?? r.lane.color, 10, r.y + r.h / 2 - 5, 10, 10);
          c.fillStyle = INK;
          c.font = `500 11px ${FONT}`;
          c.fillText(this.ellipsis(c, r.lane.label, GUTTER - 36), 28, r.y + r.h / 2);
        } else {
          c.fillStyle = INK;
          c.font = `500 11px ${FONT}`;
          c.fillText(this.v.errorsLabel || "vs. reference", 10, r.y + r.h / 2);
        }
      }
      c.fillStyle = BORDER;
      c.fillRect(0, r.y + r.h - 0.5, W, 0.5);
    }
    c.fillStyle = BORDER;
    c.fillRect(GUTTER - 0.5, RULER_H, 0.5, H - RULER_H);

    c.save();
    c.beginPath();
    c.rect(GUTTER, 0, W - GUTTER, H);
    c.clip();
    const top = RULER_H + WAVE_H;
    for (const [s, e] of this.v.shade || []) {
      if (e < this.start || s > this.end) continue;
      const x0 = this.x(s), w = this.x(e) - x0;
      c.fillStyle = "rgba(79,39,36,0.05)";
      c.fillRect(x0, top, w, H - top);
      this.hatch(c, x0, top, w, H - top, "#4f2724", 0.14);
    }
    for (const r of rows) {
      if (r.kind === "lane") {
        for (const [s, e] of r.lane.turns) {
          if (e < this.start || s > this.end) continue;
          this.drawTurn(c, s, e, r.y + 2, r.h - 4, r.lane.fill ?? r.lane.color);
        }
      } else if (r.kind === "errors") {
        for (const [s, e, kind] of this.v.errors) {
          if (e < this.start || s > this.end) continue;
          const x0 = this.x(s), w = Math.max(1, this.x(e) - x0);
          c.fillStyle = ERROR_COLORS[kind];
          c.fillRect(x0, r.y + 3, w, r.h - 6);
          // a faint band through the lanes above, so the error lines up with the turns
          c.globalAlpha = 0.07;
          c.fillRect(x0, RULER_H + WAVE_H, w, r.y - RULER_H - WAVE_H);
          c.globalAlpha = 1;
        }
      }
    }
    c.restore();
  }

  drawRuler(c, W) {
    c.fillStyle = "#ffffff";
    c.fillRect(0, 0, W, RULER_H);
    c.fillStyle = BORDER;
    c.fillRect(0, RULER_H - 0.5, W, 0.5);
    const steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200];
    const step = steps.find((s) => s * this.pps >= 72) || 1800;
    const minor = step / 5;
    c.fillStyle = MUTED2;
    c.font = `500 10px ${FONT}`;
    const first = Math.floor(this.start / minor) * minor;
    for (let t = first; t <= this.end + minor; t += minor) {
      const x = this.x(t);
      if (x < GUTTER) continue;
      const major = Math.abs(t / step - Math.round(t / step)) < 1e-6;
      c.fillRect(x, major ? RULER_H - 9 : RULER_H - 4, 0.75, major ? 9 : 4);
      if (major) c.fillText(fmtTime(t, step < 1 ? 1 : 0), x + 4, 9);
    }
  }

  drawWave(c) {
    const peaks = this.v.peaks;
    const y0 = RULER_H, mid = y0 + WAVE_H / 2, amp = WAVE_H / 2 - 4;
    c.fillStyle = "rgba(79,39,36,0.025)";
    c.fillRect(GUTTER, y0, this.plotW, WAVE_H);
    c.fillStyle = BORDER;
    c.fillRect(GUTTER, y0 + WAVE_H - 0.5, this.plotW, 0.5);
    if (!peaks) return;
    const n = peaks.length / 2;
    c.fillStyle = "#8f7f82";
    for (let px = 0; px < this.plotW; px++) {
      const t0 = this.start + px / this.pps, t1 = t0 + 1 / this.pps;
      let i0 = Math.floor(t0 / PEAK_HOP), i1 = Math.max(i0 + 1, Math.floor(t1 / PEAK_HOP));
      if (i0 >= n) break;
      i1 = Math.min(i1, n);
      let lo = 127, hi = -127;
      for (let i = i0; i < i1; i++) {
        const a = peaks[2 * i], b = peaks[2 * i + 1];
        if (a < lo) lo = a;
        if (b > hi) hi = b;
      }
      const top = mid - (hi / 127) * amp, bot = mid - (lo / 127) * amp;
      c.fillRect(GUTTER + px, top, 1, Math.max(1, bot - top));
    }
  }

  // colour: a speaker colour index, or a CSS colour string
  drawTurn(c, s, e, y, h, colour) {
    const x0 = this.x(s), w = Math.max(1.5, this.x(e) - x0);
    c.save();
    this.roundRect(c, x0, y, w, h, Math.min(3, w / 2, h / 2));
    c.globalAlpha = 0.85;
    c.fillStyle = typeof colour === "string" ? colour : speakerColor(colour);
    c.fill();
    if (typeof colour !== "string" && speakerHatched(colour)) { c.clip(); this.hatch(c, x0, y, w, h, "#ffffff", 0.9); }
    c.restore();
  }

  drawOverlay(c) {
    const H = this.heightNeeded();
    if (this.hoverX !== null && this.hoverX >= GUTTER) {
      c.fillStyle = "rgba(25,16,20,0.25)";
      c.fillRect(this.hoverX, RULER_H, 1, H - RULER_H);
    }
    const px = this.x(this.v.time);
    if (px >= GUTTER - 1 && px <= this.width + 1) {
      c.fillStyle = PRIMARY;
      c.fillRect(px - 1, 0, 2, H);
      c.beginPath();
      c.moveTo(px - 6, 0); c.lineTo(px + 6, 0); c.lineTo(px, 8); c.closePath();
      c.fill();
    }
  }

  // overview: the error track over the whole recording, and the viewport
  drawOverview() {
    const dpr = window.devicePixelRatio || 1;
    const c = this.octx, W = this.ov.clientWidth, H = 30, D = Math.max(1, this.v.duration);
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, W, H);
    c.fillStyle = "#ffffff";
    c.fillRect(0, 0, W, H);
    const X = (t) => (t / D) * W;
    for (const [s, e] of this.v.shade || []) this.hatch(c, X(s), 3, Math.max(1, X(e) - X(s)), 24, "#4f2724", 0.5);
    for (const [s, e, kind] of this.v.errors || []) {
      c.fillStyle = ERROR_COLORS[kind];
      c.fillRect(X(s), 7, Math.max(0.6, X(e) - X(s)), 16);
    }
    const v0 = X(this.start), v1 = X(this.end);
    c.fillStyle = "rgba(79,39,36,0.07)";
    c.fillRect(v0, 0, Math.max(2, v1 - v0), H);
    c.strokeStyle = INK;
    c.lineWidth = 1;
    c.strokeRect(v0 + 0.5, 0.5, Math.max(2, v1 - v0) - 1, H - 1);
    c.fillStyle = PRIMARY;
    c.fillRect(X(this.v.time) - 0.75, 0, 1.5, H);
  }

  // ---------------------------------------------------------------- primitives
  roundRect(c, x, y, w, h, r) {
    c.beginPath();
    if (c.roundRect) c.roundRect(x, y, w, h, r);
    else c.rect(x, y, w, h);
  }
  hatch(c, x, y, w, h, color, alpha) {
    if (w <= 0 || h <= 0) return;
    c.save();
    c.beginPath();
    c.rect(x, y, w, h);
    c.clip();
    c.strokeStyle = color;
    c.globalAlpha = alpha;
    c.lineWidth = 1;
    c.beginPath();
    for (let k = -h; k < w; k += 5) { c.moveTo(x + k, y + h); c.lineTo(x + k + h, y); }
    c.stroke();
    c.restore();
  }
  swatch(c, colour, x, y, w, h) {
    c.fillStyle = typeof colour === "string" ? colour : speakerColor(colour);
    c.beginPath();
    this.roundRect(c, x, y, w, h, 2);
    c.fill();
    if (typeof colour !== "string" && speakerHatched(colour)) this.hatch(c, x, y, w, h, "#ffffff", 0.9);
  }
  ellipsis(c, text, maxW) {
    if (c.measureText(text).width <= maxW) return text;
    let lo = 0, hi = text.length;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (c.measureText(text.slice(0, mid) + "…").width <= maxW) lo = mid; else hi = mid - 1;
    }
    return text.slice(0, lo) + "…";
  }

  // the turn or error run under (x, y), as [start, end], or null
  spanAt(x, y) {
    const t = this.t(x);
    for (const r of this.rows()) {
      if (y < r.y || y >= r.y + r.h) continue;
      const spans = r.kind === "lane" ? r.lane.turns : r.kind === "errors" ? this.v.errors || [] : [];
      const hit = spans.find(([s, e]) => t >= s && t <= e);
      return hit ? [hit[0], hit[1]] : null;
    }
    return null;
  }

  // ---------------------------------------------------------------- interaction
  _bind() {
    const cv = this.cv;
    const pos = (e) => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    const timeAt = (x) => clamp(this.t(x), 0, this.v.duration);

    cv.addEventListener("pointermove", (e) => {
      const [x, y] = pos(e);
      this.hoverX = x;
      this.v.onHover?.(x >= GUTTER ? this.t(x) : null);
      if (this.drag) this.v.seek(timeAt(x));
      else cv.style.cursor = y < RULER_H ? "col-resize" : x >= GUTTER && this.spanAt(x, y) ? "pointer" : "default";
      this.requestDraw();
    });
    cv.addEventListener("pointerleave", () => { this.hoverX = null; this.v.onHover?.(null); this.requestDraw(); });
    cv.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      const [x, y] = pos(e);
      if (x < GUTTER) return;
      if (y < RULER_H) { cv.setPointerCapture(e.pointerId); this.drag = true; }
      this.v.seek(timeAt(x));
    });
    cv.addEventListener("pointerup", () => (this.drag = false));
    cv.addEventListener("pointercancel", () => (this.drag = false));
    cv.addEventListener("dblclick", (e) => {
      const [x, y] = pos(e);
      const span = x >= GUTTER && this.spanAt(x, y);
      if (span) this.v.playSpan(span[0], span[1]);
    });
    cv.addEventListener("wheel", (e) => {
      e.preventDefault();
      if (e.ctrlKey || e.metaKey) {
        const [x] = pos(e);
        this.zoomAt(Math.pow(1.0018, -e.deltaY), x >= GUTTER ? this.t(x) : null);
      } else {
        const d = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
        this.setStart(this.start + d / this.pps);
        this.invalidate();
      }
    }, { passive: false });

    // overview: click / drag to move the view
    let ovDrag = false;
    const ovMove = (e) => {
      const r = this.ov.getBoundingClientRect();
      const t = ((e.clientX - r.left) / r.width) * this.v.duration;
      this.setStart(t - this.viewDur / 2);
      this.invalidate();
    };
    this.ov.addEventListener("pointerdown", (e) => { ovDrag = true; this.ov.setPointerCapture(e.pointerId); ovMove(e); });
    this.ov.addEventListener("pointermove", (e) => { if (ovDrag) ovMove(e); });
    this.ov.addEventListener("pointerup", () => (ovDrag = false));
  }
}
