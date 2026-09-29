// Small dependency-free canvas charts. Each renderer plots only the values
// supplied by the live result/model and labels unavailable series as such.
const NirnayaCharts = (() => {
  const COLORS = { amber: '#F5A623', blue: '#5B9DF9', green: '#4FBE87',
    red: '#E5646C', dim: '#5C6773', grid: '#1D242B', text: '#93A0AD' };

  function context(canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#0F1318'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.font = '10px ui-monospace, monospace';
    return ctx;
  }
  function label(ctx, text, x, y, color = COLORS.text) {
    ctx.fillStyle = color; ctx.fillText(String(text), x, y);
  }
  function empty(canvas, text) {
    const ctx = context(canvas); label(ctx, text, 14, 22); return false;
  }

  function renderSpyPlot(canvas, model) {
    const ctx = context(canvas), w = canvas.width, h = canvas.height;
    if (!model || !model.variables || !model.constraints) return empty(canvas, 'no model loaded');
    const index = Object.fromEntries(model.variables.map((v, i) => [v.name, i]));
    const pad = 24, pw = w - pad * 2, ph = h - pad * 2;
    const cw = pw / Math.max(1, model.variables.length), ch = ph / Math.max(1, model.constraints.length);
    ctx.strokeStyle = '#262E37'; ctx.strokeRect(pad, pad, pw, ph);
    ctx.fillStyle = COLORS.amber;
    model.constraints.forEach((row, r) => Object.keys(row.coefficients || {}).forEach((name) => {
      const c = index[name]; if (c === undefined) return;
      ctx.beginPath(); ctx.arc(pad + (c + .5) * cw, pad + (r + .5) * ch,
        Math.max(1, Math.min(cw, ch) * .35), 0, Math.PI * 2); ctx.fill();
    }));
    label(ctx, `${model.constraints.length} rows × ${model.variables.length} columns`, pad, h - 6);
  }

  function renderProgress(canvas, history) {
    if (!history || history.length < 2) return empty(canvas, 'iteration history unavailable from this solver response');
    const ctx = context(canvas), w = canvas.width, h = canvas.height, p = 28;
    const values = history.map(x => Number(x.objective)).filter(Number.isFinite);
    if (values.length < 2) return empty(canvas, 'iteration history unavailable from this solver response');
    const min = Math.min(...values), max = Math.max(...values), span = max - min || 1;
    ctx.strokeStyle = COLORS.grid; ctx.beginPath(); ctx.moveTo(p, p); ctx.lineTo(p, h - p); ctx.lineTo(w - p, h - p); ctx.stroke();
    ctx.strokeStyle = COLORS.amber; ctx.lineWidth = 2; ctx.beginPath();
    values.forEach((v, i) => { const x = p + i * (w - 2 * p) / (values.length - 1); const y = h - p - (v - min) / span * (h - 2 * p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke(); label(ctx, `objective ${min.toPrecision(4)} … ${max.toPrecision(4)}`, p, 14);
    return true;
  }

  function renderResiduals(canvas, residuals) {
    if (!residuals || !Object.keys(residuals).length) return empty(canvas, 'verified residuals unavailable');
    const ctx = context(canvas), entries = Object.entries(residuals).slice(0, 24);
    const max = Math.max(1e-12, ...entries.map(([, v]) => Math.abs(Number(v))));
    const rowH = Math.max(7, (canvas.height - 12) / entries.length), left = 156, usable = canvas.width - left - 12;
    entries.forEach(([name, value], i) => {
      const y = 10 + i * rowH, width = Math.max(1, Math.abs(Number(value)) / max * usable);
      label(ctx, name.slice(0, 24), 6, y + 9);
      ctx.fillStyle = Math.abs(value) <= 1e-7 ? COLORS.green : Math.abs(value) <= 1e-5 ? COLORS.amber : COLORS.red;
      ctx.fillRect(left, y, width, Math.max(3, rowH - 3));
      label(ctx, Number(value).toExponential(1), Math.min(canvas.width - 64, left + width + 4), y + 9);
    });
    return true;
  }

  function renderVarDistribution(canvas, variables) {
    if (!variables || !Object.keys(variables).length) return empty(canvas, 'primal values unavailable');
    const vals = Object.values(variables).map(Number).filter(Number.isFinite);
    if (!vals.length) return empty(canvas, 'primal values unavailable');
    const min = Math.min(...vals), max = Math.max(...vals), binsN = Math.min(12, Math.max(4, Math.round(Math.sqrt(vals.length))));
    const span = max - min || 1, bins = Array(binsN).fill(0);
    vals.forEach(v => bins[Math.min(binsN - 1, Math.max(0, Math.floor((v - min) / span * binsN)))]++);
    const ctx = context(canvas), top = 18, bottom = 24, bw = (canvas.width - 24) / binsN, peak = Math.max(1, ...bins);
    bins.forEach((count, i) => {
      const bh = count / peak * (canvas.height - top - bottom), x = 12 + i * bw;
      ctx.fillStyle = COLORS.blue; ctx.fillRect(x + 2, canvas.height - bottom - bh, Math.max(2, bw - 5), bh);
      label(ctx, count, x + 3, canvas.height - 7);
    });
    label(ctx, `${vals.length} values · range ${min.toPrecision(3)} to ${max.toPrecision(3)}`, 12, 12);
    return true;
  }

  function renderBenchmarkChart(canvas, runs) {
    const valid = (runs || []).filter(r => Number.isFinite(r.solve_time_ms));
    if (!valid.length) return empty(canvas, 'no measured solve times in the loaded report');
    const ctx = context(canvas), max = Math.max(...valid.map(r => r.solve_time_ms), 1);
    const rowH = Math.max(14, (canvas.height - 8) / valid.length), left = 180, usable = canvas.width - left - 12;
    valid.slice(0, Math.floor((canvas.height - 8) / rowH)).forEach((r, i) => {
      const y = 6 + i * rowH, width = r.solve_time_ms / max * usable;
      label(ctx, String(r.problem).slice(0, 28), 6, y + 10);
      ctx.fillStyle = r.backend_requested === 'gpu' ? COLORS.amber : COLORS.blue;
      ctx.fillRect(left, y, Math.max(1, width), Math.max(3, rowH - 3));
      label(ctx, `${Number(r.solve_time_ms).toPrecision(3)} ms`, Math.min(canvas.width - 58, left + width + 4), y + 10);
    });
    return true;
  }

  return { renderSpyPlot, renderProgress, renderResiduals, renderVarDistribution, renderBenchmarkChart };
})();
window.NirnayaCharts = NirnayaCharts;
