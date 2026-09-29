// app.js — application state and view wiring.

(function () {
  const apiInput = document.getElementById('api-base-url');
  const configuredApiUrl = new URLSearchParams(window.location.search).get('api');
  apiInput.value = configuredApiUrl || `${window.location.protocol}//${window.location.hostname}:8000`;
  const state = {
    api: new NirnayaApiClient(apiInput.value),
    connected: false,
    model: null,
    modelName: null,
    lastResult: null,
    lastBackend: 'cpu',
    sysInfo: null,
    benchmarkRuns: [],
    benchmarkMeta: null,
    scenarioRuns: [],
    validationReport: null,
  };

  // ---------- small utilities ----------
  function $(id) { return document.getElementById(id); }
  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[char]);
  }
  function fmtNum(v, digits) {
    if (v === null || v === undefined) return null;
    if (typeof v !== 'number') return escapeHtml(v);
    if (Math.abs(v) !== 0 && (Math.abs(v) < 1e-4 || Math.abs(v) >= 1e6)) return v.toExponential(digits === undefined ? 3 : digits);
    return v.toLocaleString(undefined, { maximumFractionDigits: digits === undefined ? 4 : digits });
  }
  function unavailable() { return '<span class="unavailable">unavailable</span>'; }
  function safeFilename(value) {
    const cleaned = String(value || 'result').replace(/[^a-zA-Z0-9._-]/g, '_').slice(0, 80);
    return cleaned || 'result';
  }
  function humanizeIdentifier(value) {
    return String(value).split(/[_\s-]+/).filter(Boolean)
      .map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
  }
  function csvCell(value) {
    let text = String(value);
    if (/^[=+\-@\t\r]/.test(text)) text = `'${text}`;
    return `"${text.replace(/"/g, '""')}"`;
  }

  function log(panelId, message, kind) {
    const panel = $(panelId);
    const line = document.createElement('div');
    line.className = 'log-line' + (kind ? ' ' + kind : '');
    const t = new Date().toLocaleTimeString();
    const stamp = document.createElement('span');
    stamp.className = 't';
    stamp.textContent = `[${t}] `;
    line.appendChild(stamp);
    line.appendChild(document.createTextNode(message));
    panel.appendChild(line);
    panel.scrollTop = panel.scrollHeight;
  }

  // ---------- navigation ----------
  document.querySelectorAll('.nav-item').forEach((item) => {
    item.addEventListener('click', () => {
      document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
      document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
      item.classList.add('active');
      $('view-' + item.dataset.view).classList.add('active');
    });
  });

  // ---------- connection ----------
  function setConnStatus(ok, text) {
    const pill = $('conn-status');
    pill.classList.toggle('ok', !!ok);
    pill.classList.toggle('bad', ok === false);
    $('conn-status-text').textContent = text;
    $('ov-conn').textContent = text;
    $('ov-conn').className = 'metric-value ' + (ok ? 'green' : 'dim');
    $('api-offline-banner').hidden = !!ok;
  }

  function clearLiveMetrics() {
    state.lastResult = null;
    $('ov-status').textContent = 'unavailable';
    $('ov-status').className = 'metric-value dim';
    $('ov-backend').textContent = 'unavailable';
    $('ov-objective').textContent = 'unavailable';
    $('ov-feasibility').textContent = 'unavailable';
    $('ov-feasibility').className = 'metric-value dim';
    $('ov-violation').textContent = 'max violation unavailable';
    $('ov-presolve').textContent = 'unavailable';
    $('ov-solve-time').textContent = 'unavailable';
    $('sol-status').textContent = 'unavailable';
    $('sol-objective').innerHTML = unavailable();
    $('sol-iterations').innerHTML = unavailable();
    $('sol-time').innerHTML = unavailable();
    $('sol-variables-table').innerHTML = '<div class="empty-state">Live solution unavailable until the Nirnaya API is connected and a solve completes.</div>';
    $('sol-explanation').innerHTML = '<div class="help-text">Live objective analysis unavailable.</div>';
    ['resid-canvas', 'var-dist-canvas', 'progress-canvas'].forEach(id => {
      const canvas = $(id);
      const context = canvas && canvas.getContext('2d');
      if (context) context.clearRect(0, 0, canvas.width, canvas.height);
    });
    $('btn-export-json').disabled = true;
    $('btn-export-csv').disabled = true;
    $('diag-max-resid').innerHTML = unavailable();
    $('diag-gap').innerHTML = unavailable();
    $('diag-presolve').innerHTML = unavailable();
    $('diag-reduction').textContent = 'unavailable';
    $('diag-dimensions').innerHTML = '<div class="empty-state">Live presolve data unavailable.</div>';
    $('diag-table').innerHTML = '<div class="empty-state">Live feasibility data unavailable.</div>';
    $('diag-stats').innerHTML = '';
    state.scenarioRuns = [];
    renderScenarioResults();
  }

  async function connect(options) {
    const quiet = !!(options && options.quiet);
    const url = $('api-base-url').value.trim();
    if (url !== state.api.baseUrl) {
      state.connected = false;
      state.sysInfo = null;
      clearLiveMetrics();
    }
    state.api.setBaseUrl(url);
    setConnStatus(null, 'checking API…');
    try {
      const health = await state.api.health();
      const checks = health.checks || {};
      if (health.status !== 'ok' || Object.values(checks).some(value => value !== true)) {
        throw new Error(`API health is ${health.status || 'unknown'}; one or more solver packages are unavailable`);
      }
      state.connected = true;
      setConnStatus(true, 'CONNECTED');
      if (!state.sysInfo) {
        try {
          state.sysInfo = await state.api.systemInfo();
          renderGpuView();
        } catch (infoError) {
          if (!quiet) log('event-log', `API is healthy; optional hardware details unavailable: ${infoError.message}`, 'warn');
        }
      }
      if (!quiet) log('event-log', `Connected to Nirnaya API at ${url}`, 'ok');
    } catch (e) {
      state.connected = false;
      state.sysInfo = null;
      setConnStatus(false, 'API OFFLINE');
      clearLiveMetrics();
      $('gpu-cpu-info').innerHTML = '<div class="unavailable">unavailable while the API is offline</div>';
      $('gpu-gpu-info').innerHTML = '<div class="unavailable">unavailable while the API is offline</div>';
      $('gpu-versions').innerHTML = '<div class="unavailable">unavailable while the API is offline</div>';
      if (!quiet) log('event-log', `Nirnaya API is offline at ${url}: ${e.message}`, 'err');
    }
  }
  $('btn-connect').addEventListener('click', connect);

  // ---------- Model view ----------
  const DEMO = window.NIRNAYA_DEMO_DATA || { models: {} };
  const demoOrder = ['refinery_blending', 'crude_allocation', 'production_planning', 'transportation', 'resource_allocation', 'sparse_large_lp'];

  function buildModelPicker() {
    const wrap = $('demo-model-picker');
    wrap.innerHTML = '';
    demoOrder.forEach((key) => {
      const m = DEMO.models[key];
      if (!m) return;
      const chip = document.createElement('div');
      chip.className = 'chip';
      chip.textContent = m.title || key;
      chip.dataset.key = key;
      chip.addEventListener('click', () => loadDemoModel(key));
      wrap.appendChild(chip);
    });
    const flagship = document.createElement('div');
    flagship.className = 'chip flagship-chip';
    flagship.textContent = 'Refinery decision scenario · synthetic';
    flagship.addEventListener('click', loadFlagshipModel);
    wrap.appendChild(flagship);
  }

  async function loadFlagshipModel() {
    try {
      const response = await fetch('data/refinery_decision_demo.json');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const model = await response.json();
      state.model = model;
      state.modelName = model.name;
      $('model-editor').value = JSON.stringify(model, null, 2);
      document.querySelectorAll('#demo-model-picker .chip').forEach(c => c.classList.toggle('active', c.classList.contains('flagship-chip')));
      renderModelSummary(model);
      NirnayaCharts.renderSpyPlot($('spy-canvas'), model);
      updateOverviewFromModel(model);
      log('event-log', 'Loaded the documented MRPL-inspired synthetic refinery scenario.', 'ok');
    } catch (e) {
      log('event-log', `Could not load the bundled refinery scenario: ${e.message}`, 'err');
    }
  }

  function loadDemoModel(key) {
    const model = DEMO.models[key];
    if (!model) return;
    state.model = model;
    state.modelName = key;
    $('model-editor').value = JSON.stringify(model, null, 2);
    document.querySelectorAll('#demo-model-picker .chip').forEach(c => c.classList.toggle('active', c.dataset.key === key));
    renderModelSummary(model);
    NirnayaCharts.renderSpyPlot($('spy-canvas'), model);
    updateOverviewFromModel(model);
    log('event-log', `Loaded demo model ${model.title || key} (${model.variables.length} vars, ${model.constraints.length} constraints).`);
  }

  $('model-editor').addEventListener('change', () => {
    try {
      const model = JSON.parse($('model-editor').value);
      state.model = model;
      state.modelName = model.name || 'custom';
      renderModelSummary(model);
      NirnayaCharts.renderSpyPlot($('spy-canvas'), model);
      updateOverviewFromModel(model);
      document.querySelectorAll('#demo-model-picker .chip').forEach(c => c.classList.remove('active'));
      log('event-log', 'Model updated from editor.');
    } catch (e) {
      log('event-log', `Model JSON is invalid: ${e.message}`, 'err');
    }
  });

  $('btn-upload-model').addEventListener('click', () => $('model-file-input').click());
  $('model-file-input').addEventListener('change', (ev) => {
    const file = ev.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const model = JSON.parse(reader.result);
        state.model = model;
        state.modelName = model.name || file.name;
        $('model-editor').value = JSON.stringify(model, null, 2);
        renderModelSummary(model);
        NirnayaCharts.renderSpyPlot($('spy-canvas'), model);
        updateOverviewFromModel(model);
        document.querySelectorAll('#demo-model-picker .chip').forEach(c => c.classList.remove('active'));
        log('event-log', `Uploaded model file ${file.name}.`, 'ok');
      } catch (e) {
        log('event-log', `Failed to parse uploaded file: ${e.message}`, 'err');
      }
    };
    reader.readAsText(file);
  });

  function renderModelSummary(model) {
    const nnz = model.constraints.reduce((s, c) => s + Object.keys(c.coefficients || {}).length, 0);
    const boundedVars = model.variables.filter(v => v.upper !== null && v.upper !== undefined).length;
    const html = `
      <div class="kv"><span class="k">TITLE</span><span class="v">${escapeHtml(model.title || model.name || 'untitled')}</span></div>
      <div class="kv"><span class="k">SENSE</span><span class="v">${escapeHtml(model.sense)}</span></div>
      <div class="kv"><span class="k">VARIABLES</span><span class="v">${model.variables.length}</span></div>
      <div class="kv"><span class="k">CONSTRAINTS</span><span class="v">${model.constraints.length}</span></div>
      <div class="kv"><span class="k">NONZEROS</span><span class="v">${nnz}</span></div>
      <div class="kv"><span class="k">DENSITY</span><span class="v">${(100 * nnz / (model.variables.length * model.constraints.length)).toFixed(3)}%</span></div>
      <div class="kv"><span class="k">BOUNDED VARS</span><span class="v">${boundedVars} / ${model.variables.length}</span></div>
      <div class="kv"><span class="k">DOMAIN</span><span class="v">${escapeHtml(model.domain || '')}</span></div>
      <div class="kv"><span class="k">DATA SOURCE</span><span class="v">${escapeHtml((model.metadata || {}).source || 'user-supplied; not stated')}</span></div>
      <div class="kv"><span class="k">DATA LICENSE</span><span class="v">${escapeHtml((model.metadata || {}).license || 'not stated')}</span></div>
      ${model.description ? `<div style="margin-top:10px;color:var(--text-dim);font-size:12.5px;">${escapeHtml(model.description)}</div>` : ''}
    `;
    $('model-summary').innerHTML = html;
  }

  function updateOverviewFromModel(model) {
    $('ov-model').textContent = model.title || model.name || 'custom';
    $('ov-model').className = 'metric-value blue';
    const nnz = model.constraints.reduce((s, c) => s + Object.keys(c.coefficients || {}).length, 0);
    $('ov-vars').textContent = model.variables.length;
    $('ov-cons').textContent = model.constraints.length;
    $('ov-nnz').textContent = nnz;
  }

  // ---------- Solver view ----------
  document.querySelectorAll('#backend-toggle button').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#backend-toggle button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.lastBackend = btn.dataset.backend;
      $('ov-backend').textContent = state.lastBackend;
      $('backend-help').textContent = state.lastBackend === 'gpu'
        ? 'GPU requests are rejected for LP solving in this release. GPU device discovery and numerical primitives do not mean the simplex engine ran on a GPU.'
        : 'The revised-simplex LP solver runs on the API host CPU. Reference solvers are validation-only.';
    });
  });

  $('btn-run-solve').addEventListener('click', async () => {
    if (!state.model) {
      log('solver-log', 'No model loaded. Go to the Model tab and pick or upload one.', 'err');
      return;
    }
    if (!state.connected) {
      log('solver-log', 'Not connected to an API. Set the API Base URL and click Connect.', 'err');
      return;
    }
    const options = {
      iteration_limit: parseInt($('opt-max-iter').value, 10) || 10000,
      feasibility_tol: parseFloat($('opt-tolerance').value) || 1e-7,
      optimality_tol: parseFloat($('opt-tolerance').value) || 1e-7,
      time_limit: parseInt($('opt-time-limit').value, 10) || 30,
      presolve: $('opt-presolve').checked,
      record_iteration_history: true,
      verbose: true,
    };
    const btn = $('btn-run-solve');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Solving…';
    $('ov-status').textContent = 'running';
    $('ov-status').className = 'metric-value amber';
    log('solver-log', `Submitting solve · backend=${state.lastBackend} · vars=${state.model.variables.length} · constraints=${state.model.constraints.length}`);
    try {
      const t0 = performance.now();
      const result = await state.api.solve(state.model, state.lastBackend, options);
      const wall = (performance.now() - t0).toFixed(1);
      state.lastResult = result;
      log('solver-log', `Response received in ${wall} ms (wall clock, network+solve).`);
      if (result.status === 'error') {
        log('solver-log', `Solver returned an error: ${result.error || 'unknown'}${result.gpu_unavailable_reason ? ' — ' + result.gpu_unavailable_reason : ''}`, 'err');
      } else {
        log('solver-log', `status=${escapeHtml(result.status)} objective=${fmtNum(result.objective)} iterations=${result.iterations}`, result.status === 'optimal' ? 'ok' : 'warn');
      }
      renderSolution(result);
      renderDiagnostics(result);
      updateOverviewFromResult(result);
    } catch (e) {
      log('solver-log', `Request failed: ${e.message}`, 'err');
      clearLiveMetrics();
      await connect({quiet: true});
    } finally {
      btn.disabled = false;
      btn.textContent = 'Run optimization';
    }
  });

  function updateOverviewFromResult(result) {
    $('ov-status').textContent = result.status || 'unknown';
    $('ov-status').className = 'metric-value ' + (result.status === 'optimal' ? 'green' : result.status === 'error' ? 'dim' : 'amber');
    const observability = result.observability || {};
    $('ov-backend').textContent = observability.backend || result.backend || 'unavailable';
    $('ov-objective').textContent = result.objective == null ? 'unavailable' : fmtNum(result.objective);
    const residual = (result.numerical || {}).max_residual;
    const feasible = result.status === 'optimal' && Number.isFinite(residual) && residual <= 1e-7;
    $('ov-feasibility').textContent = feasible ? 'VERIFIED' : 'unavailable';
    $('ov-feasibility').className = `metric-value ${feasible ? 'green' : 'dim'}`;
    const maxViolation = (result.numerical || {}).max_residual;
    $('ov-violation').textContent = maxViolation == null ? 'max violation unavailable' : `max violation ${fmtNum(maxViolation, 3)}`;
    $('ov-solve-time').textContent = result.solve_time_ms == null ? 'unavailable' : `${fmtNum(result.solve_time_ms, 3)} ms`;
    const stats = ((result.presolve || {}).statistics) || {};
    const rowBefore = stats.original_num_constraints, rowAfter = stats.reduced_num_constraints;
    const colBefore = stats.original_num_variables, colAfter = stats.reduced_num_variables;
    if ([rowBefore, rowAfter, colBefore, colAfter].every(Number.isFinite)) {
      const before = rowBefore + colBefore, after = rowAfter + colAfter;
      $('ov-presolve').textContent = before ? `${fmtNum(100 * (before - after) / before, 2)}%` : '0%';
    } else $('ov-presolve').textContent = 'unavailable';
  }

  // ---------- Solution view ----------
  function renderSolution(result) {
    $('sol-status').innerHTML = `<span class="badge ${escapeHtml(result.status)}">${escapeHtml(result.status)}</span>`;
    $('sol-status').className = 'metric-value dim';
    $('sol-objective').innerHTML = result.objective != null ? fmtNum(result.objective) : unavailable();
    $('sol-iterations').innerHTML = result.iterations != null ? result.iterations : unavailable();
    $('sol-time').innerHTML = result.solve_time_ms != null ? `${fmtNum(result.solve_time_ms, 3)}<span class="metric-unit">ms</span>` : unavailable();

    if (result.variables) {
      const variableDefinitions = new Map((state.model && state.model.variables || []).map(variable => [variable.name, variable]));
      const rows = Object.entries(result.variables).map(([k, v]) => {
        const definition = variableDefinitions.get(k) || {};
        const lower = definition.lower ?? definition.lb;
        const upper = definition.upper ?? definition.ub;
        const atLower = lower != null && Math.abs(v - Number(lower)) <= 1e-7 * Math.max(1, Math.abs(Number(lower)));
        const atUpper = upper != null && Math.abs(v - Number(upper)) <= 1e-7 * Math.max(1, Math.abs(Number(upper)));
        const bounds = `${lower == null ? '−∞' : fmtNum(Number(lower))} to ${upper == null ? '+∞' : fmtNum(Number(upper))}`;
        const active = atLower ? 'at lower bound' : atUpper ? 'at upper bound' : '';
        return `<tr><td>${escapeHtml(humanizeIdentifier(k))}<div class="help-text">${escapeHtml(k)}</div></td><td>${fmtNum(v)}</td><td>${escapeHtml(bounds)}</td><td>${escapeHtml(definition.unit || 'unavailable')}</td><td>${active || 'within bounds'}</td></tr>`;
      }).join('');
      $('sol-variables-table').innerHTML = `<table><thead><tr><th>DECISION VARIABLE</th><th>VALUE</th><th>BOUNDS</th><th>UNIT</th><th>BOUND STATUS</th></tr></thead><tbody>${rows}</tbody></table>`;
      NirnayaCharts.renderVarDistribution($('var-dist-canvas'), result.variables);
      $('btn-export-json').disabled = false;
      $('btn-export-csv').disabled = false;
    } else {
      $('sol-variables-table').innerHTML = `<div class="empty-state"><div class="glyph">✓</div>No variable values — status is "${escapeHtml(result.status)}".</div>`;
      $('btn-export-json').disabled = true;
      $('btn-export-csv').disabled = true;
    }

    const hasHistory = NirnayaCharts.renderProgress($('progress-canvas'), result.iteration_history);
    $('progress-help').innerHTML = hasHistory
      ? ''
      : 'Iteration-by-iteration objective history is <span class="unavailable">unavailable</span> from the connected solver for this run.';

    const objective = state.model && (state.model.objective || {});
    const coefficients = objective.coefficients || {};
    const contributions = Object.entries(coefficients).map(([name, coefficient]) => ({
      name, coefficient: Number(coefficient), value: Number((result.variables || {})[name] || 0),
      contribution: Number(coefficient) * Number((result.variables || {})[name] || 0),
    })).filter(item => item.value !== 0).sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).slice(0, 8);
    const binding = (state.model && state.model.constraints || []).filter(row => {
      const activity = Object.entries(row.coefficients || {}).reduce((sum, [name, a]) => sum + Number(a) * Number((result.variables || {})[name] || 0), 0);
      const gap = row.sense === '<=' ? Number(row.rhs) - activity : row.sense === '>=' ? activity - Number(row.rhs) : Math.abs(activity - Number(row.rhs));
      return Math.abs(gap) <= 1e-7 * Math.max(1, Math.abs(Number(row.rhs)));
    }).map(row => row.name);
    const contributionRows = contributions.map(item => `<tr><td>${escapeHtml(humanizeIdentifier(item.name))}</td><td>${fmtNum(item.value, 5)}</td><td>${fmtNum(item.coefficient, 5)}</td><td>${fmtNum(item.contribution, 5)}</td></tr>`).join('');
    $('sol-explanation').innerHTML = `<div class="panel-title">LARGEST DIRECT OBJECTIVE CONTRIBUTIONS</div>
      <div class="help-text">These terms show how the supplied objective evaluates this primal point; they do not imply a causal explanation or sensitivity certificate.</div>
      ${contributionRows ? `<div class="table-scroll"><table><thead><tr><th>VARIABLE</th><th>VALUE</th><th>OBJECTIVE COEFFICIENT</th><th>CONTRIBUTION</th></tr></thead><tbody>${contributionRows}</tbody></table></div>` : `<div class="help-text">Objective contribution values unavailable for this solve.</div>`}
      <div class="kv section-gap"><span class="k">BINDING MODEL CONSTRAINTS (ABSOLUTE TOLERANCE 1e-7 SCALED BY RHS)</span><span class="v">${binding.length ? escapeHtml(binding.join(', ')) : 'none detected'}</span></div>
      <div class="help-text">Validated shadow prices and marginal sensitivities are not produced by this solver and are not shown.</div>`;
  }

  // ---------- Diagnostics view ----------
  function renderDiagnostics(result) {
    const numerical = result.numerical || {};
    $('diag-max-resid').innerHTML = numerical.max_residual != null ? fmtNum(numerical.max_residual, 3) : unavailable();
    $('diag-gap').innerHTML = numerical.duality_gap != null ? fmtNum(numerical.duality_gap, 3) : unavailable();
    $('diag-presolve').innerHTML = result.presolve_time_ms != null ? `${fmtNum(result.presolve_time_ms, 3)}<span class="metric-unit">ms</span>` : unavailable();
    const presolveStats = (result.presolve || {}).statistics || {};
    if (presolveStats.original_num_variables != null) {
      const dimensions = [
        ['Rows', presolveStats.original_num_constraints, presolveStats.reduced_num_constraints],
        ['Columns', presolveStats.original_num_variables, presolveStats.reduced_num_variables],
        ['Nonzeros', presolveStats.original_num_nonzeros, presolveStats.reduced_num_nonzeros],
      ];
      const reduction = (before, after) => Number.isFinite(before) && Number.isFinite(after) && before > 0
        ? `${fmtNum(100 * (before - after) / before, 2)}%` : 'unavailable';
      $('diag-dimensions').innerHTML = `<table><thead><tr><th>DIMENSION</th><th>ORIGINAL</th><th>REDUCED</th><th>REMOVED</th><th>REDUCTION</th></tr></thead><tbody>${dimensions.map(([name, before, after]) => `<tr><td>${name}</td><td>${before ?? 'unavailable'}</td><td>${after ?? 'unavailable'}</td><td>${Number.isFinite(before) && Number.isFinite(after) ? before - after : 'unavailable'}</td><td>${reduction(before, after)}</td></tr>`).join('')}</tbody></table>`;
      const transformations = [
        ['Fixed variables', presolveStats.variables_fixed],
        ['Redundant constraints removed', presolveStats.rows_removed_redundant],
        ['Bound tightenings', presolveStats.bounds_tightened],
        ['Infeasibility detected', presolveStats.infeasible == null ? null : (presolveStats.infeasible ? 'yes' : 'not triggered')],
      ];
      $('diag-dimensions').innerHTML += `<div class="panel-title section-gap">RECORDED PRESOLVE TRANSFORMATIONS</div><table><thead><tr><th>TRANSFORMATION</th><th>COUNT / STATUS</th></tr></thead><tbody>${transformations.map(([name, value]) => `<tr><td>${name}</td><td>${value ?? 'unavailable'}</td></tr>`).join('')}</tbody></table>`;
      $('diag-reduction').textContent = `${presolveStats.original_num_constraints}→${presolveStats.reduced_num_constraints} rows`;
    } else {
      $('diag-reduction').textContent = 'unavailable';
      $('diag-dimensions').innerHTML = '<div class="empty-state">Presolve dimensions unavailable for this result.</div>';
    }

    const hasResid = NirnayaCharts.renderResiduals($('resid-canvas'), result.constraint_residuals);
    if (!hasResid) {
      $('resid-canvas').getContext('2d').clearRect(0, 0, $('resid-canvas').width, $('resid-canvas').height);
    }

    if (result.constraint_residuals) {
      const modelRows = state.model && state.model.constraints || [];
      const values = result.variables || {};
      const rows = modelRows.map((row) => {
        const activity = Object.entries(row.coefficients || {}).reduce((sum, [name, a]) => sum + Number(a) * Number(values[name] || 0), 0);
        const sense = row.sense === '==' ? '=' : row.sense;
        const slack = sense === '<=' ? Number(row.rhs) - activity : sense === '>=' ? activity - Number(row.rhs) : -Math.abs(activity - Number(row.rhs));
        const scale = Math.max(1, Math.abs(Number(row.rhs)));
        const binding = Math.abs(slack) <= 1e-7 * scale;
        const violation = slack < -1e-7 * scale;
        const cls = violation ? 'bad' : binding ? 'warn' : '';
        const stateText = violation ? 'violated' : binding ? 'binding' : 'slack';
        const utilization = sense !== '=' && Number(row.rhs) > 0
          ? `${fmtNum(100 * activity / Number(row.rhs), 2)}%` : 'not applicable';
        return `<tr><td>${escapeHtml(row.name)}</td><td>${escapeHtml(sense)}</td><td>${fmtNum(activity, 5)}</td><td>${fmtNum(row.rhs, 5)}</td><td>${fmtNum(slack, 5)}</td><td>${utilization}</td><td><span class="${cls}">${stateText}</span></td></tr>`;
      }).join('');
      $('diag-table').innerHTML = rows ? `<table><thead><tr><th>CONSTRAINT</th><th>SENSE</th><th>ACTIVITY</th><th>LIMIT</th><th>SLACK</th><th>UTILIZATION</th><th>INTERPRETATION</th></tr></thead><tbody>${rows}</tbody></table>` : `<div class="empty-state">Constraint activities unavailable for this model.</div>`;
    } else {
      $('diag-table').innerHTML = `<div class="empty-state"><div class="glyph">△</div>Constraint residuals are <span class="unavailable">unavailable</span> for status "${escapeHtml(result.status)}".</div>`;
    }

    const stats = [
      ['Status', result.status],
      ['Backend', result.backend],
      ['Device', result.device || 'unavailable'],
      ['Iterations', result.iterations != null ? result.iterations : 'unavailable'],
      ['Solve time', result.solve_time_ms != null ? fmtNum(result.solve_time_ms, 3) + ' ms' : 'unavailable'],
      ['Presolve time', result.presolve_time_ms != null ? fmtNum(result.presolve_time_ms, 3) + ' ms' : 'unavailable'],
      ['Condition estimate', numerical.condition_estimate != null ? fmtNum(numerical.condition_estimate, 3) : 'unavailable'],
      ['Basis factorization', (numerical.extra || {}).factorization_seconds != null ? fmtNum((numerical.extra).factorization_seconds * 1000, 2) + ' ms' : 'unavailable'],
      ['Pricing', (numerical.extra || {}).pricing_seconds != null ? fmtNum((numerical.extra).pricing_seconds * 1000, 2) + ' ms' : 'unavailable'],
      ['Ratio test', (numerical.extra || {}).ratio_test_seconds != null ? fmtNum((numerical.extra).ratio_test_seconds * 1000, 2) + ' ms' : 'unavailable'],
      ['Dual prices', 'unavailable · solver does not publish a validated sensitivity certificate'],
    ];
    $('diag-stats').innerHTML = stats.map(([k, v]) => `<div class="kv"><span class="k">${escapeHtml(k.toUpperCase())}</span><span class="v">${escapeHtml(v)}</span></div>`).join('');
  }

  // ---------- GPU view ----------
  function renderGpuView() {
    if (!state.sysInfo) return;
    const cpu = state.sysInfo.cpu || {};
    const gpu = state.sysInfo.gpu || {};
    $('gpu-cpu-info').innerHTML = `
      <div class="kv"><span class="k">MODEL</span><span class="v">${escapeHtml(cpu.model || 'unavailable')}</span></div>
      <div class="kv"><span class="k">PHYSICAL CORES</span><span class="v">${cpu.physical_cores ?? 'unavailable'}</span></div>
      <div class="kv"><span class="k">LOGICAL CORES</span><span class="v">${cpu.logical_cores ?? 'unavailable'}</span></div>
    `;
    $('gpu-gpu-info').innerHTML = gpu.available
      ? `<div class="kv"><span class="k">STATUS</span><span class="v"><span class="badge optimal">available</span></span></div>
         <div class="kv"><span class="k">DEVICE</span><span class="v">${escapeHtml(gpu.device_name || 'unavailable')}</span></div>`
      : `<div class="kv"><span class="k">STATUS</span><span class="v"><span class="badge unavailable">unavailable</span></span></div>
         <div class="kv"><span class="k">REASON</span><span class="v">${escapeHtml(gpu.reason || 'not reported')}</span></div>`;
    const versions = state.sysInfo.backend_versions || {};
    $('gpu-versions').innerHTML = Object.entries(versions).map(([k, v]) => `<div class="kv"><span class="k">${escapeHtml(k.toUpperCase())}</span><span class="v">${escapeHtml(v)}</span></div>`).join('') || 'unavailable';
  }
  $('btn-refresh-sysinfo').addEventListener('click', async () => {
    if (!state.connected) { log('event-log', 'Connect to the API first.', 'err'); return; }
    state.sysInfo = await state.api.systemInfo();
    renderGpuView();
    log('event-log', 'Refreshed system info.');
  });

  // ---------- Real refinery scenario analysis ----------
  let refineryScenarios = [];
  async function loadScenarioDefinitions() {
    if (refineryScenarios.length) return refineryScenarios;
    const response = await fetch('data/scenarios.json');
    if (!response.ok) throw new Error(`Scenario definitions returned HTTP ${response.status}`);
    const payload = await response.json();
    refineryScenarios = payload.scenarios || [];
    $('scenario-cards').innerHTML = refineryScenarios.map(s => `<div class="panel scenario-card"><div class="panel-title">${escapeHtml(s.label)}</div><p>${escapeHtml(s.description)}</p>${(s.transformations || []).map(t => `<div class="scenario-transform"><span class="badge">${escapeHtml(t.classification)}</span> ${escapeHtml(t.reason)}</div>`).join('') || '<div class="help-text">No assumptions changed.</div>'}</div>`).join('');
    return refineryScenarios;
  }

  function applyScenario(base, scenario) {
    const model = JSON.parse(JSON.stringify(base));
    const changes = [];
    for (const transform of scenario.transformations || []) {
      let matched = 0;
      if (transform.type === 'constraint_rhs_multiplier') {
        model.constraints.forEach(row => {
          if (row.name.includes(transform.match)) {
            const old = row.rhs; row.rhs *= transform.factor; matched++;
            changes.push({scenario: scenario.label, target: row.name, operation: `rhs × ${transform.factor}`, before: old, after: row.rhs, reason: transform.reason});
          }
        });
      } else if (transform.type === 'variable_upper_multiplier') {
        model.variables.forEach(v => {
          if (v.name.startsWith(transform.match) && v.upper !== null && v.upper !== undefined) {
            const old = v.upper; v.upper *= transform.factor; matched++;
            changes.push({scenario: scenario.label, target: v.name, operation: `upper bound × ${transform.factor}`, before: old, after: v.upper, reason: transform.reason});
          }
        });
      } else if (transform.type === 'objective_coefficient_delta') {
        Object.keys(model.objective.coefficients).forEach(name => {
          if (name.startsWith(transform.match)) {
            const old = model.objective.coefficients[name]; model.objective.coefficients[name] += transform.delta; matched++;
            changes.push({scenario: scenario.label, target: name, operation: `objective coefficient ${transform.delta > 0 ? '+' : ''}${transform.delta}`, before: old, after: model.objective.coefficients[name], reason: transform.reason});
          }
        });
      } else throw new Error(`Unsupported scenario operation: ${transform.type}`);
      if (!matched) throw new Error(`Scenario rule matched no model entries: ${transform.match}`);
    }
    model.name = `${base.name}__${scenario.id}`;
    return {model, changes};
  }

  function bindingConstraints(model, values) {
    return model.constraints.filter(row => {
      const activity = Object.entries(row.coefficients || {}).reduce((sum, [name, a]) => sum + Number(a) * Number(values[name] || 0), 0);
      const gap = row.sense === '<=' ? Number(row.rhs) - activity : row.sense === '>=' ? activity - Number(row.rhs) : Math.abs(activity - Number(row.rhs));
      return Math.abs(gap) <= 1e-7 * Math.max(1, Math.abs(Number(row.rhs)));
    }).map(row => row.name);
  }

  function renderScenarioResults() {
    const base = state.scenarioRuns.find(x => x.scenario_id === 'baseline');
    const rows = state.scenarioRuns.map(run => {
      const values = run.variables || {};
      const delta = base && Number.isFinite(run.objective) && Number.isFinite(base.objective)
        ? run.objective - base.objective : null;
      const decision = ['crude_light', 'crude_medium', 'crude_heavy', 'ship_domestic_regular', 'ship_domestic_premium']
        .map(name => `${name}: ${Number.isFinite(values[name]) ? fmtNum(values[name], 3) : 'n/a'}`).join('<br>');
      return `<tr><td>${escapeHtml(run.label)}</td><td><span class="badge ${run.status === 'optimal' ? 'optimal' : 'unavailable'}">${escapeHtml(run.status)}</span></td><td>${run.objective == null ? unavailableInline() : fmtNum(run.objective, 4)}</td><td>${delta == null ? unavailableInline() : fmtNum(delta, 4)}</td><td>${run.iterations ?? unavailableInline()}</td><td>${fmtNum(run.solve_time_ms, 3) ?? unavailableInline()}</td><td>${escapeHtml((run.binding || []).join(', ') || (run.status === 'optimal' ? 'none detected' : 'unavailable'))}</td><td>${decision}</td></tr>`;
    }).join('');
    $('scenario-results').innerHTML = rows ? `<table><thead><tr><th>SCENARIO</th><th>STATUS</th><th>OBJECTIVE</th><th>Δ VS BASELINE</th><th>ITERATIONS</th><th>SOLVE ms</th><th>BINDING LIMITS</th><th>KEY DECISIONS</th></tr></thead><tbody>${rows}</tbody></table>` : '<div class="empty-state">No scenarios have run.</div>';
    const changes = state.scenarioRuns.flatMap(run => run.audit || []);
    $('scenario-audit').innerHTML = changes.length ? `<table><thead><tr><th>SCENARIO</th><th>MODEL FIELD</th><th>OPERATION</th><th>BEFORE</th><th>AFTER</th><th>REASON</th></tr></thead><tbody>${changes.map(c => `<tr><td>${escapeHtml(c.scenario)}</td><td>${escapeHtml(c.target)}</td><td>${escapeHtml(c.operation)}</td><td>${fmtNum(c.before, 5)}</td><td>${fmtNum(c.after, 5)}</td><td>${escapeHtml(c.reason)}</td></tr>`).join('')}</tbody></table>` : '<div class="empty-state">No transformations applied.</div>';
  }

  $('btn-run-scenarios').addEventListener('click', async () => {
    const button = $('btn-run-scenarios');
    if (!state.connected) { $('scenario-progress').textContent = 'Connect to the production API first.'; return; }
    button.disabled = true; state.scenarioRuns = [];
    try {
      const response = await fetch('data/refinery_decision_demo.json');
      if (!response.ok) throw new Error(`Flagship model returned HTTP ${response.status}`);
      const base = await response.json();
      const definitions = await loadScenarioDefinitions();
      for (let i = 0; i < definitions.length; i++) {
        const scenario = definitions[i];
        $('scenario-progress').textContent = `Solving ${scenario.label} (${i + 1}/${definitions.length})…`;
        const {model, changes} = applyScenario(base, scenario);
        try {
          const result = await state.api.solve(model, 'cpu', {presolve: true, time_limit: 30, max_iterations: 20000, tolerance: 1e-8, verbose: true});
          state.scenarioRuns.push({scenario_id: scenario.id, label: scenario.label, status: result.status,
            objective: result.objective, iterations: result.iterations, solve_time_ms: result.solve_time_ms,
            variables: result.variables || {}, binding: result.status === 'optimal' ? bindingConstraints(model, result.variables || {}) : [], audit: changes});
        } catch (error) {
          state.scenarioRuns.push({scenario_id: scenario.id, label: scenario.label, status: 'request_error',
            audit: changes, error: error.message});
        }
        renderScenarioResults();
      }
      const counts = state.scenarioRuns.reduce((a, x) => (a[x.status] = (a[x.status] || 0) + 1, a), {});
      $('scenario-progress').textContent = `Completed ${state.scenarioRuns.length} scenarios. Outcomes: ${Object.entries(counts).map(([k,v]) => `${k} ${v}`).join(' · ')}.`;
    } catch (error) {
      $('scenario-progress').textContent = `Scenario run failed: ${error.message}`;
    } finally { button.disabled = false; }
  });

  // ---------- Recorded validation evidence ----------
  $('btn-load-validation').addEventListener('click', async () => {
    if (!state.connected) { $('validation-table').innerHTML = '<div class="empty-state">Connect to the API to read the latest validation report.</div>'; return; }
    try {
      const report = await state.api.validationReport(); state.validationReport = report;
      if (!report.available) {
        $('validation-summary').innerHTML = `<div class="panel">${escapeHtml(report.reason || 'Validation data unavailable')}</div>`;
        $('validation-table').innerHTML = '<div class="empty-state">No executed validation report is available.</div>'; return;
      }
      const summary = report.summary || {};
      $('validation-summary').innerHTML = [['TOTAL', summary.total], ['PASSED', summary.passed], ['FAILED', summary.failed], ['UNAVAILABLE', summary.unavailable]].map(([k,v]) => `<div class="metric"><div class="metric-label">${k}</div><div class="metric-value">${v ?? 'unavailable'}</div></div>`).join('');
      const rows = (report.runs || []).map(r => `<tr><td>${escapeHtml(r.case)}</td><td>${escapeHtml(r.backend)}</td><td>${escapeHtml(r.reference_status)}</td><td>${escapeHtml(r.nirnaya_status)}</td><td>${fmtNum(r.reference_objective, 5) ?? unavailableInline()}</td><td>${fmtNum(r.nirnaya_objective, 5) ?? unavailableInline()}</td><td>${fmtNum(r.objective_abs_difference, 4) ?? unavailableInline()}</td><td>${fmtNum(r.max_violation, 4) ?? unavailableInline()}</td><td>${fmtNum(r.primal_max_abs_difference_to_reference, 4) ?? unavailableInline()}</td><td>${r.iterations ?? unavailableInline()}</td><td><span class="badge ${r.passed === true ? 'optimal' : r.passed === false ? 'bad' : 'unavailable'}">${r.passed === true ? 'PASS' : r.passed === false ? 'FAIL' : 'UNAVAILABLE'}</span></td></tr>`).join('');
      $('validation-table').innerHTML = `<table><thead><tr><th>CASE</th><th>BACKEND</th><th>REFERENCE</th><th>NIRNAYA</th><th>REF OBJECTIVE</th><th>NIRNAYA OBJECTIVE</th><th>OBJECTIVE Δ</th><th>MAX VIOLATION</th><th>PRIMAL Δ</th><th>ITER</th><th>RESULT</th></tr></thead><tbody>${rows}</tbody></table>`;
      $('validation-table').insertAdjacentHTML('beforebegin', `<div class="help-text">Run ${escapeHtml(report.generated_at_utc || 'timestamp unavailable')} · reference: ${escapeHtml(report.reference_solver || 'unavailable')} · ${escapeHtml(report.reference_solver_role || '')}</div>`);
    } catch (error) { $('validation-table').innerHTML = `<div class="empty-state">Could not load validation report: ${escapeHtml(error.message)}</div>`; }
  });

  // ---------- Benchmarks view ----------
  $('btn-load-benchmarks').addEventListener('click', async () => {
    if (!state.connected) { log('event-log', 'Connect to the API first.', 'err'); return; }
    try {
      const data = await state.api.benchmarks();
      state.benchmarkRuns = data.runs || [];
      state.benchmarkMeta = data;
      renderBenchmarks();
      log('event-log', data.available
        ? `Loaded ${state.benchmarkRuns.length} measured benchmark run(s) from ${data.run_id}.`
        : `Benchmark report unavailable: ${data.reason || 'no recorded run'}`, data.available ? 'ok' : 'warn');
    } catch (e) {
      log('event-log', `Failed to load benchmarks: ${e.message}`, 'err');
    }
  });

  function renderBenchmarks() {
    const runs = state.benchmarkRuns;
    const summary = (state.benchmarkMeta || {}).summary || {};
    const countRows = [
      ['Total cases', summary.total], ['Optimal', summary.optimal],
      ['Infeasible', summary.infeasible], ['Unbounded', summary.unbounded],
      ['Time limit', summary.time_limit], ['Numerical errors', summary.numerical_error],
      ['Execution errors', summary.execution_error],
    ];
    $('benchmark-summary').innerHTML = countRows.map(([label, value]) => `<div class="metric"><div class="metric-label">${label.toUpperCase()}</div><div class="metric-value ${value == null ? 'dim' : ''}">${value ?? 'unavailable'}</div></div>`).join('');
    if (!runs || runs.length === 0) {
      $('bench-table').innerHTML = `<div class="empty-state"><div class="glyph">▬</div>${escapeHtml((state.benchmarkMeta || {}).reason || 'No benchmark data loaded. Run the recorded benchmark matrix, then click Load from API.')}</div>`;
      return;
    }
    NirnayaCharts.renderBenchmarkChart($('bench-canvas'), runs);
    const rows = runs.map(r => `
      <tr>
        <td>${escapeHtml(r.problem)}</td>
        <td>${r.num_variables ?? '—'}</td>
        <td>${r.num_constraints ?? '—'}</td>
        <td>${r.nonzeros ?? '—'}</td>
        <td>${escapeHtml(r.backend_requested)}</td>
        <td><span class="badge ${escapeHtml(r.status === 'optimal' ? 'optimal' : (r.status || 'unavailable'))}">${escapeHtml(r.status || 'unavailable')}</span></td>
        <td>${r.iterations ?? unavailableInline()}</td>
        <td>${r.solve_time_ms != null ? fmtNum(r.solve_time_ms, 3) : unavailableInline()}</td>
        <td>${r.max_residual != null ? fmtNum(r.max_residual, 3) : unavailableInline()}</td>
        <td>${escapeHtml(r.reference_status || 'unavailable')}</td>
        <td>${r.objective_abs_difference != null ? fmtNum(r.objective_abs_difference, 5) : unavailableInline()}</td>
      </tr>`).join('');
    $('bench-table').innerHTML = `<table><thead><tr>
      <th>PROBLEM</th><th>VARS</th><th>CONS</th><th>NNZ</th><th>BACKEND</th><th>STATUS</th><th>ITERS</th><th>TIME (ms)</th><th>MAX RESID</th><th>REFERENCE</th><th>OBJECTIVE Δ</th>
      </tr></thead><tbody>${rows}</tbody></table>`;
    const methodology = state.benchmarkMeta && state.benchmarkMeta.methodology;
    if (methodology && methodology.timing_comparison_warning) {
      $('benchmark-methodology').lastElementChild.textContent = `Recorded ${state.benchmarkMeta.generated_at_utc || 'timestamp unavailable'} · ${methodology.timing_comparison_warning}`;
    }
  }
  function unavailableInline() { return '<span class="unavailable">n/a</span>'; }

  // ---------- Export ----------
  function download(filename, content, mime) {
    const blob = new Blob([content], { type: mime });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
  }
  $('btn-export-json').addEventListener('click', () => {
    if (!state.lastResult) return;
    download(`nirnaya_solution_${safeFilename(state.modelName)}.json`, JSON.stringify(state.lastResult, null, 2), 'application/json');
  });
  $('btn-export-csv').addEventListener('click', () => {
    if (!state.lastResult || !state.lastResult.variables) return;
    const rows = ['variable,value', ...Object.entries(state.lastResult.variables).map(([k, v]) => `${csvCell(k)},${csvCell(v)}`)];
    download(`nirnaya_solution_${safeFilename(state.modelName)}.csv`, rows.join('\n'), 'text/csv');
  });

  // ---------- init ----------
  buildModelPicker();
  loadFlagshipModel();
  log('event-log', 'Nirnaya console loaded. Connecting to the integrated Nirnaya API.');
  connect();
  window.setInterval(() => connect({quiet: true}), 15000);
})();
