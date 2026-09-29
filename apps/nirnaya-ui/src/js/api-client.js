// api-client.js — thin wrapper around the Nirnaya API contract
// (docs/API_CONTRACT.md). No solving happens here; this file only talks
// HTTP to whatever base URL the user configures. If a call fails, callers
// must surface that honestly rather than falling back to invented data.

class NirnayaApiClient {
  constructor(baseUrl) {
    this.baseUrl = (baseUrl || '').replace(/\/+$/, '');
  }

  setBaseUrl(url) {
    this.baseUrl = (url || '').replace(/\/+$/, '');
  }

  async _get(path) {
    const res = await fetch(this.baseUrl + path, { method: 'GET' });
    if (!res.ok) throw new Error(`GET ${path} -> HTTP ${res.status}`);
    return res.json();
  }

  async _post(path, body) {
    const res = await fetch(this.baseUrl + path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`POST ${path} -> HTTP ${res.status}`);
    return res.json();
  }

  async health() {
    return this._get('/api/health');
  }

  async systemInfo() {
    return this._get('/api/system/info');
  }

  async validateModel(model) {
    return this._post('/api/model/validate', { model });
  }

  async solve(model, backend, options) {
    return this._post('/api/solve', { model, backend, options: options || {} });
  }

  async benchmarks() {
    return this._get('/api/benchmarks');
  }

  async validationReport() {
    return this._get('/api/validation');
  }
}

window.NirnayaApiClient = NirnayaApiClient;
