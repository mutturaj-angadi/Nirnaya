// test_api_client.js — unit tests for src/js/api-client.js using a mocked
// `fetch`, run under plain Node (no test framework dependency required).
//
// Usage: node tests/unit/test_api_client.js

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const clientSrc = fs.readFileSync(path.join(__dirname, '../../src/js/api-client.js'), 'utf8');

function makeSandbox(fetchImpl) {
  const sandbox = { window: {}, fetch: fetchImpl, console };
  vm.createContext(sandbox);
  vm.runInContext(clientSrc, sandbox);
  return sandbox;
}

let passed = 0, failed = 0;
function assert(cond, msg) {
  if (cond) { passed++; console.log(`PASS ${msg}`); }
  else { failed++; console.log(`FAIL ${msg}`); }
}

async function run() {
  // --- test: successful solve call shapes the request correctly ---
  {
    let capturedUrl, capturedBody;
    const fetchImpl = async (url, opts) => {
      capturedUrl = url;
      capturedBody = JSON.parse(opts.body);
      return { ok: true, json: async () => ({ status: 'optimal', objective: 42 }) };
    };
    const sandbox = makeSandbox(fetchImpl);
    const client = new sandbox.window.NirnayaApiClient('http://localhost:8000');
    const model = { name: 'm', variables: [], constraints: [], objective: { coefficients: {} } };
    const result = await client.solve(model, 'cpu', { max_iterations: 10 });

    assert(capturedUrl === 'http://localhost:8000/api/solve', 'solve() posts to /api/solve');
    assert(capturedBody.backend === 'cpu', 'solve() includes backend in body');
    assert(capturedBody.options.max_iterations === 10, 'solve() includes options in body');
    assert(result.objective === 42, 'solve() returns parsed JSON response');
  }

  // --- test: non-ok response throws ---
  {
    const fetchImpl = async () => ({ ok: false, status: 500 });
    const sandbox = makeSandbox(fetchImpl);
    const client = new sandbox.window.NirnayaApiClient('http://x');
    let threw = false;
    try { await client.health(); } catch (e) { threw = true; }
    assert(threw, 'non-ok HTTP response throws an error rather than returning fake data');
  }

  // --- test: base URL trailing slash is normalized ---
  {
    let capturedUrl;
    const fetchImpl = async (url) => { capturedUrl = url; return { ok: true, json: async () => ({}) }; };
    const sandbox = makeSandbox(fetchImpl);
    const client = new sandbox.window.NirnayaApiClient('http://localhost:8000///');
    await client.systemInfo();
    assert(capturedUrl === 'http://localhost:8000/api/system/info', 'trailing slashes are stripped from base URL');
  }

  console.log(`\n${passed} passed, ${failed} failed`);
  process.exit(failed > 0 ? 1 : 0);
}

run();
