# Three-to-five-minute judge demo

## One-command local launch

```powershell
python scripts/launch_demo.py
```

This generates the deterministic flagship model, starts the production API on `127.0.0.1:8000`, serves the static dashboard on `127.0.0.1:8080`, and opens the UI. It uses no validation/reference solver and does not require a CDN for charts or fonts. Press Ctrl+C in the launch terminal to stop both servers. Use `--no-browser` in a headless room.

The launcher configures the API's CORS allowlist for that exact UI origin. If opening the static UI separately, set `NIRNAYA_CORS_ORIGINS` on the API to the exact dashboard origin.

## Script and on-screen path

**0:00 — Decision problem.** “Choose a crude slate, blend available streams into two gasoline grades, meet illustrative domestic/export minimums and linear quality limits, and maximize modeled contribution.” Point out that all parameters are documented synthetic assumptions. There is no MRPL/live-plant data.

**0:20 — Data and validation.** Open Model. Show the 15 decision variables, 17 constraints, nonzero count, source, and license. Mention the read-only preparation audit and no silent data deletion.

**0:40 — Optimize.** Confirm the connected API, select CPU, and run the flagship model. Explain two-phase revised simplex and the actual iteration count; GPU primitives are separate and simplex executes on CPU only.

**1:05 — Solution.** Show the live objective and named crude, blending, and market decisions with units, bounds, and binding limits. The contribution table is direct objective arithmetic; no shadow prices are invented.

**1:35 — Presolve and verification.** Show measured original/reduced rows, columns, nonzeros, fixed variables, removed rows, bound tightenings, and original-space feasibility. Point out postsolve recovery. Counts not reported by the solver remain unavailable.

**2:00 — Scenario comparison.** Run the six documented cases through the production API. The audit table shows which RHS, bound, or objective coefficient changed. Compare status, objective, and key decisions; failures remain visible.

**2:35 — Independent correctness evidence.** Open Validation. Show the run timestamp, SciPy/HiGHS and Nirnaya status/objective, feasibility, coordinate difference, and pass/fail. State: “The reference solver is validation-only.” Alternate feasible optimal solutions may have different variable vectors with essentially the same objective.

**3:05 — Architecture, security, and GPU.** Show package flow, model/input limits, original-space checks, and that GPU simplex is not implemented. GPU availability reflects actual device detection.

**3:35 — Close.** “Nirnaya is an inspectable continuous LP system. Its current evidence and limitations are recorded here; it does not claim to be a universal or fastest solver.”

## Offline fallback

The dashboard, bundled models, scenario runner, and canvas visualizations are local. Solving and scenario comparison require the local API process. Independent reference comparison requires SciPy, installed with the repository development dependencies. If the browser cannot launch, open `http://127.0.0.1:8080` manually. The solver does not fall back to SciPy.
