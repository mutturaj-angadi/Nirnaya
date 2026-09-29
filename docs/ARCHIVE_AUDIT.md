# Input archive audit

All five original ZIP files were extracted under a temporary inspection directory before integration. Each contains one top-level project directory:

| Archive | Files | Top-level directory | Integrated path |
|---|---:|---|---|
| `nirnaya-core.zip` | 42 | `nirnaya-core/` | `packages/nirnaya-core/` |
| `nirnaya-presolve.zip` | 32 | `nirnaya-presolve/` | `packages/nirnaya-presolve/` |
| `nirnaya-gpu.zip` | 23 | `nirnaya-gpu/` | `packages/nirnaya-gpu/` |
| `nirnaya-api.zip` | 32 | `nirnaya-api/` | `packages/nirnaya-api/` |
| `nirnaya-ui-tests.zip` | 47 | `nirnaya-ui-tests/` | `apps/nirnaya-ui/` |

Overlapping paths were reviewed before copying. Repeated README, package metadata, integration-contract and architecture files were kept within their package contexts; a single root README and root architecture docs now explain the integrated contract. The presolve and API `tests/conftest.py` files are package-specific. GPU and UI benchmark entry points share a basename but are different implementations for different scopes. Core and GPU supplied different package licenses, so their per-package license files remain in place.

The unchanged ZIPs are retained under `source-archives/`. Their SHA-256 values after relocation are:

| Archive | SHA-256 |
|---|---|
| `nirnaya-api.zip` | `5489876FCDD0F0F9D571A715BED47E08E9D517FA7C70B44271827FF9C2CF82C1` |
| `nirnaya-core.zip` | `044E1ABFA44FDFE12093A0D0A007504CBB71FDEB5E10FF2058C421E99CA788F7` |
| `nirnaya-gpu.zip` | `0181EA6ECE8EE1AFE1C80EA884A5B7AEF1CFFC6442AD9ECE50C985A1FEC671FB` |
| `nirnaya-presolve.zip` | `CB268D6A1C95F2EFAC02E68D66E4DBBF645F2BBF0F6A3DF0D0AE4C5ACC15CE74` |
| `nirnaya-ui-tests.zip` | `2AF02B8FAE309187D4850CE80D8E4F9DC0D5FA35E4DF9B91272E6850C96C0616` |

The archive's generated reference solutions and benchmark report are retained as source artifacts. They are not evidence of results from this integrated solver. SciPy's `linprog` call appears only under the UI validation reference solver; the production `nirnaya-solver` path does not import `scipy.optimize`.
