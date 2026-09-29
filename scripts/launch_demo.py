"""Start the local Nirnaya production API and static UI with one command."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = ROOT / "apps/nirnaya-ui/src"
PACKAGES = ("nirnaya-core", "nirnaya-presolve", "nirnaya-gpu", "nirnaya-solver", "nirnaya-api")


def _check_bindable(host: str, port: int, label: str) -> None:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    probe = socket.socket(family, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
    except OSError as exc:
        raise SystemExit(f"{label} cannot bind {host}:{port}: {exc}") from exc
    finally:
        probe.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--api-url", help="Browser-reachable API base URL; defaults to the bind address")
    parser.add_argument("--ui-host", default="127.0.0.1")
    parser.add_argument("--ui-port", type=int, default=8080)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if args.api_port == args.ui_port:
        raise SystemExit("Nirnaya API and UI require different TCP ports")
    _check_bindable(args.api_host, args.api_port, "Nirnaya API")
    _check_bindable(args.ui_host, args.ui_port, "Nirnaya UI")

    for package in PACKAGES:
        source = ROOT / "packages" / package / "src"
        sys.path.insert(0, str(source))
    subprocess.run([sys.executable, str(ROOT / "examples/flagship/generate_refinery_demo.py")], check=True)

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Uvicorn is required. Install the repository packages first.") from exc

    ui_server = ThreadingHTTPServer((args.ui_host, args.ui_port),
        partial(SimpleHTTPRequestHandler, directory=str(UI_ROOT)))
    advertised_host = "127.0.0.1" if args.api_host in ("0.0.0.0", "::") else args.api_host
    api_url = args.api_url or f"http://{advertised_host}:{args.api_port}"
    ui_advertised_host = "127.0.0.1" if args.ui_host in ("0.0.0.0", "::") else args.ui_host
    ui_origin = f"http://{ui_advertised_host}:{args.ui_port}"
    configured_origins = [
        origin.strip()
        for origin in os.environ.get("NIRNAYA_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    os.environ["NIRNAYA_CORS_ORIGINS"] = ",".join(dict.fromkeys([*configured_origins, ui_origin]))
    config = uvicorn.Config("nirnaya_api.server:app", host=args.api_host,
        port=args.api_port, log_level="warning")
    api_server = uvicorn.Server(config)
    api_thread = threading.Thread(target=api_server.run, name="nirnaya-api", daemon=True)
    api_thread.start()
    ui_thread = threading.Thread(target=ui_server.serve_forever, name="nirnaya-ui", daemon=True)
    ui_thread.start()

    ui_url = f"{ui_origin}/?{urlencode({'api': api_url})}"
    health_host = "127.0.0.1" if args.api_host in ("0.0.0.0", "::") else args.api_host
    health_url = f"http://{health_host}:{args.api_port}/health"
    healthy = False
    for _ in range(75):
        # Do not mistake an already-running process on the requested port for
        # this launcher's API. Uvicorn sets .started only after this process
        # has bound the socket and begun serving its own imported app.
        if not api_server.started:
            if not api_thread.is_alive():
                ui_server.shutdown()
                ui_server.server_close()
                api_server.should_exit = True
                raise SystemExit(f"Nirnaya API process exited before becoming healthy at {health_url}")
            time.sleep(.2)
            continue
        try:
            with urllib.request.urlopen(health_url, timeout=1) as response:
                report = json.loads(response.read().decode("utf-8"))
                if response.status == 200 and report.get("status") == "ok":
                    healthy = True
                    break
        except Exception:
            time.sleep(.2)
    if not healthy:
        ui_server.shutdown()
        ui_server.server_close()
        api_server.should_exit = True
        api_thread.join(timeout=3)
        raise SystemExit(f"Nirnaya API did not become healthy at {health_url}")

    print("Nirnaya judge demo is running locally.")
    print(f"Dashboard: {ui_url}")
    print(f"Production API: {api_url}")
    print(f"API health: {health_url}")
    import nirnaya_api.server as api_module
    import nirnaya_solver.solver as solver_module
    print(f"API source: {Path(api_module.__file__).resolve()}")
    print(f"Solver source: {Path(solver_module.__file__).resolve()}")
    print("Flagship scenario: MRPL-inspired synthetic assumptions; no plant operating data.")
    print("Press Ctrl+C to stop both local servers.")
    if not args.no_browser:
        import webbrowser
        webbrowser.open(ui_url)
    try:
        while api_thread.is_alive() and ui_thread.is_alive():
            time.sleep(.5)
    except KeyboardInterrupt:
        print("Stopping Nirnaya demo servers.")
    finally:
        ui_server.shutdown()
        ui_server.server_close()
        api_server.should_exit = True
        api_thread.join(timeout=3)


if __name__ == "__main__":
    main()
