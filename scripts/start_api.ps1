$ErrorActionPreference = 'Stop'
python -m uvicorn nirnaya_api.server:app --host 127.0.0.1 --port 8000
