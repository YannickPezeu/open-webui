#!/bin/bash
# Override PYTHONPATH via Windows env for the python.exe process
export CORS_ALLOW_ORIGIN=http://localhost:5173/
PORT="${PORT:-8080}"

cd /mnt/c/Dev/openwebui_k8s_v7/open-webui/backend

# Use cmd.exe to set PYTHONPATH for the Windows python process, overriding the system-level one
cmd.exe /C "set PYTHONPATH=C:\Dev\openwebui_k8s_v7\open-webui\backend&& set PYTHONIOENCODING=utf-8&& C:\Users\pezeu\AppData\Local\anaconda3\envs\openwebui-k8s\python.exe -m uvicorn open_webui.main:app --port $PORT --host 0.0.0.0 --forwarded-allow-ips '*' --reload"
