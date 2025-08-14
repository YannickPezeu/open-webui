export CORS_ALLOW_ORIGIN=http://localhost:5173/
PORT="${PORT:-8080}"
/mnt/c/Users/pezeu/AppData/Local/anaconda3/envs/openwebui-k8s/python.exe -m uvicorn open_webui.main:app --port $PORT --host 0.0.0.0 --forwarded-allow-ips '*' --reload
