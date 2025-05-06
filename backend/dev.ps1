$env:PORT = 8080
uvicorn open_webui.main:app --port $env:PORT --host 0.0.0.0 --forwarded-allow-ips "*" --reload