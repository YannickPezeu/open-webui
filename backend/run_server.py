#!/usr/bin/env python3
import os
import sys
import uvicorn

# Add the backend directory to Python path
backend_dir = r"C:\Dev\openwebui-k8s\local_code\open-webui\backend"
sys.path.insert(0, backend_dir)

# Change to the backend directory
os.chdir(backend_dir)

# Run uvicorn programmatically
if __name__ == "__main__":
    uvicorn.run(
        "open_webui.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        forwarded_allow_ips="*"
    )