#!/usr/bin/env powershell

# Set environment variable
$env:PORT = 8080

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Run uvicorn from a clean directory to avoid file conflicts
Push-Location C:\
try {
    python -m uvicorn open_webui.main:app --app-dir "$ScriptDir" --port $env:PORT --host 0.0.0.0 --forwarded-allow-ips "*" --reload
} finally {
    Pop-Location
}