$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$env:HF_HOME = Join-Path $root ".data\cache\huggingface"
$env:TORCH_HOME = Join-Path $root ".data\cache\torch"
$env:PIP_CACHE_DIR = Join-Path $root ".data\cache\pip"
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Ejecuta primero scripts\setup.ps1." }
& $python -c "import streamlit" 2>$null
if ($LASTEXITCODE -ne 0) { throw "El entorno no funciona. Ejecuta scripts\setup.ps1 para repararlo." }
Push-Location $root
try {
    & $python -m streamlit run (Join-Path $root "app.py") --server.address 127.0.0.1
} finally { Pop-Location }
exit $LASTEXITCODE
