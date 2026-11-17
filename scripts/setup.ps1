param(
    [string]$PythonExecutable,
    [switch]$Recreate,
    [switch]$Segmentation,
    [switch]$Yolo
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$env:PIP_CACHE_DIR = Join-Path $root ".data\cache\pip"
$env:HF_HOME = Join-Path $root ".data\cache\huggingface"
$env:TORCH_HOME = Join-Path $root ".data\cache\torch"

function Test-Python([string]$Executable) {
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { return $false }
    try {
        & $Executable -c "import sys, venv; sys.exit(0 if (3,11) <= sys.version_info[:2] < (3,14) else 1)" 2>$null
        return $LASTEXITCODE -eq 0
    } catch { return $false }
}

if ($Recreate -or -not (Test-Python $python)) {
    $base = $null
    if ($PythonExecutable) {
        if (-not (Test-Python $PythonExecutable)) { throw "El Python indicado no funciona o no es 3.11-3.13." }
        $base = $PythonExecutable
    } else {
        $local = Join-Path $root ".data\python\3.12\python.exe"
        if (Test-Python $local) { $base = $local }
        if (-not $base -and (Get-Command py -ErrorAction SilentlyContinue)) {
            foreach ($version in @('-3.12', '-3.11', '-3.13')) {
                try {
                    $candidate = & py $version -c "import sys; print(sys.executable)" 2>$null
                    if ($LASTEXITCODE -eq 0 -and (Test-Python $candidate)) { $base = $candidate; break }
                } catch { }
            }
        }
        if (-not $base) {
            $command = Get-Command python -ErrorAction SilentlyContinue
            if ($command -and (Test-Python $command.Source)) { $base = $command.Source }
        }
    }
    if (-not $base) { throw "Instala Python 3.12 o usa -PythonExecutable con su ruta completa. No copies un .venv de otro equipo." }
    if ($base.StartsWith((Join-Path $root '.venv'), [StringComparison]::OrdinalIgnoreCase)) {
        throw "Para recrear .venv indica un Python base fuera de ese entorno."
    }
    if (Test-Path -LiteralPath (Join-Path $root '.venv')) {
        $backup = Join-Path $root ('.data\backups\venv-' + [guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $backup) | Out-Null
        Move-Item -LiteralPath (Join-Path $root '.venv') -Destination $backup
        Write-Host "Entorno anterior conservado en $backup"
    }
    & $base -m venv (Join-Path $root '.venv')
    if ($LASTEXITCODE -ne 0) { throw "No se pudo crear el entorno." }
}
& $python -m pip install --disable-pip-version-check -r (Join-Path $root "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "No se pudieron instalar las dependencias." }
if ($Segmentation) {
    & $python -m pip install -r (Join-Path $root "requirements-segmentation.txt")
    if ($LASTEXITCODE -ne 0) { throw "No se pudo instalar la segmentacion opcional." }
}
if ($Yolo) {
    & $python -m pip install -r (Join-Path $root "requirements-yolo.txt")
    if ($LASTEXITCODE -ne 0) { throw "No se pudo instalar YOLO opcional." }
}
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw "Hay dependencias incompatibles." }
Write-Host "Entorno listo. Inicia con scripts\run.ps1. No se descargaron modelos ni se consulto OpenAI."
