param(
    [Parameter(Mandatory=$true)][int]$TrainingProcessId,
    [Parameter(Mandatory=$true)][string]$RunDirectory
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$run = [System.IO.Path]::GetFullPath((Join-Path $root $RunDirectory))
$reportPath = Join-Path $run "training_report.json"
$report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
if ($report.status -ne "completado_experimental") {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $TrainingProcessId"
    if ($null -ne $process) {
        if ($process.CommandLine -notmatch "train_yolo.py" -or $process.CommandLine -notmatch [regex]::Escape((Split-Path $run -Leaf))) {
            throw "El PID no corresponde a este entrenamiento; no se esperara a otro proceso."
        }
        Write-Host "Esperando el entrenamiento existente, sin reiniciarlo ni consultar su progreso."
        try {
            # Espera del sistema operativo, no bucle de sondeo ni reentrenamiento.
            $nativeProcess = [System.Diagnostics.Process]::GetProcessById($TrainingProcessId)
            $nativeProcess.WaitForExit()
        } catch [System.ArgumentException] {
            # El proceso pudo terminar entre la consulta y la espera; se valida el informe abajo.
        }
    }
}
$report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
if ($report.status -ne "completado_experimental") {
    throw "El entrenamiento no termino con evaluacion completa. No se activara un checkpoint parcial."
}
& $python -B (Join-Path $root "scripts\finalize_yolo.py") --run $run --activate-experimental
if ($LASTEXITCODE -ne 0) { throw "No se pudo activar el modelo; los resultados del entrenamiento se conservan." }
& $python -B (Join-Path $root "scripts\smoke_test.py")
if ($LASTEXITCODE -ne 0) { throw "El modelo se activo, pero fallo la comprobacion HTTP; revisar el servidor." }
Write-Host "YOLO_READY: modelo experimental de residuos activado y servidor comprobado."
