$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $scriptDir "venv\Scripts\python.exe"
$pythonHome = Join-Path $scriptDir "python310"

if (-not (Test-Path $python)) {
    [Console]::Error.WriteLine("Python interpreter not found: $python")
    exit 1
}

if (Test-Path $pythonHome) {
    $env:PYTHONHOME = $pythonHome
}

[Console]::Error.WriteLine("Python: $python")
[Console]::Error.WriteLine("Version: $(& $python --version 2>&1)")
[Console]::Error.WriteLine("yaml check: $(& $python -c 'import yaml; print(""OK"")' 2>&1)")

& $python (Join-Path $scriptDir "main.py") @args
exit $LASTEXITCODE
