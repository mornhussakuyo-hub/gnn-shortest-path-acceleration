param(
    [ValidateSet("screening", "full")]
    [string]$Mode = "screening",
    [string]$Variants = "",
    [string]$Seeds = "",
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv-gnn\Scripts\python.exe"
$runner = Join-Path $PSScriptRoot "run_nbfnet_ablation.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "CUDA environment not found: $python"
}

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$arguments = @($runner, "--mode", $Mode)
if ($Variants) {
    $arguments += @("--variants", $Variants)
}
if ($Seeds) {
    $arguments += @("--seeds", $Seeds)
}
if ($DryRun) {
    $arguments += "--dry-run"
}
if ($Force) {
    $arguments += "--force"
}

Push-Location $repoRoot
try {
    & $python @arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
