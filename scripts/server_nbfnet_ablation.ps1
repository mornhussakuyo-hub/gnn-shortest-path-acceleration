param(
    [ValidateSet("start", "status", "tail")]
    [string]$Action = "status",
    [ValidateSet("screening", "full")]
    [string]$Mode = "full",
    [string]$RemoteRepo = "~/gnn-shortest-path-acceleration",
    [int]$TailLines = 40
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $repoRoot ".server.env"

if (-not (Test-Path -LiteralPath $configPath)) {
    throw "Server config not found: $configPath"
}
if ($RemoteRepo -notmatch '^[A-Za-z0-9_./~$-]+$') {
    throw "RemoteRepo contains unsupported shell characters."
}

function Import-ServerEnv {
    param([string]$Path)
    $values = @{}
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
            $value = $matches[2].Trim()
            if (
                $value.Length -ge 2 -and
                (($value.StartsWith('"') -and $value.EndsWith('"')) -or
                 ($value.StartsWith("'") -and $value.EndsWith("'")))
            ) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            $values[$matches[1]] = $value
        }
    }
    return $values
}

$server = Import-ServerEnv -Path $configPath
foreach ($required in "SERVER_HOST", "SERVER_USER") {
    if (-not $server[$required]) {
        throw "Missing $required in $configPath"
    }
}

$sshExecutable = "ssh"
$sshArguments = @()
if ($server["SERVER_SSH_COMMAND"]) {
    $parseErrors = $null
    $tokens = [System.Management.Automation.PSParser]::Tokenize(
        $server["SERVER_SSH_COMMAND"],
        [ref]$parseErrors
    ) | Where-Object {
        $_.Type -in @("Command", "CommandArgument")
    }
    if ($parseErrors -or -not $tokens) {
        throw "SERVER_SSH_COMMAND could not be parsed safely."
    }
    $sshExecutable = $tokens[0].Content
    if ($tokens.Count -gt 1) {
        $sshArguments = @($tokens[1..($tokens.Count - 1)].Content)
    }
}
else {
    if ($server["SERVER_PORT"]) {
        $sshArguments += @("-p", $server["SERVER_PORT"])
    }
    $sshArguments += "$($server['SERVER_USER'])@$($server['SERVER_HOST'])"
}

$resultRoot = "results/gnn_v2/nbfnet_ablation/$Mode"
$launcherLog = "$resultRoot/launcher.log"
$pidFile = "$resultRoot/runner.pid"

switch ($Action) {
    "start" {
        $remoteCommand = @"
set -eu
cd $RemoteRepo
git switch main
git pull --ff-only origin main
mkdir -p $resultRoot/logs
if [ -f $pidFile ] && kill -0 `$(cat $pidFile) 2>/dev/null; then
  echo "already running pid=`$(cat $pidFile)"
  exit 2
fi
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv-gnn/bin/python scripts/run_nbfnet_ablation.py --mode $Mode \
  > $launcherLog 2>&1 < /dev/null &
echo `$! > $pidFile
echo "started pid=`$(cat $pidFile) mode=$Mode log=$launcherLog"
"@
    }
    "status" {
        $remoteCommand = @"
set -eu
cd $RemoteRepo
if [ -f $pidFile ] && kill -0 `$(cat $pidFile) 2>/dev/null; then
  echo "running pid=`$(cat $pidFile)"
else
  echo "not running"
fi
if [ -f $resultRoot/manifest.json ]; then
  .venv-gnn/bin/python -c 'import json; p=json.load(open("$resultRoot/manifest.json", encoding="utf-8")); runs=p.get("runs",{}); print("status="+p.get("status","unknown"), "completed="+str(sum(v.get("status")=="completed" for v in runs.values())), "failed="+str(sum(v.get("status")=="failed" for v in runs.values())))'
fi
if [ -f $launcherLog ]; then
  tail -n $TailLines $launcherLog
fi
"@
    }
    "tail" {
        $remoteCommand = @"
set -eu
cd $RemoteRepo
if [ -f $launcherLog ]; then
  tail -n $TailLines $launcherLog
else
  echo "log not found: $launcherLog"
fi
"@
    }
}

Write-Host "server=$($server['SERVER_HOST']) action=$Action mode=$Mode"
& $sshExecutable @sshArguments $remoteCommand
exit $LASTEXITCODE
