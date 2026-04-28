# Run from the project root: .\start.ps1

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$env:PATH = "C:\Program Files\nodejs;" + $env:PATH

# ── kill any stale server on port 8000 ───────────────────────────────────────
$stale = netstat -ano | Select-String "127.0.0.1:8000" | Select-String "LISTENING"
if ($stale) {
    $stalePid = ($stale.ToString().Trim() -split '\s+')[-1]
    Write-Host "Killing stale process on port 8000 (PID $stalePid)..."
    Stop-Process -Id $stalePid -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}

# ── find python ───────────────────────────────────────────────────────────────
if (Test-Path "venv\Scripts\python.exe") {
    $python = "$Root\venv\Scripts\python.exe"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $python = "python"
} else {
    Write-Error "python not found. Install Python from https://python.org then re-run."
    exit 1
}

# ── find npm ──────────────────────────────────────────────────────────────────
$npmCmd = "C:\Program Files\nodejs\npm.cmd"
if (-not (Test-Path $npmCmd)) {
    $found = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $found) {
        Write-Error "npm not found. Install Node.js from https://nodejs.org then re-run."
        exit 1
    }
    $npmCmd = $found.Source
}

Write-Host "python: $python"
Write-Host "npm:    $(& cmd.exe /c `"$npmCmd`" --version)"
Write-Host ""

# ── start FastAPI server ──────────────────────────────────────────────────────
Write-Host "Starting FastAPI server on http://127.0.0.1:8000 ..."
$server = Start-Process -FilePath $python `
    -ArgumentList "-m uvicorn server:app --host 127.0.0.1 --port 8000" `
    -WorkingDirectory $Root `
    -NoNewWindow -PassThru

# ── start Plasmo extension dev server ────────────────────────────────────────
Write-Host "Starting extension dev server..."
$ext = Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/c `"$npmCmd`" run dev" `
    -WorkingDirectory "$Root\extension" `
    -NoNewWindow -PassThru

Write-Host "Both servers running. Press Ctrl+C to stop."
try {
    Wait-Process -Id $server.Id -ErrorAction SilentlyContinue
} finally {
    Stop-Process -Id $server.Id -ErrorAction SilentlyContinue
    Stop-Process -Id $ext.Id   -ErrorAction SilentlyContinue
}
