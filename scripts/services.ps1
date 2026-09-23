param([ValidateSet('start','stop','restart')][string]$Action = 'start',
      [ValidateSet('all','web','bot')][string]$Service = 'all')
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonExe = Join-Path $projectRoot 'venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $pythonExe)) { throw 'Run setup_windows.bat first.' }
$runtimePath = Join-Path $projectRoot 'runtime'
New-Item -ItemType Directory -Path $runtimePath -Force | Out-Null
$services = if ($Service -eq 'all') { @('web','bot') } else { @($Service) }
function Get-DotEnvValue([string]$Name, [string]$Default) {
    $envFile = Join-Path $projectRoot '.env'
    if (Test-Path -LiteralPath $envFile) {
        $match = Get-Content -LiteralPath $envFile | Where-Object { $_ -match "^\s*$([regex]::Escape($Name))\s*=" } | Select-Object -Last 1
        if ($match) { return ($match -split '=', 2)[1].Trim() }
    }
    return $Default
}
$webHost = Get-DotEnvValue 'WEB_HOST' '127.0.0.1'
$webPort = [int](Get-DotEnvValue 'WEB_PORT' '5000')
$webTimeoutSeconds = [int](Get-DotEnvValue 'WEB_START_TIMEOUT_SECONDS' '90')
$botTimeoutSeconds = [int](Get-DotEnvValue 'BOT_START_TIMEOUT_SECONDS' '30')
Push-Location $projectRoot
try {
    if ($Action -in @('stop','restart')) {
        & $pythonExe -m services.launcher_check
        if ($LASTEXITCODE -ne 0) { throw 'Tasks are active. Stop them in Studio and wait before stopping services.' }
        foreach ($name in $services) {
            $recordPath = Join-Path $runtimePath "$name-process.json"
            if (Test-Path -LiteralPath $recordPath) {
                $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
                $process = Get-CimInstance Win32_Process -Filter "ProcessId = $($record.pid)" -ErrorAction SilentlyContinue
                if ($process -and $process.CommandLine.Contains($record.script) -and
                    $process.CreationDate.ToUniversalTime().ToString('o') -eq $record.created) {
                    Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
                }
                Remove-Item -LiteralPath $recordPath -Force -ErrorAction SilentlyContinue
            }
            $targetScript = Join-Path $projectRoot $(if ($name -eq 'web') { 'app.py' } else { 'bot.py' })
            $orphans = Get-CimInstance Win32_Process | Where-Object {
                $_.CommandLine -and $_.CommandLine.Contains($targetScript)
            }
            foreach ($p in $orphans) {
                Write-Host "Stopping existing $name instance (PID $($p.ProcessId))"
                Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
            }
        }
    }
    if ($Action -eq 'restart') { Start-Sleep -Milliseconds 800 }
    if ($Action -in @('start','restart')) {
        foreach ($name in $services) {
            $script = Join-Path $projectRoot $(if ($name -eq 'web') { 'app.py' } else { 'bot.py' })
            $existing = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($script) }
            if ($existing) { Write-Host "$name already running"; continue }
            $process = Start-Process -FilePath $pythonExe -ArgumentList @('-u', "`"$script`"") -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimePath "$name.stdout.log") -RedirectStandardError (Join-Path $runtimePath "$name.stderr.log")
            Start-Sleep -Milliseconds 500
            $info = Get-CimInstance Win32_Process -Filter "ProcessId = $($process.Id)"
            if (!$info) { throw "$name exited. Check runtime logs." }
            if ($name -eq 'web') {
                $ready = $false
                $healthUrl = "http://${webHost}:${webPort}/api/device"
                $deadline = (Get-Date).AddSeconds($webTimeoutSeconds)
                Write-Host "Waiting up to ${webTimeoutSeconds}s for Web health check: $healthUrl"
                while ((Get-Date) -lt $deadline) {
                    $running = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($script) }
                    if (!$running) { break }
                    try {
                        $response = Invoke-WebRequest -UseBasicParsing -Uri $healthUrl -TimeoutSec 3
                        if ($response.StatusCode -eq 200) { $ready = $true; break }
                    } catch { Start-Sleep -Milliseconds 500 }
                }
                if (!$ready) {
                    $orphans = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($script) }
                    foreach ($p in $orphans) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
                    throw "Web did not become ready on $healthUrl within ${webTimeoutSeconds}s. Check runtime/web.stdout.log and runtime/web.stderr.log."
                }
            } else {
                $deadline = (Get-Date).AddSeconds($botTimeoutSeconds)
                $stillRunning = $null
                while ((Get-Date) -lt $deadline) {
                    $stillRunning = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($script) }
                    if ($stillRunning) { break }
                    Start-Sleep -Milliseconds 500
                }
                if (!$stillRunning) { throw "Bot exited during startup within ${botTimeoutSeconds}s. Check runtime/bot.stderr.log." }
            }
            @{pid=$process.Id; script=$script; created=$info.CreationDate.ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtimePath "$name-process.json") -Encoding UTF8
            Write-Host "$name ready (PID $($process.Id)); logs: $runtimePath"
        }
    }
} finally { Pop-Location }
