$ErrorActionPreference = "Stop"
$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $AppDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location $AppDir

$detectorPath = Join-Path $AppDir "ai-detector.exe"
$webPath = Join-Path $AppDir "ai-detector-web.exe"
$detectorLog = Join-Path $LogDir "detector.log"
$detectorErrorLog = Join-Path $LogDir "detector-error.log"
$webLog = Join-Path $LogDir "web.log"
$webErrorLog = Join-Path $LogDir "web-error.log"

$detectorProcess = $null
$webProcess = $null
try {
    $detectorProcess = Start-Process `
        -FilePath $detectorPath `
        -WorkingDirectory $AppDir `
        -RedirectStandardOutput $detectorLog `
        -RedirectStandardError $detectorErrorLog `
        -PassThru
    $webProcess = Start-Process `
        -FilePath $webPath `
        -WorkingDirectory $AppDir `
        -RedirectStandardOutput $webLog `
        -RedirectStandardError $webErrorLog `
        -PassThru

    for ($attempt = 0; $attempt -lt 80; $attempt++) {
        if ($detectorProcess.HasExited) {
            throw "The detector stopped during startup. See $detectorErrorLog"
        }
        if ($webProcess.HasExited) {
            throw "The web application stopped during startup. See $webErrorLog"
        }
        try {
            Invoke-WebRequest `
                -UseBasicParsing `
                -Uri "http://127.0.0.1/" `
                -TimeoutSec 1 | Out-Null
            Start-Process "http://localhost"
            break
        }
        catch {
            Start-Sleep -Milliseconds 250
        }
    }

    Wait-Process -Id $webProcess.Id
}
finally {
    if ($null -ne $webProcess -and -not $webProcess.HasExited) {
        Stop-Process -Id $webProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($null -ne $detectorProcess -and -not $detectorProcess.HasExited) {
        Stop-Process -Id $detectorProcess.Id -Force -ErrorAction SilentlyContinue
    }
}
