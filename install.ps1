[CmdletBinding()]
param(
    [switch]$Yes,
    [switch]$DryRun,
    [switch]$NoStart,
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "AI Detector")
)

$ErrorActionPreference = "Stop"
$Repository = "ESchouten/ai-detector"
$ApiRoot = "https://api.github.com/repos/$Repository"
$RawRoot = "https://raw.githubusercontent.com/$Repository"

function Write-Step([string]$Message) {
    Write-Host "==> $Message"
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Confirm-Checksum(
    [string]$Archive,
    [string]$Expected
) {
    $normalized = ($Expected -replace "^sha256:", "").Trim().ToLowerInvariant()
    if ($normalized -notmatch "^[0-9a-f]{64}$") {
        throw "Release checksum is invalid."
    }
    if ((Get-Sha256 $Archive) -ne $normalized) {
        throw "Release checksum verification failed."
    }
}

function Get-VerifiedReleaseAsset(
    [object[]]$Releases,
    [string]$TagPrefix,
    [string]$ArchivePattern
) {
    foreach ($release in $Releases) {
        if (
            $release.draft -or
            $release.prerelease -or
            -not ([string]$release.tag_name).StartsWith(
                $TagPrefix,
                [StringComparison]::Ordinal
            )
        ) {
            continue
        }

        $archives = @($release.assets | Where-Object { $_.name -match $ArchivePattern })
        if ($archives.Count -eq 0) {
            continue
        }
        $archive = $archives[0]
        $digest = [string]$archive.digest
        if ($digest -notmatch "^sha256:[0-9a-fA-F]{64}$") {
            continue
        }

        return [PSCustomObject]@{
            Archive = $archive
            Digest = $digest
            Tag = [string]$release.tag_name
        }
    }

    throw "No checksum-verified release asset matched $TagPrefix / $ArchivePattern"
}

$architecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
if ($architecture -ne [System.Runtime.InteropServices.Architecture]::X64) {
    throw "The native Windows application requires Windows x64; detected $architecture."
}

$gpuNames = @(
    Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
        ForEach-Object { $_.Name }
)
$gpuLabel = if ($gpuNames.Count) { $gpuNames -join ", " } else { "WindowsML auto-detection" }

Write-Step "Platform: Windows x64"
Write-Step "Acceleration: WindowsML ($gpuLabel)"
Write-Step "Install directory: $InstallDir"
Write-Step "Plan: download the latest detector + app releases, verify SHA-256, and preserve local data"

if ($DryRun) {
    exit 0
}

if (-not $Yes) {
    $reply = Read-Host "Continue? [y/N]"
    if ($reply -notmatch "^[Yy]$") {
        exit 0
    }
}

$temporaryDirectory = Join-Path ([IO.Path]::GetTempPath()) ("ai-detector-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $temporaryDirectory | Out-Null

try {
    $headers = @{
        Accept = "application/vnd.github+json"
        "User-Agent" = "ai-detector-installer"
    }
    $releases = @(
        Invoke-RestMethod -UseBasicParsing -Headers $headers `
            -Uri "$ApiRoot/releases?per_page=100"
    )
    $detectorRelease = Get-VerifiedReleaseAsset `
        -Releases $releases `
        -TagPrefix "detector/v" `
        -ArchivePattern "^aidetector-winml-.*[.]zip$"
    $webRelease = Get-VerifiedReleaseAsset `
        -Releases $releases `
        -TagPrefix "web/v" `
        -ArchivePattern "^aidetector-web-windows-.*[.]zip$"

    Write-Step "Detector release: $($detectorRelease.Tag) ($($detectorRelease.Archive.name))"
    Write-Step "App release: $($webRelease.Tag) ($($webRelease.Archive.name))"

    $detectorArchive = Join-Path $temporaryDirectory "detector.zip"
    $webArchive = Join-Path $temporaryDirectory "web.zip"
    Invoke-WebRequest -UseBasicParsing `
        -Uri $detectorRelease.Archive.browser_download_url `
        -OutFile $detectorArchive
    Invoke-WebRequest -UseBasicParsing `
        -Uri $webRelease.Archive.browser_download_url `
        -OutFile $webArchive

    Confirm-Checksum $detectorArchive $detectorRelease.Digest
    Confirm-Checksum $webArchive $webRelease.Digest

    $detectorDirectory = Join-Path $temporaryDirectory "detector"
    $webDirectory = Join-Path $temporaryDirectory "web"
    Expand-Archive -Path $detectorArchive -DestinationPath $detectorDirectory
    Expand-Archive -Path $webArchive -DestinationPath $webDirectory

    $detectorExecutable = Get-ChildItem -Path $detectorDirectory -Recurse -File `
        -Filter "aidetector-winml-*.exe" |
        Select-Object -First 1
    $webExecutable = Get-ChildItem -Path $webDirectory -Recurse -File `
        -Filter "aidetector-web-windows-*.exe" |
        Select-Object -First 1
    if ($null -eq $detectorExecutable) {
        throw "Detector release contains no WindowsML executable."
    }
    if ($null -eq $webExecutable) {
        throw "Web release contains no Windows executable."
    }

    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    Copy-Item -Path $detectorExecutable.FullName `
        -Destination (Join-Path $InstallDir "ai-detector.exe") -Force
    Copy-Item -Path $webExecutable.FullName `
        -Destination (Join-Path $InstallDir "ai-detector-web.exe") -Force
    Invoke-WebRequest -UseBasicParsing `
        -Uri "$RawRoot/main/packaging/native/AI%20Detector.ps1" `
        -OutFile (Join-Path $InstallDir "AI Detector.ps1")
    Invoke-WebRequest -UseBasicParsing `
        -Uri "$RawRoot/main/packaging/native/README.txt" `
        -OutFile (Join-Path $InstallDir "README.txt")
}
finally {
    Remove-Item -Path $temporaryDirectory -Recurse -Force -ErrorAction SilentlyContinue
}

$launcher = Join-Path $InstallDir "AI Detector.ps1"
if (-not $NoStart) {
    Start-Process powershell.exe -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$launcher`""
    )
    Write-Step "AI Detector is starting and will open in your browser."
}
else {
    Write-Step "Installed. Start $launcher"
}
