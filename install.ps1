<#
.SYNOPSIS
    Smartloop installer for Windows (PowerShell).

.DESCRIPTION
    One-click install. Downloads the pre-baked relocatable distribution from
    dl.smartloop.ai, verifies SHA256 checksums, extracts it to
    %USERPROFILE%\.smartloop\<version>, writes a `slp` launcher shim on PATH, and
    registers a per-user scheduled task that keeps the agent running.

    This is the Windows counterpart of install.sh. Run it one-click with:

        irm https://smartloop.ai/install.ps1 | iex
#>

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

# Render the UTF-8 box-drawing banner correctly. Windows PowerShell 5.1 defaults
# the console to the OEM code page, which mojibakes the block glyphs.
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$Version    = '1.2.35'
$BaseUrl    = "https://dl.smartloop.ai/slp/$Version"
$BaseDir    = Join-Path $HOME '.smartloop'
$InstallDir = Join-Path $BaseDir $Version
$CacheDir   = Join-Path $HOME '.cache\smartloop'

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
function Write-Muted { param([string]$Message) Write-Host $Message -ForegroundColor DarkGray }
function Write-Info  { param([string]$Message) Write-Host $Message }
function Write-Ok    { param([string]$Message) Write-Host $Message -ForegroundColor Green }
function Die {
    param([string]$Message)
    Write-Host "Error: $Message" -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------------------
# Platform check — only x86_64 Windows is published.
# ---------------------------------------------------------------------------
function Test-Platform {
    if (-not [Environment]::Is64BitOperatingSystem) {
        Die 'Only 64-bit Windows is supported'
    }
    $arch = $env:PROCESSOR_ARCHITECTURE
    if ($arch -eq 'ARM64') {
        Die 'Only x86_64 (amd64) is supported on Windows'
    }
    if (-not (Get-Command tar.exe -ErrorAction SilentlyContinue)) {
        Die 'tar.exe not found. Windows 10 (1803+) or Windows 11 is required.'
    }
}

# ---------------------------------------------------------------------------
# Vulkan runtime requirements.
#
# The Windows build links llama.cpp's Vulkan backend (ggml-vulkan.dll). The
# Vulkan loader (vulkan-1.dll) is bundled into the release archive next to
# ggml-vulkan.dll at build time (see smartloop.spec), so nothing is
# provisioned here. The GPU driver itself (NVIDIA/AMD/Intel) must come from
# the host; a System32 vulkan-1.dll is a reliable sign one is installed. We
# only report status up front -- without a driver, inference falls back to CPU.
# ---------------------------------------------------------------------------
function Test-VulkanRuntime {
    return (Test-Path (Join-Path $env:WINDIR 'System32\vulkan-1.dll'))
}

# ---------------------------------------------------------------------------
# Checksums — fetched from the release at install time.
# File format (one per line): "<sha256>  <filename>"
# ---------------------------------------------------------------------------
$script:Checksums = @{}

function Get-Checksums {
    $url = "$BaseUrl/checksums-sha256.txt"
    try {
        $text = (Invoke-WebRequest -Uri $url -UseBasicParsing).Content
    } catch {
        Die "Failed to download checksums: $($_.Exception.Message)"
    }
    foreach ($line in ($text -split "`n")) {
        $line = $line.Trim()
        if (-not $line) { continue }
        # "<hash><space><space><filename>"
        $parts = $line -split '\s+', 2
        if ($parts.Count -eq 2) {
            $script:Checksums[$parts[1].Trim()] = $parts[0].Trim()
        }
    }
}

function Get-ExpectedSha256 {
    param([string]$FileName)
    if ($script:Checksums.ContainsKey($FileName)) { return $script:Checksums[$FileName] }
    return $null
}

function Test-Checksum {
    param([string]$Path, [string]$Expected)
    if (-not $Expected) { return $true }   # nothing to verify against
    $actual = (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLower()
    return ($actual -eq $Expected.ToLower())
}

# ---------------------------------------------------------------------------
# Download a file with a progress bar (streams so we can show progress).
# ---------------------------------------------------------------------------
function Save-Url {
    param(
        [string]$Url,
        [string]$OutFile,
        [string]$Activity = 'Downloading',
        [long]$KnownLength = 0
    )

    Add-Type -AssemblyName System.Net.Http
    $client = [System.Net.Http.HttpClient]::new()
    $client.Timeout = [TimeSpan]::FromHours(2)
    try {
        $resp = $client.GetAsync($Url, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
        if (-not $resp.IsSuccessStatusCode) {
            throw "HTTP $([int]$resp.StatusCode) for $Url"
        }
        $total = $KnownLength
        if ($total -le 0 -and $resp.Content.Headers.ContentLength) { $total = [long]$resp.Content.Headers.ContentLength }

        $src = $resp.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
        $dst = [System.IO.File]::Create($OutFile)
        try {
            $buffer = New-Object byte[] (1MB)
            $read = 0
            $readTotal = [long]0
            $lastShown = -1
            while (($read = $src.Read($buffer, 0, $buffer.Length)) -gt 0) {
                $dst.Write($buffer, 0, $read)
                $readTotal += $read
                if ($total -gt 0) {
                    $pct = [int](($readTotal * 100) / $total)
                    if ($pct -ne $lastShown) {
                        $lastShown = $pct
                        $mb = '{0:N1}' -f ($readTotal / 1MB)
                        $totMb = '{0:N1}' -f ($total / 1MB)
                        Write-Progress -Activity $Activity -Status "$mb MB / $totMb MB" -PercentComplete $pct
                    }
                }
            }
        } finally {
            $dst.Close()
            $src.Close()
        }
        Write-Progress -Activity $Activity -Completed
    } finally {
        $client.Dispose()
    }
}

function Get-UrlContentLength {
    param([string]$Url)
    Add-Type -AssemblyName System.Net.Http
    $client = [System.Net.Http.HttpClient]::new()
    try {
        $req  = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Head, $Url)
        $resp = $client.SendAsync($req).GetAwaiter().GetResult()
        if (-not $resp.IsSuccessStatusCode) { return -1 }
        if ($resp.Content.Headers.ContentLength) { return [long]$resp.Content.Headers.ContentLength }
        return 0
    } catch {
        return -1
    } finally {
        $client.Dispose()
    }
}

# ---------------------------------------------------------------------------
# Download the release archive into the cache. Hosted on GCS (dl.smartloop.ai),
# so there is no asset size limit and no split parts.
# ---------------------------------------------------------------------------
function Get-Archive {
    New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
    $cached = Join-Path $CacheDir "slp-$Version-windows-amd64.zip"
    $archiveName = 'windows-amd64-slp.zip'

    # Reuse a valid cached archive.
    if (Test-Path $cached) {
        $expected = Get-ExpectedSha256 $archiveName
        if (Test-Checksum -Path $cached -Expected $expected) {
            Write-Muted "Hit:1 $CacheDir smartloop $Version"
            return $cached
        }
    }

    Save-Url -Url "$BaseUrl/$archiveName" -OutFile $cached -Activity "Get:1 smartloop $Version"
    $expected = Get-ExpectedSha256 $archiveName
    if (-not (Test-Checksum -Path $cached -Expected $expected)) {
        Die 'Checksum verification failed.'
    }
    return $cached
}

# ---------------------------------------------------------------------------
# PATH — add the base dir (where the slp shim lives) to the user PATH.
# ---------------------------------------------------------------------------
function Add-ToUserPath {
    param([string]$Dir)
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if ($null -eq $userPath) { $userPath = '' }
    $entries = $userPath -split ';' | Where-Object { $_ -ne '' }
    if ($entries -notcontains $Dir) {
        $newPath = (@($Dir) + $entries) -join ';'
        [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
        Write-Muted "Added slp to PATH ($Dir)"
    }
    # Make it usable in the current session too.
    if (($env:Path -split ';') -notcontains $Dir) {
        $env:Path = "$Dir;$env:Path"
    }
}

# The agent does not need a Windows service: `slp` auto-starts the background
# agent on first run if it isn't already running (see commands/cli.py).

function Show-Banner {
    # Build the block glyphs from code points so the source stays pure ASCII —
    # `irm | iex` on PS 5.1 decodes the script as Latin-1 and would otherwise
    # corrupt any raw UTF-8 characters baked into the file.
    $f = [char]0x2588  # full block  (█)
    $t = [char]0x2580  # upper half  (▀)
    $b = [char]0x2584  # lower half  (▄)
    $line1 = 'FT FTBTF BTF FTF TFT F   FTF FTF FTF'.Replace('F', $f).Replace('T', $t).Replace('B', $b)
    $line2 = 'BF F T F FTF FTB  F  FBB FBF FBF FTT'.Replace('F', $f).Replace('T', $t).Replace('B', $b)

    # Brand pink (256-color 205) via ANSI — brighter than the console's dull
    # "Magenta". Matches PINK in install.sh.
    $esc   = [char]0x1b
    $pink  = "$esc[38;5;205m"
    $reset = "$esc[0m"

    Write-Host ''
    Write-Host "$pink$line1$reset"
    Write-Host "$pink$line2$reset"
    Write-Host ''
    Write-Muted "Version: $Version"
    Write-Host ''
    Write-Muted 'To get started, open a new terminal and run:'
    Write-Host ''
    Write-Host '  slp  ' -NoNewline; Write-Muted '# Start the agent harness'
    Write-Host ''
    Write-Muted 'For more information visit https://smartloop.ai/docs/intro/'
    Write-Host ''
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function Install-Smartloop {
    Write-Muted 'Reading package lists... '
    Test-Platform
    Write-Muted 'Reading package lists... Done'

    # Preflight: report GPU status. The published build is GPU-accelerated via
    # Vulkan, so flag a missing Vulkan runtime up front.
    if (Test-VulkanRuntime) {
        Write-Muted 'Vulkan runtime detected; GPU acceleration is available.'
    } else {
        Write-Host 'Note: no Vulkan runtime detected. Install a current GPU driver (NVIDIA/AMD/Intel) for GPU acceleration; inference will otherwise run on CPU.' -ForegroundColor Yellow
    }

    $slpExisting = Join-Path $InstallDir 'bin\slp.cmd'

    if (Test-Path $slpExisting) {
        Write-Muted "smartloop is already the newest version ($Version)."
    } else {
        Write-Muted 'The following NEW packages will be installed:'
        Write-Host '  smartloop'

        Get-Checksums

        Write-Muted "[1/3] Downloading smartloop ($Version)"
        $archive = Get-Archive

        Write-Muted "[2/3] Unpacking smartloop ($Version)"
        $tmpDir = Join-Path ([System.IO.Path]::GetTempPath()) ("slp-x-" + [System.Guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
        try {
            # bsdtar (Windows 10 1803+) auto-detects and extracts zip archives.
            & tar.exe -xf $archive -C $tmpDir
            if ($LASTEXITCODE -ne 0) { Die 'Failed to extract archive' }

            Write-Muted "[3/3] Setting up smartloop ($Version)"
            New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
            Get-ChildItem -Path $InstallDir -Force | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

            $appDir = Join-Path $tmpDir 'slp'
            if (-not (Test-Path $appDir)) { Die "Unexpected archive layout: $appDir not found" }
            Copy-Item -Path (Join-Path $appDir '*') -Destination $InstallDir -Recurse -Force
        } finally {
            Remove-Item -Recurse -Force $tmpDir -ErrorAction SilentlyContinue
        }
    }

    # Track installed version.
    Set-Content -Path (Join-Path $BaseDir 'version') -Value $Version -Encoding ascii
    $installedFile = Join-Path $BaseDir 'installed'
    $alreadyListed = (Test-Path $installedFile) -and ((Get-Content $installedFile) -contains $Version)
    if (-not $alreadyListed) { Add-Content -Path $installedFile -Value $Version }

    # The distribution is self-contained: the single slp.exe sits at the top
    # of the install dir next to its bundled DLLs (Vulkan loader, MSVC runtime)
    # and embedding model (embeddings\).
    $slpBin = Join-Path $InstallDir 'slp.exe'
    if (-not (Test-Path $slpBin)) { Die "Executable not found at $slpBin" }

    # Write a stable shim on PATH that forwards to the versioned launcher
    # (Windows equivalent of the ~/.smartloop/slp symlink on Unix).
    $shim = Join-Path $BaseDir 'slp.cmd'
    @"
@echo off
"$slpBin" %*
"@ | Set-Content -Path $shim -Encoding ascii

    # Verify the install.
    & $slpBin --help *> $null
    if ($LASTEXITCODE -ne 0) {
        Die "Installation verification failed: 'slp --help' did not succeed"
    }

    Add-ToUserPath -Dir $BaseDir

    Show-Banner
}

Install-Smartloop
