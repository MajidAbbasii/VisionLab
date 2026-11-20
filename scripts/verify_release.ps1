param(
    [string]$Python = "",
    [switch]$SkipWheel
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

if ([string]::IsNullOrWhiteSpace($Python)) {
    $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) { $Python = $venvPython }
    else { $Python = "python" }
}

function Invoke-Checked([string]$File, [string[]]$Arguments) {
    Write-Host "`n> $File $($Arguments -join ' ')" -ForegroundColor Cyan
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE : $File $($Arguments -join ' ')"
    }
}

$required = @(
    "README.md",
    "CHANGELOG.md",
    "RELEASE_CHECKLIST.md",
    "pyproject.toml",
    "requirements.txt",
    "VisionLab.sln",
    "models/exported/best.onnx",
    "datasets/classes.txt",
    "tests/VisionLab.Core.Tests/VisionLab.Core.Tests.csproj"
)
foreach ($relative in $required) {
    if (-not (Test-Path (Join-Path $repoRoot $relative) -PathType Leaf)) {
        throw "Required release file is missing: $relative"
    }
}

Invoke-Checked $Python @("-m", "pip", "check")
Invoke-Checked $Python @("-m", "pytest", "-q")
Invoke-Checked $Python @("-m", "compileall", "-q", "src/visionlab", "scripts", "tests")
Invoke-Checked $Python @("-m", "visionlab.cli", "--help")
Invoke-Checked "dotnet" @("test", "VisionLab.sln", "-c", "Release")
Invoke-Checked "dotnet" @("build", "VisionLab.sln", "-c", "Release", "--no-restore")
Invoke-Checked "dotnet" @("run", "--no-build", "-c", "Release", "--project", "src/VisionLab.App", "--", "--help")

# Explicitly verify the solution with an installed .NET 8 SDK, not only the newest SDK on PATH.
$sdk8Directory = Join-Path $repoRoot "artifacts\sdk8-validation"
New-Item -ItemType Directory -Force -Path $sdk8Directory | Out-Null
Set-Content -Path (Join-Path $sdk8Directory "global.json") -Value '{"sdk":{"version":"8.0.101","rollForward":"latestPatch","allowPrerelease":false}}' -Encoding UTF8
Push-Location $sdk8Directory
try {
    $selectedSdk = (& dotnet --version).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $selectedSdk.StartsWith("8.")) {
        throw "The .NET 8 SDK could not be selected for release verification (selected: $selectedSdk)."
    }
    Write-Host "`n> .NET 8 SDK selected: $selectedSdk" -ForegroundColor Cyan
    Invoke-Checked "dotnet" @("test", (Join-Path $repoRoot "VisionLab.sln"), "-c", "Release", "--no-restore")
    Invoke-Checked "dotnet" @("build", (Join-Path $repoRoot "VisionLab.sln"), "-c", "Release", "--no-restore")
}
finally {
    Pop-Location
}

$publishDirectory = Join-Path $repoRoot "artifacts\publish\dotnet-app"
if (Test-Path $publishDirectory) {
    $resolvedPublish = (Resolve-Path $publishDirectory).Path
    $resolvedPublishRoot = (Resolve-Path (Join-Path $repoRoot "artifacts\publish")).Path
    if (-not $resolvedPublish.StartsWith($resolvedPublishRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clean a publish directory outside artifacts/publish: $resolvedPublish"
    }
    Remove-Item -LiteralPath $resolvedPublish -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $publishDirectory | Out-Null
Invoke-Checked "dotnet" @("publish", "src/VisionLab.App/VisionLab.App.csproj", "-c", "Release", "-o", $publishDirectory)

# The target package is Windows x64. Remove native runtime payloads for other OS/architectures.
$runtimeDirectory = Join-Path $publishDirectory "runtimes"
if (Test-Path $runtimeDirectory) {
    Get-ChildItem -LiteralPath $runtimeDirectory -Directory |
        Where-Object { $_.Name -ne "win-x64" } |
        Remove-Item -Recurse -Force
}
Get-ChildItem -LiteralPath $publishDirectory -Filter "*.pdb" -File |
    Remove-Item -Force

$modelDestination = Join-Path $publishDirectory "models\exported"
$labelsDestination = Join-Path $publishDirectory "datasets"
New-Item -ItemType Directory -Force -Path $modelDestination, $labelsDestination | Out-Null
Copy-Item -LiteralPath (Join-Path $repoRoot "models\exported\best.onnx") -Destination $modelDestination -Force
Copy-Item -LiteralPath (Join-Path $repoRoot "datasets\classes.txt") -Destination $labelsDestination -Force
Copy-Item -LiteralPath (Join-Path $repoRoot "docs\DOTNET_RUNTIME.md") -Destination (Join-Path $publishDirectory "README-RUNTIME.md") -Force

$smokeDirectory = Join-Path $repoRoot "artifacts\smoke-test"
New-Item -ItemType Directory -Force -Path $smokeDirectory | Out-Null
Push-Location $publishDirectory
try {
    Invoke-Checked ".\VisionLab.App.exe" @("--help")
    Invoke-Checked ".\VisionLab.App.exe" @(
        "image",
        (Join-Path $repoRoot "bus.jpg"),
        "--output",
        (Join-Path $smokeDirectory "dotnet-published-app-bus.jpg")
    )
}
finally {
    Pop-Location
}

$archivePath = Join-Path $repoRoot "artifacts\VisionLab-v1.0.0-rc1-dotnet-win-x64.zip"
Compress-Archive -Path (Join-Path $publishDirectory "*") -DestinationPath $archivePath -Force

$archiveSmokeDirectory = Join-Path $repoRoot "artifacts\archive-smoke"
if (Test-Path $archiveSmokeDirectory) {
    Remove-Item -LiteralPath $archiveSmokeDirectory -Recurse -Force
}
Expand-Archive -LiteralPath $archivePath -DestinationPath $archiveSmokeDirectory -Force
Push-Location $archiveSmokeDirectory
try {
    Invoke-Checked ".\VisionLab.App.exe" @(
        "image",
        (Join-Path $repoRoot "bus.jpg"),
        "--output",
        (Join-Path $smokeDirectory "dotnet-archive-bus.jpg")
    )
}
finally {
    Pop-Location
    Remove-Item -LiteralPath $archiveSmokeDirectory -Recurse -Force
}

if (-not $SkipWheel) {
    $wheelDirectory = Join-Path $repoRoot "artifacts\wheels"
    New-Item -ItemType Directory -Force -Path $wheelDirectory | Out-Null
    Invoke-Checked $Python @("-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation", "--wheel-dir", $wheelDirectory)

    $wheelPath = Join-Path $wheelDirectory "visionlab-1.0.0rc1-py3-none-any.whl"
    if (-not (Test-Path $wheelPath -PathType Leaf)) { throw "Expected Python wheel was not created: $wheelPath" }
    $wheelSmokeDirectory = Join-Path $repoRoot "artifacts\wheel-smoke"
    if (Test-Path $wheelSmokeDirectory) {
        Remove-Item -LiteralPath $wheelSmokeDirectory -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $wheelSmokeDirectory | Out-Null
    Invoke-Checked $Python @("-m", "pip", "install", "--no-deps", "--target", $wheelSmokeDirectory, $wheelPath)
    Invoke-Checked $Python @("-c", "import sys; sys.path.insert(0, sys.argv[1]); import visionlab; assert visionlab.__version__ == '1.0.0rc1'; print('Wheel import/version OK:', visionlab.__version__)", $wheelSmokeDirectory)
}

Invoke-Checked $Python @("scripts/write_release_manifest.py")

Write-Host "`nRelease verification passed. No model training or experiment search was started." -ForegroundColor Green
