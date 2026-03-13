param(
    [string]$PythonExe,
    [switch]$DryRun,
    [switch]$SkipPipUpgrade
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$requirementsPath = Join-Path $projectRoot "requirements.txt"

function Resolve-PythonExe {
    param(
        [string]$RequestedPythonExe
    )

    if ($RequestedPythonExe) {
        return $RequestedPythonExe
    }

    foreach ($candidate in @("py", "python")) {
        if (Get-Command $candidate -ErrorAction SilentlyContinue) {
            return $candidate
        }
    }

    throw "Python was not found. Install Python 3 first, then rerun this script."
}

function Format-Command {
    param(
        [string]$Command,
        [string[]]$Arguments
    )

    $parts = @($Command) + ($Arguments | ForEach-Object {
        if ($_ -match "\s") {
            '"' + $_ + '"'
        } else {
            $_
        }
    })

    return ($parts -join " ")
}

function Invoke-Step {
    param(
        [string]$Description,
        [string]$Command,
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "==> $Description"
    Write-Host ("    " + (Format-Command -Command $Command -Arguments $Arguments))

    if ($DryRun) {
        return
    }

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed with exit code ${LASTEXITCODE}: $Description"
    }
}

if (-not (Test-Path $requirementsPath)) {
    throw "requirements.txt was not found at: $requirementsPath"
}

$pythonCommand = Resolve-PythonExe -RequestedPythonExe $PythonExe

Write-Host "Project root: $projectRoot"
Write-Host "Using Python command: $pythonCommand"

Invoke-Step -Description "Show Python interpreter" -Command $pythonCommand -Arguments @(
    "-c",
    "import sys; print(sys.executable)"
)

if (-not $SkipPipUpgrade) {
    Invoke-Step -Description "Upgrade pip" -Command $pythonCommand -Arguments @(
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip"
    )
}

Invoke-Step -Description "Install project dependencies" -Command $pythonCommand -Arguments @(
    "-m",
    "pip",
    "install",
    "-r",
    $requirementsPath
)

Write-Host ""
if ($DryRun) {
    Write-Host "Dry run complete. No packages were installed."
} else {
    Write-Host "Install complete."
}
Write-Host "Next step: run '$pythonCommand ollama_expert_bench.py' from $projectRoot"
