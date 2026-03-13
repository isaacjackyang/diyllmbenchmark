param(
    [string]$PythonExe,
    [switch]$DryRun,
    [switch]$SkipPipUpgrade
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$requirementsPath = Join-Path $projectRoot "requirements.txt"
$entryScriptPath = Join-Path $projectRoot "ollama_expert_bench.py"
$runtimeCheckCode = @'
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

module_to_package = {
    'openai': 'openai',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'questionary': 'questionary',
    'prompt_toolkit': 'prompt_toolkit',
    'requests': 'requests',
    'tabulate': 'tabulate',
    'openpyxl': 'openpyxl',
}

for module_name, package_name in module_to_package.items():
    import_module(module_name)
    try:
        package_version = version(package_name)
    except PackageNotFoundError:
        package_version = 'unknown'
    print(package_name + '==' + package_version)
'@

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

if (-not (Test-Path $entryScriptPath)) {
    throw "Main entry script was not found at: $entryScriptPath"
}

$pythonCommand = Resolve-PythonExe -RequestedPythonExe $PythonExe

Write-Host "Project root: $projectRoot"
Write-Host "Requirements file: $requirementsPath"
Write-Host "Entry script: $entryScriptPath"
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

Invoke-Step -Description "Verify runtime imports" -Command $pythonCommand -Arguments @(
    "-c",
    $runtimeCheckCode
)

Write-Host ""
if ($DryRun) {
    Write-Host "Dry run complete. No packages were installed."
} else {
    Write-Host "Install and runtime verification complete."
}
Write-Host "Next step: run '$pythonCommand ollama_expert_bench.py' from $projectRoot"
