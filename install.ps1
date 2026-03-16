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
$venvPath = Join-Path $projectRoot ".venv"
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

function Get-VenvPythonPath {
    param(
        [string]$VirtualEnvPath
    )

    if ($env:OS -eq "Windows_NT") {
        return Join-Path $VirtualEnvPath "Scripts\python.exe"
    }

    return Join-Path $VirtualEnvPath "bin/python"
}

function Get-VenvActivatePath {
    param(
        [string]$VirtualEnvPath
    )

    if ($env:OS -eq "Windows_NT") {
        return Join-Path $VirtualEnvPath "Scripts\Activate.ps1"
    }

    return Join-Path $VirtualEnvPath "bin/activate"
}

function Format-Command {
    param(
        [string]$Command,
        [string[]]$Arguments
    )

    $formattedCommand = if ($Command -match "\s") {
        '"' + $Command + '"'
    } else {
        $Command
    }

    $parts = @($formattedCommand) + ($Arguments | ForEach-Object {
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
$venvPython = Get-VenvPythonPath -VirtualEnvPath $venvPath
$venvActivatePath = Get-VenvActivatePath -VirtualEnvPath $venvPath

Write-Host "Project root: $projectRoot"
Write-Host "Requirements file: $requirementsPath"
Write-Host "Entry script: $entryScriptPath"
Write-Host "Virtual environment path: $venvPath"
Write-Host "Using base Python command: $pythonCommand"

Invoke-Step -Description "Show base Python interpreter" -Command $pythonCommand -Arguments @(
    "-c",
    "import sys; print(sys.executable)"
)

if (-not (Test-Path $venvPath)) {
    Invoke-Step -Description "Create project virtual environment" -Command $pythonCommand -Arguments @(
        "-m",
        "venv",
        $venvPath
    )
} else {
    Write-Host ""
    Write-Host "==> Reuse existing virtual environment"
    Write-Host "    $venvPath"
}

if (-not $DryRun -and -not (Test-Path $venvPython)) {
    throw "Virtual environment Python was not found at: $venvPython"
}

Write-Host "Using virtual environment Python: $venvPython"

if (-not $SkipPipUpgrade) {
    Invoke-Step -Description "Upgrade pip inside virtual environment" -Command $venvPython -Arguments @(
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip"
    )
}

Invoke-Step -Description "Install project dependencies into virtual environment" -Command $venvPython -Arguments @(
    "-m",
    "pip",
    "install",
    "-r",
    $requirementsPath
)

Invoke-Step -Description "Verify runtime imports inside virtual environment" -Command $venvPython -Arguments @(
    "-c",
    $runtimeCheckCode
)

Write-Host ""
if ($DryRun) {
    Write-Host "Dry run complete. No virtual environment was created and no packages were installed."
} else {
    Write-Host "Virtual environment setup and runtime verification complete."
}
Write-Host ("Next step (PowerShell): & '" + $venvActivatePath + "'")
Write-Host ("Or run directly: & '" + $venvPython + "' '" + $entryScriptPath + "'")
