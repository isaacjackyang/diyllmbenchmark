param(
    [string]$PythonExe,
    [string]$VenvPath = ".venv",
    [switch]$DryRun,
    [switch]$SkipPipUpgrade
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$requirementsPath = Join-Path $projectRoot "requirements.txt"
$entryScriptPath = Join-Path $projectRoot "llm_expert_bench.py"
$venvRoot = if ([System.IO.Path]::IsPathRooted($VenvPath)) {
    $VenvPath
} else {
    Join-Path $projectRoot $VenvPath
}
$venvPythonPath = Join-Path $venvRoot "Scripts\python.exe"
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
        if (Test-PythonExe -Candidate $RequestedPythonExe) {
            return $RequestedPythonExe
        }
        throw "The requested Python command could not run: $RequestedPythonExe"
    }

    if (Get-Command "py" -ErrorAction SilentlyContinue) {
        foreach ($versionSelector in @("-3.13", "-3.12", "-3.11", "-3.10", "-3")) {
            try {
                $launcherOutput = & py $versionSelector -c "import sys; print(sys.executable)" 2>$null
                $launcherExitCode = $LASTEXITCODE
                $resolvedPath = $launcherOutput | Select-Object -First 1
                if ($launcherExitCode -eq 0 -and $resolvedPath -and (Test-PythonExe -Candidate $resolvedPath.Trim())) {
                    return $resolvedPath.Trim()
                }
            } catch {
            }
        }
    }

    foreach ($candidate in @("python", "python3")) {
        if (Test-PythonExe -Candidate $candidate) {
            return $candidate
        }
    }

    throw "Python was not found. Install Python 3 first, then rerun this script."
}

function Test-PythonExe {
    param(
        [string]$Candidate
    )

    if (-not (Get-Command $Candidate -ErrorAction SilentlyContinue)) {
        return $false
    }

    try {
        & $Candidate -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
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

$basePythonCommand = Resolve-PythonExe -RequestedPythonExe $PythonExe

Write-Host "Project root: $projectRoot"
Write-Host "Requirements file: $requirementsPath"
Write-Host "Entry script: $entryScriptPath"
Write-Host "Virtual environment: $venvRoot"
Write-Host "Base Python: $basePythonCommand"

Invoke-Step -Description "Show base Python interpreter" -Command $basePythonCommand -Arguments @(
    "-c",
    "import sys; print(sys.executable)"
)

if (-not (Test-Path -LiteralPath $venvPythonPath -PathType Leaf)) {
    Invoke-Step -Description "Create project virtual environment" -Command $basePythonCommand -Arguments @(
        "-m",
        "venv",
        $venvRoot
    )
}

if (-not $DryRun -and -not (Test-PythonExe -Candidate $venvPythonPath)) {
    throw "The project virtual environment is invalid: $venvPythonPath"
}

$pythonCommand = $venvPythonPath
Write-Host "Runtime Python: $pythonCommand"

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
Write-Host "Next step: double-click 'llm_expert_bench.cmd', or run '.\.venv\Scripts\python.exe .\llm_expert_bench.py' from $projectRoot"
