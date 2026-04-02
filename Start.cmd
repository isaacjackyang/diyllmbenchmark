@echo off
setlocal EnableExtensions

if /I "%~1"=="/?" goto :usage
if /I "%~1"=="-h" goto :usage
if /I "%~1"=="--help" goto :usage

cd /d "%~dp0" || (
    echo Failed to enter project folder:
    echo %~dp0
    pause
    exit /b 1
)

set "ENTRY_SCRIPT=%CD%\expert_LLM_benchmark.py"
set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"
set "EXIT_CODE=0"

if not exist "%ENTRY_SCRIPT%" (
    echo Main entry script was not found:
    echo %ENTRY_SCRIPT%
    goto :fail
)

if exist "%VENV_PYTHON%" (
    echo Starting expert_LLM_benchmark with .venv Python...
    "%VENV_PYTHON%" "%ENTRY_SCRIPT%" %*
    set "EXIT_CODE=%ERRORLEVEL%"
    goto :finish
)

where py >nul 2>&1
if not errorlevel 1 (
    echo .venv Python was not found. Falling back to py...
    py "%ENTRY_SCRIPT%" %*
    set "EXIT_CODE=%ERRORLEVEL%"
    goto :finish
)

where python >nul 2>&1
if not errorlevel 1 (
    echo .venv Python was not found. Falling back to python...
    python "%ENTRY_SCRIPT%" %*
    set "EXIT_CODE=%ERRORLEVEL%"
    goto :finish
)

echo Python was not found.
echo Run install.ps1 first, or install Python and create .venv.
goto :fail

:finish
if "%EXIT_CODE%"=="0" exit /b 0

echo.
echo Benchmark exited with code %EXIT_CODE%.
goto :fail_with_code

:fail
set "EXIT_CODE=1"

:fail_with_code
echo.
echo Start.cmd failed.
echo Hint: powershell -ExecutionPolicy Bypass -File .\install.ps1
pause
exit /b %EXIT_CODE%

:usage
echo Usage:
echo   Start.cmd [arguments forwarded to expert_LLM_benchmark.py]
echo.
echo Launch order:
echo   1. .venv\Scripts\python.exe
echo   2. py
echo   3. python
echo.
echo If the virtual environment is missing, run:
echo   powershell -ExecutionPolicy Bypass -File .\install.ps1
exit /b 0
