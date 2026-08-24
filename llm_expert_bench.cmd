@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"
set "INSTALL_SCRIPT=%~dp0install.ps1"

if not exist "%VENV_PYTHON%" goto bootstrap_venv

"%VENV_PYTHON%" -c "import openai, pandas, matplotlib, questionary, prompt_toolkit, requests, tabulate, openpyxl" >nul 2>&1
if errorlevel 1 goto repair_venv
if "%LLM_BENCH_CHECK_ONLY%"=="1" goto check_only_ok
goto run_venv

:bootstrap_venv
echo.
echo [SETUP] Creating the project .venv and installing dependencies...
echo The first setup may take several minutes.
goto install_venv

:repair_venv
echo.
echo [REPAIR] The project .venv is incomplete. Reinstalling requirements...

:install_venv
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%INSTALL_SCRIPT%"
if errorlevel 1 goto install_failed

"%VENV_PYTHON%" -c "import openai, pandas, matplotlib, questionary, prompt_toolkit, requests, tabulate, openpyxl" >nul 2>&1
if errorlevel 1 goto install_failed
if "%LLM_BENCH_CHECK_ONLY%"=="1" goto check_only_ok

:run_venv
"%VENV_PYTHON%" "%~dp0llm_expert_bench.py"
set "BENCH_EXIT_CODE=%ERRORLEVEL%"
if "%BENCH_EXIT_CODE%"=="0" exit /b 0

echo.
echo [ERROR] llm_expert_bench.py exited with code %BENCH_EXIT_CODE%.
echo See llm_expert_bench_crash.log for details.
echo.
pause
exit /b %BENCH_EXIT_CODE%

:check_only_ok
echo [OK] Project virtual environment is ready.
exit /b 0

:install_failed
echo.
echo [ERROR] Failed to create or repair the project .venv.
echo Run this command in PowerShell:
echo powershell -ExecutionPolicy Bypass -File .\install.ps1
echo.
pause
exit /b 1
