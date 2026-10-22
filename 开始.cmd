@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" -m smallai %*
) else (
  python -m smallai %*
)
set "smallai_exit=%errorlevel%"
if "%~1"=="" pause
exit /b %smallai_exit%
