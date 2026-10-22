@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\pythonw.exe" (
  start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0训练界面.pyw"
) else (
  start "" pythonw "%~dp0训练界面.pyw"
)
