@echo off
chcp 65001 >nul
call "%~dp0开始.cmd" chat --checkpoint outputs/cpu_demo/refined/last.pt
pause
