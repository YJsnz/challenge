@echo off
setlocal
wsl.exe -d Ubuntu --cd "%~dp0." --exec bash scripts/launch_demo.sh %*
if errorlevel 1 (
    echo.
    echo Demo launch failed. See the message above and docs\WSL.md.
    pause
    exit /b 1
)
