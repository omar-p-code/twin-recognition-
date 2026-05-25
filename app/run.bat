@echo off
title Twin Recognition – Server + localtunnel

:: 1. Install localtunnel if not present
echo Checking localtunnel...
call npm list -g localtunnel >nul 2>&1
if errorlevel 1 (
    echo Installing localtunnel globally...
    call npm install -g localtunnel
)

:: 2. Start the Python server in a new window
echo Starting Python server...
start "PythonServer" cmd /k "cd /d %~dp0 && uvicorn server:app --host 127.0.0.1 --port 8000"

:: 3. Wait for server to start
timeout /t 3 /nobreak >nul

:: 4. Start localtunnel with a fixed subdomain (change "twin-recognition" if needed)
echo Starting localtunnel...
start "Localtunnel" cmd /k "lt --port 8000 --subdomain twin-recognition"

echo.
echo ==================================================
echo  Python server: http://localhost:8000
echo  Public URL:    https://twin-recognition.loca.lt
echo.
echo  Keep this window open. Press any key to stop everything.
echo ==================================================
pause >nul

:: Cleanup
taskkill /f /im "cmd.exe" /fi "WINDOWTITLE eq PythonServer*" >nul 2>&1
taskkill /f /im "cmd.exe" /fi "WINDOWTITLE eq Localtunnel*" >nul 2>&1