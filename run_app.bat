@echo off
title PDF Translator Launcher

echo ==================================================
echo      Starting PDF Translator System...
echo ==================================================

echo.
echo [1/2] Launching Backend Server (Port 8080)...
start "Backend Server" cmd /k "cd backend && call venv\Scripts\activate && python main.py"

echo.
echo [2/2] Launching Frontend Server...
start "Frontend Server" cmd /k "cd frontend && npm run dev"

echo.
echo ==================================================
echo      Both servers are launching!
echo      Please wait for the windows to appear.
echo.
echo      Go to: http://localhost:5173
echo ==================================================
echo.
pause
