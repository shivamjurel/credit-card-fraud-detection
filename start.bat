@echo off
REM ============================================================
REM  CreditGuard - one-click start
REM  Opens two windows (backend + frontend) and then the browser.
REM  To stop the project: close both black windows.
REM ============================================================

echo Starting CreditGuard backend (FastAPI)...
start "CreditGuard Backend" cmd /k "cd /d "%~dp0backend" && ..\.venv\Scripts\python.exe -m uvicorn main:app --reload"

echo Starting CreditGuard frontend (React)...
start "CreditGuard Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo Waiting for the servers to start (Spark needs about 20 seconds)...
timeout /t 20 /nobreak >nul

start http://localhost:5173
