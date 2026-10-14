@echo off
REM ============================================================
REM  CreditGuard - real-time streaming demo
REM  Start the website first (start.bat), then run this.
REM  Opens two windows:
REM    1. Spark Structured Streaming job (scores transactions, prints fraud alerts)
REM    2. Producer (sends real test transactions every 2 seconds)
REM  On the website: Dashboard -> "Live refresh" to watch numbers update.
REM  Stop: close both windows.
REM ============================================================
cd /d "%~dp0"

start "CreditGuard Streaming (Spark)" cmd /k ".venv\Scripts\python.exe spark_pipeline\s06_streaming.py"

echo Waiting 25 seconds for Spark Streaming to start...
timeout /t 25 /nobreak >nul

start "CreditGuard Producer" cmd /k ".venv\Scripts\python.exe spark_pipeline\s06_stream_producer.py --rows 1500 --batch 50 --interval 2"
