@echo off
REM ============================================================
REM  CreditGuard - run the complete Spark Big Data pipeline
REM  Steps 1-5 + 7 (about 12 minutes on a 16 GB laptop).
REM  Needs: .venv (pip install -r backend\requirements.txt),
REM         Java 17, C:\hadoop\bin\winutils.exe + hadoop.dll,
REM         data\creditcard.csv and data\sparkov\fraudTrain.csv + fraudTest.csv
REM ============================================================
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe

echo.
echo [1/7] Ingestion and cleaning...
%PY% spark_pipeline\s01_ingest_clean.py || goto :error
echo.
echo [2/7] EDA and statistics...
%PY% spark_pipeline\s02_eda.py || goto :error
echo.
echo [3/7] Feature engineering...
%PY% spark_pipeline\s03_features.py || goto :error
echo.
echo [4/7] Training 5 models on 2 datasets (slowest step)...
%PY% spark_pipeline\s04_train_models.py || goto :error
echo.
echo [5/7] Demo samples for the website...
%PY% spark_pipeline\s04b_demo_samples.py || goto :error
echo.
echo [6/7] Evaluation charts...
%PY% spark_pipeline\s05_evaluate.py || goto :error
echo.
echo [7/7] Power BI export...
%PY% spark_pipeline\s07_export_powerbi.py || goto :error

echo.
echo ===== Pipeline finished. Charts: reports\figures   Power BI data: powerbi\data =====
pause
exit /b 0

:error
echo.
echo ***** A step failed - read the error message above. *****
pause
exit /b 1
