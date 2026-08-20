@echo off
REM Start the YOLO12 detection web app on Windows.
REM Creates .venv and installs dependencies on first run.
setlocal
cd /d "%~dp0"

if not exist ".venv" (
  echo ==^> Creating virtualenv ^(.venv^)
  py -3 -m venv .venv || python -m venv .venv
  .venv\Scripts\python.exe -m pip install --upgrade pip
  .venv\Scripts\pip.exe install -r requirements.txt
)

if "%PORT%"=="" set PORT=8000
if "%HOST%"=="" set HOST=127.0.0.1
echo ==^> http://%HOST%:%PORT%
.venv\Scripts\python.exe -m uvicorn backend.app:app --host %HOST% --port %PORT%
