@echo off
cd /d "%~dp0"
echo Depth of Field Studio - http://127.0.0.1:7860
start "" powershell -WindowStyle Hidden -Command "Start-Sleep 3; Start-Process 'http://127.0.0.1:7860'"
".\venv\Scripts\python.exe" app.py
pause
