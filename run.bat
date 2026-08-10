@echo off
if not exist ".venv\Scripts\python.exe" (
    echo 找不到虛擬環境，請先執行 install.bat。
    pause
    exit /b 1
)
.venv\Scripts\python.exe webui.py
