@echo off
setlocal enabledelayedexpansion
echo ============================================================
echo   RAW Photo Pipeline - 安裝程式
echo ============================================================
echo.

REM --- 檢查 Python ---
where py >nul 2>nul
if errorlevel 1 (
    echo [錯誤] 找不到 Python。請先從 https://www.python.org/downloads/ 安裝 Python 3.10 或 3.11，
    echo         安裝時記得勾選 "Add Python to PATH"。
    pause
    exit /b 1
)

echo [1/4] 建立虛擬環境...
if not exist ".venv" (
    py -3.11 -m venv .venv 2>nul || py -3.10 -m venv .venv 2>nul || py -3 -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
    echo [錯誤] 建立虛擬環境失敗。
    pause
    exit /b 1
)

set PY=.venv\Scripts\python.exe

echo [2/4] 安裝基礎套件...
%PY% -m pip install --upgrade pip
%PY% -m pip install pyyaml tqdm gradio "numpy<2" rawpy "opencv-python>=4.9,<5" "opencv-contrib-python>=4.9,<5" Pillow imageio

echo [3/4] 偵測顯卡並安裝 PyTorch...
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo   未偵測到 NVIDIA 顯卡工具，將安裝 CPU 版本（處理速度會非常慢，
    echo   一張照片可能要好幾分鐘到十幾分鐘，只建議用來測試流程是否正常）。
    %PY% -m pip install torch==2.2.2 torchvision==0.17.2
) else (
    echo   偵測到 NVIDIA 顯卡，安裝 CUDA 版本 PyTorch...
    %PY% -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
)

echo [4/4] 下載模型與外部工具...
where git >nul 2>nul
if errorlevel 1 (
    echo [錯誤] 找不到 git，請先安裝 Git for Windows：https://git-scm.com/download/win
    pause
    exit /b 1
)
%PY% scripts\download_models.py

echo.
echo ============================================================
echo   安裝完成！執行 run.bat 開啟介面。
echo ============================================================
pause
