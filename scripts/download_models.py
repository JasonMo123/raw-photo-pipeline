"""
下載所有需要的模型檔案跟外部工具。執行一次即可，已經下載過的會自動跳過。

來源與授權（詳見專案根目錄 THIRD_PARTY_LICENSES.md）：
- NAFNet-SIDD-width64.pth：megvii-research/NAFNet，MIT License
- face_detection_yunet_2023mar.onnx：opencv/opencv_zoo，MIT License
- Image-Adaptive-3DLUT：HuiZeng/Image-Adaptive-3DLUT，Apache License 2.0
  （預訓練LUT權重已經包在這個repo裡，不用另外下載）
- realesrgan-ncnn-vulkan：xinntao/Real-ESRGAN，BSD-3-Clause（選用，放大功能才需要）
- exiftool：Phil Harvey，Artistic License / GPL 雙授權（以獨立執行檔呼叫，非連結）
"""
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
TOOLS_DIR = ROOT / "tools"


def log(msg):
    print(f"[download_models] {msg}", flush=True)


def download_file(url: str, dest: Path, desc: str):
    if dest.exists():
        log(f"{desc} 已存在，略過：{dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"下載 {desc} ...")
    urllib.request.urlretrieve(url, dest)
    log(f"完成：{dest}")


def download_nafnet():
    dest = MODELS_DIR / "NAFNet-SIDD-width64.pth"
    if dest.exists():
        log(f"NAFNet 模型已存在，略過：{dest}")
        return
    log("下載 NAFNet 降噪模型（來源：Google Drive，約440MB）...")
    try:
        import gdown
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)
        import gdown
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    gdown.download(
        "https://drive.google.com/uc?id=14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR",
        str(dest), quiet=False,
    )


def download_yunet():
    dest = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
    download_file(
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        dest, "YuNet 人臉偵測模型",
    )


def clone_image_adaptive_3dlut():
    dest = TOOLS_DIR / "Image-Adaptive-3DLUT"
    if dest.exists() and (dest / "pretrained_models" / "sRGB" / "LUTs.pth").exists():
        log(f"Image-Adaptive-3DLUT 已存在，略過：{dest}")
        return
    log("clone Image-Adaptive-3DLUT repo（包含預訓練LUT權重）...")
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1",
                     "https://github.com/HuiZeng/Image-Adaptive-3DLUT.git", str(dest)], check=True)


def download_exiftool():
    dest = TOOLS_DIR / "exiftool.exe"
    if dest.exists():
        log(f"exiftool 已存在，略過：{dest}")
        return
    log("下載 exiftool（約11MB）...")
    tmp_zip = TOOLS_DIR / "_exiftool_tmp.zip"
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(
        "https://sourceforge.net/projects/exiftool/files/exiftool-13.59_64.zip/download", tmp_zip,
    )
    with zipfile.ZipFile(tmp_zip) as zf:
        zf.extractall(TOOLS_DIR / "_exiftool_extract")
    extracted_dir = next((TOOLS_DIR / "_exiftool_extract").glob("exiftool-*"))
    (extracted_dir / "exiftool(-k).exe").rename(dest)
    files_src = extracted_dir / "exiftool_files"
    files_dst = TOOLS_DIR / "exiftool_files"
    if files_src.exists() and not files_dst.exists():
        files_src.rename(files_dst)
    tmp_zip.unlink(missing_ok=True)
    import shutil
    shutil.rmtree(TOOLS_DIR / "_exiftool_extract", ignore_errors=True)
    log(f"完成：{dest}")


def download_realesrgan_ncnn(optional: bool = True):
    dest_dir = TOOLS_DIR / "realesrgan-ncnn-vulkan"
    dest_exe = dest_dir / "realesrgan-ncnn-vulkan.exe"
    if dest_exe.exists():
        log(f"realesrgan-ncnn-vulkan 已存在，略過：{dest_exe}")
        return
    log("下載 realesrgan-ncnn-vulkan（放大功能用，約45MB）...")
    tmp_zip = TOOLS_DIR / "_realesrgan_tmp.zip"
    dest_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip",
        tmp_zip,
    )
    with zipfile.ZipFile(tmp_zip) as zf:
        zf.extractall(dest_dir)
    tmp_zip.unlink(missing_ok=True)
    log(f"完成：{dest_exe}")


def main():
    log("開始下載模型與工具，已下載過的會自動跳過...")
    download_nafnet()
    download_yunet()
    clone_image_adaptive_3dlut()
    download_exiftool()
    download_realesrgan_ncnn()
    log("全部完成！")


if __name__ == "__main__":
    main()
