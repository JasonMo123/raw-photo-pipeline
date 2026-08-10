"""用exiftool把原始RAW檔的EXIF/GPS/拍攝日期複製到最終輸出的JPEG"""

import subprocess
from pathlib import Path


def copy_exif(exiftool_exe: str, source_raw: str, target_jpeg: str, logger=None) -> bool:
    if not Path(exiftool_exe).exists():
        if logger:
            logger.warning(f"找不到exiftool執行檔：{exiftool_exe}，略過EXIF複製。")
        return False

    cmd = [exiftool_exe, "-TagsFromFile", source_raw, "-all:all", "-overwrite_original", target_jpeg]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if logger:
            logger.warning(f"EXIF複製失敗：{source_raw} - {result.stderr.strip()}")
        return False
    return True
