"""
批次處理主流程（通用版）：掃描輸入資料夾 -> 對每張RAW照片跑
解碼->降噪->調色->(選用)放大->品質驗證->輸出JPEG->複製EXIF。

架構重點：CPU解碼用一個小的背景行程池「永遠領先GPU處理進度幾張」，解碼
結果直接在記憶體裡透過IPC傳給GPU處理，不寫暫存檔到硬碟——避免大量照片
同時解碼佔用數百GB~數TB硬碟空間的問題。

以generator方式實作，每處理完一張yield一次進度，方便WebUI或CLI即時顯示。
"""

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from .raw_decode import prepare_image
from .denoise import load_denoise_model, denoise
from .color_grade import load_color_grade_model, color_grade
from .upscale import face_aware_upscale
from .exif_utils import copy_exif
from .validate import check_tile_seams

RAW_EXTENSIONS = {".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2",
                   ".raf", ".orf", ".ori", ".rw2", ".dng", ".pef", ".ptx", ".srw",
                   ".x3f", ".3fr", ".mrw", ".kdc", ".dcr", ".erf", ".mef", ".mos", ".raw"}

LOOKAHEAD = 3


def scan_raw_files(input_dir: str, recursive: bool = True) -> list:
    input_path = Path(input_dir)
    pattern = "**/*" if recursive else "*"
    files = [p for p in input_path.glob(pattern) if p.is_file() and p.suffix.lower() in RAW_EXTENSIONS]
    return sorted(files)


def _prepare_one(raw_path: str, wb_strength: float, max_ev: float) -> np.ndarray:
    return prepare_image(raw_path, wb_strength, max_ev)


class DoneTracker:
    """記錄哪些檔案已經成功處理過，用來斷點續跑"""

    def __init__(self, tracker_path: Path):
        self.path = tracker_path
        self.done_set = self._load()

    def _load(self) -> set:
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                return set(json.load(f))
        return set()

    def is_done(self, key: str) -> bool:
        return key in self.done_set

    def mark_done(self, key: str):
        self.done_set.add(key)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(sorted(self.done_set), f, ensure_ascii=False, indent=2)


def run_pipeline(input_dir: str, output_dir: str, cfg: dict, work_dir: str, mirror_structure: bool = True):
    """
    產生器：每處理完一張(或跳過已完成的)就yield一次進度dict。

    input_dir:  來源資料夾(遞迴掃描所有RAW檔案)
    output_dir: 輸出資料夾
    work_dir:   放tracker/log/暫存的工作目錄
    mirror_structure: True的話輸出JPEG會鏡射輸入資料夾底下的相對路徑結構，
                       False則全部平鋪輸出到output_dir
    """
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    tmp_dir = work_path / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tracker = DoneTracker(work_path / "done.json")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_files = scan_raw_files(input_dir)
    total = len(all_files)
    yield {"stage": "scan", "total": total, "message": f"掃描到 {total} 張RAW照片"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    yield {"stage": "init", "message": f"使用裝置：{device}" + ("" if device == "cuda" else "（未偵測到NVIDIA GPU，會用CPU運算，速度非常慢）")}

    wb_strength = cfg["processing"]["auto_tone"]["wb_strength"]
    max_ev = cfg["processing"]["auto_tone"]["max_exposure_compensation_ev"]
    tile_size = cfg["processing"]["denoise"]["tile_size"]
    quality = cfg["output"]["jpeg_quality"]
    preserve_exif = cfg["output"]["preserve_exif"]
    upscale_enabled = cfg["processing"]["upscale"]["enabled"]

    denoise_model = load_denoise_model(cfg["paths"]["nafnet_model"], device) if cfg["processing"]["denoise"]["enabled"] else None
    lut_model = load_color_grade_model(cfg["paths"]["lut_pretrained"], cfg["paths"]["lut_classifier"], device) if cfg["processing"]["color_grade"]["enabled"] else None

    face_detector = None
    if upscale_enabled and cfg["processing"]["upscale"]["face_aware"]:
        from .face_detect import FaceDetector
        face_detector = FaceDetector(cfg["paths"]["yunet_model"], cfg["processing"]["upscale"]["face_detection_confidence"])

    # 過濾掉已完成的
    pending = []
    already_done = 0
    for raw_path in all_files:
        rel = raw_path.relative_to(input_path)
        out_path = (output_path / rel).with_suffix(".jpg") if mirror_structure else (output_path / (rel.stem + ".jpg"))
        key = str(rel)
        if tracker.is_done(key) or out_path.exists():
            already_done += 1
            continue
        pending.append((raw_path, rel, out_path, key))

    yield {"stage": "resume", "total": total, "already_done": already_done, "pending": len(pending),
           "message": f"斷點續跑：{already_done} 張先前已完成，剩下 {len(pending)} 張"}

    success, failed = 0, 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=LOOKAHEAD) as executor:
        it = iter(pending)
        in_flight = []

        def submit_next():
            entry = next(it, None)
            if entry is None:
                return None
            raw_path, rel, out_path, key = entry
            fut = executor.submit(_prepare_one, str(raw_path), wb_strength, max_ev)
            return (raw_path, rel, out_path, key, fut)

        for _ in range(LOOKAHEAD):
            e = submit_next()
            if e:
                in_flight.append(e)

        while in_flight:
            raw_path, rel, out_path, key, fut = in_flight.pop(0)
            e = submit_next()
            if e:
                in_flight.append(e)

            out_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                image = fut.result()

                if denoise_model is not None:
                    image = denoise(denoise_model, image, device, tile_size=tile_size)

                if lut_model is not None:
                    image = color_grade(lut_model, image, device)

                if upscale_enabled:
                    image = face_aware_upscale(image, face_detector, cfg, str(tmp_dir))
                else:
                    image = (image / 257).astype(np.uint8)

                seam_problems = check_tile_seams(image, tile_size)
                if seam_problems:
                    raise ValueError(f"輸出品質驗證失敗，偵測到圖塊接縫異常: {seam_problems[:3]}")

                iio.imwrite(out_path, image, quality=quality)

                if preserve_exif:
                    copy_exif(cfg["paths"]["exiftool_exe"], str(raw_path), str(out_path))

                tracker.mark_done(key)
                success += 1
                status = "success"
                error = None

            except Exception as e:
                failed += 1
                status = "failed"
                error = str(e)

            elapsed = time.time() - start_time
            done_count = already_done + success
            rate = success / elapsed * 3600 if elapsed > 0 and success > 0 else None
            remaining = total - done_count - failed
            eta_hours = (remaining / (success / elapsed)) / 3600 if rate and success > 0 else None

            yield {
                "stage": "processing", "total": total, "done": done_count, "failed": failed,
                "remaining": remaining, "percent": round(done_count / total * 100, 2) if total else 0,
                "last_file": raw_path.name, "last_status": status, "last_error": error,
                "elapsed_hours": round(elapsed / 3600, 2), "rate_per_hour": round(rate, 1) if rate else None,
                "eta_hours": round(eta_hours, 1) if eta_hours else None,
            }

    yield {"stage": "done", "total": total, "success": success, "failed": failed,
           "already_done": already_done, "message": f"完成。成功 {success}，失敗 {failed}"}
