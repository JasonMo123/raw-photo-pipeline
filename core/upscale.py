"""
放大（選用功能）。只用 Real-ESRGAN 的 ncnn-vulkan 執行檔（BSD-3-Clause授權，
不需要額外的Python深度學習環境），人臉區域用「傳統/AI混合降級」的保守做法
降低過度銳化/變形的風險。

注意：這裡刻意不使用GFPGAN——它依賴的StyleGAN2/DFDNet元件授權限定「僅供
研究或評測用途」，不適合公開發布給大眾自由使用的工具內建，詳見專案根目錄
THIRD_PARTY_LICENSES.md。
"""

import subprocess
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import PIL.Image

# 這裡處理的都是自己RAW檔案放大後的中繼PNG，2x放大22MP以上的照片容易超過
# Pillow預設的解壓縮炸彈保護門檻，因此關掉這個保護(僅影響本流程內部暫存讀寫)
PIL.Image.MAX_IMAGE_PIXELS = None


def upscale_ncnn_vulkan(exe_path: str, image_uint8_rgb: np.ndarray, model_name: str, tile_size: int, scale: int, tmp_dir: str) -> np.ndarray:
    """
    呼叫外部ncnn-vulkan執行檔做放大。一定要帶-s scale：不帶的話exe會用模型
    原生倍率(通常是4倍)，跟設定的放大倍率不符，也容易在高解析度照片上
    產生過大的中繼檔案。
    """
    tmp_in = Path(tmp_dir) / "_tmp_upscale_in.png"
    tmp_out = Path(tmp_dir) / "_tmp_upscale_out.png"

    iio.imwrite(tmp_in, image_uint8_rgb)
    cmd = [exe_path, "-i", str(tmp_in), "-o", str(tmp_out), "-n", model_name, "-s", str(scale), "-t", str(tile_size)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"realesrgan-ncnn-vulkan 執行失敗: {result.stderr}")

    output = iio.imread(tmp_out)
    tmp_in.unlink(missing_ok=True)
    tmp_out.unlink(missing_ok=True)
    return output


def upscale_with_face_blend_fallback(
    exe_path: str, image_uint8_rgb: np.ndarray, faces, model_name: str,
    tile_size: int, scale: int, blend_strength: float, tmp_dir: str,
) -> np.ndarray:
    """
    對偵測到人臉的區域，把「傳統Lanczos放大」跟「AI放大」依blend_strength
    混合，降低通用放大模型在臉部造成的過度銳化/變形程度。
    """
    ai_result = upscale_ncnn_vulkan(exe_path, image_uint8_rgb, model_name, tile_size, scale, tmp_dir)

    h_ratio = ai_result.shape[0] / image_uint8_rgb.shape[0]
    w_ratio = ai_result.shape[1] / image_uint8_rgb.shape[1]

    for (x, y, fw, fh, _conf) in faces:
        pad = int(max(fw, fh) * 0.2)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(image_uint8_rgb.shape[1], x + fw + pad), min(image_uint8_rgb.shape[0], y + fh + pad)

        face_crop = image_uint8_rgb[y0:y1, x0:x1]
        target_w = int((x1 - x0) * w_ratio)
        target_h = int((y1 - y0) * h_ratio)

        lanczos_face = cv2.resize(face_crop, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        ax0, ay0 = int(x0 * w_ratio), int(y0 * h_ratio)
        ax1, ay1 = ax0 + target_w, ay0 + target_h
        ai_face = ai_result[ay0:ay1, ax0:ax1]

        if ai_face.shape != lanczos_face.shape:
            lanczos_face = cv2.resize(lanczos_face, (ai_face.shape[1], ai_face.shape[0]))

        blended = (ai_face.astype(np.float32) * (1 - blend_strength) + lanczos_face.astype(np.float32) * blend_strength)
        ai_result[ay0:ay1, ax0:ax1] = np.clip(blended, 0, 255).astype(np.uint8)

    return ai_result


def face_aware_upscale(image_uint16_rgb: np.ndarray, face_detector, cfg: dict, tmp_dir: str) -> np.ndarray:
    upscale_cfg = cfg["processing"]["upscale"]

    image_8bit_rgb = (image_uint16_rgb / 257).astype(np.uint8)
    image_8bit_bgr = cv2.cvtColor(image_8bit_rgb, cv2.COLOR_RGB2BGR)

    faces = []
    if upscale_cfg["face_aware"] and face_detector is not None:
        faces = face_detector.detect(image_8bit_bgr)

    exe_path = cfg["paths"]["realesrgan_exe"]
    model_name = upscale_cfg["model_name"]
    tile_size = upscale_cfg["tile_size"]
    scale = upscale_cfg["scale_factor"]

    if not faces:
        return upscale_ncnn_vulkan(exe_path, image_8bit_rgb, model_name, tile_size, scale, tmp_dir)

    return upscale_with_face_blend_fallback(
        exe_path, image_8bit_rgb, faces, model_name, tile_size, scale,
        upscale_cfg["fallback_face_blend_strength"], tmp_dir,
    )
