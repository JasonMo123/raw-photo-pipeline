"""
RAW 解碼 + 自動白平衡/曝光。

使用 rawpy(LibRaw 的 Python 封裝）解碼，輸出 sRGB gamma 的16-bit影像，
符合後續 AI 模型（在一般 sRGB 照片上訓練）預期的色彩空間。
"""

import numpy as np
import rawpy


def decode_raw(raw_path: str) -> np.ndarray:
    """把RAW檔案解碼成uint16 (H,W,3) sRGB影像"""
    with rawpy.imread(raw_path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            half_size=False,
            no_auto_bright=True,  # 亮度交給 auto_levels 處理
            output_bps=16,
        )
    return rgb.astype(np.uint16)


def gray_world_white_balance(image: np.ndarray, strength: float = 0.9) -> np.ndarray:
    """Gray-World 自動白平衡：假設整張圖平均起來應該接近灰色，依此校正色偏"""
    img = image.astype(np.float32)
    means = img.reshape(-1, img.shape[2]).mean(axis=0)
    gray_mean = means.mean()
    gains = gray_mean / np.maximum(means, 1e-6)
    gains = 1.0 + strength * (gains - 1.0)
    return img * gains


def auto_levels(image: np.ndarray, max_ev_compensation: float = 1.5) -> np.ndarray:
    """自動曝光：依1%/99%百分位數拉伸對比，限制最大補償幅度避免過曝/死黑"""
    img = image.astype(np.float32)
    max_val = 65535.0

    low = np.percentile(img, 1)
    high = np.percentile(img, 99)
    if high <= low:
        return img

    current_range = high - low
    implied_ev = np.log2(max_val / max(current_range, 1e-6))
    implied_ev = np.clip(implied_ev, -max_ev_compensation, max_ev_compensation)
    effective_range = max_val / (2 ** implied_ev)
    scale = effective_range / current_range

    stretched = (img - low) * scale
    return np.clip(stretched, 0, max_val)


def prepare_image(raw_path: str, wb_strength: float = 0.9, max_ev: float = 1.5) -> np.ndarray:
    """一次跑完解碼+白平衡+自動曝光，回傳uint16 (H,W,3)"""
    image = decode_raw(raw_path)
    image = gray_world_white_balance(image, wb_strength)
    image = auto_levels(image, max_ev)
    return np.clip(image, 0, 65535).astype(np.uint16)
