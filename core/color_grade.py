"""AI 調色（Image-Adaptive-3DLUT）。全解析度直接跑，不用切塊（運算量很小）。"""

import numpy as np
import torch

from .lut_color_grade import load_lut_bundle


def load_color_grade_model(lut_pretrained_path: str, lut_classifier_path: str, device: str):
    return load_lut_bundle(lut_pretrained_path, lut_classifier_path, device)


def color_grade(model_bundle, image: np.ndarray, device: str) -> np.ndarray:
    """image: uint16 (H,W,3)，數值範圍0~65535。回傳同格式。"""
    tensor = torch.from_numpy(image.astype(np.float32) / 65535.0).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model_bundle(tensor)
    if not torch.isfinite(output).all():
        raise ValueError("調色輸出含有NaN/Inf，判定為失敗")
    output = output.clamp(0, 1).squeeze(0).permute(1, 2, 0)
    return (output * 65535.0).round().clamp(0, 65535).cpu().numpy().astype(np.uint16)
