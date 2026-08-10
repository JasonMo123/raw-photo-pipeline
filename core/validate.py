"""
輸出品質驗證。

只靠「程式有沒有crash」抓不到降噪圖塊處理相關的問題——有可能算出來的數值
是錯的，但程式本身完全沒有出錯。這裡專門在圖塊網格線位置(tile_size的整數
倍，也就是已知容易出問題的圖塊邊界)量測像素跳動程度，拿來跟旁邊正常區域
的變化量比較，只檢查已知的網格線位置，不會誤判草地/樹葉這種本來就有高頻
細節的正常內容。
"""

import numpy as np


def check_tile_seams(image: np.ndarray, tile_size: int, ratio_threshold: float = 6.0) -> list:
    """回傳有問題的網格線位置列表，空list代表沒偵測到異常"""
    h, w, _ = image.shape
    gray = image.mean(axis=2).astype(np.float32)
    problems = []

    for y in range(tile_size, h, tile_size):
        if y + 2 >= h or y - 2 < 0:
            continue
        boundary_jump = np.abs(gray[y] - gray[y - 1]).mean()
        baseline = np.abs(gray[y - 2] - gray[y - 1]).mean()
        if boundary_jump > ratio_threshold * max(baseline, 1.0):
            problems.append(("row", y, float(boundary_jump), float(baseline)))

    for x in range(tile_size, w, tile_size):
        if x + 2 >= w or x - 2 < 0:
            continue
        boundary_jump = np.abs(gray[:, x] - gray[:, x - 1]).mean()
        baseline = np.abs(gray[:, x - 2] - gray[:, x - 1]).mean()
        if boundary_jump > ratio_threshold * max(baseline, 1.0):
            problems.append(("col", x, float(boundary_jump), float(baseline)))

    return problems
