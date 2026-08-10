"""
人臉偵測模組（OpenCV YuNet）

用途：判斷一張照片裡有沒有人臉，如果有，放大階段就不要對整張圖
無腦套用同一個通用放大模型，而是把臉部區域交給專門設計來處理臉部的
GFPGAN流程，避免通用放大模型常見的「過度銳化/整臉變形/五官長歪」問題。

YuNet是OpenCV官方提供的輕量人臉偵測模型，純CPU就能跑，速度快，
不會佔用GPU資源，適合在CPU平行處理階段就先做完人臉偵測判斷。

模型下載：https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
（Claude Code請下載 face_detection_yunet_2023mar.onnx 放到 models/ 資料夾，
並確認連結是否仍然有效）
"""

from pathlib import Path

import cv2
import numpy as np


class FaceDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.6):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"找不到YuNet人臉偵測模型：{model_path}，"
                "請從 https://github.com/opencv/opencv_zoo 下載 "
                "face_detection_yunet_2023mar.onnx"
            )
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.detector = None  # 延遲初始化，因為需要先知道圖片尺寸

    def _ensure_detector(self, width: int, height: int):
        if self.detector is None or self._last_size != (width, height):
            self.detector = cv2.FaceDetectorYN.create(
                self.model_path,
                "",
                (width, height),
                score_threshold=self.confidence_threshold,
            )
            self._last_size = (width, height)

    def detect(self, image_uint8_bgr: np.ndarray):
        """
        輸入：8-bit BGR圖片（OpenCV慣用格式）
        輸出：list of (x, y, w, h, confidence)，座標是原圖尺寸下的像素位置
        """
        h, w = image_uint8_bgr.shape[:2]
        self._ensure_detector(w, h)

        _, faces = self.detector.detect(image_uint8_bgr)
        if faces is None:
            return []

        results = []
        for face in faces:
            x, y, fw, fh = face[0:4].astype(int)
            confidence = float(face[14])
            results.append((x, y, fw, fh, confidence))
        return results

    def has_face(self, image_uint8_bgr: np.ndarray) -> bool:
        return len(self.detect(image_uint8_bgr)) > 0
