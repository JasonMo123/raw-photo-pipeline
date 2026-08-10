"""
AI 降噪（NAFNet）。大圖用「重疊+加權融合」的方式切塊處理，避免整張圖一次
進模型把顯卡記憶體塞爆，同時避免圖塊邊界出現接縫（業界標準tiled inference
作法，Real-ESRGAN等工具的tile模式也是這樣處理）。
"""

import numpy as np
import torch

from .nafnet_arch import load_nafnet_sidd_width64, SCAContext


def load_denoise_model(model_path: str, device: str):
    return load_nafnet_sidd_width64(model_path, device)


def _edge_ramp(length: int, pad_before: int, pad_after: int, device, dtype) -> torch.Tensor:
    """
    一維權重曲線：中間是1.0，只有在「實際有padding的那一側」(代表那一側
    有鄰居圖塊可以融合)才往0衰減；圖片真正的邊緣(沒有鄰居)維持1.0。
    """
    w = torch.ones(length, device=device, dtype=dtype)
    if pad_before > 0:
        w[:pad_before] = torch.linspace(0, 1, pad_before, device=device, dtype=dtype)
    if pad_after > 0:
        w[length - pad_after:] = torch.linspace(1, 0, pad_after, device=device, dtype=dtype)
    return w


def _compute_global_sca_context(model, tensor: torch.Tensor, max_dim: int = 768) -> SCAContext:
    """
    修正棋盤格色塊損壞：NAFBlock內的channel attention(SCA)用全域平均池化，
    訓練時網路看到的是完整照片，這個「全域」平均才有代表性；但切塊推論
    每次只餵一小塊進去，內容差異大的相鄰圖塊(例如大面積平坦背景搭配小面積
    高對比細節)會算出差異懸殊的SCA縮放係數，導致拼回去後出現明顯可見的
    色塊(詳見nafnet_arch.SCAContext的說明)。

    做法：先把整張圖縮小到一次就能完整跑完的尺寸，正常跑一次NAFNet，
    快取每個NAFBlock在「看得到全貌」情況下算出的SCA值；之後每個圖塊
    推論時改套用這組快取的全域值，不再各自重算。用area模式縮圖是因為
    它本質上就是平均池化，最貼近這裡需要保留的「全域平均統計」。
    """
    _, c, h, w = tensor.shape
    scale = min(1.0, max_dim / max(h, w))
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    small = torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode="area")

    ctx = SCAContext()
    ctx.start_collect()
    with torch.inference_mode():
        model(small, sca_context=ctx)
    return ctx


def _tiled_inference(model, tensor: torch.Tensor, tile_size: int, tile_pad: int = 32) -> torch.Tensor:
    """把大圖切成小塊分別跑模型，用重疊+加權融合拼回去"""
    _, c, h, w = tensor.shape
    if h <= tile_size and w <= tile_size:
        with torch.inference_mode():
            result = model(tensor)
        if not torch.isfinite(result).all():
            raise ValueError("降噪輸出含有NaN/Inf，判定為失敗")
        return result

    sca_context = _compute_global_sca_context(model, tensor)

    padded_size = tile_size + 2 * tile_pad

    coords = []
    for y0 in range(0, h, tile_size):
        y1 = min(y0 + tile_size, h)
        py0, py1 = max(y0 - tile_pad, 0), min(y1 + tile_pad, h)
        pad_top, pad_bottom_side = y0 - py0, py1 - y1
        for x0 in range(0, w, tile_size):
            x1 = min(x0 + tile_size, w)
            px0, px1 = max(x0 - tile_pad, 0), min(x1 + tile_pad, w)
            pad_left, pad_right_side = x0 - px0, px1 - x1
            coords.append((py0, py1, px0, px1, pad_top, pad_bottom_side, pad_left, pad_right_side))

    output_accum = torch.zeros_like(tensor)
    weight_accum = torch.zeros((1, 1, h, w), device=tensor.device, dtype=tensor.dtype)

    for entry in coords:
        py0, py1, px0, px1, pad_top, pad_bottom_side, pad_left, pad_right_side = entry
        tile = tensor[:, :, py0:py1, px0:px1]
        pad_bottom = padded_size - tile.shape[2]
        pad_right = padded_size - tile.shape[3]
        if pad_bottom > 0 or pad_right > 0:
            # replicate(邊緣值延伸)沒有reflect的「padding量不能超過原尺寸」限制
            tile = torch.nn.functional.pad(tile, (0, pad_right, 0, pad_bottom), mode="replicate")

        sca_context.start_apply()
        with torch.inference_mode():
            out = model(tile, sca_context=sca_context)

        th, tw = py1 - py0, px1 - px0
        out_tile = out[:1, :, :th, :tw]

        wy = _edge_ramp(th, pad_top, pad_bottom_side, tensor.device, tensor.dtype)
        wx = _edge_ramp(tw, pad_left, pad_right_side, tensor.device, tensor.dtype)
        w_tile = (wy.view(-1, 1) * wx.view(1, -1)).clamp_min(1e-3)

        output_accum[:, :, py0:py1, px0:px1] += out_tile * w_tile
        weight_accum[:, :, py0:py1, px0:px1] += w_tile

    result = output_accum / weight_accum
    if not torch.isfinite(result).all():
        raise ValueError("降噪輸出含有NaN/Inf，判定為失敗")
    return result


def denoise(model, image: np.ndarray, device: str, tile_size: int = 512) -> np.ndarray:
    """image: uint16 (H,W,3)，數值範圍0~65535。回傳同格式。"""
    tensor = torch.from_numpy(image.astype(np.float32) / 65535.0).permute(2, 0, 1).unsqueeze(0).to(device)
    output = _tiled_inference(model, tensor, tile_size)
    output = output.clamp(0, 1).squeeze(0).permute(1, 2, 0)
    return (output * 65535.0).round().clamp(0, 65535).cpu().numpy().astype(np.uint16)
