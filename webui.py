"""
RAW Photo Pipeline - WebUI

啟動後會在瀏覽器開啟一個本地網頁介面，選輸入/輸出資料夾、調整參數、
按開始就能批次處理，即時看進度。
"""
import sys
import traceback
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))

from core.pipeline_config import load_config
from core.pipeline import run_pipeline, scan_raw_files

WORK_DIR = Path(__file__).parent / "work"


def check_environment():
    """啟動時檢查GPU/模型檔案是否齊全，回傳給使用者看得懂的訊息"""
    messages = []
    try:
        import torch
        if torch.cuda.is_available():
            messages.append(f"✅ 偵測到顯卡：{torch.cuda.get_device_name(0)}")
        else:
            messages.append("⚠️ 沒有偵測到 NVIDIA 顯卡，會用 CPU 運算，速度會非常慢（可能一張要好幾分鐘到十幾分鐘）")
    except Exception as e:
        messages.append(f"❌ PyTorch 載入失敗：{e}")

    try:
        cfg = load_config()
        checks = [
            ("降噪模型 (NAFNet)", cfg["paths"]["nafnet_model"]),
            ("調色模型 (3D LUT)", cfg["paths"]["lut_pretrained"]),
            ("exiftool", cfg["paths"]["exiftool_exe"]),
        ]
        for name, path in checks:
            if Path(path).exists():
                messages.append(f"✅ {name} 已就緒")
            else:
                messages.append(f"❌ 找不到 {name}：{path}（請執行 scripts/download_models.py）")
    except Exception as e:
        messages.append(f"❌ 讀取設定檔失敗：{e}")

    return "\n".join(messages)


def preview_folder(input_dir):
    if not input_dir or not Path(input_dir).exists():
        return "請輸入有效的資料夾路徑"
    try:
        files = scan_raw_files(input_dir)
        return f"找到 {len(files)} 張 RAW 照片" if files else "這個資料夾（含子資料夾）裡沒有找到 RAW 照片"
    except Exception as e:
        return f"掃描失敗：{e}"


def run_batch(input_dir, output_dir, quality, enable_upscale, mirror_structure, progress=gr.Progress()):
    if not input_dir or not Path(input_dir).exists():
        yield "❌ 輸入資料夾不存在", ""
        return
    if not output_dir:
        yield "❌ 請指定輸出資料夾", ""
        return

    try:
        cfg = load_config()
        cfg["output"]["jpeg_quality"] = int(quality)
        cfg["processing"]["upscale"]["enabled"] = bool(enable_upscale)

        log_lines = []
        last_status = ""

        for update in run_pipeline(input_dir, output_dir, cfg, str(WORK_DIR), mirror_structure=bool(mirror_structure)):
            stage = update.get("stage")

            if stage == "scan":
                log_lines.append(update["message"])
            elif stage == "init":
                log_lines.append(update["message"])
            elif stage == "resume":
                log_lines.append(update["message"])
            elif stage == "processing":
                progress(update["percent"] / 100, desc=f"{update['done']}/{update['total']}")
                mark = "✓" if update["last_status"] == "success" else "✗"
                line = f"[{update['done']}/{update['total']}] {mark} {update['last_file']}"
                if update["last_error"]:
                    line += f"  錯誤：{update['last_error']}"
                log_lines.append(line)
                eta = f"，預估剩餘 {update['eta_hours']} 小時" if update["eta_hours"] else ""
                last_status = f"進度：{update['percent']}%（{update['done']}/{update['total']}，失敗 {update['failed']}）{eta}"
            elif stage == "done":
                log_lines.append(update["message"])
                last_status = f"✅ {update['message']}"

            yield last_status, "\n".join(log_lines[-50:])

    except Exception:
        yield "❌ 發生錯誤", traceback.format_exc()


with gr.Blocks(title="RAW Photo Pipeline") as demo:
    gr.Markdown("# 📷 RAW Photo Pipeline")
    gr.Markdown("批次對 RAW 照片做 AI 降噪 + 自動白平衡/曝光 + AI 調色（選用：放大）。")

    with gr.Row():
        env_status = gr.Textbox(label="環境檢查", value=check_environment(), interactive=False, lines=5)

    with gr.Row():
        with gr.Column():
            input_dir = gr.Textbox(label="輸入資料夾（會遞迴掃描所有RAW檔案）", placeholder=r"例如 D:\我的照片")
            preview_btn = gr.Button("掃描這個資料夾", size="sm")
            preview_result = gr.Textbox(label="掃描結果", interactive=False)
        with gr.Column():
            output_dir = gr.Textbox(label="輸出資料夾", placeholder=r"例如 D:\我的照片_處理後")
            quality = gr.Slider(label="JPEG 品質", minimum=70, maximum=100, value=95, step=1)
            enable_upscale = gr.Checkbox(label="放大（較慢，需要額外下載的模型）", value=False)
            mirror_structure = gr.Checkbox(label="輸出時保留原本的子資料夾結構", value=True)

    start_btn = gr.Button("開始處理", variant="primary")
    status_box = gr.Textbox(label="狀態", interactive=False)
    log_box = gr.Textbox(label="處理紀錄", interactive=False, lines=15, max_lines=15)

    preview_btn.click(preview_folder, inputs=[input_dir], outputs=[preview_result])
    start_btn.click(
        run_batch,
        inputs=[input_dir, output_dir, quality, enable_upscale, mirror_structure],
        outputs=[status_box, log_box],
    )

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True)
