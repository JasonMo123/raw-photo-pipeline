# RAW Photo Pipeline
### AI 全自動 RAW 相片批次處理工具 · Automated AI RAW Photo Processing Pipeline

批次處理 RAW 相片：AI 降噪、自動白平衡與曝光校正、AI 調色，並提供選用的
畫面放大功能。內建網頁介面，無需撰寫程式或使用命令列。

完全在本機執行（建議搭配 Windows + NVIDIA 顯卡），照片不會上傳至任何伺服器。

## 功能 Features

- **AI 降噪**：採用 [NAFNet](https://github.com/megvii-research/NAFNet)，
  對高 ISO 雜訊有良好抑制效果；搭配切塊（tiled）推論，大尺寸照片也能在
  消費級顯卡的 VRAM 限制下處理，並修正了常見的圖塊接縫與色塊瑕疵（詳見
  「已知問題與修正記錄」）。
- **自動白平衡與曝光**：Gray-world 白平衡演算法，搭配自動色階伸展。
- **AI 調色**：採用 [Image-Adaptive-3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT)，
  依照片內容動態生成 3D LUT，而非套用固定濾鏡。
- **選用的畫面放大**：採用 [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
  （ncnn-vulkan 版本，免安裝額外的 Python 深度學習套件）；偵測到人臉的
  區域改用較保守的放大強度，降低過度銳化與變形。
- **EXIF 保留**：拍攝日期、GPS 座標、相機資訊等原始 metadata 會完整複製
  至輸出的 JPEG。
- **斷點續跑**：處理中斷（當機、手動停止）後重新執行，會自動略過已完成
  的照片，無需重跑整批。
- **支援格式**：CR2 / CR3 / CRW（Canon）、NEF / NRW（Nikon）、ARW / SRF /
  SR2（Sony）、RAF（Fujifilm）、ORF（Olympus）、RW2（Panasonic）、DNG、
  PEF / PTX（Pentax）、SRW（Samsung）、X3F（Sigma）等主流機型格式。

## 需求 Requirements

- Windows 10 / 11
- Python 3.10 或 3.11
- [Git for Windows](https://git-scm.com/download/win)（下載模型時需要）
- NVIDIA 顯卡，建議 6GB 以上 VRAM（無顯卡亦可運作，但改用 CPU 運算，
  速度可能慢上數十倍，單張照片需時數分鐘至十餘分鐘）

## 安裝 Installation

```bash
git clone https://github.com/JasonMo123/raw-photo-pipeline.git
cd raw-photo-pipeline
install.bat
```

`install.bat` 會自動建立獨立的 Python 虛擬環境（`.venv`）、依顯卡狀況安裝
對應版本的 PyTorch、下載降噪／調色／人臉偵測所需的模型權重（約數百 MB，
首次安裝需數分鐘）。整個安裝過程不會影響系統既有的 Python 環境或全域套件。

## 使用 Usage

```bash
run.bat
```

會自動於瀏覽器開啟網頁介面：

1. 上方顯示環境檢查結果（顯卡偵測狀態、模型是否齊全）
2. 填入輸入資料夾（遞迴掃描所有子資料夾內的 RAW 檔案）與輸出資料夾
3. 按「掃描這個資料夾」確認找到的照片數量
4. 視需求調整 JPEG 品質、是否放大、是否保留原始子資料夾結構
5. 按「開始處理」，即時檢視處理進度與每張照片的成功／失敗狀態

處理中斷後，重新執行 `run.bat`、填入相同的輸入／輸出資料夾並按下開始，
即會自動略過已處理完成的照片。

## 設定檔 Configuration

進階參數位於 `config.yaml`，包含降噪圖塊大小（`tile_size`）、白平衡強度、
最大自動曝光補償、放大倍率等。一般使用無需調整，WebUI 上可設定的選項
（JPEG 品質、是否放大等）已涵蓋多數常見需求。

## 已知限制 Limitations

- **未內建 GFPGAN 專業人臉修復**：另一款常見的人臉放大模型 GFPGAN，其
  授權條款限定「僅供研究用途」，與本專案期望所有人（含商業用途）皆可
  自由使用的目標相牴觸，故刻意未予內建，詳見
  [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。放大功能改採較
  單純的 Real-ESRGAN，人臉區域會混合傳統放大結果以降低變形機率，但效果
  不及專門的人臉修復模型。
- **無 NVIDIA 顯卡時處理速度顯著較慢**：所有 AI 步驟皆仰賴 GPU 加速，
  CPU-only 模式僅適合驗證流程是否正常，不建議用於大量照片處理。
- **大尺寸照片需要一定處理時間**：以 6GB VRAM 顯卡處理一張 3000 萬像素
  照片（降噪＋調色）約需一分鐘，實際時間依顯卡效能與照片解析度而異。

## 已知問題與修正記錄 Notable Fixes

開發過程中發現並修正過幾個實際的影像品質問題，記錄於此供有興趣了解細節、
或欲貢獻程式碼者參考：

- **圖塊邊界接縫**：切塊（tiled）推論若僅以硬裁切拼接圖塊，NAFNet 這類
  深層網路的實際感受野遠大於 padding 範圍，會導致圖塊邊界輸出不連續，
  放大檢視可見明顯接縫。修正方式為圖塊輸出改採重疊區域加權融合（業界
  標準 tiled inference 作法）。
- **棋盤格狀色塊瑕疵**：即使解決接縫問題，內容差異極端的照片（例如大面積
  純色背景搭配小面積高對比細節）仍可能出現明顯的棋盤格狀色塊。根因是
  NAFNet 內部的 channel attention（SCA）模組採用全域平均池化——訓練時
  網路看到的是完整照片，此「全域」平均才具代表性；但切塊推論每次僅輸入
  一小塊區域，導致相鄰圖塊各自算出差異懸殊的縮放係數。修正方式是先對
  縮小後的完整照片執行一次推論，快取各層的全域統計值，供所有圖塊共用，
  使行為與模型訓練時一致（詳見 `core/nafnet_arch.py` 中 `SCAContext`
  的說明）。

## 致謝 Acknowledgements

本專案整合數個既有開源 AI 模型與工具為一套完整的批次處理流程，核心研究
成果均來自以下專案的原作者：

- [NAFNet](https://github.com/megvii-research/NAFNet)（MIT License）—— 降噪
- [Image-Adaptive-3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT)（Apache 2.0）—— AI 調色
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)（BSD-3-Clause）—— 畫面放大
- [OpenCV Zoo / YuNet](https://github.com/opencv/opencv_zoo)（MIT License）—— 人臉偵測
- [LibRaw](https://www.libraw.org/) / [rawpy](https://github.com/letmaik/rawpy)（LGPL 2.1 / CDDL、MIT）—— RAW 解碼
- [exiftool](https://exiftool.org/)（Artistic License / GPL）—— EXIF 複製

完整第三方授權清單見 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。

## 授權 License

本專案原創程式碼採用 [MIT License](LICENSE)。所使用之第三方模型／工具／
函式庫各自維持其原始授權條款，詳見上方「致謝」與
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。
