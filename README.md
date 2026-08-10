# RAW Photo Pipeline

用 AI 批次處理 RAW 相片的免費工具：降噪、自動白平衡/曝光、AI 調色，
選用的畫面放大。內建網頁介面（WebUI），不需要寫程式或用命令列。

跑在你自己的電腦上（需要 Windows + NVIDIA 顯卡效果最好），照片不會上傳到
任何地方。

## 功能

- **AI 降噪**：[NAFNet](https://github.com/megvii-research/NAFNet)，對高
  ISO 雜訊有感的降噪效果，且做了切塊(tiled)推論，大圖也能在一般消費級顯卡
  的 VRAM 限制下處理，不會有圖塊邊界接縫或色塊問題（見下方「已知問題與修
  正」）。
- **自動白平衡 / 曝光**：Gray-world 白平衡 + 自動對比extend。
- **AI 調色**：[Image-Adaptive-3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT)，
  依照片內容自動生成 3D LUT 調色，不是套用固定濾鏡。
- **選用的畫面放大**：[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
  (ncnn-vulkan 版本，免額外裝 Python 深度學習套件)，對偵測到人臉的區域改用
  較保守的放大強度，降低過度銳化/變形。
- **保留 EXIF**：拍攝日期、GPS、相機資訊等原始metadata會複製到輸出的JPEG。
- **可斷點續跑**：處理到一半中斷（當機、手動停止）重新執行會自動跳過已完成
  的照片，不用重跑整批。
- **支援的 RAW 格式**：CR2 / CR3 / CRW（Canon）、NEF / NRW（Nikon）、ARW /
  SRF / SR2（Sony）、RAF（Fujifilm）、ORF（Olympus）、RW2（Panasonic）、
  DNG、PEF / PTX（Pentax）、SRW（Samsung）、X3F（Sigma）等主流機型格式。

## 需求

- Windows 10/11
- Python 3.10 或 3.11
- [Git for Windows](https://git-scm.com/download/win)（下載模型時需要）
- NVIDIA 顯卡 + 6GB 以上 VRAM（建議；沒有顯卡也能跑，但會改用CPU，
  速度可能慢上數十倍，單張照片要等好幾分鐘到十幾分鐘）

## 安裝

```bash
git clone https://github.com/<your-username>/raw-photo-pipeline.git
cd raw-photo-pipeline
install.bat
```

`install.bat` 會自動：建立獨立的 Python 虛擬環境（`.venv`）、依你有沒有
NVIDIA 顯卡安裝對應版本的 PyTorch、下載降噪/調色/人臉偵測所需的模型權重
（約數百MB，第一次安裝需要幾分鐘）。整個安裝過程不會動到你電腦上其他
Python環境或全域套件。

## 使用

```bash
run.bat
```

會自動在瀏覽器開啟一個網頁介面：

1. 上方顯示環境檢查結果（有沒有偵測到顯卡、模型是否齊全）
2. 填入輸入資料夾（會遞迴掃描所有子資料夾裡的RAW檔案）跟輸出資料夾
3. 按「掃描這個資料夾」確認找到幾張照片
4. 需要的話調整 JPEG 品質、是否放大、是否保留原本的子資料夾結構
5. 按「開始處理」，即時看處理進度跟每張照片的成功/失敗狀態

處理中斷後（關掉視窗、電腦重開機）重新執行 `run.bat`、填一樣的輸入/輸出
資料夾再按開始，會自動跳過已經處理完成的照片。

## 設定檔

進階參數在 `config.yaml`，包含降噪的圖塊大小(`tile_size`)、白平衡強度、
最大自動曝光補償、放大倍率等。一般使用不需要修改，WebUI上能調整的選項
（JPEG品質、是否放大等）已經涵蓋大多數常見需求。

## 已知限制

- **沒有 GFPGAN 專業人臉修復**：另一款熱門的人臉放大模型 GFPGAN 因為授權
  條款限定「僅供研究用途」，跟本專案想讓所有人（含商業用途）自由使用的目標
  衝突，所以刻意沒有內建，詳見 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。
  放大功能改用較單純的 Real-ESRGAN，人臉區域會混合傳統放大結果降低變形機率，
  但效果不如專門的人臉修復模型。
- **沒有 NVIDIA 顯卡的話速度非常慢**：所有 AI 步驟都靠 GPU 加速，CPU-only
  模式僅供測試流程是否正常，不建議用來跑大量照片。
- **大圖需要一定的處理時間**：例如 6GB VRAM 的顯卡處理一張 3000 萬像素的
  照片(降噪+調色)大約需要一分鐘左右，實際時間依顯卡效能與照片解析度而定。

## 已知問題與修正記錄

開發過程中發現並修正過幾個真實的圖片品質問題，記錄下來給有興趣了解細節、
或是之後想貢獻程式碼的人參考：

- **圖塊邊界接縫**：切塊(tiled)推論如果圖塊間只用硬裁切拼接，NAFNet這種
  深層網路的感受野遠大於padding，放大看得出方塊狀接縫。修正方式是圖塊輸出
  改用重疊+加權融合(業界標準tiled inference作法)。
- **棋盤格色塊損壞**：即使解決了接縫問題，內容差異極端的照片（例如大面積
  純色背景搭配小面積高對比細節）仍可能出現明顯的棋盤格狀色塊。根因是
  NAFNet內部的channel attention (SCA) 用全域平均池化，訓練時網路看到的是
  完整照片、這個「全域」平均才有代表性，但切塊推論每次只餵一小塊進去，
  不同圖塊各自算出差異懸殊的縮放係數。修正方式是先對縮小後的整張圖跑一次
  推論，把每層的全域統計值快取起來，讓每個圖塊共用同一組數值，跟模型訓練
  時的行為保持一致（詳見 `core/nafnet_arch.py` 裡 `SCAContext` 的說明）。

## 致謝 / 使用的開源專案

本專案是把幾個現有的開源AI模型跟工具串接成一個好用的批次處理流程，
真正困難的AI研究都是以下專案的原作者完成的：

- [NAFNet](https://github.com/megvii-research/NAFNet)（MIT License）—— 降噪
- [Image-Adaptive-3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT)（Apache 2.0）—— AI調色
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)（BSD-3-Clause）—— 畫面放大
- [OpenCV Zoo / YuNet](https://github.com/opencv/opencv_zoo)（MIT License）—— 人臉偵測
- [LibRaw](https://www.libraw.org/) / [rawpy](https://github.com/letmaik/rawpy)（LGPL 2.1 / CDDL、MIT）—— RAW解碼
- [exiftool](https://exiftool.org/)（Artistic License / GPL）—— EXIF複製

完整的第三方授權清單見 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。

## 授權

本專案原創程式碼採用 [MIT License](LICENSE)。使用到的第三方模型/工具/
函式庫各自維持自己的授權條款，詳見上方「致謝」與
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。
