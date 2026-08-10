# 第三方授權說明

本專案的原創程式碼採用 [MIT License](LICENSE)。但這個 pipeline 串接了幾個
外部 AI 模型、工具與函式庫，它們各自有自己的授權條款。本檔案列出所有會被
下載、內建或作為相依套件安裝的第三方元件，方便使用者（尤其是想商業使用的
使用者）在使用前確認授權相容性。

## 安裝時下載的模型權重 / 外部工具

這些檔案由 `scripts/download_models.py` 在安裝時下載到 `models/` 或
`tools/` 目錄，**不包含在 git repo 裡**，執行 `install.bat` 時才會取得。

| 項目 | 來源 | 授權 | 用途 |
|---|---|---|---|
| NAFNet-SIDD-width64.pth | [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) | MIT License | AI 降噪模型權重 |
| face_detection_yunet_2023mar.onnx | [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo) | MIT License | YuNet 人臉偵測模型 |
| Image-Adaptive-3DLUT（整個 repo，含預訓練 LUT 權重） | [HuiZeng/Image-Adaptive-3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT) | Apache License 2.0 | AI 調色（3D LUT 生成） |
| realesrgan-ncnn-vulkan（獨立執行檔） | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) release | BSD-3-Clause | 選用的一般畫面放大 |
| exiftool.exe | [Phil Harvey / exiftool.org](https://exiftool.org/) | Artistic License 或 GPL（雙授權，使用者可擇一） | 複製 EXIF/GPS/拍攝日期到輸出檔 |

exiftool 是以獨立子行程（subprocess）方式呼叫，不是連結（link）進本專案的
程式碼，屬於「執行外部程式」而非「衍生作品」，所以不會讓本專案的程式碼被
GPL 條款拘束。

## Python 相依套件（`requirements.txt`）

| 套件 | 授權 |
|---|---|
| torch / torchvision | BSD-3-Clause（PyTorch自訂條款，內容近似修改版BSD） |
| numpy | BSD-3-Clause |
| opencv-python / opencv-contrib-python | Apache License 2.0 |
| rawpy | MIT License（本身是對 **LibRaw** 的 Python 封裝） |
| Pillow | HPND License（歷史上稱PIL License，條款近似MIT） |
| imageio | BSD-2-Clause |
| gradio | Apache License 2.0 |
| pyyaml | MIT License |
| tqdm | MIT License 與 MPL-2.0 雙授權 |
| gdown（僅安裝時 `download_models.py` 用來下載NAFNet權重） | MIT License |

### 關於 LibRaw（rawpy 的底層引擎）

`rawpy` 這個 Python 套件本身是 MIT 授權，但它包裝的 **LibRaw** 函式庫採用
**LGPL 2.1 / CDDL 雙授權**（使用者可任選一種遵循）。LibRaw 是以動態連結
（dynamic linking）的方式被 rawpy 使用，這種用法下 LGPL 不會要求本專案
的程式碼也開源，但如果之後有人想把 LibRaw 換成靜態連結或修改 LibRaw 原始碼
再散布，就要另外確認 LGPL/CDDL 條款是否遵循。

## 刻意不內建的元件：GFPGAN 人臉修復路徑

原本規劃過用 **GFPGAN** 做「人臉感知放大」（偵測到人臉時用專門模型修復五官
細節），但 GFPGAN 內部依賴的 **StyleGAN2** 與 **DFDNet** 權重，其授權條款
明確寫著「僅供研究或評測用途（non-commercial, research or evaluation
purposes only）」，跟本專案想開放給所有人（含商業用途）自由使用的目標互相
衝突，所以刻意不把這條路徑放進公開版本，`requirements.txt` 裡也沒有列出
`gfpgan` / `basicsr` / `facexlib` 等相關套件。

放大功能改用單純的 **Real-ESRGAN ncnn-vulkan** 獨立執行檔（BSD-3-Clause，
上表已列出），對偵測到人臉的區域則混合傳統 Lanczos 放大結果來降低過度銳化
/變形的機率，雖然效果不如GFPGAN專業，但沒有授權疑慮。

## 商業使用前的建議

以上大多數元件都是寬鬆授權（MIT / BSD / Apache 2.0），可商用；比較需要留意
的是 **LibRaw 的 LGPL/CDDL 條款**（如果你要修改/靜態連結它）以及
**exiftool 的 GPL 選項**（如果你要把它的原始碼一起散布而非只呼叫執行檔）。
如果不確定，建議直接看各專案原始 repo 裡的 LICENSE 檔案以最新版本為準——
本檔案只反映撰寫當下（2026年）各專案的授權狀態。
