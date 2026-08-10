"""
Image-Adaptive-3DLUT 調色模組（獨立重新實作，不依賴官方 repo 的 trilinear_cpp CUDA擴充套件）

官方 repo (https://github.com/HuiZeng/Image-Adaptive-3DLUT) 的三線性插值是用
自己編譯的CUDA擴充套件(trilinear_cpp)做的，但本機nvcc(11.0)跟torch編譯用的
CUDA版本(12.1)對不上，編譯會直接失敗(RuntimeError: detected CUDA version
mismatch)。官方README自己也提到可以改用torch.nn.functional.grid_sample
取代(https://github.com/HuiZeng/Image-Adaptive-3DLUT/issues/14)，
這裡就是用grid_sample重新實作同樣的三線性查表邏輯，數學上等價、不需要編譯
任何東西，也不受本機CUDA工具鏈版本限制。

模型結構(Classifier)照抄官方 models_x.py 的 Classifier class；
LUT權重來自 pretrained_models/sRGB/{LUTs.pth,classifier.pth}
（這兩個檔案本身很小，已經包在git repo裡，不需要另外下載）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def discriminator_block(in_filters, out_filters, normalization=False):
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers


class Classifier(nn.Module):
    """對照 Image-Adaptive-3DLUT/models_x.py 的 Classifier"""

    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256), mode="bilinear"),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32, normalization=True),
            *discriminator_block(32, 64, normalization=True),
            *discriminator_block(64, 128, normalization=True),
            *discriminator_block(128, 128),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 3, 8, padding=0),
        )

    def forward(self, img_input):
        return self.model(img_input)


def apply_3dlut(lut: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """
    用 grid_sample 對 3D LUT 做三線性插值查表，等價於官方trilinear_cpp的CUDA kernel。

    lut:   (3, dim, dim, dim)，維度順序為 [輸出色彩通道, B索引, G索引, R索引]
    image: (N, 3, H, W)，數值範圍 [0,1] 的 sRGB 圖片
    回傳:  (N, 3, H, W)，套用LUT後的圖片

    軸對應是對照官方trilinear_kernel.cu反推出來的，不是隨意假設：
    CUDA kernel算flat index用 id = r_id + g_id*dim + b_id*dim*dim，
    也就是r_id stride=1(最快變化)、b_id stride=dim*dim(最慢變化)。
    PyTorch tensor (3,dim,dim,dim)是C-contiguous，最後一個維度(W軸)
    才是記憶體上最快變化的軸，所以「最後一維(W)=R、中間維(H)=G、
    第一個空間維(D)=B」，跟一般直覺「第一維是R」剛好相反。
    （這裡曾經因為軸對應弄反導致R/B色頻互換，整張圖膚色偏藍紫色，
    已對照kernel原始碼修正。）
    """
    dim = lut.shape[-1]
    n, c, h, w = image.shape

    volume = lut.unsqueeze(0).expand(n, -1, -1, -1, -1)  # (N,3,dim,dim,dim) as (N,C,D=B,H=G,W=R)

    r = image[:, 0:1]
    g = image[:, 1:2]
    b = image[:, 2:3]
    # grid_sample的座標順序是(x,y,z)對應input的(W,H,D)=(R,G,B)
    x = (r.clamp(0, 1) * 2 - 1)
    y = (g.clamp(0, 1) * 2 - 1)
    z = (b.clamp(0, 1) * 2 - 1)
    grid = torch.cat([x, y, z], dim=1).permute(0, 2, 3, 1).unsqueeze(1)  # (N,1,H,W,3)

    out = F.grid_sample(volume, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.squeeze(2)  # (N,3,H,W)


class LUTBundle:
    def __init__(self, classifier: Classifier, lut0: torch.Tensor, lut1: torch.Tensor, lut2: torch.Tensor):
        self.classifier = classifier
        self.lut0 = lut0
        self.lut1 = lut1
        self.lut2 = lut2

    def generate_lut(self, image: torch.Tensor) -> torch.Tensor:
        pred = self.classifier(image).squeeze()
        return pred[0] * self.lut0 + pred[1] * self.lut1 + pred[2] * self.lut2

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        lut = self.generate_lut(image)
        return apply_3dlut(lut, image)


def load_lut_bundle(lut_pretrained_path: str, lut_classifier_path: str, device: str) -> LUTBundle:
    classifier = Classifier()
    classifier.load_state_dict(torch.load(lut_classifier_path, map_location="cpu"))
    classifier.eval().to(device)

    luts = torch.load(lut_pretrained_path, map_location="cpu")
    lut0 = luts["0"]["LUT"].to(device) if "LUT" in luts["0"] else luts["0"].to(device)
    lut1 = luts["1"]["LUT"].to(device) if "LUT" in luts["1"] else luts["1"].to(device)
    lut2 = luts["2"]["LUT"].to(device) if "LUT" in luts["2"] else luts["2"].to(device)

    return LUTBundle(classifier, lut0, lut1, lut2)
