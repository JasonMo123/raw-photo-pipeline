"""
NAFNet 架構定義（獨立抽出自 https://github.com/megvii-research/NAFNet
basicsr/models/archs/NAFNet_arch.py 與 arch_util.py，僅保留推論(eval)所需部分）。

之所以獨立抽出而不是直接 import 官方 repo，是因為官方 repo 內建了自己
fork 的 basicsr 套件（basicsr.models.archs.*），會跟 requirements.txt 裡
pip 安裝的官方 basicsr 套件（gfpgan/realesrgan 依賴的那個）路徑衝突。

對應 checkpoint：models/NAFNet-SIDD-width64.pth
建構參數對照 NAFNet repo 的 options/test/SIDD/NAFNet-width64.yml：
    width=64, enc_blk_nums=[2,2,4,8], middle_blk_num=12, dec_blk_nums=[2,2,2,2]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCAContext:
    """
    讓切塊(tiled)推論時，每個NAFBlock的channel attention(SCA)改用「整張圖」
    算出的全域平均，而不是各自只看自己那一小塊圖塊的局部平均。

    背景：NAFBlock裡的SCA用nn.AdaptiveAvgPool2d(1)對輸入做全域平均池化，
    訓練時網路看到的是完整照片，這個「全域」平均一直有代表性；但切塊推論
    每次只餵一小塊進去，等於每個圖塊各自算出一個不同的「假全域」統計值，
    導致內容差異大的相鄰圖塊(例如一塊近乎全黑、隔壁有大片高對比線稿)算出
    差異懸殊的縮放係數，拼回去後在圖塊邊界產生明顯可見的色調/亮度落差
    (棋盤格狀色塊損壞)。

    用法：先用collect模式對縮小後的整張圖跑一次forward，依36個NAFBlock的
    執行順序把各自算出的SCA值存起來；之後每個圖塊用apply模式跑forward時，
    照相同順序把快取的全域SCA值套用進去，不再各自重算。
    """

    def __init__(self):
        self.mode = "collect"
        self.values = []
        self._idx = 0

    def start_collect(self):
        self.mode = "collect"
        self.values = []
        self._idx = 0

    def start_apply(self):
        self.mode = "apply"
        self._idx = 0

    def store(self, val: torch.Tensor):
        self.values.append(val.detach())

    def next_value(self) -> torch.Tensor:
        val = self.values[self._idx]
        self._idx += 1
        return val


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, groups=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, groups=1, bias=True),
        )

        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, sca_context: "SCAContext | None" = None):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)

        if sca_context is not None and sca_context.mode == "apply":
            sca_val = sca_context.next_value().to(device=x.device, dtype=x.dtype)
        else:
            sca_val = self.sca(x)
            if sca_context is not None and sca_context.mode == "collect":
                sca_context.store(sca_val)

        x = x * sca_val
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=(), dec_blk_nums=()):
        super().__init__()

        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, groups=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, sca_context: "SCAContext | None" = None):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            for blk in encoder:
                x = blk(x, sca_context=sca_context)
            encs.append(x)
            x = down(x)

        for blk in self.middle_blks:
            x = blk(x, sca_context=sca_context)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            for blk in decoder:
                x = blk(x, sca_context=sca_context)

        x = self.ending(x)
        x = x + inp
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))


def load_nafnet_sidd_width64(checkpoint_path: str, device: str) -> NAFNet:
    """建立跟 NAFNet-SIDD-width64.pth 對應的架構並載入權重"""
    model = NAFNet(img_channel=3, width=64, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["params"] if isinstance(ckpt, dict) and "params" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model
