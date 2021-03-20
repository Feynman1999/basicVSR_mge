import numpy as np
import megengine
import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
from edit.models.common import ResBlocks, default_init_weights
from edit.models.builder import BACKBONES
import math

backwarp_tenGrid = {}

def pad_W(x):
    # 在图像右方添一列 replicate模式
    # x: [B,C,H,W]
    return F.concat([x, x[:,:,:,-1:]], axis=3)

def pad_H(x):
    # 在图像下方添一行 replicate模式
    return F.concat([x, x[:,:,-1:, :]], axis=2)

def backwarp(tenInput, tenFlow, border_mode):
    """
        CONSTANT(0)    REPLICATE
    """
    _, _, H, W = tenFlow.shape
    if str(tenFlow.shape) not in backwarp_tenGrid.keys():
        x_list = np.linspace(0., W - 1., W).reshape(1, 1, 1, W)
        x_list = x_list.repeat(H, axis=2)
        y_list = np.linspace(0., H - 1., H).reshape(1, 1, H, 1)
        y_list = y_list.repeat(W, axis=3)
        xy_list = np.concatenate((x_list, y_list), 1)  # [1,2,H,W]
        backwarp_tenGrid[str(tenFlow.shape)] = megengine.tensor(xy_list.astype(np.float32))
    return F.nn.remap(inp = tenInput, map_xy=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).transpose(0, 2, 3, 1), border_mode=border_mode)

class Basic(M.Module):
    def __init__(self, intLevel):
        super(Basic, self).__init__()
        self.netBasic = M.Sequential(
            Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), # 8=3+3+2
            M.ReLU(),
            Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            M.ReLU(),
            Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
            M.ReLU(),
            Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
            M.ReLU(),
            Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
        )

    def forward(self, tenInput):
        return self.netBasic(tenInput)

class Spynet(M.Module):
    def __init__(self, num_layers, pretrain_ckpt_path = None, blocktype = None):
        super(Spynet, self).__init__()
        assert num_layers in (1, 2, 3, 4, 5)
        self.num_layers = num_layers
        self.threshold = 8
        self.pretrain_ckpt_path = pretrain_ckpt_path
        self.blocktype = blocktype
        self.blocktype = "resblock"

        if self.blocktype == "resblock":
            basic_list = [ Basic(intLevel) for intLevel in range(num_layers) ]
            self.border_mode = "REPLICATE"
        elif self.blocktype == "shuffleblock":
            basic_list = [ Basic_Shuffle(intLevel) for intLevel in range(num_layers) ]
            self.border_mode = "REPLICATE"
        elif self.blocktype == "MobileNeXt":
            basic_list = [ Basic_CA(intLevel) for intLevel in range(num_layers) ]
            self.border_mode = "REPLICATE"
        self.netBasic = M.Sequential(*basic_list)

    def preprocess(self, tenInput):
        tenRed = (tenInput[:, 0:1, :, :] - 0.485) / 0.229
        tenGreen = (tenInput[:, 1:2, :, :] - 0.456) / 0.224
        tenBlue = (tenInput[:, 2:3, :, :] - 0.406 ) / 0.225
        return F.concat([tenRed, tenGreen, tenBlue], axis=1) # [B,3,H,W]

    def forward(self, tenFirst, tenSecond):
        tenFirst = [self.preprocess(tenFirst)]
        tenSecond = [self.preprocess(tenSecond)]

        for intLevel in range(self.num_layers - 1):
            if tenFirst[0].shape[2] >= self.threshold or tenFirst[0].shape[3] >= self.threshold:
                tenFirst.insert(0, F.avg_pool2d(inp=tenFirst[0], kernel_size=2, stride=2))
                tenSecond.insert(0, F.avg_pool2d(inp=tenSecond[0], kernel_size=2, stride=2))
        
        tenFlow = F.zeros([tenFirst[0].shape[0], 2, int(math.floor(tenFirst[0].shape[2] / 2.0)), int(math.floor(tenFirst[0].shape[3] / 2.0))])
        # print(len(tenFirst))
        for intLevel in range(len(tenFirst)): 
            # normal:  5 for training  (4*4, 8*8, 16*16, 32*32, 64*64)  5 for test  (11*20, 22*40, 45*80, 90*160, 180*320)
            # small:   3 for training  (16*16, 32*32, 64*64)       3 for test  (45*80, 90*160, 180*320)
            tenUpsampled = F.nn.interpolate(inp=tenFlow, scale_factor=2, mode='BILINEAR', align_corners=True) * 2.0
            if tenUpsampled.shape[2] != tenFirst[intLevel].shape[2]:
                tenUpsampled = pad_H(tenUpsampled)
            if tenUpsampled.shape[3] != tenFirst[intLevel].shape[3]:
                tenUpsampled = pad_W(tenUpsampled)
            tenFlow = self.netBasic[intLevel]( F.concat([tenFirst[intLevel], backwarp(tenInput=tenSecond[intLevel], tenFlow=tenUpsampled, border_mode=self.border_mode), tenUpsampled], axis=1) ) + tenUpsampled
        return tenFlow

    def init_weights(self, strict=True):
        # load ckpt from path
        if self.blocktype == "resblock":
            if self.pretrain_ckpt_path is not None:
                print("loading pretrained model for Spynet 🤡🤡🤡🤡🤡🤡...")
                state_dict = megengine.load(self.pretrain_ckpt_path)
                self.load_state_dict(state_dict, strict=strict)
            else:
                pass
        else:
            default_init_weights(self.netBasic, scale=0.2)

class PixelShuffle(M.Module):
    def __init__(self, scale=2):
        super(PixelShuffle, self).__init__()
        self.scale = scale

    def forward(self, inputs):
        # N C iH iW
        N, C, iH, iW = inputs.shape
        oH = iH * self.scale
        oW = iW * self.scale
        oC = C // (self.scale ** 2)
        # N C s s iH iW
        output = inputs.reshape(N, oC, self.scale, self.scale, iH, iW)
        # N C iH s iW s
        output = output.transpose(0, 1, 4, 3, 5, 2)
        # N C oH oW
        output = output.reshape(N, oC, oH, oW)
        return output

class PixelShufflePack(M.Module):
    def __init__(self, in_channels, out_channels, scale_factor, upsample_kernel):
        super(PixelShufflePack, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel

        self.upsample_conv = M.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        self.pixel_shuffle = PixelShuffle(scale_factor)

        self.init_weights()

    def init_weights(self):
        """Initialize weights for PixelShufflePack.
        """
        default_init_weights(self, 1, nonlinearity="leaky_relu")

    def forward(self, x):
        """Forward function for PixelShufflePack.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = self.pixel_shuffle(x)
        return x

@BACKBONES.register_module()
class BasicVSR(M.Module):
    def __init__(self, in_channels, 
                        out_channels, 
                        hidden_channels,
                        init_nums,
                        blocknums, 
                        reconstruction_blocks, 
                        upscale_factor, 
                        pretrained_optical_flow_path, 
                        flownet_layers = 5,
                        blocktype = "resblock",
                        Lambda = 0.002):
        super(BasicVSR, self).__init__()
        assert blocktype in ("resblock", "shuffleblock", "MobileNeXt")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.blocknums = blocknums
        self.upscale_factor = upscale_factor
        self.reconstruction_blocknums = reconstruction_blocks
        self.Lambda = Lambda

        self.flownet = Spynet(num_layers=flownet_layers, pretrain_ckpt_path=pretrained_optical_flow_path, blocktype = blocktype)

        self.conv1 = M.ConvRelu2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1) # need init
        self.conv2 = ResBlocks(channel_num=hidden_channels, resblock_num=init_nums, blocktype=blocktype)
        self.border_mode = "REPLICATE"
        self.feature_extracter = ResBlocks(channel_num=hidden_channels, resblock_num=blocknums, blocktype=blocktype)
        self.reconstruction = ResBlocks(channel_num=hidden_channels, resblock_num=reconstruction_blocks, blocktype=blocktype)
        self.conv3 = M.ConvRelu2d(2*hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1) # need init
        self.conv4 = M.ConvRelu2d(2*hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1) # need init
        
        self.upsample1 = PixelShufflePack(hidden_channels, hidden_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(hidden_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = M.Conv2d(64, 64, 3, 1, 1)  # need init
        self.conv_last = M.Conv2d(64, out_channels, 3, 1, 1)
        self.lrelu = M.LeakyReLU(negative_slope=0.01)

    def do_upsample(self, forward_hidden, backward_hidden):
        # 处理某一个time stamp的Hidden
        out = self.conv4(F.concat([forward_hidden, backward_hidden], axis=1))
        out = self.reconstruction(out)
        out = self.lrelu(self.upsample1(out))
        out = self.lrelu(self.upsample2(out))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        return out # [B, 3, 4*H, 4*W]

    def forward(self, hidden, flow, now_frame):
        # hidden [B, C, H, W]
        now_frame = self.conv2(self.conv1(now_frame)) # [B, C, H, W]
        mid_hidden = backwarp(hidden, flow, self.border_mode) # [B, C, H, W]
        mid_hidden = self.conv3(F.concat([now_frame, mid_hidden], axis=1))
        mid_hidden = self.feature_extracter(mid_hidden)
        return mid_hidden

    def init_weights(self, pretrained):
        self.flownet.init_weights(strict=False)

        for m in [self.conv1, self.conv3, self.conv4]:
            default_init_weights(m)
        default_init_weights(self.conv_hr, nonlinearity='leaky_relu')

    def prepare_for_layer_2(self, forward_hidden, backward_hidden):
        out = self.conv4(F.concat([forward_hidden, backward_hidden], axis=1))
        out = self.reconstruction(out)
        return out