import torch
import torch.nn as nn
import numpy as np

# ===========================
# build model
# ===========================

def build_model(name, *args, **kwargs):
    if name in MODELS:
        return MODELS.get(name)(*args, **kwargs)
    else:
        raise NotImplementedError(f'{name} is not implemented in the MODELS list')

MODELS = {}

def add2model(cls):
    if cls.__name__ in MODELS:
        raise ValueError(f'{cls.__name__} is already in the LOSSES list')
    else:
        MODELS[cls.__name__] = cls
    return cls

# ===========================
# models
# ===========================

class ConvBlock(nn.Module):
    """ implement conv+ReLU two times """

    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        conv_relu = []
        conv_relu.append(nn.Conv2d(in_channels=in_channels, out_channels=middle_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.ReLU())
        conv_relu.append(nn.Conv2d(in_channels=middle_channels, out_channels=out_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.ReLU())
        self.conv_ReLU = nn.Sequential(*conv_relu)

    def forward(self, x):
        out = self.conv_ReLU(x)
        return out


@add2model
class UNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3):
        super().__init__()

        # encoder
        self.left_conv_1 = ConvBlock(
            in_channels=in_nc, middle_channels=64, out_channels=64)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_2 = ConvBlock(
            in_channels=64, middle_channels=128, out_channels=128)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_3 = ConvBlock(
            in_channels=128, middle_channels=256, out_channels=256)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_4 = ConvBlock(
            in_channels=256, middle_channels=512, out_channels=512)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_5 = ConvBlock(
            in_channels=512, middle_channels=1024, out_channels=1024)

        # decoder
        self.deconv_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.right_conv_1 = ConvBlock(
            in_channels=1024, middle_channels=512, out_channels=512)

        self.deconv_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.right_conv_2 = ConvBlock(
            in_channels=512, middle_channels=256, out_channels=256)

        self.deconv_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.right_conv_3 = ConvBlock(
            in_channels=256, middle_channels=128, out_channels=128)

        self.deconv_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.right_conv_4 = ConvBlock(
            in_channels=128, middle_channels=64, out_channels=64)
        self.right_conv_5 = nn.Conv2d(
            in_channels=64, out_channels=out_nc, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # padding to the dimension that can be divided by 16
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/16)*16-h)
        paddingRight = int(np.ceil(w/16)*16-w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        # 1：encode
        feature_1 = self.left_conv_1(x)
        feature_1_pool = self.pool_1(feature_1)

        feature_2 = self.left_conv_2(feature_1_pool)
        feature_2_pool = self.pool_2(feature_2)

        feature_3 = self.left_conv_3(feature_2_pool)
        feature_3_pool = self.pool_3(feature_3)

        feature_4 = self.left_conv_4(feature_3_pool)
        feature_4_pool = self.pool_4(feature_4)

        feature_5 = self.left_conv_5(feature_4_pool)

        # 2：decode
        de_feature_1 = self.deconv_1(feature_5)
        temp = torch.cat((feature_4, de_feature_1), dim=1)
        de_feature_1_conv = self.right_conv_1(temp)

        de_feature_2 = self.deconv_2(de_feature_1_conv)
        temp = torch.cat((feature_3, de_feature_2), dim=1)
        de_feature_2_conv = self.right_conv_2(temp)

        de_feature_3 = self.deconv_3(de_feature_2_conv)

        temp = torch.cat((feature_2, de_feature_3), dim=1)
        de_feature_3_conv = self.right_conv_3(temp)

        de_feature_4 = self.deconv_4(de_feature_3_conv)
        temp = torch.cat((feature_1, de_feature_4), dim=1)
        de_feature_4_conv = self.right_conv_4(temp)

        out = self.right_conv_5(de_feature_4_conv)
        out = out[..., :h, :w]

        return out
