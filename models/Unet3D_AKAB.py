import torch.nn as nn
from collections import OrderedDict
import torch

class LK(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pw1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.perNorm = nn.BatchNorm3d(in_channels)
        self.convl = nn.Conv3d(out_channels, out_channels, kernel_size=7, stride=1, padding=7//2, groups=out_channels, bias=False)
        self.convs = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnl = nn.BatchNorm3d(out_channels)
        self.bns = nn.BatchNorm3d(out_channels)
        self.pw2 = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.perNorm(x)
        x = self.pw1(x)
        h1 = self.bnl(self.convl(x))
        h2 = self.bns(self.convs(x))
        x = self.pw2(h1 + h2)
        x = self.act(x)
        return x


class UNet3d_AKAB(nn.Module):
    """
    Unet3d implement
    """

    def __init__(self, in_channels, out_channels, init_features=16):
        super(UNet3d_AKAB, self).__init__()
        self.features = init_features
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Adapt_intput_K = nn.Parameter(torch.ones([1]), requires_grad=True)
        self.Adapt_intput_B = nn.Parameter(torch.ones([1]), requires_grad=True)


        self.encoder1 = UNet3d_AKAB._block(self.in_channels, self.features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3d_AKAB._block(self.features, self.features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3d_AKAB._block(self.features * 2, self.features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3d_AKAB._block(self.features * 4, self.features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        # self.bottleneck = UNet3d_AKAB._LKblock(self.features * 8, self.features * 16, name="bottleneck")
        self.bottleneck = LK(self.features * 8, self.features * 16)
        self.upconv4 = nn.ConvTranspose3d(self.features * 16, self.features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet3d_AKAB._block((self.features * 8) * 2, self.features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(self.features * 8, self.features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet3d_AKAB._block((self.features * 4) * 2, self.features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(self.features * 4, self.features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet3d_AKAB._block((self.features * 2) * 2, self.features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(self.features * 2, self.features, kernel_size=2, stride=2)
        self.decoder1 = UNet3d_AKAB._block(self.features * 2, self.features, name="dec1")
        self.conv = nn.Conv3d(in_channels=self.features, out_channels=self.out_channels, kernel_size=1)

    def forward(self, x):
        x = (x * self.Adapt_intput_K) + self.Adapt_intput_B
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        out_logit = self.conv(dec1)

        return out_logit

    @staticmethod
    def _block(in_channels, features, name, prob=0.2):
        block = nn.Sequential(OrderedDict([
            (name + "conv1", nn.Conv3d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False, ),),
            (name + "norm1", nn.GroupNorm(num_groups=8, num_channels=features)),
            (name + "droupout1", nn.Dropout3d(p=prob, inplace=True)),
            (name + "relu1", nn.ReLU(inplace=True)),
            (name + "conv2", nn.Conv3d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False, ),),
            (name + "norm2", nn.GroupNorm(num_groups=8, num_channels=features)),
            (name + "droupout2", nn.Dropout3d(p=prob, inplace=True)),
            (name + "relu2", nn.ReLU(inplace=True)),
        ]))
        return block

    def _LKblock(in_channels, features, name, prob=0.2):
        block = nn.Sequential(OrderedDict([
            (name + "conv1", nn.Conv3d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=13,
                padding=13 // 2,
                bias=False, ),),
            (name + "norm1", nn.GroupNorm(num_groups=8, num_channels=features)),
            (name + "droupout1", nn.Dropout3d(p=prob, inplace=True)),
            (name + "relu1", nn.ReLU(inplace=True)),
            (name + "conv2", nn.Conv3d(
                in_channels=features,
                out_channels=features,
                kernel_size=13,
                padding=13 // 2,
                bias=False, ),),
            (name + "norm2", nn.GroupNorm(num_groups=8, num_channels=features)),
            (name + "droupout2", nn.Dropout3d(p=prob, inplace=True)),
            (name + "relu2", nn.ReLU(inplace=True)),
        ]))
        return block
    

if __name__ == "__main__":
    model = UNet3d_AKAB(in_channels=1, out_channels=1)
    x = torch.randn(1, 1, 644, 440, 440)
    y = model(x)
    print(y.shape)
    pass 