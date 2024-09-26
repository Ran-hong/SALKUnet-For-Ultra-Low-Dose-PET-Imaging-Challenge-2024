import torch.nn as nn
from collections import OrderedDict
import torch


class UNet3d_deep(nn.Module):
    """
    Unet3d implement
    """

    def __init__(self, in_channels, out_channels, init_features=16):
        super(UNet3d_deep, self).__init__()
        self.features = init_features
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder1 = UNet3d_deep._block(self.in_channels, self.features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3d_deep._block(self.features, self.features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3d_deep._block(self.features * 2, self.features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3d_deep._block(self.features * 4, self.features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder5 = UNet3d_deep._block(self.features * 8, self.features * 16, name="enc5")
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder6 = UNet3d_deep._block(self.features * 16, self.features * 32, name="enc6")
        self.pool6 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet3d_deep._block(self.features * 32, self.features * 64, name="bottleneck")

        self.upconv6 = nn.ConvTranspose3d(self.features * 64, self.features * 32, kernel_size=2, stride=2)
        self.decoder6 = UNet3d_deep._block((self.features * 32) * 2, self.features * 32, name="dec6")
        self.upconv5 = nn.ConvTranspose3d(self.features * 32, self.features * 16, kernel_size=2, stride=2)
        self.decoder5 = UNet3d_deep._block((self.features * 16) * 2, self.features * 16, name="dec5")
        self.upconv4 = nn.ConvTranspose3d(self.features * 16, self.features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet3d_deep._block((self.features * 8) * 2, self.features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(self.features * 8, self.features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet3d_deep._block((self.features * 4) * 2, self.features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(self.features * 4, self.features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet3d_deep._block((self.features * 2) * 2, self.features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(self.features * 2, self.features, kernel_size=2, stride=2)
        self.decoder1 = UNet3d_deep._block(self.features * 2, self.features, name="dec1")
        self.conv = nn.Conv3d(in_channels=self.features, out_channels=self.out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))
        enc6 = self.encoder6(self.pool5(enc5))

        bottleneck = self.bottleneck(self.pool6(enc6))

        dec6 = self.upconv6(bottleneck)
        dec6 = torch.cat((dec6, enc6), dim=1)
        dec6 = self.decoder6(dec6)
        dec5 = self.upconv5(dec6)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.decoder5(dec5)
        dec4 = self.upconv4(dec5)
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
    

if __name__ == "__main__":
    model = UNet3d_deep(in_channels=1, out_channels=1)
    x = torch.randn(1, 1, 256, 256, 256)
    y = model(x)
    pass 