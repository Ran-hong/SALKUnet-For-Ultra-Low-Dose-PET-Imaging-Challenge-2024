import torch.nn as nn
import torch

class ConvBn3d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups),
            nn.BatchNorm3d(out_channels),
        )

class ConvFFN(nn.Module):
    def __init__(self, in_channels, mlp_ratio=4):
        super().__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.conv1 = ConvBn3d(in_channels=in_channels, out_channels=mlp_ratio * in_channels, kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU()
        self.conv2 = ConvBn3d(in_channels=mlp_ratio * in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x = self.bn(x)
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return x
    
class LKBlock(nn.Module):
    def __init__(self, lkSize, skSize, in_channels):
        # param : lkSize and skSize must be odd.
        super().__init__()
        self.perBn = nn.BatchNorm3d(in_channels)
        self.perConv = ConvBn3d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.lkConv = ConvBn3d(in_channels=in_channels, out_channels=in_channels, kernel_size=lkSize, padding=lkSize//2, groups=in_channels)
        self.lkBn = nn.BatchNorm3d(in_channels)
        self.skConv = ConvBn3d(in_channels=in_channels, out_channels=in_channels, kernel_size=skSize, padding=skSize//2, groups=in_channels)
        self.skBn = nn.BatchNorm3d(in_channels)
        self.postConv = ConvBn3d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x = self.perBn(x)
        x = self.perConv(x)
        h1 = self.lkBn(self.lkConv(x))
        h2 = self.skBn(self.skConv(x))
        x = h1 + h2
        x = self.postConv(x)
        return x
    
class RepLKBlock(nn.Module):
    def __init__(self, in_channels, mlp_ratio, lkSize, skSize):
        super().__init__()
        self.LKBlock = LKBlock(lkSize, skSize, in_channels)
        self.FFN = ConvFFN(in_channels, mlp_ratio)

    def forward(self, x):
        h = self.LKBlock(x)
        x = h + x
        h = self.FFN(x)
        x = h + x
        return x
        
class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBn3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = ConvBn3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.act2 = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return x
    
class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = ConvBn3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = ConvBn3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels)
        self.act2 = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.act1(self.conv1(self.up(x)))
        x = self.act2(self.conv2(x))
        return x

    
class RepLKUnet3d(nn.Module):
    def __init__(self, 
                 in_channels = 1, 
                 out_channels = 1, 
                 mlp_ratio = 4,
                 features = [16, 32, 64, 128],
                 lkSize = [31, 27, 21, 17],
                 skSize = [5, 5, 5, 5],
                 ):
        super().__init__()
        self.block_init = nn.Sequential(
            ConvBn3d(in_channels=in_channels, out_channels=features[0], kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            ConvBn3d(in_channels=features[0], out_channels=features[0], kernel_size=3, stride=1, padding=1, groups=features[0]),
            nn.ReLU(inplace=True),
            ConvBn3d(in_channels=features[0], out_channels=features[0], kernel_size=1, stride=1, padding=0),
        )

        self.encoder0 = nn.Sequential(
            RepLKBlock(features[0], mlp_ratio, lkSize[0], skSize[0]),
            # RepLKBlock(features[0], mlp_ratio, lkSize[0], skSize[0]),
        )
        self.down0 = DownSampling(features[0], features[1])
        self.encoder1 = nn.Sequential(
            RepLKBlock(features[1], mlp_ratio, lkSize[1], skSize[1]),
            # RepLKBlock(features[1], mlp_ratio, lkSize[1], skSize[1]),
        )
        self.down1 = DownSampling(features[1], features[2])
        self.encoder2 = nn.Sequential(
            RepLKBlock(features[2], mlp_ratio, lkSize[2], skSize[2]),
            # RepLKBlock(features[2], mlp_ratio, lkSize[2], skSize[2]),
        )
        self.down2 = DownSampling(features[2], features[3])

        self.mid = nn.Sequential(
            ConvBn3d(in_channels=features[3], out_channels=features[3], kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            ConvBn3d(in_channels=features[3], out_channels=features[3], kernel_size=3, stride=1, padding=1, groups=features[3]),
            nn.ReLU(inplace=True),
            ConvBn3d(in_channels=features[3], out_channels=features[3], kernel_size=1, stride=1, padding=0),
        )

        self.up1 = UpSampling(features[3], features[2])
        self.decoder1 = nn.Sequential(
            ConvBn3d(in_channels=features[2] * 2, out_channels=features[2], kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            RepLKBlock(features[2], mlp_ratio, lkSize[2], skSize[2]),
            # RepLKBlock(features[2], mlp_ratio, lkSize[2], skSize[2]),
        )
        self.up2 = UpSampling(features[2], features[1])
        self.decoder2 = nn.Sequential(
            ConvBn3d(in_channels=features[1] * 2, out_channels=features[1], kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            RepLKBlock(features[1], mlp_ratio, lkSize[1], skSize[1]),
            # RepLKBlock(features[1], mlp_ratio, lkSize[1], skSize[1]),
        )
        self.up3 = UpSampling(features[1], features[0])
        self.decoder3 = nn.Sequential(
            ConvBn3d(in_channels=features[0] * 2, out_channels=features[0], kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            RepLKBlock(features[0], mlp_ratio, lkSize[0], skSize[0]),
            # RepLKBlock(features[0], mlp_ratio, lkSize[0], skSize[0]),
        )

        self.block_out = nn.Sequential(
            ConvBn3d(in_channels=features[0] * 2, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            ConvBn3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
            nn.ReLU(inplace=True),
            ConvBn3d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.block_init(x)
        en1 = self.down0(self.encoder0(x))
        en2 = self.down1(self.encoder1(en1))
        mid = self.down2(self.encoder2(en2))
        mid = self.mid(mid)
        dn1 = torch.cat((self.up1(mid), en2), dim=1)
        dn2 = torch.cat((self.up2(self.decoder1(dn1)), en1), dim=1)
        dn3 = torch.cat((self.up3(self.decoder2(dn2)), x), dim=1)

        out = self.block_out(dn3)
        return out


if __name__ == "__main__":
    import os  
    os.environ["CUDA_VISIBLE_DEVICES"]="1"  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RepLKUnet3d().to(device)
    x = torch.randn(1, 1, 112, 112, 112).to(device)
    y = model(x)
    pass