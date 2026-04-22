import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetMasker(nn.Module):
    """v1/v2: Basic U-Net generator"""

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Conv2d(64, 32, 3, padding=1)
        self.dec1 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(self.pool(e1)))
        d2 = F.relu(self.dec2(self.up(e2)))
        if d2.size() != e1.size():
            d2 = F.interpolate(d2, size=e1.shape[2:])
        mask = torch.sigmoid(self.dec1(d2))
        return x * mask, mask


class Discriminator(nn.Module):
    """v2: Basic discriminator"""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    """v3: Residual block"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.leaky_relu(x + self.conv(x), 0.2)


class UpgradedUNet(nn.Module):
    """v3: Final model — Concat skip connection + ResBlock U-Net"""

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1), nn.LeakyReLU(0.2))
        self.res1 = ResBlock(64)
        self.down1 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.res2 = ResBlock(128)
        self.down2 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.res3 = ResBlock(256)
        self.up2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), ResBlock(128))
        self.up1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), ResBlock(64))
        self.final = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        e1 = self.res1(self.enc1(x))
        e2 = self.res2(F.leaky_relu(self.down1(e1), 0.2))
        e3 = self.res3(F.leaky_relu(self.down2(e2), 0.2))
        d2 = self.up2(e3)
        if d2.size() != e2.size():
            d2 = F.interpolate(d2, size=e2.shape[2:])
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.size() != e1.size():
            d1 = F.interpolate(d1, size=e1.shape[2:])
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        mask = torch.sigmoid(self.final(d1))
        return x * mask, mask


class UpgradedDiscriminator(nn.Module):
    """v3: Upgraded discriminator"""

    def __init__(self):
        super().__init__()

        def conv_block(in_f, out_f):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_f),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.model = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.model(x)
