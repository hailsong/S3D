import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=64, bottleneck_features=512):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = UNet._block(features * 8, bottleneck_features, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(bottleneck_features, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)
        
    def forward(self, x):
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

        # return torch.sigmoid(self.conv(dec1))
        return self.conv(dec1)
    
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
    

class UNetMod(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=64, bottleneck_features=512):
        super(UNetMod, self).__init__()

        features = init_features
        # 기존 인코더 레이어들
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 추가된 인코더 레이어들
        self.encoder5 = self._block(features * 8, features * 16, name="enc5")
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder6 = self._block(features * 16, features * 32, name="enc6")
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder7 = self._block(features * 32, features * 64, name="enc7")
        self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self._block(features * 64, bottleneck_features, name="bottleneck")

        # 선형 계층 (변경 없음)
        self.bottleneck_fc = nn.Linear(bottleneck_features * 4 * 4, 7 * 512)
        self.expand_fc = nn.Linear(7 * 512, bottleneck_features * 4 * 4)

        # 디코더 레이어들 (추가된 부분)
        self.upconv7 = nn.ConvTranspose2d(bottleneck_features, features * 64, kernel_size=2, stride=2)
        self.decoder7 = self._block(features * 64 * 2, features * 64, name="dec7")

        self.upconv6 = nn.ConvTranspose2d(features * 64, features * 32, kernel_size=2, stride=2)
        self.decoder6 = self._block(features * 32 * 2, features * 32, name="dec6")

        self.upconv5 = nn.ConvTranspose2d(features * 32, features * 16, kernel_size=2, stride=2)
        self.decoder5 = self._block(features * 16 * 2, features * 16, name="dec5")

        # 기존 디코더 레이어들
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block(features * 8 * 2, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block(features * 4 * 2, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(features * 2 * 2, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # 인코더
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))
        enc6 = self.encoder6(self.pool5(enc5))
        enc7 = self.encoder7(self.pool6(enc6))

        bottleneck = self.bottleneck(self.pool7(enc7))

        # Bottleneck 처리
        bottleneck_flat = bottleneck.view(bottleneck.size(0), -1)
        bottleneck_encoded = self.bottleneck_fc(bottleneck_flat)
        bottleneck_encoded = bottleneck_encoded.view(bottleneck.size(0), 7, 512)

        bottleneck_expanded = self.expand_fc(bottleneck_encoded.view(bottleneck.size(0), -1))
        bottleneck_expanded = bottleneck_expanded.view(bottleneck.size())

        # 디코더
        dec7 = self.upconv7(bottleneck_expanded)
        dec7 = torch.cat((dec7, enc7), dim=1)
        dec7 = self.decoder7(dec7)

        dec6 = self.upconv6(dec7)
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

        return self.conv(dec1)
    
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )


class UNetStyleDistil(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=64, bottleneck_features=512):
        super(UNetStyleDistil, self).__init__()

        # 512 x 512
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 256 * 256
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 128 * 128
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 64 * 64
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 32 * 32
        self.encoder5 = UNet._block(features * 8, features * 8, name="enc5")
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 16 * 16
        self.encoder6 = UNet._block(features * 8, features * 8, name="enc6")
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 8 * 8
        self.encoder7 = UNet._block(features * 8, features * 8, name="enc7")
        self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck (4 * 4)
        self.bottleneck = UNet._block(features * 8, bottleneck_features, name="bottleneck")
        self.downconv1x1 = nn.Conv2d(bottleneck_features, bottleneck_features, kernel_size=1, stride=1, padding=0)
        self.downfc = nn.Linear(16, 7)
        self.upfc = nn.Linear(7, 16)
        self.upconv1x1 = nn.Conv2d(bottleneck_features, bottleneck_features, kernel_size=1, stride=1, padding=0)

        # 8 * 8
        self.upconv7 = nn.ConvTranspose2d(bottleneck_features, features * 8, kernel_size=2, stride=2)
        self.decoder7 = UNet._block((features * 8) * 2, features * 8, name="dec7")

        # 16 * 16
        self.upconv6 = nn.ConvTranspose2d(features * 8, features * 8, kernel_size=2, stride=2)
        self.decoder6 = UNet._block((features * 8) * 2, features * 8, name="dec6")

        # 32 * 32
        self.upconv5 = nn.ConvTranspose2d(features * 8, features * 8, kernel_size=2, stride=2)
        self.decoder5 = UNet._block((features * 8) * 2, features * 8, name="dec5")

        # 64 * 64
        self.upconv4 = nn.ConvTranspose2d(bottleneck_features, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")

        # 128 * 128
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")

        # 256 * 256
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")

        # 512 * 512
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)
        
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))
        enc6 = self.encoder6(self.pool5(enc5))
        enc7 = self.encoder7(self.pool6(enc6))

        _bottleneck = self.bottleneck(self.pool7(enc7))
        _bottleneck = self.downconv1x1(_bottleneck)
        
        # get style vector
        b, c, w, h = _bottleneck.shape
        _bottleneck = _bottleneck.view(b, c, w * h)
        style_embed = self.downfc(_bottleneck)
        bottleneck = self.upfc(style_embed)
        bottleneck = bottleneck.view(b, c, w, h)
        bottleneck = self.upconv1x1(bottleneck)
        # bottleneck = bottleneck + _bottleneck

        dec7 = self.upconv7(bottleneck)
        dec7 = torch.cat((dec7, enc7), dim=1)
        dec7 = self.decoder7(dec7)

        dec6 = self.upconv6(dec7)
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

        # return torch.sigmoid(self.conv(dec1))
        return self.conv(dec1), style_embed
    
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
