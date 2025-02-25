import os
import numpy as np
import torch
import torch.nn as nn
from dataset import BirdSegmentationDataset, train_loader, val_loader

class DoubleConv(nn.Module):
    """
    Double Convolution Block:
    (Conv2d -> BatchNorm -> ReLU) * 2
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = \
            nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.double_conv(x)
        
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):


        super().__init__()

        # Downsampling Path
        self.downs = nn.ModuleList()

        # Upsampling Path
        self.ups = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Downsampling)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))

            in_channels = feature

        # Bottom (Bridge)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Decoder (Upsampling)
        for feature in reversed(features):
            # Upconvolution
            self.ups.append(
                nn.ConvTranspose2d( feature*2,
                                    feature,
                                    kernel_size=2,
                                    stride = 2)
            )
            # Double conv after concatenation
            self.ups.append(
                DoubleConv(feature*2, feature)
            )


        # Final Convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottom
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1] # reversing skip connections

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)

            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = torch.nn.functional.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x =  self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    
def test_unet():
    x = torch.randn((1, 3, 160, 160))
    model = UNET(in_channels=3, out_channels=1)
    preds = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")

if __name__ == "__main__":
    test_unet()