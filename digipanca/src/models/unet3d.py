import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Two consecutive 3D convolutions with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    """3D U-Net implementation"""
    def __init__(self, in_channels=1, out_channels=5, base_channels=32):
        super().__init__()
        
        # Encoder path
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)
        
        # Decoder path
        self.up4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, 
                                      kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)
        
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, 
                                      kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 
                                      kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, 
                                      kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)
        
        # Final output layer
        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.dec4(torch.cat([self.up4(bottleneck), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))
        
        # Output segmentation map
        return self.out_conv(dec1)

# if __name__ == "__main__":
#     # TEST UNet3D
#     model = UNet3D(in_channels=1, out_channels=5)
#     # print(model)
#     x = torch.randn((1, 1, 64, 128, 128))  # Batch size 1, 1 channel, depth 64, height 128, width 128
#     y = model(x)
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Trainable params: {trainable_params:,}")
#     print(y.shape)  # Expected: (1, 5, 64, 128, 128)
#     # Print values to verify the model's output
#     print("Output values:")
#     print(y)