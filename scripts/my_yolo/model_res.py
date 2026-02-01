import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding, bias=False) # No bias for conv layers followed by batch norm
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        """
        Forward pass for the convolutional block.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, H/2, W/2).
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        """
        Forward pass for the residual block.
        Similar to a standard ResNet block.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        residual = x
        out = self.conv1(x) # extracts features
        out = self.bn1(out) # stabilizes learning
        out = self.relu(out) # introduces non-linearity
        out = self.conv2(out) # refining features
        out = self.bn2(out) # stabilizes learning
        out += residual # skip connection
        return self.relu(out) # final activation

class MyYolo_res(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Backbone: Progressive reduction of resolution (from 640 to 20)
        self.backbone = nn.Sequential(
            ConvBlock(3, 16),   # 320x320
            ConvBlock(16, 32),  # 160x160
            ConvBlock(32, 64),  # 80x80
            ConvBlock(64, 128), # 40x40
            ConvBlock(128, 256) # 20x20
        )

        # Neck: we use dropout for regularization
        self.neck = nn.Sequential(
            ResidualBlock(256),
            nn.Dropout2d(0.2),
            ResidualBlock(256),
            nn.Dropout2d(0.2),
            ResidualBlock(256) 
        )

        self.out_channels = 5 + num_classes
        self.head = nn.Conv2d(256, self.out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass for the MyYolo_res model.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 3, 640, 640).

        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, 20, 20).
        """
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x