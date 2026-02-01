import torch
import torch.nn as nn

# similar structure to YOLO but simplified (15M parameters, instead of 2.6M of YOLO11nano, \
# which uses depthwise separable convolutions and other advanced techniques)

class ConvBlock(nn.Module):
    """
    Base convolutional block: Conv (stride 2) -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride=2, padding=padding, bias=False) 
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class NeckBlock(nn.Module):
    """
    Deep convolutional block to process features at 20x20 resolution
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class MyYolo(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Ridotto il numero di canali: 16 -> 32 -> 64 -> 128 -> 256
        self.backbone = nn.Sequential(
            ConvBlock(3, 16),
            ConvBlock(16, 32), 
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256)
        )

        # Neck più leggero (256 canali invece di 512)
        self.neck = nn.Sequential(
            NeckBlock(256, 256),
            NeckBlock(256, 256)
        )

        self.out_channels = 5 + num_classes
        self.head = nn.Conv2d(256, self.out_channels, kernel_size=1)

    def forward(self, x):
        # x: [Batch, 3, 640, 640]
        x = self.backbone(x)  # Result: [Batch, 256, 20, 20]
        x = self.neck(x)      # Result: [Batch, 256, 20, 20]
        x = self.head(x)      # Result: [Batch, 5+N, 20, 20]
        
        # final activation will be applied in the loss function and at inference
        return x


if __name__ == "__main__":
    model = MyYolo(num_classes=15)
    test_input = torch.randn(1, 3, 640, 640)
    output = model(test_input)

    print(f"Input Shape:  {test_input.shape}")
    print(f"Output Shape: {output.shape}") 

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")