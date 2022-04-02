import torch
import torch.nn as nn
from fourier_unit import FourierUnit

class SpectralTransform(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride = 1,
    ):
        super(SpectralTransform,self).__init__()
        
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2,2),stride=2)
        else:
            self.downsample = nn.Identity()

        self.stirde = stride

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, bias=False)
    
    def forward(self,x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        output = self.conv2(x+output)

        return output
        
