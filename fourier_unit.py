import torch
import torch.nn as nn




class FourierUnit(nn.Module):
    
    def __init__(
        self,
        in_channels,
        out_channels,
        ffc3d=False,
        fft_norm='ortho'
    ):
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm
        self.conv_layer = nn.Conv2d(in_channels= in_channels*2, out_channels=out_channels*2,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn = nn.BatchNorm2d(out_channels*2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        batch = x.shape[0] #b
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1) # c, H , W
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous() #(B,c,2,H,W/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted) #( B, c*2,h,w/2+1)
        ffted = self.relu(self.bn(ffted))
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        return output