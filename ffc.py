import torch
import torch.nn as nn
from spectral_trasform_unit import SpectralTransform


class FFC(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio_gin,
        ratio_gout,
        stride = 1,
        padding = 0,
        dilation = 1,
        bias = False,
        padding_type = 'reflect',
        gated = False,
        **spectral_kwargs
    ):
        super(FFC, self).__init__()
        self.stride = stride
        
        in_cg = int(in_channels*ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels*ratio_gout)
        out_cl = out_channels -out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl,out_cl,kernel_size,stride,padding,dilation,bias=bias,padding_mode=padding_type)

        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl,out_cg,kernel_size,stride,padding,dilation,bias=bias,padding_mode=padding_type)

        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg,out_cl,kernel_size,stride,padding,dilation,bias=bias,padding_mode=padding_type)

        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg,out_cg,stride=1)
        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)
    
    def forward(self,x):
        x_l,x_g = x if type(x) is tuple else (x,0)
        out_xl,out_xg = 0, 0

        if self.gated:
            total_input_parts =[x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts,dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate,l2g_gate = 1,1
        
        if self.ratio_gout != 1:
            out_xl = self.convg2l(x_l) + self.convg2l(x_g)*g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2l(x_l)*l2g_gate+self.convg2g(x_g)
        
        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio_gin,
        ratio_gout,
        stride =1,
        padding=0,
        dilation=1,
        bias = False,
        norm_layer = nn.BatchNorm2d,
        activation_layer = nn.Identity,
        padding_type = 'reflect',
        **kwargs
    ):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels,out_channels,kernel_size,ratio_gin,ratio_gout,stride,padding,dilation,bias,padding_type=padding_type,**kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    
    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g




class FFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3,ratio_gin=0.5,ratio_gout=0.5, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3,ratio_gin=0.5,ratio_gout=0.5, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.inline = inline
    
    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)
        id_l, id_g = x_l, x_g
        
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out