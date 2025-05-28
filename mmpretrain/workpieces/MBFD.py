import math
from typing import Sequence
from mmpretrain.models.utils import to_2tuple
import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
from mmengine.model import BaseModule
__all__ = ['MBFD_E', 'CMBFD', 'CMBFD_16x', 'MBFD', 'FMBFD', 'MBFD_E_GN', 'CMBFD_GN', 'MBFD_E_V2', 'CMBFD_V2']

class TiedBlockConv2d(nn.Module):
    '''Tied Block Conv2d'''
    def __init__(self, in_planes, planes, kernel_size, stride=1, padding=0, bias=True, B=1, args=None, dropout_tbc=0.0, groups=1, dilation=1):
        super(TiedBlockConv2d, self).__init__()
        assert planes % B == 0
        assert in_planes % B == 0
        self.B = B
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.out_planes = planes
        self.kernel_size = kernel_size
        self.dropout_tbc = dropout_tbc
        self.conv = nn.Conv2d(in_planes//self.B, planes//self.B, kernel_size=kernel_size, stride=stride, \
                    padding=padding, bias=bias, groups=groups, dilation=dilation)
        if self.dropout_tbc > 0.0:
            self.drop_out = nn.Dropout(self.dropout_tbc)
    def forward(self, x):
        n, c, h, w = x.size()
        x = x.contiguous().view(n*self.B, c//self.B, h, w)
        h_o = (h - self.kernel_size - (self.kernel_size-1)*(self.dilation-1) + 2*self.padding) // self.stride + 1
        w_o = (w - self.kernel_size - (self.kernel_size-1)*(self.dilation-1) + 2*self.padding) // self.stride + 1
        x = self.conv(x)
        x = x.view(n, self.out_planes, h_o, w_o)
        if self.dropout_tbc > 0:
            x = self.drop_out(x)
        return x

class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),  
            nn.ReLU(inplace=True), 
        )
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, :]  # 水平高频
        y_LH = yH[0][:, :, 1, :]  # 垂直高频
        y_HH = yH[0][:, :, 2, :]  # 对角高频
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x
    
class Down_wt_GN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt_GN, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.GroupNorm(1, out_ch),  
            nn.ReLU(inplace=True), 
        )
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, :]  # 水平高频
        y_LH = yH[0][:, :, 1, :]  # 垂直高频
        y_HH = yH[0][:, :, 2, :]  # 对角高频
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))

class Conv_GN(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.gn = nn.GroupNorm(1, c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.gn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))

class PTConv(nn.Module):
    def __init__(self, dim, k=3, s=1, p=1, d=1, n_div=2, nwa=True):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, kernel_size=k, stride=s, padding=p, dilation=d, bias=False)
        self.tied_conv = TiedBlockConv2d(self.dim_untouched, self.dim_untouched, kernel_size=k, stride=s, padding=p, dilation=d, bias=False)
        self.nwa = nwa
        if nwa:
            self.norm = nn.BatchNorm2d(dim)
            self.act = nn.SiLU()
        
    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x2 = self.tied_conv(x2)
        x2_1, x2_2 = torch.chunk(x2, 2, dim=1)
        x = torch.cat((x2_1, x1, x2_2), 1)
        if self.nwa:
            x = self.act(self.norm(x))
        return x

class PTConv_GN(nn.Module):
    def __init__(self, dim, k=3, s=1, p=1, d=1, n_div=2, nwa=True):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, kernel_size=k, stride=s, padding=p, dilation=d, bias=False)
        self.tied_conv = TiedBlockConv2d(self.dim_untouched, self.dim_untouched, kernel_size=k, stride=s, padding=p, dilation=d, bias=False)
        self.nwa = nwa
        if nwa:
            self.norm = nn.GroupNorm(1, dim)
            self.act = nn.SiLU()
        
    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x2 = self.tied_conv(x2)
        x2_1, x2_2 = torch.chunk(x2, 2, dim=1)
        x = torch.cat((x2_1, x1, x2_2), 1)
        if self.nwa:
            x = self.act(self.norm(x))
        return x

class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc):
        super().__init__()
        self.conv = Conv(inc * 4, ouc, act=nn.ReLU(inplace=True), k=3, s=1, p=1)

    def forward(self, x):
        x = torch.cat([x[...,  ::2,  ::2], 
                       x[..., 1::2,  ::2], 
                       x[...,  ::2, 1::2], 
                       x[..., 1::2, 1::2]
                      ], 1)
        x = self.conv(x)
        return x
    
class SPDConv_GN(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc):
        super().__init__()
        self.conv = Conv_GN(inc * 4, ouc, act=nn.ReLU(inplace=True), k=3, s=1, p=1)

    def forward(self, x):
        x = torch.cat([x[...,  ::2,  ::2], 
                       x[..., 1::2,  ::2], 
                       x[...,  ::2, 1::2], 
                       x[..., 1::2, 1::2]
                      ], 1)
        x = self.conv(x)
        return x

class CMBFD(BaseModule):
    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 conv_type='Conv2d',
                 kernel_size=16,
                 stride=16,
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 input_size=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.FMBFD = FMBFD(in_channels=in_channels, out_channels=embed_dims//4)
        self.MBFD = MBFD(in_channels=embed_dims//4, out_channels=embed_dims)
        
        if input_size:
            input_size = to_2tuple(input_size)
            self.init_input_size = input_size
            self.init_out_size = (input_size[0]//4, input_size[1]//4)
        else:
            self.init_input_size = None
            self.init_out_size = None
    def forward(self, x):
        x = self.FMBFD(x)
        x = self.MBFD(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        return x, out_size
    
class CMBFD_V2(BaseModule):
    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 conv_type='Conv2d',
                 kernel_size=16,
                 stride=16,
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 input_size=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.FMBFD = FMBFD_V2(in_channels=in_channels, out_channels=embed_dims//4)
        self.MBFD = MBFD_V2(in_channels=embed_dims//4, out_channels=embed_dims)
        
        if input_size:
            input_size = to_2tuple(input_size)
            self.init_input_size = input_size
            self.init_out_size = (input_size[0]//4, input_size[1]//4)
        else:
            self.init_input_size = None
            self.init_out_size = None
    def forward(self, x):
        x = self.FMBFD(x)
        x = self.MBFD(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        return x, out_size
    
    
class CMBFD_GN(BaseModule):
    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 conv_type='Conv2d',
                 kernel_size=16,
                 stride=16,
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 input_size=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.FMBFD = FMBFD_GN(in_channels=in_channels, out_channels=embed_dims//4)
        self.MBFD = MBFD_GN(in_channels=embed_dims//4, out_channels=embed_dims)
        
        if input_size:
            input_size = to_2tuple(input_size)
            self.init_input_size = input_size
            self.init_out_size = (input_size[0]//4, input_size[1]//4)
        else:
            self.init_input_size = None
            self.init_out_size = None
    def forward(self, x):
        x = self.FMBFD(x)
        x = self.MBFD(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        return x, out_size

class CMBFD_16x(BaseModule):
    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 conv_type='Conv2d',
                 kernel_size=16,
                 stride=16,
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 input_size=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.FMBFD = FMBFD(in_channels=in_channels, out_channels=embed_dims//16)
        self.MBFD1 = MBFD(in_channels=embed_dims//16, out_channels=embed_dims//8)
        self.MBFD2 = MBFD(in_channels=embed_dims//8, out_channels=embed_dims//4)
        self.MBFD3 = MBFD(in_channels=embed_dims//4, out_channels=embed_dims)
        
        if input_size:
            input_size = to_2tuple(input_size)
            self.init_input_size = input_size
            self.init_out_size = (input_size[0]//16, input_size[1]//16)
        else:
            self.init_input_size = None
            self.init_out_size = None
    def forward(self, x):
        x = self.FMBFD(x)
        x = self.MBFD1(x)
        x = self.MBFD2(x)
        x = self.MBFD3(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        return x, out_size
        
class FMBFD(BaseModule):
    """
    首层多分支融合下采样模块（First-layer Multi-branch Fusion Downsampling, FMBFD）
    用于首层（从三通道出发）的下采样
    """
    def __init__(self, in_channels=3, out_channels=16, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.proj_first = Conv(in_channels, out_channels//2, k=3, s=1, p=1)
        
        self.conv1 = SPDConv(in_channels, out_channels//2)
        
        self.conv2 = Conv(out_channels//2, out_channels//2, k=3, s=2, p=1, g=out_channels//2)
        
        self.conv3 = PTConv(out_channels//2, k=3, s=2, p=1, d=1, n_div=2)
        
        self.conv4 = Down_wt(in_channels, out_channels//2)
        
        self.proj_last = Conv(2*out_channels, out_channels)

    def forward(self, x):
        c = self.proj_first(x)

        c1 = self.conv1(x)     
        c2 = self.conv2(c)
        c3 = self.conv3(c)
        c4 = self.conv4(x)

        x = torch.cat([c1, c2, c3, c4], dim=1)
        x = self.proj_last(x)
        return x

class FMBFD_V2(BaseModule):
    """
    首层多分支融合下采样模块（First-layer Multi-branch Fusion Downsampling, FMBFD）
    用于首层（从三通道出发）的下采样
    """
    def __init__(self, in_channels=3, out_channels=16, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.proj_first = Conv(in_channels, out_channels//2, k=3, s=1, p=1, act=False)
        
        self.conv1 = SPDConv(in_channels, out_channels//2)
        
        self.conv2 = Conv(out_channels//2, out_channels//2, k=3, s=2, p=1, g=out_channels//2)
        
        self.conv3 = PTConv(out_channels//2, k=3, s=2, p=1, d=1, n_div=2)
        
        self.conv4 = Down_wt(in_channels, out_channels//2)
        
        self.proj_last = Conv(2*out_channels, out_channels, act=False)

    def forward(self, x):
        c = self.proj_first(x)

        c1 = self.conv1(x)     
        c2 = self.conv2(c)
        c3 = self.conv3(c)
        c4 = self.conv4(x)

        x = torch.cat([c1, c2, c3, c4], dim=1)
        x = self.proj_last(x)
        return x

class FMBFD_GN(BaseModule):
    """
    首层多分支融合下采样模块（First-layer Multi-branch Fusion Downsampling, FMBFD）
    用于首层（从三通道出发）的下采样
    """
    def __init__(self, in_channels=3, out_channels=16, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.proj_first = Conv_GN(in_channels, out_channels//2, k=3, s=1, p=1, act=False)
        
        self.conv1 = SPDConv_GN(in_channels, out_channels//2)
        
        self.conv2 = Conv_GN(out_channels//2, out_channels//2, k=3, s=2, p=1, g=out_channels//2)
        
        self.conv3 = PTConv_GN(out_channels//2, k=3, s=2, p=1, d=1, n_div=2)
        
        self.conv4 = Down_wt_GN(in_channels, out_channels//2)
        
        self.proj_last = Conv_GN(2*out_channels, out_channels,act=False)

    def forward(self, x):
        c = self.proj_first(x)

        c1 = self.conv1(x)     
        c2 = self.conv2(c)
        c3 = self.conv3(c)
        c4 = self.conv4(x)

        x = torch.cat([c1, c2, c3, c4], dim=1)
        x = self.proj_last(x)
        return x

class MBFD(BaseModule):
    """
    多分支融合下采样模块（Multi-branch Fusion Downsampling, MBFD）
    该模块融合了深度可分离卷积、部分通道卷积和小波变换三种不同的下采样方式，
    通过通道拼接和1x1卷积实现高效特征融合，提升下采样阶段的特征表达能力。
    """
    def __init__(self, in_channels, out_channels, nwa=True, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.proj_first = Conv(in_channels, out_channels, k=3, s=1, p=1, g=math.gcd(in_channels,out_channels))
        assert out_channels % 2 == 0, 'out_channels must be even'
        self.conv1 = Conv(out_channels, out_channels//2, k=3, s=2, p=1, g=out_channels//2)
        
        self.conv2 = PTConv(out_channels, k=3, s=2, p=1, d=1, n_div=2)
        
        self.harr = Down_wt(in_channels, out_channels // 2)
        
        if nwa:
            self.proj_last = Conv(2*out_channels, out_channels, k=1, s=1)
        else:
            self.proj_last = nn.Conv2d(2*out_channels, out_channels, 1, 1)

    def forward(self, x):
        c = self.proj_first(x)

        c1 = self.conv1(c)     
        c2 = self.conv2(c)
        w = self.harr(x)

        x = torch.cat([c1, c2, w], dim=1)
        x = self.proj_last(x)
        return x
    
class MBFD_V2(BaseModule):
    """
    多分支融合下采样模块（Multi-branch Fusion Downsampling, MBFD）
    该模块融合了深度可分离卷积、部分通道卷积和小波变换三种不同的下采样方式，
    通过通道拼接和1x1卷积实现高效特征融合，提升下采样阶段的特征表达能力。
    no act
    """
    def __init__(self, in_channels, out_channels, nwa=True, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.proj_first = Conv(in_channels, out_channels, k=3, s=1, p=1, g=math.gcd(in_channels,out_channels))
        assert out_channels % 2 == 0, 'out_channels must be even'
        self.conv1 = Conv(out_channels, out_channels//2, k=3, s=2, p=1, g=out_channels//2, act=False)
        
        self.conv2 = PTConv(out_channels, k=3, s=2, p=1, d=1, n_div=2)
        
        self.harr = Down_wt(in_channels, out_channels // 2)
        
        if nwa:
            self.proj_last = Conv(2*out_channels, out_channels, k=1, s=1, act=False)
        else:
            self.proj_last = nn.Conv2d(2*out_channels, out_channels, 1, 1)

    def forward(self, x):
        c = self.proj_first(x)

        c1 = self.conv1(c)     
        c2 = self.conv2(c)
        w = self.harr(x)

        x = torch.cat([c1, c2, w], dim=1)
        x = self.proj_last(x)
        return x
    
class MBFD_GN(BaseModule):
    """
    多分支融合下采样模块（Multi-branch Fusion Downsampling, MBFD）
    该模块融合了深度可分离卷积、部分通道卷积和小波变换三种不同的下采样方式，
    通过通道拼接和1x1卷积实现高效特征融合，提升下采样阶段的特征表达能力。
    """
    def __init__(self, in_channels, out_channels, nwa=True, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.proj_first = Conv_GN(in_channels, out_channels, k=3, s=1, p=1, g=math.gcd(in_channels,out_channels),act=False)
        assert out_channels % 2 == 0, 'out_channels must be even'
        self.conv1 = Conv_GN(out_channels, out_channels//2, k=3, s=2, p=1, g=out_channels//2)
        
        self.conv2 = PTConv_GN(out_channels, k=3, s=2, p=1, d=1, n_div=2)
        
        self.harr = Down_wt_GN(in_channels, out_channels // 2)
        
        if nwa:
            self.proj_last = Conv_GN(2*out_channels, out_channels, k=1, s=1, act=False)
        else:
            self.proj_last = nn.Conv2d(2*out_channels, out_channels, 1, 1)

    def forward(self, x):
        c = self.proj_first(x)

        c1 = self.conv1(c)     
        c2 = self.conv2(c)
        w = self.harr(x)

        x = torch.cat([c1, c2, w], dim=1)
        x = self.proj_last(x)
        return x

class MBFD_E(BaseModule):
    """
    多分支融合下采样模块（Multi-branch Fusion Downsampling, MBFD）
    该模块融合了深度可分离卷积、部分通道卷积和小波变换三种不同的下采样方式，
    通过通道拼接和1x1卷积实现高效特征融合，提升下采样阶段的特征表达能力。
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 stride=None,
                 padding='corner',
                 dilation=1,
                 bias=False,
                 norm_cfg=dict(type='LN'),
                 use_post_norm=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.proj_first = Conv(in_channels, out_channels, k=3, s=1, p=1, g=in_channels)
        
        self.conv1 = Conv(out_channels, out_channels//2, k=3, s=2, p=1, g=out_channels//2)
        
        self.conv2 = PTConv(out_channels, k=3, s=2, p=1, d=1, n_div=2)
        
        self.harr = Down_wt(in_channels, out_channels // 2)
        
        self.proj_last = Conv(2*out_channels, out_channels, k=1, s=1)
        
    def forward(self, x, input_size):
        B, L, C = x.shape
        assert isinstance(input_size, Sequence), f'Expect ' \
                                                 f'input_size is ' \
                                                 f'`Sequence` ' \
                                                 f'but get {input_size}'

        H, W = input_size
        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W
        
        c = self.proj_first(x)

        c1 = self.conv1(c)     
        c2 = self.conv2(c)
        w = self.harr(x)

        x = torch.cat([c1, c2, w], dim=1)
        x = self.proj_last(x)
        
        output_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        return x, output_size

class MBFD_E_V2(BaseModule):
    """
    多分支融合下采样模块（Multi-branch Fusion Downsampling, MBFD）
    该模块融合了深度可分离卷积、部分通道卷积和小波变换三种不同的下采样方式，
    通过通道拼接和1x1卷积实现高效特征融合，提升下采样阶段的特征表达能力。
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 stride=None,
                 padding='corner',
                 dilation=1,
                 bias=False,
                 norm_cfg=dict(type='LN'),
                 use_post_norm=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.proj_first = Conv(in_channels, out_channels, k=3, s=1, p=1, g=in_channels, act=False)
        
        self.conv1 = Conv(out_channels, out_channels//2, k=3, s=2, p=1, g=out_channels//2)
        
        self.conv2 = PTConv(out_channels, k=3, s=2, p=1, d=1, n_div=2)
        
        self.harr = Down_wt(in_channels, out_channels // 2)
        
        self.proj_last = Conv(2*out_channels, out_channels, k=1, s=1, act=False)
        
    def forward(self, x, input_size):
        B, L, C = x.shape
        assert isinstance(input_size, Sequence), f'Expect ' \
                                                 f'input_size is ' \
                                                 f'`Sequence` ' \
                                                 f'but get {input_size}'

        H, W = input_size
        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W
        
        c = self.proj_first(x)

        c1 = self.conv1(c)     
        c2 = self.conv2(c)
        w = self.harr(x)

        x = torch.cat([c1, c2, w], dim=1)
        x = self.proj_last(x)
        
        output_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        return x, output_size

class MBFD_E_GN(BaseModule):
    """
    多分支融合下采样模块（Multi-branch Fusion Downsampling, MBFD）
    该模块融合了深度可分离卷积、部分通道卷积和小波变换三种不同的下采样方式，
    通过通道拼接和1x1卷积实现高效特征融合，提升下采样阶段的特征表达能力。
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 stride=None,
                 padding='corner',
                 dilation=1,
                 bias=False,
                 norm_cfg=dict(type='LN'),
                 use_post_norm=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.proj_first = Conv_GN(in_channels, out_channels, k=3, s=1, p=1, g=in_channels,act=False)
        
        self.conv1 = Conv_GN(out_channels, out_channels//2, k=3, s=2, p=1, g=out_channels//2)
        
        self.conv2 = PTConv_GN(out_channels, k=3, s=2, p=1, d=1, n_div=2)
        
        self.harr = Down_wt_GN(in_channels, out_channels // 2)
        
        self.proj_last = Conv_GN(2*out_channels, out_channels, k=1, s=1,act=False)
        
    def forward(self, x, input_size):
        B, L, C = x.shape
        assert isinstance(input_size, Sequence), f'Expect ' \
                                                 f'input_size is ' \
                                                 f'`Sequence` ' \
                                                 f'but get {input_size}'

        H, W = input_size
        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W
        
        c = self.proj_first(x)

        c1 = self.conv1(c)     
        c2 = self.conv2(c)
        w = self.harr(x)

        x = torch.cat([c1, c2, w], dim=1)
        x = self.proj_last(x)
        
        output_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        return x, output_size

if __name__ == "__main__":
    # 测试MBFD模块的用例
    # 假设输入为(batch_size, H*W, C)，例如batch_size=2，H=W=32，C=16
    batch_size = 2
    H, W = 32, 32  # 输入特征图的高和宽
    C = 16         # 输入通道数
    out_channels = 384  # MBFD模块输出通道数

    # 构造输入张量，形状为(batch_size, H*W, C)
    x = torch.randn(batch_size, H * W, C)
    x2 = torch.randn(batch_size, 3, H,  W)

    # 实例化MBFD模块
    mbfd = MBFD_E(in_channels=C, out_channels=out_channels)
    # f = CMBFD_16x(3, out_channels)
    # 前向传播，传入x和输入尺寸
    out, out_size = mbfd(x, (H, W))
    # out, out_size = f(x2)

    # 打印输入输出形状和输出尺寸
    print("输入张量形状:", x.shape)            # (2, 1024, 16)
    print("输出张量形状:", out.shape)         # (2, H/2*W/2, out_channels)
    print("输出特征图空间尺寸:", out_size)    # (16, 16)