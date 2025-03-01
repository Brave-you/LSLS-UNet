
import torch
from torch import nn
import torch.nn.functional as F
from thop import profile
from timm.models.layers import trunc_normal_
import math
from module.LKselection import LSKBlock
from module.GSconv_3 import MSPCB
from module.attention import *
from module.Deformconv import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=(3 + (3 - 1) * (dilation - 1)) // 2,stride=stride,  groups=dim_in),
        )

        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))

class ConvLayer(nn.Module):
    def __init__(self, dim,dilation=7):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=(3 + (3 - 1) * (dilation - 1)) // 2,stride=1,dilation=dilation, groups=dim, padding_mode='reflect') # depthwise conv
        self.norm1 = nn.BatchNorm2d(dim)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(dim, 4 * dim, kernel_size=1, padding=0, stride=1) #DeformConv2d(dim, dim, kernel_size=3, padding=1,stride=1) #
        self.conv3 = nn.Conv2d(4 * dim, 1, kernel_size=1, padding=0, stride=1)
        self.norm2 = nn.BatchNorm2d(1)
        self.act2 = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.act1(x)


        x = self.conv3(x)
        x = self.norm2(x)
        x = self.act2(x)

        # x = self.conv2(x)
        # x = self.norm2(x)
        # x = self.act2(x)

        # y = x.reshape(x.shape[0], 2, x.shape[1] // 2, x.shape[2], x.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # x = y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])


        return x

class Down(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(self.bn(x))

class Down2(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, groups=in_channels)

    def forward(self, x):
        return self.conv(self.bn(x))

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class BAJDGA(nn.Module):
    def __init__(self, dim_xh, dim_xl):
        super().__init__()
        self.pre_project = nn.Sequential(nn.Conv2d(1, 1, 1),
                                         )
        self.pre_project2 = nn.Sequential(nn.Conv2d(dim_xl, 1, 1),
                                         )
        self.pre_project3 = nn.Sequential(
                                           nn.Conv2d(3, 1, kernel_size=3, padding=1, groups=1),
                                          )
        self.mask1 = simam_module()
        self.mask2 = SpatialAttention(3)
        self.ca = ChannelAttention(3)
        #self.cbam = CBAM_Block(3)
        self.share_space1 = nn.Parameter(torch.Tensor(1, 1, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space1)
        self.share_space2 = nn.Parameter(torch.Tensor(1, 1, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space2)
        self.share_space3 = nn.Parameter(torch.Tensor(1, 1, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space3)
        self.share_space4 = nn.Parameter(torch.Tensor(1, 1, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space4)
        self.share_space5 = nn.Parameter(torch.Tensor(1, 1, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space5)
        self.share_space6 = nn.Parameter(torch.Tensor(1, 1, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space6)

        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=2, data_format='channels_first'),
            nn.Conv2d(2, 1, 1)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, groups=1),
            nn.GELU(),
            nn.Conv2d(1, 1, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, groups=1),
            nn.GELU(),
            nn.Conv2d(1, 1, 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, groups=1),
            nn.GELU(),
            nn.Conv2d(1, 1, 1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, groups=1),
            nn.GELU(),
            nn.Conv2d(1, 1, 1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, groups=1),
            nn.GELU(),
            nn.Conv2d(1, 1, 1)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, groups=1),
            nn.GELU(),
            nn.Conv2d(1, 1, 1)
        )
        self.w = nn.Parameter(torch.ones(6)).cuda()

    def forward(self, xh, xl, mask, out):
        # 解码特征，  编码器特征，   掩码
        a = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        b = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        c = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))
        #d = torch.exp(self.w[3]) / torch.sum(torch.exp(self.w))
        e = torch.exp(self.w[4]) / torch.sum(torch.exp(self.w))
        f = torch.exp(self.w[5]) / torch.sum(torch.exp(self.w))

        #out = self.pre_project(out)
        out = F.interpolate(out, size=[xl.size(2), xl.size(3)], mode='bilinear', align_corners=True)

        xll = self.pre_project2(xl)

        # t1 = self.pre_project3(t1)
        # ref = F.interpolate(ref, size=[xl.size(2), xl.size(3)], mode='bilinear', align_corners=True)


        paraa = self.conv1(F.interpolate(self.share_space1, size=xl.shape[2:4], mode='bilinear', align_corners=True))
        parab = self.conv2(F.interpolate(self.share_space2, size=xl.shape[2:4], mode='bilinear', align_corners=True))
        parac = self.conv3(F.interpolate(self.share_space3, size=xl.shape[2:4], mode='bilinear', align_corners=True))
        #parad = self.conv4(F.interpolate(self.share_space4, size=xl.shape[2:4], mode='bilinear', align_corners=True))
        parah = self.conv5(F.interpolate(self.share_space5, size=xh.shape[2:4], mode='bilinear', align_corners=True))
        paral = self.conv6(F.interpolate(self.share_space6, size=xl.shape[2:4], mode='bilinear', align_corners=True))

        #fuse  =     torch.sigmoid(mask * a + paraa ) * mask  +  torch.sigmoid(out * b + parab ) *out  +torch.sigmoid(xll * c + parac )*xll
        #atten =     (xl + torch.sigmoid( d*xl + paral) * xl ) * torch.sigmoid (fuse)
        #x = atten + (xh + torch.sigmoid( e*xh + parah) * xh ) * torch.sigmoid (atten)

        # fuse  =     (torch.sigmoid(mask * a + paraa ) * mask  +  torch.sigmoid(out * b + parab ) *out  + torch.sigmoid(xll * c + parac )*xll)
        # atten =     (xl + torch.sigmoid( d*xl + paral) * xl ) * torch.sigmoid (fuse)
        # x = atten + (xh + torch.sigmoid( e*xh + parah) * xh ) * torch.sigmoid (fuse)

#效果最佳0.8218,0.818     0.819（3）  0.821(2)   0.814(4)
        fuse  =   self.pre_project3(torch.cat([ torch.sigmoid(mask * a + paraa ) * mask,
                                                        torch.sigmoid(out * b + parab ) * out,
                                                        torch.sigmoid(xll * c + parac ) * xll,
                                                      # torch.sigmoid(ref * d + parad ) * ref,
                                                        ], dim=1 ))
        # atten =     (xl +  torch.sigmoid(  e*xl + paral) * xl ) * torch.sigmoid (fuse)
        # x = atten + (xh +  torch.sigmoid(  f*xh + parah) * xh ) * torch.sigmoid (fuse)

        # atten =     (xl +  torch.sigmoid(  e*xl + paral) * xl ) * torch.sigmoid (fuse)
        # x = atten + (xh +  torch.sigmoid(  f*xh + parah) * xh )# * torch.sigmoid (fuse)

        atten =     (xl +  torch.sigmoid(  e*xl + paral) * xl ) * torch.sigmoid (fuse)
        x = xl  + atten + (xh +  torch.sigmoid(  f*xh + parah) * xh )# * torch.sigmoid (fuse)

        # atten =     (xl +  torch.sigmoid(  e*xl + paral) * xl ) * torch.sigmoid (fuse)
        # x = xl + xh + atten + (xh +  torch.sigmoid(  f*xh + parah) * xh ) * torch.sigmoid (fuse)



#0.822  加上shift mlp是0.820
        # atten = xl +  torch.sigmoid((mask * a + out * b) * paral) * xl
        # x = xl + (xh + torch.sigmoid((mask * d + out * e) * parah) * xh )* atten

#0.820
        # atten = xl +  torch.sigmoid((mask * a + out * b + ref * c ) * paral) * xl
        # x = xl + (xh + torch.sigmoid((mask * d + out * e+ ref * f)  * parah) * xh )* atten


        return x, fuse

class MlpChannel(nn.Module):
    def __init__(self, c_dim, shift):
        super().__init__()
        self.pad = 2  # 需要的填充大小
        self.shift = shift
        self.conv1 = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, kernel_size=3, padding=1, groups=c_dim),
            nn.GELU(),
            nn.Conv2d(c_dim, c_dim, 1)
        )

    def forward(self, x1):
        B, C, H, W = x1.shape
        x1 = F.pad(x1, [self.pad, self.pad, self.pad, self.pad], "constant", 0)
        x_shift = torch.roll(x1, shifts=self.shift, dims=(2, 3))
        x_cat = torch.narrow(x_shift, 2, self.pad, H)
        x = torch.narrow(x_cat, 3, self.pad, W)

        x = self.conv1(x)

        return x



class Dual_Channel_Mixing_Shift_Convolution_Block(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        c_dim = dim_in // 4

        self.share_space1 = nn.Parameter(torch.Tensor(1, c_dim, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, kernel_size=3, padding=1, groups=c_dim),
            nn.GELU(),
            nn.Conv2d(c_dim, c_dim, 1)
        )
        self.share_space2 = nn.Parameter(torch.Tensor(1, c_dim, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, kernel_size=3, padding=1, groups=c_dim),
            nn.GELU(),
            nn.Conv2d(c_dim, c_dim, 1)
        )
        self.share_space3 = nn.Parameter(torch.Tensor(1, c_dim, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space3)
        self.conv3 = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, kernel_size=3, padding=1, groups=c_dim),
            nn.GELU(),
            nn.Conv2d(c_dim, c_dim, 1)
        )
        self.share_space4 = nn.Parameter(torch.Tensor(1, c_dim, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space4)
        self.conv4 = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, kernel_size=3, padding=1, groups=c_dim),
            nn.GELU(),
            nn.Conv2d(c_dim, c_dim, 1)
        )

        self.shuf = nn.Conv2d(dim_in, dim_in, 1, 1, 0, bias=False)

        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')

        self.ldw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_out, 1),
        )

        self.shift1 = MlpChannel(c_dim, [1, 1])
        self.shift2 = MlpChannel(c_dim, [-1, 1])
        self.shift3 = MlpChannel(c_dim, [-1, -1])
        self.shift4 = MlpChannel(c_dim, [1, -1])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # C*H*W
        self.conv5 = nn.Sequential(
            #LiteDynamicDWConv(dim_in,dim_in)
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            # nn.BatchNorm2d(dim_in),
            # nn.Conv2d(dim_in,  dim_in, 1),
            # act_layer('gelu', inplace=True),
        )

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        B, C, H, W = x1.size()

        # x11 = x1 * self.conv1(F.interpolate(self.share_space1, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        # x22 = x2 * self.conv2(F.interpolate(self.share_space2, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        # x33 = x3 * self.conv3(F.interpolate(self.share_space3, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        # x44 = x4 * self.conv4(F.interpolate(self.share_space4, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        # xshift = torch.cat([x22, x44, x11, x33], dim=1)

        # xshift = torch.cat([self.shift2(x2),self.shift4(x4),self.shift1(x1),self.shift3(x3)],dim=1)

        xshift = torch.cat([self.shift2(x2),
                                   self.shift4(x4),
                                   self.shift1(x1),
                                   self.shift3(x3)], dim=1)

        xres = torch.cat([x1, x2, x3, x4],dim=1)#

        x = xshift + self.conv5(xres)  # xres

        y = x.reshape(x.shape[0], 2, x.shape[1] // 2, x.shape[2], x.shape[3])
        y = y.permute(0, 2, 1, 3, 4)
        x = y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])

        x = self.norm2(x)
        x = self.ldw(x)

        return x


class LSLSNet(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], bridge=True, gt_ds=True):
        super().__init__()

        self.bridge = bridge
        self.gt_ds = gt_ds
        upscale_factor = 2
        self.encoder1 = nn.Sequential(
            #nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
            MSPCB(input_channels, c_list[0], 3, False),  # 7
        )
        self.encoder2 = nn.Sequential(
            #nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
            MSPCB(c_list[0], c_list[1], 3, True),
            # ConvLayer(c_list[1]),

        )
        self.encoder3 = nn.Sequential(
            #nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
            MSPCB(c_list[1], c_list[2], 3, True),
            # Grouped_multi_axis_Hadamard_Product_Attention(c_list[2], c_list[2]),
            # ConvLayer(c_list[2]),
        )
        self.encoder4 = nn.Sequential(
            #nn.Conv2d(c_list[2], c_list[3], 3, stride=1, padding=1),
            Dual_Channel_Mixing_Shift_Convolution_Block(c_list[2], c_list[3]),
            # GSConv(c_list[2], c_list[3]),
        )
        self.encoder5 = nn.Sequential(
            #nn.Conv2d(c_list[3], c_list[4], 3, stride=1, padding=1),
            Dual_Channel_Mixing_Shift_Convolution_Block(c_list[3], c_list[4]),


        )
        self.encoder6 = nn.Sequential(
            #nn.Conv2d(c_list[4], c_list[5], 3, stride=1, padding=1),
            Dual_Channel_Mixing_Shift_Convolution_Block(c_list[4], c_list[5]),

        )

        self.bottle = nn.Sequential(
           # nn.Conv2d(c_list[5], 1, 1),
           nn.Conv2d(c_list[5], c_list[5], kernel_size=3, padding=1, groups=c_list[5]),
           nn.GELU(),
           nn.Conv2d(c_list[5], 1, 1),
        )


        self.Down1 = Down2(c_list[0])
        self.Down2 = Down2(c_list[1])
        self.Down3 = Down2(c_list[2])

        if bridge:
            self.BAJDGA1 = BAJDGA(c_list[1], c_list[0])
            self.BAJDGA2 = BAJDGA(c_list[2], c_list[1])
            self.BAJDGA3 = BAJDGA(c_list[3], c_list[2])
            self.BAJDGA4 = BAJDGA(c_list[4], c_list[3])
            self.BAJDGA5 = BAJDGA(c_list[5], c_list[4])
            print('group_aggregation_bridge was used')
        if gt_ds:
            self.gt_conv1 = nn.Sequential(nn.Conv2d(c_list[4], 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(c_list[3], 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(c_list[2], 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(c_list[1], 1, 1))
            self.gt_conv5 = nn.Sequential(nn.Conv2d(c_list[0], 1, 1))

            self.edge_conv1 = nn.Sequential(nn.Conv2d(1, 1, 1))
            self.edge_conv2 = nn.Sequential(nn.Conv2d(1, 1, 1))
            self.edge_conv3 = nn.Sequential(nn.Conv2d(1, 1, 1))
            self.edge_conv4 = nn.Sequential(nn.Conv2d(1, 1, 1))
            self.edge_conv5 = nn.Sequential(nn.Conv2d(1, 1, 1))

            print('gt deep supervision was used')

        self.decoder1 = nn.Sequential(
            #nn.Conv2d(c_list[5], c_list[4], 3, stride=1, padding=1),
            Dual_Channel_Mixing_Shift_Convolution_Block(c_list[5], c_list[4]),
        )
        self.decoder2 = nn.Sequential(
            #nn.Conv2d(c_list[4], c_list[3], 3, stride=1, padding=1),
            Dual_Channel_Mixing_Shift_Convolution_Block(c_list[4], c_list[3]),
        )
        self.decoder3 = nn.Sequential(
            #nn.Conv2d(c_list[3], c_list[2], 3, stride=1, padding=1),
            Dual_Channel_Mixing_Shift_Convolution_Block(c_list[3], c_list[2]),
        )
        self.decoder4 = nn.Sequential(
            #nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
            MSPCB (c_list[2], c_list[1], 3, True)
        )
        self.decoder5 = nn.Sequential(
            #nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
            MSPCB (c_list[1], c_list[0], 3, False)
        )

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])


        self.final = nn.Sequential(

            nn.Conv2d(c_list[0], 1, kernel_size=1)

        )

        self.w = nn.Parameter(torch.ones(5)).cuda()


        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):


        # out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        out = F.gelu(self.Down1(self.ebn1(self.encoder1(x))))
        t1 = out  # b, 8, 128, 128

        # out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        out = F.gelu(self.Down2(self.ebn2(self.encoder2(out))))
        t2 = out  # b, 16, 64, 64


        # out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        out = F.gelu(self.Down3(self.ebn3(self.encoder3(out))))
        t3 = out  # b, 24, 32, 32

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # b, 32, 16, 16

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # b, 48, 8, 8

        out = F.gelu(self.encoder6(out))  # b, 64, 8, 8


        #out = self.bottle5(out) # MLP

        t6 = out
        t6 = self.bottle(t6)


        out5 = F.gelu(self.dbn1(self.decoder1(out)))  # b, 48, 8, 8




        gt_pre5 = self.gt_conv1(out5)
        out5, atten5 = self.BAJDGA5(out5, t5, gt_pre5, t6)
        atten55 = self.edge_conv1(atten5)
        gt_pre55 = F.interpolate(gt_pre5, scale_factor=32, mode='bilinear', align_corners=True)
        edge_gt5 = F.interpolate(atten55, scale_factor=32, mode='bilinear', align_corners=True)



        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',align_corners=True))  # b, c3, H/16, W/16
        gt_pre4 = self.gt_conv2(out4)
        out4, atten4 = self.BAJDGA4(out4, t4, gt_pre4, atten55)
        atten44 = self.edge_conv2(atten4)
        gt_pre44 = F.interpolate(gt_pre4, scale_factor=16, mode='bilinear', align_corners=True)
        edge_gt4 = F.interpolate(atten44, scale_factor=16, mode='bilinear', align_corners=True)



        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',align_corners=True))  # b, c2, H/8, W/8
        gt_pre3 = self.gt_conv3(out3)
        out3, atten3 = self.BAJDGA3(out3, t3, gt_pre3, atten44)
        atten33 = self.edge_conv3(atten3)
        gt_pre33 = F.interpolate(gt_pre3, scale_factor=8, mode='bilinear', align_corners=True)
        edge_gt3 = F.interpolate(atten33, scale_factor=8, mode='bilinear', align_corners=True)



        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',align_corners=True))  # b, c1, H/4, W/4
        gt_pre2 = self.gt_conv4(out2)
        out2, atten2 = self.BAJDGA2(out2, t2, gt_pre2, atten33)
        atten22 = self.edge_conv4(atten2)
        gt_pre22 = F.interpolate(gt_pre2, scale_factor=4, mode='bilinear', align_corners=True)
        edge_gt2 = F.interpolate(atten22, scale_factor=4, mode='bilinear', align_corners=True)



        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',align_corners=True))  # b, c0, H/2, W/2
        gt_pre1 = self.gt_conv5(out1)
        out1, atten1 = self.BAJDGA1(out1, t1, gt_pre1, atten22)
        atten11 = self.edge_conv5(atten1)
        gt_pre11 = F.interpolate(gt_pre1, scale_factor=2, mode='bilinear', align_corners=True)
        edge_gt1 = F.interpolate(atten11, scale_factor=2, mode='bilinear', align_corners=True)


        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear', align_corners=True)  # b, num_class, H, W




        return (
            (torch.sigmoid(gt_pre55), torch.sigmoid(gt_pre44), torch.sigmoid(gt_pre33), torch.sigmoid(gt_pre22),torch.sigmoid(gt_pre11)),

            torch.sigmoid(out0),


            (torch.sigmoid(edge_gt5), torch.sigmoid(edge_gt4), torch.sigmoid(edge_gt3),torch.sigmoid(edge_gt2), torch.sigmoid(edge_gt1)),

        )



if __name__ == '__main__':
    net = LSLSNet(num_classes=1).cuda()
    input = torch.randn(1, 3, 256, 256).cuda()
    flops, params = profile(net, (input,))
    outputs = net(input)
    print('flops: ', flops, 'params: ', params, 'outputs: ', outputs[1].shape)
