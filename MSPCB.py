import torch
import torch.nn as nn
from thop import profile
import torch.nn.functional as F


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


import torch
import torch.nn as nn
import torch.nn.functional as F



class MSPCB(nn.Module):

    def __init__(self, dim, dim_out, kernel=3,shuffle = True,skip_connection=True,kernel_sizes=[3,5,7]):

        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.use_skip_connection = skip_connection
        self.shuffle = shuffle
        dim_in = dim_out//2
        self.cv1 = nn.Sequential(
              nn.Conv2d(in_channels=dim,out_channels=dim_in, kernel_size=kernel,stride=1,
                        padding=(kernel - 1) // 2,groups=1),
              nn.BatchNorm2d(dim_in),
              act_layer('gelu', inplace=True)
        )


        self.dw = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=3,
                          padding=(3 + (3 - 1) * (kernel_size - 1)) // 2,
                          dilation=kernel_size,groups = dim_in, bias=False),
                nn.BatchNorm2d(dim_in),
                act_layer('gelu', inplace=True),

            )
            for kernel_size in self.kernel_sizes
        ])

        # self.dw = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size,
        #                   padding= (kernel_size + (kernel_size - 1) * (3 - 1)) // 2,
        #                   dilation=3,groups = dim_in, bias=False),
        #         nn.BatchNorm2d(dim_in),
        #         act_layer('gelu', inplace=True),
        #
        #     )
        #     for kernel_size in self.kernel_sizes
        # ])


        self.pconv2 = nn.Sequential(

           nn.Conv2d(dim_out, dim_out, 1, 1, 0, bias=False),
           #nn.BatchNorm2d(dim_out),
           act_layer('gelu', inplace=True),
           nn.Conv2d(dim_out, dim_out, 1, 1, 0, bias=False),
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim_out, 1, 1, 0, bias=False),
        )


    # ---------------------------- GSConv ---------------------------------

    def forward(self, x0):

            x1 = self.cv1(x0)
            outputs = []
            for dwconv in self.dw:
                dw_out = dwconv(x1)
                outputs.append(dw_out)
            dout = 0
            for dwout in outputs:
                dout = dout + dwout
            x2 = torch.cat((x1, dout), 1)
            #x2 = dout+x1
            if self.shuffle:
                y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
                y = y.permute(0, 2, 1, 3, 4)
                y = y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])
            else:
                y = x2
            y = self.pconv2(y)

            if self.use_skip_connection:
                    x0 = self.conv1x1(x0)
                    y =x0 + y

                    return  y
            else:

                    return y




    # def forward(self, x0):
    #     x1 = self.cv1(x0)
    #     outputs = []
    #     for dwconv in self.dw:
    #         dw_out = dwconv(x0)
    #         outputs.append(dw_out)
    #     dout = 0
    #     for dwout in outputs:
    #         dout = dout + dwout
    #     #x2 = torch.cat((x1, dout), 1)
    #     x2 = dout+x1
    #     if self.shuffle:
    #         y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
    #         y = y.permute(0, 2, 1, 3, 4)
    #         y = y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])
    #     else:
    #         y = x2
    #     y = self.pconv2(y)
    #     if self.use_skip_connection:
    #             x0 = self.conv1x1(x0)
    #             return x0 + y
    #     else:
    #         return y



        # self.pconv1 = nn.Sequential(
        #     # pointwise convolution
        #     nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
        #     nn.BatchNorm2d(dim),
        #     act_layer('gelu', inplace=True)
        # )

        # self.pconv1 = nn.Sequential(
        #     # pointwise convolution
        #
        #      nn.Conv2d(c_, 4 * c_, kernel_size=1, padding=0, stride=1),
        #      nn.GELU(),
        #      nn.Conv2d(4 * c_, c_, kernel_size=1, padding=0, stride=1),
        #      nn.BatchNorm2d(c_),
        #      nn.GELU(),
        # )

        #x2 = self.norms (x2)

        # b, n, h, w = x2.data.size()
        # b_n = b * n // 2
        # y = x2.reshape(b_n, 2, h * w)
        # y = y.permute(1, 0, 2)
        # y = y.reshape(2, -1, n // 2, h, w)
        # y = torch.cat((y[0], y[1]), 1)




# class GSConvns(GSConv):
#     # GSConv with a normative-shuffle https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
#         super().__init__(c1, c2, k=1, s=1, g=1, act=True)
#         c_ = c2 // 2
#         self.shuf = nn.Conv2d(c_ * 2, c2, 1, 1, 0, bias=False)
#
#     def forward(self, x):
#         x1 = self.cv1(x)
#         x2 = torch.cat((x1, self.dw(x1)), 1)
#         # normative-shuffle, TRT supported
#         return nn.ReLU(self.shuf(x2))



#
# class GSBottleneck(nn.Module):
#     # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=3, s=1, e=0.5):
#         super().__init__()
#         c_ = int(c2 *e)
#         # for lighting
#         self.conv_lighting = nn.Sequential(
#             GSConv(c1, c_, 1, 1),
#             GSConv(c_, c2, 3, 1, act=False))
#         self.shortcut = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
#
#     def forward(self, x):
#         return self.conv_lighting(x) + self.shortcut(x)
#
#
# class DWConv(Conv):
#     # Depth-wise convolution class
#     def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)
#
#
#
#
#
# class VoVGSCSP(nn.Module):
#     # VoVGSCSP module with GSBottleneck
#     def __init__(self, cx, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         # self.gc1 = GSConv(c_, c_, 1, 1)
#         # self.gc2 = GSConv(c_, c_, 1, 1)
#         # self.gsb = GSBottleneck(c_, c_, 1, 1)
#         self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
#         self.res = Conv(c_, c_, 3, 1, act=False)
#         self.cv3 = Conv(2 * c_, c2, 1)  #
#
#
#     def forward(self, x):
#         x1 = self.gsb(self.cv1(x))
#         y = self.cv2(x)
#         return self.cv3(torch.cat((y, x1), dim=1))
#
#
# class VoVGSCSPC(VoVGSCSP):
#     # cheap VoVGSCSP module with GSBottleneck
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__(c1, c2)
#         c_ = int(c2 * 0.5)  # hidden channels
#         self.gsb = GSBottleneckC(c_, c_, 1, 1)


