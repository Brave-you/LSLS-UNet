import torch
import torch.nn as nn
from thop import profile
import torch.nn.functional as F
import torch
import torch.nn as nn
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


        self.pconv2 = nn.Sequential(

           nn.Conv2d(dim_out, dim_out, 1, 1, 0, bias=False),
           #nn.BatchNorm2d(dim_out),
           act_layer('gelu', inplace=True),
           nn.Conv2d(dim_out, dim_out, 1, 1, 0, bias=False),
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim_out, 1, 1, 0, bias=False),
        )


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
