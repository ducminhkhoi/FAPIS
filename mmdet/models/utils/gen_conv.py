import torch.nn as nn
import torch.nn.functional as F
import torch


class GenConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', special=False):
        super(GenConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.transposed = None
        self.output_padding = None
        
        self.use_bias = bias
        self.r1, self.r2, self.r3 = 16, 16, kernel_size * kernel_size

        self.weight_predict = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(256, 256),
            # nn.ReLU(True),
            # nn.Dropout()
        )

        self.gen_A = nn.Linear(256, in_channels * self.r1)
        self.gen_B = nn.Linear(256, out_channels // groups * self.r2)
        self.gen_G = nn.Linear(256, self.r1 * self.r2 * self.r3)
        
        if self.use_bias:
            self.bias_predict = nn.Sequential(
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    # nn.Linear(256, 256),
                    # nn.ReLU(True),
                    # nn.Dropout(),
                    nn.Linear(256, out_channels)
                )

    def forward(self, input, ref_feat):
        N, C, H, W = input.size()
        weight_base = self.weight_predict(ref_feat)
        A = self.gen_A(weight_base).view(-1, self.in_channels, self.r1, ) # I1 * r1
        B = self.gen_B(weight_base).view(-1, self.r2, self.out_channels // self.groups) # r2 * I2
        G = self.gen_G(weight_base).view(-1, self.r1, self.r3*self.r2) #  r1 * (r3 * r2)

        weight = (A @ G).view(-1, self.in_channels * self.r3, self.r2) # (I1 * r3) * r2
        weight = (weight @ B).view(-1, self.in_channels, self.r3, self.out_channels // self.groups) # I1 * r3 * I2
        weight = weight.permute(0, 3, 1, 2).contiguous() # I2 * I1 * r3
        weight = weight.view(N * self.out_channels//self.groups, self.in_channels, self.kernel_size, self.kernel_size) # I1 * I2 * I3

        if self.use_bias:
            bias = self.bias_predict(ref_feat).view(N * self.out_channels)
        else:
            bias = None

        input = input.view(1, N*C, H, W)

        output = F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, groups=N)
                        
        H, W = output.shape[-2:]

        output = output.view(N, -1, H, W)

        return output