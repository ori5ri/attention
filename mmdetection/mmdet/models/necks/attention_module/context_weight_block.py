import torch
from mmcv.cnn import constant_init, kaiming_init
from torch import nn
import matplotlib.pyplot as plt


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


class ContextWeightBlcok(nn.Module):
    def __init__(self,
                 inplanes,
                 levels,
                 ratio=1. / 4,
                 pooling_type='att',
                 weight_type=False,
                 eps=0.0001,
                 viz=False):
        super(ContextWeightBlcok, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert levels > 0, 'at least one feature should be used'
        assert weight_type in [False, 'before_attention', 'before_sigmoid', 'after_attention']
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.levels = levels
        self.weight_type = weight_type
        self.eps = eps
        self.relu = nn.ReLU(inplace=False)
        self.viz = viz
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_add_conv =  nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(self.planes, self.inplanes // self.levels, kernel_size=1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        x = torch.cat(x, dim=1)
        context = self.spatial_pool(x)

        # [N, C, 1, 1]
        add_maps = self.channel_add_conv(context)

        return add_maps
