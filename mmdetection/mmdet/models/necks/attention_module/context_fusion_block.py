import torch
from mmcv.cnn import constant_init, kaiming_init
from torch import nn


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 levels,
                 repeated=1,
                 residual=False,
                 ratio=1. / 4,
                 pooling_type='att',
                 fusion_types=('channel_add', 'channel_mul')):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        assert levels > 0, 'at least one feature should be used'
        self.repeated = repeated
        self.residual = residual
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        self.levels = levels
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.ModuleList()
            for i in range(self.repeated):
                self.channel_add_conv.append(nn.ModuleList())
                for level in range(self.levels):
                    self.channel_add_conv[i].append(nn.Sequential(
                        nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                        nn.LayerNorm([self.planes, 1, 1]),
                        nn.ReLU(inplace=True),  # yapf: disable
                        nn.Conv2d(self.planes, self.inplanes // self.levels, kernel_size=1)))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.ModuleList()
            for i in range(self.repeated):
                self.channel_mul_conv.append(nn.ModuleList())
                for level in range(self.levels):
                    self.channel_mul_conv[i].append(nn.Sequential(
                        nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                        nn.LayerNorm([self.planes, 1, 1]),
                        nn.ReLU(inplace=True),  # yapf: disable
                        nn.Conv2d(self.planes, self.inplanes // self.levels, kernel_size=1)))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

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
        # [N, C, 1, 1]
        outs = x
        x = torch.cat(x, dim=1)
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            mul_maps = [torch.sigmoid(self.channel_mul_conv[0][i](context)) for i in range(self.levels)]
            for repeat in range(1, self.repeated):
                if self.residual:
                    mul_maps_ = [torch.sigmoid(self.channel_mul_conv[repeat][i](torch.cat(mul_maps, dim=1)))
                                 for i in range(self.levels)]
                    mul_maps = [mul_map + mul_map_ for mul_map, mul_map_ in zip(mul_maps, mul_maps_)]
                else:
                    mul_maps = [torch.sigmoid(self.channel_mul_conv[repeat][i](torch.cat(mul_maps, dim=1)))
                                for i in range(self.levels)]
            outs = [out * mul_map for out, mul_map in zip(outs, mul_maps)]
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            add_maps = [self.channel_add_conv[0][i](context) for i in range(self.levels)]
            for repeat in range(1, self.repeated):
                if self.residual:
                    add_maps_ = [self.channel_add_conv[repeat][i](torch.cat(add_maps, dim=1)) for i in range(self.levels)]
                    add_maps = [add_map + add_map_ for add_map, add_map_ in zip(add_maps, add_maps_)]
                else:
                    add_maps = [self.channel_add_conv[repeat][i](torch.cat(add_maps, dim=1)) for i in range(self.levels)]
            outs = [out + add_map for out, add_map in zip(outs, add_maps)]

        return sum(outs)
