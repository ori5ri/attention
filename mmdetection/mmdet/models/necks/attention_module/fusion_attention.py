import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelAttention(nn.Module):
    def __init__(self, gate_channels, levels, reduction_ratio=16, pool_types=['avg', 'max'], map_repeated=1,
                 map_residual=False):
        super(ChannelAttention, self).__init__()
        assert all(pool_type in ['avg', 'max', 'lp', 'lse'] for pool_type in pool_types)
        self.gate_channels = gate_channels
        self.levels = levels
        self.mlp = nn.ModuleList()
        for _ in range(levels):
            self.mlp.append(nn.Sequential(
                Flatten(),
                nn.Linear(gate_channels, gate_channels // reduction_ratio),
                nn.ReLU(),
                nn.Linear(gate_channels // reduction_ratio, gate_channels // levels)
            ))
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        attention = torch.cat(x, dim=1)

        for pool_type in self.pool_types:
            if pool_type == 'avg':
                pool = F.avg_pool2d(attention, (attention.size(2), attention.size(3)),
                                    stride=(attention.size(2), attention.size(3)))
            elif pool_type == 'max':
                pool = F.max_pool2d(attention, (attention.size(2), attention.size(3)),
                                    stride=(attention.size(2), attention.size(3)))
            elif pool_type == 'lp':
                pool = F.lp_pool2d(attention, 2, (attention.size(2), attention.size(3)),
                                   stride=(attention.size(2), attention.size(3)))
            elif pool_type == 'lse':  # LSE pool only
                pool = logsumexp_2d(attention)

            channel_att_raw = [self.mlp[j](pool) for j in range(self.levels)]

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = [raw + sum for raw, sum in zip(channel_att_raw, channel_att_sum)]

        channel_att_sum = [torch.sigmoid(att_sum) for att_sum in channel_att_sum]

        scale = [sum.unsqueeze(2).unsqueeze(3).expand_as(x[0]) for sum in channel_att_sum]

        return scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialAttention(nn.Module):
    def __init__(self, levels, kernel_size=7, map_repeated=1, map_residual=False):
        super(SpatialAttention, self).__init__()
        self.levels = levels
        self.map_repeated = map_repeated
        self.map_residual = map_residual
        self.compress = ChannelPool()
        self.spatial = nn.ModuleList()
        for _ in range(levels):
            self.spatial.append(
                BasicConv(2 * levels, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False))

    def forward(self, x):
        attention = torch.cat([(self.compress(_)) for _ in x], dim=1)
        scale = [torch.sigmoid(self.spatial[j](attention)) for j in range(self.levels)]

        return scale


class FusionAttention(nn.Module):
    def __init__(self, gate_channels, levels, reduction_ratio=16, kernel_size=7, pool_types=['avg', 'max'],
                 no_channel=False, no_spatial=False, stacking=1, residual=False, map_repeated=1, map_residual=False):
        super(FusionAttention, self).__init__()
        assert 1 <= stacking
        self.levels = levels
        self.no_channel = no_channel
        self.no_spatial = no_spatial
        self.stacking = stacking
        self.residual = residual
        self.map_repeated = map_repeated

        if not no_channel:
            self.channel_attention = nn.ModuleList()
            for _ in range(stacking):
                self.channel_attention.append(nn.ModuleList())
                for _ in range(map_repeated):
                    self.channel_attention[-1].append(ChannelAttention(
                        gate_channels, levels, reduction_ratio, pool_types, map_repeated, map_residual))

        if not no_spatial:
            self.spatial_attention = nn.ModuleList()
            for _ in range(stacking):
                self.spatial_attention.append(nn.ModuleList())
                for _ in range(map_repeated):
                    self.spatial_attention[-1].append(SpatialAttention(levels, kernel_size, map_repeated, map_residual))

    def forward(self, x):
        for i in range(self.stacking):
            if not self.no_channel:
                channel_attention_maps = self.channel_attention[i][0](x)
                for repeated in range(1, self.map_repeated):
                    channel_attention_maps2 = self.channel_attention[i][repeated](channel_attention_maps)
                    channel_attention_maps = [maps + maps2
                                              for maps, maps2 in zip(channel_attention_maps, channel_attention_maps2)]
                if self.residual:
                    x = [x[level] * channel_attention_maps[level] + x[level] for level in range(self.levels)]
                else:
                    x = [x[level] * channel_attention_maps[level] for level in range(self.levels)]

            if not self.no_spatial:
                spatial_attention_maps = self.spatial_attention[i][0](x)
                for repeated in range(1, self.map_repeated):
                    spatial_attention_maps2 = self.spatial_attention[i][repeated](spatial_attention_maps)
                    spatial_attention_maps = [maps + maps2
                                              for maps, maps2 in zip(spatial_attention_maps, spatial_attention_maps2)]
                if self.residual:
                    x = [x[level] * spatial_attention_maps[level] + x[level] for level in range(self.levels)]
                else:
                    x = [x[level] * spatial_attention_maps[level] for level in range(self.levels)]

        return sum(x)
