import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16

from ..builder import NECKS


@NECKS.register_module()
class Attention(nn.Module):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = Attention(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 reduction_ratio=16,
                 kernel_size=7,
                 no_channel=False,
                 no_spatial=False,
                 stacking=1):
        super(Attention, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.fusion_attentions = nn.ModuleList()
        self.downsample_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            d_conv = nn.ModuleList()
            for j in range(self.start_level, self.backbone_end_level):
                temp = []
                for _ in range(i - j):
                    temp.append(ConvModule(
                        out_channels,
                        out_channels,
                        3,
                        stride=2,
                        padding=1,
                        inplace=False))
                d_conv.append(nn.Sequential(*temp))
            self.fusion_attentions.append(
                FusionAttention(
                    gate_channels=out_channels * (self.num_ins - start_level),
                    levels=self.num_ins - start_level,
                    reduction_ratio=reduction_ratio,
                    kernel_size=kernel_size,
                    no_channel=no_channel,
                    no_spatial=no_spatial,
                    stacking=stacking
                )
            )
            self.downsample_convs.append(d_conv)

        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        outs = []
        for i in range(used_backbone_levels):
            shape = laterals[i].shape[2:]
            samples = []
            samples.extend([self.downsample_convs[i][j](laterals[j]) for j in
                            range(i)])
            samples.append(laterals[i])
            samples.extend([F.interpolate(laterals[j], size=shape, **self.upsample_cfg) for j in
                            range(i + 1, used_backbone_levels)])
            outs.append(self.fusion_attentions[i](samples))

        # build outputs+
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](outs[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


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
    def __init__(self, gate_channels, levels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelAttention, self).__init__()
        self.gate_channels = gate_channels
        self.levels = levels
        self.mlp = nn.ModuleList()
        for _ in range(levels):
            if reduction_ratio != 1:
                self.mlp.append(nn.Sequential(
                    Flatten(),
                    nn.Linear(gate_channels, gate_channels // reduction_ratio),
                    nn.ReLU(),
                    nn.Linear(gate_channels // reduction_ratio, gate_channels // levels)
                ))
            else:
                self.mlp.append(nn.Sequential(
                    Flatten(),
                    nn.Linear(gate_channels, gate_channels // levels)
                ))
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        attention = torch.cat(x, dim=1)
        scale = []
        for i in range(self.levels):
            for pool_type in self.pool_types:
                if pool_type == 'avg':
                    avg_pool = F.avg_pool2d(attention, (attention.size(2), attention.size(3)), stride=(attention.size(2), attention.size(3)))
                    channel_att_raw = self.mlp[i](avg_pool)
                elif pool_type == 'max':
                    max_pool = F.max_pool2d(attention, (attention.size(2), attention.size(3)), stride=(attention.size(2), attention.size(3)))
                    channel_att_raw = self.mlp[i](max_pool)
                elif pool_type == 'lp':
                    lp_pool = F.lp_pool2d(attention, 2, (attention.size(2), attention.size(3)), stride=(attention.size(2), attention.size(3)))
                    channel_att_raw = self.mlp[i](lp_pool)
                elif pool_type == 'lse':
                    # LSE pool only
                    lse_pool = logsumexp_2d(attention)
                    channel_att_raw = self.mlp[i](lse_pool)

                if channel_att_sum is None:
                    channel_att_sum = channel_att_raw
                else:
                    channel_att_sum = channel_att_sum + channel_att_raw
            scale.append(torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x[i]))
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
    def __init__(self, levels, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.levels = levels
        self.compress = ChannelPool()
        self.spatial = nn.ModuleList()
        for _ in range(levels):
            self.spatial.append(
                BasicConv(2 * levels, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False))

    def forward(self, x):
        attention = torch.cat([(self.compress(_)) for _ in x], dim=1)
        scale = [torch.sigmoid(self.spatial[i](attention)) for i in range(self.levels)]
        return scale


class FusionAttention(nn.Module):
    def __init__(self, gate_channels, levels, reduction_ratio=16, kernel_size=7, pool_types=['avg', 'max'],
                 no_channel=False, no_spatial=False, stacking=1):
        super(FusionAttention, self).__init__()
        assert 1 <= stacking
        self.levels = levels
        self.no_channel = no_channel
        self.no_spatial = no_spatial
        self.stacking = stacking

        if not no_channel:
            self.channel_attention = nn.ModuleList()
            for _ in range(stacking):
                self.channel_attention.append(ChannelAttention(gate_channels, levels, reduction_ratio, pool_types))

        if not no_spatial:
            self.spatial_attention = nn.ModuleList()
            for _ in range(stacking):
                self.spatial_attention.append(SpatialAttention(levels, kernel_size))

    def forward(self, x):
        if not self.no_channel:
            channel_attention_maps = self.channel_attention[0](x)
            for stack in range(1, self.stacking):
                channel_attention_maps2 = self.channel_attention[stack](channel_attention_maps)
                channel_attention_maps = [maps + maps2
                                          for maps, maps2 in zip(channel_attention_maps, channel_attention_maps2)]
            x = [x[level] * channel_attention_maps[level] for level in range(self.levels)]

        if not self.no_spatial:
            spatial_attention_maps = self.spatial_attention[0](x)
            for stack in range(1, self.stacking):
                spatial_attention_maps2 = self.spatial_attention[stack](spatial_attention_maps)
                spatial_attention_maps = [maps + maps2
                                          for maps, maps2 in zip(spatial_attention_maps, spatial_attention_maps2)]
            x = [x[level] * spatial_attention_maps[level] for level in range(self.levels)]

        return sum(x)
