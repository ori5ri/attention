import torch
from mmcv.cnn import constant_init, kaiming_init
from torch import nn
import matplotlib.pyplot as plt


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
                 fusion_types=('channel_add', 'channel_mul'),
                 weight_type=False,
                 eps=0.0001,
                 viz=False):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        assert levels > 0, 'at least one feature should be used'
        assert weight_type in [False, 'before_attention', 'before_sigmoid', 'after_attention']
        self.repeated = repeated
        self.residual = residual
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
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

        if self.weight_type == 'before_attention' or self.weight_type == 'after_attention':
            self.weight = nn.Parameter(torch.Tensor(self.levels).fill_(1.0))
        elif self.weight_type == 'before_sigmoid':
            self.weight = nn.Parameter(torch.Tensor(self.repeated, self.levels).fill_(1.0))
            self.weight_add = nn.Parameter(torch.Tensor(self.repeated, self.levels).fill_(1.0))

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
        if self.weight_type == 'before_attention':
            w = self.relu(self.weight)
            w /= (w.sum() + self.eps)
            x = [x[i] * w[i] for i in range(self.levels)]
        outs = x
        x = torch.cat(x, dim=1)
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            if self.weight_type == 'before_sigmoid':
                w = self.relu(self.weight[0])
                w /= (w.sum() + self.eps)
                mul_maps = [torch.sigmoid(self.channel_mul_conv[0][i](context) * w[i])
                            for i in range(self.levels)]
            else:
                mul_maps = [torch.sigmoid(self.channel_mul_conv[0][i](context)) for i in range(self.levels)]
            for repeat in range(1, self.repeated):
                if self.weight_type == 'before_sigmoid':
                    w = self.relu(self.weight[repeat])
                    w /= (w.sum() + self.eps)
                if self.residual:
                    if self.weight_type == 'before_sigmoid':
                        mul_maps_ = [torch.sigmoid
                                     (self.channel_mul_conv[repeat][i](torch.cat(mul_maps, dim=1)) * w[i])
                                     for i in range(self.levels)]
                    else:
                        mul_maps_ = [torch.sigmoid(self.channel_mul_conv[repeat][i](torch.cat(mul_maps, dim=1)))
                                     for i in range(self.levels)]
                    mul_maps = [mul_map + mul_map_ for mul_map, mul_map_ in zip(mul_maps, mul_maps_)]
                else:
                    if self.weight_type == 'before_sigmoid':
                        mul_maps = [torch.sigmoid
                                    (self.channel_mul_conv[repeat][i](torch.cat(mul_maps, dim=1)) * w[i])
                                    for i in range(self.levels)]
                    else:
                        mul_maps = [torch.sigmoid(self.channel_mul_conv[repeat][i](torch.cat(mul_maps, dim=1)))
                                    for i in range(self.levels)]
            if self.viz:
                for i, (out, mul_map) in enumerate(zip(outs, mul_maps)):
                    import numpy as np
                    height = mul_map[0].squeeze().cpu().numpy()
                    large_index = height.argsort()[-9:][::-1]
                    small_index = height.argsort()[:9][::-1]
                    label = range(18)
                    left = range(18)
                    print('-'*50+str(i) + '-'*50)
                    print(large_index, small_index)
                    plt.bar(left, height[np.concatenate((large_index, small_index))], tick_label=label, width=0.8, color=['red'])

                    # plt.ylim([-1, 1])
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.title('good')

                    plt.show()

                    fig, axarr = plt.subplots(3, 3)
                    for idx in range(9):
                        axarr[idx // 3][idx % 3].imshow(out.squeeze()[large_index[idx]].squeeze().cpu())
                    plt.show()

                    fig, axarr = plt.subplots(3, 3)
                    for idx in range(9):
                        axarr[idx // 3][idx % 3].imshow(out.squeeze()[small_index[idx]].squeeze().cpu())
                    plt.show()

                    temp = out * mul_map

                    fig, axarr = plt.subplots(3, 3)
                    for idx in range(9):
                        axarr[idx // 3][idx % 3].imshow(temp.squeeze()[large_index[idx]].squeeze().cpu())
                    plt.show()

                    fig, axarr = plt.subplots(3, 3)
                    for idx in range(9):
                        axarr[idx // 3][idx % 3].imshow(temp.squeeze()[small_index[idx]].squeeze().cpu())
                    plt.show()

            outs = [out * mul_map for out, mul_map in zip(outs, mul_maps)]

        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            add_maps = [self.channel_add_conv[0][i](context) for i in range(self.levels)]
            if self.weight_type == 'before_sigmoid':
                w = self.relu(self.weight_add[0])
                w /= (w.sum() + self.eps)
                add_maps = [add_maps[i] * w[i] for i in range(self.levels)]
            for repeat in range(1, self.repeated):
                if self.residual:
                    add_maps_ = [self.channel_add_conv[repeat][i](torch.cat(add_maps, dim=1)) for i in
                                 range(self.levels)]
                    add_maps = [add_map + add_map_ for add_map, add_map_ in zip(add_maps, add_maps_)]
                else:
                    add_maps = [self.channel_add_conv[repeat][i](torch.cat(add_maps, dim=1)) for i in
                                range(self.levels)]

                if self.weight_type == 'before_sigmoid':
                    w = self.relu(self.weight_add[repeat])
                    w /= (w.sum() + self.eps)
                    add_maps = [add_maps[i] * w[i] for i in range(self.levels)]
            if self.viz:
                for i, (out, mul_map) in enumerate(zip(outs, add_maps)):
                    import numpy as np
                    height = mul_map[0].squeeze().cpu().numpy()
                    large_index = height.argsort()[-9:][::-1]
                    small_index = height.argsort()[:9][::-1]
                    label = range(18)
                    left = range(18)
                    print('-'*50+str(i) + '-'*50)
                    print(large_index, small_index)
                    plt.bar(left, height[np.concatenate((large_index, small_index))], tick_label=label, width=0.8, color=['red'])

                    # plt.ylim([-1, 1])
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.title('good')

                    plt.show()

                    fig, axarr = plt.subplots(3, 3)
                    for idx in range(9):
                        axarr[idx // 3][idx % 3].imshow(out.squeeze()[large_index[idx]].squeeze().cpu())
                    plt.show()

                    fig, axarr = plt.subplots(3, 3)
                    for idx in range(9):
                        axarr[idx // 3][idx % 3].imshow(out.squeeze()[small_index[idx]].squeeze().cpu())
                    plt.show()

                    temp = out * mul_map

                    fig, axarr = plt.subplots(3, 3)
                    for idx in range(9):
                        axarr[idx // 3][idx % 3].imshow(temp.squeeze()[large_index[idx]].squeeze().cpu())
                    plt.show()

                    fig, axarr = plt.subplots(3, 3)
                    for idx in range(9):
                        axarr[idx // 3][idx % 3].imshow(temp.squeeze()[small_index[idx]].squeeze().cpu())
                    plt.show()

            outs = [out + add_map for out, add_map in zip(outs, add_maps)]

        if self.weight_type == 'after_attention':
            w = self.relu(self.weight)
            w /= (w.sum() + self.eps)
            outs = [outs[i] * w[i] for i in range(self.levels)]

        return sum(outs)
