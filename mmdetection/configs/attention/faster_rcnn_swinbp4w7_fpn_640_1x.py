_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    './_base_/dataset_640.py',
    './_base_/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False),
    neck=dict(
        type='Attention',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=5,
        attention_type='context',
        fusion_types=['channel_add'],
        add_extra_convs='on_lateral',
    )
)

load_from = 'cascade_mask_rcnn_swin_base_patch4_window7.pth'
