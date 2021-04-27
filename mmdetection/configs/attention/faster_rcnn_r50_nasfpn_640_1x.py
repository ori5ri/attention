_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    './_base_/dataset_640.py',
    './_base_/schedule_1x.py', '../_base_/default_runtime.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    neck=dict(
        type='NASFPN',
        # in_channels=[256, 512, 1024, 2048],
        # out_channels=256,
        # start_level=1,
        # num_outs=5,
        # add_extra_convs='on_input',
        stack_times=7,
        norm_cfg=norm_cfg
    ),
)
