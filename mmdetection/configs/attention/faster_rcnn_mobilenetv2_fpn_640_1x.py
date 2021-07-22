_base_ = [
    '../_base_/models/faster_rcnn_mobilenet_v2_fpn.py',
    './_base_/dataset_640.py',
    './_base_/schedule_1x.py', '../_base_/default_runtime.py'
] 
model = dict(
    backbone=dict(
        type='MobileNetV2',
        widen_factor=1.0,
        out_indices=(1,2,4,7),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU6'),
        norm_eval=False,
        with_cp=False
        )

    )
# load_from ='work_dirs/faster_rcnn_mobilenetv2_fpn_640_1x/latest.pth'