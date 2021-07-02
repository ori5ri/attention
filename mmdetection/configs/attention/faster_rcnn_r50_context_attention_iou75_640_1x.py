_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    './_base_/dataset_640.py',
    './_base_/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    neck=dict(
        type='Attention',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        attention_type='context'
    ),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.75,
                neg_iou_thr=0.75,
                min_pos_iou=0.75,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
    )
)
