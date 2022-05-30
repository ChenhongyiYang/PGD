_base_ = "../base/1x_setting.py"

fp16 = dict(loss_scale=512.)
model = dict(
    type='FCOS',
    backbone=dict(
        type='MobileNetV2',
        out_indices=(1, 2, 4, 7),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        act_cfg=dict(type='ReLU6'),
        init_cfg=dict(type='Pretrained', checkpoint='data/pretrain_models/mobilenet_v2_batch256_imagenet-ff34753d.pth')),
    neck=dict(
        type='FPN',
        in_channels=[24, 32, 96, 1280],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='FCOSDistillHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        identity_pos=0),
    # training and testing settings 
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        kd_assigner=dict(
            cls_assigner=dict(
                type='GaussianMLEWeightCalculatorAnchorfree',
                topk=30,
                alpha=0.8,
                low_bound=0.),
            reg_assigner=dict(
                type='GaussianMLEWeightCalculatorAnchorfree',
                topk=30,
                alpha=0.6,
                low_bound=0.)
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
