_base_ = ["faster-rcnn_x101-64x4d_fpn_1x_coco.py"]

user_conf = dict(
    BATCH_SIZE = 8,
    LR = 0.001,
    EPOCHS = 10,
)

classes = ("covid")

work_dir = "/media/my_ftp/TFTs/amoure/TFM_MUIT/src/models/mmdet/train_results/"
data_root = "/media/my_ftp/BasesDeDatos_Torax_RX_CT/COVID19_CT/processed/segmentation/rgb_images"
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        type='CheckpointHook',
        save_best="coco/bbox_mAP_50",
        by_epoch=True,
        out_dir = "/media/my_ftp/TFTs/amoure/TFM_MUIT/src/models/mmdet/train_results/"
        )
     )

model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            num_classes=1,
        )
    )
)

train_cfg = dict(max_epochs=user_conf["EPOCHS"], type='EpochBasedTrainLoop', val_interval=1)
optim_wrapper = dict(
    optimizer=dict(lr=user_conf["LR"], momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')

train_dataloader = dict(
    dataset=dict(
        metainfo=dict(classes=classes),
        ann_file="/media/my_ftp/BasesDeDatos_Torax_RX_CT/COVID19_CT/processed/object_detection/coco_annotations/coco_train.json",
        data_prefix=dict(img=""),
        data_root=data_root,
    ),
    num_workers=4,
    batch_size=user_conf["BATCH_SIZE"]
)

val_dataloader = dict(
    dataset=dict(
        metainfo=dict(classes=classes),
        ann_file="/media/my_ftp/BasesDeDatos_Torax_RX_CT/COVID19_CT/processed/object_detection/coco_annotations/coco_val.json",
        data_prefix=dict(img=""),
        data_root=data_root,
    ),
    num_workers=4,
    batch_size=user_conf["BATCH_SIZE"]
)

val_evaluator = dict(
    ann_file="/media/my_ftp/BasesDeDatos_Torax_RX_CT/COVID19_CT/processed/object_detection/coco_annotations/coco_val.json",
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')

test_dataloader = dict(
    dataset=dict(
        metainfo=dict(classes=classes),
        ann_file="/media/my_ftp/BasesDeDatos_Torax_RX_CT/COVID19_CT/processed/object_detection/coco_annotations/coco_val.json",
        data_prefix=dict(img=""),
        data_root=data_root,
    ),
    num_workers=2,
    batch_size=2
)

test_evaluator = dict(
    ann_file="/media/my_ftp/BasesDeDatos_Torax_RX_CT/COVID19_CT/processed/object_detection/coco_annotations/coco_val.json",
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type="WandbVisBackend", init_kwargs=dict(project="mmdetection", name="test"))
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type="WandbVisBackend", init_kwargs=dict(project="mmdetection", name="test"))
    ])
