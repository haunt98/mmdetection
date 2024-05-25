# https://mmdetection.readthedocs.io/en/latest/user_guides/train.html#prepare-the-customized-dataset
_base_ = "../mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py"

model = dict(
    roi_head=dict(bbox_head=dict(num_classes=4), mask_head=dict(num_classes=4))
)

data_root = "data/manga109/"
metainfo = {
    "classes": ("body", "face", "frame", "text"),
}

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="annotations_coco_format/manga109s_coco_90train.json",
        data_prefix=dict(img="images/"),
    ),
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="annotations_coco_format/manga109s_coco_4val.json",
        data_prefix=dict(img="images/"),
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + "annotations_coco_format/manga109s_coco_4val.json"
)
test_evaluator = val_evaluator

# https://github.com/open-mmlab/mmdetection/tree/main/configs/mask_rcnn
load_from = "https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_2x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_2x_coco_bbox_mAP-0.403__segm_mAP-0.365_20200504_231822-a75c98ce.pth"
