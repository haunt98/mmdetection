# https://mmdetection.readthedocs.io/en/latest/user_guides/train.html#prepare-the-customized-dataset
_base_ = "../yolox/yolox_tiny_8xb8-300e_coco.py"


model = dict(bbox_head=dict(num_classes=4))

data_root = "data/Manga109s/"
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
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="annotations_coco_format/manga109s_coco_15test.json",
        data_prefix=dict(img="images/"),
    )
)

val_evaluator = dict(
    ann_file=data_root + "annotations_coco_format/manga109s_coco_4val.json"
)
test_evaluator = dict(
    ann_file=data_root + "annotations_coco_format/manga109s_coco_15test.json"
)

# https://github.com/open-mmlab/mmdetection/tree/main/configs/yolox
load_from = "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"
