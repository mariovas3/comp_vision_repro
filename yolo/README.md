# You Only Look Once (YOLO)

> *Will implement object detection using YOLO. This is a "single-stage" object detection method as opposed to "two-stage" object detection that is done by region proposal-based methods.*


## Data Prep
* targets are modified as shown below:

```python
def resize_and_crop_bbox(
    bbox: MutableSequence,
    old_w: int,
    old_h: int,
    resize_size: int,
    crop_size: int,
) -> Tuple[MutableSequence, bool]:
    """
    Get the coords of bbox after resize and crop and bool indicator
    if bbox is usable.

    bbox should be in xmin, ymin, xmax, ymax format.

    Return: (bbox, bad_crop)

    Since we are resizing first and then cropping, it is possible
    we lose a big part of the bbox if it was close to the edge
    of the image. If the width or height decrease by more than
    40%, we return bad_crop=True alongside the bbox, otherwise
    bad_crop=False.
    """
    assert len(bbox) == 4
    new_w, new_h = get_resized_wh(old_w, old_h, size=resize_size)
    w_mult = new_w / old_w
    # shifting is for the crop operation since
    # we adjust coords by subtracting half ignored region
    # after the crop;
    w_shift = (new_w - crop_size) / 2
    h_mult = new_h / old_h
    h_shift = (new_h - crop_size) / 2
    # adjust for resizing;
    bbox[0], bbox[2] = bbox[0] * w_mult, bbox[2] * w_mult
    bbox[1], bbox[3] = bbox[1] * h_mult, bbox[3] * h_mult
    old_bw, old_bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # adjust for cropping;
    for i, c in enumerate(bbox):
        if i % 2 == 0:
            bbox[i] = clip(0, int(c - w_shift), crop_size)
            assert (
                bbox[i] <= crop_size
            ), f"{c=}, {bbox[i]=}, {crop_size=}, {new_w=}, {old_w=}, {old_h=}"
        else:
            bbox[i] = clip(0, int(c - h_shift), crop_size)
            assert (
                bbox[i] <= crop_size
            ), f"{c=}, {bbox[i]=}, {crop_size=}, {new_h=}, {old_w=}, {old_h=}"
    new_bw, new_bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # don't add this box if cropping left too much out;
    bad_crop = new_bw < old_bw * 0.6 or new_bh < old_bh * 0.6
    return bbox, bad_crop
```

* For clustering to get the anchor boxes I set `scale_box_dims=True` when getting the boxes to cluster:

```python
def get_all_boxes(dataset, resize_size, scale_box_dims=False, crop_size=224):
    all_boxes = []
    for _, info in dataset:
        _, boxes, img_size = get_labels_and_boxes_and_size(info["annotation"])
        if scale_box_dims:
            new_boxes = []
            for i, box in enumerate(boxes):
                # get XYXY after T.Resize and T.CenterCrop applied
                # to image.
                box, bad_crop = resize_and_crop_bbox(
                    bbox=box,
                    old_w=img_size["width"],
                    old_h=img_size["height"],
                    resize_size=resize_size,
                    crop_size=crop_size,
                )
                if bad_crop:
                    continue
                new_boxes.append(box)
            boxes = new_boxes
        all_boxes.extend(boxes)
    return all_boxes
```

```bash
export WANDB_START_METHOD="thread"
```

```bash
python yolo/run_training.py fit --config yolo_training.yaml --trainer.max_epochs=2 --trainer.limit_train_batches=2 --trainer.limit_val_batches=2 --trainer.check_val_every_n_epoch=2 --trainer.log_every_n_steps=1 --data.batch_size=64
```

## Implementation details:
* Will use tips and tricks from both `YOLO-v1` and `YOLO9000`. 
* Will not do the `WordTree` trick from `YOLO9000` that allows to combine the ImageNet and COCO datasets' labels. Instead, I will train and eval on `PASCAL-VOC`, because the goal is to get to a working PoC.
* Since both use pretrained "backbones", I will do so as well, trying something like `resnet50` and see where that goes.
    * I chose to use `resnet50` since based on <a href="https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights">this</a> table from PyTorch, it achieves `80.858%` accuracy on ImageNet, compared to `69.758%` for `resnet18`.
    * Also `resnet50` requires `4.09` GFLOPS compared to `1.81` GFLOPS for `resnet18` - shouldn't be a problem on modern hardware. On the other hand, the `resnet152` costs `11.52` GFLOPS and only does `82.284%` accuracy - pretty marginal improvement considering the extra params relative to `resnet50`.
    * I only use the resnet params up to the last conv block, so I don't use the avg pooling and ffn.
* The `resnet50` has an image compression factor of 32, so I will use multiples of 32 as inputs. 

### YOLO-v1 tricks:
* Single neural net gives all the outputs - object detection prob, class probs, bounding box coords.
* Paper quotes 45 fps on Titan X. Since I might use a different backbone, the goal is to get faster than 30 fps.