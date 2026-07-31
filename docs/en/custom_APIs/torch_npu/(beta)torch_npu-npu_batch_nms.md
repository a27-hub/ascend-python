# (beta) torch_npu.npu_batch_nms

> [!NOTE]
> This API supports the Ascend 950 backend (provided through the ops-cv installation package). When using it on Ascend 950, ensure the ops-cv 950 operator package is installed and its `set_env.bash` environment is loaded.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas training products</term>                                      |    √     |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Ascend 950 products</term>                                         |    √     |

## Function

Performs batched per-class non-maximum suppression (NMS) on bounding boxes to remove redundant detections, and outputs the remaining bounding boxes along with their corresponding classes and scores.

## Prototype

```python
torch_npu.npu_batch_nms(self, scores, score_threshold, iou_threshold, max_size_per_class, max_total_size, change_coordinate_frame=False, transpose_box=False) -> (Tensor, Tensor, Tensor, Tensor)
```

## Parameters

- **`self`** (`Tensor`): Required. Input box tensor containing the batch size. The data type must be `float16` or `float32`. When `transpose_box=False`, the shape is `(batch_size, num_anchors, q, 4)`. When `transpose_box=True`, the shape is `(batch_size, q, 4, num_anchors)`. `q` must be `1` or `num_classes`.
- **`scores`** (`Tensor`): Required. Input score tensor. The data type must be `float16` or `float32`, matching the dtype of `self`. Shape is `(batch_size, num_anchors, num_classes)`.
- **`score_threshold`** (`float`): Required. Score filter threshold used to screen boxes and remove boxes with low scores. Negative values are allowed. Must be a finite value.
- **`iou_threshold`** (`float`): Required. NMS IoU threshold used to set a threshold and remove boxes higher than the threshold. Must be in the range [0, 1].
- **`max_size_per_class`** (`int`): Required. Maximum number of boxes for each class. Must be a positive integer.
- **`max_total_size`** (`int`): Required. Maximum number of boxes for each batch. Must be a positive integer.
- **`change_coordinate_frame`** (`bool`): Optional. Specifies whether to normalize the coordinates of output boxes. The default value is `False`. **Setting this to `True` is not supported** by `npu_batch_nms` because it requires a `clip_window` input that this API does not provide.
- **`transpose_box`** (`bool`): Optional. Specifies the layout of boxes. The default value is `False`. If set to `True`, boxes use the `[B, q, 4, N]` layout. If set to `False`, boxes use the `[B, N, q, 4]` layout.

## Return Values

- **`nmsed_boxes`** (`Tensor`): NMS boxes output for each batch. This parameter must be a 3D tensor with shape `(batch, max_total_size, 4)`. The data type matches the input (`float16` or `float32`).
- **`nmsed_scores`** (`Tensor`): NMS scores output for each batch. This parameter must be a 2D tensor with shape `(batch, max_total_size)`. The data type matches the input (`float16` or `float32`).
- **`nmsed_classes`** (`Tensor`): NMS classes output for each batch. This parameter must be a 2D tensor with shape `(batch, max_total_size)`. The data type matches the input (`float16` or `float32`).
- **`nmsed_num`** (`Tensor`): Valid number of `nmsed_boxes`. This parameter must be a 1D tensor with shape `(batch)`. The data type is `int32`.

Outputs are fixed-size tensors. The number of valid outputs per batch is indicated by `nmsed_num`. Invalid regions are padded with zeros.

## Constraints

- `self` and `scores` must be NPU tensors on the same device with matching dtypes.
- `self` must have rank 4 and `scores` must have rank 3.
- `iou_threshold` must be in [0, 1]. `max_size_per_class` and `max_total_size` must be positive integers.
- `change_coordinate_frame=True` is not supported by this API.
- On Ascend 950, the ops-cv 950 operator package must be installed. Without it, the runtime will return an operator selection error.

## Example

```python
>>> import torch
>>> import torch_npu
>>> boxes = torch.randn(2, 100, 1, 4, dtype=torch.float16).to("npu")
>>> scores = torch.randn(2, 100, 80, dtype=torch.float16).to("npu")
>>> nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = torch_npu.npu_batch_nms(
...     boxes, scores, 0.3, 0.5, 10, 20)
>>> nmsed_boxes.shape
torch.Size([2, 20, 4])
>>> nmsed_num.shape
torch.Size([2])
```
