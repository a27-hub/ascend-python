# （beta）torch\_npu.npu\_batch\_nms

> [!NOTE]
> 该接口已支持 Ascend 950 后端（通过 ops-cv 安装包提供）。在 Ascend 950 上使用时，需确保已安装 ops-cv 950 算子包并加载其 `set_env.bash` 环境。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas 训练系列产品</term>                                       |    √     |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Ascend 950 系列产品</term>                                      |    √     |

## 功能说明

以批量处理方式对每个类别的检测框进行非极大值抑制（Non-Maximum Suppression，NMS），从而去除冗余检测框，输出保留下来的检测框及其对应的类别和得分。

## 函数原型

```python
torch_npu.npu_batch_nms(self, scores, score_threshold, iou_threshold, max_size_per_class, max_total_size, change_coordinate_frame=False, transpose_box=False) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

- **self** \(`Tensor`)：必填值，输入框的tensor，包含batch大小，数据类型支持`float16`或`float32`。当`transpose_box=False`时，输入shape为\[batch\_size, num\_anchors, q, 4\]；当`transpose_box=True`时，输入shape为\[batch\_size, q, 4, num\_anchors\]。其中q=1或q=num\_classes。
- **scores** \(`Tensor`)：必填值，输入tensor，数据类型支持`float16`或`float32`，需与self的dtype一致，输入shape为\[batch\_size, num\_anchors, num\_classes\]。
- **score\_threshold** \(`float`)：必填值，指定评分过滤器的阈值，用于筛选框，去除得分较低的框。允许负值，必须为有限值。
- **iou\_threshold** \(`float`)：必填值，指定nms的iou\_threshold，用于设定阈值，去除高于阈值的框。取值范围为\[0, 1\]。
- **max\_size\_per\_class** \(`int`)：必填值，指定每个类别的最大可选的框数，必须为正整数。
- **max\_total\_size** \(`int`)：必填值，指定每个batch最大可选的框数，必须为正整数。
- **change\_coordinate\_frame** \(`bool`)：可选值，是否正则化输出框坐标矩阵，默认值为False。**当前`npu_batch_nms`接口不支持设置为True**，因为该功能依赖`clip_window`输入，而本接口未提供该参数。
- **transpose\_box** \(`bool`)：可选值，确定boxes的布局，默认值为False。True表示boxes使用\[B, q, 4, N\]排布，False表示boxes使用\[B, N, q, 4\]排布。

## 返回值说明

- **nmsed_boxes** (`Tensor`)：shape为\(batch, max\_total\_size, 4\)的3D张量，指定每批次输出的nms框，数据类型与输入一致（`float16`或`float32`）。
- **nmsed_scores** (`Tensor`)：shape为\(batch, max\_total\_size\)的2D张量，指定每批次输出的nms分数，数据类型与输入一致（`float16`或`float32`）。
- **nmsed_classes**(`Tensor`)：shape为\(batch, max\_total\_size\)的2D张量，指定每批次输出的nms类，数据类型与输入一致（`float16`或`float32`）。
- **nmsed_num**(`Tensor`)：shape为\(batch\)的1D张量，指定nmsed\_boxes的有效数量，数据类型为`int32`。

输出为定长张量，有效输出数量由`nmsed_num`指定，无效区域填充为0。

## 约束说明

- `self`和`scores`必须为同一设备上的NPU Tensor，且dtype一致。
- `self`的rank必须为4，`scores`的rank必须为3。
- `iou_threshold`取值范围为\[0, 1\]，`max_size_per_class`和`max_total_size`必须为正整数。
- `change_coordinate_frame=True`在当前接口中不可用。
- 在 Ascend 950 上使用时，需安装 ops-cv 950 算子包。未安装时，运行时将返回算子选择错误。

## 调用示例

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
