# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.

import torch
import torch_npu  # noqa: F401
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


class TestSortedNms(TestCase):
    def test_sorted_nms_meta(self):
        boxes = torch.empty((3, 4), dtype=torch.float16, device="meta")
        sorted_scores = torch.empty((3,), dtype=torch.float16, device="meta")
        input_indices = torch.empty((3,), dtype=torch.int32, device="meta")
        max_output_size = torch.empty((1,), dtype=torch.int32, device="meta")
        iou_threshold = torch.empty((1,), dtype=torch.float16, device="meta")
        score_threshold = torch.empty((1,), dtype=torch.float16, device="meta")

        result = torch.ops.npu.sorted_nms(
            boxes, sorted_scores, input_indices, max_output_size,
            iou_threshold, score_threshold, offset=1,
        )

        self.assertEqual(result.shape, (3,))
        self.assertEqual(result.dtype, torch.int32)
        self.assertEqual(result.device.type, "meta")

    @SupportedDevices(["Ascend950"])
    def test_sorted_nms(self):
        boxes = torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [20.0, 20.0, 30.0, 30.0]],
            dtype=torch.float32,
            device="npu",
        )
        sorted_scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32, device="npu")
        input_indices = torch.tensor([0, 1, 2], dtype=torch.int32, device="npu")
        max_output_size = torch.tensor([2], dtype=torch.int32, device="npu")
        iou_threshold = torch.tensor([0.5], dtype=torch.float32, device="npu")
        score_threshold = torch.tensor([0.0], dtype=torch.float32, device="npu")

        result = torch.ops.npu.sorted_nms(
            boxes, sorted_scores, input_indices, max_output_size,
            iou_threshold, score_threshold,
        )

        self.assertRtolEqual(result.cpu(), torch.tensor([0, 2, -1], dtype=torch.int32))


if __name__ == "__main__":
    run_tests()
