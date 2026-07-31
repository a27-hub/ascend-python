import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestBatchNms(TestCase):

    def test_output_shapes_and_dtypes_fp32(self):
        boxes = torch.randn(2, 100, 1, 4, dtype=torch.float32).npu()
        scores = torch.randn(2, 100, 80, dtype=torch.float32).npu()
        nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = torch_npu.npu_batch_nms(
            boxes, scores, 0.3, 0.5, 10, 20
        )
        self.assertEqual(nmsed_boxes.shape, (2, 20, 4))
        self.assertEqual(nmsed_scores.shape, (2, 20))
        self.assertEqual(nmsed_classes.shape, (2, 20))
        self.assertEqual(nmsed_num.shape, (2,))
        self.assertEqual(nmsed_boxes.dtype, torch.float32)
        self.assertEqual(nmsed_scores.dtype, torch.float32)
        self.assertEqual(nmsed_classes.dtype, torch.float32)
        self.assertEqual(nmsed_num.dtype, torch.int32)

    def test_output_shapes_and_dtypes_fp16(self):
        boxes = torch.randn(2, 100, 1, 4, dtype=torch.float16).npu()
        scores = torch.randn(2, 100, 80, dtype=torch.float16).npu()
        nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = torch_npu.npu_batch_nms(
            boxes, scores, 0.3, 0.5, 10, 20
        )
        self.assertEqual(nmsed_boxes.shape, (2, 20, 4))
        self.assertEqual(nmsed_scores.shape, (2, 20))
        self.assertEqual(nmsed_classes.shape, (2, 20))
        self.assertEqual(nmsed_num.shape, (2,))
        self.assertEqual(nmsed_boxes.dtype, torch.float16)
        self.assertEqual(nmsed_scores.dtype, torch.float16)
        self.assertEqual(nmsed_classes.dtype, torch.float16)
        self.assertEqual(nmsed_num.dtype, torch.int32)

    def test_q_equals_c(self):
        boxes = torch.randn(1, 50, 3, 4, dtype=torch.float32).npu()
        scores = torch.randn(1, 50, 3, dtype=torch.float32).npu()
        nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = torch_npu.npu_batch_nms(
            boxes, scores, 0.1, 0.5, 5, 10
        )
        self.assertEqual(nmsed_boxes.shape, (1, 10, 4))
        self.assertEqual(nmsed_scores.shape, (1, 10))
        self.assertEqual(nmsed_classes.shape, (1, 10))
        self.assertEqual(nmsed_num.shape, (1,))

    def test_transpose_box_true(self):
        boxes = torch.randn(2, 1, 4, 100, dtype=torch.float32).npu()
        scores = torch.randn(2, 100, 80, dtype=torch.float32).npu()
        nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = torch_npu.npu_batch_nms(
            boxes, scores, 0.3, 0.5, 10, 20, transpose_box=True
        )
        self.assertEqual(nmsed_boxes.shape, (2, 20, 4))
        self.assertEqual(nmsed_scores.shape, (2, 20))
        self.assertEqual(nmsed_classes.shape, (2, 20))
        self.assertEqual(nmsed_num.shape, (2,))

    def test_non_contiguous_input(self):
        boxes_base = torch.randn(2, 100, 2, 4, dtype=torch.float32)
        boxes = boxes_base[:, :, :1, :].npu()
        scores = torch.randn(2, 100, 80, dtype=torch.float32).npu()
        nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = torch_npu.npu_batch_nms(
            boxes, scores, 0.3, 0.5, 10, 20
        )
        self.assertEqual(nmsed_boxes.shape, (2, 20, 4))
        self.assertEqual(nmsed_num.shape, (2,))

    def test_error_wrong_dtype(self):
        boxes = torch.randint(0, 10, (2, 100, 1, 4), dtype=torch.int32).npu()
        scores = torch.randint(0, 10, (2, 100, 80), dtype=torch.int32).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_batch_nms(boxes, scores, 0.3, 0.5, 10, 20)

    def test_error_mismatched_dtype(self):
        boxes = torch.randn(2, 100, 1, 4, dtype=torch.float16).npu()
        scores = torch.randn(2, 100, 80, dtype=torch.float32).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_batch_nms(boxes, scores, 0.3, 0.5, 10, 20)

    def test_error_wrong_rank_boxes(self):
        boxes = torch.randn(2, 100, 4, dtype=torch.float32).npu()
        scores = torch.randn(2, 100, 80, dtype=torch.float32).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_batch_nms(boxes, scores, 0.3, 0.5, 10, 20)

    def test_error_wrong_rank_scores(self):
        boxes = torch.randn(2, 100, 1, 4, dtype=torch.float32).npu()
        scores = torch.randn(2, 100, dtype=torch.float32).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_batch_nms(boxes, scores, 0.3, 0.5, 10, 20)

    def test_error_batch_mismatch(self):
        boxes = torch.randn(2, 100, 1, 4, dtype=torch.float32).npu()
        scores = torch.randn(3, 100, 80, dtype=torch.float32).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_batch_nms(boxes, scores, 0.3, 0.5, 10, 20)

    def test_error_change_coordinate_frame_true(self):
        boxes = torch.randn(2, 100, 1, 4, dtype=torch.float32).npu()
        scores = torch.randn(2, 100, 80, dtype=torch.float32).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_batch_nms(
                boxes, scores, 0.3, 0.5, 10, 20, change_coordinate_frame=True
            )

    def test_error_invalid_iou_threshold(self):
        boxes = torch.randn(2, 100, 1, 4, dtype=torch.float32).npu()
        scores = torch.randn(2, 100, 80, dtype=torch.float32).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_batch_nms(boxes, scores, 0.3, 1.5, 10, 20)

    def test_error_invalid_max_size_per_class(self):
        boxes = torch.randn(2, 100, 1, 4, dtype=torch.float32).npu()
        scores = torch.randn(2, 100, 80, dtype=torch.float32).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_batch_nms(boxes, scores, 0.3, 0.5, 0, 20)

    def test_error_invalid_max_total_size(self):
        boxes = torch.randn(2, 100, 1, 4, dtype=torch.float32).npu()
        scores = torch.randn(2, 100, 80, dtype=torch.float32).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_batch_nms(boxes, scores, 0.3, 0.5, 10, -1)

    def test_all_filtered_by_score_threshold(self):
        boxes = torch.randn(1, 10, 1, 4, dtype=torch.float32).npu()
        scores = torch.randn(1, 10, 5, dtype=torch.float32).npu()
        nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = torch_npu.npu_batch_nms(
            boxes, scores, 999.0, 0.5, 5, 10
        )
        nmsed_num_cpu = nmsed_num.cpu()
        self.assertEqual(nmsed_num_cpu.item(), 0)

    def test_single_box_single_class(self):
        boxes = torch.tensor([[[[10.0, 20.0, 50.0, 60.0]]]], dtype=torch.float32).npu()
        scores = torch.tensor([[[0.9]]], dtype=torch.float32).npu()
        nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = torch_npu.npu_batch_nms(
            boxes, scores, 0.3, 0.5, 5, 10
        )
        nmsed_num_cpu = nmsed_num.cpu()
        self.assertEqual(nmsed_num_cpu.item(), 1)
        nmsed_boxes_cpu = nmsed_boxes.cpu()
        self.assertAlmostEqual(nmsed_boxes_cpu[0, 0, 0].item(), 10.0, places=2)
        self.assertAlmostEqual(nmsed_boxes_cpu[0, 0, 1].item(), 20.0, places=2)
        self.assertAlmostEqual(nmsed_boxes_cpu[0, 0, 2].item(), 50.0, places=2)
        self.assertAlmostEqual(nmsed_boxes_cpu[0, 0, 3].item(), 60.0, places=2)


if __name__ == "__main__":
    run_tests()
