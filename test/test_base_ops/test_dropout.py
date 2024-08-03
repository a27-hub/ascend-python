import torch

import torch_npu 
from torch_npu.testing.testcase import TestCase, run_tests


class TestDropout(TestCase):

    def _test_dropout_randomness(self, dtype, p):
        input_tensor = torch.randn(10, 10, requires_grad=True, dtype=dtype).npu()
        dropout_layer = torch.nn.Dropout(p=p)

        output1 = dropout_layer(input_tensor)
        output2 = dropout_layer(input_tensor)

        self.assertNotEqual(output1, output2)

    def _test_dropout_inplace(self, dtype, p):
        input_tensor = torch.randn(10, 10, requires_grad=True, dtype=dtype).npu()
        dropout_layer = torch.nn.Dropout(p=p, inplace=True)

        output = dropout_layer(input_tensor)

        self.assertTrue(input_tensor is output)

    def _test_dropout_inplace_vs_noninplace(self, dtype, p):
        input_tensor = torch.randn(5, 5, requires_grad=True, dtype=dtype).npu()

        torch.manual_seed(0)
        dropout_layer = torch.nn.Dropout(p=p, inplace=False)
        output_noninplace = dropout_layer(input_tensor.clone())

        torch.manual_seed(0)
        input_tensor_clone = input_tensor.clone()
        dropout_layer_inplace = torch.nn.Dropout(p=p, inplace=True)
        dropout_layer_inplace(input_tensor_clone)

        self.assertEqual(output_noninplace, input_tensor_clone)

    def test_dropout_randomness(self):
        dtypes = [torch.float32, torch.float16]
        probabilities = [0.0, 0.6, 1.0]

        for dtype in dtypes:
            for p in probabilities:
                with self.subTest(dtype=dtype, p=p):
                    self._test_dropout_randomness(dtype, p)

    def test_dropout_inplace(self):
        dtypes = [torch.float32, torch.float16]
        probabilities = [0.0, 0.6, 1.0]

        for dtype in dtypes:
            for p in probabilities:
                with self.subTest(dtype=dtype, p=p):
                    self._test_dropout_inplace(dtype, p)

    def test_dropout_inplace_vs_noninplace(self):
        dtypes = [torch.float32, torch.float16]
        probabilities = [0.0, 0.6, 1.0]

        for dtype in dtypes:
            for p in probabilities:
                with self.subTest(dtype=dtype, p=p):
                    self._test_dropout_inplace_vs_noninplace(dtype, p)


if __name__ == '__main__':
    run_tests()
