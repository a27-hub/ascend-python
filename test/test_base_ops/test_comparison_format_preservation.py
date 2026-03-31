import torch
import torch_npu
import numpy as np
from torch_npu.testing.testcase import TestCase, run_tests


class TestComparisonFormatPreservation(TestCase):
    """Test that comparison operators preserve input tensor formats"""
    
    def setUp(self):
        """Set up test environment"""
        super().setUp()
        # Enable private formats for testing
        self.original_allow_internal_format = torch_npu.npu.config.allow_internal_format
        torch_npu.npu.config.allow_internal_format = True
    
    def tearDown(self):
        """Restore original settings"""
        super().tearDown()
        torch_npu.npu.config.allow_internal_format = self.original_allow_internal_format
    
    def create_test_tensor(self, shape, dtype, format_type):
        """Create a test tensor with specified shape, dtype, and format"""
        np_tensor = np.random.uniform(-10, 10, shape).astype(dtype)
        tensor = torch.from_numpy(np_tensor).npu()
        # Apply specified format
        tensor = torch_npu.npu_format_cast(tensor, format_type)
        return tensor
    
    def test_comparison_operators_format_preservation(self):
        """Test that all comparison operators preserve input tensor formats"""
        # Test shapes and dtypes
        test_cases = [
            # (shape, dtype, input_format)
            ((2, 3, 4, 5), np.float32, torch_npu.Format.NCHW),      # Non-private format
            ((2, 3, 4, 5), np.float32, torch_npu.Format.ND),         # Non-private format
            ((2, 3, 4, 5), np.float32, torch_npu.Format.FRACTAL_Z),  # Private format
            ((2, 3, 4, 5), np.float32, torch_npu.Format.FRACTAL_NZ), # Private format
            ((2, 3, 4, 5), np.float32, torch_npu.Format.NC1HWC0),    # Private format
        ]
        
        # Comparison operators to test
        comparison_ops = [
            (torch.gt, ">"),   # greater than
            (torch.ge, ">="),  # greater equal
            (torch.lt, "<"),   # less than
            (torch.le, "<="),  # less equal
            (torch.eq, "=="),  # equal
            (torch.ne, "!="),  # not equal
        ]
        
        for shape, dtype, input_format in test_cases:
            for op, op_name in comparison_ops:
                # Test with two tensors of the same format
                tensor1 = self.create_test_tensor(shape, dtype, input_format)
                tensor2 = self.create_test_tensor(shape, dtype, input_format)
                
                # Get original formats
                orig_format1 = torch_npu.get_npu_format(tensor1)
                orig_format2 = torch_npu.get_npu_format(tensor2)
                
                # Perform comparison operation
                result = op(tensor1, tensor2)
                
                # Verify input formats are preserved
                self.assertEqual(
                    torch_npu.get_npu_format(tensor1), orig_format1,
                    f"{op_name} operation changed tensor1 format from {orig_format1} to {torch_npu.get_npu_format(tensor1)}"
                )
                self.assertEqual(
                    torch_npu.get_npu_format(tensor2), orig_format2,
                    f"{op_name} operation changed tensor2 format from {orig_format2} to {torch_npu.get_npu_format(tensor2)}"
                )
                
                # Test with tensor and scalar
                scalar = 0.5
                tensor = self.create_test_tensor(shape, dtype, input_format)
                orig_format = torch_npu.get_npu_format(tensor)
                
                # Perform comparison operation with scalar
                result = op(tensor, scalar)
                
                # Verify input format is preserved
                self.assertEqual(
                    torch_npu.get_npu_format(tensor), orig_format,
                    f"{op_name} operation with scalar changed tensor format from {orig_format} to {torch_npu.get_npu_format(tensor)}"
                )
    
    def test_inplace_comparison_operators(self):
        """Test that inplace comparison operators work correctly"""
        # Inplace comparison operators are not supported in PyTorch
        # They return a new tensor rather than modifying inplace
        shape = (2, 3, 4, 5)
        dtype = np.float32
        input_format = torch_npu.Format.FRACTAL_Z
        
        tensor = self.create_test_tensor(shape, dtype, input_format)
        orig_format = torch_npu.get_npu_format(tensor)
        
        # Create a result tensor with same shape and device
        result = torch.empty_like(tensor, dtype=torch.bool)
        
        # Test out variant of comparison operators
        torch.gt(tensor, 0.5, out=result)
        
        # Verify input format is preserved
        self.assertEqual(
            torch_npu.get_npu_format(tensor), orig_format,
            "out variant of gt operation changed input tensor format"
        )


if __name__ == "__main__":
    run_tests()