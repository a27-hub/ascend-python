import os
import unittest
import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuSaveTensor(TestCase):
    def _torch_scalar_type_to_torch_dtype(self, dtype_value):
        if dtype_value == 0:
            return torch.uint8
        if dtype_value == 1:
            return torch.int8
        if dtype_value == 2:
            return torch.int16
        if dtype_value == 3:
            return torch.int32
        if dtype_value == 4:
            return torch.int64
        if dtype_value == 5:
            return torch.float16
        if dtype_value == 6:
            return torch.float32
        if dtype_value == 7:
            return torch.float64
        if dtype_value == 8:
            return torch.complex32
        if dtype_value == 9:
            return torch.complex64
        if dtype_value == 10:
            return torch.complex64128
        if dtype_value == 11:
            return torch.bool
        if dtype_value == 12:
            return torch.qint8
        if dtype_value == 13:
            return torch.quint8
        if dtype_value == 14:
            return torch.qint32
        if dtype_value == 15:
            return torch.bfloat16
        if dtype_value == 16:
            return torch.quint4x2
        if dtype_value == 17:
            return torch.quint2x4
        if dtype_value == 23:
            return torch.float8_e5m2
        if dtype_value == 24:
            return torch.float8_e4m3fn
        raise ValueError(f"Unsupported torch scalar type value {dtype_value}")

    def _load_tensor_from_bin(self, path):
        if not path.endswith(".bin") or not os.path.exists(path):
            raise ValueError(f"Invalid file path or file format: {path}")
        
        import numpy as np
        with open(path, 'rb') as f:
            dtype_line = f.readline()
            scalar_type = int(dtype_line.decode().strip()[7:])
            
            shape_line = f.readline()
            shape_str = shape_line.decode().strip()[8:-1]
            shape = tuple(map(int, shape_str.split(', '))) if shape_str else ()
            
            data = f.read()
        
        torch_dtype = self._torch_scalar_type_to_torch_dtype(scalar_type)
        expected_bytes = torch_dtype.itemsize * (np.prod(shape) if shape else 1)
        if len(data) < expected_bytes:
            raise RuntimeError(f"File size does not match, probably file is broken: {path}")
        
        tensor = torch.frombuffer(data, dtype=torch_dtype)
        if shape:
            tensor = tensor.reshape(shape)
        
        return tensor
    
    @unittest.skip("skip until CANN is updated to support aclrtLaunchHostFunc")
    def test_npu_tensor_save(self):
        x = torch.randn([5, 5]).npu()
        torch_npu.save_npugraph_tensor(x, save_path = "./x.bin")
        
        self.assertTrue(os.path.exists("./x_device_0.bin"))
        actual = self._load_tensor_from_bin("./x_device_0.bin")
        self.assertEqual(x.dtype, actual.dtype)
        self.assertEqual(x.shape, actual.shape)
        self.assertRtolEqual(x, actual, 0.001)


if __name__ == "__main__":
    run_tests()